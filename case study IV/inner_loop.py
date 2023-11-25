# File for loops in optimization
# Author: Amit Dilip Patil

import itertools
import multiprocessing
import pandapower as pp
import pandapower.control as control
import pandas as pd
import pandapower.networks as nw
import pandapower.shortcircuit as sc
import pandapower.topology as top
import dres_scenario
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import random
import warnings
import time
from pandapower.plotting import pf_res_plotly
from statistics import mean
import dres_scenario

numba = True
from pandapower.protection import oc_relay_model as oc_protection
from multiprocessing import Pool
import multiprocessing as mp
from tqdm import tqdm
from contextlib import contextmanager
import sys, os
import file_io
import network_data

trip_index_time = []
trip_index_time_false = []


# ====================================================================================================================================================================================
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# ====================================================================================================================================================================================

def blinding_case(net, sgen_state, fault_location, relay_settings_IDMT, dres_location, dres_dist, setting, data):

    if len(net.sgen) > 0:
        net.sgen.in_service = sgen_state

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        trip_decisions_IDMT, net_sc = oc_protection.run_fault_scenario_oc(net, sc_line_id=fault_location,
                                                                          sc_location=0.5,
                                                                          relay_settings=relay_settings_IDMT)

    df = pd.DataFrame(trip_decisions_IDMT)
    fault_current = df['Fault Current [kA]'].tolist()  # Extract fault current values

    trip_times_with_inf = []
    for l in range(len(net.switch)):
        temp = trip_decisions_IDMT[l]['Trip time [s]']
        temp_pair = (l, temp)
        trip_times_with_inf.append(temp_pair)

    trip_times = pd.DataFrame(trip_times_with_inf)
    with pd.option_context('mode.use_inf_as_null', True):
        #trip_times = trip_times.dropna()
        trip_times = trip_times.fillna(99999)
    trip_times.columns = ['index', 'time']

    trip_index_time_w_sgen = []
    for m in range(len(trip_times)):
        x = trip_times.at[trip_times.index[m], 'index']
        y = trip_times.at[trip_times.index[m], 'time']
        trip_index_time_w_sgen.append(tuple((x, y)))

    new_row = {'dres_location': dres_location, 'dres_capacity': dres_dist, 'fault_location': fault_location,
               'trip_times': trip_index_time_w_sgen, 'settings': setting, 'fault current': fault_current}
    data = data.append(new_row, ignore_index=True)

    return data, trip_times

#====================================================================================================================================================================================

def fault_calculation_loop(net, default_net, relay_settings_IDMT, dres_location, setting, blinding_data, no_blinding_data,
                           relative_data, dres_capacity_list, fault_location):

    # with suppress_stdout():

    """
    Loop for each dres capacity
    """

    """
    default_net for no sgen scenario, net for sgen scenario
    """

    for j in range(len(dres_capacity_list)):

        random.shuffle(dres_capacity_list)
        dres_dist = dres_capacity_list.pop()
        print('dres dist:', dres_dist)

        # dres_dist = random.choice(dres_capacity_list)
        # dres_capacity_list.remove(dres_dist)
        # print('dres dist:', dres_dist)

        """
        No sgen scenario
        """

        sgen_state = False
        print('State: ', sgen_state)
        no_blinding_data, trip_times = blinding_case(default_net, sgen_state, fault_location, relay_settings_IDMT,
                                                     dres_location, dres_dist, setting, no_blinding_data)

        """
        sgen scenario
        """

        new_net = dres_scenario.power_network()
        net = dres_scenario.dres_penetration(new_net, dres_location, dres_dist)

        sgen_state = True
        print('State: ', sgen_state)
        blinding_data, trip_times_blind = blinding_case(net, sgen_state, fault_location, relay_settings_IDMT, dres_location, dres_dist, setting, blinding_data)

        """
        Find the CBs that tripped in both cases and calculate difference
        """

        num_line_feeders, num_bus_feeders, num_feeder_hops, switch_locations, switches_in_feeders, cumulative_line_feeders, feeder_merge_bus_index = network_data.network_info(net)
        original_switch_locations = switch_locations.copy()

        check_list_blind = trip_times_blind['time'].tolist()  # Creates a list containing finite trip times
        check_list_no_blind = trip_times['time'].tolist()

        if check_list_blind:

            min_trip_time = trip_times_blind['time'].idxmin()
            min_trip_time_index = trip_times_blind.at[min_trip_time, 'index']

            primary_index, secondary_index, path, hops = network_data.find_info_about_case(net, switch_locations, original_switch_locations, num_feeder_hops, num_line_feeders, fault_location)
            print('path:', path)

            #Pop primary and secondary from the list and compute the following for the remaining CBs

            if len(path) <= 2:
                all_above_threshold = all(num > 90000 for num in check_list_blind)   # 90000 is used as 99999 is assigned for no trips
                if all_above_threshold:
                    print("No CBs have tripped")

                exist_count = path.count(min_trip_time_index)
                if exist_count == 0:            #Eliminates cases with symapthetic tripping; Masks blinding as those cases are removed
                    print("Sympathetic tripping")

                """
                Calculate blinding time delta t
                """
                primary_switch_index = net.switch.index[net.switch['element'] == primary_index].values
                primary_switch_index = primary_switch_index[0]
                secondary_switch_index = net.switch.index[net.switch['element'] == secondary_index].values
                secondary_switch_index = secondary_switch_index[0]

                primary_time_blind = trip_times_blind.at[trip_times_blind.index[primary_switch_index], 'time']
                secondary_time_blind = trip_times_blind.at[trip_times_blind.index[secondary_switch_index], 'time']

                primary_time = trip_times.at[trip_times.index[primary_switch_index], 'time']
                secondary_time = trip_times.at[trip_times.index[secondary_switch_index], 'time']

                primary_difference = primary_time_blind - primary_time
                secondary_difference = secondary_time_blind - secondary_time

                if secondary_difference >= primary_difference:
                    difference = primary_difference
                    tripped_index = primary_index
                else:
                    difference = secondary_difference
                    tripped_index = secondary_index

                new_row = {'dres_location': dres_location, 'dres_capacity': dres_dist, 'fault_location': fault_location, 'primary_index': primary_index,
                           'primary_trip_time': round(primary_time, 3), 'primary_trip_time_blind': round(primary_time_blind, 3), 'primary_delta_trip_time':round(primary_difference, 3),
                           'tripped_index': tripped_index, 'tripped_time': round(secondary_time, 3), 'tripped_time_blind': round(secondary_time_blind, 3), 'delta_trip_times': round(difference, 3)}

                relative_data = relative_data.append(new_row, ignore_index=True)

                print('path test')
                remain_path = path.copy()
                if len(remain_path) > 1:
                    remain_path.remove(primary_index)
                    remain_path.remove(secondary_index)



    return blinding_data, no_blinding_data, relative_data
