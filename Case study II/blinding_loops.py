#File for loops in optimization
#Author: Amit Dilip Patil

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

blinded = []
not_blinded = []
trip_index_time = []
trip_index_time_false = []

#====================================================================================================================================================================================
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

#====================================================================================================================================================================================

def fault_calculation_loop(net, num_lines, relay_settings_IDMT, dres_location, dres_dist, setting, blinding_data, no_blinding_data, relative_data):

    with suppress_stdout():
        fault_location_list = list(range(0, num_lines - 1))
        len_fault_list = len(fault_location_list)

        """
        Loop for each fault location
        """
        for k in range(len_fault_list):

            print('fault location: ', k)

            """
            Run short circuit calculation for blinding scenario
            """
            trip_time_local = []
            net.sgen.in_service = True

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                trip_decisions_IDMT, net_sc = oc_protection.run_fault_scenario_oc(net, sc_line_id=k, sc_location=0.5,
                                                                                  relay_settings=relay_settings_IDMT)
                df = pd.DataFrame(trip_decisions_IDMT)
                fault_current = df['Fault Current [kA]'].tolist()   #Extract fault current values

            """
            Find tripped CBs and the respective trip times
            """
            trip_times_with_inf = []
            for l in range(len(net.switch)):
                temp = trip_decisions_IDMT[l]['Trip time [s]']
                temp_pair = (l, temp)
                trip_times_with_inf.append(temp_pair)

            trip_times_blind = pd.DataFrame(trip_times_with_inf)
            with pd.option_context('mode.use_inf_as_null', True):
                trip_times_blind = trip_times_blind.dropna()


            trip_times_blind.columns = ['index', 'time']
            for m in range(len(trip_times_blind)):
                x = trip_times_blind.at[trip_times_blind.index[m], 'index']
                y = trip_times_blind.at[trip_times_blind.index[m], 'time']
                trip_time_local.append(tuple((x, y)))

            data = [dres_location, dres_dist, k, trip_time_local, setting]
            blinded.append(data)

            # (columns=['dres_location', 'dres_capacity', 'fault_location', 'trip_times', 'settings'])
            new_row = {'dres_location': dres_location, 'dres_capacity': dres_dist, 'fault_location': k,
                       'trip_times': trip_time_local, 'settings': setting, 'fault current': fault_current}
            blinding_data = blinding_data.append(new_row, ignore_index=True)

            """
            Run short circuit calculation for no dres scenario
            """
            trip_time_local = []
            net.sgen.in_service = False
            print('fault location: ', k)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                trip_decisions_IDMT, net_sc = oc_protection.run_fault_scenario_oc(net, sc_line_id=k, sc_location=0.5,
                                                                                  relay_settings=relay_settings_IDMT)
                df = pd.DataFrame(trip_decisions_IDMT)
                fault_current = df['Fault Current [kA]'].tolist()

            """
            Find tripped CBs and the respective trip times
            """
            trip_times_with_inf = []
            for l in range(len(net.switch)):
                temp = trip_decisions_IDMT[l]['Trip time [s]']
                temp_pair = (l, temp)
                trip_times_with_inf.append(temp_pair)

            trip_times = pd.DataFrame(trip_times_with_inf)
            with pd.option_context('mode.use_inf_as_null', True):
                trip_times = trip_times.dropna()

            trip_times.columns = ['index', 'time']
            for m in range(len(trip_times)):
                x = trip_times.at[trip_times.index[m], 'index']
                y = trip_times.at[trip_times.index[m], 'time']
                trip_time_local.append(tuple((x, y)))

            data = [dres_location, dres_dist, k, trip_time_local, setting]
            not_blinded.append(data)

            new_row = {'dres_location': dres_location, 'dres_capacity': dres_dist, 'fault_location': k,
                       'trip_times': trip_time_local, 'settings': setting, 'fault current': fault_current}
            no_blinding_data = no_blinding_data.append(new_row, ignore_index=True)

            """
            Find the CBs that tripped in both cases and calculate difference
            """
            for i in range(len(trip_times_blind)):
                ix = trip_times_blind.at[trip_times_blind.index[i], 'index']
                iy = trip_times_blind.at[trip_times_blind.index[i], 'time']

                for j in range(len(trip_times)):
                    jx = trip_times.at[trip_times.index[j], 'index']
                    jy = trip_times.at[trip_times.index[j], 'time']

                    if ix == jx:
                        new_row = {'dres_location': dres_location, 'dres_capacity': dres_dist, 'fault_location': k,
                                   'trip_times': iy-jy, 'settings': setting, 'fault current': fault_current}
                        relative_data = relative_data.append(new_row, ignore_index=True)


    return blinding_data, no_blinding_data, relative_data

#====================================================================================================================================================================================

def capacity_loop(dres_capacity_list, len_cap_list, default_net, dres_location, setting, blinding_data, no_blinding_data, relative_data, num_lines, relay_settings_IDMT):

    with suppress_stdout():

        for j in range(len_cap_list):

            dres_dist = random.choice(dres_capacity_list)
            dres_capacity_list.remove(dres_dist)
            #print('dres_distribution: ', dres_dist)
            net = default_net
            net = dres_scenario.dres_penetration(net, dres_location, dres_dist)

            blinding_data, no_blinding_data, relative_data = fault_calculation_loop(net, num_lines, relay_settings_IDMT, dres_location, dres_dist, setting, blinding_data, no_blinding_data, relative_data)

    return blinding_data, no_blinding_data, relative_data


#====================================================================================================================================================================================


def blinding_impact(dres_location_list, dres_capacity_list, blinding_data, no_blinding_data, relative_data, num_lines):
    #Three loops:
    # 1) dres_location
    # 2) dres capacity
    # 3) fault location

    with suppress_stdout():

        setting = []
        net = dres_scenario.power_network()
        default_net = net
        net, relay_settings_IDMT = dres_scenario.baseline_settings(net)
        setting.append(relay_settings_IDMT['I_s[kA]'].tolist())
        dres_capacity_list_stored = dres_capacity_list.copy()

        pbar = tqdm(total=len(dres_location_list))
        len_loc_list = len(dres_location_list)
        for i in range(len_loc_list):
            dres_location = random.choice(dres_location_list)
            dres_location_list.remove(dres_location)
            print('dres_location: ', dres_location)
            dres_capacity_list = dres_capacity_list_stored.copy()
            len_cap_list = len(dres_capacity_list)
            blinding_data, no_blinding_data, relative_data = capacity_loop(dres_capacity_list, len_cap_list, default_net, dres_location, setting, blinding_data, no_blinding_data, relative_data, num_lines, relay_settings_IDMT)

            pbar.update(1)
        pbar.close()

    return blinding_data, no_blinding_data, relative_data


#====================================================================================================================================================================================
