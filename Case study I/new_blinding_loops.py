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
import network_data
import inner_loop

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
def fault_location_loop(net, default_net, dres_location, setting, blinding_data, no_blinding_data, relative_data, num_lines, relay_settings_IDMT, dres_capacity_list):

    with suppress_stdout():

        fault_location_list = list(range(0, num_lines))
        dres_capacity_list_stored = dres_capacity_list.copy()

        for k in fault_location_list:
            """
            Check here for blinding necessary condition
            """
            print('Fault location: ', k)
            blinding_condition = network_data.check_necessary_condition(default_net, k, dres_location)
            if blinding_condition:
                dres_capacity_list = dres_capacity_list_stored.copy()
                blinding_data, no_blinding_data, relative_data = inner_loop.fault_calculation_loop(net, default_net, relay_settings_IDMT, dres_location, setting,
                                                                                                   blinding_data, no_blinding_data, relative_data, dres_capacity_list, k)

    return blinding_data, no_blinding_data, relative_data


#====================================================================================================================================================================================

def blinding_impact(dres_location_list, dres_capacity_list, blinding_data, no_blinding_data, relative_data, num_lines):

    """
    loop 1: select location
    loop 2: while loop to select fault location and condition check
    loop 3: add capacities to gens and fault calculation
    """

    setting = []
    net = dres_scenario.power_network()
    default_net = dres_scenario.power_network()
    net, relay_settings_IDMT = dres_scenario.baseline_settings(net)
    setting.append(relay_settings_IDMT['I_s[kA]'].tolist())
    #print('settings', relay_settings_IDMT)
    pbar = tqdm(total=len(dres_location_list))
    len_loc_list = len(dres_location_list)

    for i in range(len_loc_list):
        dres_location = random.choice(dres_location_list)
        dres_location_list.remove(dres_location)
        print('dres_location: ', dres_location)
        blinding_data, no_blinding_data, relative_data = fault_location_loop(net, default_net, dres_location, setting, blinding_data, no_blinding_data, relative_data, num_lines, relay_settings_IDMT, dres_capacity_list)

        pbar.update(1)
    pbar.close()

    return blinding_data, no_blinding_data, relative_data


#====================================================================================================================================================================================
