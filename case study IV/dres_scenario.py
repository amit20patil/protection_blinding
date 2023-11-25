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

def power_network():
    '''
    This function initializes the power system
    :param state:
    :return:
    '''
    net = nw.case33bw()

    net.ext_grid["s_sc_min_mva"] = 0
    net.ext_grid["s_sc_max_mva"] = 25
    net.ext_grid["rx_min"] = 0.2
    net.ext_grid["rx_max"] = 0.35
    net.ext_grid["r0x0_max"] = 0.4
    net.ext_grid["x0x_max"] = 1.0
    net.ext_grid["r0x0_min"] = 0.1
    net.ext_grid["x0x_min"] = 0.1
    net.line["r0_ohm_per_km"] = 0.244
    net.line["x0_ohm_per_km"] = 0.336
    net.line["c0_nf_per_km"] = 2000
    net.ext_grid['p_mw'] = 0.05
    net.line["endtemp_degree"] = 80

    switch0 = pp.create_switch(net, 0, 0, closed=True, et="l", type='CB_IDMT', name="s0")
    switch1 = pp.create_switch(net, 1, 17, closed=True, et="l", type='CB_IDMT', name="s1")
    switch2 = pp.create_switch(net, 2, 21, closed=True, et="l", type='CB_IDMT', name="s2")
    switch3 = pp.create_switch(net, 5, 24, closed=True, et="l", type='CB_IDMT', name="s3")

    switch4 = pp.create_switch(net, 5, 5, closed=True, et="l", type='CB_IDMT', name="s4")
    switch5 = pp.create_switch(net, 9, 9, closed=True, et="l", type='CB_IDMT', name="s5")
    switch6 = pp.create_switch(net, 13, 13, closed=True, et="l", type='CB_IDMT', name="s6")
    switch7 = pp.create_switch(net, 28, 28, closed=True, et="l", type='CB_IDMT', name="s7")

    for index, row in net.switch.iterrows():
        line_index = net.switch.at[index, 'element']
        net.line.at[line_index, "cb_index"] = index

    return net

#====================================================================================================================================================================================

def generate_dres_lists( num_feeder, num_buses_feeder1, num_buses_feeder2, num_buses_feeder3, num_buses_feeder4, sn_mva, capacity_list_split, location_list_split):

    dres_capacity_list = []

    """
    Create permutations of dres capacity distributed across all feeders
    """
    numbers = list(itertools.chain(range(0, sn_mva + 1)))
    for p in itertools.product(numbers, repeat=num_feeder):
        if sum(p) == sn_mva:
            dres_capacity_list.append(p)

    # for i in range(1, sn_mva+1):
    #     cap = [i, 0, 0, 0]
    #     dres_capacity_list.append(cap)

    np.random.seed(1235)

    """
    Dirichlet approach
    """
    # for i in range(100):
    #     dres_dist = np.random.dirichlet(np.ones(num_feeder))
    #     dres_dist = np.round(dres_dist, 2)
    #     dres_capacity_list.append(dres_dist)
    #
    # print("Before pop: ", dres_capacity_list)
    # print(len(dres_capacity_list))


    """
    Randomly remove elements of capacity list for shorter test simulations
    """
    n = round(len(dres_capacity_list) * capacity_list_split)
    for i in range(n):
        random_index = np.random.randint(0, len(dres_capacity_list))
        dres_capacity_list.pop(random_index)
    print(len(dres_capacity_list))

    """
    Create list of dres locations
    """
    feeder1_list = list(range(0, (num_buses_feeder1) - 2))
    feeder2_list = list(range(num_buses_feeder1, num_buses_feeder1 + num_buses_feeder2 - 2))
    feeder3_list = list(
        range(num_buses_feeder1 + num_buses_feeder2, num_buses_feeder1 + num_buses_feeder2 + num_buses_feeder3 - 1))
    feeder4_list = list(range(num_buses_feeder1 + num_buses_feeder2 + num_buses_feeder3,
                              num_buses_feeder1 + num_buses_feeder2 + num_buses_feeder3 + num_buses_feeder4 - 1))
    dres_location_list = [[a, b, c, d] for a in feeder1_list for b in feeder2_list for c in feeder3_list for d in
                          feeder4_list]

    """
    Keep m elements of location list for shorter test simulations
    """

    dres_location_list = [[7, 18, 23, 25]]
    #dres_location_list = [[1, 20, 25, 29], [7, 18, 23, 25]]

    #Comment the two lines below and uncomment the following code line for only keeping one or m values.
    # m = round(len(dres_location_list) * location_list_split)

    # m = location_list_split
    #
    # temp = []
    # for i in range(m):
    #     random_index = np.random.randint(0, len(dres_location_list))
    #     location = dres_location_list.pop(random_index)
    #     temp.append(location)
    #
    # dres_location_list = temp.copy()
    # print("After pop: ", dres_location_list)
    # print(len(dres_location_list))

    return dres_location_list, dres_capacity_list

#====================================================================================================================================================================================

def dres_penetration(net, dres_location_list, dres_capacity):

    total_capacity = 0
    for i in range(len(dres_capacity)):
        total_capacity = total_capacity + dres_capacity[i]

    net.ext_grid["s_sc_max_mva"].at[0] = net.ext_grid["s_sc_max_mva"].at[0] - total_capacity

    #Place sgens
    pp.create_sgen(net, dres_location_list[0], p_mw=0.050, q_mvar=0.1, sn_mva=dres_capacity[0], type="WP", scaling=1.0, in_service=True, current_source=True, k=1.3)
    pp.create_sgen(net, dres_location_list[1], p_mw=0.050, q_mvar=0.1, sn_mva=dres_capacity[1], type="WP", scaling=1.0, in_service=True, current_source=True, k=1.3)
    pp.create_sgen(net, dres_location_list[2], p_mw=0.050, q_mvar=0.1, sn_mva=dres_capacity[2], type="WP", scaling=1.0, in_service=True, current_source=True, k=1.3)
    pp.create_sgen(net, dres_location_list[3], p_mw=0.050, q_mvar=0.1, sn_mva=dres_capacity[3], type="WP", scaling=1.0, in_service=True, current_source=True, k=1.3)

    return net
#====================================================================================================================================================================================

def baseline_settings(net):

    net.sgen.in_service = False
    tripping_time = pd.DataFrame({'switch_id': [0, 1, 2, 3, 4, 5, 6, 7],
                                  'tms': [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                  't_grade': [3.5, 2, 2, 2, 1.75, 1.5, 1.0, 1.75]})
    # tripping_time = pd.DataFrame({'switch_id': [0, 1, 2, 3, 4, 5, 6, 7],
    #                               'tms': [1, 1, 1, 1, 1, 1, 1, 1],
    #                               't_grade': [0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6]})
    with suppress_stdout():
        #initial settings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            relay_settings_IDMT = oc_protection.oc_parameters(net, time_settings=tripping_time, relay_type='IDMT', curve_type='standard_inverse')

        #Finding corrected settings

        for i in range(len(net.line)):
            trip_decisions_IDMT, net_sc = oc_protection.run_fault_scenario_oc(net, sc_line_id=i, sc_location=0.5, relay_settings=relay_settings_IDMT)

            df = pd.DataFrame(trip_decisions_IDMT, columns=['Switch ID', 'Switch Type', 'Trip type', 'Trip', 'Fault Current [kA]', 'Trip time [s]'])

            min = df.loc[df['Fault Current [kA]'] > 0, 'Fault Current [kA]'].min()
            min = round(min, 5)

            switch_id = df.index[round(df['Fault Current [kA]'], 5) == min].tolist()

            for i in range(len(switch_id)):
                if ( min < relay_settings_IDMT.loc[switch_id[i], 'I_s[kA]']):
                    relay_settings_IDMT.loc[switch_id[i], 'I_s[kA]'] = min

        net.sgen.in_service = True

    return net, relay_settings_IDMT

#====================================================================================================================================================================================

# def plot_histogram(y):
#
#     # An "interface" to matplotlib.axes.Axes.hist() method
#     plt.hist(y, bins='auto', density=True)
#     plt.xlim(xmin=-0.05, xmax=5.75)
#     plt.ylim(ymin=0, ymax=10)
#     plt.show()
#     plt.grid(axis='y', alpha=0.75)
#     plt.xlabel('Trip time (seconds)')
#     plt.ylabel('Frequency')
#     # plt.title('My Very Own Histogram')
#     plt.text(1, 1, r'$\mu=15, b=3$')
#     # Set a clean upper y-axis limit.
#     plt.ylim(1)
#     plt.show()

#====================================================================================================================================================================================

# def calculate_relative_time(data_blinding, data_no_blinding):
#
#         print('Calculating relative time')
#
#         for i in range(len(data_blinding)):
#             i_temp_data = data_blinding.iloc[i]
#             i_dres_location = i_temp_data[0]
#             i_dres_distribution = i_temp_data[1]
#             i_fault_location = i_temp_data[2]
#             i_times = i_temp_data[3]
#
#             for j in range(len(data_no_blinding)):
#                 j_temp_data = data_blinding.iloc[j]
#                 j_dres_location = j_temp_data[0]
#                 j_dres_distribution = j_temp_data[1]
#                 j_fault_location = j_temp_data[2]
#                 j_times = j_temp_data[3]
#
#                 print('i_times ',i_times)
#                 i_time = i_times[0]
#                 print('i_times 0 ', i_time[0])
#
#                 print('j_times ', j_times)
#                 j_time = j_times[0]
#                 print('j_times 0 ', j_time[0])
#
#                 if( i_dres_location == j_dres_location and
#                         i_dres_distribution == j_dres_distribution and
#                         i_fault_location == j_fault_location and
#                         i_time[0] == j_time[0]):
#
#                             print('diff ', i_time[1] - j_time[1])
#
#         return

#====================================================================================================================================================================================

