#File to extract network data
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
from Lib.collections import Counter
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
    switch3 = pp.create_switch(net, 5, 5, closed=True, et="l", type='CB_IDMT', name="s3")
    switch4 = pp.create_switch(net, 5, 24, closed=True, et="l", type='CB_IDMT', name="s4")
    switch5 = pp.create_switch(net, 9, 9, closed=True, et="l", type='CB_IDMT', name="s5")
    switch6 = pp.create_switch(net, 13, 13, closed=True, et="l", type='CB_IDMT', name="s6")
    switch7 = pp.create_switch(net, 28, 28, closed=True, et="l", type='CB_IDMT', name="s7")

    net.switch['trip_time'] = 0

    for index, row in net.switch.iterrows():
        line_index = net.switch.at[index, 'element']
        net.line.at[line_index, "cb_index"] = index

    return net

#====================================================================================================================================================================================

def network_info(net):

    #num_feeder = 4
    num_bus_feeders = []
    num_line_feeders = []
    num_feeder_hops = [0] # 0 for the main feeder
    feeder_merge_bus_index = [0] #for the main feeder
    previous_line = 0
    previous_bus = 0

    for i in range(len(net.line)):
        if net.line.at[i, 'in_service'] == True:
            tb = net.line.at[i, 'to_bus']
            fb_next = net.line.at[i+1, 'from_bus']
            if tb != fb_next:
                num_line_feeders.append(i+1 - previous_line)    # We consider i+1 as total number of lines; For actual pandapower indices, use n-1, i.e., i
                num_bus_feeders.append(net.line.at[i, 'to_bus'] + 1 - previous_bus)
                previous_line = i + 1
                previous_bus = net.line.at[i, 'to_bus'] + 1
                if net.line.at[i+1, 'in_service'] == True:
                    num_feeder_hops.append(fb_next+1)
                    feeder_merge_bus_index.append(net.line.at[i + 1, 'from_bus'])

    switch_locations = []
    for i in range(len(net.switch)):
        switch_locations.append(net.switch.at[i, 'element'])
    switch_locations.sort()

    switches_in_feeders = []
    for i in range(len(num_line_feeders)):
        threshold = sum(num_line_feeders[:i + 1])
        sublist = [x for x in switch_locations if x < threshold and x not in sum(switches_in_feeders, [])]
        switches_in_feeders.append(sublist)

    for sublist_index in range(1, len(switches_in_feeders)):
        line_index = switches_in_feeders[sublist_index]
        from_bus_index = net.line.at[line_index[0], 'from_bus']     #Gives the origin bus of the first line in this feeder
        previous_line_index = net.line.index[net.line['to_bus'] == from_bus_index].tolist() #Finds all lines which where to_bus is the same as from_bus
        less_than_x_values = [value for value in switch_locations if value < previous_line_index]
        if less_than_x_values:
            switches_in_feeders[sublist_index].insert(0, less_than_x_values[0])
        else: switches_in_feeders[sublist_index].insert(0, 0)

    cumulative_line_feeders = []
    total = 0
    for i in range(len(num_line_feeders)):
        x = num_line_feeders[i]
        total = total + x
        cumulative_line_feeders.append(total)

    return num_line_feeders, num_bus_feeders, num_feeder_hops, switch_locations, switches_in_feeders, cumulative_line_feeders, feeder_merge_bus_index

#============================================================================================================================================================================

def find_feeder(cumulative_line_feeders, fault_location):

    total = 0
    fault_feeder_index = 0
    for i in range(len(cumulative_line_feeders)):
        if fault_location < cumulative_line_feeders[i]:
            fault_feeder_index = i
            break
        else:
            total = total + cumulative_line_feeders[i]
    return fault_feeder_index

#============================================================================================================================================================================

def find_switch_for_fault(net, switch_locations, fault_location, num_line_feeders):

    switch_locations.append(fault_location)
    switch_locations.sort()

    """
    Find switches by feeders; Gives a list of lists, where each sublist contains the swictches in the feeder of the index
    """
    switches_in_feeders = []
    for i in range(len(num_line_feeders)):
        threshold = sum(num_line_feeders[:i + 1])
        sublist = [x for x in switch_locations if x < threshold and x not in sum(switches_in_feeders, [])]
        switches_in_feeders.append(sublist)

    for sublist_index in range(1, len(switches_in_feeders)):
        line_index = switches_in_feeders[sublist_index]
        from_bus_index = net.line.at[line_index[0], 'from_bus']     #Gives the origin bus of the first line in this feeder
        previous_line_index = net.line.index[net.line['to_bus'] == from_bus_index].tolist() #Finds all lines which where to_bus is the same as from_bus
        less_than_x_values = [value for value in switch_locations if value < previous_line_index[0]]
        if less_than_x_values:
            switches_in_feeders[sublist_index].insert(0, less_than_x_values[0])
        else: switches_in_feeders[sublist_index].insert(0, 0)

    return switches_in_feeders
#============================================================================================================================================================================

def find_info_about_case(net, switch_locations, original_switch_locations, num_feeder_hops, num_line_feeders, fault_location):

    with suppress_stdout():
        """
        Find primary and backup relays
        1) Find feeder of the fault and CBs in that feeder
        """
        switch_locations = original_switch_locations.copy()
        switches_in_feeders_with_fault = find_switch_for_fault(net, switch_locations, fault_location, num_line_feeders)
        print('switches_in_feeders_with_fault: ', switches_in_feeders_with_fault)
        i = 0 #gives index of feeder
        for sublist in switches_in_feeders_with_fault:
            if fault_location in sublist:
                break
            else: i = i + 1
        fault_feeder_id = i
        value = switches_in_feeders_with_fault[fault_feeder_id]
        print('Feeder id and CBs: ',fault_feeder_id, value)

        """
        2) Find precisely the two relays
        """

        fault_index = value.index(fault_location, 0, len(value))
        fault_at_CB = 0
        CBs_in_same_feeder = 0

        print('original_switch_locations: ', original_switch_locations)
        frequency = Counter(original_switch_locations)
        if (frequency[fault_location] > 0):
            fault_at_CB = 1

        if fault_at_CB:
            primary_index = value[fault_index]
            if primary_index == 0:
                secondary_index = 0
            else:
                secondary_index = value[fault_index - 1]

        else:
            value.pop(fault_index)
            primary_index = value[fault_index - 1]
            if primary_index == 0:
                secondary_index = 0
            else:
                secondary_index = value[fault_index - 2]

        print('Indices: ', primary_index, secondary_index)

        #Eliminate all values that come after primary index to obtain actual path

        index = value.index(primary_index)
        value = value[:index+1]

    # Given: fault_location, primary index, secondary index, net.line dataframe
    primary_hop = 0
    secondary_hop = 0

    if primary_index == fault_location:
        primary_hop = 0
    else:
        for i in range(fault_location, primary_index, -1):
            from_index = net.line.at[i, 'from_bus']
            to_index = net.line.at[i - 1, 'to_bus']
            if from_index == to_index:
                primary_hop = primary_hop + 1
            else:
                break

    if CBs_in_same_feeder == 0:
        secondary_hop = num_feeder_hops[fault_feeder_id] + primary_hop
    else:
        for i in range(fault_location, secondary_index, -1):
            from_index = net.line.at[i, 'from_bus']
            to_index = net.line.at[i - 1, 'to_bus']
            if from_index == to_index:
                secondary_hop = secondary_hop + 1
            else:
                break

    hops = [primary_hop, secondary_hop]

    return primary_index, secondary_index, value, hops

#============================================================================================================================================================================

def check_necessary_condition(net, fault_location, dres_location):
    """
    switch has bus index (k)
    DRES has bus index (l)
    fault line has to_bus (i) and from_bus (j)
    condition: k < l < i,j iff all in same feeder
    Don't consider faults at switch lines (remove switch_location from fault_loc_list)
    """
    #net = power_network()
    num_line_feeders, num_bus_feeders, num_feeder_hops, switch_locations, switches_in_feeders, cumulative_line_feeders, feeder_merge_bus_index = network_info(net)
    original_switch_locations = switch_locations.copy()

    if fault_location not in original_switch_locations:
        """
        Collect all necessary indices
        """
        fault_feeder_index = find_feeder(cumulative_line_feeders, fault_location)
        fault_line_to_bus = net.line.at[ fault_location, 'to_bus']
        fault_line_from_bus = net.line.at[fault_location, 'from_bus']
        dres_bus_index = dres_location[fault_feeder_index]  #Assumes only one dres location in this feeder
        primary_index, secondary_index, path, hops = find_info_about_case(net, switch_locations, original_switch_locations, num_feeder_hops, num_line_feeders, fault_location)
        switch_index = net.switch[net.switch['element'] == primary_index].index.values     #Returned as a list
        switch_index = switch_index[0]
        switch_bus_index = net.switch.at[ switch_index, 'bus']      #Obtains bus index where the primary protection device is connected

        feeder_merge_bus_index.pop(0)                               #No feeder merges at bus 0; specific to 33 bus system
        if fault_feeder_index == 0:
            feeder_merge_bus_index = feeder_merge_bus_index[0]
            feeder_merge_bus_index = [feeder_merge_bus_index]
        else:
            feeder_merge_bus_index = feeder_merge_bus_index[:fault_feeder_index]

        print(switch_bus_index , dres_bus_index , fault_line_from_bus)

        if( switch_bus_index < dres_bus_index <= fault_line_from_bus):
            print('Necessary condition satisfied')
            return 1
        elif any(switch_bus_index < element <= fault_line_from_bus for element in feeder_merge_bus_index):
            print('Necessary condition satisfied: external feeder')
            return 1
        else:
            print('Necessary condition NOT satisfied')
            return 0
    else:
        print('Necessary condition NOT satisfied: is switch')

        return 0

#============================================================================================================================================================================

def find_bus_feeder(num_bus_feeders, fault_feeder_index, dres_locations, dres_cap_original, feeder_merge_bus_index, fault_location):

    """
    :param num_bus_feeders: Number of buses in each feeder
    :param fault_feeder_index: Index f the faulty feeder
    :param dres_loc: Bus location of the dres
    :param dres_cap_original: Original dres capacity distribution
    :param feeder_merge_bus_index: Buses at which the feeders merge with original feeder
    :param fault_location: Line index of the fault
    :return:
    """

    """
    Obtain the network and build the list with buses in a cumulative manner
    """
    dres_locations_original = dres_locations.copy()
    sgen_locations = dres_locations.copy()
    new_net = dres_scenario.power_network()
    net = dres_scenario.dres_penetration(new_net, dres_locations, dres_cap_original)
    cumulative_bus_feeders = []
    total = 0
    for i in range(len(num_bus_feeders)):
        x = num_bus_feeders[i]
        total = total + x
        cumulative_bus_feeders.append(total)
    #print('test:', cumulative_bus_feeders)

    """
    Find the buses of the faulty feeder to identify the dres bus
    """
    if fault_feeder_index == 0:
        start_bus_index = 0
        end_bus_index = cumulative_bus_feeders[fault_feeder_index] - 1  #Gives the INDEX of the last bus
        buses = 0
        max_buses = cumulative_bus_feeders[fault_feeder_index]          #Gives the NUMBER of buses

    else:
        start_bus_index = cumulative_bus_feeders[fault_feeder_index] - num_bus_feeders[fault_feeder_index]
        end_bus_index = cumulative_bus_feeders[fault_feeder_index] - 1
        buses = cumulative_bus_feeders[fault_feeder_index-1]
        max_buses = cumulative_bus_feeders[fault_feeder_index]
    #print('test:', start_bus_index, end_bus_index, buses, max_buses)

    """
    WE KNOW THE FEEDER INDEX OF THE FAULT: CAN REMOVE ALL THOSE INDICES FROM sgens WHICH ARE NOT IN THIS FEEDER
    """
    sgens = feeder_merge_bus_index + list(set(sgen_locations) - set(feeder_merge_bus_index))
    if dres_locations[0] != 0:
        sgens.pop(0) #There is no sgen at bus 0, this value is in feeder_merge_bus_index for indexing purposes
    sgens.sort()
    sgens_sorted = [x for x in sgens if buses < x < max_buses]
    #print('test:', sgens_sorted)

    """
    Find the DRES index between fault and start of feeder
    """
    from_bus_index = net.line.at[fault_location, 'from_bus']
    dres_location = 0
    for i in range(len(sgens_sorted)):
        index = sgens_sorted[i]
        if start_bus_index <= index <= from_bus_index:
            dres_location = index   #Could hold index of bus in main feeder where another feeder merges
    #print('test:', dres_location)

    """
    If DRES not from same feeder, find in parallel feeder
    """
    exist_count = dres_locations_original.count(dres_location)  #If count is 0, blinding is due to DRES in parallel feeder
    if exist_count == 0:
        index = feeder_merge_bus_index.index(dres_location)
        dres_location = dres_locations_original[index]

    return dres_location

#=================================================================================================================================================================================

def find_line_feeder(cumulative_line_feeders, fault_location):
    total = 0
    fault_feeder_index = 0
    for i in range(len(cumulative_line_feeders)):
        if fault_location < cumulative_line_feeders[i]:
            fault_feeder_index = i
            break
        else:
            total = total + cumulative_line_feeders[i]
    return fault_feeder_index

#============================================================================================================================================================================

