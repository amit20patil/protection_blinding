#Plotting data as heatmap
#v3 extends the x-axis representation: Uses normalized impedance values
#Author: Amit Dilip Patil

import itertools
import random
import pandapower as pp
import dres_scenario
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import warnings
from statistics import mean
numba = True
import numpy as np
import matplotlib.colors as mcolors
import matplotlib
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import seaborn as sns
import file_io
import sys, os
from contextlib import contextmanager
import pandas as pd
from pandas import DataFrame
import dres_scenario
from collections import Counter
from tqdm import tqdm
#import heat_map_supplement
from electrical_distance import get_equivalent_impedance as impedance
import network_data

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

"""
Define variables
"""
k1 = 0
k2 = 0
heterogeneity = 0
minimum = 0
sn_mva = 6
homogeneous_capacity = [0.25, 0.25, 0.25, 0.25]
fault_location = 0
trip_times = 0
primary_index = 0
secondary_index = 0
primary_hop = 0
secondary_hop = 0
ymin = 0
#ymax = sn_mva
ymax = 6   #6 for 6, 8 for 8, 9 for 10, 11 for 12
location_list_split = 1 # No. of instances to keep: CHANGE RANDOM NUMBER SEED FOR ANOTHER SET
capacity_list_split = 0.0   # % of set to remove
num_buses = 33
fontsize = 10
hmap_time_limit = 30

#====================================================================================================================================================================================

def read_blinding_files(sn_mva):

    filename_relative = 'data/data_relative_' + str(capacity_list_split) + str(location_list_split)
    filename_relative = filename_relative + str(sn_mva) + '_' + str(num_buses)
    print('File read ', filename_relative)
    with open('%s.pkl' % filename_relative, 'rb') as f:
        data = pickle.load(f)

    return data

#============================================================================================================================================================================

def normalized_distance(net, fault_from_bus_index, dres_locations):

    z_fault_all_dres = []
    for i in range(len(dres_locations)):
        z = impedance(net, dres_locations[i], fault_from_bus_index)
        z = abs(z)
        z_fault_all_dres.append(z)

    normalized_value_list = []
    total = sum(z_fault_all_dres)

    for i in range(len(dres_locations)):
        normalized_value = z_fault_all_dres[i]/total
        normalized_value = round(normalized_value, 3)
        normalized_value_list.append(normalized_value)

    return normalized_value_list, total

#============================================================================================================================================================================
def hop_count(net, primary_index, dres_location):
    hop = 0

    if primary_index == dres_location:
        hop = 0
    else:
        for i in range(dres_location, primary_index, -1):
            from_index = net.line.at[i, 'from_bus']
            to_index = net.line.at[i - 1, 'to_bus']
            if from_index == to_index:
                hop = hop + 1
            else:
                break

    return hop
#============================================================================================================================================================================

def extract_data(net, data, i, sn_mva):

    """
    Extract blinding data from previous simulations
    filename_relative = 'data/data_relative_' + str(capacity_list_split)  + str(location_list_split)
    filename_relative = filename_relative + str(sn_mva) + '_' + str(num_buses)
    relative_data = pd.DataFrame(columns=['dres_location', 'dres_capacity', 'fault_location', 'primary_index', 'primary_trip_time', 'primary_trip_time_blind', 'primary_delta_trip_time',
                                          'tripped_index', 'tripped_time', 'tripped_time_blind','delta_trip_times'])

    """
    temp = data.iloc[i]
    dres_locations = temp[0]
    dres_cap_original = temp[1]
    dres_cap_normalized = [x/sn_mva for x in dres_cap_original]
    fault_location = temp[2]
    primary_index = temp[3]
    primary_delta_trip_time = temp[6]
    delta_trip_time = temp[10]

    """
    Obtain info about the network
    """
    num_line_feeders, num_bus_feeders, num_feeder_hops, switch_locations, switches_in_feeders, cumulative_line_feeders, feeder_merge_bus_index = network_data.network_info(net)
    original_switch_locations = switch_locations.copy()

    """
    Find feeder of the fault
    """
    fault_feeder_index = network_data.find_line_feeder(cumulative_line_feeders, fault_location)
    num_feeders = len(num_bus_feeders)
    fault_feeder_list = [0] * num_feeders

    # Check if the index is within the valid range of the list
    if 0 <= fault_feeder_index < len(fault_feeder_list):
        fault_feeder_list[fault_feeder_index] = 1

    primary_index, secondary_index, value, hops = network_data.find_info_about_case(net, switch_locations, original_switch_locations, num_feeder_hops, num_line_feeders, fault_location)

    """
    Find index of the dres causing the blinding
    """
    dres_bus_index = network_data.find_bus_feeder(num_bus_feeders, fault_feeder_index, dres_locations, dres_cap_original, feeder_merge_bus_index, fault_location)
    #dres_bus_index = 1

    """
    Find index of the bus upstream of the fualt
    """
    fault_from_bus_index = net.line.at[fault_location, 'from_bus']

    """
    Find bus to which primary protection device is attached
    """
    cond = (net.switch['element'] == primary_index)
    switch_bus_index = net.switch[cond].bus.values[0]

    """
    Find electrical distance between switch and fault AND dres and fault
    """
    z_fault_switch = impedance(net, switch_bus_index, fault_from_bus_index)
    z_dres_switch = impedance(net, dres_bus_index, switch_bus_index)
    z_dres_switch = abs(z_dres_switch)
    z_dres_switch = round(z_dres_switch, 4)

    """
    Find impedance of lines
    """
    r_line = net.line.at[fault_location, 'r_ohm_per_km']
    x_line = net.line.at[fault_location, 'x_ohm_per_km']
    z_line = math.sqrt(r_line * r_line + x_line * x_line) #Impedance per km
    z_length = net.line.at[fault_location, 'length_km']
    z_line = z_line * z_length

    z_fault_switch = abs(z_fault_switch) + z_line/2
    z_fault_switch = round(z_fault_switch, 2)

    """
    Use below code when the impact of multiple DRESs is considered, taking mean of impedance
    """
    z_dres_switch = 0
    switch_bus_locations = []
    for i in range(len(net.switch)):
        switch_bus_locations.append(net.switch.at[i, 'bus'])
    switch_bus_locations.sort()

    z_list = []
    for i in range(len(dres_locations)):
        z_dres_switch = impedance(net, dres_locations[i], switch_bus_locations[i])
        z_dres_switch = abs(z_dres_switch)
        z_dres_switch = round(z_dres_switch, 2)
        z_list.append(z_dres_switch)
    mean = sum(z_list) / len(z_list)
    z_normalized = ( mean / z_fault_switch )
    z_normalized = round(z_normalized, 4)

    """
    Use below code when the impact of single DRES is considered, change dres_bus_index
    """
    # z_normalized = ( z_dres_switch / z_fault_switch )
    # z_normalized = round(z_normalized, 4)

    """
    Select trip time for heatmap
    """
    if delta_trip_time > primary_delta_trip_time:
        trip_times = primary_delta_trip_time
    else:
        trip_times = delta_trip_time

    """
    Calculate heterogeneity index
    """

    #type4 dres_cap_original * ( x_hom - x_het ) * fault_feeder_index
    dres_cap_homogeneous = list(map(lambda z: z * sn_mva, (i for i in homogeneous_capacity)))
    X_i = [abs(a - b) for a, b in zip(dres_cap_homogeneous, dres_cap_original)]
    dot_product = sum([ (lambda x,y: x*y)(x,y) for x,y in zip(X_i, fault_feeder_list)])
    dot_product = dot_product/sn_mva
    x = round(dot_product, 2)
    heterogeneity = ( x * dres_cap_normalized[fault_feeder_index])
    heterogeneity = round(heterogeneity, 2)

    # print(switch_bus_locations, dres_locations)
    # print(z_list, mean)
    # print( z_dres_switch, z_fault_switch, z_normalized)
    # print(dres_bus_index, fault_from_bus_index, switch_bus_index, abs(impedance(net, switch_bus_index, dres_bus_index)))

    return heterogeneity, z_normalized, trip_times, dres_bus_index, switch_bus_index

#============================================================================================================================================================================


def plot_heatmap(data):

    x = [0.0] #for 6 (case 4)
    #x = [0.01, 0.38] #for 8
    #x = [ 0.01, 0.21, 0.59] #for 10
    #x = [0.01, 0.03, 0.19, 0.38, 0.61] #for 12
    column = 'het_index'
    data.drop(data[data[column].isin(x)].index, axis=0, inplace=True)

    """
    Normalize the trip times
    """
    column = 'trip_time'
    data[column] = (data[column] - data[column].min()) / ( data[column].max() - data[column].min())

    data = data.assign(
        PopGroup=pd.cut(data.distance, bins=[0.15, 0.2, 0.25, 0.3, np.inf], labels=['<100', '100-200', '200-300', '>300']))

    data = data.groupby(['het_index', 'distance'])['trip_time'].mean().reset_index()

    pivot = data.pivot(index='het_index', columns='distance', values='trip_time')

    fig, ax = plt.subplots()

    ax = sns.heatmap(pivot, cmap="YlGnBu", mask=pivot.isnull(),vmin=0, vmax=1, cbar_kws={'label': r'Normalized blinding time $\Delta t$'},
                     annot=False, xticklabels=1, yticklabels=1, annot_kws={"fontsize":fontsize})
    #ax.invert_yaxis()
    #ax.set_xlabel( r'Distance between fault and protection device $|Z_{f}^{pd}|$ (Ohm)')
    # ax.set_xlabel(r' Electrical distance ratio $\frac{ Z_{dres}^{cb}}{Z_{f}^{cb}}$')
    ax.set_xlabel('EDR')
    ax.set_ylabel('Heterogeneity index H')
    #ax.set_ylabel('Normalized DRES fault level')
    plt.ylim(ymin, ymax)

    # # set the ticks first
    #ax.set_yticks(range(7))
    # # set the labels
    ax.set_yticklabels(['0', '0.05', '0.1', '0.3', '0.5', '0.75'])

    filename = 'heatmap_' + str(sn_mva) + '.pdf'
    plt.savefig(filename, dpi=2000, format='pdf', bbox_inches='tight')
    plt.show()

    return


#============================================================================================================================================================================

def Main():
    """
    Initialize dataframe to store the data
    | k1, k2 | gen index | trip time |
    """
    heatmap_data = pd.DataFrame(columns=['distance', 'het_index', 'trip_time'])
    heatmap_data = heatmap_data.astype('int64')

    """
    Define power network
    """
    net = dres_scenario.power_network()
    data = read_blinding_files(sn_mva)

    print('len', len(data))
    orig_data = data.copy()

    # ==================================================================================================================
    """
    Extract required information for interpretation
    """
    pbar = tqdm(total=len(data))
    for i in range(len(data)):
        heterogeneity, distance, trip_times, dres_bus_index, switch_bus_index = extract_data(net, data, i, sn_mva)
        new_row = {'distance': distance, 'het_index': heterogeneity, 'trip_time': trip_times}

        heatmap_data = heatmap_data.append(new_row, ignore_index=True)
        pbar.update(1)
    pbar.close()

    filename = 'data/heat_map_data' + str(sn_mva)
    pickle.dump(heatmap_data, open('%s.pkl' % filename, 'wb'))
    # ==================================================================================================================

    filename = 'data/heat_map_data' + str(sn_mva)
    print('File read')
    with open('%s.pkl' % filename, 'rb') as f:
        data = pickle.load(f)

    data["trip_time"] = np.where(data["trip_time"] > hmap_time_limit, hmap_time_limit, data["trip_time"])
    data["trip_time"] = np.where(data["trip_time"] < 0, 0, data["trip_time"])
    plot_heatmap(data)

    extracted_col = data["distance"]
    orig_data = orig_data.join(extracted_col)
    extracted_col = data["het_index"]
    orig_data = orig_data.join(extracted_col)
    #print(orig_data)

    file_name = 'data.xlsx'
    orig_data.to_excel(file_name)

if __name__ == "__main__":
    Main()
#============================================================================================================================================================================

