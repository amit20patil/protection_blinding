#Code for I/O operations with picklt files
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

#====================================================================================================================================================================================

def create_files(sn_mva, num_buses, capacity_list_split, location_list_split):
    # Initialize pickle filename
    filename_blinding = 'data/data_blinding_' + str(capacity_list_split)  + str(location_list_split)
    filename_blinding = filename_blinding + str(sn_mva) + '_' + str(num_buses)
    blinding_data = pd.DataFrame(columns=['dres_location', 'dres_capacity', 'fault_location', 'trip_times', 'settings', 'fault current'])
    blinding_data = blinding_data.astype('object')

    filename_no_blinding = 'data/data_no_blinding_' + str(capacity_list_split)  + str(location_list_split)
    filename_no_blinding = filename_no_blinding + str(sn_mva) + '_' + str(num_buses)
    no_blinding_data = pd.DataFrame(columns=['dres_location', 'dres_capacity', 'fault_location', 'trip_times', 'settings', 'fault current'])
    no_blinding_data = no_blinding_data.astype('object')

    filename_relative = 'data/data_relative_' + str(capacity_list_split)  + str(location_list_split)
    filename_relative = filename_relative + str(sn_mva) + '_' + str(num_buses)
    relative_data = pd.DataFrame(columns=['dres_location', 'dres_capacity', 'fault_location', 'primary_index', 'primary_trip_time', 'primary_trip_time_blind', 'primary_delta_trip_time',
                                          'tripped_index', 'tripped_time', 'tripped_time_blind','delta_trip_times'])
    relative_data = relative_data.astype('object')

    return blinding_data, no_blinding_data, filename_blinding, filename_no_blinding, relative_data, filename_relative

#====================================================================================================================================================================================

def file_read(filename):

    """"
    Read pickle file with results
    """
    print('File read')
    with open('%s.pkl' % filename,  'rb') as f:
       data = pickle.load(f)

    trip_times = []
    relay_id = []
    trip_time = []
    for i in range(len(data)):
        trip_times.append(data.at[i, 'trip_times'])

    for i in range(len(trip_times)):
        temp = trip_times[i]
        if isinstance(temp, list):
            length = len(temp)
            for pairs in range(length):
                pair = temp[pairs]
                relay_id.append(pair[0])
                trip_time.append(pair[1])
        elif(temp > 0):
            trip_time.append(temp)

    print('Mean: ', mean(trip_time))
    print('Min: ', min(trip_time))
    print('Max: ', max(trip_time))
    print('Length: ', len(trip_time))

#====================================================================================================================================================================================


def read_blinding_files(sn_mva, num_buses):

    filename_relative = 'data/data_blinding_'

    filename_relative = filename_relative + str(sn_mva[0]) + '_' + str(num_buses)
    print('File read ', filename_relative)
    with open('%s.pkl' % filename_relative, 'rb') as f:
        data_sn_mva0 = pickle.load(f)

    #filename_relative = 'data/data_relative_'
    #filename_relative = 'data/data_no_blinding_'
    filename_relative = 'data/data_blinding_'
    filename_relative = filename_relative + str(sn_mva[1]) + '_' + str(num_buses)
    print('File read ', filename_relative)
    with open('%s.pkl' % filename_relative, 'rb') as f:
        data_sn_mva1 = pickle.load(f)

    #filename_relative = 'data/data_relative_'
    #filename_relative = 'data/data_no_blinding_'
    filename_relative = 'data/data_blinding_'
    filename_relative = filename_relative + str(sn_mva[2]) + '_' + str(num_buses)
    print('File read ', filename_relative)
    with open('%s.pkl' % filename_relative, 'rb') as f:
        data_sn_mva2 = pickle.load(f)

    #filename_relative = 'data/data_relative_'
    #filename_relative = 'data/data_no_blinding_'
    filename_relative = 'data/data_blinding_'
    filename_relative = filename_relative + str(sn_mva[3]) + '_' + str(num_buses)
    print('File read ', filename_relative)
    with open('%s.pkl' % filename_relative, 'rb') as f:
        data_sn_mva3 = pickle.load(f)

    return data_sn_mva0, data_sn_mva1, data_sn_mva2, data_sn_mva3

#============================================================================================================================================================================
