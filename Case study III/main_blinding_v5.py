#main file for optimization
#Author: Amit Dilip Patil

from itertools import chain
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
import new_blinding_loops as loops

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

num_buses = 33 #33
num_lines = 32 # 32
num_feeder = 4
num_buses_feeder1 = 18 #18
num_buses_feeder2 = 4 #4
num_buses_feeder3 = 3 #3
num_buses_feeder4 = 8 #8
sn_mva = 6   #Vary to break symmetry
location_list_split = 1 # % of set to remove; Can also be instances to keep, check dres_scenario
capacity_list_split = 0.0   # % of set to remove
parts = 1
dres_capacity_list = []
trip_time_dict = {}

times = []
setting = []
fault_current = []
#====================================================================================================================================================================================


def Main():

    print('Blinding impact')
    p = Pool(parts)

    """
    Create files and dataframes for data
    """
    blinding_data, no_blinding_data, filename_blinding, filename_no_blinding, relative_data, filename_relative = file_io.create_files(sn_mva, num_buses, capacity_list_split, location_list_split)

    """
    Create lists of dres location and assign capacity distribution
    """
    dres_location_list, dres_capacity_list = dres_scenario.generate_dres_lists( num_feeder, num_buses_feeder1, num_buses_feeder2, num_buses_feeder3, num_buses_feeder4, sn_mva, capacity_list_split, location_list_split)

    """
    Split location list into parts for parallelization
    """
    dres_loc_len = len(dres_location_list)
    n = dres_loc_len / parts
    n = math.ceil(n)
    x = [dres_location_list[i:i + n] for i in range(0, len(dres_location_list), n)]

    """
    Create argument list and call function
    """
    args = [[ x[i], dres_capacity_list, blinding_data, no_blinding_data, relative_data, num_lines] for i in range(parts)]
    with Pool(processes=parts) as pool:
        blinding_data, no_blinding_data, relative_data = zip(*p.starmap(loops.blinding_impact, args))
    p.close()
    p.join()

    """
    Store data in correct format
    """
    blinding_data = pd.concat(blinding_data)
    no_blinding_data = pd.concat(no_blinding_data)
    relative_data = pd.concat(relative_data)
    print('len: ', len(relative_data))

    """
    Store data in pickle file
    """
    pickle.dump(blinding_data, open('%s.pkl' % filename_blinding, 'wb'))
    pickle.dump(no_blinding_data, open('%s.pkl' % filename_no_blinding, 'wb'))
    pickle.dump(relative_data, open('%s.pkl' % filename_relative, 'wb'))

    print('File read ', filename_relative)
    with open('%s.pkl' % filename_relative, 'rb') as f:
        data = pickle.load(f)


    import heat_map_paper_v3
    heat_map_paper_v3.Main()

if __name__ == "__main__":
    Main()

#====================================================================================================================================================================================