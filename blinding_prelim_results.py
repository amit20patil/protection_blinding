"""
Main file for Preliminary numerical results from the paper "Analysis of Protection Blinding in Renewable Penetrated Distribution Systems"
Author: Amit Dilip Patil
This code allows you replicate the results in Figure 4 of the paper.
Change the relevant parameters in Main() function
"""

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
numba = True
from pandapower.protection import oc_relay_model as oc_protection
from multiprocessing import Pool
import multiprocessing as mp
from tqdm import tqdm
from contextlib import contextmanager
import sys, os
from pandapower.shortcircuit import calc_sc
from pandapower.pypower.idx_bus_sc import IKCV


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def power_network():
    """
    This function initializes the power system
    """

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
    net.ext_grid["p_mw"] = 0.05
    net.line["endtemp_degree"] = 80

    switch0 = pp.create_switch(
        net, 0, 0, closed=True, et="l", type="CB_IDMT", name="s0"
    )
    switch1 = pp.create_switch(
        net, 1, 17, closed=True, et="l", type="CB_IDMT", name="s1"
    )
    switch2 = pp.create_switch(
        net, 2, 21, closed=True, et="l", type="CB_IDMT", name="s2"
    )
    switch3 = pp.create_switch(
        net, 5, 24, closed=True, et="l", type="CB_IDMT", name="s3"
    )

    """
    Comment all the switches below if only studying impact on feeder protection
    """
    # switch4 = pp.create_switch(
    #     net, 5, 5, closed=True, et="l", type='CB_IDMT', name="s4"
    # )
    # switch5 = pp.create_switch(
    #     net, 9, 9, closed=True, et="l", type='CB_IDMT', name="s5"
    # )
    # switch6 = pp.create_switch(
    #     net, 13, 13, closed=True, et="l", type='CB_IDMT', name="s6"
    # )
    # switch7 = pp.create_switch(
    #     net, 28, 28, closed=True, et="l", type='CB_IDMT', name="s7"
    # )

    net.switch["trip_time"] = 0

    for index, row in net.switch.iterrows():
        line_index = net.switch.at[index, "element"]
        net.line.at[line_index, "cb_index"] = index

    return net


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# ------------------------------------------------------------------------------------------------------------------------


def dres_penetration(net, dres_location_list, dres_capacity):
    """
    :param net: Power network
    :param dres_location_list: List of DRES location, i.e., bus indices for DRESs
    :param dres_capacity: List of DRES fault levels
    :return: Network with DRES integrated
    """

    """
    Reduce the fault level from external grid to represent decommissioning of synchronous generators
    """
    total_capacity = 0
    for i in range(len(dres_capacity)):
        total_capacity = total_capacity + dres_capacity[i]

    net.ext_grid["s_sc_max_mva"].at[0] = (
        net.ext_grid["s_sc_max_mva"].at[0] - total_capacity
    )

    pp.create_sgen(
        net,
        dres_location_list[0],
        p_mw=0.050,
        q_mvar=0.1,
        sn_mva=dres_capacity[0],
        type="WP",
        scaling=1.0,
        in_service=True,
        current_source=True,
        k=1.3,
    )
    pp.create_sgen(
        net,
        dres_location_list[1],
        p_mw=0.050,
        q_mvar=0.1,
        sn_mva=dres_capacity[1],
        type="WP",
        scaling=1.0,
        in_service=True,
        current_source=True,
        k=1.3,
    )
    pp.create_sgen(
        net,
        dres_location_list[2],
        p_mw=0.050,
        q_mvar=0.1,
        sn_mva=dres_capacity[2],
        type="WP",
        scaling=1.0,
        in_service=True,
        current_source=True,
        k=1.3,
    )
    pp.create_sgen(
        net,
        dres_location_list[3],
        p_mw=0.050,
        q_mvar=0.1,
        sn_mva=dres_capacity[3],
        type="WP",
        scaling=1.0,
        in_service=True,
        current_source=True,
        k=1.3,
    )

    return net


# ------------------------------------------------------------------------------------------------------------------------


def baseline_settings(net):
    """
    :param net: Power network
    :return: Default settings of the protection devices based on fault current from only the external grid
    """


    net.sgen.in_service = False
    tripping_time = pd.DataFrame(
        {
            "switch_id": [0, 1, 2, 3, 4, 5, 6, 7],
            "tms": [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "t_grade": [3.5, 2, 2, 2, 1.75, 1.5, 1.0, 1.75],
        }
    )

    with suppress_stdout():
        # initial settings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            relay_settings_IDMT = oc_protection.oc_parameters(
                net,
                time_settings=tripping_time,
                relay_type="IDMT",
                curve_type="standard_inverse",
            )

        # Finding corrected settings

        for i in range(len(net.line)):
            trip_decisions_IDMT, net_sc = oc_protection.run_fault_scenario_oc(
                net, sc_line_id=i, sc_location=0.5, relay_settings=relay_settings_IDMT
            )

            df = pd.DataFrame(
                trip_decisions_IDMT,
                columns=[
                    "Switch ID",
                    "Switch Type",
                    "Trip type",
                    "Trip",
                    "Fault Current [kA]",
                    "Trip time [s]",
                ],
            )
            print("df", df)
            min = df.loc[df["Fault Current [kA]"] > 0, "Fault Current [kA]"].min()
            min = round(min, 5)
            print("min", min)
            switch_id = df.index[round(df["Fault Current [kA]"], 5) == min].tolist()

            for i in range(len(switch_id)):
                if min < relay_settings_IDMT.loc[switch_id[i], "I_s[kA]"]:
                    relay_settings_IDMT.loc[switch_id[i], "I_s[kA]"] = min

        # print(relay_settings_IDMT)
        net.sgen.in_service = True

    return net, relay_settings_IDMT


# ------------------------------------------------------------------------------------------------------------------------

def Main():

    """
    Specify the fault location, DRES location, capacities (fault level)
    """
    fault_location = 18
    dres_location = [1, 18, 22, 25]
    dres_cap = (3, 1, 1, 1)

    """
    Depending on the location of the fault, the index of the primary CB will change. 
    Those indices will have to be changed below to calculate the blinding time.
    e.g., for fault location 18, the index below should be 2. 
    """
    index_wo_sgen = 1
    index_w_sgen = 1

    net = power_network()
    net = dres_penetration(net, dres_location, dres_cap)
    net, relay_settings_IDMT = baseline_settings(net)

    """
    Don't forget to check state of sgens
    """
    net.sgen.in_service = False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        trip_decisions_IDMT, net_sc = oc_protection.run_fault_scenario_oc(
            net,
            sc_line_id=fault_location,
            sc_location=0.5,
            relay_settings=relay_settings_IDMT,
        )

    trip_times_with_inf = []
    for l in range(len(net.switch)):
        temp = trip_decisions_IDMT[l]["Trip time [s]"]
        temp_pair = (l, temp)
        trip_times_with_inf.append(temp_pair)

    trip_times = pd.DataFrame(trip_times_with_inf)
    with pd.option_context("mode.use_inf_as_null", True):
        trip_times = trip_times.dropna()
    print(trip_times)

    times = []
    trip_index_time = []
    trip_times.columns = ["index", "time"]
    times.extend(trip_times["time"].tolist())
    for m in range(len(trip_times)):
        x = trip_times.at[trip_times.index[m], "index"]
        y = trip_times.at[trip_times.index[m], "time"]
        trip_index_time.append(tuple((x, y)))

    """
    Don't forget to check state of sgens
    """
    net.sgen.in_service = True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        trip_decisions_IDMT, net_sc = oc_protection.run_fault_scenario_oc(
            net,
            sc_line_id=fault_location,
            sc_location=0.5,
            relay_settings=relay_settings_IDMT,
        )
    trip_times_with_inf = []
    for l in range(len(net.switch)):
        temp = trip_decisions_IDMT[l]["Trip time [s]"]
        temp_pair = (l, temp)
        trip_times_with_inf.append(temp_pair)

    trip_times = pd.DataFrame(trip_times_with_inf)
    with pd.option_context("mode.use_inf_as_null", True):
        # trip_times = trip_times.dropna()
        trip_times = trip_times.fillna(9999)

    times = []
    trip_index_time_w_sgen = []
    trip_times.columns = ["index", "time"]
    times.extend(trip_times["time"].tolist())
    for m in range(len(trip_times)):
        x = trip_times.at[trip_times.index[m], "index"]
        y = trip_times.at[trip_times.index[m], "time"]
        trip_index_time_w_sgen.append(tuple((x, y)))

    """
    Compare to the trip times with the ones below
    """
    print("w/o sgen: ", trip_index_time)
    print("w/ sgen: ", trip_index_time_w_sgen)

    x = trip_index_time[index_wo_sgen]
    x = x[1]

    y = trip_index_time_w_sgen[index_w_sgen]
    y = y[1]

    print("Difference (w/sgen - w/o sgen): ", y - x)


if __name__ == "__main__":
    Main()
