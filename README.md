# Preliminary results
The preliminary results can be replicated using "blinding_prelim_results.py".
The DRES location and capacity must be changed accordingly.
The index of the switch in the pandapower list should be set to primary protection device index for a fault.

# Case study I
To run the code, open "main_blinding_v5.py".
Change the variable sn_mva in this file and "histogram.py" to obtain the histogram.


# Case study II
The respective files are in the folder with the relevant name.
To run the code, open "main_blinding_v5.py".
Change the variable sn_mva in this file and "heatmap_paper_v3.py" to obtain the heatmap.


# Case study III
Same as case study II

# Case study IV
Same as case study II

# Case study V
add values for r_fault_ohm and/or x_fault_ohm in function calc_sc() under "run_fault_scenario_sc()"

# Case study VI
Open run_fault_scenario_oc() -> open calc_sc()
Change fault argument in cal_sc for different type of fault.

Note: 
1) Due to random sampling, the resulting visualization may vary.
2) Some rows in heatmap_paper_v3.py may need to be dropped to obtain heatmaps with identical sizes.
