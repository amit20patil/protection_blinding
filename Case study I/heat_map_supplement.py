
import itertools
import multiprocessing
import pandapower as pp
import dres_scenario
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import warnings
import time
from pandapower.plotting import pf_res_plotly
from statistics import mean
import dres_scenario
numba = True
import numpy as np
import matplotlib.colors as mcolors
import matplotlib
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import seaborn as sns
import pandapower.networks as nw
import file_io
import pandas as pd
#====================================================================================================================================================================================
sn_mva = 6
hops = [2, 4]
ymin = 0
ymax = 10

filename_relative = 'data/data_relative_0.08' + str(sn_mva) + '_33'
print('File read ', filename_relative)
with open('%s.pkl' % filename_relative, 'rb') as f:
    data_sn_mva0 = pickle.load(f)

print(len(data_sn_mva0))
print(data_sn_mva0.iloc[41])

filename = 'data/heat_map_data' + str(sn_mva)
print('File read')
with open('%s.pkl' % filename, 'rb') as f:
    data = pickle.load(f)

#data_sn_mva0 = data_sn_mva0.loc[data_sn_mva0['dres_capacity'] == (5, 1, 0, 0)]
#print(data.iloc[9])
#print(data.iloc[100]['k1_k2'])
#print(data['trip_time'].max())
#print(data['het_index'].max())
# desired_rows = data[data['k1_k2'].apply(lambda x: x == hops)]
# print(desired_rows)

#df = data.loc[data['k1_k2'] == hops]
#print(df)

# for x in data['het_index']:
#     print(x)

data = data[(data['trip_time'] > 0)]
print(data)

heatmap_data = pd.crosstab(index=data['het_index'],
                           columns=[data['distance'].apply(lambda x: x[0]), data['distance'].apply(lambda x: x[1])],
                           values=data['trip_time'], aggfunc='first')

fig, ax = plt.subplots()
# Customize tick labels
y_ticks = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
#y_ticks = [0, 0.25, 0.5, 0.75, 1, 1.25]
#x_ticks = [0 - 0, 0 - 2, 0 - 3, 1 - 1, 2 - 2]
x_ticks = range(len(heatmap_data.columns))
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)

# Add labels and title
ax.set_xlabel('Pair')
ax.set_ylabel('Index')
plt.title('Heatmap of blinding time')
ax = sns.heatmap(heatmap_data, cmap="viridis", mask=heatmap_data.isnull(), cbar_kws={'label': 'Blinding time (seconds)'},
                 annot=True, xticklabels=1, yticklabels=1, vmin=0, vmax=30)
ax.invert_yaxis()
ax.set_xlabel('Pair')
ax.set_ylabel('Index')
plt.ylim(ymin, ymax)
plt.savefig('heatmap.pdf', dpi=2000, format='pdf', bbox_inches='tight')
plt.show()

#WHITE spaces => no blinding
