import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import file_io

sn_mva = [6, 8, 10, 12]
num_buses = 33
location_list_split = 1
capacity_list_split = 0.0

#============================================================================================================================================================================

def plot_histogram(trip_time0, trip_time1, trip_time2, trip_time3):

    random.shuffle(trip_time0)
    random.shuffle(trip_time1)
    random.shuffle(trip_time2)
    random.shuffle(trip_time3)

    trip_time0 = trip_time0[0:1000]
    trip_time1 = trip_time1[0:1000]
    trip_time2 = trip_time2[0:1000]
    trip_time3 = trip_time3[0:1000]

    data = [trip_time0, trip_time1, trip_time2, trip_time3]

    # plt.hist(trip_time3, bins=np.logspace(start=np.log10(0.01), stop=np.log10(100), num=20), density=True, label="50 % mva", alpha=0.25)
    # plt.hist(trip_time2, bins=np.logspace(start=np.log10(0.01), stop=np.log10(100), num=20), density=True, label="42 % mva", alpha=0.25)
    # plt.hist(trip_time1, bins=np.logspace(start=np.log10(0.01), stop=np.log10(100), num=20), density=True, label="33 % mva", alpha=0.25)
    # plt.hist(trip_time0, bins=np.logspace(start=np.log10(0.01), stop=np.log10(100), num=20), density=True, label="25 % mva", alpha=0.25)

    # z = sns.histplot(data=trip_time0,  color="orange", kde=True, log_scale=True, label="25 % mva")
    # z = sns.histplot(data=trip_time1, color="blue", kde=True, log_scale=True, label="33 % mva")
    # z = sns.histplot(data=trip_time2, color="green", kde=True, log_scale=True, label="42 % mva")
    # z = sns.histplot(data=trip_time3, color="red", kde=True, log_scale=True, label="50 % mva")

    # names = ["25 % mva", "33 % mva", "42 % mva", "50 % mva"]
    # plt.tight_layout()
    # plt.legend()
    # plt.xlabel(u'Δt Blinding time of protection device (seconds)')
    # plt.ylabel('Frequency')
    # #plt.xlim(0, 4)
    # plt.ylim(0.01,100)
    # plt.gca().set_xscale("log")
    # plt.gca().set_yscale("log")
    # plt.savefig('histogram.pdf', dpi=2000, format='pdf', bbox_inches='tight')
    # plt.show()
    #
    # plt.hist(data, bins=np.logspace(start=np.log10(0.01), stop=np.log10(100), num=20), stacked=True, label=names)
    # plt.gca().set_xscale("log")
    # plt.gca().set_yscale("log")
    # plt.legend()
    # plt.xlabel(u'Δt Blinding time of protection device (seconds)')
    # plt.ylabel('Frequency')
    # plt.savefig('histogram.pdf', dpi=2000, format='pdf', bbox_inches='tight')
    # plt.show()

    colors = ['#E69F00', '#56B4E9', '#F0E442', '#009E73']
    names = ["25 % mva", "33 % mva", "42 % mva", "50 % mva"]
    #hatch = ['/', '-', 'x', 'o']
    #plt.hist([trip_time0, trip_time1, trip_time2, trip_time3], bins=np.logspace(start=np.log10(0.01), stop=np.log10(100), num=20), color = colors, label=names)
    plt.hist([trip_time0, trip_time1, trip_time2, trip_time3], bins=np.logspace(start=np.log10(1), stop=np.log10(50), num=10), color=colors, label=names)

    # Plot formatting
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    plt.xlim(0, 100)
    plt.legend()
    plt.xlabel(u'Δt Blinding time of protection device (seconds)')
    plt.ylabel('Frequency')
    #plt.title('Side-by-Side Histogram with Multiple Airlines')
    plt.savefig('histogram.pdf', dpi=2000, format='pdf', bbox_inches='tight')
    plt.show()
#====================================================================================================================================================================================

def read_blinding_files(sn_mva, num_buses):

    filename_relative = 'data/data_relative_' + str(capacity_list_split) + str(location_list_split)
    filename_relative = filename_relative + str(sn_mva) + '_' + str(num_buses)
    print('File read ', filename_relative)
    with open('%s.pkl' % filename_relative, 'rb') as f:
        data = pickle.load(f)

    return data

#============================================================================================================================================================================

def Main():

    data_sn_mva0 = read_blinding_files(sn_mva[0], num_buses)
    data_sn_mva1 = read_blinding_files(sn_mva[1], num_buses)
    data_sn_mva2 = read_blinding_files(sn_mva[2], num_buses)
    data_sn_mva3 = read_blinding_files(sn_mva[3], num_buses)

    data_sn_mva0 = data_sn_mva0[ data_sn_mva0['delta_trip_times'] >= 0]
    data_sn_mva1 = data_sn_mva1[data_sn_mva1['delta_trip_times'] >= 0]
    data_sn_mva2 = data_sn_mva2[data_sn_mva2['delta_trip_times'] >= 0]
    data_sn_mva3 = data_sn_mva3[data_sn_mva3['delta_trip_times'] >= 0]

    trip_time0 = data_sn_mva0['delta_trip_times'].tolist()
    trip_time1 = data_sn_mva1['delta_trip_times'].tolist()
    trip_time2 = data_sn_mva2['delta_trip_times'].tolist()
    trip_time3 = data_sn_mva3['delta_trip_times'].tolist()

    #print(trip_time0)
    plot_histogram(trip_time0, trip_time1, trip_time2, trip_time3)

if __name__ == "__main__":
    Main()
#============================================================================================================================================================================
