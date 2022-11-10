## Distributions of HRC outputs


"""
Get all HRC output for all data types. This is for plotting the histogram of the HRC output.
"""


import sys
sys.path.append('../stimulus_core/')
sys.path.append('../models_core/')
sys.path.append('../helper/')

import os
import numpy as np

import helper_functions as hpfn


figure_path = '/Volumes/Baohua/research/loom_detection/results/final_figures_for_paper_exp/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

M = 64
set_number = np.int(1000 + M)
data_path = '/Volumes/Baohua/data_on_hd/loom/multi_lplc2_D5_L4_exp/set_{}/'.format(set_number)
data_types = ['hit', 'miss', 'retreat', 'rotation']

V_hit, V_miss, V_retreat, V_rotation = hpfn.get_velocity_nonzero(data_path, data_types)

save_path = figure_path + '/hrc_tuning/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

np.savez(save_path+f'HRC_output_M{M}', V_hit, V_miss, V_retreat, V_rotation)