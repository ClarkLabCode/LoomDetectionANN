#!/usr/bin/env python


"""
Here, we calculate the incoming theta angles of the isotropic hit signals.
"""


import sys
sys.path.append('../stimulus_core/')
sys.path.append('../models_core/')
sys.path.append('../helper/')

import os
import numpy as np

import optical_signal as opsg

figure_path = '/Volumes/Baohua/research/loom_detection/results/final_figures_for_paper_exp/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)
if not os.path.exists(figure_path+'theta_angles/'):
    os.makedirs(figure_path+'theta_angles/')

# data
M = 1
K = 12
L = 4
KK = K**2
pad = 2*L
set_number = np.int(1000 + M)
NNs = 10
data_path = '/Volumes/Baohua/data_on_hd/loom/multi_lplc2_D5_L4_exp/set_{}/'.format(set_number)
data_types = ['hit', 'miss', 'retreat', 'rotation']
data_type = data_types[0]
path_traj = data_path+'other_info'+'/'+data_type+'/'+'trajectories/'
path_dist = data_path+'other_info'+'/'+data_type+'/'+'distances/'

theta_angles = []
for sample_number in range(8000):
    sample_list_number = sample_number//NNs+1
    traj_list = np.load(path_traj+'traj_list_{}.npy'.format(sample_list_number), allow_pickle=True)
    dist_list = np.load(path_dist+'distance_list_{}.npy'.format(sample_list_number), allow_pickle=True)
    traj = traj_list[sample_number%NNs-1]
    dist = dist_list[sample_number%NNs-1]
    x, y, z = traj[-1, 0, :]
    vec = traj[-2, 0, :]-traj[-1, 0, :]
    angles = opsg.get_angles_between_lplc2_and_vec(M, vec)
    theta_angles.append(angles[0])
    
save_file = figure_path+'theta_angles/theta_angles'
np.save(save_file, theta_angles)