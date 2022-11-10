#!/usr/bin/env python


import sys
sys.path.append('../stimulus_core/')
sys.path.append('../models_core/')
sys.path.append('../helper/')

import os
import numpy as np

import dynamics_3d as dn3d
import optical_signal as opsg
import samples_generation_multi_units as smgnmu
import flow_field as flfd
                                                                            

M = 1 # number of LPLC2 units
D = 10 # initial distance of the incoming object
D_max = 9 # maximum distance to travel
theta_max = 150 # maximum incoming angle
na = 9 # half dimension size of the lattice
R = 1 # radius of the ball
V = 5. * R # maximum speed (/sec)
dt = 0.01 # time step (sec)
dynamics_fun = dn3d.dynamics_fun_zero_field # dynamics that is imposed on the object
eta_1 = 0. # random force added on the ball (/sec^2)
sigma = 0. # noise added to images
theta_r = np.deg2rad(30) # half of the receptive field width (rad)
K = 12 # K*K is the total number of elements
L = 100 # L is the dimension of each element
sample_dt = 0.03 # sampling resolution
delay_dt = 0.03 # delay in the motion detector
savepath = '../../data/loom/grid_conv2_M1_L100_par_exp/' # path to save the results
n_cores = 36 # number of cores used


if not os.path.exists(savepath+'trajectories'):
    os.makedirs(savepath+'trajectories')
if not os.path.exists(savepath+'intensities_samples_cg'):
    os.makedirs(savepath+'intensities_samples_cg')
if not os.path.exists(savepath+'UV_flow_samples'):
    os.makedirs(savepath+'UV_flow_samples')
smgnmu.generate_samples_grid2_par(M, D, D_max, theta_max, R, V, dt, dynamics_fun, eta_1, sigma, theta_r,\
                           K, L, sample_dt, delay_dt, savepath, n_cores)
            
            