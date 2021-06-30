#!/usr/bin/env python


'''
Generates samples for linear law, angular size dependency, angular velocity dependency.
'''

import sys
sys.path.append('../stimulus_core/')
sys.path.append('../models_core/')
sys.path.append('../helper/')

import numpy as np
import os
from joblib import Parallel, delayed

import dynamics_3d as dn3d
import flow_field as flfd
import helper_functions as hpfn
import optical_signal as opsg
import samples_generation_multi_units as smgnmu

R_over_v_list = [0.01, 0.02, 0.04, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
# R_over_v_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, \
#                  0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]

def get_samples_linear_law(R_over_v):

    R = 1 # radius of the ball (m)
    Rs = np.array([R])
    dt = 0.001 # time step (sec)
    dynamics_fun = dn3d.dynamics_fun_zero_field # dynamics that is imposed on the object
    eta_1 = 0. # random force added on the ball (m/sec^2)
    sigma = 0. # noise added to images
    theta_r = np.deg2rad(30) # half of the receptive field width (rad)
    K = 12 # K*K is the total number of elements
    L = 20 # L is the dimension of each element
    N = K*L + 4*L # Size of each frame, the additional 4*L is for spatial filtering and will be deleted afterwards.
    pad = 2*L
    leftup_corners = opsg.get_leftup_corners(K, L, pad)
    coord_y = np.arange(N) - (N-1)/2.
    coord_x = np.arange(N) - (N-1)/2.
    coords_y, coords_x = np.meshgrid(coord_y, -coord_x) # coordinates of each point in the frame
    dm = np.sqrt(coords_y**2 + coords_x**2) # distance matrix
    theta_r = np.deg2rad(30) # half of the receptive field width (rad)
    space_filter = flfd.get_space_filter(L/2, 4)
    theta_matrix, phi_matrix = opsg.get_angle_matrix(theta_r, coords_x, coords_y, dm, K, L)
    coord_matrix = opsg.get_coord_matrix(phi_matrix, theta_matrix, 1.)
    sample_dt = 0.001 # sampling resolution
    delay_dt = 0.03 # delay in the motion detector
    D_max = 10.1*R # maximum initial distance (m)
    data_type = 'hit'
    # save_path = '../../data/loom/linear_law/'+data_type+'/'
    save_path = '/Volumes/Baohua/data_on_hd/loom/linear_law/'+data_type+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for M in [1, 256]:
        if M == 1:
            lplc2_units_coords = np.array([[0, 0, 1]])
        else:
            _, lplc2_units_coords = opsg.get_lplc2_units(M)
        
        x = lplc2_units_coords[0][0]
        y = lplc2_units_coords[0][1]
        z = lplc2_units_coords[0][2]
        vx = x * R / R_over_v
        vy = y * R / R_over_v
        vz = z * R / R_over_v
        traj, distance = smgnmu.generate_one_trajectory_hit(M, R, D_max, theta_r, dt, dynamics_fun, eta_1, x, y, z, vx, vy, vz)

        _, intensities_sample_cg, UV_flow_sample, _ = \
            hpfn.generate_one_sample_exp(M, Rs, traj, sigma, theta_r, theta_matrix, coord_matrix, space_filter, K, L, dt, sample_dt, delay_dt)
        
        steps = len(intensities_sample_cg)
        print(f'The length of the movie is {steps}.')

        np.save(save_path+f'traj_{R_over_v}_M{M}', traj)
        np.save(save_path+f'distance_{R_over_v}_M{M}', distance)
        np.save(save_path+f'intensities_sample_cg_{R_over_v}_M{M}', intensities_sample_cg)
        np.save(save_path+f'UV_flow_sample_{R_over_v}_M{M}', UV_flow_sample)

n_cores = 2
Parallel(n_jobs=n_cores)(delayed(get_samples_linear_law)(R_over_v) for R_over_v in R_over_v_list)