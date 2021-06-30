#!/usr/bin/env python


"""
This module generates data samples for the movies.
"""


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
import helper_functions as hpfn

figure_path = '/Volumes/Baohua/research/loom_detection/results/final_figures_for_paper_exp/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)
    

# Stimuli movies

# Here, we generate trajectories, intensity, flow field samples for different types of stimuli:
# hit, miss, retreat, rotation for single-unit model (M = 1)

M = 1
R = 1 # radius of the ball (m)
Rs = np.array([R])
dt = 0.01 # time step (sec)
dynamics_fun = dn3d.dynamics_fun_zero_field # dynamics that is imposed on the object
eta_1 = 0. # random force added on the ball (m/sec^2)
sigma = 0. # noise added to images
theta_r = np.deg2rad(30) # half of the receptive field width (rad)
K = 12 # K*K is the total number of elements
L = 50 # L is the dimension of each element
sample_dt = 0.01 # sampling resolution
delay_dt = 0.03 # delay in the motion detector
D_max = 10.1*R # maximum initial distance (m)

data_types = ['hit', 'miss', 'retreat', 'rotation']
for data_type in data_types:

    if data_type == 'hit':
        x = 0
        y = 0
        z = 1.1
        vx = 0
        vy = 0
        vz = 5
        traj, _ = smgnmu.generate_one_trajectory_hit(M, R, D_max, theta_r, dt, dynamics_fun, eta_1, x, y, z, vx, vy, vz)
    elif data_type == 'miss':
        x = 0
        y = 0
        z = 10
        vx = -0.1
        vy = 0.5
        vz = -5
        traj, dist = smgnmu.generate_one_trajectory_miss(M, R, D_max, theta_r, dt, dynamics_fun, eta_1, x, y, z, vx, vy, vz)
        min_D = np.argmin(dist)
        traj = traj[:min_D+6]
    elif data_type == 'retreat':
        x = 0
        y = 0
        z = 1.1
        vx = 0
        vy = 0
        vz = 5
        traj, _ = smgnmu.generate_one_trajectory_retreat(M, R, D_max, theta_r, dt, dynamics_fun, eta_1, x, y, z, vx, vy, vz)
    elif data_type == 'rotation':
        P = 100
        steps = 200
        D_min = 5*R
        D_max = 15*R
        Rs = np.random.random(P)*R
        theta_s = np.arccos(2*np.random.random()-1)
        phi_s = 2.*np.pi*np.random.random()
    #     xa = np.sin(theta_s)*np.cos(phi_s)
    #     ya = np.sin(theta_s)*np.sin(phi_s)
    #     za = np.cos(theta_s)
        xa = 1
        ya = 1
        za = 0
        fixed_z = -200*R
        traj = smgnmu.generate_one_trajectory_rot(M, D_min, D_max, P, steps, dt, \
                                                  xa, ya, za, random_z=False, fixed_z=fixed_z)
    print(f'The length of the trajectory is {len(traj)}.')

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

    frames_sample, _, UV_flow_sample, _ = \
        hpfn.generate_one_sample_exp(1, Rs, traj, sigma, theta_r, theta_matrix, coord_matrix, space_filter, K, L, dt, sample_dt, delay_dt)
    steps = len(frames_sample)
    print(f'The length of the movie is {steps}.')

    save_path = figure_path+'movies/stimuli/movie_frames/'+data_type+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save(save_path+'traj', traj)
    np.save(save_path+'frames_sample', frames_sample)
    np.save(save_path+'UV_flow_sample', UV_flow_sample)

    if data_type == 'rotation':
        np.save(save_path+'Rs', Rs)
    