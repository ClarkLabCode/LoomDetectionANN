#!/usr/bin/env python


"""
This module generates the trajectories, intensities, and flow fields for each type of stimuli: 
hit, miss, retreat, rotation.
"""
import sys
sys.path.append('../stimulus_core/')
sys.path.append('../models_core/')
sys.path.append('../helper/')

import os
import numpy as np
import random
import time
import importlib
from matplotlib.colors import LinearSegmentedColormap

import dynamics_3d as dn3d
import optical_signal as opsg
import samples_generation_multi_units as smgnmu
import flow_field as flfd
import helper_functions as hpfn


figure_path = '/Volumes/Baohua/research/loom_detection/results/final_figures_for_paper/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

has_inhibition = True
if not os.path.exists(figure_path+'movies/model_response/movie_frames_multi_units/'):
    os.makedirs(figure_path+'movies/model_response/movie_frames_multi_units/hit/')
    os.makedirs(figure_path+'movies/model_response/movie_frames_multi_units/miss/')
    os.makedirs(figure_path+'movies/model_response/movie_frames_multi_units/retreat/')
    os.makedirs(figure_path+'movies/model_response/movie_frames_multi_units/rotation/')


M = 32
R = 1 # radius of the ball (m)
Rs = np.array([R])
dt = 0.01 # time step (sec)
dynamics_fun = dn3d.dynamics_fun_zero_field # dynamics that is imposed on the object
eta_1 = 0. # random force added on the ball (m/sec^2)
sigma = 0. # noise added to images
theta_r = np.deg2rad(30) # half of the receptive field width (rad)
K = 12 # K*K is the total number of units
L = 50 # L is the dimension of each 
sample_dt = 0.01 # sampling resolution
delay_dt = 0.03 # delay in the motion detector
D_max = 5.1*R # maximum initial distance (m)

data_types = ['hit', 'miss', 'retreat', 'rotation']
data_type = data_types[0]

if M == 1:
    lplc2_units = np.array([[0, 0]])
else:
    lplc2_units = opsg.get_lplc2_units_xy_angles(M)
    _, lplc2_units_coords = opsg.get_lplc2_units(M)
angle_mc = lplc2_units[1]
x0, y0, z0 = lplc2_units_coords[1]
print(f'Coordinates of the central unit is {x0, y0, z0}.')

for data_type in ['miss', 'rotation']:
    if data_type == 'hit':
        x = 1.1*x0
        y = 1.1*y0
        z = 1.1*z0
        vx = 5*(x0)
        vy = 5*(y0)
        vz = 5*(z0)
        traj, _ = smgnmu.generate_one_trajectory_hit(M, R, D_max, theta_r, dt, dynamics_fun, eta_1, x, y, z, vx, vy, vz)
    elif data_type == 'miss':
        x = 10*x0
        y = 10*y0
        z = 10*z0
        vx = (-x0)*0
        vy = (-y0)*0
        vz = (-z0)*1
        v = np.sqrt(vx**2+vy**2+vz**2)
        vx = 5*vx/v
        vy = 5*vy/v
        vz = 5*vz/v
        print(x, y, z)
        print(vx, vy, vz)
        traj, dist = smgnmu.generate_one_trajectory_miss(M, R, D_max, theta_r, dt, dynamics_fun, eta_1, x, y, z, vx, vy, vz)
        min_D = np.argmin(dist)
        traj = traj[:min_D+6]
    elif data_type == 'retreat':
        x = 1.1*x0
        y = 1.1*y0
        z = 1.1*z0
        vx = 5*(x0)
        vy = 5*(y0)
        vz = 5*(z0)
        traj, _ = smgnmu.generate_one_trajectory_retreat(M, R, D_max, theta_r, dt, dynamics_fun, eta_1, x, y, z, vx, vy, vz)
    elif data_type == 'rotation':
        P = 100
        steps = 50
        D_min = 5*R
        D_max = 50*R
        Rs = np.ones(P)*R
        theta_s = np.arccos(2*np.random.random()-1)
        phi_s = 2.*np.pi*np.random.random()
        xa = np.sin(theta_s)*np.cos(phi_s)
        ya = np.sin(theta_s)*np.sin(phi_s)
        za = np.cos(theta_s)
        fixed_z = 200*R
        traj = smgnmu.generate_one_trajectory_rot(M, D_min, D_max, P, steps, dt, xa, ya, za, random_z=False, fixed_z=fixed_z)
    print(f'Length of the trajectory is {len(traj)}.')

    N = K*L + 4*L # Size of each frame, the additional 4*L is for spatial filtering and will be deleted afterwards.
    pad = 2*L
    leftup_corners = opsg.get_leftup_corners(K, L, pad)
    coord_y = np.arange(N) - (N-1)/2.
    coord_x = np.arange(N) - (N-1)/2.
    coords_y, coords_x = np.meshgrid(coord_y, -coord_x) # coordinates of each point in the frame
    dm = np.sqrt(coords_y**2 + coords_x**2) # distance matrix
    theta_r = np.deg2rad(30) # half of the receptive field width (rad)
    myheat = LinearSegmentedColormap.from_list('br', ["b", "w", "r"], N=256)
    space_filter = flfd.get_space_filter(L/2, 4)
    theta_matrix, phi_matrix = opsg.get_angle_matrix(theta_r, coords_x, coords_y, dm, K, L)
    coord_matrix = opsg.get_coord_matrix(phi_matrix, theta_matrix, 1.)

    intensities_sample, _, UV_flow_sample, _ = \
        hpfn.generate_one_sample(M, Rs, traj, sigma, theta_r, space_filter, K, L, dt, sample_dt, delay_dt)
    steps = len(intensities_sample)
    print(steps)

    save_path = figure_path+'movies/model_response/movie_intensities_multi_units/'+data_type+f'/M_{M}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save(save_path+'traj_list', [traj])
    np.save(save_path+'intensities_sample_list', [intensities_sample])
    np.save(save_path+'UV_flow_sample_list', [UV_flow_sample])