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
import samples_generation_multi_units_revision as smgnmur
import flow_field as flfd
import helper_functions as hpfn

figure_path = '/Volumes/Baohua/research/loom_detection/results/revision/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)
    

M = 32
R = 1 # radius of the ball (m)
Rs = np.array([R])
dt = 0.01 # time step (sec)
dynamics_fun = dn3d.dynamics_fun_zero_field # dynamics that is imposed on the object
around_z_angles = np.random.random(M) * 0 # rotation angles around the z axis
eta_1 = 0. # random force added on the ball (m/sec^2)
sigma = 0. # noise added to images
theta_r = np.deg2rad(30) # half of the receptive field width (rad)
K = 12 # K*K is the total number of units
L = 20 # L is the dimension of each of the K*K cells/motion detection units
sample_dt = 0.01 # sampling resolution
delay_dt = 0.03 # delay in the motion detector
D_max = 5.1*R # maximum initial distance (m)

data_types = ['hit', 'miss', 'retreat', 'rotation']

if M == 1:
    lplc2_units = np.array([[0, 0]])
else:
    lplc2_units = opsg.get_lplc2_units_xy_angles(M)
    _, lplc2_units_coords = opsg.get_lplc2_units(M)
angle_mc = lplc2_units[1]
x0, y0, z0 = lplc2_units_coords[1]
print(f'Coordinates of the central unit is {x0, y0, z0}.')

for data_type in data_types:
    if data_type == 'hit':
        x, y, z = 1.1 * hpfn.get_normalized_vector([0.33224794, -0.71471, -0.621718])
        vx, vy, vz = 5. * hpfn.get_normalized_vector([0.016, -0.04, -0.035])
        traj, _ = smgnmur.generate_one_trajectory_hit(R, D_max, dt, dynamics_fun, eta_1, x, y, z, vx, vy, vz)
    elif data_type == 'miss':
        x = 5*x0
        y = 5*y0
        z = 5*z0
        vx = -x
        vy = -y
        vz = -z
        vx, vy, vz = 5. * hpfn.get_normalized_vector([vx, vy, vz])
        v_add = 1.2 * hpfn.get_normalized_vector(np.array([0, 1./vy, -1./vz]))
        vx = vx + v_add[0]
        vy = vy + v_add[1]
        vz = vz + v_add[2]
        traj, dist = smgnmur.generate_one_trajectory_miss(R, D_max, dt, dynamics_fun, eta_1, x, y, z, vx, vy, vz)
        min_D = np.argmin(dist)
        traj = traj[:min_D+6]
    elif data_type == 'retreat':
        x = 1.*x0
        y = 1.5*y0
        z = 1.*z0
        x, y, z = 1.1 * hpfn.get_normalized_vector([x, y, z])
        vx = x
        vy = y
        vz = z
        vx, vy, vz = 5 * hpfn.get_normalized_vector([vx, vy, vz])
        traj, _ = smgnmur.generate_one_trajectory_retreat(R, D_max, dt, dynamics_fun, eta_1, x, y, z, vx, vy, vz)
    elif data_type == 'rotation':
        P = 100
        steps = 100
        D_min_r = 5*R
        D_max_r = 15*R
        Rs = np.random.random(P) * R
        theta_s = np.arccos(2*np.random.random()-1)
        phi_s = 2.*np.pi*np.random.random()
        xa = 1
        ya = 1
        za = 1
        fixed_z = -200*R
        traj = smgnmur.generate_one_trajectory_rot(D_min_r, D_max_r, P, steps, dt, \
                                                  xa, ya, za, scal=200, random_z=False, fixed_z=fixed_z)
    print(f'Length of the trajectory is {len(traj)}.')

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

    intensities_sample_cg_list, UV_flow_sample_list, _ = \
        smgnmur.generate_one_sample_exp(M, Rs, [traj], around_z_angles, sigma, theta_r, theta_matrix, coord_matrix, \
                            space_filter, K, L, dt, sample_dt, delay_dt)
    intensities_sample_cg = intensities_sample_cg_list[0]
    UV_flow_sample = UV_flow_sample_list[0]
    steps = len(intensities_sample_cg)
    print(f'Length of the sample is {steps}.')

    save_path = figure_path + f'movies/model_response/movie_intensities_multi_units/M_{M}/' 
    if not os.path.exists(save_path+data_type):
        os.makedirs(save_path+'other_info/'+data_type+'/trajectories/')
        os.makedirs(save_path + data_type + '/intensities_samples_cg/')
        os.makedirs(save_path + data_type + '/UV_flow_samples/')

    np.save(save_path + 'other_info/' + data_type + '/trajectories/traj_list_0', [traj])
    np.save(save_path + data_type + '/intensities_samples_cg/intensities_sample_cg_list_0', [intensities_sample_cg])
    np.save(save_path + data_type + '/UV_flow_samples/UV_flow_sample_list_0', [UV_flow_sample])
