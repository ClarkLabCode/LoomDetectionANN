#!/usr/bin/env python


"""
This script generates stimuli and flow field to analyze the tuning curves of the HRC models.
"""


import sys
sys.path.append('../stimulus_core/')
sys.path.append('../models_core/')
sys.path.append('../helper/')

import numpy as np
import os
import time
from time import sleep
from joblib import Parallel, delayed

import dynamics_3d as dn3d
import optical_signal as opsg
import flow_field as flfd
import helper_functions as hpfn
import get_Klapoetke_stimuli as gKs

# Flow field estimation
# Here, we look at the flow field estimation using three types of stimuli: an expanding disk, a moving edge and a moving bar, all of which have constant edge velocities for any individual trajectories or movies.


# Useful functions

## Expanding disk
def get_expanding_disk(K, L, pad, dt, v_deg):
    N = K * L
    ro = (N - 1) / 2.
    co = (N - 1) / 2.
    R = 2 * L
    v = (v_deg / 5) * L

    expanding_disk = []
    while R <= (N + 1) / 2:
        one_disk = gKs.get_one_disk(K, L, pad, ro, co, R)
        expanding_disk.append(one_disk)
        R = R + v * dt
    expanding_disk = np.array(expanding_disk)
    
    return expanding_disk


## Moving bar
def get_moving_bar(K, L, pad, dt, v_deg):
    N = K * L
    ro = (N - 1) / 2.
    co = -L
    theta_a = np.pi * 0
    L1 = L * 1
    L2 = (N - 1) / 2.
    L3 = L * 1
    L4 = (N - 1) / 2.
    v = (v_deg / 5) * L 
    
    moving_bar = []
    while co <= N + L:
        one_bar = gKs.get_one_bar(K, L, pad, ro, co, theta_a, L1, L2, L3, L4)
        moving_bar.append(one_bar)
        co = co + v * dt
    moving_bar = np.array(moving_bar)
    
    return moving_bar


## Moving edge
def get_moving_edge(K, L, pad, dt, v_deg):
    N = K * L
    ro = (N - 1) / 2.
    co = -L
    theta_a = np.pi * 0
    L1 = 0
    L2 = (N - 1) / 2.
    L3 = 0
    L4 = (N - 1) / 2.
    v = (v_deg / 5) * L
    
    moving_edge = []
    while L1 <= N - co:
        one_bar = gKs.get_one_bar(K, L, pad, ro, co, theta_a, L1, L2, L3, L4)
        moving_edge.append(one_bar)
        L1 = L1 + v * dt
    moving_edge = np.array(moving_edge)
    
    return moving_edge


########## Start simulations ###########
K = 12
L = 50
dt = 0.001
p = 1
pad = 2 * L
delay_dt = 0.03
n_cores = 20
v_deg_list = range(5, 2001, 5)
space_filter = flfd.get_space_filter(L/2, 4)
leftup_corners = opsg.get_leftup_corners(K, L, pad)


############ Delta filter #############

## Expanding disk
save_path = '../../data/loom/hrc_tuning/expanding_disk_delta/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def get_expanding_disk_par(v_deg, space_filter, K, L, pad, dt, delay_dt, leftup_corners, save_path):
    # intensity
    expanding_disk = get_expanding_disk(K, L, pad, dt, v_deg)
    # calculate the flow field
    UV_flows = gKs.get_UV_flows(space_filter, K, L, pad, dt, delay_dt, leftup_corners, expanding_disk)
    np.save(save_path+f'UV_flow_L{L}_v_{v_deg}_delta', UV_flows)
    
start_time = time.time()

Parallel(n_jobs=n_cores)\
    (delayed(get_expanding_disk_par)\
        (v_deg, space_filter, K, L, pad, dt, delay_dt, leftup_corners, save_path) for v_deg in v_deg_list)
    
print(f'This takes {time.time()-start_time}.')


## Moving bar
save_path = '../../data/loom/hrc_tuning/moving_bar_delta/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def get_moving_bar_par(v_deg, space_filter, K, L, pad, dt, delay_dt, leftup_corners, save_path):
    # intensity
    moving_bar = get_moving_bar(K, L, pad, dt, v_deg)
    # calculate the flow field
    UV_flows = gKs.get_UV_flows(space_filter, K, L, pad, dt, delay_dt, leftup_corners, moving_bar)
    np.save(save_path+f'UV_flow_L{L}_v_{v_deg}_delta', UV_flows)
    
start_time = time.time()

Parallel(n_jobs=n_cores)\
    (delayed(get_moving_bar_par)\
        (v_deg, space_filter, K, L, pad, dt, delay_dt, leftup_corners, save_path) for v_deg in v_deg_list)
    
print(f'This takes {time.time()-start_time}.')


## Moving_edge
save_path = '../../data/loom/hrc_tuning/moving_edge_delta/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def get_moving_edge_par(v_deg, space_filter, K, L, pad, dt, delay_dt, leftup_corners, save_path):
    # intensity
    moving_edge = get_moving_edge(K, L, pad, dt, v_deg)
    # calculate the flow field
    UV_flows = gKs.get_UV_flows(space_filter, K, L, pad, dt, delay_dt, leftup_corners, moving_edge)
    np.save(save_path+f'UV_flow_L{L}_v_{v_deg}_delta', UV_flows)
    
start_time = time.time()

Parallel(n_jobs=n_cores)\
    (delayed(get_moving_edge_par)\
        (v_deg, space_filter, K, L, pad, dt, delay_dt, leftup_corners, save_path) for v_deg in v_deg_list)
    
print(f'This takes {time.time()-start_time}.')


############ Exponential filter #############

## Expanding disk
save_path = '../../data/loom/hrc_tuning/expanding_disk_exp/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def get_expanding_disk_par(v_deg, space_filter, K, L, pad, dt, delay_dt, leftup_corners, save_path):
    # intensity
    expanding_disk = get_expanding_disk(K, L, pad, dt, v_deg)
    # calculate the flow field
    UV_flows = gKs.get_UV_flows_exp(space_filter, K, L, pad, dt, delay_dt, leftup_corners, expanding_disk)
    np.save(save_path+f'UV_flow_L{L}_v_{v_deg}_exp', UV_flows)
    
start_time = time.time()

Parallel(n_jobs=n_cores)\
    (delayed(get_expanding_disk_par)\
        (v_deg, space_filter, K, L, pad, dt, delay_dt, leftup_corners, save_path) for v_deg in v_deg_list)
    
print(f'This takes {time.time()-start_time}.')


## Moving bar
save_path = '../../data/loom/hrc_tuning/moving_bar_exp/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def get_moving_bar_par(v_deg, space_filter, K, L, pad, dt, delay_dt, leftup_corners, save_path):
    # intensity
    moving_bar = get_moving_bar(K, L, pad, dt, v_deg)
    # calculate the flow field
    UV_flows = gKs.get_UV_flows_exp(space_filter, K, L, pad, dt, delay_dt, leftup_corners, moving_bar)
    np.save(save_path+f'UV_flow_L{L}_v_{v_deg}_exp', UV_flows)
    
start_time = time.time()

Parallel(n_jobs=n_cores)\
    (delayed(get_moving_bar_par)\
        (v_deg, space_filter, K, L, pad, dt, delay_dt, leftup_corners, save_path) for v_deg in v_deg_list)
    
print(f'This takes {time.time()-start_time}.')


## Moving_edge
save_path = '../../data/loom/hrc_tuning/moving_edge_exp/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def get_moving_edge_par(v_deg, space_filter, K, L, pad, dt, delay_dt, leftup_corners, save_path):
    # intensity
    moving_edge = get_moving_edge(K, L, pad, dt, v_deg)
    # calculate the flow field
    UV_flows = gKs.get_UV_flows_exp(space_filter, K, L, pad, dt, delay_dt, leftup_corners, moving_edge)
    np.save(save_path+f'UV_flow_L{L}_v_{v_deg}_exp', UV_flows)
    
start_time = time.time()

Parallel(n_jobs=n_cores)\
    (delayed(get_moving_edge_par)\
        (v_deg, space_filter, K, L, pad, dt, delay_dt, leftup_corners, save_path) for v_deg in v_deg_list)
    
print(f'This takes {time.time()-start_time}.')
    


