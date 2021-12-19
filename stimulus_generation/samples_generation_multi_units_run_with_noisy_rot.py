#!/usr/bin/env python


'''
Generates samples for training.
'''


import sys
sys.path.append('../stimulus_core/')
sys.path.append('../models_core/')
sys.path.append('../helper/')

import os
import numpy as np
import time
import multiprocessing
import importlib
import dynamics_3d as dn3d
import optical_signal as opsg
import flow_field as flfd
import samples_generation_multi_units_revision as smgnmur
import helper_functions as hpfn


for M in [1, 2, 4, 8, 16, 32, 64, 128, 192, 256]:

    R = 1 # radius of the ball
    dt = 0.01 # time step (sec)
    dynamics_fun = dn3d.dynamics_fun_zero_field # dynamics that is imposed on the object
    around_z_angles = np.random.random(M) * 0. # rotation angles around the z axis
    eta_1 = 0. # random force added on the ball (/sec^2)
    sigma = 0. # noise added to images
    theta_r = np.deg2rad(30) # half of the receptive field width (rad)
    K = 12 # K*K is the total number of elements
    L = 4 # L is the dimension of each element
    sample_dt = 0.03 # sampling resolution (sec)
    delay_dt = 0.03 # delay in the motion detector (sec)
    D_min = 5. * R # minimum initial distance 
    D_max = 5.01 * R # maximum initial distance 
    v_min = 2. * R # minimum speed (/sec)
    v_max = 10. * R # maximum speed (/sec)
    P = 100 # number of balls in rotation scenes
    steps_r = 50 # number of steps in rotation scenes
    D_min_r = 5 * R # minimum distance of the balls in the rotation scenes
    D_max_r = 15 * R # maximum distance of the balls in the rotation scenes
    scal = 100. # rotational speed (deg/sec)
    hl = 0.2 * 1000 # half life (sec)

    # number of samples in one list
    NNs = 10
    
    # number of training sample lists, and real sample size shold times NNs
    N1 = 100 # hit
    N2 = 50 # miss
    N3 = 50 # retreat
    
    # number of testing sample lists, and real sample size shold times NNs
    M1 = 30 # hit
    M2 = 15 # miss
    M3 = 15 # retreat
    
    if M in [1, 2, 4]:
        N1 = np.int(8 / M * N1)
        N2 = np.int(8 / M * N2)
        N3 = np.int(8 / M * N3)
        M1 = np.int(8 / M * M1)
        M2 = np.int(8 / M * M2)
        M3 = np.int(8 / M * M3)
    
    # set number
    set_number = 1000 + M
    # path to save the sample data
    savepath = f'../../data/loom/multi_lplc2_D{np.int(D_min)}_L{np.int(L)}_exp_with_noisy_rot_scal{int(scal)}/'
    # savepath = f'/Volumes/Baohua/data_on_hd/loom/multi_lplc2_D{np.int(D_min)}_L{np.int(L)}_exp_with_noisy_rot_test/'
    # make the folder to save the samples
    hpfn.make_set_folder(set_number, savepath)

    # number of cores used in parallel
    num_cores = 36

    # generate the samples
    start_time = time.time()
    smgnmur.generate_samples_par_exp_with_noisy_rot(M, R, dt, dynamics_fun, around_z_angles, eta_1, sigma, theta_r, K, L, sample_dt, delay_dt, D_min, D_max, v_min, v_max,\
                                                    P, steps_r, D_min_r, D_max_r, scal, hl, N1, N2, N3, M1, M2, M3, NNs,\
                                                    set_number, savepath, num_cores)
    print(time.time() - start_time)








