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
import samples_generation_multi_units as smgnmu
import helper_functions as hpfn


for M in [32]:

    R = 1 # radius of the ball
    dt = 0.01 # time step (sec)
    dynamics_fun = dn3d.dynamics_fun_zero_field # dynamics that is imposed on the object
    around_z_angles = np.random.random(M) * 0 * np.pi # rotation angles around the z axis
    # around_z_angles = [3.48656203, 5.4032646,  5.56937198, 2.94184824, 1.99006079, 3.94869006, 3.57920114, 2.12201486, 
                       # 5.22594498, 0.03105057, 1.67954443, 4.84723275, 5.84375018, 4.1627342,  2.89210876, 4.00368211, 
                       # 5.38178864, 4.4826508, 4.25779883, 4.79747045, 4.49131731, 2.1991297,  3.7946034,  2.15303421, 
                       # 5.74870705, 3.94106405, 1.06121123, 5.88680446, 2.24776055, 4.22813591, 5.94538818, 1.26232594]
    eta_1 = 0. # random force added on the ball (/sec^2)
    sigma = 0. # noise added to images
    theta_r = np.deg2rad(30) # half of the receptive field width (rad)
    K = 12 # K*K is the total number of elements
    L = 4 # L is the dimension of each element
    sample_dt = 0.03 # sampling resolution
    delay_dt = 0.03 # delay in the motion detector
    D_min = 5. * R # minimum initial distance 
    D_max = 5.01 * R # maximum initial distance 
    v_min = 2. * R # minimum speed (/sec)
    v_max = 10. * R # maximum speed (/sec)
    P = 100 # number of balls in rotation scenes
    steps_r = 50 # number of steps in rotation scenes
    D_min_r = 5 * R # minimum distance of the balls in the rotation scenes
    D_max_r = 15 * R # maximum distance of the balls in the rotation scenes
    scal = 200. # rotational speed (deg/sec)
    hl = 0.2 * 1000 # half life (sec)

    # number of samples in one list
    NNs = 10
    
    # number of training sample lists, and real sample size shold times NNs
    N1 = 100 # hit
    N2 = 50 # miss
    N3 = 50 # retreat
    N4 = 200 # rotation
    
    # number of testing sample lists, and real sample size shold times NNs
    M1 = 30 # hit
    M2 = 15 # miss
    M3 = 15 # retreat
    M4 = 60 # rotation
    
    if M in [1, 2, 4]:
        N1 = np.int(8 / M * N1)
        N2 = np.int(8 / M * N2)
        N3 = np.int(8 / M * N3)
        N4 = np.int(8 / M * N4)
        M1 = np.int(8 / M * M1)
        M2 = np.int(8 / M * M2)
        M3 = np.int(8 / M * M3)
        M4 = np.int(8 / M * M4)

    # N1 = np.int(8 * N1)
    # N2 = np.int(8 * N2)
    # N3 = np.int(8 * N3)
    # N4 = np.int(8 * N4)
    # M1 = np.int(8 * M1)
    # M2 = np.int(8 * M2)
    # M3 = np.int(8 * M3)
    # M4 = np.int(8 * M4)
    
    # set number
    set_number = 1000 + M
    # path to save the sample data
    savepath = f'../../data/loom/multi_lplc2_D{np.int(D_min)}_L{np.int(L)}_exp_with_half_constant_rot_scal{int(scal)}/'
    # make the folder to save the samples
    hpfn.make_set_folder(set_number, savepath)

    # number of cores used in parallel
    num_cores = 36

    # generate the samples
    start_time = time.time()
    smgnmu.generate_samples_par_exp(M, R, dt, dynamics_fun, around_z_angles, eta_1, sigma, theta_r, K, L, sample_dt, delay_dt, \
                                D_min, D_max, v_min, v_max, P, steps_r, D_min_r, D_max_r, scal, \
                                N1, N2, N3, N4, M1, M2, M3, M4, NNs, set_number, savepath, num_cores)

    # generate the samples with background noise
    # start_time = time.time()
    # smgnmu.generate_samples_par_exp_with_noisy_rot(M, R, dt, dynamics_fun, around_z_angles, eta_1, sigma, theta_r, K, L, sample_dt, delay_dt, D_min, D_max, v_min, v_max,\
    #                                                 P, steps_r, D_min_r, D_max_r, scal, hl, N1, N2, N3, M1, M2, M3, NNs,\
    #                                                 set_number, savepath, num_cores)

    print(time.time() - start_time)








