#!/usr/bin/env python


"""
This module contains functions to generate samples for training and testing.
"""


import numpy as np
from scipy.stats import truncnorm
from scipy.spatial.transform import Rotation as R3d
from joblib import Parallel, delayed
import glob

import dynamics_3d as dn3d
import optical_signal as opsg
import flow_field as flfd


# # Generate certain amount of samples for each type
# def generate_samples_par(M, R, dt, dynamics_fun, eta_1, sigma, theta_r, K, L, sample_dt, delay_dt, D_min, D_max, v_min, v_max,\
#                          P, steps_r, D_min_r, D_max_r, scal, N1, N2, N3, N4, M1, M2, M3, M4, NNs,\
#                          set_number, savepath, num_cores):
#     """
#     Args:
#     M: # of lplc2 units
#     R: radius of the ball 
#     dt: time step (sec)
#     dynamics_fun: predefined dynamics of the object, return the accelerations (/sec^2)
#     eta_1: random force added on the ball (/sec^2)
#     sigma: noise added to images
#     theta_r: half of the receptive field width (rad)
#     K: K*K is the total number of elements.
#     L: element dimension.
#     sample_dt: timescale of sampling
#     delay_dt: timescale of delay in the motion detector.
#     D_min: minimum initial distance 
#     D_max: maximum initial distance 
#     v_min: minimum velocity (sec^-1)
#     v_max: maximum velocity (sec^-1)
#     P: number of balls in rotation scenes
#     steps_r: number of steps in rotation scenes
#     D_min_r: minimum distance of the balls in the rotation scenes
#     D_max_r: maximum distance of the balls in the rotation scenes
#     scal: scale of the rotaion, in degrees
#     N1: # of training samples of hit signals
#     N2: # of training samples of miss signals
#     N3: # of training samples of retreat signals
#     N4: # of training samples of rotation signals
#     M1: # of testing samples of hit signals
#     M2: # of testing samples of miss signals
#     M3: # of testing samples of retreat signals
#     M4: # of testing samples of rotation signals
#     NNs: # of sample trajectories in each list
#     set_number: int, set number
#     savepath: path to save the sample data
#     num_cores: # of cores used in parallel
#     """
    
#     f0 = open(savepath+'set_{}/general_info.txt'.format(set_number), 'a')
#     f0.write('M = {}: # of lplc2 units\n'.format(M))
#     f0.write('R = {}: radius of the ball \n'.format(R))
#     f0.write('dt = {}: time step (sec)\n'.format(dt))
#     f0.write('dynamics_fun = '+dynamics_fun.__name__+': predefined dynamics of the object, return the accelerations (m/sec^2)\n')
#     f0.write('eta_1 = {}: random force added on the ball (m/sec^2)\n'.format(eta_1))
#     f0.write('sigma = {}: noise added to images\n'.format(sigma))
#     f0.write('theta_r = {:.4g}: half of the receptive field width (rad)\n'.format(theta_r))
#     f0.write('K = {}: K*K is the total number of elements\n'.format(K))
#     f0.write('L = {}: element dimension\n'.format(L))
#     f0.write('sample_dt = {}: timescale of sampling\n'.format(sample_dt))
#     f0.write('delay_dt = {}: timescale of delay in the motion detector\n'.format(delay_dt))
#     f0.write('D_min = {}: minimum initial distance \n'.format(D_min))
#     f0.write('D_max = {}: maximum initial distance \n'.format(D_max))
#     f0.write('v_min = {}: minimum velocity (sec^-1) \n'.format(v_min))
#     f0.write('v_max = {}: maximum velocity (sec^-1) \n'.format(v_max))
#     f0.write('P = {}: number of balls in rotation scenes \n'.format(P))
#     f0.write('steps_r = {}: number of steps in rotation scenes \n'.format(steps_r))
#     f0.write('D_min_r = {}: minimum distance of the balls in the rotation scenes \n'.format(D_min_r))
#     f0.write('D_max_r = {}: maximum distance of the balls in the rotation scenes \n'.format(D_max_r))
#     f0.write('scal = {}: scale of the rotaion (deg*sec^-1) \n'.format(scal))
#     f0.write('N1 = {}: # of lists of training samples of hit signals\n'.format(N1))
#     f0.write('N2 = {}: # of lists of training samples of miss signals\n'.format(N2))
#     f0.write('N3 = {}: # of lists of training samples of retreat signals\n'.format(N3))
#     f0.write('N4 = {}: # of lists of training samples of rotation signals\n'.format(N4))
#     f0.write('M1 = {}: # of lists of testing samples of hit signals\n'.format(M1))
#     f0.write('M2 = {}: # of lists of testing samples of miss signals\n'.format(M2))
#     f0.write('M3 = {}: # of lists of testing samples of retreat signals\n'.format(M3))
#     f0.write('M4 = {}: # of lists of testing samples of rotation signals\n'.format(M4))
#     f0.write('NNs = {}: # of samples in each list\n'.format(NNs))
#     f0.close()   
    
#     N = K * L + 4 * L # Size of each frame, the additional 4*L is for spatial filtering and will be deleted afterwards.
#     coord_y = np.arange(N) - (N - 1) / 2.
#     coord_x = np.arange(N) - (N - 1) / 2.
#     coords_y, coords_x = np.meshgrid(coord_y, -coord_x) # coordinates of each point in the frame
#     dm = np.sqrt(coords_y**2 + coords_x**2) # distance matrix
#     theta_matrix, phi_matrix = opsg.get_angle_matrix(theta_r, coords_x, coords_y, dm, K, L)
#     coord_matrix = opsg.get_coord_matrix(phi_matrix, theta_matrix, D=1.)
#     space_filter = flfd.get_space_filter(L/2, 4)
    
#     # hit samples
#     print('Generating hit samples:')
#     datatype = 'hit'
#     Rs = np.array([R])
#     generate_trajectory_hit(\
#         M, R, D_max, v_min, v_max, theta_r, dt, delay_dt, dynamics_fun, eta_1, N1+M1, NNs, savepath, set_number, datatype)
#     Parallel(n_jobs=num_cores)\
#             (delayed(generate_sample_par)\
#              (n1, N1, M1, M, Rs, sigma, theta_r, theta_matrix, coord_matrix, space_filter, K, L,\
#               dt, sample_dt, delay_dt, savepath, set_number, datatype)\
#              for n1 in range(1, N1+M1+1))
#     print('{} trainging samples have been generated!'.format(N1))
#     print('{} testing samples have been generated!'.format(M1))
    
#     # miss samples
#     print('Generating miss samples:')
#     datatype = 'miss'
#     Rs = np.array([R])
#     generate_trajectory_miss(\
#         M, R, D_min, D_max, v_min, v_max, theta_r, dt, delay_dt, dynamics_fun, eta_1, N2+M2, NNs, savepath, set_number, datatype)
#     Parallel(n_jobs=num_cores)\
#             (delayed(generate_sample_par)\
#              (n2, N2, M2, M, Rs, sigma, theta_r, theta_matrix, coord_matrix, space_filter, K, L,\
#               dt, sample_dt, delay_dt, savepath, set_number, datatype)\
#              for n2 in range(1, N2+M2+1))
#     print('{} trainging samples have been generated!'.format(N2))
#     print('{} testing samples have been generated!'.format(M2))
    
#     # retreat samples
#     print('Generating retreat samples:')
#     datatype = 'retreat'
#     Rs = np.array([R])
#     generate_trajectory_retreat(\
#         M, R, D_max, v_min, v_max, theta_r, dt, delay_dt, dynamics_fun, eta_1, N3+M3, NNs, savepath, set_number, datatype)
#     Parallel(n_jobs=num_cores)\
#             (delayed(generate_sample_par)\
#              (n3, N3, M3, M, Rs, sigma, theta_r, theta_matrix, coord_matrix, space_filter, K, L,\
#               dt, sample_dt, delay_dt, savepath, set_number, datatype)\
#              for n3 in range(1, N3+M3+1))
#     print('{} trainging samples have been generated!'.format(N3))
#     print('{} testing samples have been generated!'.format(M3))
    
#     # rotation scene samples
#     print('Generating rotation scene samples:')
#     datatype = 'rotation'
#     Rs = np.random.random(P) * R
#     generate_trajectory_rot(M, D_min_r, D_max_r, P, steps_r, dt, scal, N4+M4, NNs, savepath, set_number, datatype)
#     Parallel(n_jobs=num_cores)\
#             (delayed(generate_sample_par)\
#              (n4, N4, M4, M, Rs, sigma, theta_r, theta_matrix, coord_matrix, space_filter, K, L,\
#               dt, sample_dt, delay_dt, savepath, set_number, datatype)\
#              for n4 in range(1, N4+M4+1)) 
#     print('{} training samples have been generated!'.format(N4))
#     print('{} testing samples have been generated!'.format(M4))
    
    
# # Generate one sample, in parallel
# def generate_sample_par(n, N0, M0, M, Rs, sigma, theta_r, theta_matrix, coord_matrix, space_filter, K, L,\
#                         dt, sample_dt, delay_dt, savepath, set_number, datatype):

#     print('{} out of {} are Finished!'.format(n, N0+M0), end='\r')
    
#     traj_list = np.load(savepath+'set_{}/'.format(set_number)+'other_info/'+datatype+'/trajectories/traj_list_{}.npy'\
#                         .format(n), allow_pickle=True)
#     intensities_sample_cg_list, UV_flow_sample_list, distance_sample_list \
#         = generate_one_sample(M, Rs, traj_list, sigma, theta_r, theta_matrix, coord_matrix, space_filter, K, L,\
#                               dt, sample_dt, delay_dt)
#     if n <= N0:
#         np.save(savepath+'set_{}/'.format(set_number)+'training/'+datatype\
#                 +'/intensities_samples_cg/intensities_sample_cg_list_{}'\
#                 .format(n), intensities_sample_cg_list)
#         np.save(savepath+'set_{}/'.format(set_number)+'training/'+datatype\
#                 +'/UV_flow_samples/UV_flow_sample_list_{}'\
#                 .format(n), UV_flow_sample_list)
#     else:
#         np.save(savepath+'set_{}/'.format(set_number)+'testing/'+datatype\
#                 +'/intensities_samples_cg/intensities_sample_cg_list_{}'\
#                 .format(n), intensities_sample_cg_list)
#         np.save(savepath+'set_{}/'.format(set_number)+'testing/'+datatype\
#                 +'/UV_flow_samples/UV_flow_sample_list_{}'\
#                 .format(n), UV_flow_sample_list)
#     if datatype != 'rotation':
#         np.save(savepath+'set_{}/'.format(set_number)+'other_info/'+datatype\
#                 +'/distances/sample/distance_sample_list_{}'\
#                 .format(n), distance_sample_list)
        
        
# # Generate one sample of optical stimulus        
# def generate_one_sample(\
#     M, Rs, traj_list, sigma, theta_r, theta_matrix, coord_matrix, space_filter, K, L, dt, sample_dt, delay_dt):
#     """
#     Args:
#     M: # of lplc2 units
#     Rs: radii of the balls 
#     traj_list: a list of trajectories
#     sigma: noise added to images
#     theta_r: half of the receptive field width (rad)
#     theta_matrix: theta matrix
#     coord_matrix: coordinate matrix
#     K: K*K is the total number of elements.
#     L: element dimension.
#     dt: timescale of the simulations
#     sample_dt: timescale of sampling
#     delay_dt: timescale of delay in the motion detector.
    
#     Returns:
#     intensities_sample_cg_list: list of intensities_sample_cg (coarse-grained (cg) optical stimulus, steps (or lower) by M by K*K)
#     UV_flow_sample_list: list of flow fields (steps (or lower) by M by K*K by 4)
#     distance_sample_list: list of distance (steps by P by 0)
#     """
#     N = K * L + 4 * L # Size of each frame, the additional 4*L is for spatial filtering and will be deleted afterwards.
#     pad = 2 * L
#     leftup_corners = opsg.get_leftup_corners(K, L, pad)
    
#     sample_step = np.int(sample_dt/dt)
#     delay_step = np.int(delay_dt/dt)
    
#     intensities_sample_cg_list = []
#     UV_flow_sample_list = []
#     distance_sample_list = []
    
#     NN = len(traj_list)
#     for nn in range(NN):
#         traj = traj_list[nn]
#         steps = len(traj)
#         intensities_sample_cg = []
#         UV_flow_sample = []
#         distance = []
#         signal_filtered_all = np.zeros((M, K*K, 4))
#         assert steps > sample_step and steps > delay_step, print('Error: trajectory is too short!')
#         for step in range(delay_step, steps):
#             # Calculate the distance
#             Ds = dn3d.get_radial_distances(traj[step])
#             if step > 0 and step % sample_step == 0:
#                 # the previous frame
#                 pos1 = traj[step-delay_step]
#                 _, cf_raw1, _ = opsg.get_one_intensity(M, pos1, Rs, theta_r, theta_matrix, coord_matrix, K, L, pad, sigma)
#                 _, signal_cur1 = \
#                     flfd.get_filtered_and_current(signal_filtered_all, cf_raw1, \
#                                                   leftup_corners, space_filter, K, L, pad, dt, delay_dt)
#                 # the current frame
#                 pos = traj[step]
#                 _, cf_raw, _ = opsg.get_one_intensity(M, pos, Rs, theta_r, theta_matrix, coord_matrix, K, L, pad, sigma)
#                 _, signal_cur = \
#                     flfd.get_filtered_and_current(signal_filtered_all, cf_raw, \
#                                                   leftup_corners, space_filter, K, L, pad, dt, delay_dt)
#                 # Obtain the coarse-grained frame
#                 intensity_cg = opsg.get_intensity_cg(cf_raw, leftup_corners, K, L, pad)
#                 intensities_sample_cg.append(intensity_cg)
#                 # Calculate the flow field: U, V  
#                 UV_flow = flfd.get_flow_fields(signal_cur1, signal_cur, leftup_corners, K, L, pad)
#                 UV_flow_sample.append(UV_flow)
#                 distance.append(Ds)
#         intensities_sample_cg_list.append(intensities_sample_cg)
#         UV_flow_sample_list.append(UV_flow_sample)
#         distance_sample_list.append(distance)
    
#     return intensities_sample_cg_list, UV_flow_sample_list, distance_sample_list


####### Exponential filter #######
# Generate certain amount of samples for each type
def generate_samples_par_exp(M, R, dt, dynamics_fun, eta_1, sigma, theta_r, K, L, sample_dt, delay_dt, D_min, D_max, v_min, v_max,\
                         P, steps_r, D_min_r, D_max_r, scal, N1, N2, N3, N4, M1, M2, M3, M4, NNs,\
                         set_number, savepath, num_cores):
    """
    Args:
    M: # of lplc2 units
    R: radius of the ball 
    dt: time step (sec)
    dynamics_fun: predefined dynamics of the object, return the accelerations (/sec^2)
    eta_1: random force added on the ball (/sec^2)
    sigma: noise added to images
    theta_r: half of the receptive field width (rad)
    K: K*K is the total number of elements.
    L: element dimension.
    sample_dt: timescale of sampling
    delay_dt: timescale of delay in the motion detector.
    D_min: minimum initial distance 
    D_max: maximum initial distance 
    v_min: minimum velocity (sec^-1)
    v_max: maximum velocity (sec^-1)
    P: number of balls in rotation scenes
    steps_r: number of steps in rotation scenes
    D_min_r: minimum distance of the balls in the rotation scenes
    D_max_r: maximum distance of the balls in the rotation scenes
    scal: scale of the rotaion, in degrees
    N1: # of training samples of hit signals
    N2: # of training samples of miss signals
    N3: # of training samples of retreat signals
    N4: # of training samples of rotation signals
    M1: # of testing samples of hit signals
    M2: # of testing samples of miss signals
    M3: # of testing samples of retreat signals
    M4: # of testing samples of rotation signals
    NNs: # of sample trajectories in each list
    set_number: int, set number
    savepath: path to save the sample data
    num_cores: # of cores used in parallel
    """
    
    f0 = open(savepath+'set_{}/general_info.txt'.format(set_number), 'a')
    f0.write('M = {}: # of lplc2 units\n'.format(M))
    f0.write('R = {}: radius of the ball \n'.format(R))
    f0.write('dt = {}: time step (sec)\n'.format(dt))
    f0.write('dynamics_fun = '+dynamics_fun.__name__+': predefined dynamics of the object, return the accelerations (m/sec^2)\n')
    f0.write('eta_1 = {}: random force added on the ball (m/sec^2)\n'.format(eta_1))
    f0.write('sigma = {}: noise added to images\n'.format(sigma))
    f0.write('theta_r = {:.4g}: half of the receptive field width (rad)\n'.format(theta_r))
    f0.write('K = {}: K*K is the total number of elements\n'.format(K))
    f0.write('L = {}: element dimension\n'.format(L))
    f0.write('sample_dt = {}: timescale of sampling\n'.format(sample_dt))
    f0.write('delay_dt = {}: timescale of delay in the motion detector\n'.format(delay_dt))
    f0.write('D_min = {}: minimum initial distance \n'.format(D_min))
    f0.write('D_max = {}: maximum initial distance \n'.format(D_max))
    f0.write('v_min = {}: minimum velocity (sec^-1) \n'.format(v_min))
    f0.write('v_max = {}: maximum velocity (sec^-1) \n'.format(v_max))
    f0.write('P = {}: number of balls in rotation scenes \n'.format(P))
    f0.write('steps_r = {}: number of steps in rotation scenes \n'.format(steps_r))
    f0.write('D_min_r = {}: minimum distance of the balls in the rotation scenes \n'.format(D_min_r))
    f0.write('D_max_r = {}: maximum distance of the balls in the rotation scenes \n'.format(D_max_r))
    f0.write('scal = {}: scale of the rotaion (deg*sec^-1) \n'.format(scal))
    f0.write('N1 = {}: # of lists of training samples of hit signals\n'.format(N1))
    f0.write('N2 = {}: # of lists of training samples of miss signals\n'.format(N2))
    f0.write('N3 = {}: # of lists of training samples of retreat signals\n'.format(N3))
    f0.write('N4 = {}: # of lists of training samples of rotation signals\n'.format(N4))
    f0.write('M1 = {}: # of lists of testing samples of hit signals\n'.format(M1))
    f0.write('M2 = {}: # of lists of testing samples of miss signals\n'.format(M2))
    f0.write('M3 = {}: # of lists of testing samples of retreat signals\n'.format(M3))
    f0.write('M4 = {}: # of lists of testing samples of rotation signals\n'.format(M4))
    f0.write('NNs = {}: # of samples in each list\n'.format(NNs))
    f0.close()   
    
    N = K * L + 4 * L # Size of each frame, the additional 4*L is for spatial filtering and will be deleted afterwards.
    coord_y = np.arange(N) - (N - 1) / 2.
    coord_x = np.arange(N) - (N - 1) / 2.
    coords_y, coords_x = np.meshgrid(coord_y, -coord_x) # coordinates of each point in the frame
    dm = np.sqrt(coords_y**2 + coords_x**2) # distance matrix
    theta_matrix, phi_matrix = opsg.get_angle_matrix(theta_r, coords_x, coords_y, dm, K, L)
    coord_matrix = opsg.get_coord_matrix(phi_matrix, theta_matrix, D=1.)
    space_filter = flfd.get_space_filter(L/2, 4)
    
    # hit samples
    print('Generating hit samples:')
    datatype = 'hit'
    Rs = np.array([R])
    generate_trajectory_hit(\
        M, R, D_max, v_min, v_max, theta_r, dt, delay_dt, dynamics_fun, eta_1, N1+M1, NNs, savepath, set_number, datatype)
    Parallel(n_jobs=num_cores)\
            (delayed(generate_sample_par_exp)\
             (n1, N1, M1, M, Rs, sigma, theta_r, theta_matrix, coord_matrix, space_filter, K, L,\
              dt, sample_dt, delay_dt, savepath, set_number, datatype)\
             for n1 in range(1, N1+M1+1))
    print('{} trainging samples have been generated!'.format(N1))
    print('{} testing samples have been generated!'.format(M1))
    
    # miss samples
    print('Generating miss samples:')
    datatype = 'miss'
    Rs = np.array([R])
    generate_trajectory_miss(\
        M, R, D_min, D_max, v_min, v_max, theta_r, dt, delay_dt, dynamics_fun, eta_1, N2+M2, NNs, savepath, set_number, datatype)
    Parallel(n_jobs=num_cores)\
            (delayed(generate_sample_par_exp)\
             (n2, N2, M2, M, Rs, sigma, theta_r, theta_matrix, coord_matrix, space_filter, K, L,\
              dt, sample_dt, delay_dt, savepath, set_number, datatype)\
             for n2 in range(1, N2+M2+1))
    print('{} trainging samples have been generated!'.format(N2))
    print('{} testing samples have been generated!'.format(M2))
    
    # retreat samples
    print('Generating retreat samples:')
    datatype = 'retreat'
    Rs = np.array([R])
    generate_trajectory_retreat(\
        M, R, D_max, v_min, v_max, theta_r, dt, delay_dt, dynamics_fun, eta_1, N3+M3, NNs, savepath, set_number, datatype)
    Parallel(n_jobs=num_cores)\
            (delayed(generate_sample_par_exp)\
             (n3, N3, M3, M, Rs, sigma, theta_r, theta_matrix, coord_matrix, space_filter, K, L,\
              dt, sample_dt, delay_dt, savepath, set_number, datatype)\
             for n3 in range(1, N3+M3+1))
    print('{} trainging samples have been generated!'.format(N3))
    print('{} testing samples have been generated!'.format(M3))
    
    # rotation scene samples
    print('Generating rotation scene samples:')
    datatype = 'rotation'
    Rs = np.random.random(P) * R
    generate_trajectory_rot(M, D_min_r, D_max_r, P, steps_r, dt, scal, N4+M4, NNs, savepath, set_number, datatype)
    Parallel(n_jobs=num_cores)\
            (delayed(generate_sample_par_exp)\
             (n4, N4, M4, M, Rs, sigma, theta_r, theta_matrix, coord_matrix, space_filter, K, L,\
              dt, sample_dt, delay_dt, savepath, set_number, datatype)\
             for n4 in range(1, N4+M4+1)) 
    print('{} training samples have been generated!'.format(N4))
    print('{} testing samples have been generated!'.format(M4))


# Generate one sample, in parallel
def generate_sample_par_exp(n, N0, M0, M, Rs, sigma, theta_r, theta_matrix, coord_matrix, space_filter, K, L,\
                        dt, sample_dt, delay_dt, savepath, set_number, datatype):

    print('{} out of {} are Finished!'.format(n, N0+M0), end='\r')
    
    traj_list = np.load(savepath+'set_{}/'.format(set_number)+'other_info/'+datatype+'/trajectories/traj_list_{}.npy'\
                        .format(n), allow_pickle=True)
    intensities_sample_cg_list, UV_flow_sample_list, distance_sample_list \
        = generate_one_sample_exp(M, Rs, traj_list, sigma, theta_r, theta_matrix, coord_matrix, space_filter, K, L,\
                              dt, sample_dt, delay_dt)
    if n <= N0:
        np.save(savepath+'set_{}/'.format(set_number)+'training/'+datatype\
                +'/intensities_samples_cg/intensities_sample_cg_list_{}'\
                .format(n), intensities_sample_cg_list)
        np.save(savepath+'set_{}/'.format(set_number)+'training/'+datatype\
                +'/UV_flow_samples/UV_flow_sample_list_{}'\
                .format(n), UV_flow_sample_list)
    else:
        np.save(savepath+'set_{}/'.format(set_number)+'testing/'+datatype\
                +'/intensities_samples_cg/intensities_sample_cg_list_{}'\
                .format(n), intensities_sample_cg_list)
        np.save(savepath+'set_{}/'.format(set_number)+'testing/'+datatype\
                +'/UV_flow_samples/UV_flow_sample_list_{}'\
                .format(n), UV_flow_sample_list)
    if datatype != 'rotation':
        np.save(savepath+'set_{}/'.format(set_number)+'other_info/'+datatype\
                +'/distances/sample/distance_sample_list_{}'\
                .format(n), distance_sample_list)
        
        
# Generate one sample of optical stimulus        
def generate_one_sample_exp(\
    M, Rs, traj_list, sigma, theta_r, theta_matrix, coord_matrix, space_filter, K, L, dt, sample_dt, delay_dt):
    """
    Args:
    M: # of lplc2 units
    Rs: radii of the balls 
    traj_list: a list of trajectories
    sigma: noise added to images
    theta_r: half of the receptive field width (rad)
    theta_matrix: theta matrix
    coord_matrix: coordinate matrix
    K: K*K is the total number of elements.
    L: element dimension.
    dt: timescale of the simulations
    sample_dt: timescale of sampling
    delay_dt: timescale of delay in the motion detector.
    
    Returns:
    intensities_sample_cg_list: list of intensities_sample_cg (coarse-grained (cg) optical stimulus, steps (or lower) by M by K*K)
    UV_flow_sample_list: list of flow fields (steps (or lower) by M by K*K by 4)
    distance_sample_list: list of distance (steps by P by 0)
    """
    N = K * L + 4 * L # Size of each frame, the additional 4*L is for spatial filtering and will be deleted afterwards.
    pad = 2 * L
    leftup_corners = opsg.get_leftup_corners(K, L, pad)
    
    sample_step = np.int(sample_dt/dt)
    delay_step = np.int(delay_dt/dt)
    
    intensities_sample_cg_list = []
    UV_flow_sample_list = []
    distance_sample_list = []
    
    NN = len(traj_list)
    for nn in range(NN):
        traj = traj_list[nn]
        steps = len(traj)
        intensities_sample_cg = []
        UV_flow_sample = []
        distance = []
        signal_filtered_all = np.zeros((M, K*K, 4))
        assert steps > sample_step and steps > delay_step, print('Error: trajectory is too short!')
        for step in range(steps):
            # Calculate the distance
            Ds = dn3d.get_radial_distances(traj[step])
            if step > 0 and step % sample_step == 0:
                # the current frame
                pos = traj[step]
                cf, cf_raw, hit = opsg.get_one_intensity(M, pos, Rs, theta_r, theta_matrix, coord_matrix, K, L, pad, sigma)
                # filtered signal
                signal_filtered_all, signal_cur = \
                    flfd.get_filtered_and_current(signal_filtered_all, cf_raw, leftup_corners, space_filter, K, L, pad, dt, delay_dt)
                # Obtain the coarse-grained frame
                intensity_cg = opsg.get_intensity_cg(cf_raw, leftup_corners, K, L, pad)
                intensities_sample_cg.append(intensity_cg)
                # Calculate the flow field: U, V  
                UV_flow = flfd.get_flow_fields(signal_filtered_all, signal_cur, leftup_corners, K, L, pad)
                UV_flow_sample.append(UV_flow)
                distance.append(Ds)
        intensities_sample_cg_list.append(intensities_sample_cg)
        UV_flow_sample_list.append(UV_flow_sample)
        distance_sample_list.append(distance)
    
    return intensities_sample_cg_list, UV_flow_sample_list, distance_sample_list
##################################


# Generate the trajectory for hit optical stimulus        
def generate_trajectory_hit(\
    M, R, D_max, v_min, v_max, theta_r, dt, delay_dt, dynamics_fun, eta_1, Ns, NNs, savepath, set_number, datatype):
    """
    Args:
    M: # of lplc2 units
    R: radius of the ball 
    D_max: maximum distance from the origin 
    v_min: minimum velocity (sec^-1)
    v_max: maximum velocity (sec^-1)
    dt: time step (sec)
    delay_dt: timescale of delay in the motion detector (sec)
    dynamics_fun: predefined dynamics of the object, return the accelerations (/sec^2)
    eta_1: random force added on the ball (/sec^2)
    Ns: # of sample lists
    NNs: # of sample trajectories in each list
    set_number: int, set number
    savepath: path to save the trajectory samples
    """
    n = 1
    while n <= Ns:
        traj_list = []
        distance_list = []
        nn = 1
        while nn <= NNs:
            # Initiation
            x, y, z, vx, vy, vz = generate_init_condition(0, R, v_min, v_max)
            traj,distance = \
                generate_one_trajectory_hit(M, R, D_max, theta_r, dt, dynamics_fun, eta_1, x, y, z, vx, vy, vz)
            if len(traj) * dt > delay_dt:
                traj_list.append(traj)
                distance_list.append(distance)
                nn = nn + 1
        np.save(savepath+'set_{}/'.format(set_number)+'other_info/'+datatype+'/trajectories/traj_list_{}'\
                    .format(n), traj_list)
        np.save(savepath+'set_{}/'.format(set_number)+'other_info/'+datatype+'/distances/distance_list_{}'\
                    .format(n), distance_list)
        n = n + 1
    

# Generate the trajectory for miss optical stimulus        
def generate_trajectory_miss(\
    M, R, D_min, D_max, v_min, v_max, theta_r, dt, delay_dt, dynamics_fun, eta_1, Ns, NNs, savepath, set_number, datatype):
    """
    Args:
    M: # of lplc2 units
    R: radius of the ball 
    D_min: minimum initial value of distance , bigger than R
    D_max: maximum initial value of distance 
    v_min: minimum velocity (sec^-1)
    v_max: maximum velocity (sec^-1)
    dt: time step (sec)
    delay_dt: timescale of delay in the motion detector (sec).
    dynamics_fun: predefined dynamics of the object, return the accelerations (/sec^2)
    eta_1: random force added on the ball (/sec^2)
    Ns: # of sample lists
    NNs: # of sample trajectories in each list
    set_number: int, set number
    savepath: path to save the trajectory samples
    """
    n = 1
    while n <= Ns:
        traj_list = []
        distance_list = []
        occupancy_list = []
        nn = 1
        while nn <= NNs:
            # Initiation
            x, y, z, vx, vy, vz = generate_init_condition(D_min, D_max, v_min, v_max)
            vx = np.abs(vx) * (-np.sign(x))
            vy = np.abs(vy) * (-np.sign(y))
            vz = np.abs(vz) * (-np.sign(z))
            traj, distance =\
                generate_one_trajectory_miss(M, R, D_max, theta_r, dt, dynamics_fun, eta_1, x, y, z, vx, vy, vz)
            if len(distance) > 0:
                D = np.min(distance)
            else:
                D = 0
            if D > R:
                min_D = np.argmin(distance)
                if len(traj[:min_D]) * dt > delay_dt:
                    traj_list.append(traj[:min_D])
                    distance_list.append(distance[:min_D])
                    nn = nn + 1
        np.save(savepath+'set_{}/'.format(set_number)+'other_info/'+datatype+'/trajectories/traj_list_{}'\
                .format(n), traj_list)
        np.save(savepath+'set_{}/'.format(set_number)+'other_info/'+datatype+'/distances/distance_list_{}'\
                .format(n), distance_list)
        n = n + 1  
        
        
# Generate the trajectory for retreat optical stimulus        
def generate_trajectory_retreat(\
        M, R, D_max, v_min, v_max, theta_r, dt, delay_dt, dynamics_fun, eta_1, Ns, NNs, savepath, set_number, datatype):
    """
    Args:
    M: # of lplc2 units
    R: radius of the ball 
    D_max: maximum distance from the origin 
    v_min: minimum velocity (sec^-1)
    v_max: maximum velocity (sec^-1)
    dt: time step (sec)
    delay_dt: timescale of delay in the motion detector (sec)
    dynamics_fun: predefined dynamics of the object, return the accelerations (/sec^2)
    eta_1: random force added on the ball (/sec^2)
    Ns: # of sample lists
    NNs: # of sample trajectories in each list
    set_number: int, set number
    savepath: path to save the trajectory samples
    """
    n = 1
    while n <= Ns:
        traj_list = []
        distance_list = []
        nn = 1
        while nn <= NNs:
            # Initiation
            x, y, z, vx, vy, vz = generate_init_condition(0, R, v_min, v_max)
            traj, distance = \
                generate_one_trajectory_retreat(M, R, D_max, theta_r, dt, dynamics_fun, eta_1, x, y, z, vx, vy, vz)
            if len(traj) * dt > delay_dt:
                traj_list.append(traj)
                distance_list.append(distance)
                nn = nn + 1
        np.save(savepath+'set_{}/'.format(set_number)+'other_info/'+datatype+'/trajectories/traj_list_{}'\
                    .format(n), traj_list)
        np.save(savepath+'set_{}/'.format(set_number)+'other_info/'+datatype+'/distances/distance_list_{}'\
                    .format(n), distance_list)
        n = n + 1
        
        
# Generate trajectory for rotation scene      
def generate_trajectory_rot(M, D_min, D_max, P, steps, dt, scal, Ns, NNs, savepath, set_number, datatype):
    """
    Args:
    M: # of lplc2 units
    D_min: minimum distance
    D_max: maximum distance
    P: # of balls
    steps: # of steps
    dt: time step (sec)
    scal: scale of the rotaion, in degrees
    Ns: # of sample lists
    NNs: # of sample trajectories in each list
    savepath: path to save the trajectory samples
    set_number: int, set number
    datatype: data type
    """
    n = 1
    while n <= Ns:
        traj_list = []
        nn = 1
        while nn <= NNs:
            theta_s = np.arccos(2*np.random.random()-1)
            phi_s = 2. * np.pi * np.random.random()
            xa = np.sin(theta_s) * np.cos(phi_s)
            ya = np.sin(theta_s) * np.sin(phi_s)
            za = np.cos(theta_s)
            traj = generate_one_trajectory_rot(M, D_min, D_max, P, steps, dt, xa, ya, za, scal)
            traj_list.append(traj)
            nn = nn + 1
        np.save(savepath+'set_{}/'.format(set_number)+'other_info/'+datatype+'/trajectories/traj_list_{}'\
                .format(n), traj_list)
        n = n + 1
        
        
# Generate one trajectory for hit optical stimulus        
def generate_one_trajectory_hit(M, R, D_max, theta_r, dt, dynamics_fun, eta_1, x, y, z, vx, vy, vz):
    """
    Args:
    M: # of lplc2 units
    R: radius of the ball 
    D_max: maximum distance from the origin 
    theta_r: half of the receptive field width (rad)
    dt: time step (sec)
    dynamics_fun: predefined dynamics of the object, return the accelerations (/sec^2)
    eta_1: random force added on the ball (/sec^2)
    x, y, z, vx, vy, vz: initial position and velocity
    """
    D = dn3d.get_radial_distance(x, y, z)
    step = 0
    traj = []
    distance = []
    while D <= D_max:
        if D > R:
            theta_b = np.arcsin(np.minimum(R/D, 1)) * 180. / np.pi
            angles_with_lplc2 = opsg.get_angles_between_lplc2_and_vec(M, [x, y, z])
            traj.insert(0, [[x, y, z]])
            distance.insert(0, D)
        # Update the dynamics
        x, y, z = dn3d.update_position(x, y, z, vx, vy, vz, dt)
        ax, ay, az = dynamics_fun(x, y, z, vx, vy, vz, eta_1, step, dt)
        vx, vy, vz = dn3d.update_velocity(vx, vy, vz, ax, ay, az, dt)
        step = step + 1
        D = dn3d.get_radial_distance(x, y, z)
        
    return np.array(traj, np.float32), np.array(distance, np.float32)
            

# Generate one trajectory for miss optical stimulus        
def generate_one_trajectory_miss(M, R, D_max, theta_r, dt, dynamics_fun, eta_1, x, y, z, vx, vy, vz):
    """
    Args:
    M: # of lplc2 units
    R: radius of the ball 
    D_max: maximum distance from the origin 
    theta_r: half of the receptive field width (rad)
    dt: time step (sec)
    dynamics_fun: predefined dynamics of the object, return the accelerations (/sec^2)
    eta_1: random force added on the ball (/sec^2)
    x, y, z, vx, vy, vz: initial position and velocity
    """
    D = dn3d.get_radial_distance(x, y, z)
    step = 0
    traj = []
    distance = []
    while D <= D_max:
        if D <= R:
            traj = []
            distance = []
            break
        theta_b = np.arcsin(np.minimum(R/D, 1)) * 180. / np.pi
        angles_with_lplc2 = opsg.get_angles_between_lplc2_and_vec(M, [x, y, z])
        traj.append([[x, y, z]])
        distance.append(D)
        # Update the dynamics
        x, y, z = dn3d.update_position(x, y, z, vx, vy, vz, dt)
        ax, ay, az = dynamics_fun(x, y, z, vx, vy, vz, eta_1, step, dt)
        vx, vy, vz = dn3d.update_velocity(vx, vy, vz, ax, ay, az, dt)
        step = step + 1
        D = dn3d.get_radial_distance(x, y, z)
        
    return np.array(traj, np.float32), np.array(distance, np.float32)
        
        
# Generate one trajectory for retreat optical stimulus        
def generate_one_trajectory_retreat(M, R, D_max, theta_r, dt, dynamics_fun, eta_1, x, y, z, vx, vy, vz):
    """
    Args:
    M: # of lplc2 units
    R: radius of the ball 
    D_max: maximum distance from the origin 
    theta_r: half of the receptive field width (rad)
    dt: time step (sec)
    dynamics_fun: predefined dynamics of the object, return the accelerations (/sec^2)
    eta_1: random force added on the ball (/sec^2)
    x, y, z, vx, vy, vz: initial position and velocity
    """
    D = dn3d.get_radial_distance(x, y, z)
    step = 0
    traj = []
    distance = []
    while D <= D_max:
        if D > R:
            theta_b = np.arcsin(np.minimum(R/D, 1)) * 180. / np.pi
            angles_with_lplc2 = opsg.get_angles_between_lplc2_and_vec(M, [x, y, z])
            traj.append([[x, y, z]])
            distance.append(D)
        # Update the dynamics
        x, y, z = dn3d.update_position(x, y, z, vx, vy, vz, dt)
        ax, ay, az = dynamics_fun(x, y, z, vx, vy, vz, eta_1, step, dt)
        vx, vy, vz = dn3d.update_velocity(vx, vy, vz, ax, ay, az, dt)
        step = step + 1
        D = dn3d.get_radial_distance(x, y, z)
                
    return np.array(traj, np.float32), np.array(distance, np.float32)
        
        
# Generate one trajectory for rotation scene      
def generate_one_trajectory_rot(M, D_min, D_max, P, steps, dt, xa, ya, za, scal=200, random_z=True, fixed_z=0):
    """
    Args:
    M: # of lplc2 units
    D_min: minimum distance
    D_max: maximum distance
    P: # of balls
    steps: # of steps
    dt: time step (sec)
    xa, ya, za: the direction of the rotation
    scal: scale of the rotaion, in degrees
    random_z: whether randomly select the rotation speed around 'z' (rotated z, not the original)
    fixed_z: predetermined rotation speed around 'z' (rotated z, not the original)
    """
    traj = []
    pos = []
    for p in range(P):
        D = D_min + (D_max - D_min) * np.random.random()
        theta_s = np.arccos(2*np.random.random()-1)
        phi_s = 2. * np.pi * np.random.random()
        x = D * np.sin(theta_s) * np.cos(phi_s)
        y = D * np.sin(theta_s) * np.sin(phi_s)
        z = D * np.cos(theta_s)
        pos.append([x, y, z])
    pos = np.array(pos)
    angle = opsg.get_xy_angles(xa, ya, za)
    pos_r = opsg.get_rotated_coordinates(-angle, pos)
    if random_z:
        around_z = np.random.normal(scale=scal*dt)
    else:
        around_z = fixed_z*dt
    r = R3d.from_euler('ZYX', [around_z, 0, 0], degrees=True)
    for step in range(steps):
        traj.append(pos)
        pos_r = r.apply(pos_r)
        pos = opsg.get_rotated_coordinates_rev(angle, pos_r)
        
    return np.array(traj, np.float32)


# Generate initial conditions
def generate_init_condition(D_min, D_max, v_min, v_max):
    """
    Args:
    D_min: minimum initial value of distance , bigger than R
    D_max: maximum initial value of distance 
    v_min: minimum velocity (sec^-1)
    v_max: maximum velocity (sec^-1)
    
    Returns:
    x, y, z: initial values of the coordinate 
    vx, vy, vz: initial values of the velocity (/sec)
    """
    D = D_min + (D_max - D_min) * np.random.random()
    theta_s = np.arccos(2*np.random.random()-1)
    phi_s = 2. * np.pi * np.random.random()
    x = D * np.sin(theta_s) * np.cos(phi_s)
    y = D * np.sin(theta_s) * np.sin(phi_s)
    z = D * np.cos(theta_s)
    
    v = v_min + (v_max - v_min) * np.random.random()
    theta_s = np.arccos(2*np.random.random()-1)
    phi_s = 2. * np.pi * np.random.random()
    vx = v * np.sin(theta_s) * np.cos(phi_s)
    vy = v * np.sin(theta_s) * np.sin(phi_s)
    vz = v * np.cos(theta_s)

    return x, y, z, vx, vy, vz


##########################################
####### generate samples in a grid #######
##########################################
    
    
# Get grid of initials
def get_grid_init(D, angle_xy, na):
    """
    Args:
    D: distance
    angle_xy: 
    na: 
    
    Returns:
    grid_init:
    """
    NA = np.int(2*na+1)
    grid_init = np.zeros((NA, NA, 3))
    x_angles = np.arange(np.int(na*5), -np.int(na*5+1), -5)
    y_angles = np.arange(np.int(na*5), -np.int(na*5+1), -5)
    arr_c = np.zeros(3)
    arr_c[2] = D
    for i in range(NA):
        for j in range(NA):
            phi1 = np.deg2rad(5.)
            theta1 = np.deg2rad(x_angles[j])
            scale = np.rad2deg(2*np.arcsin(np.sin(phi1/2)/np.cos(theta1))) / 5.
            if x_angles[j] == 90:
                r = R3d.from_euler('xzy', [x_angles[j], y_angles[i]*scale, 0], degrees=True)
            elif x_angles[j] == -90:
                r = R3d.from_euler('xzy', [x_angles[j], -y_angles[i]*scale, 0], degrees=True)
            else:
                r = R3d.from_euler('xyz', [x_angles[j], y_angles[i]*scale, 0], degrees=True)
            grid_init[i, j, :] = r.apply(arr_c)[:]
    grid_init_reshaped = grid_init.reshape((NA*NA, 3))
    grid_init_reshaped = opsg.get_rotated_coordinates_rev(angle_xy, grid_init_reshaped)
    grid_init = grid_init_reshaped.reshape((NA, NA, 3))
                                      
    return grid_init


def generate_one_trajectory_grid(D_max, dt, dynamics_fun, eta_1, x, y, z, vx, vy, vz):
    """
    Args:
    D_max: maximum distance from the origin 
    dt: time step (sec)
    dynamics_fun: predefined dynamics of the object, return the accelerations (/sec^2)
    eta_1: random force added on the ball (/sec^2)
    x, y, z, vx, vy, vz: initial position and velocity

    Returns:
    traj: trajectory
    distance:
    """
    x0, y0, z0 = x, y, z
    D = dn3d.get_radial_distance(x-x0, y-y0, z-z0)
    step = 0
    traj = []
    distance = []
    while D <= D_max:
        traj.append([[x, y, z]])
        distance.append(D)
        # Update the dynamics
        x, y, z = dn3d.update_position(x, y, z, vx, vy, vz, dt)
        ax, ay, az = dynamics_fun(x, y, z, vx, vy, vz, eta_1, step, dt)
        vx, vy, vz = dn3d.update_velocity(vx, vy, vz, ax, ay, az, dt)
        step = step + 1
        D = dn3d.get_radial_distance(x-x0, y-y0, z-z0)
        
    return np.array(traj, np.float32), np.array(distance, np.float32)
        
        
# Generate samples for grid
def generate_samples_grid(M, D, D_max, na, R, V, dt, dynamics_fun, eta_1, sigma, theta_r, K, L, \
                          sample_dt, delay_dt, savepath, data_type):
    """
    Args:
    M: number of LPLC2 units
    D: initial distance of the incoming object
    D_max: maximum distance to travel
    na: half dimension size of the lattice
    R: radius of the ball
    V: maximum speed (/sec)
    dt: time step (sec)
    dynamics_fun: dynamics that is imposed on the object
    eta_1: random force added on the ball (/sec^2)
    sigma: noise added to images
    theta_r: half of the receptive field width (rad)
    K: K*K is the total number of elements
    L: L is the dimension of each element
    sample_dt: sampling resolution
    delay_dt: delay in the motion detector
    savepath: path to save the results
    data_type: type of data
    """
    N = K * L + 4 * L # Size of each frame, the additional 4*L is for spatial filtering and will be deleted afterwards.
    coord_y = np.arange(N) - (N-1) / 2.
    coord_x = np.arange(N) - (N-1) / 2.
    coords_y, coords_x = np.meshgrid(coord_y, -coord_x) # coordinates of each point in the frame
    dm = np.sqrt(coords_y**2 + coords_x**2) # distance matrix
    theta_matrix, phi_matrix = opsg.get_angle_matrix(theta_r, coords_x, coords_y, dm, K, L)
    coord_matrix = opsg.get_coord_matrix(phi_matrix, theta_matrix, D=1.)
    space_filter = flfd.get_space_filter(L/2, 4)
    
    Rs = np.array([R])
    if M == 1:
        angle_xy = np.zeros(2)
        arr_c = np.zeros(3)
        arr_c[2] = D
    else:
        angle_xy = opsg.get_lplc2_units_xy_angles(M)[0]
        _, lplc2_units_coords = opsg.get_lplc2_units(M)
        arr_c = lplc2_units_coords[0] * D
    grid_init = get_grid_init(D, angle_xy, na)
    d1 = grid_init.shape[0]
    d2 = grid_init.shape[1]   
    for i in range(d1):
        for j in range(d2):
            if data_type == 'convergence':
                x, y, z = grid_init[i, j, :]
                vx, vy, vz = (V / D) * (-x), (V / D) * (-y), (V / D) * (-z)
            elif data_type == 'divergence':
                x, y, z = grid_init[i, j, :]
                vx, vy, vz = (V / D) * (-x), (V / D) * (-y), (V / D) * (-z)
                x, y, z = arr_c
            elif data_type == 'parallel':
                x, y, z = grid_init[i, j, :]
                vx, vy, vz = (V / D) * (-arr_c[0]), (V / D) * (-arr_c[1]), (V / D) * (-arr_c[2])
            traj, _ = generate_one_trajectory_grid(D_max, dt, dynamics_fun, eta_1, x, y, z, vx, vy, vz)
            intensities_sample_cg_list, UV_flow_sample_list, distance_sample_list = generate_one_sample_exp(\
                M, Rs, np.array([traj]), sigma, theta_r, theta_matrix, coord_matrix, space_filter, K, L, dt, sample_dt, delay_dt)
            np.save(savepath+'trajectories/traj_{}_{}'.format(i+1,j+1), [traj])
            np.save(savepath+'intensities_samples_cg/intensities_sample_cg_{}_{}'.format(i+1,j+1), intensities_sample_cg_list)
            np.save(savepath+'UV_flow_samples/UV_flow_sample_{}_{}'.format(i+1,j+1), UV_flow_sample_list)
            

# Get grid of initials
def get_grid_init2(D,theta_max):
    """
    Args:
    D: distance
    theta_max: maximum theta 
    
    Returns:
    grid_init:
    """
    theta_angles = np.arange(0, theta_max+1, 5)
    phi_angles = np.arange(0, 360, 5)
    N1 = len(theta_angles)
    N2 = len(phi_angles)
    grid_init = np.zeros((N1, N2, 3))
    
    for i in range(N1):
        for j in range(N2):
            coord = dn3d.get_coord(D, np.deg2rad(phi_angles[j]), np.deg2rad(theta_angles[i]))
            grid_init[i, j, :] = coord[:]
                                      
    return grid_init
            
    
# Generate samples for grid
def generate_samples_grid2(M, D, D_max, theta_max, R, V, dt, dynamics_fun, eta_1, sigma, theta_r,\
                           K, L, sample_dt, delay_dt, savepath, grid_init, ij):
    """
    Args:
    M: number of LPLC2 units
    D: initial distance of the incoming object
    D_max: maximum distance to travel
    theta_max: maximum incoming angle
    R: radius of the ball
    V: maximum speed (/sec)
    dt: time step (sec)
    dynamics_fun: dynamics that is imposed on the object
    eta_1: random force added on the ball (/sec^2)
    sigma: noise added to images
    theta_r: half of the receptive field width (rad)
    K: K*K is the total number of elements
    L: L is the dimension of each element
    sample_dt: sampling resolution
    delay_dt: delay in the motion detector
    savepath: path to save the results
    grid_init: intitials of the trajectories
    ij: position indicator of the grid
    """
    N = K * L + 4 * L # Size of each frame, the additional 4*L is for spatial filtering and will be deleted afterwards.
    coord_y = np.arange(N) - (N - 1) / 2.
    coord_x = np.arange(N) - (N - 1) / 2.
    coords_y, coords_x = np.meshgrid(coord_y, -coord_x) # coordinates of each point in the frame
    dm = np.sqrt(coords_y**2 + coords_x**2) # distance matrix
    theta_matrix, phi_matrix = opsg.get_angle_matrix(theta_r, coords_x, coords_y, dm, K, L)
    coord_matrix = opsg.get_coord_matrix(phi_matrix, theta_matrix, D=1.)
    space_filter = flfd.get_space_filter(L/2, 4)
    
    Rs = np.array([R])
    i, j = ij
    x, y, z = grid_init[i, j, :]
    vx, vy, vz = (V / D) * (-x), (V / D) * (-y), (V / D) * (-z)
    traj, _ = generate_one_trajectory_grid(D_max, dt, dynamics_fun, eta_1, x, y, z, vx, vy, vz)
    intensities_sample_cg_list, UV_flow_sample_list, distance_sample_list = generate_one_sample_exp(\
        M, Rs, np.array([traj]), sigma, theta_r, theta_matrix, coord_matrix, space_filter, K, L, dt, sample_dt, delay_dt)
    np.save(savepath+'trajectories/traj_{}_{}'.format(i+1,j+1), [traj])
    np.save(savepath+'intensities_samples_cg/intensities_sample_cg_{}_{}'.format(i+1,j+1), intensities_sample_cg_list)
    np.save(savepath+'UV_flow_samples/UV_flow_sample_{}_{}'.format(i+1,j+1), UV_flow_sample_list)

            
def generate_samples_grid2_par(M, D, D_max, theta_max, R, V, dt, dynamics_fun, eta_1, sigma, theta_r,\
                           K, L, sample_dt, delay_dt, savepath, n_cores):   
    """
    Args:
    M: number of LPLC2 units
    D: initial distance of the incoming object
    D_max: maximum distance to travel
    theta_max: maximum incoming angle
    R: radius of the ball
    V: maximum speed (/sec)
    dt: time step (sec)
    dynamics_fun: dynamics that is imposed on the object
    eta_1: random force added on the ball (/sec^2)
    sigma: noise added to images
    theta_r: half of the receptive field width (rad)
    K: K*K is the total number of elements
    L: L is the dimension of each element
    sample_dt: sampling resolution
    delay_dt: delay in the motion detector
    savepath: path to save the results
    n_cores: number of cores used
    """
    grid_init = get_grid_init2(D, theta_max)
    d1 = grid_init.shape[0]
    d2 = grid_init.shape[1] 
    ij_list = []
    for i in range(d1):
        for j in range(d2):
            ij_list.append((i, j))

    Parallel(n_jobs=n_cores)\
            (delayed(generate_samples_grid2)\
                (M, D, D_max, theta_max, R, V, dt, dynamics_fun, eta_1, sigma, theta_r,\
                           K, L, sample_dt, delay_dt, savepath, grid_init, ij) for ij in ij_list)
    