#!/usr/bin/env python
# coding: utf-8


'''
This file gets the landscape of the models.
'''


import sys
sys.path.append('../stimulus_core/')
sys.path.append('../models_core/')
sys.path.append('../helper/')

import numpy as np
import importlib
import glob
import os
import tensorflow as tf
import multiprocessing
from datetime import datetime
from joblib import Parallel, delayed
import time

import flow_field as flfd
import helper_functions as hpfn
import optical_signal as opsg
import lplc2_models as lplc2


#################################### WARNING #########################################
########### DOUBLE CHECK args['use_ln'] AND args['restrict_nonneg_weight'] ###########
######################################################################################

def get_loss_landscape_f(M, n_cores):
    # some hyperparameters 
    args = {}
    args['use_ln'] = True # whether train an ln model without individual inhibitory units
    args['restrict_nonneg_weight'] = False # whether restrict the weights to be nonnegative
    args['glm'] = False # whether reduce the model to a glm, need args['use_ln'] to be true and args['restrict_nonneg_weight'] false
    args['rectified_inhibition'] = False # whether rectify the inhibition or not
    args['M'] = M # total number of model units
    args['save_folder'] = 'loss_landscape_ln_small_range_with_zero_solution/' # folder that stores the results
    args['data_path'] = '/home/bz242/project/data/loom/multi_lplc2_D5_L4_exp/' # path for the training and testing dataset
    args['set_number'] = np.int(1000+args['M']) # indicate which dataset to be used for training
    args['N_epochs'] = 2000 # number of training epochs
    args['rotational_fraction'] = 1.0 # fraction of rotational data that is used in training and testing
    args['K'] = 12 # size of one of the dimensions of the square receptive field matrix
    args['L'] = 4 # resoluton of of each element in the args['K'] by args['K'] receptive field.
    args['lr'] = 1e-3 # learning rate
    args['restrict_nonpos_intercept'] = False # whether restrict the intercepts to be nonpositive 
    args['rotation_symmetry'] = True # whether impose 90-deg rotation symmetry in the weights
    args['flip_symmetry'] = True # whether impose left-right or up-down symmetry in the weights
    args['train_a'] = False # whether train the slope a inside the probabilistic model (the sigmoid function)
    args['report_num'] = 400 # how many steps to report the training process once
    args['max_response'] = False # whether use maximum response or the averaged one over the whole trajectory
    args['temporal_filter'] = False # whether use the temporal filter to convolve the model response
    args['n'] = 1 # order of temporal filters, which is not used in our current model
    args['dt'] = 0.01 # time resolution of the temporal filters
    args['tau'] = 2.0 # timescale of the temporal filter, 0->1s, 1->368ms, 2-> 135ms, 3->50 ms,  4->18ms
    args['learn_tau'] = False # whether train the timescale args['tau'] of the temporal filter
    args['activation'] = 0 # activation function used in the model, 0 is ReLU, 1 is Leaky ReLU, 2 is ELU, 3 is tanh
    args['leaky_relu_constant'] = 0.02 # the coefficient for the negative part of the leaky ReLU
    args['square_activation'] = False # whether square the response of each model unit
    args['save_intermediate_weights'] = False # whether save intermediate weights
    args['save_steps'] = np.int(args['N_epochs']/10) # how many training epochs to save the intermediate weights once
    args['use_step_weight'] = False # whether use step weights to weight the response or not
    args['l1_regu_we'] = 0. # L1 regularization strength for excitatory weights
    args['l1_regu_wi'] = 0. # L1 regularization strength for inhibitory weights
    args['l1_regu_a'] = 0. # L1 regularization strength for slope a if args['train_a'] is set to be true
    args['l2_regu_we'] = 1e-4 # L2 regularization strength for excitatory weights
    args['l2_regu_wi'] = 1e-4 # L2 regularization strength for inhibitory weights
    args['l2_regu_a'] = 0. # L2 regularization strength for slope a if args['train_a'] is set to be true
    args['NNs'] = 10 # number of data samples that are combined into a list to reduce the amount of files
    args['S'] = 5 # number of lists of data samples in one batch, and thus batch size is args['NNs'] * args['S']
    if M in [1, 2, 4]:
        args['S'] = np.int(args['S'] * 8 / M)
    args['in_or_out'] = 0 # indicates inward or outward
    args['parameter_id'] = 0 # indicates which parameter

    # make the folders that store the training results
    args['save_path'] = '../results/' + args['save_folder']
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])

    # Regularization
    l1_l2_regu_we = tf.contrib.layers.l1_l2_regularizer(scale_l1=args["l1_regu_we"], scale_l2=args["l2_regu_we"], scope=None)
    l1_l2_regu_wi = tf.contrib.layers.l1_l2_regularizer(scale_l1=args["l1_regu_wi"], scale_l2=args["l2_regu_wi"], scope=None)
    l1_l2_regu_a = tf.contrib.layers.l1_l2_regularizer(scale_l1=args["l1_regu_a"], scale_l2=args["l2_regu_a"], scope=None)

    # Check if need process the training and testing data
    data_path = args['data_path']
    set_number = args['set_number']
    condition1 = os.path.exists(data_path + 'set_{}/training/'.format(set_number)+'train_flow_files.npy')
    condition2 = os.path.exists(data_path + 'set_{}/training/'.format(set_number)+'train_flow_snapshots.npy')
    if not (condition1 and condition2):
        # generate the training and testing samples, whole trajectories
        train_flow_files, _, train_labels, train_distances, \
        test_flow_files, _, test_labels, test_distances \
            = hpfn.generate_train_test_file_paths(args)  
        # generate the training samples, single frames (for each trajectory, one frame is randomly selected)
        hpfn.generate_train_data(args)

    # Load data
    train_flow_files = np.load(data_path + 'set_{}/training/'.format(set_number)+'train_flow_files.npy', allow_pickle=True)
    train_intensity_files = np.load(data_path + 'set_{}/training/'.format(set_number)+'train_intensity_files.npy', allow_pickle=True)
    train_labels = np.load(data_path + 'set_{}/training/'.format(set_number)+'train_labels.npy', allow_pickle=True)
    train_distances = np.load(data_path + 'set_{}/training/'.format(set_number)+'train_distances.npy', allow_pickle=True)
    train_flow_snapshots = np.load(data_path + 'set_{}/training/'.format(set_number)+'train_flow_snapshots.npy', allow_pickle=True)
    train_labels_snapshots = np.load(data_path + 'set_{}/training/'.format(set_number)+'train_labels_snapshots.npy', allow_pickle=True)
    args['train_distances'] = train_distances # distances for the trajectories of the training samples
    flow_shape = train_flow_snapshots.shape
    label_shape = train_labels_snapshots.shape
    train_flow_snapshots_r = train_flow_snapshots.reshape((1, flow_shape[0]*flow_shape[2], M, 144, 4))
    train_labels_snapshots_r = train_labels_snapshots.reshape((1, label_shape[0]*label_shape[2]))
    
    
    # combine those lists of hyper parameters
    config_list = []
    for in_or_out in [0, 1, 2]:
        for parameter_id in range(74):
            config_list.append({'in_or_out':in_or_out, \
                                'parameter_id':parameter_id})
    
    # train the model in parallel for the elements in config_list
    Parallel(n_jobs=n_cores)\
            (delayed(run_get_loss_landscape)\
                (config, args, train_flow_snapshots_r, train_labels_snapshots_r) for config in config_list)


def run_get_loss_landscape(config, args, train_flow_snapshots_r, train_labels_snapshots_r):
    """
    This function calculates the loss landscape.
    """

    # override configurations/hyperparameters
    for key in config:
        args[key] = config[key]
    
    # model parameters
    M = args['M']
    in_or_out = args['in_or_out']
    figure_path = '../results/'
    data_path = figure_path + 'model_clustering_ln_relu_correct_initialization/clusterings/'
    model_folders = np.load(data_path+'model_folders_M{}.npy'.format(M), allow_pickle=True)
    parameters_in_opt = np.zeros(147, dtype=np.float32)
    model_path = model_folders[in_or_out][0] + '/'
    model_path = model_path.replace('/Volumes/Baohua/research_large_files/loom_detection/', '../results/')
    a = np.load(model_path + "trained_a.npy")
    b = np.load(model_path + "trained_b.npy")  
    intercept_e = np.load(model_path + "trained_intercept_e.npy")
    weights_e = np.load(model_path + "trained_weights_e.npy")
    intercept_i = np.load(model_path + "trained_intercept_i.npy")
    weights_i = np.load(model_path + "trained_weights_i.npy")
    parameters_in_opt[0] = b
    parameters_in_opt[1] = intercept_e
    parameters_in_opt[2:74] = weights_e[:72, 0]
    parameters_in_opt[74] = intercept_i
    parameters_in_opt[75:147]= weights_i[:72, 0]

    # loss landscape
    sampling_vec = np.linspace(-1, 3, 101)
    loss_res_vec = np.zeros((2, len(sampling_vec)))
    parameter_id = args['parameter_id']
    parameters_in = parameters_in_opt.copy()
    parameter_interest = parameters_in_opt[parameter_id]
    start_time = time.time()
    for ii in range(len(sampling_vec)):
        if abs(parameter_interest) > 1e-4:
            parameters_in[parameter_id] = sampling_vec[ii] * parameter_interest
        else:
            parameters_in[parameter_id] = (sampling_vec[ii] - 1.) / 2.
        loss_res = \
            hpfn.get_gradient_hessian(args, train_flow_snapshots_r, train_labels_snapshots_r, parameters_in)
        loss_res_vec[0, ii] = parameters_in[parameter_id]
        loss_res_vec[1, ii] = loss_res
    print(f'Total time used for one parameter is {time.time()-start_time}.')

    np.save(args['save_path'] + f'loss_landscape_{in_or_out}_{parameter_id}', loss_res_vec)
