#!/usr/bin/env python
# coding: utf-8


'''
This file trains and tests the models.
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

import flow_field as flfd
import helper_functions as hpfn
import optical_signal as opsg
import lplc2_models as lplc2


#################################### WARNING #########################################
########### DOUBLE CHECK args['use_ln'] AND args['restrict_nonneg_weight'] ###########
######################################################################################

def train_multiple_units_f(M, seed_left, seed_right, n_cores):
    """
    This function trains the model in parallel of many combinations of hyperparameters.
    
    Args:
    M: number of model units
    seed_left: left boundary of the seed for random intialization
    seed_right: right boundary of the seed for random intialization
    n_cores: number of cpu cores used for training
    """
    # some hyperparameters 
    args = {}
    args['use_ln'] = True # whether train an ln model without individual inhibitory units
    args['restrict_nonneg_weight'] = False # whether restrict the weights to be nonnegative
    args['glm'] = False # whether reduce the model to a glm, need args['use_ln'] to be true and args['restrict_nonneg_weight'] false
    args['rectified_inhibition'] = False # whether rectify the inhibition or not
    args['M'] = M # total number of model units
    args['save_folder'] = 'multi_lplc2_training_D5_exp_ln_relu_with_half_constant_rot_scal200/' # folder that stores the trained results
    args['data_path'] = '../../data/loom/multi_lplc2_D5_L4_exp_with_half_constant_rot_scal200/' # path for the training and testing dataset
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
    train_flow_files = np.load(data_path + 'set_{}/training/'.format(set_number)+'train_flow_files.npy', \
                               allow_pickle=True)
    train_intensity_files = np.load(data_path + 'set_{}/training/'.format(set_number)+'train_intensity_files.npy', \
                                    allow_pickle=True)
    train_labels = np.load(data_path + 'set_{}/training/'.format(set_number)+'train_labels.npy', \
                           allow_pickle=True)
    train_distances = np.load(data_path + 'set_{}/training/'.format(set_number)+'train_distances.npy', \
                              allow_pickle=True)
    test_flow_files = np.load(data_path + 'set_{}/testing/'.format(set_number)+'test_flow_files.npy', \
                              allow_pickle=True)
    test_intensity_files = np.load(data_path + 'set_{}/testing/'.format(set_number)+'test_intensity_files.npy', \
                                   allow_pickle=True)
    test_labels = np.load(data_path + 'set_{}/testing/'.format(set_number)+'test_labels.npy', \
                          allow_pickle=True)
    test_distances = np.load(data_path + 'set_{}/testing/'.format(set_number)+'test_distances.npy', \
                             allow_pickle=True)
    train_flow_snapshots = np.load(data_path + 'set_{}/training/'.format(set_number)+'train_flow_snapshots.npy', \
                                   allow_pickle=True)
    train_labels_snapshots = np.load(data_path + 'set_{}/training/'.format(set_number)+'train_labels_snapshots.npy', \
                                     allow_pickle=True)
    
    args['train_distances'] = train_distances # distances for the trajectories of the training samples
    args['test_distances'] = test_distances # distances for the trajectories of the testing samples

    
    # get lists of hyper parameters
    lr_list = [1e-3] # list of learning rates
    regu_list = [1e-4] # list of regularization strength
    random_seed_list = list(range(seed_left, seed_right+1)) # do multiple initializations to be more robust
    
    
    # combine those lists of hyper parameters
    config_list = []
    for lr in lr_list:
        for regu in regu_list:
            for seed in random_seed_list:
                config_list.append({'lr':lr, \
                                    'l2_regu_we':regu, \
                                    'l2_regu_wi':regu, \
                                    'random_seed':seed})
    
    # train the model in parallel for the elements in config_list
    Parallel(n_jobs=n_cores)\
            (delayed(run_training_procedure)\
                (config, args, train_flow_snapshots, train_labels_snapshots, \
                     train_flow_files, train_labels, test_flow_files, test_labels) for config in config_list)


def run_training_procedure(config, args, train_flow_snapshots, train_labels_snapshots, \
                           train_flow_files, train_labels, test_flow_files, test_labels):
    """
    This function trains the models.
    """
    # seed
    np.random.seed(config['random_seed'])
    tf.random.set_random_seed(config['random_seed'])

    # override configurations/hyperparameters
    for key in config:
        args[key] = config[key]
    
    # make the folders that store the training results
    args['save_path'] =\
    '../results/'+args['save_folder']+'M_{}/lr_{}/regu_{}/seed_{}_{}/'\
        .format(args['M'], args['lr'], args['l2_regu_we'], config['random_seed'], datetime.now().strftime('%Y%m%d%H%M%S'))
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])
    
    # train and test
    weights_e_out, weights_i_out, intercept_e_out, intercept_i_out, tau_1_out, b_out, a_out\
        =lplc2.train_and_test_model(args, train_flow_files, train_labels, test_flow_files, test_labels, \
                                         train_flow_snapshots, train_labels_snapshots)

    # plot the trained weights
    weight_mask = opsg.get_disk_mask(args['K'], args['L'])
    colormap = 'RdBu'
    hpfn.plot_flow_weights(weights_e_out, weight_mask, 
                       colormap, args['save_path'] + '/weights_e_out.png')
    hpfn.plot_sym_flow_weights(weights_e_out, weight_mask, 
                       colormap, args['save_path'] + '/sym_weights_e_out.png')
    hpfn.plot_flow_weights(weights_i_out, weight_mask, 
                       colormap, args['save_path'] + '/weights_i_out.png')
    hpfn.plot_sym_flow_weights(weights_i_out, weight_mask, 
                       colormap, args['save_path'] + '/sym_weights_i_out.png')
