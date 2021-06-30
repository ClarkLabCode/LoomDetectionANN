#!/usr/bin/env python


"""
This module contains functions that define the LPLC2 models with four individual inhibitory units.
"""


import os
import numpy as np
import math
import tensorflow as tf
import random
import glob
import time
import sklearn.metrics as sm

import dynamics_3d as dn3d
import optical_signal as opsg
import helper_functions as hpfn


# Output of the model at a specific time point
def get_output_LPLC2(args, weights_e, weights_i, intercept_e, intercept_i, UV_flow_t):
    """
    This function takes the flow field as inputs, and outputs the sum of the responses 
    from all the M model units at a specific time point t. 
    
    Args:
    args: a dictionary that contains problem parameters, see train_multiple_units.py for definitions.
    weights_e: tensor, weights for excitatory units. 
               shape = K*K by 1 (with rotation symmetry).
               shape = K*K by 4 (without rotation symmetry), where
               for axis=1, 0: right motion, 1: left motion, 2: up motion, 3: down motion
    weights_i: tensor, weights for inhibitory units. 
               shape = K*K by 1 (with rotation symmetry).
               shape = K*K by 4 (without rotation symmetry), where
               for axis=1, 0: right motion, 1: left motion, 2: up motion, 3: down motion
    intercept_e: tensor, intercept in the activation function for the LPLC2 unit, shape = 0
    intercept_i: tensor, intercept in the activation function for inhibitory units, shape = 0
    UV_flow_t: tensor, flow field at time point t, shape = batch_size by M by K*K by 4
    
    Returns:
    output_t: output of the model at a specific time point t, shape = batch_size
    """
    K = args['K'] # size of one of the dimensions of the square receptive field matrix
    intercept_e = tf.dtypes.cast(intercept_e, tf.float32)  
    intercept_i = tf.dtypes.cast(intercept_i, tf.float32) 
    
    # add the intercept for the activation function of the lplc2 unit output
    output_t = intercept_e
        
    # rotation symmetry, counter-clockwise, 0: right motion, 1: left motion, 2: up motion, 3: down motion
    rotate_num_list = [0, 2, 1, 3] 
    
    # excitatory 
    for i in range(4):
        if args['rotation_symmetry']:
            weights_e_reshaped = tf.reshape(weights_e, [K, K, 1])
            weights_e_rotated = tf.image.rot90(weights_e_reshaped, rotate_num_list[i])
            weights_e_reshaped_back = tf.reshape(weights_e_rotated, [K*K, 1])
            output_t = output_t + tf.tensordot(UV_flow_t[:, :, :, i], weights_e_reshaped_back, axes=[[2], [0]])
        else:
            output_t = output_t + tf.tensordot(UV_flow_t[:, :, :, i], weights_e[:, i], axes=[[2], [0]])
    
    # inhibitory
    for i in range(4):
        if args['rotation_symmetry']:
            weights_i_reshaped = tf.reshape(weights_i, [K, K, 1])
            weights_i_rotated = tf.image.rot90(weights_i_reshaped, rotate_num_list[i])
            weights_i_reshaped_back = tf.reshape(weights_i_rotated, [K*K, 1])
            if args['activation'] == 0:
                output_t = output_t - tf.nn.relu(tf.tensordot(UV_flow_t[:, :, :, i], 
                    weights_i_reshaped_back, axes=[[2], [0]])+intercept_i)
            elif args['activation'] == 1:
                output_t = output_t - tf.nn.leaky_relu(tf.tensordot(UV_flow_t[:, :, :, i], 
                    weights_i_reshaped_back, axes=[[2], [0]])+intercept_i, alpha=args['leaky_relu_constant'])
            elif args['activation'] == 2:
                output_t = output_t - tf.nn.elu(tf.tensordot(UV_flow_t[:, :, :, i], 
                    weights_i_reshaped_back, axes=[[2], [0]])+intercept_i)
            elif args['activation'] == 3:
                output_t = output_t - tf.math.tanh(tf.tensordot(UV_flow_t[:, :, :, i], 
                    weights_i_reshaped_back, axes=[[2], [0]])+intercept_i)
        else:
            if args['activation'] == 0:
                output_t = output_t - tf.nn.relu(tf.tensordot(UV_flow_t[:, :, :, i], 
                    weights_i[:, i], axes=[[2], [0]])+intercept_i)
            elif args['activation'] == 1:
                output_t = output_t - tf.nn.leaky_relu(tf.tensordot(UV_flow_t[:, :, :, i], 
                    weights_i[:, i], axes=[[2], [0]])+intercept_i, alpha=args['leaky_relu_constant'])
            elif args['activation'] == 2:
                output_t = output_t - tf.nn.elu(tf.tensordot(UV_flow_t[:, :, :, i], 
                    weights_i[:, i], axes=[[2], [0]])+intercept_i)
            elif args['activation'] == 3:
                output_t = output_t - tf.math.tanh(tf.tensordot(UV_flow_t[:, :, :, i], 
                    weights_i[:, i], axes=[[2], [0]])+intercept_i)
    
    # sum responses across multiple units
    if args['activation'] == 0:
        if not args['square_activation']:
            output_t = tf.reduce_sum(tf.nn.relu(output_t), axis=1)
        else:
            output_t = tf.reduce_sum(tf.math.square(tf.nn.relu(output_t)), axis=1)
    elif args['activation'] == 1:
        if not args['square_activation']:
            output_t = tf.reduce_sum(tf.nn.leaky_relu(output_t, alpha=args['leaky_relu_constant']), axis=1)
        else:
            output_t = tf.reduce_sum(tf.math.squre(tf.nn.leaky_relu(output_t, alpha=args['leaky_relu_constant'])), axis=1)
    elif args['activation'] == 2:
        if not args['square_activation']:
            output_t = tf.reduce_sum(tf.nn.elu(output_t), axis=1)
        else:
            output_t = tf.reduce_sum(tf.math.square(tf.nn.elu(output_t)), axis=1)
    elif args['activation'] == 3:
        if not args['square_activation']:
            output_t = tf.reduce_sum(tf.math.tanh(output_t), axis=1)
        else:
            output_t = tf.reduce_sum(tf.math.square(tf.math.tanh(output_t)), axis=1)
    output_t = tf.reshape(output_t, [-1]) # shape = batch_size
    
    return output_t
    
    
# output of the model for a whole sequential signal
def get_output_LPLC2_T(args, weights_e, weights_i, intercept_e, intercept_i, UV_flow, tau_1):
    """
    This function takes the flow field as inputs, and outputs the sum of the responses 
    from all the M model units along the whole trajectory with total time steps T. 
    
    Args:
    args: a dictionary that contains problem parameters
    weights_e: tensor, weights for excitatory units. 
               shape = K*K by 1 (with rotation symmetry).
               shape = K*K by 4 (without rotation symmetry), where
               for axis=1, 0: right motion, 1: left motion, 2: up motion, 3: down motion
    weights_i: tensor, weights for inhibitory units. 
               shape = K*K by 1 (with rotation symmetry).
               shape = K*K by 4 (without rotation symmetry), where
               for axis=1, 0: right motion, 1: left motion, 2: up motion, 3: down motion
    intercept_e: tensor, intercept in the activation function for the LPLC2 unit, shape = 0
    intercept_i: tensor, intercept in the activation function for inhibitory units, shape = 0
    UV_flow: tensor, flow field, shape = steps by shape = batch_size by M by K*K by 4
    tau_1: log of the timescale of the filter
    
    Returns:
    output_T: tensor, output of the model, shape = steps by batch_size
    """
    output_T = tf.map_fn(lambda x:\
        get_output_LPLC2(args, weights_e, weights_i, intercept_e, intercept_i, x), \
            UV_flow, dtype=tf.float32) # shape = steps by batch_size
    if args['temporal_filter']:
        output_T = get_LPLC2_response_filtered(args, tau_1, output_T) # shape = steps by batch_size
    
    return output_T
    
    
#  Loss and predicted probabilities
def get_loss_and_prob(args, weights_e, weights_i, intercept_e, intercept_i, UV_flow, \
                   labels, tau_1, a, b, step_weights, regularizer_we, regularizer_wi, regularizer_a):
    """
    This function calculates the cross entropy loss and predicted probability.
    
    Args:
    args: a dictionary that contains problem parameters
    weights_e: tensor, weights for excitatory units. 
               shape = K*K by 1 (with rotation symmetry).
               shape = K*K by 4 (without rotation symmetry), where
               for axis=1, 0: right motion, 1: left motion, 2: up motion, 3: down motion
    weights_i: tensor, weights for inhibitory units. 
               shape = K*K by 1 (with rotation symmetry).
               shape = K*K by 4 (without rotation symmetry), where
               for axis=1, 0: right motion, 1: left motion, 2: up motion, 3: down motion
    intercept_e: tensor, intercept in the activation function for the LPLC2 unit, shape = 0
    intercept_i: tensor, intercept in the activation function for inhibitory units, shape = 0
    UV_flow: tensor, flow field, shape = steps by shape = batch_size by M by K*K by 4
    labels: labels of the stimuli (hit, 1; others, 0), shape = steps by batch_size
    tau_1: log of the timescale of the filter
    a: linear coefficient of the logistic classifier
    b: intercept of the logistic classifier
    step_weights: weights on steps, indicating how important each step or time point is, shape=steps
    regularizer_we, regularizer_wi, regularizer_a: regularizers on weights_e, weights_i, and a.
    
    Returns
    loss: cross entropy loss averaged cross the current batch, shape = 0
    probabilities: predicted probabilities of hit, shape = batch_size
    """
    output_T = get_output_LPLC2_T(args, weights_e, weights_i, \
        intercept_e, intercept_i, UV_flow, tau_1) # shape = steps by batch_size
    logits = tf.abs(a) * output_T + b # shape = steps by batch_size
    
    # loss
    if args['max_response']:
        loss = cross_entropy_loss(tf.reduce_max(logits), labels[0], step_weights[0]) # shape = 0
    else:
        loss = cross_entropy_loss(logits, labels, step_weights) # shape = 0
    
    # Apply regularization
    regularization_penalty_we = tf.contrib.layers.apply_regularization(regularizer_we, [weights_e])
    regularization_penalty_wi = tf.contrib.layers.apply_regularization(regularizer_wi, [weights_i])
    regularization_penalty_a = tf.contrib.layers.apply_regularization(regularizer_a, [a])
    loss = loss + regularization_penalty_we + regularization_penalty_wi + regularization_penalty_a # shape = 0
    
    # probability of hit
    probabilities = tf.nn.sigmoid(logits) # shape = steps by batch_size
    probabilities = tf.math.multiply(probabilities, step_weights) # shape = steps by batch_size
    probabilities = tf.reduce_sum(probabilities, axis=0) # average over trajectory, shape = batch_size
    
    return loss, probabilities


# Train and test the model
def train_and_test_model(args, train_flow_files, train_labels, test_flow_files, test_labels, \
                         train_flow_snapshots, train_labels_snapshots):
    """
    Args:
    args: a dictionary that contains problem parameters
    train_flow_files: flow fields for training
    train_labels: labels (probability of hit, either 0 or 1) for training
    test_flow_files: flow fields for testing
    test_labels: labels (probability of hit, either 0 or 1) for testing
    train_flow_snapshots: snapshot training samples, flow field
    train_labels_snapshots: corresponding labels for snapshot samples
    """
    start = time.time()
    
    # make sure there are the same number of snapshots and trajectories
    assert(len(train_flow_files) == len(train_flow_snapshots))
    
    # allocate certain amount of training samples for validation
    validation_portion = 0.2
    valid_flow_snapshots = train_flow_snapshots[int((1-validation_portion)*len(train_flow_snapshots)):]
    train_flow_snapshots = train_flow_snapshots[:int((1-validation_portion)*len(train_flow_snapshots))]
    valid_labels_snapshots = train_labels_snapshots[int((1-validation_portion)*len(train_labels_snapshots)):]
    train_labels_snapshots = train_labels_snapshots[:int((1-validation_portion)*len(train_labels_snapshots))]
    
    N_train = len(train_flow_snapshots) # number of train batches
    N_valid = len(valid_flow_snapshots) # number of validation batches
    N_test = len(test_flow_files) # number of test batches
    
    M = args['M'] # number of models units
    K = args['K'] # size of one of the dimensions of the square receptive field matrix
    lr = args['lr'] # learning rate
    N_epochs = args['N_epochs'] # number of training epochs
    log_file = args['save_path'] + 'log.txt'
        
    # output variables
    train_loss_output = []
    valid_loss_output = []
    a_output = []
    b_output = []
    intercept_e_output = []
    intercept_i_output = []
    y_true_train = []
    y_pred_train = []
    y_true_valid = []
    y_pred_valid = []
    y_true_test = []
    y_pred_test = []
    
    
    with open(log_file, 'w') as f:
        f.write('Model setup:\n')
        f.write('----------------------------------------\n')
        f.write('set_number: {}\n'.format(args['set_number']))
        f.write('data_path: {}\n'.format(args['data_path']))
        f.write('rotational_fraction: {}\n'.format(args['rotational_fraction']))
        f.write('Number of training examples: {}\n'.format(len(train_flow_files)))
        f.write('Number of testing examples: {}\n'.format(len(test_flow_files)))
        f.write('M: {}\n'.format(args['M']))
        f.write('n: {}\n'.format(args['n']))
        f.write('dt: {}\n'.format(args['dt']))
        f.write('K: {}\n'.format(args['K']))
        f.write('N_epochs: {}\n'.format(args['N_epochs']))
        f.write('lr: {}\n'.format(args['lr']))
        f.write('activation: {}\n'.format(args['activation']))
        f.write('square_activation: {}\n'.format(args['square_activation']))
        f.write('leaky_relu_constant: {}\n'.format(args['leaky_relu_constant']))
        f.write('fine_tune_weights: {}\n'.format(args['fine_tune_weights'])) 
        f.write('fine_tune_intercepts_and_b: {}\n'.format(args['fine_tune_intercepts_and_b'])) 
        f.write('fine_tune_model_dir: {}\n'.format(args['fine_tune_model_dir']))  
        f.write('save_intermediate_weights: {}\n'.format(args['save_intermediate_weights']))
        f.write('save_steps: {}\n'.format(args['save_steps']))
        f.write('use_step_weight: {}\n'.format(args['use_step_weight']))
        f.write('l1 regularization on excitatory weights: {}\n'.format(args['l1_regu_we']))
        f.write('l2 regularization on excitatory weights: {}\n'.format(args['l2_regu_we']))
        f.write('l1 regularization on inhibitory weights: {}\n'.format(args['l1_regu_wi']))
        f.write('l2 regularization on inhibitory weights: {}\n'.format(args['l2_regu_wi']))
        f.write('l1 regularization on a: {}\n'.format(args['l1_regu_a']))
        f.write('l2 regularization on a: {}\n'.format(args['l2_regu_a']))
        f.write('restrict_nonneg_weight: {}\n'.format(args['restrict_nonneg_weight']))
        f.write('restrict_nonpos_intercept: {}\n'.format(args['restrict_nonpos_intercept']))
        f.write('rotation_symmetry: {}\n'.format(args['rotation_symmetry']))
        f.write('flip_symmetry: {}\n'.format(args['flip_symmetry']))
        f.write('train_a: {}\n'.format(args['train_a']))
        f.write('max_response: {}\n'.format(args['max_response']))
        f.write('temporal_filter: {}\n'.format(args['temporal_filter']))
        f.write('learn_tau: {}\n'.format(args['learn_tau']))
        f.write('tau in standard scale: '+str(np.around(np.exp(args['tau']), 3))+'\n\n')
    
    # inputs
    UV_flow = tf.compat.v1.placeholder(tf.float32, [None, None, M, K*K, 4], name='UV_flow')
    step_weights = tf.compat.v1.placeholder(tf.float32, [None, None], name='step_weights')
    labels = tf.compat.v1.placeholder(tf.float32, [None, None], name='labels')

    # variables
    with tf.compat.v1.variable_scope('with_inhibitory'):
        # define some initializers
        scale_initializer = tf.compat.v1.keras.initializers.VarianceScaling()
        positive_initializer_e = tf.compat.v1.keras.initializers.TruncatedNormal(mean=0.2, stddev=0.1)
        positive_initializer_i = tf.compat.v1.keras.initializers.TruncatedNormal(mean=0.2, stddev=0.1)
        symm_initializer_b = tf.compat.v1.keras.initializers.TruncatedNormal(mean=0., stddev=1.)
        negative_initializer_b = tf.compat.v1.keras.initializers.TruncatedNormal(mean=-2., stddev=1.)
        zero_initializer = tf.constant_initializer(0)
        positive_initializer = tf.constant_initializer(1.)
        negative_initializer = tf.constant_initializer(-1.)
        # flow field weights
        if args['rotation_symmetry']:
            if not args['flip_symmetry']:
                weights_e = tf.Variable(initializer([K*K, 1]), name='weights_e')
                weights_i = tf.Variable(initializer([K*K, 1]), name='weights_i')
            else:
                if args['fine_tune_weights']:
                    saved_weights_e = np.float32(np.load(args['fine_tune_model_dir']+'trained_weights_e.npy'))
                    weights_e_raw = tf.Variable(initial_value=saved_weights_e[:(K+1)//2*K, 0], name='weights_e')
                    weights_e = expand_weight(weights_e_raw, (K+1)//2, K, K%2==0)
                    saved_weights_i = np.float32(np.load(args['fine_tune_model_dir']+'trained_weights_i.npy'))
                    weights_i_raw = tf.Variable(initial_value=saved_weights_i[:(K+1)//2*K, 0], name='weights_i')
                    weights_i = expand_weight(weights_i_raw, (K+1)//2, K, K%2==0)
                else:
                    weights_e_raw = tf.Variable(positive_initializer_e([(K+1)//2*K, 1]), name='weights_e')
                    weights_e = expand_weight(weights_e_raw, (K+1)//2, K, K%2==0)
                    weights_i_raw = tf.Variable(positive_initializer_i([(K+1)//2*K, 1]), name='weights_i')
                    weights_i = expand_weight(weights_i_raw, (K+1)//2, K, K%2==0)
        else:
            weights_e = tf.Variable(positive_initializer_e([K*K, 4]), name='weights_e')
            weights_i = tf.Variable(positive_initializer_i([K*K, 4]), name='weights_i')
        # slope or scale on the summed response
        if args['train_a']:
            a = tf.Variable(positive_initializer([1]), name='a')
        else:
            a = 1.
        # intercepts and b
        if args['fine_tune_intercepts_and_b']:
            saved_intercept_e = np.float32(np.load(args['fine_tune_model_dir']+'trained_intercept_e.npy'))
            intercept_e = tf.Variable(initial_value=saved_intercept_e, name='intercept_e')
            saved_intercept_i = np.float32(np.load(args['fine_tune_model_dir']+'trained_intercept_i.npy'))
            intercept_i = tf.Variable(initial_value=saved_intercept_i, name='intercept_i')
            saved_b = np.float32(np.load(args['fine_tune_model_dir']+'trained_b.npy'))
            b = tf.Variable(initial_value=saved_b, name='b')
        else:
            intercept_e = tf.Variable(scale_initializer([1]), name='intercept_e')
            intercept_i = tf.Variable(scale_initializer([1]), name='intercept_i')
            b = tf.Variable(negative_initializer_b([1]), name='b') # Put a negative prior on b
        # log of the timescale in the temporal filters
        if args['learn_tau']:
            tau_1 = tf.Variable(negative_initializer([-1]), name='tau_1')
        else:
            tau_1 = args['tau']
    
    # save initial weights
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        np.save(args['save_path']+'initial_weights_e', weights_e.eval())
        np.save(args['save_path']+'initial_weights_i', weights_i.eval())
            
    # Regularization
    l1_l2_regu_we = tf.contrib.layers.l1_l2_regularizer(scale_l1=args["l1_regu_we"], scale_l2=args["l2_regu_we"], scope=None)
    l1_l2_regu_wi = tf.contrib.layers.l1_l2_regularizer(scale_l1=args["l1_regu_wi"], scale_l2=args["l2_regu_wi"], scope=None)
    l1_l2_regu_a = tf.contrib.layers.l1_l2_regularizer(scale_l1=args["l1_regu_a"], scale_l2=args["l2_regu_a"], scope=None)
    
    # loss and probability, shape = 0, batch_size
    loss, probabilities = \
    get_loss_and_prob(args, weights_e, weights_i, intercept_e, intercept_i, UV_flow, \
        labels, tau_1, a, b, step_weights, l1_l2_regu_we, l1_l2_regu_wi, l1_l2_regu_a)

    # Optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(lr, beta1=0.9)
    opt = optimizer.minimize(loss)
    opt_with_clip = opt
    if args['restrict_nonneg_weight']:
        with tf.control_dependencies([opt]): # constrained optimization
            if not args['flip_symmetry']:
                clip_weights_e = weights_e.assign(tf.maximum(0., weights_e))
                clip_weights_i = weights_i.assign(tf.maximum(0., weights_i))
            else:
                clip_weights_e = weights_e_raw.assign(tf.maximum(0., weights_e_raw))
                clip_weights_i = weights_i_raw.assign(tf.maximum(0., weights_i_raw))
            opt_with_clip = tf.group(clip_weights_e, clip_weights_i)
    if args['restrict_nonpos_intercept']:
        with tf.control_dependencies([opt_with_clip]): 
            clip_intercept_e = intercept_e.assign(tf.minimum(0., intercept_e))
            clip_intercept_i = intercept_i.assign(tf.minimum(0., intercept_i))
            opt_with_clip = tf.group(clip_intercept_e, clip_intercept_i)
    
    # Train, validate
    counter_1 = 0 # counter of epoch
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for epoch in range(N_epochs):
            #######################################
            ############## training ###############
            #######################################
            counter_1 = counter_1 + 1
            train_loss = 0 # this is to accumulate train loss for each epoch
            train_counter = 0 # counter of training batches, should be equal to N_train at the end of each epoch.
            for sample_i in range(N_train):
                UV_flow_i = train_flow_snapshots[sample_i].copy()
                labels_i = train_labels_snapshots[sample_i].copy()
                step_weights_i = np.ones_like(labels_i) # uniform step weights
                # shape = _, 0, batch_size
                _, loss_i, probabilities_i = \
                sess.run([opt_with_clip, loss, probabilities], 
                    {UV_flow:UV_flow_i, labels:labels_i, step_weights:step_weights_i})
                
                # append current values of loss and some trained variables for output
                train_loss_output.append(loss_i)
                if args['train_a']:
                    a_output.append(a.eval())
                else:
                    a_output.append(a)
                b_output.append(b.eval())
                intercept_e_output.append(intercept_e.eval())
                intercept_i_output.append(intercept_i.eval())
                
                # accumulate training status for the current epoch
                train_loss = train_loss + loss_i 
                train_counter = train_counter + 1
                
                # output and save (later in the code) the true labels and predicted probabilities for the last epoch
                if epoch == N_epochs - 1:
                    for ii in range(labels_i.shape[1]):
                        y_true_train.append(labels_i[0][ii])
                        y_pred_train.append(probabilities_i[ii])
                        
                # save intermediate weights and variables if needed
                if args["save_intermediate_weights"] and counter_1 % args["save_steps"] == 1:
                    if not os.path.isdir(args['save_path'] + 'checkpoints'):
                        os.mkdir(args['save_path'] + 'checkpoints')
                    temp_dir = args['save_path'] + 'checkpoints/' + str(epoch) + '-' + str(sample_i+1) + '/'
                    os.mkdir(temp_dir)
                    if args['rotation_symmetry']:
                        #0: right motion, 1: left motion, 2: up motion, 3: down motion
                        rotate_num_list = [0, 2, 1, 3] 
                        weights_e_list = []
                        weights_i_list = []
                        for i in range(4):
                            weights_e_reshaped = tf.reshape(weights_e, [K, K, 1])
                            weights_e_rotated = tf.image.rot90(weights_e_reshaped, rotate_num_list[i])
                            weights_e_reshaped_back = tf.reshape(weights_e_rotated, [K*K, 1])
                            weights_e_list.append(weights_e_reshaped_back)
                            weights_i_reshaped = tf.reshape(weights_i, [K, K, 1])
                            weights_i_rotated = tf.image.rot90(weights_i_reshaped, rotate_num_list[i])
                            weights_i_reshaped_back = tf.reshape(weights_i_rotated, [K*K, 1])
                            weights_i_list.append(weights_i_reshaped_back)
                        weights_e_save = tf.concat(weights_e_list, axis=1)
                        weights_i_save = tf.concat(weights_i_list, axis=1)
                    weights_e_out = weights_e_save.eval()
                    weights_i_out = weights_i_save.eval()
                    intercept_e_out = intercept_e.eval()
                    intercept_i_out = intercept_i.eval()
                    b_out = b.eval()
                    if args['learn_tau']:
                        tau_1_out = tau_1.eval()
                    else:
                        tau_1_out = tau_1
                    if args['train_a']:
                        a_out = a.eval()
                    else:
                        a_out = a
                    np.save(temp_dir+'trained_weights_e', weights_e_out)
                    np.save(temp_dir+'trained_weights_i', weights_i_out)
                    np.save(temp_dir+'trained_intercept_e', intercept_e_out)
                    np.save(temp_dir+'trained_intercept_i', intercept_i_out)
                    np.save(temp_dir+'trained_tau_1', tau_1_out)
                    np.save(temp_dir+'trained_b', b_out)
                    np.save(temp_dir+'trained_a', a_out)
                    
            ############################################
            ############## validation ##################
            ############################################
            valid_loss = 0
            valid_counter = 0 # counter of validation batches, should be equal to N_valid at the end
            for sample_i in range(N_valid):
                UV_flow_i = valid_flow_snapshots[sample_i].copy()
                labels_i = valid_labels_snapshots[sample_i].copy()
                step_weights_i = np.ones_like(labels_i)
                # shape = _, 0, batch_size
                _, loss_i, probabilities_i = \
                sess.run([opt_with_clip, loss, probabilities], 
                    {UV_flow:UV_flow_i, labels:labels_i, step_weights:step_weights_i})
                
                # append current values of validation loss for output
                valid_loss_output.append(loss_i)
                
                # accumulate validation status for the current epoch
                valid_loss = valid_loss + loss_i 
                valid_counter = valid_counter + 1
                
                # output and save (later in the code) the true labels and predicted probabilities for the last epoch
                if epoch == N_epochs - 1:
                    for ii in range(labels_i.shape[1]):
                        y_true_valid.append(labels_i[0][ii])
                        y_pred_valid.append(probabilities_i[ii])
            
            # Print and record intermediate training and validation results
            assert(train_counter == N_train)
            assert(valid_counter == N_valid)
            print('epoch {}: train loss (epoch average) is {:.4g}, and validation loss (epoch average) is {:.4g}'\
                .format(epoch, train_loss/train_counter, valid_loss/valid_counter))
            with open(log_file, 'a+') as f:
                f.write('epoch {}: train loss (epoch average) is {:.4g}, and validation loss (epoch average) is {:.4g}\n'\
                    .format(epoch, train_loss/train_counter, valid_loss/valid_counter))
        
        # save the training and validation results (for all the epochs)
        np.save(args['save_path'] + 'train_loss_output', train_loss_output)
        np.save(args['save_path'] + 'valid_loss_output', valid_loss_output)
        np.save(args['save_path'] + 'a_output', a_output)
        np.save(args['save_path'] + 'b_output', b_output)
        np.save(args['save_path'] + 'intercept_e_output', intercept_e_output)
        np.save(args['save_path'] + 'intercept_i_output', intercept_i_output)
        
        # save the final training and validation results (for the last epoch)
        if args['rotation_symmetry']:
            #0: right motion, 1: left motion, 2: up motion, 3: down motion
            rotate_num_list = [0, 2, 1, 3] 
            weights_e_list = []
            weights_i_list = []
            for i in range(4):
                weights_e_reshaped = tf.reshape(weights_e, [K, K, 1])
                weights_e_rotated = tf.image.rot90(weights_e_reshaped, rotate_num_list[i])
                weights_e_reshaped_back = tf.reshape(weights_e_rotated, [K*K, 1])
                weights_e_list.append(weights_e_reshaped_back)
                weights_i_reshaped = tf.reshape(weights_i, [K, K, 1])
                weights_i_rotated = tf.image.rot90(weights_i_reshaped, rotate_num_list[i])
                weights_i_reshaped_back = tf.reshape(weights_i_rotated, [K*K, 1])
                weights_i_list.append(weights_i_reshaped_back)
            weights_e = tf.concat(weights_e_list, axis=1)
            weights_i = tf.concat(weights_i_list, axis=1)
        weights_e_out = weights_e.eval()
        weights_i_out = weights_i.eval()
        intercept_e_out = intercept_e.eval()
        intercept_i_out = intercept_i.eval()
        b_out = b.eval()
        if args['learn_tau']:
            tau_1_out = tau_1.eval()
        else:
            tau_1_out = tau_1
        if args['train_a']:
            a_out = a.eval()
        else:
            a_out = a   
        np.save(args['save_path'] + 'trained_weights_e', weights_e_out)
        np.save(args['save_path'] + 'trained_weights_i', weights_i_out)
        np.save(args['save_path'] + 'trained_intercept_e', intercept_e_out)
        np.save(args['save_path'] + 'trained_intercept_i', intercept_i_out)
        np.save(args['save_path'] + 'trained_tau_1', tau_1_out)
        np.save(args['save_path'] + 'trained_b', b_out)
        np.save(args['save_path'] + 'trained_a', a_out)
        np.save(args['save_path'] + 'y_true_train', y_true_train)
        np.save(args['save_path'] + 'y_pred_train', y_pred_train)
        np.save(args['save_path'] + 'y_true_valid', y_true_valid)
        np.save(args['save_path'] + 'y_pred_valid', y_pred_valid)
        
                
        ######################################################
        ############## testing on the whole trajectories #####
        ######################################################
        test_loss = 0
        test_counter = 0 # counter for total test batches, should be equal to N_test
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for sample_i, UV_flow_file_i, labels_i, step_weights_i in \
            zip(range(N_test), test_flow_files, test_labels, args['test_distances']):
            # flatten the labels for the current test batch, # from args['S'] by args['NNs'] to args['S'] * args['NNs']
            labels_i = np.array(labels_i).flatten()
            # calculate the loss and predicted probabilities for the current batch
            UV_flow_i, step_weights_i, labels_i =\
                hpfn.load_compressed_dataset(args, UV_flow_file_i, labels_i)
            loss_i, probabilities_i = sess.run([loss, probabilities],\
                {UV_flow:UV_flow_i, labels:labels_i, step_weights: step_weights_i})
            # accumulate test loss for printing and log
            test_loss = test_loss + loss_i
            test_counter = test_counter + 1
            # accumulate true labels and predicted probabilities for test data
            for ii in range(labels_i.shape[1]):
                y_true_test.append(labels_i[0][ii])
                y_pred_test.append(probabilities_i[ii])
                if labels_i[0][ii] == 0:
                    if probabilities_i[ii] < 0.5:
                        true_negative = true_negative + 1
                    elif probabilities_i[ii] >= 0.5:
                        false_positive = false_positive + 1
                elif labels_i[0][ii] == 1:
                    if probabilities_i[ii] < 0.5:
                        false_negative = false_negative + 1
                    elif probabilities_i[ii] >= 0.5:
                        true_positive = true_positive + 1
        # calculate the auc for roc and pr curves
        auc_roc = sm.roc_auc_score(y_true_test, y_pred_test)
        auc_pr = sm.average_precision_score(y_true_test, y_pred_test)
        
        # Print and record testing results
        assert(test_counter == N_test)
        print('Test loss (epoch average) is {:.4g}, test AUCROC is {:.3g}, test PRAUC is {:.3g}'\
            .format(test_loss/test_counter, auc_roc, auc_pr))
        print('True positive is {}, False positive is {} \n \
            False negative is {}, True negative is {}'\
                .format(true_positive, false_positive, false_negative, true_negative))
        with open(log_file, 'a+') as f:
            f.write('Test loss (epoch average) is {:.4g}, test AUCROC is {:.3g}, test PRAUC is {:.3g}\n'\
                .format(test_loss/test_counter, auc_roc, auc_pr))
            f.write('True positive is {}, False positive is {} \n \
                False negative is {}, True negative is {}\n'\
                    .format(true_positive, false_positive, false_negative, true_negative))
        # save the testing results (after the last epoch of training)
        np.save(args['save_path'] + 'auc_roc', auc_roc)
        np.save(args['save_path'] + 'auc_pr', auc_pr)
        np.save(args['save_path'] + 'y_true_test', y_true_test)
        np.save(args['save_path'] + 'y_pred_test', y_pred_test)

    # Print and record the trained parameters except the weights
    print('Trained tau_1 is ', np.around(tau_1_out, 3))
    print('Trained b is ', np.around(b_out, 3))
    print('Trained a is ', np.around(a_out, 3))
    print('Trained intercept_e is ', np.around(intercept_e_out, 3))
    print('Trained intercept_i is ', np.around(intercept_i_out, 3))
    with open(log_file, 'a+') as f:
        f.write('Trained tau_1 is ' + str(np.around(tau_1_out, 3)) + '\n')
        f.write('Trained b is ' + str(np.around(b_out, 3 )) + '\n')
        f.write('Trained a is ' + str(np.around(a_out, 3)) + '\n')
        f.write('Trained intercept_e is '  + str(np.around(intercept_e_out, 3)) + '\n')
        f.write('Trained intercept_i is '  + str(np.around(intercept_i_out, 3)) + '\n')
    
    end = time.time()
    print('Training took {:.0f} second'.format(end - start))
    with open(log_file, 'a+') as f:
        f.write('Training took {:.0f} second\n'.format(end - start))
    
    return weights_e_out, weights_i_out, \
           intercept_e_out, intercept_i_out, tau_1_out, b_out, a_out


############################################
########## helper functions ################
############################################
def expand_weight(weights, num_row, num_column, is_even):
    """
    Get the full mirror symmetric weight from half of the weight.
    """
    weights_reshaped = tf.reshape(weights, [num_row, num_column, 1])
    if is_even:
        assert(num_row*2 == num_column)
        weights_flipped = tf.concat([weights_reshaped, tf.reverse(weights_reshaped, axis=[0])], axis=0)
    else:
        assert((num_row*2-1) == num_column)
        weights_flipped = tf.concat([weights_reshaped, tf.reverse(weights_reshaped[:-1], axis=[0])], axis=0)
    weights_reshaped_back = tf.reshape(weights_flipped, [num_column**2, 1])
    
    return weights_reshaped_back


# general temporal filter
def general_temp_filter(args, tau_1, T):
    """
    Args:
    args: a dictionary that contains problem parameters
    tau_1: log of the timescale of the filter
    T: total length of the filter
    
    Returns:
    G_n: general temporal filter, len(G_n) = T
    """
    n = args['n']
    dt = args['dt']
    
    tau_1 = tf.exp(tau_1)
    T = tf.dtypes.cast(T, tf.float32)
    ts = dt * tf.range(T)
    if n == 1.:
        #G_n = (1./tau_1) * tf.exp(-ts/tau_1)
        G_n = tau_1 * tf.exp(-ts*tau_1)
    else:
        G_n = (1. / tf.exp(tf.lgamma(n-1.))) * \
        (ts**(n-1.) / (tau_1**n)) * tf.exp(-ts/tau_1)
    G_n = G_n / tf.reduce_sum(G_n)
    
    return G_n


# get filtered signal
def get_filtered_signal(args, tau_1, signal_seq):
    """
    Args:
    args: a dictionary that contains problem parameters
    tau_1: log of the timescale of the filter
    signal_seq: tensor, signal sequence to be filtered
    
    Returns:
    filtered_sig: filtered signal, single data point
    """

    if n == 0:
        filtered_sig = signal_seq[-1]
    else:
        T = tf.shape(signal_seq)[-1]
        G_n = general_temp_filter(args, tau_1, T)
    filtered_sig = tf.tensordot(signal_seq, tf.reverse(G_n, [0]), axes=1)
    
    return filtered_sig
   
    
# get the filtered response of a LPLC2 unit
def get_LPLC2_response_filtered(args, tau_1, input_T):
    """
    Args:
    args: a dictionary that contains problem parameters
    tau_1: log of the timescale of the filter
    input_T: tensor, input to a LPLC2 unit, len(input_T_tf) = steps
    
    Returns:
    filtered_res_tf: tensor, filtered response, len(filtered_res_tf) = steps
    """
    n = args['n']

    if n == 0:
        filtered_res = input_T
    else:
        steps = tf.shape(input_T)
        G_n = general_temp_filter(args, tau_1, steps[0])
        filter_matrix = tf.tile(G_n, [steps[0]])
        filter_matrix = tf.reshape(filter_matrix, [steps[0], steps[0]])
        filter_matrix = tf.linalg.LinearOperatorLowerTriangular(filter_matrix)
        seq_lens = tf.range(1, steps[0]+1)
        filter_matrix = tf.reverse_sequence(filter_matrix.to_dense(), seq_lens, 
            seq_axis=1, batch_axis=0)
        filter_matrix = filter_matrix / tf.reduce_sum(filter_matrix, 
            axis=-1, keepdims=True)
        filtered_res = tf.tensordot(filter_matrix, input_T, axes=1)
    
    return filtered_res


# cross entropy loss function
def cross_entropy_loss(logits, labels, step_weights):
    """
    This function calculates the cross entropy loss averaged aross the current batch.
    
    Args:
    logits: predicted logits, shape=steps by batch_size
    labels: labels of the stimuli (hit, 1; others, 0), shape = steps by batch_size
    step_weights: weights on steps, indicating how important each step or time point is, shape=steps by batch_size
    
    Returns:
    celoss: cross entropy loss (averaged over batch), shape=0
    """
    celoss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels) # shape = steps by batch_size
    step_weights = tf.convert_to_tensor(step_weights, dtype=tf.float32) 
    celoss = tf.math.multiply(celoss, step_weights) # shape = steps by batch_size
    celoss = tf.reduce_sum(celoss, axis=0) # average over time steps, shape = batch_size
    celoss = tf.reduce_mean(celoss) # average over batch, shape = 0
    
    return celoss
    
    