#!/usr/bin/env python


'''
This module contains functions that define various LPLC2 models.
'''


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


##########################################
######## Inhibitory (pooling) ##########
##########################################


# Input to a LPLC2 neuron at a specific time point according to 
# model inhibitory
def get_input_LPLC2_with_inhibition(args, weights_e, weights_i, weights_intensity, intercept_e, 
    intercept_i, signal_t):
    ''''
    Args:
    args: a dictionary that contains problem parameters
    weights_e: tensor, weights for excitatory neurons, K*K by 4, for axis=1, 
                0: right motion, 1: left motion, 2: up motion, 3: down motion
    weights_i: tensor, weights for inhibitory neurons, K*K by 4, for axis=1, 
                0: right motion, 1: left motion, 2: up motion, 3: down motion
    intercept_e: tensor, intercept in the activation function for the LPLC2 neuron
    intercept_i: tensor, intercept in the activation function for inhibitory neurons
    signal_t: (UV_flow_t,frame_intensity_t), where
              UV_flow_t: tensor, flow field at time point t, batch_size by Q by K*K by 4
              frame_intensity_t: tensor, intensity at time point t, batch_size by Q by K*K
    
    Returns:
    input_t: input to a LPLC2 neuron at a specific time point t
    '''
    K = args['K']
    rotation_symmetry = args['rotation_symmetry']  
    UV_flow_t = signal_t[0]
    frame_intensity_t = signal_t[1]
    intercept_e = tf.dtypes.cast(intercept_e,tf.float32)
    intercept_i = tf.dtypes.cast(intercept_i,tf.float32)
    
    # add the intercept for the activation function of the lplc2 unit
    input_t = intercept_e
    
    # intensity dependence
    if args['use_intensity']:
        input_intensity_t = tf.tensordot(frame_intensity_t, 
            weights_intensity, axes=[[2],[0]])
        input_t = input_t+input_intensity_t
    
    # rotation is counter-clockwise
    # 0: right motion, 1: left motion, 2: up motion, 3: down motion
    rotate_num_list = [0, 2, 1, 3] 
    
    # excitatory input
    for i in range(4):
        if rotation_symmetry:
            weights_e_reshaped = tf.reshape(weights_e, [K,K,1])
            weights_e_rotated = tf.image.rot90(weights_e_reshaped,
                rotate_num_list[i])
            weights_e_reshaped_back = tf.reshape(weights_e_rotated, [K*K,1])
            input_t = input_t + tf.tensordot(UV_flow_t[:,:,:,i], 
                weights_e_reshaped_back,axes=[[2],[0]])
        else:
            input_t = input_t + tf.tensordot(UV_flow_t[:,:,:,i], weights_e[:,i], 
                axes=[[2],[0]])
        
    # inhibitory input
    for i in range(4):
        if rotation_symmetry:
            weights_i_reshaped = tf.reshape(weights_i, [K,K,1])
            weights_i_rotated = tf.image.rot90(weights_i_reshaped,
                rotate_num_list[i])
            weights_i_reshaped_back = tf.reshape(weights_i_rotated, [K*K,1])
            if args['activation'] == 0:
                input_t = input_t - tf.nn.relu(tf.tensordot(UV_flow_t[:,:,:,i],
                    weights_i_reshaped_back, axes=[[2],[0]]) + intercept_i)
            elif args['activation'] == 1:
                input_t = input_t - tf.nn.leaky_relu(tf.tensordot(UV_flow_t[:,:,:,i],
                    weights_i_reshaped_back, axes=[[2],[0]]) + intercept_i, 
                alpha=args['leaky_relu_constant'])
            elif args['activation'] == 2:
                input_t = input_t - tf.nn.elu(tf.tensordot(UV_flow_t[:,:,:,i],
                    weights_i_reshaped_back, axes=[[2],[0]]) + intercept_i)
            elif args['activation'] == 3:
                input_t = input_t - tf.math.tanh(tf.tensordot(UV_flow_t[:,:,:,i],
                    weights_i_reshaped_back, axes=[[2],[0]]) + intercept_i)
        else:
            if args['activation'] == 0:
                input_t = input_t - tf.nn.relu(tf.tensordot(UV_flow_t[:,:,:,i],
                    weights_i[:,i], axes=[[2],[0]]) + intercept_i)
            elif args['activation'] == 1:
                input_t = input_t - tf.nn.leaky_relu(tf.tensordot(UV_flow_t[:,:,:,i],
                    weights_i[:,i], axes=[[2],[0]]) + intercept_i, 
                alpha=args['leaky_relu_constant'])
            elif args['activation'] == 2:
                input_t = input_t - tf.nn.elu(tf.tensordot(UV_flow_t[:,:,:,i],
                    weights_i[:,i], axes=[[2],[0]]) + intercept_i)
            elif args['activation'] == 3:
                input_t = input_t - tf.math.tanh(tf.tensordot(UV_flow_t[:,:,:,i],
                    weights_i[:,i], axes=[[2],[0]]) + intercept_i)
    
    # sum pooling across multiple neurons
    if args['activation'] == 0:
        if not args['square_activation']:
            input_t = tf.reduce_sum(tf.nn.relu(input_t),axis=1)
        else:
            input_t = tf.reduce_sum(tf.math.square(tf.nn.relu(input_t)),axis=1)
    elif args['activation'] == 1:
        if not args['square_activation']:
            input_t = tf.reduce_sum(tf.nn.leaky_relu(input_t, 
                alpha=args['leaky_relu_constant']),axis=1)
        else:
            input_t = tf.reduce_sum(tf.math.squre(tf.nn.leaky_relu(input_t, 
                alpha=args['leaky_relu_constant'])),axis=1)
    elif args['activation'] == 2:
        if not args['square_activation']:
            input_t = tf.reduce_sum(tf.nn.elu(input_t),axis=1)
        else:
            input_t = tf.reduce_sum(tf.math.square(tf.nn.elu(input_t)),axis=1)
    elif args['activation'] == 3:
        if not args['square_activation']:
            input_t = tf.reduce_sum(tf.math.tanh(input_t),axis=1)
        else:
            input_t = tf.reduce_sum(tf.math.square(tf.math.tanh(input_t)),axis=1)
    input_t = tf.reshape(input_t,[-1])
    
    return input_t
    
    
# Input to a LPLC2 neuron for a whole sequential signal according to
# model inhibitory
def get_input_LPLC2_with_inhibition_T(args, weights_e, weights_i, intercept_e, 
    intercept_i, UV_flow):
    '''
    Args:
    args: a dictionary that contains problem parameters
    weights_e: tensor, weights for excitatory neurons, K*K by 4, for axis=1, 
                0: right motion, 1: left motion, 2: up motion, 3: down motion
    weights_i: tensor, weights for inhibitory neurons, K*K by 4, for axis=1, 
                0: right motion, 1: left motion, 2: up motion, 3: down motion
    intercept_e: tensor, intercept in the activation function for the LPLC2 neuron
    intercept_i: tensor, intercept in the activation function for inhibitory neurons
    UV_flow: tensor, flow field, steps by Q by K*K by 4
    
    Returns:
    input_T: tensor, input to a LPLC2 neuron, len(input_T_tf) = steps
    '''
    
    input_T = tf.map_fn(lambda x:\
                        get_input_LPLC2_with_inhibition2(args, weights_e,
                            weights_i, weights_intensity, intercept_e, intercept_i, x), (UV_flow,frame_intensity))
    
    return input_T
    
    return input_T

#  Loss and error of the classification for the inhibitory model
def loss_error_with_inhibitory(args, weights_e, weights_i, weights_intensity, intercept_e, intercept_i, UV_flow, frame_intensity, 
                             labels, tau_1, a, b, step_weights, regularizer_we, regularizer_wi, regularizer_a):
    '''
    Args:
    args: a dictionary that contains problem parameters
    weights_e: tensor, weights for excitatory neurons, K*K by 4, for axis=1, 
                0: right motion, 1: left motion, 2: up motion, 3: down motion
    weights_i: tensor, weights for inhibitory neurons, K*K by 4, for axis=1, 
                0: right motion, 1: left motion, 2: up motion, 3: down motion
    weights_intensity: tensor, weights for intensities, K*K
    intercept_e: tensor, intercept of excitatory part in the activation function
    intercept_i: tensor, intercept of inhibitory part in the activation function
    UV_flow: tensor, data, flow field, steps by Q by K*K by 4
    frame_intensity: data, coarse grained frame intensity, steps by Q by K*K
    labels: data (probability of hit), len(labels) = steps
    tau_1: log of the timescale of the filter
    a: linear coefficient of the logistic classifier
    b: intercept of the logistic classifier
    step_weights: weights on steps, indicating how important each step or time point is
    regularizer_we, regularizer_wi, regularizer_a: regularizers on weights_e, weights_i, and a.
    
    Returns
    loss: cross entropy loss function for one trajectory
    error_step: binary classification error for all steps in one trajectory
    error_trajectory: binary classification error for one trajectory
    filtered_res: filtered response for all steps in one trajectory 
    probabilities: predicted probabilities of hit for all steps in one trajectory 
    '''

    input_T = get_input_LPLC2_with_inhibition_T(args, weights_e, weights_i, 
        intercept_e, intercept_i, UV_flow)
    if args['use_intensity']:
        input_intensity_T = tf.expand_dims(get_input_LPLC2_intensity_T(args, 
            weights_intensity, frame_intensity), 1)
        input_T += input_intensity_T
    if args['temporal_filter']:
        filtered_res = get_LPLC2_response(args, tau_1, input_T)
    else:
        filtered_res = input_T
    logits = tf.abs(a) * filtered_res + b
    if args['max_response']:
        loss = cross_entropy_loss(tf.reduce_max(logits), labels[0], step_weights[0])
    else:
        loss = cross_entropy_loss(logits, labels, step_weights)
    probabilities = tf.nn.sigmoid(logits)
    probabilities = tf.math.multiply(probabilities,step_weights)
    probabilities = tf.reduce_sum(probabilities,axis=0)
    predictions = tf.round(probabilities)
#     error_step = 1 - tf.reduce_mean(tf.cast(tf.equal(predictions, labels), 
#         tf.float32))
    error_step = tf.constant(0,dtype=tf.float32)
    error_trajectory = 1 - tf.cast(tf.equal(predictions, 
        labels[0,:]), tf.float32)
    
    # Apply regularization
    regularization_penalty_we = tf.contrib.layers.apply_regularization(regularizer_we, [weights_e])
    regularization_penalty_wi = tf.contrib.layers.apply_regularization(regularizer_wi, [weights_i])
    regularization_penalty_a = tf.contrib.layers.apply_regularization(regularizer_a, [a])
    loss = loss + regularization_penalty_we + regularization_penalty_wi + regularization_penalty_a
    
    return loss, error_step, error_trajectory, filtered_res, probabilities


# Train and test the model inhibitory2 for classification
def train_and_test_model_C_i2(args, train_flow_files, train_labels, 
    test_flow_files, test_labels):
    '''
    Args:
    args: a dictionary that contains problem parameters
    train_flow_files: files for training UV flows
    train_labels: labels (probability of hit, either 0 or 1) for training
    test_flow_files: files for testing UV flows
    test_labels: labels (probability of hit, either 0 or 1) for testing
    
    Returns:
    
    '''
    start = time.time()

    Q = args['Q']
    K = args['K']
    rotation_symmetry = args['rotation_symmetry']
    flip_symmetry = args['flip_symmetry']
    restrict_nonneg_weight = args['restrict_nonneg_weight']
    lr = args['lr']
    N_epochs = args['N_epochs']
    log_file = args['save_path'] + 'log.txt'
    loss_output = []
    loss_output_test = []
    train_error_output = []
    b_output = []
    intercept_e_output = []
    intercept_i_output = []
    lplc2_cells = opsg.get_lplc2_cells_xy_angles(Q)
    
    with open(log_file, 'w') as f:
        f.write('Model setup:\n')
        f.write('----------------------------------------\n')
        f.write('Model is inhibitory2 \n')
        f.write('set_number: {}\n'.format(args['set_number']))
        f.write('data_path: {}\n'.format(args['data_path']))
        f.write('rotational_fraction: {}\n'.format(args['rotational_fraction']))
        f.write('Number of training examples: {}\n'.format(len(train_flow_files)))
        f.write('Number of testing examples: {}\n'.format(len(test_flow_files)))
        f.write('Q: {}\n'.format(args['Q']))
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
        f.write('frames to remove: {}\n'.format(args['frames_to_remove']))
        f.write('save_intermediate_weights: {}\n'.format(args['save_intermediate_weights']))
        f.write('save_steps: {}\n'.format(args['save_steps']))
        f.write('use_step_weight: {}\n'.format(args['use_step_weight']))
        f.write('l1 regularization on excitatory weights: {}\n'.format(args['l1_regu_we']))
        f.write('l2 regularization on excitatory weights: {}\n'.format(args['l2_regu_we']))
        f.write('l1 regularization on inhibitory weights: {}\n'.format(args['l1_regu_wi']))
        f.write('l2 regularization on inhibitory weights: {}\n'.format(args['l2_regu_wi']))
        f.write('l1 regularization on a: {}\n'.format(args['l1_regu_a']))
        f.write('l2 regularization on a: {}\n'.format(args['l2_regu_a']))
        f.write('restrict_nonneg_weight: {}\n'.format(
            args['restrict_nonneg_weight']))
        f.write('restrict_nonpos_intercept: {}\n'.format(
            args['restrict_nonpos_intercept']))
        f.write('rotation_symmetry: {}\n'.format(args['rotation_symmetry']))
        f.write('flip_symmetry: {}\n'.format(args['flip_symmetry']))
        f.write('use_intensity: {}\n'.format(args['use_intensity']))
        f.write('train_a: {}\n'.format(args['train_a']))
        f.write('max_response: {}\n'.format(args['max_response']))
        f.write('temporal_filter: {}\n'.format(args['temporal_filter']))
        f.write('learn_tau: {}\n'.format(args['learn_tau']))
        f.write('tau in standard scale: ' + 
            str(np.around(np.exp(-args['tau']), 3)) + '\n\n')
    
    N_train = len(train_flow_files)
    N_test = len(test_flow_files)
        
    # inputs
    UV_flow = tf.compat.v1.placeholder(tf.float32, [None,None,Q,K*K,4], 
        name = 'UV_flow')
    step_weights = tf.compat.v1.placeholder(tf.float32, [None,None], name='step_weights')
    labels = tf.compat.v1.placeholder(tf.float32, [None,None], name = 'labels')
    frame_intensity = None
    if args['use_intensity']:
        frame_intensity = tf.compat.v1.placeholder(tf.float32, 
            [None,None,Q,K*K], name='frame_intensity')

    # variables
    with tf.compat.v1.variable_scope('inhibitory_2'):
        initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=10.)
        positive_initializer = tf.compat.v1.keras.initializers.TruncatedNormal(mean=10.,stddev=9.)
        zero_initializer = tf.constant_initializer(0)
        negative_initializer = tf.constant_initializer(-1.)
        if rotation_symmetry:
            if not flip_symmetry:
                weights_e = tf.Variable(initializer([K*K,1]), name='weights_e')
                weights_i = tf.Variable(initializer([K*K,1]), name='weights_i')
            else:
                if args['fine_tune_weights']:
                    saved_weights_e = np.float32(np.load(args['fine_tune_model_dir'] + 'trained_weights_e.npy'))
                    weights_e_raw = tf.Variable(initial_value=saved_weights_e[:(K+1)//2*K,0], 
                        name='weights_e')
                    weights_e = expand_weight(weights_e_raw, (K+1)//2, K, K%2==0)

                    saved_weights_i = np.float32(np.load(args['fine_tune_model_dir'] + 'trained_weights_i.npy'))
                    weights_i_raw = tf.Variable(initial_value=saved_weights_i[:(K+1)//2*K,0], 
                        name='weights_i')
                    weights_i = expand_weight(weights_i_raw, (K+1)//2, K, K%2==0)
                else:
                    weights_e_raw = tf.Variable(positive_initializer([(K+1)//2*K,1]), 
                        name='weights_e')
                    weights_e = expand_weight(weights_e_raw, (K+1)//2, K, K%2==0)
                    weights_i_raw = tf.Variable(positive_initializer([(K+1)//2*K,1]), 
                        name='weights_i')
                    weights_i = expand_weight(weights_i_raw, (K+1)//2, K, K%2==0)
        else:
            weights_e = tf.Variable(initializer([K*K,4]), name='weights_e')
            weights_i = tf.Variable(initializer([K*K,4]), name='weights_i')
        weights_intensity = None
        if args['use_intensity']:
            weights_intensity = tf.Variable(initializer([K*K]), 
                name='weights_intensity')
        if args['fine_tune_intercepts_and_b']:
            saved_intercept_e = np.float32(np.load(args['fine_tune_model_dir'] + 'trained_intercept_e.npy'))
            intercept_e = tf.Variable(initial_value=saved_intercept_e, name='ntercept_e')

            saved_intercept_i = np.float32(np.load(args['fine_tune_model_dir'] + 'trained_intercept_i.npy'))
            intercept_i = tf.Variable(initial_value=saved_intercept_i, name='ntercept_i')
        else:
            intercept_e = tf.Variable(initializer([1]), name='intercept_e')
            intercept_i = tf.Variable(initializer([1]), name='intercept_i')
        if args['learn_tau']:
            tau_1 = tf.Variable(initializer([1]), name='tau_1')
        else:
            tau_1 = args['tau']
        #b = tf.Variable(initializer([1]), name='b')
        if args['fine_tune_intercepts_and_b']: 
            saved_b = np.float32(np.load(args['fine_tune_model_dir'] + 'trained_b.npy'))
            b = tf.Variable(initial_value=saved_b, name='b')
        else:
            b = tf.Variable(negative_initializer([1]), name='b')
        if args['train_a']:
            a = tf.Variable(initializer([1]), name='a')
        else:
            a = 1.

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        np.save(args['save_path'] + 'initial_weights_e', weights_e.eval())
        np.save(args['save_path'] + 'initial_weights_i', weights_i.eval())
            
    # Optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(lr,beta1=0.9)

    # Regularization
    l1_l2_regu_we = tf.contrib.layers.l1_l2_regularizer(scale_l1=args["l1_regu_we"],scale_l2=args["l2_regu_we"],scope=None)
    l1_l2_regu_wi = tf.contrib.layers.l1_l2_regularizer(scale_l1=args["l1_regu_wi"],scale_l2=args["l2_regu_wi"],scope=None)
    l1_l2_regu_a = tf.contrib.layers.l1_l2_regularizer(scale_l1=args["l1_regu_a"],scale_l2=args["l2_regu_a"],scope=None)

    # loss and error function
    loss, error_step, error_trajectory, _, probabilities = \
    loss_error_with_inhibitory(args, weights_e, weights_i, weights_intensity, intercept_e, intercept_i, UV_flow, frame_intensity, 
                             labels, tau_1, a, b, step_weights,l1_l2_regu_we,l1_l2_regu_wi,l1_l2_regu_a)

    # Optimization
    opt = optimizer.minimize(loss)
    opt_with_clip = opt
    if restrict_nonneg_weight:
        with tf.control_dependencies([opt]): # constrained optimization
            if not flip_symmetry:
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
    
    #config = tf.ConfigProto(inter_op_parallelism_threads=1)
    #with tf.compat.v1.Session(config=config) as sess:
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        # training
        for epoch in range(1, N_epochs+1):
            train_loss = 0
            train_error_step = 0
            train_error_trajectory = 0
            train_steps = 0
            train_samples = 0
            batch_train_loss = 0
            batch_train_error_step = 0
            batch_train_error_trajectory = 0
            batch_train_steps = 0
            batch_train_samples = 0
            for sample_i, UV_flow_file_i, labels_i, step_weights_i in zip(range(1, N_train+1), 
                train_flow_files, train_labels, args['train_distances']):
                labels_i = np.array(labels_i).flatten()
    #             steps_i = 1
    #             if not args['use_step_weight']:
    #                 step_weights_i = np.ones(steps_i)
                if args['use_intensity']:
                    if args['frames_to_remove'] == 0:
                        UV_flow_i,step_weights_i,labels_i =\
                            load_compressed_dataset(args, UV_flow_file_i, labels_i)
                        frame_intensity_i,step_weights_i,labels_i =\
                            load_compressed_dataset(args, args['train_frames'][sample_i-1],labels_i)
                    else:
                        UV_flow_i,step_weights_i,labels_i =\
                            load_compressed_dataset(args, UV_flow_file_i,labels_i)[:-args['frames_to_remove']]
                        frame_intensity_i,step_weights_i,labels_i =\
                            load_compressed_dataset(args, args['train_frames'][sample_i-1],labels_i)[:-args['frames_to_remove']]
                    _, loss_i, error_step_i, error_trajectory_i = \
                    sess.run([opt_with_clip, loss, error_step, error_trajectory],
                        {UV_flow:UV_flow_i, labels:labels_i, step_weights: step_weights_i})
                else:
                    if args['frames_to_remove'] == 0:
                        UV_flow_i,step_weights_i,labels_i =\
                            load_compressed_dataset(args, UV_flow_file_i,labels_i)
                    else:
                        UV_flow_i,step_weights_i,labels_i =\
                            load_compressed_dataset(args, UV_flow_file_i,labels_i)[:-args['frames_to_remove']]
                    _, loss_i, error_step_i, error_trajectory_i = \
                    sess.run([opt_with_clip, loss, error_step, error_trajectory],
                        {UV_flow:UV_flow_i, labels:labels_i, step_weights: step_weights_i})
                steps_i = 1
                loss_output.append(loss_i)
                b_output.append(b.eval())
                intercept_e_output.append(intercept_e.eval())
                intercept_i_output.append(intercept_i.eval())
                batch_train_loss += loss_i 
                batch_train_error_step += error_step_i * steps_i
                batch_train_steps += steps_i
                train_error_step += error_step_i * steps_i
                train_loss += loss_i 
                train_steps += steps_i
                for ii in range(labels_i.shape[1]):
                    batch_train_error_trajectory += error_trajectory_i[ii]
                    batch_train_samples += 1
                    train_error_trajectory += error_trajectory_i[ii]
                    train_samples += 1
                train_error_output.append(error_trajectory_i)
                if sample_i % args['report_num'] == 0:
                    print('epoch {} step {}/{}: batch train loss (per step) is {:.4g}, batch_train error (per step) is {:.3g} \n \
                        batch train error (per trajectory) is {:.3g}'.format(epoch, sample_i, N_train, 
                            batch_train_loss / batch_train_steps, batch_train_error_step / batch_train_steps, 
                            batch_train_error_trajectory / batch_train_samples))
                    with open(log_file, 'a+') as f:
                        f.write('epoch {} step {}/{}: batch train loss (per step) is {:.4g}, batch_train error (per step) is {:.3g} \n \
                        batch train error (per trajectory) is {:.3g}\n'.format(epoch, sample_i, N_train, 
                            batch_train_loss / batch_train_steps, batch_train_error_step / batch_train_steps, 
                            batch_train_error_trajectory / batch_train_samples))
                    batch_train_loss = 0
                    batch_train_error_step = 0
                    batch_train_error_trajectory = 0
                    batch_train_steps = 0
                    batch_train_samples = 0
                    if args['learn_tau']:
                        current_tau_1 = tau_1.eval()
                    else:
                        current_tau_1 = tau_1
                    print('tau_1:' + str(current_tau_1) + ' b: ' + str(b.eval()) + \
                        ' intercept_e: ' + str(intercept_e.eval()))
                    print('norm of excitatory filter:' + str(tf.norm(weights_e).eval()))
                    print('norm of inhibitory filter:' + str(tf.norm(weights_i).eval()))
                    with open(log_file, 'a+') as f:
                        f.write('tau_1:' + str(current_tau_1) + ' b: ' + str(b.eval()) + \
                            ' intercept_e: ' + str(intercept_e.eval()) + '\n')
                        f.write('norm of excitatory filter:' + str(tf.norm(weights_e).eval()) + '\n')
                if args["save_intermediate_weights"] and sample_i % args["save_steps"] == 1:
                    if not os.path.isdir(args['save_path'] + 'checkpoints'):
                        os.mkdir(args['save_path'] + 'checkpoints')
                    temp_dir = args['save_path'] + 'checkpoints/' + str(epoch) + '-' + str(sample_i) + '/'
                    os.mkdir(temp_dir)

                    if rotation_symmetry:
                        #0: right motion, 1: left motion, 2: up motion, 3: down motion
                        rotate_num_list = [0, 2, 1, 3] 
                        weights_e_list = []
                        weights_i_list = []
                        for i in range(4):
                            weights_e_reshaped = tf.reshape(weights_e,[K,K,1])
                            weights_e_rotated = tf.image.rot90(weights_e_reshaped,
                                rotate_num_list[i])
                            weights_e_reshaped_back = tf.reshape(weights_e_rotated, [K*K,1])
                            weights_e_list.append(weights_e_reshaped_back)
                
                            weights_i_reshaped = tf.reshape(weights_i,[K,K,1])
                            weights_i_rotated = tf.image.rot90(weights_i_reshaped,
                                rotate_num_list[i])
                            weights_i_reshaped_back = tf.reshape(weights_i_rotated, [K*K,1])
                            weights_i_list.append(weights_i_reshaped_back)
                        weights_e_save = tf.concat(weights_e_list, axis=1)
                        weights_i_save = tf.concat(weights_i_list, axis=1)
            
                    weights_e_out = weights_e_save.eval()
                    weights_i_out = weights_i_save.eval()
                    weights_intensity_out = None
                    if args['use_intensity']:
                        weights_intensity_out = weights_intensity.eval()
                    intercept_e_out = intercept_e.eval()
                    intercept_i_out = intercept_i.eval()
                    if args['learn_tau']:
                        tau_1_out = tau_1.eval()
                    else:
                        tau_1_out = tau_1
                    b_out = b.eval()
                    if args['train_a']:
                        a_out = a.eval()
                    else:
                        a_out = a
        
                    np.save(temp_dir + 'trained_weights_e', weights_e_out)
                    np.save(temp_dir + 'trained_weights_i', weights_i_out)
                    if args['use_intensity']:
                        np.save(temp_dir + 'trained_weights_intensity', 
                        weights_intensity_out)
                    np.save(temp_dir + 'trained_intercept_e', intercept_e_out)
                    np.save(temp_dir + 'trained_intercept_i', intercept_i_out)
                    np.save(temp_dir + 'trained_tau_1', tau_1_out)
                    np.save(temp_dir + 'trained_b', b_out)
                    np.save(temp_dir + 'trained_a', a_out)
            print('epoch {}: overall train loss (per step) is {:.4g}, overall train error (per step) is {:.3g}\n \
                overall train error (per trajectory) is {:.3g}'.format(epoch,
                train_loss / train_steps, train_error_step / train_steps,
                train_error_trajectory / train_samples))
            with open(log_file, 'a+') as f:
                f.write('epoch {}: overall train loss (per step) is {:.4g}, overall train error (per step) is {:.3g}\n \
                overall train error (per trajectory) is {:.3g}\n'.format(epoch,
                train_loss / train_steps, train_error_step / train_steps,
                train_error_trajectory / train_samples))

        # testing
        test_loss = 0
        test_error_step = 0
        test_error_trajectory = 0
        test_error_type1 = 0
        test_error_type2 = 0
        test_steps = 0
        test_samples = 0
        test_samples_type1 = 0
        test_samples_type2 = 0
        y_true = []
        y_score = []
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for sample_i, UV_flow_file_i, labels_i, step_weights_i in zip(range(1, N_test+1), 
            test_flow_files, test_labels, args['test_distances']):
            labels_i = np.array(labels_i).flatten()
#             steps_i = 1
#             if not args['use_step_weight']:
#                 step_weights_i = np.ones(steps_i)
            if args['use_intensity']:
                if args['frames_to_remove'] == 0:
                    UV_flow_i,step_weights_i,labels_i =\
                        load_compressed_dataset(args, UV_flow_file_i, labels_i)
                    frame_intensity_i,step_weights_i,labels_i =\
                        load_compressed_dataset(args, args['train_frames'][sample_i-1],labels_i)
                else:
                    UV_flow_i,step_weights_i,labels_i =\
                        load_compressed_dataset(args, UV_flow_file_i,labels_i)[:-args['frames_to_remove']]
                    frame_intensity_i,step_weights_i,labels_i =\
                        load_compressed_dataset(args, args['train_frames'][sample_i-1],labels_i)[:-args['frames_to_remove']]
                loss_i, error_step_i, error_trajectory_i, probabilities_i = \
                sess.run([loss, error_step, error_trajectory, probabilities],
                    {UV_flow:UV_flow_i, labels:labels_i, frame_intensity: frame_intensity_i,
                    step_weights: step_weights_i})
            else:
                if args['frames_to_remove'] == 0:
                    UV_flow_i,step_weights_i,labels_i =\
                        load_compressed_dataset(args, UV_flow_file_i,labels_i)
                else:
                    UV_flow_i,step_weights_i,labels_i =\
                        load_compressed_dataset(args, UV_flow_file_i,labels_i)[:-args['frames_to_remove']]
                loss_i, error_step_i, error_trajectory_i, probabilities_i = \
                sess.run([loss, error_step, error_trajectory, probabilities],
                    {UV_flow:UV_flow_i, labels:labels_i, step_weights: step_weights_i})
            steps_i = 1
            test_loss += loss_i
            test_error_step += error_step_i * steps_i
            test_steps += steps_i
            for ii in range(labels_i.shape[1]):
                y_true.append(labels_i[0][ii])
                y_score.append(probabilities_i[ii])
                test_error_trajectory += error_trajectory_i[ii]
                test_samples += 1
                if labels_i[0][ii] == 0:
                    test_error_type1 += error_trajectory_i[ii]
                    test_samples_type1 += 1
                    if error_trajectory_i[ii] == 0:
                        true_negative += 1
                    elif error_trajectory_i[ii] == 1:
                        false_positive += 1
                else:
                    test_error_type2 += error_trajectory_i[ii]
                    test_samples_type2 += 1
                    if error_trajectory_i[ii] == 0:
                        true_positive += 1
                    elif error_trajectory_i[ii] == 1:
                        false_negative += 1
        auc_roc = sm.roc_auc_score(y_true, y_score)
        auc_pr = sm.average_precision_score(y_true, y_score)
        print('Test loss (per step) is {:.4g}, test error (per step) is {:.3g}, test error (per trajectory) is {:.3g}\n \
            test AUCROC (per trajectory) is {:.3g}, test PRAUC (per trajectory) is {:.3g}'.format(test_loss / test_steps, 
            test_error_step / test_steps, test_error_trajectory / test_samples, auc_roc, auc_pr))
        print('True positive is {}, False positive is {} \n \
               False negative is {}, True negative is {}'.format(true_positive, 
                false_positive, false_negative, true_negative))
        with open(log_file, 'a+') as f:
            f.write('Test loss (per step) is {:.4g}, test error (per step) is {:.3g}, test error (per trajectory) is {:.3g}\n \
            test AUCROC (per trajectory) is {:.3g}, test PRAUC (per trajectory) is {:.3g}\n'.format(test_loss / test_steps, 
            test_error_step / test_steps, test_error_trajectory / test_samples, auc_roc, auc_pr))
            f.write('True positive is {}, False positive is {} \n \
               False negative is {}, True negative is {}\n'.format(true_positive, 
                false_positive, false_negative, true_negative))
        if rotation_symmetry:
            #0: right motion, 1: left motion, 2: up motion, 3: down motion
            rotate_num_list = [0,2,1,3] 
            weights_e_list = []
            weights_i_list = []
            for i in range(4):
                weights_e_reshaped = tf.reshape(weights_e,[K,K,1])
                weights_e_rotated = tf.image.rot90(weights_e_reshaped,
                    rotate_num_list[i])
                weights_e_reshaped_back = tf.reshape(weights_e_rotated, [K*K,1])
                weights_e_list.append(weights_e_reshaped_back)
                
                weights_i_reshaped = tf.reshape(weights_i,[K,K,1])
                weights_i_rotated = tf.image.rot90(weights_i_reshaped,
                    rotate_num_list[i])
                weights_i_reshaped_back = tf.reshape(weights_i_rotated, [K*K,1])
                weights_i_list.append(weights_i_reshaped_back)
            weights_e = tf.concat(weights_e_list, axis=1)
            weights_i = tf.concat(weights_i_list, axis=1)
            
        weights_e_out = weights_e.eval()
        weights_i_out = weights_i.eval()
        weights_intensity_out = None
        if args['use_intensity']:
            weights_intensity_out = weights_intensity.eval()
        intercept_e_out = intercept_e.eval()
        intercept_i_out = intercept_i.eval()
        if args['learn_tau']:
            tau_1_out = tau_1.eval()
        else:
            tau_1_out = tau_1
        b_out = b.eval()
        if args['train_a']:
            a_out = a.eval()
        else:
            a_out = a

        np.save(args['save_path'] + 'trained_weights_e', weights_e_out)
        np.save(args['save_path'] + 'trained_weights_i', weights_i_out)
        if args['use_intensity']:
            np.save(args['save_path'] + 'trained_weights_intensity', 
                weights_intensity_out)
        np.save(args['save_path'] + 'trained_intercept_e', intercept_e_out)
        np.save(args['save_path'] + 'trained_intercept_i', intercept_i_out)
        np.save(args['save_path'] + 'trained_tau_1', tau_1_out)
        np.save(args['save_path'] + 'trained_b', b_out)
        np.save(args['save_path'] + 'trained_a', a_out)
        np.save(args['save_path'] + 'loss_output', loss_output)
        np.save(args['save_path'] + 'train_error_output', train_error_output)
        np.save(args['save_path'] + 'b_output', b_output)
        np.save(args['save_path'] + 'intercept_e_output', intercept_e_output)
        np.save(args['save_path'] + 'intercept_i_output', intercept_i_output)
        np.save(args['save_path'] + 'auc_roc', auc_roc)
        np.save(args['save_path'] + 'auc_pr', auc_pr)

    print('Trained tau_1 is ', np.around(tau_1_out, 3))
    print('Trained b is ', np.around(b_out, 3))
    print('Trained a is ', np.around(a_out, 3))
    print('Trained intercept_e is ', np.around(intercept_e_out, 3))
    print('Trained intercept_i is ', np.around(intercept_i_out, 3))
    with open(log_file, 'a+') as f:
        f.write('Trained tau_1 is ' + str(np.around(tau_1_out, 3)) + '\n')
        f.write('Trained b is ' + str(np.around(b_out,3 )) + '\n')
        f.write('Trained a is ' + str(np.around(a_out, 3)) + '\n')
        f.write('Trained intercept_e is '  + str(np.around(intercept_e_out, 3)) + '\n')
        f.write('Trained intercept_i is '  + str(np.around(intercept_i_out, 3)) + '\n')

    end = time.time()
    print('Training took {:.0f} second'.format(end - start))
    with open(log_file, 'a+') as f:
        f.write('Training took {:.0f} second\n'.format(end - start))
    
    return weights_intensity_out, weights_e_out, weights_i_out, \
    intercept_e_out, intercept_i_out, tau_1_out, b_out, a_out


# Train and test the model inhibitory2 for classification
def test_model_C_i2(args, test_flow_files, test_labels, model_path):
    '''
    Args:
    args: a dictionary that contains problem parameters
    train_flow_files: files for training UV flows
    train_labels: labels (probability of hit, either 0 or 1) for training
    test_flow_files: files for testing UV flows
    test_labels: labels (probability of hit, either 0 or 1) for testing
    
    Returns:
    
    '''
    start = time.time()

    Q = args['Q']
    K = args['K']
    rotation_symmetry = args['rotation_symmetry']
    flip_symmetry = args['flip_symmetry']
    restrict_nonneg_weight = args['restrict_nonneg_weight']
    lr = args['lr']
    N_epochs = args['N_epochs']
    loss_output = []
    lplc2_cells = opsg.get_lplc2_cells_xy_angles(Q)
    
    N_test = len(test_flow_files)
        
    # inputs
    UV_flow = tf.compat.v1.placeholder(tf.float32, [None,None,Q,K*K,4], 
        name = 'UV_flow')
    step_weights = tf.compat.v1.placeholder(tf.float32, [None,None], name='step_weights')
    labels = tf.compat.v1.placeholder(tf.float32, [None,None], name = 'labels')
    frame_intensity = None
    if args['use_intensity']:
        frame_intensity = tf.compat.v1.placeholder(tf.float32, 
            [None,None,Q,K*K], name='frame_intensity')
        
    # model variables
    a_input = np.load(model_path + "trained_a.npy")
    b_input = np.load(model_path + "trained_b.npy")
    intercept_e_input = np.load(model_path + "trained_intercept_e.npy")
    tau_1_input = np.load(model_path + "trained_tau_1.npy")
    weights_e_input = np.load(model_path + "trained_weights_e.npy")
    weights_i_input = np.load(model_path + "trained_weights_i.npy")
    intercept_i_input = np.load(model_path + "trained_intercept_i.npy")

    # variables
    with tf.compat.v1.variable_scope('inhibitory_2'):
        initializer = tf.compat.v1.keras.initializers.VarianceScaling()
        positive_initializer = tf.compat.v1.keras.initializers.TruncatedNormal(mean=0.5,stddev=0.1)
        zero_initializer = tf.constant_initializer(0)
        if rotation_symmetry:
            if not flip_symmetry:
                weights_e = tf.Variable(initializer([K*K,1]), name='weights_e')
                weights_i = tf.Variable(initializer([K*K,1]), name='weights_i')
            else:
                if args['fine_tune_weights']:
                    saved_weights_e = np.float32(np.load(args['fine_tune_model_dir'] + 'trained_weights_e.npy'))
                    weights_e_raw = tf.Variable(initial_value=saved_weights_e[:(K+1)//2*K,0], 
                        name='weights_e')
                    weights_e = expand_weight(weights_e_raw, (K+1)//2, K, K%2==0)

                    saved_weights_i = np.float32(np.load(args['fine_tune_model_dir'] + 'trained_weights_i.npy'))
                    weights_i_raw = tf.Variable(initial_value=saved_weights_i[:(K+1)//2*K,0], 
                        name='weights_i')
                    weights_i = expand_weight(weights_i_raw, (K+1)//2, K, K%2==0)
                else:
                    weights_e_raw = tf.Variable(positive_initializer([(K+1)//2*K,1]), 
                        name='weights_e')
                    weights_e = expand_weight(weights_e_raw, (K+1)//2, K, K%2==0)
                    weights_i_raw = tf.Variable(positive_initializer([(K+1)//2*K,1]), 
                        name='weights_i')
                    weights_i = expand_weight(weights_i_raw, (K+1)//2, K, K%2==0)
        else:
            weights_e = tf.Variable(initializer([K*K,4]), name='weights_e')
            weights_i = tf.Variable(initializer([K*K,4]), name='weights_i')
        weights_intensity = None
        if args['use_intensity']:
            weights_intensity = tf.Variable(initializer([K*K]), 
                name='weights_intensity')
        if args['fine_tune_intercepts_and_b']:
            saved_intercept_e = np.float32(np.load(args['fine_tune_model_dir'] + 'trained_intercept_e.npy'))
            intercept_e = tf.Variable(initial_value=saved_intercept_e, name='intercept_e')

            saved_intercept_i = np.float32(np.load(args['fine_tune_model_dir'] + 'trained_intercept_i.npy'))
            intercept_i = tf.Variable(initial_value=saved_intercept_i, name='intercept_i')
        else:
            intercept_e = tf.Variable(initializer([1]), shape=(1),name='intercept_e')
            intercept_i = tf.Variable(initializer([1]), shape=(1),name='intercept_i')
        if args['learn_tau']:
            tau_1 = tf.Variable(initializer([1]), name='tau_1')
        else:
            tau_1 = args['tau']
        #b = tf.Variable(initializer([1]), name='b')
        if args['fine_tune_intercepts_and_b']: 
            saved_b = np.float32(np.load(args['fine_tune_model_dir'] + 'trained_b.npy'))
            b = tf.Variable(initial_value=saved_b, name='b')
        else:
            b = tf.Variable(zero_initializer([1]), name='b')
        if args['train_a']:
            a = tf.Variable(initializer([1]), name='a')
        else:
            a = tf.constant(np.array([1.0]),tf.float32,name='a')
    
    # Regularization
    l1_l2_regu_we = tf.contrib.layers.l1_l2_regularizer(scale_l1=args["l1_regu_we"],scale_l2=args["l2_regu_we"],scope=None)
    l1_l2_regu_wi = tf.contrib.layers.l1_l2_regularizer(scale_l1=args["l1_regu_wi"],scale_l2=args["l2_regu_wi"],scope=None)
    l1_l2_regu_a = tf.contrib.layers.l1_l2_regularizer(scale_l1=args["l1_regu_a"],scale_l2=args["l2_regu_a"],scope=None)

    # loss and error function
    loss, error_step, error_trajectory, _, probabilities = \
    loss_error_with_inhibitory(args, weights_e, weights_i, weights_intensity, intercept_e, intercept_i, UV_flow, frame_intensity, 
                             labels, tau_1, a, b, step_weights,l1_l2_regu_we,l1_l2_regu_wi,l1_l2_regu_a)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        # testing
        test_loss = 0
        test_error_step = 0
        test_error_trajectory = 0
        test_error_type1 = 0
        test_error_type2 = 0
        test_steps = 0
        test_samples = 0
        test_samples_type1 = 0
        test_samples_type2 = 0
        y_true = []
        y_score = []
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        logits_output = []
        for sample_i, UV_flow_file_i, labels_i, step_weights_i in zip(range(1, N_test+1), 
            test_flow_files, test_labels, args['test_distances']):
            labels_i = np.array(labels_i).flatten()
#             steps_i = 1
#             if not args['use_step_weight']:
#                 step_weights_i = np.ones(steps_i)
            if args['use_intensity']:
                if args['frames_to_remove'] == 0:
                    UV_flow_i,step_weights_i,labels_i =\
                        load_compressed_dataset(args, UV_flow_file_i, labels_i)
                    frame_intensity_i,step_weights_i,labels_i =\
                        load_compressed_dataset(args, args['train_frames'][sample_i-1],labels_i)
                else:
                    UV_flow_i,step_weights_i,labels_i =\
                        load_compressed_dataset(args, UV_flow_file_i,labels_i)[:-args['frames_to_remove']]
                    frame_intensity_i,step_weights_i,labels_i =\
                        load_compressed_dataset(args, args['train_frames'][sample_i-1],labels_i)[:-args['frames_to_remove']]
                loss_i, error_step_i, error_trajectory_i, probabilities_i = \
                sess.run([loss, error_step, error_trajectory, probabilities],
                    {UV_flow:UV_flow_i, labels:labels_i, frame_intensity: frame_intensity_i,
                    step_weights: step_weights_i})
            else:
                if args['frames_to_remove'] == 0:
                    UV_flow_i,step_weights_i,labels_i =\
                        load_compressed_dataset(args, UV_flow_file_i,labels_i)
                else:
                    UV_flow_i,step_weights_i,labels_i =\
                        load_compressed_dataset(args, UV_flow_file_i,labels_i)[:-args['frames_to_remove']]
                loss_i, error_step_i, error_trajectory_i, probabilities_i = \
                sess.run([loss, error_step, error_trajectory, probabilities],
                    {UV_flow:UV_flow_i, labels:labels_i, step_weights: step_weights_i,\
                     weights_e:weights_e_input[:,:1].astype(np.float32),weights_i:weights_i_input[:,:1].astype(np.float32),\
                     intercept_e:intercept_e_input.astype(np.float32),intercept_i:intercept_i_input.astype(np.float32),\
                     a:a_input.astype(np.float32), b:b_input.astype(np.float32)})
            steps_i = 1
            test_loss += loss_i
            test_error_step += error_step_i * steps_i
            test_steps += steps_i
            for ii in range(labels_i.shape[1]):
                y_true.append(labels_i[0][ii])
                y_score.append(probabilities_i[ii])
                test_error_trajectory += error_trajectory_i[ii]
                test_samples += 1
                if labels_i[0][ii] == 0:
                    test_error_type1 += error_trajectory_i[ii]
                    test_samples_type1 += 1
                    if error_trajectory_i[ii] == 0:
                        true_negative += 1
                    elif error_trajectory_i[ii] == 1:
                        false_positive += 1
                else:
                    test_error_type2 += error_trajectory_i[ii]
                    test_samples_type2 += 1
                    if error_trajectory_i[ii] == 0:
                        true_positive += 1
                    elif error_trajectory_i[ii] == 1:
                        false_negative += 1
        auc_roc = sm.roc_auc_score(y_true, y_score)
        auc_pr = sm.average_precision_score(y_true, y_score)
        cm = sm.confusion_matrix(y_true,np.round(y_score))
    
    return auc_roc,auc_pr,cm,y_true,y_score


# Train and test the model inhibitory2 for classification
def train_and_test_model_i2_snapshot(args, train_flow_files, train_labels, 
    test_flow_files, test_labels,train_flow_samples,train_frame_samples,train_labels_samples):
    '''
    Args:
    args: a dictionary that contains problem parameters
    train_flow_files: files for training UV flows
    train_labels: labels (probability of hit, either 0 or 1) for training
    test_flow_files: files for testing UV flows
    test_labels: labels (probability of hit, either 0 or 1) for testing
    train_flow_samples: snapshot training samples
    train_labels_samples: corresponding labels for snapshot samples
    '''
    start = time.time()

    assert(len(train_flow_files) == len(train_flow_samples))

    validation_portion = 0.2

    valid_flow_samples = train_flow_samples[int((1-validation_portion)*len(train_flow_samples)):]
    train_flow_samples = train_flow_samples[:int((1-validation_portion)*len(train_flow_samples))]
    valid_frame_samples = train_frame_samples[int((1-validation_portion)*len(train_frame_samples)):]
    train_frame_samples = train_frame_samples[:int((1-validation_portion)*len(train_frame_samples))]
    valid_labels_samples = train_labels_samples[int((1-validation_portion)*len(train_labels_samples)):]
    train_labels_samples = train_labels_samples[:int((1-validation_portion)*len(train_labels_samples))]

    N_train = len(train_flow_samples)
    N_valid = len(valid_flow_samples)
    N_test = len(test_flow_files)
    

    Q = args['Q']
    K = args['K']
    rotation_symmetry = args['rotation_symmetry']
    flip_symmetry = args['flip_symmetry']
    restrict_nonneg_weight = args['restrict_nonneg_weight']
    lr = args['lr']
    N_epochs = args['N_epochs']
    log_file = args['save_path'] + 'log.txt'
    loss_output = []
    loss_output_test = []
    train_error_output = []
    validation_error_output = []
    validation_loss_output = []
    a_output = []
    b_output = []
    intercept_e_output = []
    intercept_i_output = []
    lplc2_cells = opsg.get_lplc2_cells_xy_angles(Q)
    
    with open(log_file, 'w') as f:
        f.write('Model setup:\n')
        f.write('----------------------------------------\n')
        f.write('Model is inhibitory2 \n')
        f.write('set_number: {}\n'.format(args['set_number']))
        f.write('data_path: {}\n'.format(args['data_path']))
        f.write('rotational_fraction: {}\n'.format(args['rotational_fraction']))
        f.write('Number of training examples: {}\n'.format(len(train_flow_files)))
        f.write('Number of testing examples: {}\n'.format(len(test_flow_files)))
        f.write('Q: {}\n'.format(args['Q']))
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
        f.write('frames to remove: {}\n'.format(args['frames_to_remove']))
        f.write('save_intermediate_weights: {}\n'.format(args['save_intermediate_weights']))
        f.write('save_steps: {}\n'.format(args['save_steps']))
        f.write('use_step_weight: {}\n'.format(args['use_step_weight']))
        f.write('l1 regularization on excitatory weights: {}\n'.format(args['l1_regu_we']))
        f.write('l2 regularization on excitatory weights: {}\n'.format(args['l2_regu_we']))
        f.write('l1 regularization on inhibitory weights: {}\n'.format(args['l1_regu_wi']))
        f.write('l2 regularization on inhibitory weights: {}\n'.format(args['l2_regu_wi']))
        f.write('l1 regularization on a: {}\n'.format(args['l1_regu_a']))
        f.write('l2 regularization on a: {}\n'.format(args['l2_regu_a']))
        f.write('restrict_nonneg_weight: {}\n'.format(
            args['restrict_nonneg_weight']))
        f.write('restrict_nonpos_intercept: {}\n'.format(
            args['restrict_nonpos_intercept']))
        f.write('rotation_symmetry: {}\n'.format(args['rotation_symmetry']))
        f.write('flip_symmetry: {}\n'.format(args['flip_symmetry']))
        f.write('use_intensity: {}\n'.format(args['use_intensity']))
        f.write('train_a: {}\n'.format(args['train_a']))
        f.write('max_response: {}\n'.format(args['max_response']))
        f.write('temporal_filter: {}\n'.format(args['temporal_filter']))
        f.write('learn_tau: {}\n'.format(args['learn_tau']))
        f.write('tau in standard scale: ' + 
            str(np.around(np.exp(-args['tau']), 3)) + '\n\n')
    

        
    # inputs
    UV_flow = tf.compat.v1.placeholder(tf.float32, [None,None,Q,K*K,4], 
        name = 'UV_flow')
    step_weights = tf.compat.v1.placeholder(tf.float32, [None,None], name='step_weights')
    labels = tf.compat.v1.placeholder(tf.float32, [None,None], name = 'labels')
    frame_intensity = None
    if args['use_intensity']:
        frame_intensity = tf.compat.v1.placeholder(tf.float32, 
            [None,K*K], name='frame_intensity')

    # variables
    with tf.compat.v1.variable_scope('inhibitory_2'):
        initializer = tf.compat.v1.keras.initializers.TruncatedNormal(mean=0.002,stddev=0.001)
        positive_initializer_e = tf.compat.v1.keras.initializers.TruncatedNormal(mean=0.2,stddev=0.1)
        positive_initializer_i = tf.compat.v1.keras.initializers.TruncatedNormal(mean=0.2,stddev=0.1)
        zero_initializer = tf.constant_initializer(0)
        negative_initializer = tf.constant_initializer(-1.)
        if rotation_symmetry:
            if not flip_symmetry:
                weights_e = tf.Variable(initializer([K*K,1]), name='weights_e')
                weights_i = tf.Variable(initializer([K*K,1]), name='weights_i')
            else:
                if args['fine_tune_weights']:
                    saved_weights_e = np.float32(np.load(args['fine_tune_model_dir'] + 'trained_weights_e.npy'))
                    weights_e_raw = tf.Variable(initial_value=saved_weights_e[:(K+1)//2*K,0], 
                        name='weights_e')
                    weights_e = expand_weight(weights_e_raw, (K+1)//2, K, K%2==0)

                    saved_weights_i = np.float32(np.load(args['fine_tune_model_dir'] + 'trained_weights_i.npy'))
                    weights_i_raw = tf.Variable(initial_value=saved_weights_i[:(K+1)//2*K,0], 
                        name='weights_i')
                    weights_i = expand_weight(weights_i_raw, (K+1)//2, K, K%2==0)
                else:
                    weights_e_raw = tf.Variable(positive_initializer_e([(K+1)//2*K,1]), 
                        name='weights_e')
                    weights_e = expand_weight(weights_e_raw, (K+1)//2, K, K%2==0)
                    weights_i_raw = tf.Variable(positive_initializer_i([(K+1)//2*K,1]), 
                        name='weights_i')
                    weights_i = expand_weight(weights_i_raw, (K+1)//2, K, K%2==0)
        else:
            weights_e = tf.Variable(initializer([K*K,4]), name='weights_e')
            weights_i = tf.Variable(initializer([K*K,4]), name='weights_i')
        weights_intensity = None
        if args['use_intensity']:
            weights_intensity = tf.Variable(initializer([K*K]), 
                name='weights_intensity')
        if args['fine_tune_intercepts_and_b']:
            saved_intercept_e = np.float32(np.load(args['fine_tune_model_dir'] + 'trained_intercept_e.npy'))
            intercept_e = tf.Variable(initial_value=saved_intercept_e, name='ntercept_e')

            saved_intercept_i = np.float32(np.load(args['fine_tune_model_dir'] + 'trained_intercept_i.npy'))
            intercept_i = tf.Variable(initial_value=saved_intercept_i, name='ntercept_i')
        else:
            intercept_e = tf.Variable(initializer([1]), name='intercept_e')
            intercept_i = tf.Variable(initializer([1]), name='intercept_i')
        if args['learn_tau']:
            tau_1 = tf.Variable(initializer([1]), name='tau_1')
        else:
            tau_1 = args['tau']
        #b = tf.Variable(initializer([1]), name='b')
        if args['fine_tune_intercepts_and_b']: 
            saved_b = np.float32(np.load(args['fine_tune_model_dir'] + 'trained_b.npy'))
            b = tf.Variable(initial_value=saved_b, name='b')
        else:
            b = tf.Variable(negative_initializer([1]), name='b')
        if args['train_a']:
            a = tf.Variable(initializer([1]), name='a')
        else:
            a = 1.

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        np.save(args['save_path'] + 'initial_weights_e', weights_e.eval())
        np.save(args['save_path'] + 'initial_weights_i', weights_i.eval())
            
    # Optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(lr,beta1=0.9)
#     optimizer = tf.compat.v1.train.GradientDescentOptimizer(lr)
#     optimizer = tf.compat.v1.train.MomentumOptimizer(lr,momentum=0.9)
#     optimizer = tf.compat.v1.train.AdagradOptimizer(lr)

    # Regularization
    l1_l2_regu_we = tf.contrib.layers.l1_l2_regularizer(scale_l1=args["l1_regu_we"],scale_l2=args["l2_regu_we"],scope=None)
    l1_l2_regu_wi = tf.contrib.layers.l1_l2_regularizer(scale_l1=args["l1_regu_wi"],scale_l2=args["l2_regu_wi"],scope=None)
    l1_l2_regu_a = tf.contrib.layers.l1_l2_regularizer(scale_l1=args["l1_regu_a"],scale_l2=args["l2_regu_a"],scope=None)

    # loss and error function
    loss, error_step, error_trajectory, _, probabilities = \
    loss_error_with_inhibitory(args, weights_e, weights_i, weights_intensity, intercept_e, intercept_i, UV_flow, frame_intensity, 
                             labels, tau_1, a, b, step_weights,l1_l2_regu_we,l1_l2_regu_wi,l1_l2_regu_a)

    # Optimization
    opt = optimizer.minimize(loss)
    opt_with_clip = opt
    if restrict_nonneg_weight:
        with tf.control_dependencies([opt]): # constrained optimization
            if not flip_symmetry:
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
    
    #config = tf.ConfigProto(inter_op_parallelism_threads=1)
    #with tf.compat.v1.Session(config=config) as sess:
    ind_bz = 0
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        # training
        for epoch in range(1, N_epochs+1):
            train_loss = 0
            train_error_step = 0
            train_error_trajectory = 0
            train_steps = 0
            train_samples = 0
            batch_train_loss = 0
            batch_train_error_step = 0
            batch_train_error_trajectory = 0
            batch_train_steps = 0
            batch_train_samples = 0
            for sample_i in range(1, N_train+1):
                ind_bz = ind_bz+1
                UV_flow_i = train_flow_samples[sample_i-1].copy()
                frame_intensity_i = train_frame_samples[sample_i-1].copy()
                labels_i = train_labels_samples[sample_i-1]
                step_weights_i = np.ones_like(labels_i)
                
                if args['use_intensity']:
                    _, loss_i, error_step_i, error_trajectory_i = \
                    sess.run([opt_with_clip, loss, error_step, error_trajectory],
                        {UV_flow:UV_flow_i, labels:labels_i, frame_intensity: frame_intensity_i,
                        step_weights: step_weights_i})
                else:
                    _, loss_i, error_step_i, error_trajectory_i = \
                    sess.run([opt_with_clip, loss, error_step, error_trajectory],
                        {UV_flow:UV_flow_i, labels:labels_i, step_weights: step_weights_i})
                    steps_i = 1
                loss_output.append(loss_i)
                if args['train_a']:
                    a_output.append(a.eval())
                else:
                    a_output.append(a)
                b_output.append(b.eval())
                intercept_e_output.append(intercept_e.eval())
                intercept_i_output.append(intercept_i.eval())
                batch_train_loss += loss_i 
                batch_train_error_step += error_step_i * steps_i
                batch_train_steps += steps_i
                train_error_step += error_step_i * steps_i
                train_loss += loss_i 
                train_steps += steps_i
                for ii in range(labels_i.shape[1]):
                    batch_train_error_trajectory += error_trajectory_i[ii]
                    batch_train_samples += 1
                    train_error_trajectory += error_trajectory_i[ii]
                    train_samples += 1
                train_error_output.append(error_trajectory_i)
                if sample_i % args['report_num'] == 0:
                    print('epoch {} step {}/{}: batch train loss (per step) is {:.4g}, batch_train error (per step) is {:.3g} \n \
                        batch train error (per trajectory) is {:.3g}'.format(epoch, sample_i, N_train, 
                            batch_train_loss / batch_train_steps, batch_train_error_step / batch_train_steps, 
                            batch_train_error_trajectory / batch_train_samples))
                    with open(log_file, 'a+') as f:
                        f.write('epoch {} step {}/{}: batch train loss (per step) is {:.4g}, batch_train error (per step) is {:.3g} \n \
                        batch train error (per trajectory) is {:.3g}\n'.format(epoch, sample_i, N_train, 
                            batch_train_loss / batch_train_steps, batch_train_error_step / batch_train_steps, 
                            batch_train_error_trajectory / batch_train_samples))
                    batch_train_loss = 0
                    batch_train_error_step = 0
                    batch_train_error_trajectory = 0
                    batch_train_steps = 0
                    batch_train_samples = 0
                    if args['learn_tau']:
                        current_tau_1 = tau_1.eval()
                    else:
                        current_tau_1 = tau_1
                    print('tau_1:' + str(current_tau_1) + ' b: ' + str(b.eval()) + \
                        ' intercept_e: ' + str(intercept_e.eval()))
                    print('norm of excitatory filter:' + str(tf.norm(weights_e).eval()))
                    print('norm of inhibitory filter:' + str(tf.norm(weights_i).eval()))
                    with open(log_file, 'a+') as f:
                        f.write('tau_1:' + str(current_tau_1) + ' b: ' + str(b.eval()) + \
                            ' intercept_e: ' + str(intercept_e.eval()) + '\n')
                        f.write('norm of excitatory filter:' + str(tf.norm(weights_e).eval()) + '\n')
                if args["save_intermediate_weights"] and ind_bz % args["save_steps"] == 1:
                    if not os.path.isdir(args['save_path'] + 'checkpoints'):
                        os.mkdir(args['save_path'] + 'checkpoints')
                    temp_dir = args['save_path'] + 'checkpoints/' + str(epoch) + '-' + str(sample_i) + '/'
                    os.mkdir(temp_dir)
                    if rotation_symmetry:
                        #0: right motion, 1: left motion, 2: up motion, 3: down motion
                        rotate_num_list = [0, 2, 1, 3] 
                        weights_e_list = []
                        weights_i_list = []
                        for i in range(4):
                            weights_e_reshaped = tf.reshape(weights_e,[K,K,1])
                            weights_e_rotated = tf.image.rot90(weights_e_reshaped,
                                rotate_num_list[i])
                            weights_e_reshaped_back = tf.reshape(weights_e_rotated, [K*K,1])
                            weights_e_list.append(weights_e_reshaped_back)
                
                            weights_i_reshaped = tf.reshape(weights_i,[K,K,1])
                            weights_i_rotated = tf.image.rot90(weights_i_reshaped,
                                rotate_num_list[i])
                            weights_i_reshaped_back = tf.reshape(weights_i_rotated, [K*K,1])
                            weights_i_list.append(weights_i_reshaped_back)
                        weights_e_save = tf.concat(weights_e_list, axis=1)
                        weights_i_save = tf.concat(weights_i_list, axis=1)
            
                    weights_e_out = weights_e_save.eval()
                    weights_i_out = weights_i_save.eval()
                    weights_intensity_out = None
                    if args['use_intensity']:
                        weights_intensity_out = weights_intensity.eval()
                    intercept_e_out = intercept_e.eval()
                    intercept_i_out = intercept_i.eval()
                    if args['learn_tau']:
                        tau_1_out = tau_1.eval()
                    else:
                        tau_1_out = tau_1
                    b_out = b.eval()
                    if args['train_a']:
                        a_out = a.eval()
                    else:
                        a_out = a
        
                    np.save(temp_dir + 'trained_weights_e', weights_e_out)
                    np.save(temp_dir + 'trained_weights_i', weights_i_out)
                    if args['use_intensity']:
                        np.save(temp_dir + 'trained_weights_intensity', 
                        weights_intensity_out)
                    np.save(temp_dir + 'trained_intercept_e', intercept_e_out)
                    np.save(temp_dir + 'trained_intercept_i', intercept_i_out)
                    np.save(temp_dir + 'trained_tau_1', tau_1_out)
                    np.save(temp_dir + 'trained_b', b_out)
                    np.save(temp_dir + 'trained_a', a_out)
                    ######################################################
                    ############## do evaluation on validation set #######
                    ######################################################
                    valid_loss = 0
                    valid_error_step = 0
                    valid_steps = 0
                    valid_samples = 0
                    for sample_i in range(1, N_valid+1):
                        UV_flow_i = valid_flow_samples[sample_i-1].copy()
                        frame_intensity_i = valid_frame_samples[sample_i-1].copy()
                        labels_i = valid_labels_samples[sample_i-1]
                        step_weights_i = np.ones_like(labels_i)
                        
                        if args['use_intensity']:
                            _, loss_i, error_step_i, error_trajectory_i = \
                            sess.run([opt_with_clip, loss, error_step, error_trajectory],
                                {UV_flow:UV_flow_i, labels:labels_i, frame_intensity: frame_intensity_i,
                                step_weights: step_weights_i})
                        else:
                            loss_i, error_step_i, error_trajectory_i = \
                            sess.run([loss, error_step, error_trajectory],
                                {UV_flow:UV_flow_i, labels:labels_i, step_weights: step_weights_i})
                        steps_i = 1
                        
                        valid_loss += loss_i 
                        valid_error_step += error_step_i * steps_i
                        valid_steps += steps_i
                    #np.save(temp_dir + 'valid_loss', valid_loss / valid_steps)
                    #np.save(temp_dir + 'valid_error', valid_error_step / valid_steps)
                    validation_loss_output.append(valid_loss / valid_steps)
                    validation_error_output.append(valid_error_step / valid_steps)


            print('epoch {}: overall train loss (per step) is {:.4g}, overall train error (per step) is {:.3g}\n \
                overall train error (per trajectory) is {:.3g}'.format(epoch,
                train_loss / train_steps, train_error_step / train_steps,
                train_error_trajectory / train_samples))
            with open(log_file, 'a+') as f:
                f.write('epoch {}: overall train loss (per step) is {:.4g}, overall train error (per step) is {:.3g}\n \
                overall train error (per trajectory) is {:.3g}\n'.format(epoch,
                train_loss / train_steps, train_error_step / train_steps,
                train_error_trajectory / train_samples))
                
        # testing
        test_loss = 0
        test_error_step = 0
        test_error_trajectory = 0
        test_error_type1 = 0
        test_error_type2 = 0
        test_steps = 0
        test_samples = 0
        test_samples_type1 = 0
        test_samples_type2 = 0
        y_true = []
        y_score = []
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for sample_i, UV_flow_file_i, labels_i, step_weights_i in zip(range(1, N_test+1), 
            test_flow_files, test_labels, args['test_distances']):
            labels_i = np.array(labels_i).flatten()
#             steps_i = 1
#             if not args['use_step_weight']:
#                 step_weights_i = np.ones(steps_i)
            if args['use_intensity']:
                if args['frames_to_remove'] == 0:
                    UV_flow_i,step_weights_i,labels_i =\
                        load_compressed_dataset(args, UV_flow_file_i, labels_i)
                    frame_intensity_i,step_weights_i,labels_i =\
                        load_compressed_dataset(args, args['train_frames'][sample_i-1],labels_i)
                else:
                    UV_flow_i,step_weights_i,labels_i =\
                        load_compressed_dataset(args, UV_flow_file_i,labels_i)[:-args['frames_to_remove']]
                    frame_intensity_i,step_weights_i,labels_i =\
                        load_compressed_dataset(args, args['train_frames'][sample_i-1],labels_i)[:-args['frames_to_remove']]
                loss_i, error_step_i, error_trajectory_i, probabilities_i = \
                sess.run([loss, error_step, error_trajectory, probabilities],
                    {UV_flow:UV_flow_i, labels:labels_i, frame_intensity: frame_intensity_i,
                    step_weights: step_weights_i})
            else:
                if args['frames_to_remove'] == 0:
                    UV_flow_i,step_weights_i,labels_i =\
                        load_compressed_dataset(args, UV_flow_file_i,labels_i)
                else:
                    UV_flow_i,step_weights_i,labels_i =\
                        load_compressed_dataset(args, UV_flow_file_i,labels_i)[:-args['frames_to_remove']]
                loss_i, error_step_i, error_trajectory_i, probabilities_i = \
                sess.run([loss, error_step, error_trajectory, probabilities],
                    {UV_flow:UV_flow_i, labels:labels_i, step_weights: step_weights_i})
            steps_i = 1
            test_loss += loss_i
            test_error_step += error_step_i * steps_i
            test_steps += steps_i
            for ii in range(labels_i.shape[1]):
                y_true.append(labels_i[0][ii])
                y_score.append(probabilities_i[ii])
                test_error_trajectory += error_trajectory_i[ii]
                test_samples += 1
                if labels_i[0][ii] == 0:
                    test_error_type1 += error_trajectory_i[ii]
                    test_samples_type1 += 1
                    if error_trajectory_i[ii] == 0:
                        true_negative += 1
                    elif error_trajectory_i[ii] == 1:
                        false_positive += 1
                else:
                    test_error_type2 += error_trajectory_i[ii]
                    test_samples_type2 += 1
                    if error_trajectory_i[ii] == 0:
                        true_positive += 1
                    elif error_trajectory_i[ii] == 1:
                        false_negative += 1
        auc_roc = sm.roc_auc_score(y_true, y_score)
        auc_pr = sm.average_precision_score(y_true, y_score)
        print('Test loss (per step) is {:.4g}, test error (per step) is {:.3g}, test error (per trajectory) is {:.3g}\n \
            test AUCROC (per trajectory) is {:.3g}, test PRAUC (per trajectory) is {:.3g}'.format(test_loss / test_steps, 
            test_error_step / test_steps, test_error_trajectory / test_samples, auc_roc, auc_pr))
        print('True positive is {}, False positive is {} \n \
               False negative is {}, True negative is {}'.format(true_positive, 
                false_positive, false_negative, true_negative))
        with open(log_file, 'a+') as f:
            f.write('Test loss (per step) is {:.4g}, test error (per step) is {:.3g}, test error (per trajectory) is {:.3g}\n \
            test AUCROC (per trajectory) is {:.3g}, test PRAUC (per trajectory) is {:.3g}\n'.format(test_loss / test_steps, 
            test_error_step / test_steps, test_error_trajectory / test_samples, auc_roc, auc_pr))
            f.write('True positive is {}, False positive is {} \n \
               False negative is {}, True negative is {}\n'.format(true_positive, 
                false_positive, false_negative, true_negative))

        np.save(args['save_path'] + 'auc_roc', auc_roc)
        np.save(args['save_path'] + 'auc_pr', auc_pr)
        np.save(args['save_path'] + 'y_true', y_true)
        np.save(args['save_path'] + 'y_score', y_score)
        
        if rotation_symmetry:
            #0: right motion, 1: left motion, 2: up motion, 3: down motion
            rotate_num_list = [0,2,1,3] 
            weights_e_list = []
            weights_i_list = []
            for i in range(4):
                weights_e_reshaped = tf.reshape(weights_e,[K,K,1])
                weights_e_rotated = tf.image.rot90(weights_e_reshaped,
                    rotate_num_list[i])
                weights_e_reshaped_back = tf.reshape(weights_e_rotated, [K*K,1])
                weights_e_list.append(weights_e_reshaped_back)
                
                weights_i_reshaped = tf.reshape(weights_i,[K,K,1])
                weights_i_rotated = tf.image.rot90(weights_i_reshaped,
                    rotate_num_list[i])
                weights_i_reshaped_back = tf.reshape(weights_i_rotated, [K*K,1])
                weights_i_list.append(weights_i_reshaped_back)
            weights_e = tf.concat(weights_e_list, axis=1)
            weights_i = tf.concat(weights_i_list, axis=1)
            
        weights_e_out = weights_e.eval()
        weights_i_out = weights_i.eval()
        weights_intensity_out = None
        if args['use_intensity']:
            weights_intensity_out = weights_intensity.eval()
        intercept_e_out = intercept_e.eval()
        intercept_i_out = intercept_i.eval()
        if args['learn_tau']:
            tau_1_out = tau_1.eval()
        else:
            tau_1_out = tau_1
        b_out = b.eval()
        if args['train_a']:
            a_out = a.eval()
        else:
            a_out = a

        np.save(args['save_path'] + 'trained_weights_e', weights_e_out)
        np.save(args['save_path'] + 'trained_weights_i', weights_i_out)
        if args['use_intensity']:
            np.save(args['save_path'] + 'trained_weights_intensity', 
                weights_intensity_out)
        np.save(args['save_path'] + 'trained_intercept_e', intercept_e_out)
        np.save(args['save_path'] + 'trained_intercept_i', intercept_i_out)
        np.save(args['save_path'] + 'trained_tau_1', tau_1_out)
        np.save(args['save_path'] + 'trained_b', b_out)
        np.save(args['save_path'] + 'trained_a', a_out)
        np.save(args['save_path'] + 'loss_output', loss_output)
        np.save(args['save_path'] + 'train_error_output', train_error_output)
        np.save(args['save_path'] + 'a_output', a_output)
        np.save(args['save_path'] + 'b_output', b_output)
        np.save(args['save_path'] + 'intercept_e_output', intercept_e_output)
        np.save(args['save_path'] + 'intercept_i_output', intercept_i_output)
        np.save(args['save_path'] + 'validation_loss_output', validation_loss_output)
        np.save(args['save_path'] + 'validation_error_output', validation_error_output)

    print('Trained tau_1 is ', np.around(tau_1_out, 3))
    print('Trained b is ', np.around(b_out, 3))
    print('Trained a is ', np.around(a_out, 3))
    print('Trained intercept_e is ', np.around(intercept_e_out, 3))
    print('Trained intercept_i is ', np.around(intercept_i_out, 3))
    with open(log_file, 'a+') as f:
        f.write('Trained tau_1 is ' + str(np.around(tau_1_out, 3)) + '\n')
        f.write('Trained b is ' + str(np.around(b_out,3 )) + '\n')
        f.write('Trained a is ' + str(np.around(a_out, 3)) + '\n')
        f.write('Trained intercept_e is '  + str(np.around(intercept_e_out, 3)) + '\n')
        f.write('Trained intercept_i is '  + str(np.around(intercept_i_out, 3)) + '\n')
    
    end = time.time()
    print('Training took {:.0f} second'.format(end - start))
    with open(log_file, 'a+') as f:
        f.write('Training took {:.0f} second\n'.format(end - start))
    
    return weights_intensity_out, weights_e_out, weights_i_out, \
    intercept_e_out, intercept_i_out, tau_1_out, b_out, a_out


############################################
######### Linear-Nonlinear Model #################
############################################

# Input to a LPLC2 neuron at a specific time point according to 
# model inhibitory 2
def get_input_LPLC2_ln(args, weights_e, intercept_e, UV_flow_t):
    '''
    Args:
    args: a dictionary that contains problem parameters
    weights_e: tensor, weights for excitatory neurons, K*K by 4, for axis=1, 
                0: right motion, 1: left motion, 2: up motion, 3: down motion
    intercept_e: tensor, intercept in the activation function for the LPLC2 neuron
    UV_flow_t: tensor, flow field at time point t, Q by K*K by 4
    
    Returns:
    input_t: input to a LPLC2 neuron at a specific time point t
    '''
    
    K = args['K']
    rotation_symmetry = args['rotation_symmetry']
                
    intercept_e = tf.dtypes.cast(intercept_e,tf.float32)
    
    input_t = intercept_e
    
    # rotation is counter-clockwise
    # 0: right motion, 1: left motion, 2: up motion, 3: down motion
    rotate_num_list = [0, 2, 1, 3] 
    
    # excitatory input
    for i in range(4):
        if rotation_symmetry:
            weights_e_reshaped = tf.reshape(weights_e, [K,K,1])
            weights_e_rotated = tf.image.rot90(weights_e_reshaped,
                rotate_num_list[i])
            weights_e_reshaped_back = tf.reshape(weights_e_rotated, [K*K,1])
            input_t = input_t + tf.tensordot(UV_flow_t[:,:,:,i], 
                weights_e_reshaped_back,axes=[[2],[0]])
        else:
            input_t = input_t + tf.tensordot(UV_flow_t[:,:,:,i], weights_e[:,i], 
                axes=[[2],[0]])
    
    # sum pooling across multiple neurons
    if args['activation'] == 0:
        if not args['square_activation']:
            input_t = tf.reduce_sum(tf.nn.relu(input_t),axis=1)
        else:
            input_t = tf.reduce_sum(tf.math.square(tf.nn.relu(input_t)),axis=1)
    elif args['activation'] == 1:
        if not args['square_activation']:
            input_t = tf.reduce_sum(tf.nn.leaky_relu(input_t, 
                alpha=args['leaky_relu_constant']),axis=1)
        else:
            input_t = tf.reduce_sum(tf.math.squre(tf.nn.leaky_relu(input_t, 
                alpha=args['leaky_relu_constant'])),axis=1)
    elif args['activation'] == 2:
        if not args['square_activation']:
            input_t = tf.reduce_sum(tf.nn.elu(input_t),axis=1)
        else:
            input_t = tf.reduce_sum(tf.math.square(tf.nn.elu(input_t)),axis=1)
    elif args['activation'] == 3:
        if not args['square_activation']:
            input_t = tf.reduce_sum(tf.math.tanh(input_t),axis=1)
        else:
            input_t = tf.reduce_sum(tf.math.square(tf.math.tanh(input_t)),axis=1)
    input_t = tf.reshape(input_t,[-1])
    
    return input_t
    
    
# Input to a LPLC2 neuron for a whole sequential signal according to
# model LN
def get_input_LPLC2_ln_T(args, weights_e, intercept_e, UV_flow):
    '''
    Args:
    args: a dictionary that contains problem parameters
    weights_e: tensor, weights for excitatory neurons, K*K by 4, for axis=1, 
                0: right motion, 1: left motion, 2: up motion, 3: down motion
    intercept_e: tensor, intercept in the activation function for the LPLC2 neuron
    UV_flow: tensor, flow field, steps by Q by K*K by 4
    
    Returns:
    input_T: tensor, input to a LPLC2 neuron, len(input_T_tf) = steps
    '''
    
    input_T = tf.map_fn(lambda x:\
                        get_input_LPLC2_ln(args, weights_e,
                            intercept_e, x), UV_flow)
    
    return input_T

#  Loss and error of the classification for the excitatory only model
def loss_error_C_ln(args, weights_e, weights_intensity, 
    intercept_e, UV_flow, frame_intensity, labels, tau_1, a, b, step_weights,
    regularizer_we, regularizer_a):
    '''
    Args:
    args: a dictionary that contains problem parameters
    weights_e: tensor, weights for excitatory neurons, K*K by 4, for axis=1, 
                0: right motion, 1: left motion, 2: up motion, 3: down motion
    weights_intensity: tensor, weights for intensities, K*K
    intercept_e: tensor, intercept in the activation function
    UV_flow: tensor, data, flow field, steps by Q by K*K by 4
    frame_intensity: data, coarse grained frame intensity, steps by Q by K*K
    labels: data (probability of hit), len(labels) = steps
    tau_1: log of the timescale of the filter
    a: linear coefficient of the logistic classifier
    b: intercept of the logistic classifier
    
    Returns
    loss: cross entropy loss function for one trajectory
    error_step: binary classification error for all steps in one trajectory
    error_trajectory: binary classification error for one trajectory
    filtered_res: filtered response for all steps in one trajectory 
    probabilities: predicted probabilities of hit for all steps in one trajectory 
    '''

    input_T = get_input_LPLC2_ln_T(args, weights_e, intercept_e, UV_flow)
    if args['use_intensity']:
        input_intensity_T = tf.expand_dims(get_input_LPLC2_intensity_T(args, 
            weights_intensity, frame_intensity), 1)
        input_T += input_intensity_T
    if args['temporal_filter']:
        filtered_res = get_LPLC2_response(args, tau_1, input_T)
    else:
        filtered_res = input_T
    logits = tf.abs(a) * filtered_res + b
    if args['max_response']:
        loss = cross_entropy_loss(tf.reduce_max(logits), labels[0], step_weights[0])
    else:
        loss = cross_entropy_loss(logits, labels, step_weights)
    probabilities = tf.nn.sigmoid(logits)
    probabilities = tf.math.multiply(probabilities,step_weights)
    probabilities = tf.reduce_sum(probabilities,axis=0)
    predictions = tf.round(probabilities)
    #error_step = 1 - tf.reduce_mean(tf.cast(tf.equal(predictions, labels), 
    #    tf.float32))
    error_step = tf.constant(0,dtype=tf.float32)
    error_trajectory = 1 - tf.cast(tf.equal(predictions, 
        labels[0,:]), tf.float32)
    
    # Apply regularization
    regularization_penalty_we = tf.contrib.layers.apply_regularization(regularizer_we, [weights_e])
    regularization_penalty_a = tf.contrib.layers.apply_regularization(regularizer_a, [a])
    loss = loss + regularization_penalty_we + regularization_penalty_a

    return loss, error_step, error_trajectory, filtered_res, probabilities


# Train and test the LN model for classification
def train_and_test_model_ln_snapshot(args, train_flow_files, train_labels, 
    test_flow_files, test_labels,train_flow_samples,train_labels_samples):
    '''
    Args:
    args: a dictionary that contains problem parameters
    train_flow_files: files for training UV flows
    train_labels: labels (probability of hit, either 0 or 1) for training
    test_flow_files: files for testing UV flows
    test_labels: labels (probability of hit, either 0 or 1) for testing
    train_flow_samples: snapshot training samples  
    train_labels_samples: corresponding labels for snapshot samples

    Returns:
    
    '''
    start = time.time()

    assert(len(train_flow_files) == len(train_flow_samples))

    
    validation_portion = 0.2

    valid_flow_samples = train_flow_samples[int((1-validation_portion)*len(train_flow_samples)):]
    train_flow_samples = train_flow_samples[:int((1-validation_portion)*len(train_flow_samples))]
    valid_labels_samples = train_labels_samples[int((1-validation_portion)*len(train_labels_samples)):]
    train_labels_samples = train_labels_samples[:int((1-validation_portion)*len(train_labels_samples))]

    N_train = len(train_flow_samples)
    N_valid = len(valid_flow_samples)
    N_test = len(test_flow_files)
    

    Q = args['Q']
    K = args['K']
    rotation_symmetry = args['rotation_symmetry']
    flip_symmetry = args['flip_symmetry']
    restrict_nonneg_weight = args['restrict_nonneg_weight']
    lr = args['lr']
    N_epochs = args['N_epochs']
    log_file = args['save_path'] + 'log.txt'
    loss_output = []
    loss_output_test = []
    train_error_output = []
    validation_error_output = []
    validation_loss_output = []
    a_output = []
    b_output = []
    intercept_e_output = []
    lplc2_cells = opsg.get_lplc2_cells_xy_angles(Q)
    
    with open(log_file, 'w') as f:
        f.write('Model setup:\n')
        f.write('----------------------------------------\n')
        f.write('Model is inhibitory2 \n')
        f.write('set_number: {}\n'.format(args['set_number']))
        f.write('data_path: {}\n'.format(args['data_path']))
        f.write('rotational_fraction: {}\n'.format(args['rotational_fraction']))
        f.write('Number of training examples: {}\n'.format(len(train_flow_files)))
        f.write('Number of testing examples: {}\n'.format(len(test_flow_files)))
        f.write('Q: {}\n'.format(args['Q']))
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
        f.write('frames to remove: {}\n'.format(args['frames_to_remove']))
        f.write('save_intermediate_weights: {}\n'.format(args['save_intermediate_weights']))
        f.write('save_steps: {}\n'.format(args['save_steps']))
        f.write('use_step_weight: {}\n'.format(args['use_step_weight']))
        f.write('l1 regularization on excitatory weights: {}\n'.format(args['l1_regu_we']))
        f.write('l2 regularization on excitatory weights: {}\n'.format(args['l2_regu_we']))
        f.write('l1 regularization on inhibitory weights: {}\n'.format(args['l1_regu_wi']))
        f.write('l2 regularization on inhibitory weights: {}\n'.format(args['l2_regu_wi']))
        f.write('l1 regularization on a: {}\n'.format(args['l1_regu_a']))
        f.write('l2 regularization on a: {}\n'.format(args['l2_regu_a']))
        f.write('restrict_nonneg_weight: {}\n'.format(
            args['restrict_nonneg_weight']))
        f.write('restrict_nonpos_intercept: {}\n'.format(
            args['restrict_nonpos_intercept']))
        f.write('rotation_symmetry: {}\n'.format(args['rotation_symmetry']))
        f.write('flip_symmetry: {}\n'.format(args['flip_symmetry']))
        f.write('use_intensity: {}\n'.format(args['use_intensity']))
        f.write('train_a: {}\n'.format(args['train_a']))
        f.write('max_response: {}\n'.format(args['max_response']))
        f.write('temporal_filter: {}\n'.format(args['temporal_filter']))
        f.write('learn_tau: {}\n'.format(args['learn_tau']))
        f.write('tau in standard scale: ' + 
            str(np.around(np.exp(-args['tau']), 3)) + '\n\n')
        
    # inputs
    UV_flow = tf.compat.v1.placeholder(tf.float32, [None,None,Q,K*K,4], 
        name = 'UV_flow')
    step_weights = tf.compat.v1.placeholder(tf.float32, [None,None], name='step_weights')
    labels = tf.compat.v1.placeholder(tf.float32, [None,None], name = 'labels')
    frame_intensity = None
    if args['use_intensity']:
        frame_intensity = tf.compat.v1.placeholder(tf.float32, 
            [None,K*K], name='frame_intensity')

    # variables
    with tf.compat.v1.variable_scope('inhibitory_2'):
        initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1)
        positive_initializer_e = tf.compat.v1.keras.initializers.TruncatedNormal(mean=0.2,stddev=0.1)
        positive_initializer_i = tf.compat.v1.keras.initializers.TruncatedNormal(mean=0.2,stddev=0.1)
        zero_initializer = tf.constant_initializer(0)
        negative_initializer = tf.constant_initializer(-1.)
        if rotation_symmetry:
            if not flip_symmetry:
                weights_e = tf.Variable(initializer([K*K,1]), name='weights_e')
            else:
                if args['fine_tune_weights']:
                    saved_weights_e = np.float32(np.load(args['fine_tune_model_dir'] + 'trained_weights_e.npy'))
                    weights_e_raw = tf.Variable(initial_value=saved_weights_e[:(K+1)//2*K,0], 
                        name='weights_e')
                    weights_e = expand_weight(weights_e_raw, (K+1)//2, K, K%2==0)
                else:
                    weights_e_raw = tf.Variable(positive_initializer_e([(K+1)//2*K,1]), 
                        name='weights_e')
                    weights_e = expand_weight(weights_e_raw, (K+1)//2, K, K%2==0)
        else:
            weights_e = tf.Variable(initializer([K*K,4]), name='weights_e')
        weights_intensity = None
        if args['use_intensity']:
            weights_intensity = tf.Variable(initializer([K*K]), 
                name='weights_intensity')
        if args['fine_tune_intercepts_and_b']:
            saved_intercept_e = np.float32(np.load(args['fine_tune_model_dir'] + 'trained_intercept_e.npy'))
            intercept_e = tf.Variable(initial_value=saved_intercept_e, name='ntercept_e')
        else:
            intercept_e = tf.Variable(initializer([1]), name='intercept_e')
        if args['learn_tau']:
            tau_1 = tf.Variable(initializer([1]), name='tau_1')
        else:
            tau_1 = args['tau']
        #b = tf.Variable(initializer([1]), name='b')
        if args['fine_tune_intercepts_and_b']: 
            saved_b = np.float32(np.load(args['fine_tune_model_dir'] + 'trained_b.npy'))
            b = tf.Variable(initial_value=saved_b, name='b')
        else:
            b = tf.Variable(negative_initializer([1]), name='b')
        if args['train_a']:
            a = tf.Variable(initializer([1]), name='a')
        else:
            a = 1.

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        np.save(args['save_path'] + 'initial_weights_e', weights_e.eval())
            
    # Optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(lr,beta1=0.9)
#     optimizer = tf.compat.v1.train.GradientDescentOptimizer(lr)
#     optimizer = tf.compat.v1.train.MomentumOptimizer(lr,momentum=0.9)
#     optimizer = tf.compat.v1.train.AdagradOptimizer(lr)

    # Regularization
    l1_l2_regu_we = tf.contrib.layers.l1_l2_regularizer(scale_l1=args["l1_regu_we"],scale_l2=args["l2_regu_we"],scope=None)
    l1_l2_regu_a = tf.contrib.layers.l1_l2_regularizer(scale_l1=args["l1_regu_a"],scale_l2=args["l2_regu_a"],scope=None)

    # loss and error function
    loss, error_step, error_trajectory, _, probabilities = \
    loss_error_C_ln(args, weights_e, weights_intensity, intercept_e, UV_flow, frame_intensity, 
                             labels, tau_1, a, b, step_weights,l1_l2_regu_we,l1_l2_regu_a)


    # Optimization
    opt = optimizer.minimize(loss)
    opt_with_clip = opt
    if args['restrict_nonpos_intercept']:
        with tf.control_dependencies([opt_with_clip]): 
            clip_intercept_e = intercept_e.assign(tf.minimum(0., intercept_e))
            opt_with_clip = tf.group(clip_intercept_e)
    
    #config = tf.ConfigProto(inter_op_parallelism_threads=1)
    #with tf.compat.v1.Session(config=config) as sess:
    ind_bz = 0
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        # training
        for epoch in range(1, N_epochs+1):
            train_loss = 0
            train_error_step = 0
            train_error_trajectory = 0
            train_steps = 0
            train_samples = 0
            batch_train_loss = 0
            batch_train_error_step = 0
            batch_train_error_trajectory = 0
            batch_train_steps = 0
            batch_train_samples = 0
            for sample_i in range(1, N_train+1):
                ind_bz = ind_bz+1
                UV_flow_i = train_flow_samples[sample_i-1].copy()
                labels_i = train_labels_samples[sample_i-1]
                step_weights_i = np.ones_like(labels_i)
                
                _, loss_i, error_step_i, error_trajectory_i = \
                sess.run([opt_with_clip, loss, error_step, error_trajectory],
                    {UV_flow:UV_flow_i, labels:labels_i, step_weights: step_weights_i})
                steps_i = 1
                loss_output.append(loss_i)
                if args['train_a']:
                    a_output.append(a.eval())
                else:
                    a_output.append(a)
                b_output.append(b.eval())
                intercept_e_output.append(intercept_e.eval())
                batch_train_loss += loss_i 
                batch_train_error_step += error_step_i * steps_i
                batch_train_steps += steps_i
                train_error_step += error_step_i * steps_i
                train_loss += loss_i 
                train_steps += steps_i
                for ii in range(labels_i.shape[1]):
                    batch_train_error_trajectory += error_trajectory_i[ii]
                    batch_train_samples += 1
                    train_error_trajectory += error_trajectory_i[ii]
                    train_samples += 1
                train_error_output.append(error_trajectory_i)
                if sample_i % args['report_num'] == 0:
                    print('epoch {} step {}/{}: batch train loss (per step) is {:.4g}, batch_train error (per step) is {:.3g} \n \
                        batch train error (per trajectory) is {:.3g}'.format(epoch, sample_i, N_train, 
                            batch_train_loss / batch_train_steps, batch_train_error_step / batch_train_steps, 
                            batch_train_error_trajectory / batch_train_samples))
                    with open(log_file, 'a+') as f:
                        f.write('epoch {} step {}/{}: batch train loss (per step) is {:.4g}, batch_train error (per step) is {:.3g} \n \
                        batch train error (per trajectory) is {:.3g}\n'.format(epoch, sample_i, N_train, 
                            batch_train_loss / batch_train_steps, batch_train_error_step / batch_train_steps, 
                            batch_train_error_trajectory / batch_train_samples))
                    batch_train_loss = 0
                    batch_train_error_step = 0
                    batch_train_error_trajectory = 0
                    batch_train_steps = 0
                    batch_train_samples = 0
                    if args['learn_tau']:
                        current_tau_1 = tau_1.eval()
                    else:
                        current_tau_1 = tau_1
                    print('tau_1:' + str(current_tau_1) + ' b: ' + str(b.eval()) + \
                        ' intercept_e: ' + str(intercept_e.eval()))
                    print('norm of excitatory filter:' + str(tf.norm(weights_e).eval()))
                    with open(log_file, 'a+') as f:
                        f.write('tau_1:' + str(current_tau_1) + ' b: ' + str(b.eval()) + \
                            ' intercept_e: ' + str(intercept_e.eval()) + '\n')
                        f.write('norm of excitatory filter:' + str(tf.norm(weights_e).eval()) + '\n')
                if args["save_intermediate_weights"] and ind_bz % args["save_steps"] == 1:
                    if not os.path.isdir(args['save_path'] + 'checkpoints'):
                        os.mkdir(args['save_path'] + 'checkpoints')
                    temp_dir = args['save_path'] + 'checkpoints/' + str(epoch) + '-' + str(sample_i) + '/'
                    os.mkdir(temp_dir)
                    if rotation_symmetry:
                        #0: right motion, 1: left motion, 2: up motion, 3: down motion
                        rotate_num_list = [0, 2, 1, 3] 
                        weights_e_list = []
                        for i in range(4):
                            weights_e_reshaped = tf.reshape(weights_e,[K,K,1])
                            weights_e_rotated = tf.image.rot90(weights_e_reshaped,
                                rotate_num_list[i])
                            weights_e_reshaped_back = tf.reshape(weights_e_rotated, [K*K,1])
                            weights_e_list.append(weights_e_reshaped_back)
                        weights_e_save = tf.concat(weights_e_list, axis=1)
            
                    weights_e_out = weights_e_save.eval()
                    weights_intensity_out = None
                    if args['use_intensity']:
                        weights_intensity_out = weights_intensity.eval()
                    intercept_e_out = intercept_e.eval()
                    if args['learn_tau']:
                        tau_1_out = tau_1.eval()
                    else:
                        tau_1_out = tau_1
                    b_out = b.eval()
                    if args['train_a']:
                        a_out = a.eval()
                    else:
                        a_out = a
        
                    np.save(temp_dir + 'trained_weights_e', weights_e_out)
                    if args['use_intensity']:
                        np.save(temp_dir + 'trained_weights_intensity', 
                        weights_intensity_out)
                    np.save(temp_dir + 'trained_intercept_e', intercept_e_out)
                    np.save(temp_dir + 'trained_tau_1', tau_1_out)
                    np.save(temp_dir + 'trained_b', b_out)
                    np.save(temp_dir + 'trained_a', a_out)
                    ######################################################
                    ############## do evaluation on validation set #######
                    ######################################################
                    valid_loss = 0
                    valid_error_step = 0
                    valid_steps = 0
                    valid_samples = 0
                    for sample_i in range(1, N_valid+1):
                        UV_flow_i = valid_flow_samples[sample_i-1].copy()
                        labels_i = valid_labels_samples[sample_i-1]
                        step_weights_i = np.ones_like(labels_i)
                
                        loss_i, error_step_i, error_trajectory_i = \
                        sess.run([loss, error_step, error_trajectory],
                            {UV_flow:UV_flow_i, labels:labels_i, step_weights: step_weights_i})
                        steps_i = 1
                        
                        valid_loss += loss_i 
                        valid_error_step += error_step_i * steps_i
                        valid_steps += steps_i
                    #np.save(temp_dir + 'valid_loss', valid_loss / valid_steps)
                    #np.save(temp_dir + 'valid_error', valid_error_step / valid_steps)
                    validation_loss_output.append(valid_loss / valid_steps)
                    validation_error_output.append(valid_error_step / valid_steps)


            print('epoch {}: overall train loss (per step) is {:.4g}, overall train error (per step) is {:.3g}\n \
                overall train error (per trajectory) is {:.3g}'.format(epoch,
                train_loss / train_steps, train_error_step / train_steps,
                train_error_trajectory / train_samples))
            with open(log_file, 'a+') as f:
                f.write('epoch {}: overall train loss (per step) is {:.4g}, overall train error (per step) is {:.3g}\n \
                overall train error (per trajectory) is {:.3g}\n'.format(epoch,
                train_loss / train_steps, train_error_step / train_steps,
                train_error_trajectory / train_samples))
        # testing
        test_loss = 0
        test_error_step = 0
        test_error_trajectory = 0
        test_error_type1 = 0
        test_error_type2 = 0
        test_steps = 0
        test_samples = 0
        test_samples_type1 = 0
        test_samples_type2 = 0
        y_true = []
        y_score = []
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for sample_i, UV_flow_file_i, labels_i, step_weights_i in zip(range(1, N_test+1), 
            test_flow_files, test_labels, args['test_distances']):
            labels_i = np.array(labels_i).flatten()
#             steps_i = 1
#             if not args['use_step_weight']:
#                 step_weights_i = np.ones(steps_i)
            if args['use_intensity']:
                if args['frames_to_remove'] == 0:
                    UV_flow_i,step_weights_i,labels_i =\
                        load_compressed_dataset(args, UV_flow_file_i, labels_i)
                    frame_intensity_i,step_weights_i,labels_i =\
                        load_compressed_dataset(args, args['train_frames'][sample_i-1],labels_i)
                else:
                    UV_flow_i,step_weights_i,labels_i =\
                        load_compressed_dataset(args, UV_flow_file_i,labels_i)[:-args['frames_to_remove']]
                    frame_intensity_i,step_weights_i,labels_i =\
                        load_compressed_dataset(args, args['train_frames'][sample_i-1],labels_i)[:-args['frames_to_remove']]
                loss_i, error_step_i, error_trajectory_i, probabilities_i = \
                sess.run([loss, error_step, error_trajectory, probabilities],
                    {UV_flow:UV_flow_i, labels:labels_i, frame_intensity: frame_intensity_i,
                    step_weights: step_weights_i})
            else:
                if args['frames_to_remove'] == 0:
                    UV_flow_i,step_weights_i,labels_i =\
                        load_compressed_dataset(args, UV_flow_file_i,labels_i)
                else:
                    UV_flow_i,step_weights_i,labels_i =\
                        load_compressed_dataset(args, UV_flow_file_i,labels_i)[:-args['frames_to_remove']]
                loss_i, error_step_i, error_trajectory_i, probabilities_i = \
                sess.run([loss, error_step, error_trajectory, probabilities],
                    {UV_flow:UV_flow_i, labels:labels_i, step_weights: step_weights_i})
            steps_i = 1
            test_loss += loss_i
            test_error_step += error_step_i * steps_i
            test_steps += steps_i
            for ii in range(labels_i.shape[1]):
                y_true.append(labels_i[0][ii])
                y_score.append(probabilities_i[ii])
                test_error_trajectory += error_trajectory_i[ii]
                test_samples += 1
                if labels_i[0][ii] == 0:
                    test_error_type1 += error_trajectory_i[ii]
                    test_samples_type1 += 1
                    if error_trajectory_i[ii] == 0:
                        true_negative += 1
                    elif error_trajectory_i[ii] == 1:
                        false_positive += 1
                else:
                    test_error_type2 += error_trajectory_i[ii]
                    test_samples_type2 += 1
                    if error_trajectory_i[ii] == 0:
                        true_positive += 1
                    elif error_trajectory_i[ii] == 1:
                        false_negative += 1
        auc_roc = sm.roc_auc_score(y_true, y_score)
        auc_pr = sm.average_precision_score(y_true, y_score)
        print('Test loss (per step) is {:.4g}, test error (per step) is {:.3g}, test error (per trajectory) is {:.3g}\n \
            test AUCROC (per trajectory) is {:.3g}, test PRAUC (per trajectory) is {:.3g}'.format(test_loss / test_steps, 
            test_error_step / test_steps, test_error_trajectory / test_samples, auc_roc, auc_pr))
        print('True positive is {}, False positive is {} \n \
               False negative is {}, True negative is {}'.format(true_positive, 
                false_positive, false_negative, true_negative))
        with open(log_file, 'a+') as f:
            f.write('Test loss (per step) is {:.4g}, test error (per step) is {:.3g}, test error (per trajectory) is {:.3g}\n \
            test AUCROC (per trajectory) is {:.3g}, test PRAUC (per trajectory) is {:.3g}\n'.format(test_loss / test_steps, 
            test_error_step / test_steps, test_error_trajectory / test_samples, auc_roc, auc_pr))
            f.write('True positive is {}, False positive is {} \n \
               False negative is {}, True negative is {}\n'.format(true_positive, 
                false_positive, false_negative, true_negative))

        np.save(args['save_path'] + 'auc_roc', auc_roc)
        np.save(args['save_path'] + 'auc_pr', auc_pr)
        np.save(args['save_path'] + 'y_true', y_true)
        np.save(args['save_path'] + 'y_score', y_score)
        
        if rotation_symmetry:
            #0: right motion, 1: left motion, 2: up motion, 3: down motion
            rotate_num_list = [0,2,1,3] 
            weights_e_list = []
            for i in range(4):
                weights_e_reshaped = tf.reshape(weights_e,[K,K,1])
                weights_e_rotated = tf.image.rot90(weights_e_reshaped,
                    rotate_num_list[i])
                weights_e_reshaped_back = tf.reshape(weights_e_rotated, [K*K,1])
                weights_e_list.append(weights_e_reshaped_back)
                
            weights_e = tf.concat(weights_e_list, axis=1)
            
        weights_e_out = weights_e.eval()
        weights_intensity_out = None
        if args['use_intensity']:
            weights_intensity_out = weights_intensity.eval()
        intercept_e_out = intercept_e.eval()
        if args['learn_tau']:
            tau_1_out = tau_1.eval()
        else:
            tau_1_out = tau_1
        b_out = b.eval()
        if args['train_a']:
            a_out = a.eval()
        else:
            a_out = a

        np.save(args['save_path'] + 'trained_weights_e', weights_e_out)
        if args['use_intensity']:
            np.save(args['save_path'] + 'trained_weights_intensity', 
                weights_intensity_out)
        np.save(args['save_path'] + 'trained_intercept_e', intercept_e_out)
        np.save(args['save_path'] + 'trained_tau_1', tau_1_out)
        np.save(args['save_path'] + 'trained_b', b_out)
        np.save(args['save_path'] + 'trained_a', a_out)
        np.save(args['save_path'] + 'loss_output', loss_output)
        np.save(args['save_path'] + 'train_error_output', train_error_output)
        np.save(args['save_path'] + 'a_output', a_output)
        np.save(args['save_path'] + 'b_output', b_output)
        np.save(args['save_path'] + 'intercept_e_output', intercept_e_output)
        np.save(args['save_path'] + 'validation_loss_output', validation_loss_output)
        np.save(args['save_path'] + 'validation_error_output', validation_error_output)

    print('Trained tau_1 is ', np.around(tau_1_out, 3))
    print('Trained b is ', np.around(b_out, 3))
    print('Trained a is ', np.around(a_out, 3))
    print('Trained intercept_e is ', np.around(intercept_e_out, 3))
    with open(log_file, 'a+') as f:
        f.write('Trained tau_1 is ' + str(np.around(tau_1_out, 3)) + '\n')
        f.write('Trained b is ' + str(np.around(b_out,3 )) + '\n')
        f.write('Trained a is ' + str(np.around(a_out, 3)) + '\n')
        f.write('Trained intercept_e is '  + str(np.around(intercept_e_out, 3)) + '\n')

    end = time.time()
    print('Training took {:.0f} second'.format(end - start))
    with open(log_file, 'a+') as f:
        f.write('Training took {:.0f} second\n'.format(end - start))
    
    return weights_intensity_out, weights_e_out,  \
    intercept_e_out, tau_1_out, b_out, a_out


############################################
######### helper functions #################
############################################


def expand_weight(weights, num_row, num_column, is_even):
    weights_reshaped = tf.reshape(weights, [num_row,num_column,1])
    if is_even:
        assert(num_row*2 == num_column)
        weights_flipped = tf.concat([weights_reshaped, 
            tf.reverse(weights_reshaped, axis=[0])], axis=0)
    else:
        assert((num_row*2-1)  == num_column)
        weights_flipped = tf.concat([weights_reshaped, 
            tf.reverse(weights_reshaped[:-1], axis=[0])], axis=0)
    weights_reshaped_back = tf.reshape(weights_flipped, [num_column**2,1])
    return weights_reshaped_back

# general temporal filter
def general_temp_filter(args, tau_1, T):
    '''
    Args:
    args: a dictionary that contains problem parameters
    tau_1: log of the timescale of the filter
    T: total length of the filter
    
    Returns:
    G_n: general temporal filter, len(G_n) = T
    '''
    n = args['n']
    dt = args['dt']

    tau_1 = tf.exp(tau_1)
    #tau_1 = tf.abs(tau_1)
    T = tf.dtypes.cast(T,tf.float32)
    ts = dt * tf.range(T)
    if n == 1.:
        #G_n = (1./tau_1) * tf.exp(-ts/tau_1)
        G_n = tau_1 * tf.exp(-ts*tau_1)
    else:
        G_n = (1./tf.exp(tf.lgamma(n-1.))) * \
        (ts**(n-1.)/(tau_1**n)) * tf.exp(-ts/tau_1)
    G_n = G_n / tf.reduce_sum(G_n)
    
    return G_n

# get filtered signal
def get_filtered_signal(args, tau_1, signal_seq):
    '''
    Args:
    args: a dictionary that contains problem parameters
    tau_1: log of the timescale of the filter
    signal_seq: tensor, signal sequence to be filtered
    
    Returns:
    filtered_sig: filtered signal, single data point
    '''

    if n == 0:
        filtered_sig = signal_seq[-1]
    else:
        T = tf.shape(signal_seq)[-1]
        G_n = general_temp_filter(args, tau_1, T)
    filtered_sig = tf.tensordot(signal_seq, tf.reverse(G_n, [0]), axes=1)
    
    return filtered_sig
   
# get the response of a LPLC2 neuron
def get_LPLC2_response(args, tau_1, input_T):
    '''
    Args:
    args: a dictionary that contains problem parameters
    tau_1: log of the timescale of the filter
    input_T: tensor, input to a LPLC2 neuron, len(input_T_tf) = steps
    
    Returns:
    filtered_res_tf: tensor, filtered response, len(filtered_res_tf) = steps
    '''
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
def cross_entropy_loss(logits, labels, weights):
    '''
    Args:
    logits: predicted variable
    labels: data (probability of hit)
    take_maximu: If true, take maximum, otherwise, take sum
    
    Returns:
    celoss: cross entropy loss (averaged)
    '''
    celoss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    weights = tf.convert_to_tensor(weights,dtype=tf.float32)
    celoss = tf.math.multiply(celoss,weights)
    celoss = tf.reduce_sum(celoss,axis=0)
    celoss = tf.reduce_mean(celoss)
    
    return celoss

# generate file paths for train and test data
def generate_train_test_file_paths(args):
    '''
    Args:
    args: a dictionary that contains problem parameters

    Returns:
    train_flow_files: list of files for UV flows from for training
    train_frame_files: list of files for frame intensities for training
    train_labels: list of labels (probability of hit, either 0 or 1) for training
    test_flow_files: list of files for UV flows from for testing
    test_frame_files: list of files for frame intensities for testing
    test_labels: list of labels (probability of hit, either 0 or 1) for testing
    '''

    start = time.time()
    print('Generating train and test file paths')

    set_number = args['set_number']
    data_path = args['data_path']

    file_types = ['hit', 'miss', 'retreat', 'rotation']
    
    train_flow_files = []
    train_frame_files = []
    train_distances = []
    train_labels = []
    test_flow_files = []
    test_frame_files = []
    test_distances = []
    test_labels = []
    rotational_train_flow_files = []
    rotational_train_frame_files = []
    rotational_train_distances = []
    rotational_train_labels = []
    rotational_test_flow_files = []
    rotational_test_frame_files = []
    rotational_test_distances = []
    rotational_test_labels = []
    for file_type in file_types:
        # gather training paths and labels
        train_path = data_path + 'set_{}/training/'.format(set_number) + \
        file_type + '/UV_flow_samples/'
        if os.path.isdir(train_path):
            train_files = glob.glob(train_path + '*.npy')
            for train_flow_file in train_files:
                train_frame_file = train_flow_file.replace('UV_flow_samples',
                    'frames_samples_cg')
                train_frame_file = train_frame_file.replace('UV_flow_sample',
                    'frames_sample_cg')
                train_distance_file = train_flow_file.replace('training',
                    'other_info')
                train_distance_file = train_distance_file.replace('UV_flow_samples',
                    'distances')
                train_distance_file = train_distance_file.replace('UV_flow_sample',
                    'distance')
                train_flow =  np.load(train_flow_file,allow_pickle=True)
                train_steps = train_flow.shape[0]
                if file_type == 'hit':
                    train_label = np.ones(10)
                else:
                    train_label = np.zeros(10)
                if file_type == 'rotation':
                    rotational_train_flow_files.append(train_flow_file)
                    rotational_train_frame_files.append(train_frame_file)
                    rotational_train_labels.append(train_label)
                    rotational_train_distances.append(np.ones(train_steps))
                else:
                    train_flow_files.append(train_flow_file)
                    train_frame_files.append(train_frame_file)
                    train_labels.append(train_label)
                    distances = np.load(train_distance_file,allow_pickle=True)
                    distances_inverse = [1/item for sublist in distances for item in sublist]
                    train_distances.append(np.asarray(distances_inverse))

        # gather testing paths and labels
        test_path = data_path + 'set_{}/testing/'.format(set_number) + \
        file_type + '/UV_flow_samples/'
        if os.path.isdir(test_path):
            test_files = glob.glob(test_path + '*.npy')
            for test_flow_file in test_files:
                test_frame_file = test_flow_file.replace('UV_flow_samples',
                    'frames_samples_cg')
                test_frame_file = test_frame_file.replace('UV_flow_sample',
                    'frames_sample_cg')
                test_distance_file = test_flow_file.replace('testing',
                    'other_info')
                test_distance_file = test_distance_file.replace('UV_flow_samples',
                    'distances')
                test_distance_file = test_distance_file.replace('UV_flow_sample',
                    'distance')
                
                test_flow =  np.load(test_flow_file,allow_pickle=True)
                if file_type == 'hit':
                    test_label = np.ones(10)
                else:
                    test_label = np.zeros(10)
                if file_type == 'rotation':
                    rotational_test_flow_files.append(test_flow_file)
                    rotational_test_frame_files.append(test_frame_file)
                    rotational_test_labels.append(test_label)
                    rotational_test_distances.append(np.array([1]))
                else:
                    test_flow_files.append(test_flow_file)
                    test_frame_files.append(test_frame_file)
                    test_labels.append(test_label)
                    distances = np.load(test_distance_file,allow_pickle=True)
                    distances_inverse = [1/item for sublist in distances for item in sublist]
                    test_distances.append(np.asarray(distances_inverse))
    # subsample rotational files
    rotational_train_index = random.sample(range(len(rotational_train_labels)), 
        int(args['rotational_fraction']*len(rotational_train_labels)))
    for index in rotational_train_index:
        train_flow_files.append(rotational_train_flow_files[index])
        train_frame_files.append(rotational_train_frame_files[index])
        train_labels.append(rotational_train_labels[index])
        train_distances.append(rotational_train_distances[index])

    rotational_test_index = random.sample(range(len(rotational_test_labels)), 
        int(args['rotational_fraction']*len(rotational_test_labels)))
    for index in rotational_test_index:
        test_flow_files.append(rotational_test_flow_files[index])
        test_frame_files.append(rotational_test_frame_files[index])
        test_labels.append(rotational_test_labels[index])
        test_distances.append(rotational_test_distances[index])

    # shuffle data
    temp = list(zip(train_flow_files, train_frame_files, train_labels, train_distances)) 
    random.shuffle(temp) 
    train_flow_files, train_frame_files, train_labels, train_distances = zip(*temp) 

    temp = list(zip(test_flow_files, test_frame_files, test_labels, test_distances)) 
    random.shuffle(temp) 
    test_flow_files, test_frame_files, test_labels, test_distances = zip(*temp) 
    
    # group samples
    M = args['M']
    train_flow_files = get_grouped_list(train_flow_files,M)
    train_frame_files = get_grouped_list(train_frame_files,M)
    train_labels = get_grouped_list(train_labels,M)
    train_distances = get_grouped_list(train_distances,M)
    
    test_flow_files = get_grouped_list(test_flow_files,M)
    test_frame_files = get_grouped_list(test_frame_files,M)
    test_labels = get_grouped_list(test_labels,M)
    test_distances = get_grouped_list(test_distances,M)

    end = time.time()
    print('Generated {} train samples and {} test samples'.format(len(train_flow_files), 
        len(test_flow_files)))
    print('Data generation took {:.0f} second'.format(end - start))
    
    print(len(train_flow_files))
    print(len(train_flow_files[0]))
    return train_flow_files, train_frame_files, train_labels, test_flow_files, \
    test_frame_files, test_labels, train_distances, test_distances


# generate file paths for train data
def generate_train_data(args):
    '''
    Args:
    args: a dictionary that contains problem parameters
    NNs: # samples in one list

    Returns:
    train_flow_files: list of files for UV flows from for training
    train_frame_files: list of files for frame intensities for training
    train_labels: list of labels (probability of hit, either 0 or 1) for training
    '''

    set_number = args['set_number']
    data_path = args['data_path']
    NNs = args['NNs']

    file_types = ['hit', 'miss', 'retreat', 'rotation']
    
    train_flow_files = []
    train_frame_files = []
    train_distances = []
    train_labels = []
    
    rotational_train_flow_files = []
    rotational_train_frame_files = []
    rotational_train_distances = []
    rotational_train_labels = []
    
    for file_type in file_types:
        # gather training paths and labels
        train_path = data_path + 'set_{}/training/'.format(set_number) + \
        file_type + '/UV_flow_samples/'
        if os.path.isdir(train_path):
            train_files = glob.glob(train_path + '*.npy')
            for train_flow_file in train_files:
                train_frame_file = train_flow_file.replace('UV_flow_samples',
                    'frames_samples_cg')
                train_frame_file = train_frame_file.replace('UV_flow_sample',
                    'frames_sample_cg')
                train_distance_file = train_flow_file.replace('training',
                    'other_info')
                train_distance_file = train_distance_file.replace('UV_flow_samples',
                    'distances')
                train_distance_file = train_distance_file.replace('UV_flow_sample',
                    'distance')
                train_flow =  np.load(train_flow_file,allow_pickle=True)
                train_steps = train_flow.shape[0]
                if file_type == 'hit':
                    train_label = np.ones(NNs)
                else:
                    train_label = np.zeros(NNs)
                if file_type == 'rotation':
                    rotational_train_flow_files.append(train_flow_file)
                    rotational_train_frame_files.append(train_frame_file)
                    rotational_train_labels.append(train_label)
                    rotational_train_distances.append(np.ones(train_steps))
                else:
                    train_flow_files.append(train_flow_file)
                    train_frame_files.append(train_frame_file)
                    train_labels.append(train_label)
                    distances = np.load(train_distance_file,allow_pickle=True)
                    distances_inverse = [1/item for sublist in distances for item in sublist]
                    train_distances.append(np.asarray(distances_inverse))

    # subsample rotational files
    rotational_train_index = random.sample(range(len(rotational_train_labels)), 
        int(args['rotational_fraction']*len(rotational_train_labels)))
    for index in rotational_train_index:
        train_flow_files.append(rotational_train_flow_files[index])
        train_frame_files.append(rotational_train_frame_files[index])
        train_labels.append(rotational_train_labels[index])
        train_distances.append(rotational_train_distances[index])

    # shuffle data
    temp = list(zip(train_flow_files, train_frame_files, train_labels, train_distances)) 
    random.shuffle(temp) 
    train_flow_files, train_frame_files, train_labels, train_distances = zip(*temp) 
    
    # group samples
    M = args['M']
    train_flow_files = get_grouped_list(train_flow_files,M)
    train_frame_files = get_grouped_list(train_frame_files,M)
    train_labels = get_grouped_list(train_labels,M)
    train_distances = get_grouped_list(train_distances,M)
    
    # sampling
    train_flow_samples = []
    train_frame_samples = []
    train_labels_samples = []
    for file_path_flow,file_path_frame,labels in zip(train_flow_files,train_frame_files,train_labels):
        steps_list_flow = []
        steps_list_frame = []
        for m in range(M):
            data_flow = np.load(file_path_flow[m],allow_pickle=True)
            data_frame = np.load(file_path_frame[m],allow_pickle=True)
            for dd in range(len(data_flow)): # len(data) indicates number of data samples in one data sample list
                i = random.randint(0,len(data_flow[dd])-1) # randomly select one snapshot
                step_flow = []
                step_frame = []
                for j in range(len(data_flow[dd][i])): # len(data[dd][i]) indicates Q
                    if len(data_flow[dd][i][j].shape) == 1:
                        step_flow.append(np.zeros((args['K']**2,4)))
                        step_frame.append(np.zeros((args['K']**2)))
                    else:
                        step_flow.append(data_flow[dd][i][j])
                        step_frame.append(data_frame[dd][i][j])
                steps_list_flow.append(np.stack([np.stack(step_flow)]))
                steps_list_frame.append(np.stack([np.stack(step_frame)]))
        steps_list_extended_flow = []
        steps_list_extended_frame = []
        weight_list_extended = []
        label_list_extended = []
        labels = np.array(labels).flatten()
        for n in range(M*NNs):
            steps_extended_flow,weight_extended,label_extended = get_extended_array(steps_list_flow[n],labels[n],1)
            steps_extended_frame,weight_extended,label_extended = get_extended_array(steps_list_frame[n],labels[n],1)
            steps_list_extended_flow.append(steps_extended_flow)
            steps_list_extended_frame.append(steps_extended_frame)
            weight_list_extended.append(weight_extended)
            label_list_extended.append(label_extended)
        steps_list_extended_flow = np.swapaxes(np.stack(steps_list_extended_flow),0,1)
        steps_list_extended_frame = np.swapaxes(np.stack(steps_list_extended_frame),0,1)
        weight_list_extended = np.swapaxes(np.stack(weight_list_extended),0,1)
        label_list_extended = np.swapaxes(np.stack(label_list_extended),0,1)
        
        train_flow_samples.append(steps_list_extended_flow)
        train_frame_samples.append(steps_list_extended_frame)
        train_labels_samples.append(label_list_extended)
        
    np.save(data_path + 'set_{}/training/'.format(set_number)+'train_flow_samples',train_flow_samples)
    np.save(data_path + 'set_{}/training/'.format(set_number)+'train_frame_samples',train_frame_samples)
    np.save(data_path + 'set_{}/training/'.format(set_number)+'train_labels_samples',train_labels_samples)


def load_compressed_dataset(args, file_path, labels):
    N = len(file_path)
    T = 0
    steps_list = []
    for n in range(N):
        data = np.load(file_path[n],allow_pickle=True)
        for dd in range(len(data)):
            steps = []
            T = np.maximum(T,len(data[dd]))
            for i in range(len(data[dd])):
                step = []
                for j in range(args['Q']):
                    if len(data[dd][i][j].shape) == 1:
                        step.append(np.zeros((args['K']**2,4)))
                    else:
                        step.append(data[dd][i][j])
                steps.append(np.stack(step))
            steps_list.append(np.stack(steps))
    steps_list_extended = []
    weight_list_extended = []
    label_list_extended = []
    for n in range(N*len(data)):
        steps_extended,weight_extended,label_extended = get_extended_array(steps_list[n],labels[n],T)
        steps_list_extended.append(steps_extended)
        weight_list_extended.append(weight_extended)
        label_list_extended.append(label_extended)
    steps_list_extended = np.swapaxes(np.stack(steps_list_extended),0,1)
    weight_list_extended = np.swapaxes(np.stack(weight_list_extended),0,1)
    label_list_extended = np.swapaxes(np.stack(label_list_extended),0,1)
    
    return steps_list_extended,weight_list_extended,label_list_extended


def get_grouped_list(input_list,M):
    '''
    Args:
    input_list: # of sample data files in the folder, each sample data here is a list
    M: # of sample data list in one batch in mini-batch training
    
    Returns:
    output_list: a list of mini batches, len(output_list) =  # of mini batches in one training epoch
    '''
    output_list = []
    N = np.int(len(input_list)/M)
    for n in range(N):
        tem_list = input_list[n*M:(n+1)*M]
        output_list.append(tem_list)
        
    return output_list


def get_extended_array(input_array,label,T):
    T0 = input_array.shape[0]
    output_array = np.zeros((T,)+input_array.shape[1:])
    output_array[:T0] = input_array
    output_weight = np.zeros(T)
    output_weight[:T0] = 1./T0
    output_label = np.zeros(T)
    output_label[:] = label
    
    return output_array,output_weight,output_label
    
    
##########################################
######## Intensity Input #################
##########################################


# Input to a LPLC2 neuron at a specific time point from intensity
def get_input_LPLC2_intensity(args, weights_intensity, frame_intensity_t):
    '''
    Args:
    args: a dictionary that contains problem parameters
    weights_intensity: tensor, weights for intensities, K*K
    frame_intensity_t: tensor, frame intensity at time point t, Q by K*K
    
    Returns:
    input_intensity_t: input from intensity to a LPLC2 neuron at time t
    '''
    input_intensity_t = tf.tensordot(frame_intensity_t, 
        weights_intensity, axes=[[1],[0]])

    return input_intensity_t


# Input to a LPLC2 neuron for a whole sequential intensity signals
def get_input_LPLC2_intensity_T(args, weights_intensity, frame_intensity):
    '''
    Args:
    args: a dictionary that contains problem parameters
    weights_intensity: tensor, weights for intensities, K*K
    frame_intensity_t: tensor, frame intensity at time point t, Q by K*K
    
    Returns:
    input_intensity_T: input to a LPLC2 neuron from intensity, length = steps
    '''
    input_intensity_T = tf.map_fn(lambda x:\
                        get_input_LPLC2_intensity(args, weights_intensity, x), 
                        frame_intensity)

    return input_intensity_T
    
    
    