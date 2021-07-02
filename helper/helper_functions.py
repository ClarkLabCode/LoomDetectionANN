#!/usr/bin/env python

'''
Helper functions.
'''

import sys
sys.path.append('../stimulus_core/')
sys.path.append('../models_core/')
sys.path.append('../helper/')

import os
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
import glob
from joblib import Parallel, delayed
import seaborn as sns
from scipy.spatial.transform import Rotation as R3d
from scipy.ndimage import gaussian_filter
import imageio
import time
import random

import dynamics_3d as dn3d
import optical_signal as opsg
import flow_field as flfd
import samples_generation_multi_units as smgnmu
import get_Klapoetke_stimuli_experiment as gKse
import predefined_weights as pdwt
import lplc2_models as lplc2


tf.compat.v1.disable_eager_execution()

# Plot the one trained frame intensity weights
def plot_intensity_weights(input_weights, mask_d, colormap, filename):
    '''
    Args:
    input_weights: steps by K*K
    mask_d: disk mask
    colormap: color map of the image
    filename: filename to save
    '''
    K = int(np.sqrt(len(input_weights)))
    fig = plt.figure(figsize=(20, 5))
    
    weights_ = input_weights.reshape((K, K))
    weights_[mask_d] = 0.
    extreme = np.amax(np.abs(input_weights))
    color_norm = mpl.colors.Normalize(vmin=-extreme, vmax=extreme)
    plt.imshow(weights_, norm=color_norm, cmap=colormap)
    plt.colorbar()
    plt.title('intensity')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    fig.savefig(filename, bbox_inches='tight')

# Plot the four trained UV flow weights
def plot_flow_weights(input_weights, mask_d, colormap, filename):
    '''
    Args:
    input_weights: steps by K*K by 4
    mask_d: disk mask
    colormap: color map of the image
    filename: filename to save
    '''
    K = int(np.sqrt(len(input_weights)))
    fig = plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    weights_ = input_weights[:, 0].reshape((K, K))
    weights_[mask_d] = 0.

    extreme = np.amax(np.abs(weights_))
    color_norm = mpl.colors.Normalize(vmin=-extreme, vmax=extreme)

    plt.imshow(weights_, norm=color_norm, cmap=colormap)
    plt.colorbar()
    plt.title('rightward motion')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(1, 4, 2)
    weights_ = input_weights[:, 1].reshape((K, K))
    weights_[mask_d] = 0.
    plt.imshow(weights_, norm=color_norm, cmap=colormap)
    plt.colorbar()
    plt.title('leftward motion')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(1, 4, 3)
    weights_ = input_weights[:, 2].reshape((K, K))
    weights_[mask_d] = 0.
    plt.imshow(weights_, norm=color_norm, cmap=colormap)
    plt.colorbar()
    plt.title('upward motion')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(1, 4, 4)
    weights_ = input_weights[:, 3].reshape((K, K))
    weights_[mask_d] = 0.
    plt.imshow(weights_, norm=color_norm, cmap=colormap)
    plt.colorbar()
    plt.title('downward motion')
    plt.xticks([])
    plt.yticks([])
    
    plt.show()
    
    fig.savefig(filename, bbox_inches='tight')

    
def plot_predefined_weights(theta_dt, K, L, filename):
    
    leftup_corners = flfd.get_leftup_corners(K, L, 0)
    
    # Excitatory neurons
    weights_e = pdwt.get_all_weights_e(leftup_corners, theta_dt, K, L)
    
    weights_cf_right_e = pdwt.set_weights_on_frame(weights_e[:, 0], leftup_corners, K, L)
    weights_cf_left_e = pdwt.set_weights_on_frame(weights_e[:, 1], leftup_corners, K, L)
    weights_cf_up_e = pdwt.set_weights_on_frame(weights_e[:, 2], leftup_corners, K, L)
    weights_cf_down_e = pdwt.set_weights_on_frame(weights_e[:, 3], leftup_corners, K, L)
    
    # Inhibitory neurons
    weights_i = pdwt.get_all_weights_i(leftup_corners, theta_dt, K, L)
    
    weights_cf_right_i = pdwt.set_weights_on_frame(weights_i[:, 0], leftup_corners, K, L)
    weights_cf_left_i = pdwt.set_weights_on_frame(weights_i[:, 1], leftup_corners, K, L)
    weights_cf_up_i = pdwt.set_weights_on_frame(weights_i[:, 2], leftup_corners, K, L)
    weights_cf_down_i = pdwt.set_weights_on_frame(weights_i[:, 3], leftup_corners, K, L)
    
    fig = plt.figure(figsize=(20, 10))
    plt.subplot(2, 4, 1)
    plt.imshow(weights_cf_right_e, cmap='gray_r')
    plt.title('rightward motion detector')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 4, 2)
    plt.imshow(weights_cf_left_e, cmap='gray_r')
    plt.title('leftward motion detector')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 4, 3)
    plt.imshow(weights_cf_up_e, cmap='gray_r')
    plt.title('upward motion detector')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 4, 4)
    plt.imshow(weights_cf_down_e, cmap='gray_r')
    plt.title('downward motion detector')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(2, 4, 5)
    plt.imshow(weights_cf_right_i, cmap='gray_r')
    plt.title('rightward motion detector')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 4, 6)
    plt.imshow(weights_cf_left_i, cmap='gray_r')
    plt.title('leftward motion detector')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 4, 7)
    plt.imshow(weights_cf_up_i, cmap='gray_r')
    plt.title('upward motion detector')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 4, 8)
    plt.imshow(weights_cf_down_i, cmap='gray_r')
    plt.title('downward motion detector')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    fig.savefig(filename, bbox_inches='tight')
    
    
def make_set_folder(set_number, savepath):
    os.makedirs(savepath+'set_{}/'.format(set_number)+'training/hit/intensities_samples_cg')
    os.makedirs(savepath+'set_{}/'.format(set_number)+'training/hit/UV_flow_samples')
    os.makedirs(savepath+'set_{}/'.format(set_number)+'training/miss/intensities_samples_cg')
    os.makedirs(savepath+'set_{}/'.format(set_number)+'training/miss/UV_flow_samples')
    os.makedirs(savepath+'set_{}/'.format(set_number)+'training/retreat/intensities_samples_cg')
    os.makedirs(savepath+'set_{}/'.format(set_number)+'training/retreat/UV_flow_samples')
    os.makedirs(savepath+'set_{}/'.format(set_number)+'training/rotation/intensities_samples_cg')
    os.makedirs(savepath+'set_{}/'.format(set_number)+'training/rotation/UV_flow_samples')
    
    os.makedirs(savepath+'set_{}/'.format(set_number)+'testing/hit/intensities_samples_cg')
    os.makedirs(savepath+'set_{}/'.format(set_number)+'testing/hit/UV_flow_samples')
    os.makedirs(savepath+'set_{}/'.format(set_number)+'testing/miss/intensities_samples_cg')
    os.makedirs(savepath+'set_{}/'.format(set_number)+'testing/miss/UV_flow_samples')
    os.makedirs(savepath+'set_{}/'.format(set_number)+'testing/retreat/intensities_samples_cg')
    os.makedirs(savepath+'set_{}/'.format(set_number)+'testing/retreat/UV_flow_samples')
    os.makedirs(savepath+'set_{}/'.format(set_number)+'testing/rotation/intensities_samples_cg')
    os.makedirs(savepath+'set_{}/'.format(set_number)+'testing/rotation/UV_flow_samples')
    
    os.makedirs(savepath+'set_{}/'.format(set_number)+'other_info/hit/trajectories')
    os.makedirs(savepath+'set_{}/'.format(set_number)+'other_info/hit/distances')
    os.makedirs(savepath+'set_{}/'.format(set_number)+'other_info/hit/distances/sample')
    os.makedirs(savepath+'set_{}/'.format(set_number)+'other_info/miss/trajectories')
    os.makedirs(savepath+'set_{}/'.format(set_number)+'other_info/miss/distances')
    os.makedirs(savepath+'set_{}/'.format(set_number)+'other_info/miss/distances/sample')
    os.makedirs(savepath+'set_{}/'.format(set_number)+'other_info/retreat/trajectories')
    os.makedirs(savepath+'set_{}/'.format(set_number)+'other_info/retreat/distances')
    os.makedirs(savepath+'set_{}/'.format(set_number)+'other_info/retreat/distances/sample')
    os.makedirs(savepath+'set_{}/'.format(set_number)+'other_info/rotation/trajectories')


# Plot the distances versus test error of a pre-trained model
def plot_distance_error(args, pretrained_model_dir, model_number):
    '''
    Args:
    args: a dictionary that contains problem parameters
    pretrained_model_dir: a directory that contains pre-trained model parameters
    model_number: 1 or 2 or 3
    '''
    set_number = args["set_number"]
    dt = args["dt"]
    datapath = args["datapath"]
    K = args['K']

    file_types = ['loom_hit', 'loom_nohit']
    #file_types = ['loom_hit']
    test_flows = []
    test_intensities = []
    test_labels = []
    test_distances = []
    for file_type in file_types:
        # gather testing paths and labels
        test_path = datapath + 'set_{}/testing/'.format(set_number) + file_type + '/UV_flow_samples/'
        if os.path.isdir(test_path):
            test_files = glob.glob(test_path + '*.npy')
            for test_flow_file in test_files:
                test_intensity_file = test_flow_file.replace('UV_flow_samples', 'intensities_samples_cg')
                test_intensity_file = test_intensity_file.replace('UV_flow_sample', 'intensities_sample_cg')
                test_distance_file = test_flow_file.replace('testing', 'other_info/distances')
                test_distance_file = test_distance_file.replace('UV_flow_samples/UV_flow_sample', 'distance')
                test_flow =  np.load(test_flow_file)
                test_intensity = np.load(test_intensity_file)
                test_distance = np.load(test_distance_file)
                test_steps = test_flow.shape[0]
                if file_type == 'loom_hit':
                    test_label = np.ones(test_steps)
                else:
                    test_label = np.zeros(test_steps)
                test_flows.append(test_flow)
                test_intensities.append(test_intensity)
                test_distances.append(test_distance[-1])
                test_labels.append(test_label)
    N_test = len(test_flows)

    UV_flow = tf.compat.v1.placeholder(tf.float32, [None, K*K, 4], name='UV_flow')
    labels = tf.compat.v1.placeholder(tf.float32, [None], name='labels')
    intensity = None
    if args['use_intensity']:
        intensity = tf.compat.v1.placeholder(tf.float32, 
            [None, K*K], name='intensity')

    a = np.load(pretrained_model_dir + "trained_a.npy")
    b = np.load(pretrained_model_dir + "trained_b.npy")
    intercept_e = np.load(pretrained_model_dir + "trained_intercept_e.npy")
    tau_1 = np.load(pretrained_model_dir + "trained_tau_1.npy")
    weights_e = np.load(pretrained_model_dir + "trained_weights_e.npy")
    #print(a, b, tau_1, intercept_e)

    if args["symmetrize"]:
        weights_e = weights_e[:, 0]
    if model_number == 2 or model_number == 3:
        weights_i = np.load(pretrained_model_dir + "trained_weights_i.npy")
        if args["symmetrize"]:
            weights_i = weights_i[:, 0]
    if model_number == 3:
        intercept_i = np.load(pretrained_model_dir + "trained_intercept_i.npy")
        #print(intercept_i)
    weights_intensity = None
    if args["use_intensity"]:
        weights_intensity = np.load(pretrained_model_dir + "trained_weights_intensity.npy")

    if model_number == 1:
        loss, error_step, error_trajectory, filtered_res, _ = lplc2.loss_error_C_excitatory_only(args, 
        weights_e, weights_intensity, intercept_e, UV_flow, intensity, 
        labels, tau_1, a, b)
    elif model_number == 2:
        loss, error_step, error_trajectory, filtered_res, _  = lplc2.loss_error_C_inhibitory1(args, 
            weights_e, weights_i, weights_intensity, intercept_e, UV_flow, 
            intensity, labels, tau_1, a, b)
    elif model_number == 3:
        loss, error_step, error_trajectory, filtered_res, _ = lplc2.loss_error_C_inhibitory2(args, 
            weights_e, weights_i, weights_intensity, intercept_e, intercept_i, 
            UV_flow, intensity, labels, tau_1, a, b)

    error_list = []
    error_loomnohit = []
    predicted_steps = []
    distance = []
    with tf.compat.v1.Session() as sess:
        for sample_i in range(N_test):
            if args['use_intensity']:
                error_i, response_i = sess.run([error_trajectory, filtered_res], 
                    {UV_flow:test_flows[sample_i], labels:test_labels[sample_i], 
                    intensity: test_intensities[sample_i]})
            else:
                error_i, response_i = sess.run([error_trajectory, filtered_res], 
                    {UV_flow:test_flows[sample_i], labels:test_labels[sample_i]})
            if test_labels[sample_i][0] == 1:
                error_list.append(error_i)
                distance.append(test_distances[sample_i])
                logits_i = np.abs(a) * response_i + b
                predictions_i = np.round(sigmoid_array(logits_i))
                if np.amax(predictions_i) > 0:
                    predicted_steps.append(np.argmax(predictions_i) + 1)
            else:
                error_loomnohit.append(error_i)
    print("{}/{} = {} loom_hit samples are predicted wrong".format(sum(error_list), 
        len(error_list), str(np.mean(error_list))))
    if len(error_loomnohit) > 0:
        print("{}/{} = {} loom_nohit samples are predicted wrong".format(sum(error_loomnohit), 
            len(error_loomnohit), str(np.mean(error_loomnohit))))
    print("Average time step for escape is " + str(np.around(np.mean(predicted_steps), 3)))
    with open(pretrained_model_dir + "investigate_output.txt", 'w') as f:
        f.write("{}/{} = {} loom_hit samples are predicted wrong\n".format(sum(error_list), 
        len(error_list), str(np.around(np.mean(error_list), 3))))
        if len(error_loomnohit) > 0:
            f.write("{}/{} = {} loom_nohit samples are predicted wrong\n".format(sum(error_loomnohit), 
            len(error_loomnohit), str(np.around(np.mean(error_loomnohit), 3))))
        f.write("Average time step for escape is " + str(np.around(np.mean(predicted_steps), 3)))
    correct = ["correct" if error < 0.5 else "incorrect" for error in error_list]
    fig = plt.figure(figsize=(5, 5))
    sns.boxplot(x=correct, y=distance).set(ylabel='Distance')
    fig.savefig(pretrained_model_dir + "distance_error.pdf")


def sigmoid_array(x): 
    return 1 / (1 + np.exp(-x))


# temporal filter
def get_tem_filtered_traj(traj, sigma):
    traj_filtered = np.zeros_like(traj)
    P = traj.shape[1]
    dims = traj.shape[2]
    assert dims == 3, 'The dimension should be 3!'
    for p in range(P):
        for i in range(dims):
            traj_filtered[:, p, i] = gaussian_filter(traj[:, p, i], sigma=sigma, mode='nearest')
        
    return traj_filtered


# generate file paths for train and test data
def generate_train_test_file_paths(args):
    '''
    Args:
    args: a dictionary that contains problem parameters

    Returns:
    train_flow_files: list of files for UV flows from for training
    train_intensity_files: list of files for frame intensities for training
    train_labels: list of labels (probability of hit, either 0 or 1) for training
    test_flow_files: list of files for UV flows from for testing
    test_intensity_files: list of files for frame intensities for testing
    test_labels: list of labels (probability of hit, either 0 or 1) for testing
    '''
    start = time.time()
    print('Generating train and test file paths')

    set_number = args['set_number']
    data_path = args['data_path']

    file_types = ['hit', 'miss', 'retreat', 'rotation']
    
    train_flow_files = []
    train_intensity_files = []
    train_distances = []
    train_labels = []
    test_flow_files = []
    test_intensity_files = []
    test_distances = []
    test_labels = []
    rotational_train_flow_files = []
    rotational_train_intensity_files = []
    rotational_train_distances = []
    rotational_train_labels = []
    rotational_test_flow_files = []
    rotational_test_intensity_files = []
    rotational_test_distances = []
    rotational_test_labels = []
    for file_type in file_types:
        # gather training paths and labels
        train_path = data_path + 'set_{}/training/'.format(set_number) + \
        file_type + '/UV_flow_samples/'
        if os.path.isdir(train_path):
            train_files = glob.glob(train_path + '*.npy')
            for train_flow_file in train_files:
                train_intensity_file = train_flow_file.replace('UV_flow_samples', 
                    'intensities_samples_cg')
                train_intensity_file = train_intensity_file.replace('UV_flow_sample', 
                    'intensities_sample_cg')
                train_distance_file = train_flow_file.replace('training', 
                    'other_info')
                train_distance_file = train_distance_file.replace('UV_flow_samples', 
                    'distances')
                train_distance_file = train_distance_file.replace('UV_flow_sample', 
                    'distance')
                train_flow =  np.load(train_flow_file, allow_pickle=True)
                train_steps = train_flow.shape[0]
                if file_type == 'hit':
                    train_label = np.ones(10)
                else:
                    train_label = np.zeros(10)
                if file_type == 'rotation':
                    rotational_train_flow_files.append(train_flow_file)
                    rotational_train_intensity_files.append(train_intensity_file)
                    rotational_train_labels.append(train_label)
                    rotational_train_distances.append(np.ones(train_steps))
                else:
                    train_flow_files.append(train_flow_file)
                    train_intensity_files.append(train_intensity_file)
                    train_labels.append(train_label)
                    distances = np.load(train_distance_file, allow_pickle=True)
                    distances_inverse = [1/item for sublist in distances for item in sublist]
                    train_distances.append(np.asarray(distances_inverse))

        # gather testing paths and labels
        test_path = data_path + 'set_{}/testing/'.format(set_number) + \
        file_type + '/UV_flow_samples/'
        if os.path.isdir(test_path):
            test_files = glob.glob(test_path + '*.npy')
            for test_flow_file in test_files:
                test_intensity_file = test_flow_file.replace('UV_flow_samples', 
                    'intensities_samples_cg')
                test_intensity_file = test_intensity_file.replace('UV_flow_sample', 
                    'intensities_sample_cg')
                test_distance_file = test_flow_file.replace('testing', 
                    'other_info')
                test_distance_file = test_distance_file.replace('UV_flow_samples', 
                    'distances')
                test_distance_file = test_distance_file.replace('UV_flow_sample', 
                    'distance')
                
                test_flow =  np.load(test_flow_file, allow_pickle=True)
                if file_type == 'hit':
                    test_label = np.ones(10)
                else:
                    test_label = np.zeros(10)
                if file_type == 'rotation':
                    rotational_test_flow_files.append(test_flow_file)
                    rotational_test_intensity_files.append(test_intensity_file)
                    rotational_test_labels.append(test_label)
                    rotational_test_distances.append(np.array([1]))
                else:
                    test_flow_files.append(test_flow_file)
                    test_intensity_files.append(test_intensity_file)
                    test_labels.append(test_label)
                    distances = np.load(test_distance_file, allow_pickle=True)
                    distances_inverse = [1/item for sublist in distances for item in sublist]
                    test_distances.append(np.asarray(distances_inverse))
    # subsample rotational files
    rotational_train_index = random.sample(range(len(rotational_train_labels)), 
        int(args['rotational_fraction']*len(rotational_train_labels)))
    for index in rotational_train_index:
        train_flow_files.append(rotational_train_flow_files[index])
        train_intensity_files.append(rotational_train_intensity_files[index])
        train_labels.append(rotational_train_labels[index])
        train_distances.append(rotational_train_distances[index])

    rotational_test_index = random.sample(range(len(rotational_test_labels)), 
        int(args['rotational_fraction']*len(rotational_test_labels)))
    for index in rotational_test_index:
        test_flow_files.append(rotational_test_flow_files[index])
        test_intensity_files.append(rotational_test_intensity_files[index])
        test_labels.append(rotational_test_labels[index])
        test_distances.append(rotational_test_distances[index])

    # shuffle data
    temp = list(zip(train_flow_files, train_intensity_files, train_labels, train_distances)) 
    random.shuffle(temp) 
    train_flow_files, train_intensity_files, train_labels, train_distances = zip(*temp) 

    temp = list(zip(test_flow_files, test_intensity_files, test_labels, test_distances)) 
    random.shuffle(temp) 
    test_flow_files, test_intensity_files, test_labels, test_distances = zip(*temp) 
    
    # group samples
    S = args['S']
    train_flow_files = get_grouped_list(train_flow_files, S)
    train_intensity_files = get_grouped_list(train_intensity_files, S)
    train_labels = get_grouped_list(train_labels, S)
    train_distances = get_grouped_list(train_distances, S)
    
    test_flow_files = get_grouped_list(test_flow_files, S)
    test_intensity_files = get_grouped_list(test_intensity_files, S)
    test_labels = get_grouped_list(test_labels, S)
    test_distances = get_grouped_list(test_distances, S)

    end = time.time()
    print('Generated {} train samples and {} test samples'.format(len(train_flow_files), len(test_flow_files)))
    print('Data generation took {:.0f} second'.format(end - start))
    
    np.save(data_path + 'set_{}/training/'.format(set_number)+'train_flow_files', train_flow_files)
    np.save(data_path + 'set_{}/training/'.format(set_number)+'train_intensity_files', train_intensity_files)
    np.save(data_path + 'set_{}/training/'.format(set_number)+'train_labels', train_labels)
    np.save(data_path + 'set_{}/training/'.format(set_number)+'train_distances', train_distances)
    np.save(data_path + 'set_{}/testing/'.format(set_number)+'test_flow_files', test_flow_files)
    np.save(data_path + 'set_{}/testing/'.format(set_number)+'test_intensity_files', test_intensity_files)
    np.save(data_path + 'set_{}/testing/'.format(set_number)+'test_labels', test_labels)
    np.save(data_path + 'set_{}/testing/'.format(set_number)+'test_distances', test_distances)
    
    return train_flow_files, train_intensity_files, train_labels, train_distances, \
           test_flow_files, test_intensity_files, test_labels, test_distances


# generate file paths for train data
def generate_train_data(args):
    '''
    Args:
    args: a dictionary that contains problem parameters

    Returns:
    train_flow_files: list of files for UV flows from for training
    train_intensity_files: list of files for frame intensities for training
    train_labels: list of labels (probability of hit, either 0 or 1) for training
    '''

    set_number = args['set_number']
    data_path = args['data_path']
    NNs = args['NNs']

    file_types = ['hit', 'miss', 'retreat', 'rotation']
    
    train_flow_files = []
    train_intensity_files = []
    train_distances = []
    train_labels = []
    
    rotational_train_flow_files = []
    rotational_train_intensity_files = []
    rotational_train_distances = []
    rotational_train_labels = []
    
    for file_type in file_types:
        # gather training paths and labels
        train_path = data_path + 'set_{}/training/'.format(set_number) + \
        file_type + '/UV_flow_samples/'
        if os.path.isdir(train_path):
            train_files = glob.glob(train_path + '*.npy')
            for train_flow_file in train_files:
                train_intensity_file = train_flow_file.replace('UV_flow_samples', 
                    'intensities_samples_cg')
                train_intensity_file = train_intensity_file.replace('UV_flow_sample', 
                    'intensities_sample_cg')
                train_distance_file = train_flow_file.replace('training', 
                    'other_info')
                train_distance_file = train_distance_file.replace('UV_flow_samples', 
                    'distances')
                train_distance_file = train_distance_file.replace('UV_flow_sample', 
                    'distance')
                train_flow =  np.load(train_flow_file, allow_pickle=True)
                train_steps = train_flow.shape[0]
                if file_type == 'hit':
                    train_label = np.ones(NNs)
                else:
                    train_label = np.zeros(NNs)
                if file_type == 'rotation':
                    rotational_train_flow_files.append(train_flow_file)
                    rotational_train_intensity_files.append(train_intensity_file)
                    rotational_train_labels.append(train_label)
                    rotational_train_distances.append(np.ones(train_steps))
                else:
                    train_flow_files.append(train_flow_file)
                    train_intensity_files.append(train_intensity_file)
                    train_labels.append(train_label)
                    distances = np.load(train_distance_file, allow_pickle=True)
                    distances_inverse = [1/item for sublist in distances for item in sublist]
                    train_distances.append(np.asarray(distances_inverse))

    # subsample rotational files
    rotational_train_index = random.sample(range(len(rotational_train_labels)), 
        int(args['rotational_fraction']*len(rotational_train_labels)))
    for index in rotational_train_index:
        train_flow_files.append(rotational_train_flow_files[index])
        train_intensity_files.append(rotational_train_intensity_files[index])
        train_labels.append(rotational_train_labels[index])
        train_distances.append(rotational_train_distances[index])

    # shuffle data
    temp = list(zip(train_flow_files, train_intensity_files, train_labels, train_distances)) 
    random.shuffle(temp) 
    train_flow_files, train_intensity_files, train_labels, train_distances = zip(*temp) 
    
    # group samples
    S = args['S']
    train_flow_files = get_grouped_list(train_flow_files, S)
    train_intensity_files = get_grouped_list(train_intensity_files, S)
    train_labels = get_grouped_list(train_labels, S)
    train_distances = get_grouped_list(train_distances, S)
    
    # sampling
    train_flow_snapshots = []
    train_intensity_snapshots = []
    train_labels_snapshots = []
    for file_path_flow, file_path_intensity, labels in zip(train_flow_files, train_intensity_files, train_labels):
        steps_list_flow = []
        steps_list_intensity = []
        for s in range(S):
            data_flow = np.load(file_path_flow[s], allow_pickle=True)
            data_intensity = np.load(file_path_intensity[s], allow_pickle=True)
            for dd in range(len(data_flow)): # len(data) indicates number of data samples in one data sample list
                i = random.randint(0, len(data_flow[dd])-1) # randomly select one snapshot
                step_flow = []
                step_intensity = []
                for j in range(len(data_flow[dd][i])): # len(data[dd][i]) indicates M
                    if len(data_flow[dd][i][j].shape) == 1:
                        step_flow.append(np.zeros((args['K']**2, 4)))
                        step_intensity.append(np.zeros((args['K']**2)))
                    else:
                        step_flow.append(data_flow[dd][i][j])
                        step_intensity.append(data_intensity[dd][i][j])
                steps_list_flow.append(np.stack([np.stack(step_flow)]))
                steps_list_intensity.append(np.stack([np.stack(step_intensity)]))
        steps_list_extended_flow = []
        steps_list_extended_intensity = []
        weight_list_extended = []
        label_list_extended = []
        labels = np.array(labels).flatten()
        for n in range(S*NNs):
            steps_extended_flow, weight_extended, label_extended \
                = get_extended_array(steps_list_flow[n], labels[n], 1)
            steps_extended_intensity, weight_extended, label_extended \
                = get_extended_array(steps_list_intensity[n], labels[n], 1)
            steps_list_extended_flow.append(steps_extended_flow)
            steps_list_extended_intensity.append(steps_extended_intensity)
            weight_list_extended.append(weight_extended)
            label_list_extended.append(label_extended)
        steps_list_extended_flow = np.swapaxes(np.stack(steps_list_extended_flow), 0, 1)
        steps_list_extended_intensity = np.swapaxes(np.stack(steps_list_extended_intensity), 0, 1)
        weight_list_extended = np.swapaxes(np.stack(weight_list_extended), 0, 1)
        label_list_extended = np.swapaxes(np.stack(label_list_extended), 0, 1)
        
        train_flow_snapshots.append(steps_list_extended_flow)
        train_intensity_snapshots.append(steps_list_extended_intensity)
        train_labels_snapshots.append(label_list_extended)
        
    np.save(data_path + 'set_{}/training/'.format(set_number)+'train_flow_snapshots', train_flow_snapshots)
    np.save(data_path + 'set_{}/training/'.format(set_number)+'train_intensity_snapshots', train_intensity_snapshots)
    np.save(data_path + 'set_{}/training/'.format(set_number)+'train_labels_snapshots', train_labels_snapshots)


def load_compressed_dataset(args, file_path, labels):
    """
    This function loads compressed datasets, and prepares them for training fand testing.
    
    Args:
    args: a dictionary that contains problem parameters
    file_path: a list of file paths
    labels: labels of the data contained in file paths
    
    Returns:
    steps_list_extended: expanded and extended data
    weight_list_extended: extended step weights
    label_list_extended: extended labels
    """
    N = len(file_path)
    T = 0
    steps_list = []
    for n in range(N):
        data = np.load(file_path[n], allow_pickle=True)
        for dd in range(len(data)): # len(data) indicates number of data samples in one data sample list
            steps = []
            T = np.maximum(T, len(data[dd])) # len(data[dd]) indicates total time steps
            for i in range(len(data[dd])):
                step = []
                for j in range(args['M']): 
                    if len(data[dd][i][j].shape) == 1:
                        step.append(np.zeros((args['K']**2, 4)))
                    else:
                        step.append(data[dd][i][j])
                steps.append(np.stack(step))
            steps_list.append(np.stack(steps))
    steps_list_extended = []
    weight_list_extended = []
    label_list_extended = []
    for n in range(N*len(data)):
        steps_extended, weight_extended, label_extended = get_extended_array(steps_list[n], labels[n], T)
        steps_list_extended.append(steps_extended)
        weight_list_extended.append(weight_extended)
        label_list_extended.append(label_extended)
    steps_list_extended = np.swapaxes(np.stack(steps_list_extended), 0, 1)
    weight_list_extended = np.swapaxes(np.stack(weight_list_extended), 0, 1)
    label_list_extended = np.swapaxes(np.stack(label_list_extended), 0, 1)
    
    return steps_list_extended, weight_list_extended, label_list_extended


def get_grouped_list(input_list, S):
    '''
    Args:
    input_list: # of sample data files in the folder, each sample data here is a list
    S: # of sample data list in one batch in mini-batch training
    
    Returns:
    output_list: a list of mini batches, len(output_list) =  # of mini batches in one training epoch
    '''
    output_list = []
    N = np.int(len(input_list)/S)
    for n in range(N):
        tem_list = input_list[n*S:(n+1)*S]
        output_list.append(tem_list)
        
    return output_list


def get_extended_array(input_array, label, T):
    """
    This function extends the data sample (input_array) in certain batch to let it have the same
    time dimension (T) as others.
    
    Args:
    input_array: input array
    label: scaler, label of the current input array
    T: target length of the time dimension
    
    Returns:
    output_array: extended data array
    output_weight: extended step weights
    output_label: extended label
    """
    T0 = input_array.shape[0]
    output_array = np.zeros((T, )+input_array.shape[1:])
    output_array[:T0] = input_array
    output_weight = np.zeros(T)
    output_weight[:T0] = 1./T0
    output_label = np.zeros(T)
    output_label[:] = label
    
    return output_array, output_weight, output_label


# generate one sample of optical stimulus, using exponential filters     
def generate_one_sample_exp(M, Rs, traj, sigma, theta_r, theta_matrix, coord_matrix, space_filter, K, L, dt, sample_dt, delay_dt):
    '''
    Args:
    M: # of lplc2 units
    Rs: radii of the balls (m)
    traj: trajectory
    sigma: noise added to images
    theta_r: half of the receptive field width (rad)
    K: K*K is the total # of elements.
    L: element dimension.
    dt: timescale of the simulations
    sample_dt: timescale of sampling
    delay_dt: timescale of delay in the motion detector.
    
    Returns:
    intensities_sample: optical stimulus, steps+1 (or lower) by K*L by K*L
    intensities_sample_cg: coarse-grained (cg) optical stimulus, steps (or lower) by K*K
    UV_flow_sample: flow fields, steps (or lower) by K*K by 4
    traj: trajectory, steps by P by 3
    distance: distance, steps by P by 0
    '''
    N = K*L + 4*L # Size of each frame, the additional 4*L is for spatial filtering and will be deleted afterwards.
    pad = 2*L
    leftup_corners = opsg.get_leftup_corners(K, L, pad)

    sample_step = np.int(sample_dt/dt)

    steps = len(traj)
    intensities_sample = []
    intensities_sample_cg = []
    UV_flow_sample = []
    distance = []
    signal_filtered_all = np.zeros((M, K*K, 4))
    if steps < sample_step:
        print('Warning: trajectory is too short!')
    for step in range(steps):
        # the current frame
        pos = traj[step]
        cf, cf_raw, hit = opsg.get_one_intensity(M, pos, Rs, theta_r, theta_matrix, coord_matrix, K, L, pad, sigma)
        # filtered signal
        signal_filtered_all, signal_cur = \
            flfd.get_filtered_and_current(signal_filtered_all, cf_raw, leftup_corners, space_filter, K, L, pad, dt, delay_dt)
        # Calculate the distance
        Ds = dn3d.get_radial_distances(pos)
        if step>0 and step%sample_step == 0:
            intensities_sample.append(cf)
            # Obtain the coarse-grained frame
            intensity_cg = opsg.get_intensity_cg(cf_raw, leftup_corners, K, L, pad)
            intensities_sample_cg.append(intensity_cg)
            # Calculate the flow field: U, V  
            UV_flow = flfd.get_flow_fields(signal_filtered_all, signal_cur, leftup_corners, K, L, pad)
            UV_flow_sample.append(UV_flow)
            distance.append(Ds)
    
    return intensities_sample, intensities_sample_cg, UV_flow_sample, distance


# # generate one sample of optical stimulus        
# def generate_one_sample(\
#     M, Rs, traj, sigma, theta_r, theta_matrix, coord_matrix, space_filter, K, L, dt, sample_dt, delay_dt):
#     """
#     Args:
#     M: # of lplc2 units
#     Rs: radii of the balls 
#     traj: trajectory
#     sigma: noise added to images
#     theta_r: half of the receptive field width (rad)
    
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
    
#     steps = len(traj)
#     intensities_sample = []
#     intensities_sample_cg = []
#     UV_flow_sample = []
#     distance_sample = []
#     signal_filtered_all = np.zeros((M, K*K, 4))
#     assert steps > sample_step and steps > delay_step, print('Error: trajectory is too short!')
#     for step in range(delay_step, steps):
#         # Calculate the distance
#         Ds = dn3d.get_radial_distances(traj[step])
#         if step > 0 and step % sample_step == 0:
#             # the previous frame
#             pos1 = traj[step-delay_step]
#             cf1, cf_raw1, _ = opsg.get_one_intensity(M, pos1, Rs, theta_r, theta_matrix, coord_matrix, K, L, pad, sigma)
#             _, signal_cur1 = \
#                 flfd.get_filtered_and_current(signal_filtered_all, cf_raw1, \
#                                               leftup_corners, space_filter, K, L, pad, dt, delay_dt)
#             # the current frame
#             pos = traj[step]
#             cf, cf_raw, _ = opsg.get_one_intensity(M, pos, Rs, theta_r, theta_matrix, coord_matrix, K, L, pad, sigma)
#             _, signal_cur = \
#                 flfd.get_filtered_and_current(signal_filtered_all, cf_raw, \
#                                               leftup_corners, space_filter, K, L, pad, dt, delay_dt)
#             # Obtain the coarse-grained frame
#             intensity_cg = opsg.get_intensity_cg(cf_raw, leftup_corners, K, L, pad)
#             intensities_sample.append(cf)
#             intensities_sample_cg.append(intensity_cg)
#             # Calculate the flow field: U, V  
#             UV_flow = flfd.get_flow_fields(signal_cur1, signal_cur, leftup_corners, K, L, pad)
#             UV_flow_sample.append(UV_flow)
#             distance_sample.append(Ds)
    
#     return intensities_sample, intensities_sample_cg, UV_flow_sample, distance_sample
    

# produce one of the frames in the video
def get_one_intensity(M, angle, pos, Rs, theta_r, theta_matrix, coord_matrix, K, L, pad, sigma=0.0):
    '''
    Args:
    M: # of lplc2 units
    pos: P by 3, the current positions of the centers of the balls (m).
    Rs: len(Rs) = P, the radii of the balls (m).
    theta_r: angular radius of the receptive field (rad).
    coords_x: coordinates of the frame in the vertical direction (x axis)
    coords_y: coordinates of the frame in the horizontal direction (y axis)
    dm: distance matrix calculated from coords_ud and coords_lr
    K: K*K is the total # of elements
    L: element dimension
    pad: padding size
    sigma: noise level on the frame.
    
    Returns:
    cf_raw: raw current frame, slightly larger than the actual frame
    cf: M by N by N, current frame.
    '''
    N = K*L + 2*pad # Size of each frame, the additional 4*L is for spatial filtering and will be deleted afterwards.
    N_half = (N-1)/2.
    cf_raw = np.zeros((M, N, N)) # current raw frame
    cf = np.zeros((M, N-2*pad, N-2*pad)) # current raw frame
    P = pos.shape[0]
    Ds = dn3d.get_radial_distances(pos)
    hit = Ds <= Rs
    if M == 1:
        lplc2_units = angle
    else:
        lplc2_units, _ = opsg.get_lplc2_units(M)   
    # remove signals outside the receptive field
    mask_2 = np.logical_not(theta_matrix<theta_r)
    mask_2 = mask_2[pad:-pad, pad:-pad]
    for m in range(M):
        mask_1_T = np.zeros((N, N))
        angle = -lplc2_units[m]
        pos_rot = opsg.get_rotated_coordinates(angle, pos)
        for p in range(P):
            x, y, z = pos_rot[p]
            R = Rs[p]
            D = dn3d.get_radial_distance(x, y, z)
            theta_b = opsg.get_angular_size(x, y, z, R)
            angle_matrix_b = opsg.get_angle_matrix_b(coord_matrix, pos_rot[p])
            mask_1 = angle_matrix_b <= theta_b
            mask_1_T = np.logical_or(mask_1, mask_1_T)
        cf_raw[m, mask_1_T] = 1.0
        # apply spacial filter
        cf_raw[m, :, :] = gaussian_filter(cf_raw[m, :, :], sigma=L/2.)
        # add noise to the signal
        noise = sigma*np.random.normal(size=N*N).reshape(N, N)
        cf_raw[m, :, :] = cf_raw[m, :, :] + noise
        # crop the frame
        cf[m, :, :] = cf_raw[m, pad:-pad, pad:-pad]
        cf[m, mask_2] = 0.
    
    return cf_raw, cf, hit


# generate one trajectory      
def generate_one_trajectory(x, y, z, vx, vy, vz, R, dynamics_fun, eta_1, dt):
    '''
    Args:
    (x, y, z): position of the ball
    (vx, vy, vz): velocity of the ball
    R: radius of the ball (m)
    dynamics_fun: predefined dynamics of the object, return the accelerations (m/sec^2)
    eta_1: random force added on the ball (m/sec^2)
    dt: time step (sec)
    '''
    D = dn3d.get_radial_distance(x, y, z)
    D0 = dn3d.get_radial_distance(x, y, z)
    traj = []
    step = 0
    while D > R and D <= np.maximum(10.*R, D0) and z > -R:
        step = step+1
        traj.append([[x, y, z]])
        x, y, z = dn3d.update_position(x, y, z, vx, vy, vz, dt)
        ax, ay, az = dynamics_fun(x, y, z, vx, vy, vz, eta_1, step, dt)
        vx, vy, vz = dn3d.update_velocity(vx, vy, vz, ax, ay, az, dt)
        D = dn3d.get_radial_distance(x, y, z)
    
    return np.float32(traj)


def generate_one_trajectory_grid(D_max, dt, dynamics_fun, eta_1, x, y, z, vx, vy, vz):
    '''
    Args:
    D_max: maximum distance from the origin 
    dt: time step (sec)
    dynamics_fun: predefined dynamics of the object, return the accelerations (m/sec^2)
    eta_1: random force added on the ball (m/sec^2)
    x, y, z, vx, vy, vz: initial position and velocity
    '''
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
        
        
def plot_signal_flow(Rs, traj, dt, sample_dt, delay_dt, D_thres, K, L, sigma, v_max, save_path):
    '''
    Args:
    Rs: radii of the balls (m)
    traj: trajectories
    dt: time step (sec)
    sample_dt: timescale of sampling
    delay_dt: timescale of delay in the motion detector.
    D_thres: threshold distance (m)
    K: K*K is the total # of elements.
    L: element dimension.
    sigma: noise
    v_max: maximum v
    save_path: save_path for the saved file.
    '''
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
        generate_one_sample(1, Rs, traj, sigma, theta_r, space_filter, K, L, dt, sample_dt, delay_dt, D_thres)
    steps = len(intensities_sample)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
#     mask_2 = np.logical_and(theta_matrix>=theta_r-0.01, theta_matrix<=theta_r+0.01)
#     mask_2 = mask_2[2*L:-2*L, 2*L:-2*L]
    for step in range(steps):
        cf = intensities_sample[step]
        UV_flow = UV_flow_sample[step]
        
        N = K*L
        N_half = (N-1)/2.
        
        # plot the image
        fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))
        im1 = np.array(cf)
#         im1[0, mask_2] = 1
        ax1.imshow(im1[0, :, :], cmap='gray_r', vmin=0, vmax=1)
        ax1.vlines(np.arange(-0.55, N, step=N/K), -0.55, N-0.45, color='salmon', linewidth=.5)
        ax1.hlines(np.arange(-0.55, N, step=N/K), -0.55, N-0.45, color='salmon', linewidth=.5)
        ax1.add_patch(Circle((N_half, N_half), N_half, edgecolor=[0.5, 0.5, 0.5], facecolor='None'))
        label=str(np.round((step+1)*dt, 2))+' s'
        ax1.text(15, 15, label, color='red')
        fig1.savefig(save_path+'intensity_{}.png'.format(step+1))
        
        # get the uv-flow
        cf_u, cf_v = flfd.set_flow_fields_on_frame2(UV_flow, leftup_corners, K, L, pad)
        
        # calculate the u-flow
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))
        vmin=np.min(cf_u.flatten())
        vmax=np.max(cf_u.flatten())
        vmax = np.max([-vmin, vmax])
        if vmax < 1e-6:
            vmax = 1
        vmax = v_max
#         cf_u[0, mask_2] = vmax
        ax2.imshow(cf_u[0, :, :], cmap=myheat, vmin=-vmax, vmax=vmax)
        PCM=ax2.get_children()[9] #get the mappable, the 1st and the 2nd are the x and y axes
        plt.colorbar(PCM, ax=ax2)
        ax2.vlines(np.arange(-0.55, N, step=N/K), -0.55, N-0.45, color='salmon', linewidth=.5)
        ax2.hlines(np.arange(-0.55, N, step=N/K), -0.55, N-0.45, color='salmon', linewidth=.5)
        ax2.add_patch(Circle((N_half, N_half), N_half, edgecolor=[0.5, 0.5, 0.5], facecolor='None'))
        fig2.savefig(save_path+'U_flow_{}.png'.format(step+1))
        
        # calculate the v-flow
        fig3, ax3 = plt.subplots(1, 1, figsize=(5, 5))
        vmin=np.min(cf_v.flatten())
        vmax=np.max(cf_v.flatten())
        vmax = np.max([-vmin, vmax])
        if vmax < 1e-6:
            vmax = 1
        vmax = v_max
#         cf_v[0, mask_2] = vmax
        ax3.imshow(cf_v[0, :, :], cmap=myheat, vmin=-vmax, vmax=vmax)
        PCM=ax3.get_children()[9] #get the mappable, the 1st and the 2nd are the x and y axes
        plt.colorbar(PCM, ax=ax3)
        ax3.vlines(np.arange(-0.55, N, step=N/K), -0.55, N-0.45, color='salmon', linewidth=.5)
        ax3.hlines(np.arange(-0.55, N, step=N/K), -0.55, N-0.45, color='salmon', linewidth=.5)
        ax3.add_patch(Circle((N_half, N_half), N_half, edgecolor=[0.5, 0.5, 0.5], facecolor='None'))
        fig3.savefig(save_path+'V_flow_{}.png'.format(step+1))


# generate one trajectory for rotation scene      
def generate_one_trajectory_nat(D_min, D_max, P, steps, dt, scal, around_z, around_y, around_x):
    '''
    Args:
    D_min: minimum distance
    D_max: maximum distance
    P: # of balls
    steps: # of steps
    dt: time step (sec)
    scal: scale of the rotaion, in degrees
    '''
    traj = []
    pos = []
    for p in range(P):
        D = D_min + (D_max-D_min)*np.random.random()
        theta_s = np.pi*np.random.random()
        phi_s = 2*np.pi*np.random.random()
        x = D*np.sin(theta_s)*np.cos(phi_s)
        y = D*np.sin(theta_s)*np.sin(phi_s)
        z = D*np.cos(theta_s)
        pos.append([x, y, z])
    pos = np.array(pos)
    r = R3d.from_euler('ZYX', [around_z, around_y, around_x], degrees=True)
    for step in range(steps):
        traj.append(pos)
        pos = r.apply(pos)
    
    return np.array(traj, np.float32)


# generate one sample of rotation scene
def generate_rotation_scene(M, K, L, pad, theta_r, P, sigma, D_max, D_min, R_rot):
    N = K*L + 4*L # Size of each frame, the additional 4*L is for spatial filtering and will be deleted afterwards.
    pad = 2*L
    leftup_corners = opsg.get_leftup_corners(K, L, pad)
    coord_y = np.arange(N) - (N-1)/2.
    coord_x = np.arange(N) - (N-1)/2.
    coords_y, coords_x = np.meshgrid(coord_y, -coord_x) # coordinates of each point in the frame
    dm = np.sqrt(coords_y**2 + coords_x**2) # distance matrix
    theta_matrix, phi_matrix = opsg.get_angle_matrix(theta_r, coords_x, coords_y, dm, K, L)
    coord_matrix = opsg.get_coord_matrix(phi_matrix, theta_matrix, 1.)
    pos = []
    Rs = []
    for p in range(P):
        D = D_min + (D_max-D_min)*np.random.random()
        theta_s = np.pi*np.random.random()
        phi_s = 2*np.pi*np.random.random()
        x = D*np.sin(theta_s)*np.cos(phi_s)
        y = D*np.sin(theta_s)*np.sin(phi_s)
        z = D*np.cos(theta_s)
        pos.append([x, y, z])
        Rs.append(np.random.random()*R_rot)
#         Rs.append(R_rot)
    pos = np.array(pos)
    Rs = np.array(Rs)
    cf, cf_raw, hit = opsg.get_one_intensity(M, pos, Rs, theta_r, theta_matrix, coord_matrix, K, L, pad, sigma)
    intensity_cg = opsg.get_intensity_cg(cf_raw, leftup_corners, K, L, pad)
    
    return cf, intensity_cg, hit


# generate one sample of rotation scene
def generate_one_intensity_example(M, K, L, pad, theta_r, sigma, pos, Rs):
    N = K*L + 2*pad # Size of each frame, the additional 2*pad is for spatial filtering and will be deleted afterwards.
    coord_y = np.arange(N) - (N-1)/2.
    coord_x = np.arange(N) - (N-1)/2.
    coords_y, coords_x = np.meshgrid(coord_y, -coord_x) # coordinates of each point in the frame
    dm = np.sqrt(coords_y**2 + coords_x**2) # distance matrix
    cf_raw, cf, hit = opsg.get_one_intensity(M, pos, Rs, theta_r, coords_x, coords_y, dm, K, L, pad, sigma)
    
    return cf_raw, cf, hit


# Get averaged frames.
def get_ave_frames(K, datapath, data_types):
    '''
    Args:
    K: K*K is the total # of elements.
    datapath: path where the data is saved.
    data_types: data types
    '''
    L = len(data_types)
    N_frames = np.zeros(L)
    Ave_frames = np.zeros((L, K*K, 4))
    for ind, data_type in enumerate(data_types):
        path = datapath+data_type+'/UV_flow_samples/'
        if os.path.isdir(path):
            files = glob.glob(path+'*.npy')
            for index in np.arange(len(files)):
                rn_arrs = np.load(files[index], allow_pickle=True)
                for rn_arr in rn_arrs:
                    steps = len(rn_arr)
                    if len(rn_arr) == 0:
                        print(files[index])
                    M = len(rn_arr[0])
                    N_frames[ind] = N_frames[ind] + steps*M
                    N_frames[-1] = N_frames[-1] + steps*M
                    for step in range(steps):
                        for m in range(M):
                            if rn_arr[step][m].shape[0] == 1:
                                rn_arr[step][m] = np.array([0])
                            Ave_frames[ind, :, :] = Ave_frames[ind, :, :] + rn_arr[step][m]
                            Ave_frames[-1, :, :] = Ave_frames[-1, :, :] + rn_arr[step][m]
    for l in range(L):
        for i in range(4):
            Ave_frames[l, :, i] = Ave_frames[l, :, i]/N_frames[l]
    print(f'Number of frames for each type: {N_frames}.')
    print(f'The shape of the averaged frames is {Ave_frames.shape}.')
    
    return Ave_frames
    
    
# Plot the averaged samples for each data type
def plot_ave_samples(K, data_types, motion_types, Ave_frames, mask_d, colormap, filename):
    '''
    Args:
    K: K*K is the total # of elements.
    data_types: data types
    motion_types: motion types
    Ave_frames: len(data_types) by K*K by 4
    mask_d: disk mask
    colormap: color map of the image
    filename: filename to save
    '''
    L = len(data_types)
    fig = plt.figure(figsize=(16, 4*L))
    ind = 0
    for ind1, data_type in enumerate(data_types):
        for ind2, motion_type in enumerate(motion_types):
            ind = ind+1
            plt.subplot(L, 4, ind)
            ave_frame = Ave_frames[ind1, :, ind2].reshape((K, K))
            ave_frame[mask_d] = 0.
            plt.imshow(ave_frame, cmap=colormap)
            plt.colorbar()
            plt.title(data_type+'/'+motion_type)
            plt.xticks([])
            plt.yticks([])
    plt.show()
    fig.savefig(filename, bbox_inches='tight')
    
    
# Get averaged frames.
def get_ave_frames_per_unit(K, M, datapath, data_type, intensity_UV):
    '''
    Args:
    K: K*K is the total # of elements.
    datapath: path where the data is saved.
    data_type: data type
    '''
    if intensity_UV == 'UV_flow_samples':
        Ave_frames = np.zeros((M+1, K*K, 4))
    elif intensity_UV == 'intensities_samples_cg':
        Ave_frames = np.zeros((M+1, K*K, 1))
    N_frames = 0
    path = datapath+data_type+'/'+intensity_UV+'/'
    if os.path.isdir(path):
        files = glob.glob(path+'*.npy')
        for index in np.arange(len(files)):
            rn_arrs = np.load(files[index], allow_pickle=True)
            for rn_arr in rn_arrs:
                steps = len(rn_arr)
                N_frames = N_frames + steps
                for m in range(M):
                    for step in range(steps):
                        if rn_arr[step][m].shape[0] > 1:
                            if intensity_UV == 'intensities_samples_cg':
                                rn_arr_tem = np.expand_dims(rn_arr[step][m], axis=1)
                            else:
                                rn_arr_tem = rn_arr[step][m]
                            Ave_frames[m, :, :] = Ave_frames[m, :, :] + rn_arr_tem[:, :]
                            Ave_frames[-1, :, :] = Ave_frames[-1, :, :] + rn_arr_tem[:, :]
                        else:
                            Ave_frames[m, :, :] = Ave_frames[m, :, :] + 0
                            Ave_frames[-1, :, :] = Ave_frames[-1, :, :] + 0
    Ave_frames[:-1, :, :] = Ave_frames[:-1, :, :]/N_frames
    Ave_frames[-1, :, :] = Ave_frames[-1, :, :]/(N_frames*M)
    
    print(f'Number of frames for each type: {N_frames}.')
    print(f'The shape of the averaged frames is {Ave_frames.shape}.')
    
    return Ave_frames
    
    
# Plot the averaged samples for each data type
def plot_ave_samples_per_unit(K, motion_types, Ave_frames, mask_d, colormap, filename):
    '''
    Args:
    K: K*K is the total # of elements.
    motion_types: motion types
    Ave_frames: len(data_types) by K*K by 4
    mask_d: disk mask
    colormap: color map of the image
    filename: filename to save
    '''
    L = Ave_frames.shape[0]
    fig = plt.figure(figsize=(16, 4*L))
    ind = 0
    for l in range(L):
        for ind2, motion_type in enumerate(motion_types):
            ind = ind+1
            plt.subplot(L, 4, ind)
            ave_frame = Ave_frames[l, :, ind2].reshape((K, K))
            ave_frame[mask_d] = 0.
            plt.imshow(ave_frame, cmap=colormap)
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
    plt.show()
    fig.savefig(filename, bbox_inches='tight')


# Get distribution of velocities at one pixel
def get_distribution_pixel(K, data_path, set_number, data_type, pixel, motion):
    '''
    Args:
    K:
    data_path: path where the data is saved.
    data_types: data types
    set_number:
    pixel: tuple, (row, col), location of the pixel
    motion: 0, U+; 1, U-; 2, V+; 3, V-.
    '''
    velocities = []
    path = data_path+'set_{}/training/'.format(set_number)+data_type+'/UV_flow_samples/'
    if os.path.isdir(path):
        files = glob.glob(path+'*.npy')
        for index in np.arange(len(files)):
            rn_arr = np.load(files[index], allow_pickle=True)
            row = pixel[0]
            col = pixel[1]
            steps = len(rn_arr)
            M = len(rn_arr[0])
            for step in range(steps):
                for m in range(M):
                    if rn_arr[step][m].shape[0] > 1:
                        if rn_arr[step][m][row*K+col, motion] > 0:
                            velocities.append(rn_arr[step][m][row*K+col, motion])
    velocities = np.array(velocities)

    return velocities


# Get distribution of velocities all pixels
def get_distribution_pixel_all(K, data_path, set_number, data_type):
    '''
    Args:
    K:
    data_path: path where the data is saved.
    set_number:
    data_types: data types
    '''
    velocities_all = []
    path = data_path+'set_{}/training/'.format(set_number)+data_type+'/UV_flow_samples/'
    if os.path.isdir(path):
        files = glob.glob(path+'*.npy')
        for index in np.arange(len(files)):
            rn_arr = np.load(files[index])
            velocities_all.extend(rn_arr.flatten().tolist())
    velocities_all = np.array(velocities_all)
    
    return velocities_all


def plot_all_hists(arr_list, color_list, legend_list, y_scal, xl, xu, yl, yu):
    fig = plt.figure(figsize=(5, 5))
    for ind, arr in enumerate(arr_list):
        if len(arr) > 0:
            hist, bin_centers = get_hist(arr)
            plt.plot(bin_centers, hist, c=color_list[ind], alpha=1.)
    plt.yscale(y_scal)
    plt.ylim([yl, yu])
    plt.xlim([xl, xu])
    plt.xlabel('HRC output')
    plt.ylabel('Density')
    plt.legend(legend_list)
    plt.show()
    
    return fig


def get_hist(arr):
    arr = arr.flatten()
    bins = np.int(np.sqrt(arr.shape[0]))
    bins = np.minimum(bins, 1000)
    hist, bin_edges = np.histogram(arr, bins, density=True)
    bin_centers = 0.5*(bin_edges[:-1]+bin_edges[1:])
    
    return hist, bin_centers


# Plot the filtered responses on given trajectory
def plot_response_over_time(\
    args, model_path, model_type, figuretype, res_T_2Fa_max, UV_flow_files, intensity_files, ymin, ymax_dict, filename):
    '''
    Args:
    args: a dictionary that contains problem parameters
    model_path: a directory that contains pre-trained model parameters
    model_type: model type
    UV_flow_files: files that contains UV flow
    intensity_files: files that contains frame intensities
    filename: filename to save
    '''
    K = args['K']
    alpha_leak = args['alpha_leak']
    a = np.load(model_path + "trained_a.npy")
    b = np.load(model_path + "trained_b.npy")
    
    intercept_e = np.load(model_path + "trained_intercept_e.npy")
    tau_1 = np.load(model_path + "trained_tau_1.npy")
    weights_e = np.load(model_path + "trained_weights_e.npy")
#     weights_e = 1./(1.+np.exp(-weights_e))
    if model_type == 'inhibitory1' or model_type == 'inhibitory2':
        weights_i = np.load(model_path + "trained_weights_i.npy")
#         weights_i = 1./(1.+np.exp(-weights_i))
        intercept_i = np.load(model_path + "trained_intercept_i.npy")
    weights_intensity = None
    if args["use_intensity"]:
        weights_intensity = np.load(model_path + "trained_weights_intensity.npy")
    if args['temporal_filter']:
        args['n'] = 1
    
    res_max = 0.
    for ind, UV_flow_file in enumerate(UV_flow_files):
        UV_flow = np.load(UV_flow_file)
        intensity = None
        if args['use_intensity']:
            intensity = np.load(intensity_files[ind])
            assert(len(intensity) == len(UV_flow_))
        if model_type == 'excitatory':
            res_T = get_response_excitatory_only(\
                args, weights_e, intercept_e, a, b, UV_flow, weights_intensity, intensity)
        elif model_type == 'excitatory_WNR':
            res_T = get_response_excitatory_only(\
                args, weights_e, intercept_e, a, b, UV_flow, weights_intensity, intensity)
        elif model_type == 'inhibitory1':
            res_T = get_response_with_inhibition1(\
                args, weights_e, weights_i, intercept_e, intercept_i, a, b, UV_flow, weights_intensity, intensity)
        elif model_type == 'inhibitory2':
            res_rest, res_T, _, _ = get_response_with_inhibition2(\
                args, weights_e, weights_i, intercept_e, intercept_i, UV_flow, weights_intensity, intensity)
        res_max = np.maximum(res_max, res_T.max())
        
    fig = plt.figure(figsize=(1.5*len(UV_flow_files), 1.5))
    for ind, UV_flow_file in enumerate(UV_flow_files):
        UV_flow = np.load(UV_flow_file)
        intensity = None
        if args['use_intensity']:
            intensity = np.load(intensity_files[ind])
            assert(len(intensity) == len(UV_flow_))
        if model_type == 'excitatory':
            res_T = get_response_excitatory_only(\
                args, weights_e, intercept_e, a, b, UV_flow, weights_intensity, intensity)
        elif model_type == 'excitatory_WNR':
            res_T = get_response_excitatory_only(\
                args, weights_e, intercept_e, a, b, UV_flow, weights_intensity, intensity)
        elif model_type == 'inhibitory1':
            res_T = get_response_with_inhibition1(\
                args, weights_e, weights_i, intercept_e, intercept_i, a, b, UV_flow, weights_intensity, intensity)
        elif model_type == 'inhibitory2':
            res_rest, res_T, _, _ = get_response_with_inhibition2(\
                args, weights_e, weights_i, intercept_e, intercept_i, UV_flow, weights_intensity, intensity)
        plt.subplot(1, len(UV_flow_files), ind+1)
        if figuretype == '2F':
            res_to_tem = res_T-res_T[0]
            res_to_plot = np.zeros_like(res_to_tem)
            for t in range(res_to_plot.shape[0]):
                res_to_plot[t] = general_temp_filter(0, 0.1, 0.01, res_to_tem[:t+1])
            plt.plot(np.arange(len(res_T)), res_to_plot, c='k', linewidth=1)
            ymax = res_max-res_T[0]
        else:
            res_to_tem = (res_T-res_T[0])/res_T_2Fa_max
            res_to_plot = np.zeros_like(res_to_tem)
            for t in range(res_to_plot.shape[0]):
                res_to_plot[t] = general_temp_filter(0, 0.1, 0.01, res_to_tem[:t+1])
            plt.plot(np.arange(len(res_T)), res_to_plot, c='k', linewidth=1)
            ymax = (res_max-res_T[0])/res_T_2Fa_max
            ymax = ymax_dict[figuretype]
        plt.ylim([ymin, ymax])
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
    fig.savefig(filename, bbox_inches='tight')
    plt.close()
    
    
# get the filtered responses on given trajectory
def get_response_over_time(args, model_path, model_type, UV_flow, intensity=0):
    '''
    Args:
    args: a dictionary that contains problem parameters
    model_path: a directory that contains pre-trained model parameters
    model_type: model type
    UV_flow: UV flow
    intensity: frame intensity
    
    Returns:
    res_T: response
    '''
    K = args['K']
    alpha_leak = args['alpha_leak']
    intensity = None
    if args['use_intensity']:
        intensity = intensity
        assert(len(intensity) == len(UV_flow_))
    
    a = np.load(model_path + "trained_a.npy")
    b = np.load(model_path + "trained_b.npy")
    
    intercept_e = np.load(model_path + "trained_intercept_e.npy")
    tau_1 = np.load(model_path + "trained_tau_1.npy")
    weights_e = np.load(model_path + "trained_weights_e.npy")
    if model_type == 'inhibitory1' or model_type == 'inhibitory2':
        weights_i = np.load(model_path + "trained_weights_i.npy")
        intercept_i = np.load(model_path + "trained_intercept_i.npy")
        
    
    weights_intensity = None
    if args["use_intensity"]:
        weights_intensity = np.load(model_path + "trained_weights_intensity.npy")
    if args['temporal_filter']:
        args['n'] = 1
    if model_type == 'excitatory':
        res_T = get_response_excitatory_only(\
            args, weights_e, intercept_e, a, b, UV_flow, weights_intensity, intensity)
    elif model_type == 'excitatory_WNR':
        res_T = get_response_excitatory_only(\
            args, weights_e, intercept_e, a, b, UV_flow, weights_intensity, intensity)
    elif model_type == 'inhibitory1':
        res_T = get_response_with_inhibition1(\
            args, weights_e, weights_i, intercept_e, intercept_i, a, b, UV_flow, weights_intensity, intensity)
    elif model_type == 'inhibitory2':
        res_rest, res_T, _, _, _ = get_response_with_inhibition2(\
            args, weights_e, weights_i, intercept_e, intercept_i, UV_flow, weights_intensity, intensity)
    
    res_TM = res_T.sum(axis=1)
    res_rest_M = res_rest.sum()
    prob = sigmoid_array(np.abs(a)*res_TM+b)
    prob_rest = sigmoid_array(np.abs(a)*res_rest_M+b)
    
    return res_rest, res_T, prob, prob_rest


def get_sparsity(M, res_T_all, res_rest, data_type_ind):
    """
    Get the numbers of active units for signals.
    
    Args:
    M: number of units
    res_T_all: response curves for all the data types, all the signals, 
               all the time points, all the units.
    res_rest: rest response when there is no signal
    data_type_ind: 0 -> hit, 1 -> miss, 2 -> retreat, 3 -> rotation
    
    Returns:
    active_unit: a list of numbers of active units for all the signals
    """
    assert M == len(res_T_all[0][0][0])
    
    active_unit = []
    res_T_all1 = res_T_all[data_type_ind]
    res_rest1 = res_rest[data_type_ind][0]
    N = len(res_T_all1)
    for n in range(N):
        res_T_all_np = np.array(res_T_all1[n])
        res_T_max = res_T_all_np.max(axis=0)
        M_active = (res_T_max > res_rest1).sum()
        active_unit.append(M_active)
    
    return active_unit

    
# Plot the trajectories in 3d
def plot_trajectories(data_path, data_types, colors, filename):
    fig = plt.figure(figsize=(5*len(data_types), 5))
    for ind1, data_type in enumerate(data_types):
        if data_type != 'rotation':
            ax = fig.add_subplot(1, len(data_types), 1+ind1, projection='3d')
            path = data_path+data_type+'/trajectories/'
            files = glob.glob(path+'*.npy')
            for ind2, file in enumerate(files):
                if ind2%1 == 0:
                    rn_arrs = np.load(file, allow_pickle=True)
                    for rn_arr in rn_arrs:
                        rn_arr = np.array(rn_arr)
                        P = rn_arr.shape[1]
                        for p in range(P):
                            ax.plot(rn_arr[:, p, 0], rn_arr[:, p, 1], rn_arr[:, p, 2], c=colors[0], linewidth=1) 
            for ind2, file in enumerate(files):
                if ind2%1 == 0:
                    rn_arrs = np.load(file, allow_pickle=True)
                    for rn_arr in rn_arrs:
                        rn_arr = np.array(rn_arr)
                        P = rn_arr.shape[1]
                        for p in range(P):
                            ax.plot(rn_arr[0:1, p, 0], rn_arr[0:1, p, 1], rn_arr[0:1, p, 2], 'k.')
            ax.plot([0], [0], [0], 'r.')    
            ax.set_title(data_type)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
        elif data_type == 'rotation':
            ax = fig.add_subplot(1, len(data_types), 1+ind1, projection='3d')
            path = data_path+data_type+'/trajectories/'
            files = glob.glob(path+'*.npy')
            rn_arrs = np.load(files[3], allow_pickle=True)
            rn_arr = np.array(rn_arrs[9])
            P = rn_arr.shape[1]
            for p in range(P):
                if p%1 == 0:
                    ax.plot(rn_arr[:, p, 0], rn_arr[:, p, 1], rn_arr[:, p, 2], c=colors[1], linewidth=1)
                    ax.plot(rn_arr[0:1, p, 0], rn_arr[0:1, p, 1], rn_arr[0:1, p, 2], 'k.') 
            ax.plot([0], [0], [0], 'r.')    
            ax.set_title(data_type)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
    fig.savefig(filename, bbox_inches='tight')
    plt.show()
    
    
# Plot the distances
def plot_distances(data_path, data_types, colors, filename):
    fig = plt.figure(figsize=(4*len(data_types), 4))
    for ind1, data_type in enumerate(data_types):
        if data_type != 'rotation':
            T = 0
            ax = fig.add_subplot(1, len(data_types), 1+ind1)
            path = data_path+data_type+'/distances/full/'
            files = glob.glob(path+'*.npy')
            for ind2, file in enumerate(files):
                if ind2%10 == 0:
                    rn_arrs = np.load(file, allow_pickle=True)
                    for rn_arr in rn_arrs:
                        rn_arr = np.array(rn_arr)
                        ax.plot(rn_arr, c=colors[0], linewidth=0.5)
                        T = np.maximum(T, len(rn_arr))
            path = data_path+data_type+'/distances/full/'
            files = glob.glob(path+'*.npy')
            for ind3, file in enumerate(files):
                if ind3%10 == 0:
                    rn_arrs = np.load(file, allow_pickle=True)
                    for rn_arr in rn_arrs:
                        rn_arr = np.array(rn_arr)
                        ax.plot(rn_arr, c=colors[1], linewidth=0.5)
            ax.plot(np.arange(T), np.ones(T), c='k', linewidth=0.5)
            ax.set_title(data_type)
            ax.set_xlabel('time step')
            ax.set_ylabel('distance')
        elif data_type == 'rotation':
            ax = fig.add_subplot(1, len(data_types), 1+ind1)
            path = data_path+data_type+'/trajectories/'
            files = glob.glob(path+'*.npy')
            rn_arrs = np.load(files[3], allow_pickle=True)
            rn_arr = np.array(rn_arrs[9])
            T = rn_arr.shape[0]
            P = rn_arr.shape[1]
            for p in range(P):
                if p%1 == 0:
                    ci = rn_arr[0][p]
                    Di = dn3d.get_radial_distance(ci[0], ci[1], ci[2])
                    cf = rn_arr[-1][p]
                    Df = dn3d.get_radial_distance(cf[0], cf[1], cf[2])
                ax.plot([0, T], [Di, Df], c=colors[1], linewidth=0.5)
            ax.plot(np.arange(T), np.ones(T), c='k', linewidth=0.5)
            ax.set_title(data_type)
            ax.set_xlabel('time step')
            ax.set_ylabel('distance')
            
    fig.savefig(filename, bbox_inches='tight')
    plt.show()
    
    
# output to a LPLC2 neuron at a specific time point with excitatory neurons only
def get_output_excitatory_only(weights_e, intercept_e, UV_flow_t, alpha_leak, weights_intensity=0, intensity_t=0):
    '''
    Args:
    weights_e: weights for excitatory neurons, K*K by 4, for axis=1, 0:right motion, 1:left motion, 2:up motion, 3:down motion
    intercept_e: intercept in the activation function
    UV_flow_t: flow field at time point t
    alpha_leak: slope for leaky relu
    
    Returns:
    output_t: output to a LPLC2 neuron at a specific time point t
    '''
    M = len(UV_flow_t)
    output_t = intercept_e*np.ones(M)
    for m in range(M):
        if weights_intensity:
            if intensity_t[m].shape[0] > 1:
                output_t[m] = output_t[m] + np.dot(intensity_t[m][:], weights_intensity[:])
            else:
                output_t[m] = output_t[m] + 0
        # input
        for i in range(4):
            if UV_flow_t[m].shape[0] > 1:
                output_t[m] = output_t[m] + np.dot(UV_flow_t[m][:, i], weights_e[:, i])
            else:
                output_t[m] = output_t[m] + 0
        # relu
        output_t[m] = get_leaky_relu(alpha_leak, output_t[m])
    
    return output_t


# response of a LPLC2 neuron
def get_response_excitatory_only(args, weights_e, intercept_e, a, b, UV_flow, weights_intensity=0, intensity=0):
    '''
    Args:
    args: other arguments
    weights_e: weights for excitatory neurons, K*K by 4, for axis=1, 0:right motion, 1:left motion, 2:up motion, 3:down motion
    intercept_e: intercept in the activation function
    a: coefficient
    b: intercept
    UV_flow: flow field
    alpha_leak: slope for leaky relu
    
    Returns:
    res_T: response of a LPLC2 neuron
    '''
    n = args['n']
    tau_1 = args['tau_1']
    dt = args['dt']
    alpha_leak = args['alpha_leak']
    T = len(UV_flow)
    M = len(UV_flow[0])
    output_t = np.zeros((T, M))
    res_T = np.zeros((T, M))
    if weights_intensity:
        for t in range(T):
            output_t = get_output_excitatory_only(\
                weights_e, intercept_e, UV_flow[t], alpha_leak, weights_intensity, intensity[t])
            output_t[t, :] = output_t[:]
            res_T[t, :] = general_temp_filter(n, tau_1, dt, output_t[:t+1, :])
    else:
        for t in range(T):
            output_t = get_output_excitatory_only(\
                weights_e, intercept_e, UV_flow[t], alpha_leak)
            output_t[t] = output_t[:]
            res_T[t, :] = general_temp_filter(n, tau_1, dt, output_t[:t+1, :])
    
    return a*res_T+b


# output to a LPLC2 neuron at a specific time point with both excitatory and inhibitory neurons 
def get_output_with_inhibition1(weights_e, weights_i, intercept_e, intercept_i, UV_flow_t, alpha_leak, \
                               weights_intensity=0, intensity_t=0):
    '''
    Args:
    weights_e: weights for excitatory neurons, K*K by 4, for axis=1, 0:right motion, 1:left motion, 2:up motion, 3:down motion
    weights_i: weights for inhibitory neurons, K*K by 4, for axis=1, 0:right motion, 1:left motion, 2:up motion, 3:down motion
    intercept_e: intercept in the activation function for LPLC2 neuron
    intercept_i: intercept in the activation function for inhibitory neurons
    UV_flow_t: flow field at time point t
    alpha_leak: slope for leaky relu
    
    Returns:
    output_t: output to a LPLC2 neuron at a specific time point t
    '''
    M = len(UV_flow_t)
    output_t = intercept_e*np.ones(M)
    output_t_i = intercept_i*np.ones(M)
    for m in range(M):
        if weights_intensity:
            if intensity_t[m].shape[0] > 1:
                output_t[m] = output_t[m] + np.dot(intensity_t[m][:], weights_intensity[:])
            else:
                output_t[m] = output_t[m] + 0
        # excitatory input
        for i in range(4):
            if UV_flow_t[m].shape[0] > 1:
                output_t[m] = output_t[m] + np.dot(UV_flow_t[m][:, i], weights_e[:, i])
            else:
                output_t[m] = output_t[m] + 0
        # inhibitory input
        for i in range(4):
            if UV_flow_t[m].shape[0] > 1:
                output_t_i[m] = output_t_i[m] + np.dot(UV_flow_t[m][:, i], weights_i[:, i])
            else:
                output_t_i[m] = output_t_i[m] + 0
        output_t_i[m] = get_leaky_relu(alpha_leak, output_t_i[m])
        # total input
        output_t[m] = output_t[m] - output_t_i[m]
        # relu
        output_t[m] = get_leaky_relu(alpha_leak, output_t[m])
    
    return output_t


# response of a LPLC2 neuron
def get_response_with_inhibition1(\
    args, weights_e, weights_i, intercept_e, intercept_i, a, b, UV_flow, weights_intensity=0, intensity=0):
    '''
    Args:
    args: other arguments
    weights_e: weights for excitatory neurons, K*K by 4, for axis=1, 0:right motion, 1:left motion, 2:up motion, 3:down motion
    weights_i: weights for inhibitory neurons, K*K by 4, for axis=1, 0:right motion, 1:left motion, 2:up motion, 3:down motion
    intercept_e: intercept in the activation function for LPLC2 neuron
    intercept_i: intercept in the activation function for inhibitory neurons
    a: coefficient
    b: intercept
    UV_flow: flow field
    alpha_leak: slope for leaky relu
    
    Returns:
    res_T: response of a LPLC2 neuron
    '''
    n = args['n']
    tau_1 = args['tau_1']
    dt = args['dt']
    alpha_leak = args['alpha_leak']
    T = len(UV_flow)
    M = len(UV_flow[0])
    output_t = np.zeros((T, M))
    res_T = np.zeros((T, M))
    if weights_intensity:
        for t in range(T):
            output_t = get_output_with_inhibition1(\
                weights_e, weights_i, intercept_e, intercept_i, UV_flow[t], alpha_leak, weights_intensity, intensity_[t])
            output_t[t, :] = output_t[:]
            res_T[t, :] = general_temp_filter(n, tau_1, dt, output_t[:t+1, :])
    else:
        for t in range(T):
            output_t = get_output_with_inhibition1(\
                weights_e, weights_i, intercept_e, intercept_i, UV_flow[t], alpha_leak)
            output_t[t, :] = output_t[:]
            res_T[t, :] = general_temp_filter(n, tau_1, dt, output_t[:t+1, :])
    
    return a*res_T+b


# output to a LPLC2 neuron at a specific time point with both excitatory and inhibitory neurons 
def get_output_with_inhibition2(weights_e, weights_i, intercept_e, intercept_i, UV_flow_t, alpha_leak, \
                               weights_intensity=0, intensity_t=0):
    '''
    Args:
    weights_e: weights for excitatory neurons, K*K by 4, for axis=1, 0:right motion, 1:left motion, 2:up motion, 3:down motion
    weights_i: weights for inhibitory neurons, K*K by 4, for axis=1, 0:right motion, 1:left motion, 2:up motion, 3:down motion
    intercept_e: intercept in the activation function for the LPLC2 neuron
    intercept_i: intercept in the activation function for inhibitory neurons
    UV_flow_t: flow field at time point t
    alpha_leak: slope for leaky relu
    
    Returns:
    output_t: output of a LPLC2 neuron at a specific time point t
    Ie_t: excitatory component without thresholding
    Ii_t: inhibitory component with thresholding
    Ii_t2: inhibitory component without thresholding
    '''
    M = len(UV_flow_t)
    output_t = intercept_e*np.ones(M)
    Ie_t = np.zeros(M)
    Ii_t = np.zeros(M)
    Ii_t2 = np.zeros(M)
    for m in range(M):
        if weights_intensity:
            if intensity_t[m].shape[0] > 1:
                output_t[m] = output_t[m] + np.dot(intensity_t[m][:], weights_intensity[:])
            else:
                output_t[m] = output_t[m] + 0
        # excitory input
        for i in range(4):
            if UV_flow_t[m].shape[0] > 1:
                output_t[m] = output_t[m] + np.dot(UV_flow_t[m][:, i], weights_e[:, i])
                Ie_t[m] = Ie_t[m] + np.dot(UV_flow_t[m][:, i], weights_e[:, i])
            else:
                output_t[m] = output_t[m] + 0
                Ie_t[m] = Ie_t[m] + 0
        # inhibitory input
        if UV_flow_t[m].shape[0] > 1:
            h1 = intercept_i + np.dot(UV_flow_t[m][:, 0], weights_i[:, 0])
            h2 = intercept_i + np.dot(UV_flow_t[m][:, 1], weights_i[:, 1])
            h3 = intercept_i + np.dot(UV_flow_t[m][:, 2], weights_i[:, 2])
            h4 = intercept_i + np.dot(UV_flow_t[m][:, 3], weights_i[:, 3]) 
        else:
            h1 = intercept_i + 0
            h2 = intercept_i + 0
            h3 = intercept_i + 0
            h4 = intercept_i + 0
        # relu
        h12 = get_leaky_relu(alpha_leak, h1-intercept_i)
        h22 = get_leaky_relu(alpha_leak, h2-intercept_i)
        h32 = get_leaky_relu(alpha_leak, h3-intercept_i)
        h42 = get_leaky_relu(alpha_leak, h4-intercept_i)
        Ii_t2[m] = h12+h22+h32+h42
        
        h1 = get_leaky_relu(alpha_leak, h1)
        h2 = get_leaky_relu(alpha_leak, h2)
        h3 = get_leaky_relu(alpha_leak, h3)
        h4 = get_leaky_relu(alpha_leak, h4)
        Ii_t[m] = h1+h2+h3+h4
        output_t[m] = output_t[m] - (h1+h2+h3+h4)
        # relu
        output_t[m] = get_leaky_relu(alpha_leak, output_t[m])
    
    return output_t, Ie_t, Ii_t, Ii_t2


# response of a LPLC2 neuron
def get_response_with_inhibition2(\
    args, weights_e, weights_i, intercept_e, intercept_i, UV_flow, weights_intensity=0, intensity=0):
    '''
    Args:
    args: other arguments
    weights_e: weights for excitatory neurons, K*K by 4, for axis=1, 0:right motion, 1:left motion, 2:up motion, 3:down motion
    weights_i: weights for inhibitory neurons, K*K by 4, for axis=1, 0:right motion, 1:left motion, 2:up motion, 3:down motion
    intercept_e: intercept in the activation function for LPLC2 neuron
    UV_flow: flow field
    alpha_leak: slope for leaky relu
    
    Returns:
    res_rest: baseline response when there is no stimuli
    res_T: response of a LPLC2 neuron
    Ie_T: excitatory component without thresholding
    Ii_T: inhibitory component with thresholding
    Ii_T2: inhibitory component without thresholding
    '''
    n = args['n']
    tau_1 = args['tau_1']
    dt = args['dt']
    alpha_leak = args['alpha_leak']
    T = len(UV_flow)
    M = len(UV_flow[0])
    res_T = np.zeros((T, M))
    Ie_T = np.zeros((T, M))
    Ii_T = np.zeros((T, M))
    Ii_T2 = np.zeros((T, M))
    UV_flow_0 = np.zeros((M, 1))
    if weights_intensity:
        res_rest, _, _, _ = get_output_with_inhibition2(\
            weights_e, weights_i, intercept_e, intercept_i, UV_flow_0, alpha_leak, weights_intensity, intensity_[t])
        for t in range(T):
            output_t, Ie_t, Ii_t, Ii_t2 = get_output_with_inhibition2(\
                weights_e, weights_i, intercept_e, intercept_i, UV_flow[t], alpha_leak, weights_intensity, intensity_[t])
            res_T[t, :] = output_t[:]
            Ie_T[t, :] = Ie_t[:]
            Ii_T[t, :] = Ii_t[:]
            Ii_T2[t, :] = Ii_t2[:]
    else:
        res_rest, _, _, _ = get_output_with_inhibition2(\
                weights_e, weights_i, intercept_e, intercept_i, UV_flow_0, alpha_leak)
        for t in range(T):
            output_t, Ie_t, Ii_t, Ii_t2 = get_output_with_inhibition2(\
                weights_e, weights_i, intercept_e, intercept_i, UV_flow[t], alpha_leak)
            res_T[t, :] = output_t[:]
            Ie_T[t, :] = Ie_t[:]
            Ii_T[t, :] = Ii_t[:]
            Ii_T2[t, :] = Ii_t2[:]
    
    return res_rest, res_T, Ie_T, Ii_T, Ii_T2


# general temporal filter
def general_temp_filter(n, tau_1, dt, signal_seq):
    '''
    Args:
    n: filter order, 0 means no filter
    tau_1: timescale of the filter (sec)
    dt: simulation time step (sec)
    signal_seq: signal sequence to be filtered
    
    Returns:
    filtered_sig: filtered signal, single data point
    '''
    if n == 0:
        return signal_seq[-1, :]
    elif n == 1:
        T = signal_seq.shape[0]
        ts = dt*np.arange(T)
        G_n = (1./tau_1)*np.exp(-ts/tau_1)
        G_n = G_n/G_n.sum()
        filtered_sig = np.dot(np.flip(G_n), signal_seq)
        return filtered_sig
    else:
        T = signal_seq.shape[0]
        ts = dt*np.arange(T)
        G_n = (1./np.math.factorial(n-1))*(ts**(n-1)/(tau_1**n))*np.exp(-ts/tau_1)
        G_n = G_n/G_n.sum()
        filtered_sig = np.dot(np.flip(G_n), signal_seq)
        return filtered_sig
    
    
def get_angle_matrix_lplc2(M):
    angle_matrix = np.zeros((M, M))
    _, lplc2_units_coords = opsg.get_lplc2_units(M)
    for m1 in range(M):
        for m2 in range(M):
            vec_1 = np.array(lplc2_units_coords[m1])
            vec_2 = np.array(lplc2_units_coords[m2])
            angle = opsg.get_angle_two_vectors(vec_1, vec_2)
            angle_matrix[m1, m2] = angle
    
    return angle_matrix
    
    
def plot_lplc2_overlap(M_min, M_max, M_incr):
    N = np.int(np.sqrt((M_max-M_min)/M_incr)+1)
    fig, axs = plt.subplots(N, N, figsize=(20, 20))
    n = 0
    angles_out = []
    for M in range(M_min, M_max+1, M_incr):
        angle_matrix = get_angle_matrix_lplc2(M)
        angles_min = np.zeros(M)
        for m in range(M):
            angles = angle_matrix[m, :]
            angles_min[m] = angles[np.nonzero(angles)].min()
        angle = angles_min.mean()*180/np.pi
        angles_out.append(angle)
        cL = (-angle/2., 0)
        cR = (angle/2., 0)
        circleL = Circle(cL, 30, facecolor='none', 
                        edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
        circleR = Circle(cR, 30, facecolor='none', 
                        edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
        row = np.int(n/N)
        col = n%N
        ax = axs[row, col]
        ax.add_patch(circleL)
        ax.add_patch(circleR)
        ax.scatter(cL[0], cL[1], c=[(0, 0.8, 0.8)], marker='.')
        ax.scatter(cR[0], cR[1], c=[(0, 0.8, 0.8)], marker='.')
        ax.set_xlim([-90, 90])
        ax.set_ylim([-90, 90])
        ax.set_xticks([])
        ax.set_yticks([])
        if row == N-1:
            ax.set_xlabel('degree')
        if col == 0:
            ax.set_ylabel('degree')
        ax.set_title('M = {}'.format(M))
        n = n+1
    fig.savefig('../results/lplc2_overlap.pdf', bbox_inches='tight')
    
    return angles_out
    
    
# Plot only one trained UV flow weight because of symmetry
def plot_sym_flow_weights(input_weights, mask_d,colormap, filename):
    '''
    Args:
    input_weights: steps by K*K by 4
    mask_d: disk mask
    colormap: color map of the image
    filename: filename to save
    '''
    K = int(np.sqrt(len(input_weights)))
    fig = plt.figure(figsize=(5,5))
    weights_ = input_weights[:,0].reshape((K,K))
    weights_[mask_d] = 0.

    extreme = np.amax(np.abs(weights_))
    color_norm = mpl.colors.Normalize(vmin=-extreme,vmax=extreme)
    
    plt.imshow(weights_, norm=color_norm, cmap=colormap)
    plt.colorbar()
    plt.title('rightward motion')
    plt.xticks([])
    plt.yticks([])
    
    plt.show()
    
    fig.savefig(filename,bbox_inches='tight')
    
    
# plot the grid
def plot_sym_flow_weights_grid(path, natural, folder_names, model_type, condition, weight_type, mask_d, colormap):
    row = len(folder_names)
    col = len(folder_names[0])
    fig = plt.figure(figsize=(4*col, 4*row))
    n = 1
    for r in range(row):
        for c in range(col):
            plt.subplot(row, col, n)
            weight_path = path+'/'+folder_names[r][c]+'/'+model_type+'/'+condition+'/'+'/'+weight_type+'.npy'
            input_weights = np.load(weight_path)
            plot_sym_flow_weights(input_weights, mask_d, colormap)
            n = n+1     
    plt.show()
    filename = path+'/'+model_type+'_'+weight_type+'_'+natural+'.png'
    fig.savefig(filename, bbox_inches='tight')
    
    
# get grid of initials
def get_grid_init(D, na):
    grid_init = np.zeros((np.int(2*na+1), np.int(2*na+1), 3))
    arr_c = np.zeros(3)
    arr_c[2] = D
    x_angles = np.arange(np.int(na*5), -np.int(na*5+1), -5)
    y_angles = np.arange(np.int(na*5), -np.int(na*5+1), -5)
    for i in range(np.int(2*na+1)):
        for j in range(np.int(2*na+1)):
            phi1 = np.deg2rad(5.)
            theta1 = np.deg2rad(x_angles[j])
            scale = np.rad2deg(2*np.arcsin(np.sin(phi1/2)/np.cos(theta1)))/5.
            if x_angles[j] == 90:
                r = R3d.from_euler('xzy', [x_angles[j], y_angles[i]*scale, 0], degrees=True)
            elif x_angles[j] == -90:
                r = R3d.from_euler('xzy', [x_angles[j], -y_angles[i]*scale, 0], degrees=True)
            else:
                r = R3d.from_euler('xyz', [x_angles[j], y_angles[i]*scale, 0], degrees=True)
            grid_init[i, j, :] = r.apply(arr_c)[:]
                                      
    return grid_init
                                      
                                      
# generate samples for grid
def generate_samples_grid_conv(D, D_max, na, V, R, dt, dynamics_fun, eta_1, sigma, theta_r, K, L, sample_dt, delay_dt, savepath):
    N = K*L + 4*L # Size of each frame, the additional 4*L is for spatial filtering and will be deleted afterwards.
    coord_y = np.arange(N) - (N-1)/2.
    coord_x = np.arange(N) - (N-1)/2.
    coords_y, coords_x = np.meshgrid(coord_y, -coord_x) # coordinates of each point in the frame
    dm = np.sqrt(coords_y**2 + coords_x**2) # distance matrix
    theta_matrix, phi_matrix = opsg.get_angle_matrix(theta_r, coords_x, coords_y, dm, K, L)
    coord_matrix = opsg.get_coord_matrix(phi_matrix, theta_matrix, D=1.)
    space_filter = flfd.get_space_filter(L/2, 4)
    
    Rs = np.array([R])
    grid_init = get_grid_init(D, na)
    d1 = grid_init.shape[0]
    d2 = grid_init.shape[1]   
    for i in range(d1):
        for j in range(d2):
            x, y, z = grid_init[i, j, :]
            vx, vy, vz = (V/D)*(-x), (V/D)*(-y), (V/D)*(-z)
            traj, _ = generate_one_trajectory_grid(D_max, dt, dynamics_fun, eta_1, x, y, z, vx, vy, vz)
            intensities_sample_cg_list, UV_flow_sample_list, distance_sample_list = smgnmu.generate_one_sample(\
                1, Rs, np.array([traj]), sigma, theta_r, theta_matrix, coord_matrix, space_filter, K, L, dt, sample_dt, delay_dt)
            np.save(savepath+'trajectories/traj_{}_{}'.format(i+1, j+1), [traj])
            np.save(savepath+'intensities_samples_cg/intensities_sample_cg_{}_{}'.format(i+1, j+1), intensities_sample_cg_list)
            np.save(savepath+'UV_flow_samples/UV_flow_sample_{}_{}'.format(i+1, j+1), UV_flow_sample_list)
            
            
# generate samples for grid
def generate_samples_grid_div(D, D_max, na, V, R, dt, dynamics_fun, eta_1, sigma, theta_r, K, L, sample_dt, delay_dt, savepath):
    N = K*L + 4*L # Size of each frame, the additional 4*L is for spatial filtering and will be deleted afterwards.
    coord_y = np.arange(N) - (N-1)/2.
    coord_x = np.arange(N) - (N-1)/2.
    coords_y, coords_x = np.meshgrid(coord_y, -coord_x) # coordinates of each point in the frame
    dm = np.sqrt(coords_y**2 + coords_x**2) # distance matrix
    theta_matrix, phi_matrix = opsg.get_angle_matrix(theta_r, coords_x, coords_y, dm, K, L)
    coord_matrix = opsg.get_coord_matrix(phi_matrix, theta_matrix, D=1.)
    space_filter = flfd.get_space_filter(L/2, 4)
    
    Rs = np.array([R])
    grid_init = get_grid_init(D, na)
    d1 = grid_init.shape[0]
    d2 = grid_init.shape[1]   
    for i in range(d1):
        for j in range(d2):
            x, y, z = grid_init[i, j, :]
            vx, vy, vz = (V/D)*(x), (V/D)*(y), (V/D)*(1.-z)
            traj, _ = generate_one_trajectory_grid(D_max, dt, dynamics_fun, eta_1, 0, 0, D, vx, vy, vz)
            intensities_sample_cg_list, UV_flow_sample_list, distance_sample_list = smgnmu.generate_one_sample(\
                1, Rs, np.array([traj]), sigma, theta_r, theta_matrix, coord_matrix, space_filter, K, L, dt, sample_dt, delay_dt)
            np.save(savepath+'trajectories/traj_{}_{}'.format(i+1, j+1), [traj])
            np.save(savepath+'intensities_samples_cg/intensities_sample_cg_{}_{}'.format(i+1, j+1), intensities_sample_cg_list)
            np.save(savepath+'UV_flow_samples/UV_flow_sample_{}_{}'.format(i+1, j+1), UV_flow_sample_list)
    
            
# generate samples for grid
def generate_samples_grid_para(D, D_max, na, V, R, dt, dynamics_fun, eta_1, sigma, theta_r, K, L, sample_dt, delay_dt, savepath):
    N = K*L + 4*L # Size of each frame, the additional 4*L is for spatial filtering and will be deleted afterwards.
    coord_y = np.arange(N) - (N-1)/2.
    coord_x = np.arange(N) - (N-1)/2.
    coords_y, coords_x = np.meshgrid(coord_y, -coord_x) # coordinates of each point in the frame
    dm = np.sqrt(coords_y**2 + coords_x**2) # distance matrix
    theta_matrix, phi_matrix = opsg.get_angle_matrix(theta_r, coords_x, coords_y, dm, K, L)
    coord_matrix = opsg.get_coord_matrix(phi_matrix, theta_matrix, D=1.)
    space_filter = flfd.get_space_filter(L/2, 4)
    
    Rs = np.array([R])
    grid_init = get_grid_init(D, na)
    d1 = grid_init.shape[0]
    d2 = grid_init.shape[1]   
    for i in range(d1):
        for j in range(d2):
            x, y, z = grid_init[i, j, :]
            vx, vy, vz = 0, 0, (V/D)*(1.-D)
            traj, _ = generate_one_trajectory_grid(D_max, dt, dynamics_fun, eta_1, x, y, z, vx, vy, vz)
            intensities_sample_cg_list, UV_flow_sample_list, distance_sample_list = smgnmu.generate_one_sample(\
                1, Rs, np.array([traj]), sigma, theta_r, theta_matrix, coord_matrix, space_filter, K, L, dt, sample_dt, delay_dt)
            np.save(savepath+'trajectories/traj_{}_{}'.format(i+1, j+1), [traj])
            np.save(savepath+'intensities_samples_cg/intensities_sample_cg_{}_{}'.format(i+1, j+1), intensities_sample_cg_list)
            np.save(savepath+'UV_flow_samples/UV_flow_sample_{}_{}'.format(i+1, j+1), UV_flow_sample_list)

                                      
# generate response grid
def generate_response_grid(args, y_min, y_max, na, fsize, model_path, model_type, data_path, filename, intensity=0):
    res_max = 0
    NA = np.int(na*2+1)
    for i in range(1, NA+1):
        for j in range(1, NA+1):
            file = glob.glob(data_path+'UV_flow_samples/'+'UV_flow_sample_{}_{}.npy'.format(i, j))
            UV_flow = np.load(file[0], allow_pickle=True)[0]
            res_rest, res_T, _ = get_response_over_time(args, model_path, model_type, UV_flow, intensity=0)
            res_max = np.maximum(res_max, res_T.sum(axis=1).mean())
    
    fig, axes = plt.subplots(NA+1, NA+1, figsize=(fsize, fsize))
    axes[0, 0].spines["right"].set_visible(False)
    axes[0, 0].spines["top"].set_visible(False)
    axes[0, 0].spines["bottom"].set_linewidth(1)
    axes[0, 0].spines["left"].set_linewidth(1)
    axes[0, 0].set_xticks([res_T.shape[0]*0.5, res_T.shape[0]])
    axes[0, 0].set_xticklabels([-1, 0])
    axes[0, 0].set_yticks([0, y_max])
    axes[0, 0].tick_params(direction='in', length=2, width=1, labelsize=10)
    for i in range(NA+1):
        for j in range(NA+1):
            if i == 0 or j == 0:
                if i+j > 0:
                    axes[i, j].axis('off')
            elif i != 0 and j != 0:
                file = glob.glob(data_path+'UV_flow_samples/'+'UV_flow_sample_{}_{}.npy'.format(i, j))
                UV_flow = np.load(file[0], allow_pickle=True)[0]
                res_rest, res_T, _ = get_response_over_time(args, model_path, model_type, UV_flow, intensity=0)
                rlevel = (1-np.maximum(0, res_T.sum(axis=1).mean())/res_max)*0.8
                axes[i, j].plot(np.arange(res_T.shape[0]), res_T.sum(axis=1), c=[0.8, rlevel, rlevel], linewidth=1)
                axes[i, j].plot(np.arange(res_T.shape[0]), np.ones(res_T.shape[0])*res_rest.sum(), c=[0.5, 0.5, 0.5], \
                               linestyle='--', linewidth=1)
                axes[i, j].set_xlim([res_T.shape[0]*0.4, res_T.shape[0]])
                axes[i, j].set_ylim([y_min, y_max])
                axes[i, j].axis('off')
    plt.show()
    fig.savefig(filename, bbox_inches='tight')


# generate response grid
def generate_response_multiunit(args, model_path, model_type, data_path, data_type, type_dict, subtract_baseline=True, K=12):
    M = args['M']
#     if subtract_baseline:
#         UV_flow_sample0 = [[np.zeros((1, M, K*K, 4))]]
#         res_0, _ = get_response_over_time(args, model_path, model_type, UV_flow_sample0, intensity=0)
#     else:
#         res_0 = 0.
    
    res_max_all = []
    res_T_all = []
    angles_all = []
    distances_all = []
    lplc2_units, lplc2_units_coords = opsg.get_lplc2_units(M, randomize=False)
    path_uv = data_path+'testing/'+data_type+'/UV_flow_samples/'
    path_traj = data_path+'other_info/'+data_type+'/trajectories/'
    path_distance = data_path+'other_info/'+data_type+'/distances/'
    files = glob.glob(path_uv+'*.npy')
    S = len(files)
    steps = 0
    starting_number = type_dict[data_type]
    for s in range(starting_number, starting_number+S):
        UV_flow_sample_list = np.load(path_uv+'UV_flow_sample_list_{}.npy'.format(s+1), allow_pickle=True)
        traj_list = np.load(path_traj+'traj_list_{}.npy'.format(s+1), allow_pickle=True)
        distance_list = np.load(path_distance+'distance_list_{}.npy'.format(s+1), allow_pickle=True)
        for ii in range(len(UV_flow_sample_list)):
            UV_flow_sample = np.array(UV_flow_sample_list[ii])
            traj = np.array(traj_list[ii])
            distance = np.array(distance_list[ii])
            res_rest, res_T, _, _ = get_response_over_time(args, model_path, model_type, UV_flow_sample, intensity=0)
#             res_T = res_T-res_0
            vec = traj[-2, 0, :]-traj[-1, 0, :] 
            angles = opsg.get_angles_between_lplc2_and_vec(M, vec)
            res_max_all.append(res_T.max(axis=0))
            res_T_all.append(res_T)
            angles_all.append(angles)
            distances_all.append(distance)
            steps = np.maximum(steps, res_T.shape[0])
    
    return res_max_all, res_T_all, angles_all, steps, distances_all


# generate response grid
def generate_response_multiunit_all_models(args, model_path, model_types, data_path, data_type, bin_size, max_angle):
    bins = get_index_angle(bin_size, max_angle)+1
    res_max_all_models = []
    for model_type in model_types:
        res_max_all, _, angles_all, _, _= generate_response_multiunit(args, model_path+model_type+'/', model_type, data_path, data_type)
        M = res_max_all[0].shape[0]
        print(M)
        res_out = np.zeros((2, bins))
        n = np.zeros(bins)
        for counter, res_T in enumerate(res_max_all):
            for m in range(M):
                ind = get_index_angle(bin_size, angles_all[counter][m])
                res_out[0, ind] = res_out[0, ind]+res_T[m]
                res_out[1, ind] = res_out[1, ind]+res_T[m]**2.
                n[ind] = n[ind]+1     
        res_out[0, :] = np.divide(res_out[0, :], n)
        res_out[1, :] = np.divide(res_out[1, :], n)
        res_out[1, :] = np.sqrt(res_out[1, :]-res_out[0, :]**2.)/np.sqrt(n-1)
        res_max_all_models.append(res_out)
    
    return res_max_all_models


# generate heatmap for multiunit response
def generate_heatmap_multiunit(res_T_all, angles_all, steps, bin_size, max_angle):
    M = res_T_all[0].shape[1]
    bins = get_index_angle(bin_size, max_angle)+1
    res_heatmap = np.zeros((steps, bins))
    for step in range(steps):
        n = np.zeros(bins)
        for counter, res_T in enumerate(res_T_all):
            for m in range(M):
                ind = get_index_angle(bin_size, angles_all[counter][m])
                if res_T.shape[0] >= step+1:
                    res_heatmap[-step-1, ind] = res_heatmap[-step-1, ind]+res_T[-step-1, m]
                    n[ind] = n[ind]+1     
        res_heatmap[-step-1, :] = np.divide(res_heatmap[-step-1, :], n)
        
    return np.nan_to_num(res_heatmap)


# generate heatmap for multiunit response
def generate_heatmap_multiunit_dist(res_T_all, angles_all, distances_all, bin_size_a, bin_size_d, max_angle, dt, sample_dt):
    M = res_T_all[0].shape[1]
    dist_min = 10000.
    dist_max = 0.
    for distance in distances_all:
        dist_min = np.minimum(dist_min, np.floor(distance.min()))
        dist_max = np.maximum(dist_max, np.ceil(distance.max()))
    bins1 = get_index_angle(bin_size_d, dist_max-dist_min)+1
    bins2 = get_index_angle(bin_size_a, max_angle)+1
    res_heatmap = np.zeros((bins1, bins2))
#     n = np.zeros((bins1, bins2))
    n = np.ones((bins1, bins2)) # add pseudo count
    for counter, res_T in enumerate(res_T_all):
        for m in range(M):
            for ii in range(res_T.shape[0]):
                ind1 = get_index_angle(bin_size_d, distances_all[counter][np.int(ii*sample_dt/dt)]-dist_min)
                ind2 = get_index_angle(bin_size_a, angles_all[counter][m])
                res_heatmap[ind1, ind2] = res_heatmap[ind1, ind2]+res_T[ii][m]
                n[ind1, ind2] = n[ind1, ind2]+1
    res_heatmap = np.divide(res_heatmap, n)
        
    return np.nan_to_num(res_heatmap)


# get index for angle
def get_index_angle(bin_size, angle):
    ind = np.int(angle/bin_size)
    if np.mod(angle, bin_size) == 0 and angle != 0:
        ind = ind-1
        
    return ind
    
    
# get probability
def get_probability(args, model_path, model_type, data_path, set_number, data_type):
    a = np.load(model_path + "trained_a.npy")
    b = np.load(model_path + "trained_b.npy")
    probability = []
    path_UV = data_path+'set_{}/testing/'.format(set_number)+data_type+'/UV_flow_samples/'
    files = glob.glob(path_UV+'*.npy')
    for index in np.arange(len(files)):
        UV_flow = np.load(files[index], allow_pickle=True)
        res_rest, res_T, _ = get_response_over_time(args, model_path, model_type, UV_flow, intensity=0)
        res_TM = res_T.sum(axis=1)
        prob = sigmoid_array(np.abs(a)*res_TM+b)
        probability.append(np.amax(prob))
    
    return np.array(probability)


# get probability
def get_probability2(args, model_path, model_type, data_path, set_number, data_type):
    a = np.load(model_path + "trained_a.npy")
    b = np.load(model_path + "trained_b.npy")
    probability = []
    path_UV = data_path+'set_{}/testing/'.format(set_number)+data_type+'/UV_flow_samples/'
    files = glob.glob(path_UV+'*.npy')
    for index in np.arange(len(files)):
        UV_flows = np.load(files[index], allow_pickle=True)
        for ii in range(len(UV_flows)):
            UV_flow = UV_flows[ii]
            res_rest, res_T, _ = get_response_over_time(args, model_path, model_type, UV_flow, intensity=0)
            res_TM = res_T.sum(axis=1)
            prob = sigmoid_array(np.abs(a)*res_TM+b)
            probability.append(np.mean(prob))
    
    return np.array(probability)


# get loss function
def get_loss(args, a, b, weights_e, weights_i, intercept_e, intercept_i, X, y, regu_l2):
    N = X.shape[0]
    probability_all = []
    probability_T_all = []
    res_T_all = []
    Ie_T_all = []
    Ii_T_all = []
    weights_intensity = None
    intensity = None
    for n in range(N):
        UV_flow = X[n:n+1]
        res_rest, res_T, Ie_T, Ii_T = get_response_with_inhibition2(\
                args, weights_e, weights_i, intercept_e, intercept_i, UV_flow, weights_intensity, intensity)
        res_T_all.append(res_T)
        Ie_T_all.append(Ie_T)
        Ii_T_all.append(Ii_T)
        res_TM = res_T.sum(axis=1)
        prob = sigmoid_array(np.abs(a)*res_TM+b)
        probability_all.append(np.mean(prob))
        probability_T_all.append(prob)
    loss = np.dot(y, -np.log(np.array(probability_all)))/N + np.dot(1-y, -np.log(1-np.array(probability_all)))/N \
        + regu_l2*np.power(weights_e, 2).sum()/8. + regu_l2*np.power(weights_i, 2).sum()/8.
    
    return loss, probability_all, probability_T_all, res_T_all, Ie_T_all, Ii_T_all


# get loss function
def get_loss_for_training(inputs, args, X, y, regu_l2):
    
    a = 1
    b = inputs[0]
    intercept_e = inputs[1]
    intercept_i = inputs[2]
    WE_upperhalf = inputs[3:75].reshape((6, 12))
    WI_upperhalf = inputs[75:147].reshape((6, 12))
    WE = np.concatenate((WE_upperhalf, np.flip(WE_upperhalf, axis=0)))
    WI = np.concatenate((WI_upperhalf, np.flip(WI_upperhalf, axis=0)))
    weights_e = np.zeros((144, 4))
    weights_i = np.zeros((144, 4))
    weights_e[:, 0] = WE.flatten()
    weights_i[:, 0] = WI.flatten()
    weights_e[:, 1] = np.rot90(WE, 2).flatten()
    weights_i[:, 1] = np.rot90(WI, 2).flatten()
    weights_e[:, 2] = np.rot90(WE, 1).flatten()
    weights_i[:, 2] = np.rot90(WI, 1).flatten()
    weights_e[:, 3] = np.rot90(WE, 3).flatten()
    weights_i[:, 3] = np.rot90(WI, 3).flatten()
    
    N = X.shape[0]
    probability_all = []
    probability_T_all = []
    res_T_all = []
    Ie_T_all = []
    Ii_T_all = []
    weights_intensity = None
    intensity = None
    for n in range(N):
        UV_flow = X[n:n+1]
        res_rest, res_T, Ie_T, Ii_T = get_response_with_inhibition2(\
                args, weights_e, weights_i, intercept_e, intercept_i, UV_flow, weights_intensity, intensity)
        res_T_all.append(res_T)
        Ie_T_all.append(Ie_T)
        Ii_T_all.append(Ii_T)
        res_TM = res_T.sum(axis=1)
        prob = sigmoid_array(np.abs(a)*res_TM+b)
        probability_all.append(np.mean(prob))
        probability_T_all.append(prob)
    loss = np.dot(y, -np.log(np.array(probability_all)))/N + np.dot(1-y, -np.log(1-np.array(probability_all)))/N \
        + regu_l2*np.power(weights_e, 2).sum()/8. + regu_l2*np.power(weights_i, 2).sum()/8.
    
    return loss


# get the velocity 
def get_velocity_train_test(data_path, data_types, save_path=None):
    '''
    Args:
    data_path: path where the data is saved.
    data_types: data types
    '''
    LL = len(data_types)
    N_train = np.zeros(LL)
    N_test = np.zeros(LL)
    X_training = []
    y_training = []
    for data_type in data_types:
        path = data_path+'training/'+data_type+'/UV_flow_samples/'
        if os.path.isdir(path):
            files = glob.glob(path+'*.npy')
            for index in np.arange(len(files)):
                rn_arrs = np.load(files[index], allow_pickle=True)
                for rn_arr in rn_arrs:
                    steps = len(rn_arr)
                    if steps > 0:
                        if data_type == 'hit':
                            N_train[0] = N_train[0]+1
                        elif data_type == 'miss':
                            N_train[1] = N_train[1]+1
                        elif data_type == 'retreat':
                            N_train[2] = N_train[2]+1
                        elif data_type == 'rotation':
                            N_train[3] = N_train[3]+1
                        M = len(rn_arr[0])
                        XM = np.zeros(M)
                        for m in range(M):
                            vm = 0
                            for step in range(steps):
                                vm = np.maximum(vm, rn_arr[step][m].max())
                            XM[m] = vm
                        X_training.append(XM)
                        if data_type == 'hit':
                            y_training.append([1])
                        else:
                            y_training.append([0])
                    else:
                        print('Empty array!')
    X_testing = []
    y_testing = []
    for data_type in data_types:
        path = data_path+'testing/'+data_type+'/UV_flow_samples/'
        if os.path.isdir(path):
            files = glob.glob(path+'*.npy')
            for index in np.arange(len(files)):
                rn_arrs = np.load(files[index], allow_pickle=True)
                for rn_arr in rn_arrs:
                    if data_type == 'hit':
                        N_test[0] = N_test[0]+1
                    elif data_type == 'miss':
                        N_test[1] = N_test[1]+1
                    elif data_type == 'retreat':
                        N_test[2] = N_test[2]+1
                    elif data_type == 'rotation':
                        N_test[3] = N_test[3]+1
                    steps = len(rn_arr)
                    M = len(rn_arr[0])
                    XM = np.zeros(M)
                    for m in range(M):
                        vm = 0
                        for step in range(steps):
                            vm = np.maximum(vm, rn_arr[step][m].max())
                        XM[m] = vm
                    X_testing.append(XM)
                    if data_type == 'hit':
                        y_testing.append([1])
                    else:
                        y_testing.append([0])
    X_training = np.array(X_training, np.float32)
    y_training = np.array(y_training, np.float32)
    X_testing = np.array(X_testing, np.float32)
    y_testing = np.array(y_testing, np.float32)
    if save_path:
        np.savez(save_path, N_train, N_test, X_training, y_training, X_testing, y_testing)
        
    return N_train.astype(np.int), N_test.astype(np.int), X_training, y_training, X_testing, y_testing


# get the velocity design matrix
def get_velocity_nonzero(data_path, data_types):
    '''
    Args:
    data_path: path where the data is saved.
    data_types: data types
    '''
    V_hit = []
    V_miss = []
    V_retreat = []
    V_rotation = []
    for data_type in data_types:
        path = data_path+'training/'+data_type+'/UV_flow_samples/'
        if os.path.isdir(path):
            files = glob.glob(path+'*.npy')
            for index in np.arange(len(files)):
                rn_arrs = np.load(files[index], allow_pickle=True)
                for rn_arr in rn_arrs:
                    steps = len(rn_arr)
                    M = len(rn_arr[0])
                    for m in range(M):
                        for step in range(steps):
                            if rn_arr[step][m].shape[0] > 1:
                                if data_type == 'hit':
                                    r_to_append = rn_arr[step][m].flatten()
                                    V_hit.extend(r_to_append[r_to_append>0])
                                elif data_type == 'miss':
                                    r_to_append = rn_arr[step][m].flatten()
                                    V_miss.extend(r_to_append[r_to_append>0])
                                elif data_type == 'retreat':
                                    r_to_append = rn_arr[step][m].flatten()
                                    V_retreat.extend(r_to_append[r_to_append>0])
                                elif data_type == 'rotation':
                                    r_to_append = rn_arr[step][m].flatten()
                                    V_rotation.extend(r_to_append[r_to_append>0])
    
    return np.array(V_hit).flatten(), np.array(V_miss).flatten(), np.array(V_retreat).flatten(), np.array(V_rotation).flatten()


def display_weights(\
    K, trained_weights_e_all, trained_weights_i_all, has_inhibition, sample_list, n_sample, mask_d, colormap_e, colormap_i):
    fig = plt.figure(figsize=(2*n_sample*0.8, 0.8))
    for ind, label in enumerate(sample_list):
        trained_weights_e = trained_weights_e_all[label, :].reshape((K, K))
        trained_weights_e[mask_d] = 0.
        extreme_e = np.amax(np.abs(trained_weights_e))
        if has_inhibition:
            color_norm_e = mpl.colors.Normalize(vmin=0, vmax=extreme_e)
        else:
            color_norm_e = mpl.colors.Normalize(vmin=-extreme_e, vmax=extreme_e)
        plt.subplot(1, 2*n_sample, ind+1)
        plt.imshow(trained_weights_e, norm=color_norm_e, cmap=colormap_e)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar(orientation='horizontal', fraction=.03)
        if has_inhibition:
            trained_weights_i = trained_weights_i_all[label, :].reshape((K, K))
            trained_weights_i[mask_d] = 0.
            extreme_i = np.amax(np.abs(trained_weights_i))
            color_norm_i = mpl.colors.Normalize(vmin=0, vmax=extreme_i)
            plt.subplot(1, 2*n_sample, ind+1+len(sample_list))
            plt.imshow(trained_weights_i, norm=color_norm_i, cmap=colormap_i)
            plt.xticks([])
            plt.yticks([])
            plt.colorbar(orientation='horizontal', fraction=.03)
    plt.show()
    
    return fig


def display_weights2(K, input_weights, mask_d, colormap, M):
    N = input_weights.shape[0]
    fig = plt.figure(figsize=(M*4, 4))
    
    weights_matrix = input_weights[0, :].reshape((K, K))
    weights_matrix[mask_d] = 0.
    extreme = np.amax(np.abs(input_weights))
    color_norm = mpl.colors.Normalize(vmin=-extreme, vmax=extreme)
    plt.subplot(1, M, 1)
    plt.imshow(weights_matrix, norm=color_norm, cmap=colormap)
    plt.colorbar()
    
    for m in range(1, M-1):
        weights_matrix = input_weights[m*np.int((N-2)/(M-2)), :].reshape((K, K))
        weights_matrix[mask_d] = 0.
        color_norm = mpl.colors.Normalize(vmin=-extreme, vmax=extreme)
        plt.subplot(1, M, m+1)
        plt.imshow(weights_matrix, norm=color_norm, cmap=colormap)
        plt.colorbar()
        
    weights_matrix = input_weights[-1, :].reshape((K, K))
    weights_matrix[mask_d] = 0.
    color_norm = mpl.colors.Normalize(vmin=-extreme, vmax=extreme)
    plt.subplot(1, M, M)
    plt.imshow(weights_matrix, norm=color_norm, cmap=colormap)
    plt.colorbar()
    plt.show()
    
    return fig
    
    
# leaky relu
def get_leaky_relu(alpha_leak, x):
    if x >= 0:
        return x
    else:
        return alpha_leak*x
    

def sigmoid_array(x): 
    return 1 / (1 + np.exp(-x))


def plot_KLapoetke_intensities_extended(intensities, dt, savepath, name):
    intensities = np.squeeze(intensities)
    steps = intensities.shape[0]
    combined_movies_list = []
    for step in range(steps):
        save_intensity(intensities[step], step, savepath+name)
        combined_movies = imageio.imread(savepath+name+'_{}.png'.format(step+1))
        combined_movies_list.append(combined_movies)
    movie_name = savepath+name+'.mp4'
    imageio.mimsave(movie_name, combined_movies_list, fps = np.int(1/dt))

    
def save_intensity(intensity, step, save_name):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(intensity, cmap='gray_r', vmin=0, vmax=1)
#         ax.vlines(np.arange(-0.55, N, step=N/K), -0.55, N-0.45, color='salmon', linewidth=.5)
#         ax.hlines(np.arange(-0.55, N, step=N/K), -0.55, N-0.45, color='salmon', linewidth=.5)
    fig.savefig(save_name+'_{}.png'.format(step+1))
    plt.close(fig)
    
    
def plot_KLapoetke_UV_flows(UV_flow, K, L, pad, dt, savepath, name):
    leftup_corners = opsg.get_leftup_corners(K, L, pad)
    steps = UV_flow.shape[0]
    combined_movies_u_list = []
    combined_movies_v_list = []
    for step in range(steps):
        cf_u, cf_v = flfd.set_flow_fields_on_frame(UV_flow[step], leftup_corners, K, L, pad)
        u_flow = cf_u[0, :, :]
        v_flow = cf_v[0, :, :]
        save_uv_flow(u_flow, step, savepath+name+'_u')
        save_uv_flow(v_flow, step, savepath+name+'_v')
        combined_movies_u = imageio.imread(savepath+name+'_u_{}.png'.format(step+1))
        combined_movies_u_list.append(combined_movies_u)
        combined_movies_v = imageio.imread(savepath+name+'_v_{}.png'.format(step+1))
        combined_movies_v_list.append(combined_movies_v)
    movie_name_u = savepath+name+'_u.mp4'
    imageio.mimsave(movie_name_u, combined_movies_u_list, fps = np.int(1/dt))
    movie_name_v = savepath+name+'_v.mp4'
    imageio.mimsave(movie_name_v, combined_movies_v_list, fps = np.int(1/dt))
        
        
def save_uv_flow(flow, step, save_name):
    myheat = LinearSegmentedColormap.from_list('br', ["b", "w", "r"], N=256)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    vmin=np.min(flow.flatten())
    vmax=np.max(flow.flatten())
    vmax = np.max([-vmin, vmax])
    if vmax < 1e-6:
        vmax = 1
    vmax = 0.08
    ax.imshow(flow, cmap=myheat, vmin=-vmax, vmax=vmax)
    PCM=ax.get_children()[9] #get the mappable, the 1st and the 2nd are the x and y axes
    plt.colorbar(PCM, ax=ax)
#     ax.vlines(np.arange(-0.55, N, step=N/K), -0.55, N-0.45, color='salmon', linewidth=.5)
#     ax.hlines(np.arange(-0.55, N, step=N/K), -0.55, N-0.45, color='salmon', linewidth=.5)
    fig.savefig(save_name+'_{}.png'.format(step+1))
    plt.close(fig)
    
    
def plot_UV_flow_tem(UV_flow, p, image_name_u, image_name_v, movie_name_u, movie_name_v, fps, leftup_corners, K, L, pad, vmax=0.08):
    steps = UV_flow.shape[0]
    combined_movies_u_list = []
    combined_movies_v_list = []
    for step in range(steps):
        if step%p == 0:
            cf_u, cf_v = flfd.set_flow_fields_on_frame2(UV_flow[step, :, :, :], leftup_corners, K, L, pad)
            u_flow = cf_u[0, :, :]
            v_flow = cf_v[0, :, :]
            gKse.save_uv_flow(u_flow, step, image_name_u, vmax)
            gKse.save_uv_flow(v_flow, step, image_name_v, vmax)
            combined_movies_u = imageio.imread(image_name_u+'_{}.png'.format(step+1))
            combined_movies_u_list.append(combined_movies_u)
            combined_movies_v = imageio.imread(image_name_v+'_{}.png'.format(step+1))
            combined_movies_v_list.append(combined_movies_v)
    imageio.mimsave(movie_name_u, combined_movies_u_list, fps = fps)
    imageio.mimsave(movie_name_v, combined_movies_v_list, fps = fps)
    

###############################
### multi-unit illustration ###
###############################

def get_normalized_vector(vec):
    norm = np.linalg.norm(vec)
    if norm == 0: 
        raise Exception('The norm of the vec is 0!')
    else:
        return vec/norm
    
    
def get_normalized(v0, v):
    dv = v-v0
    d = np.sqrt(np.dot(dv, dv))
    if d > 0:
        v_out = v0+dv/d*0.1
    else:
        print('The distance is 0!')
    
    return v_out
    

def get_c_vec_list(angle_m, angle_r, N):
    x_axis = np.array([[1., 0, 0]])
    y_axis = np.array([[0, 1., 0]])
    z_axis = np.array([[0, 0, 1.]])
    x_axis = opsg.get_rotated_axes(angle_m, x_axis)
    y_axis = opsg.get_rotated_axes(angle_m, y_axis)
    z_axis = opsg.get_rotated_axes(angle_m, z_axis)
    r = np.tan(angle_r)
    c_vec_list = []
    for n in range(N):
        theta_ = 2*np.pi*(n/N)
        r_vec = x_axis*r*np.sin(theta_) + y_axis*r*np.cos(theta_)
        c_vec = z_axis + r_vec
        c_vec = get_normalized_vector(c_vec)
        c_vec_list.append(c_vec)
    c_vec_list.append(c_vec_list[0])
    
    return np.array(c_vec_list)

    
def get_coords_circumf(x_axis, y_axis, z_axis, angle, N):
    """
    Get the coordinates of the N points on the circumference that are angle away
    from the z_axis, where x_axis, y_axis, z_axis are the local coordinates.
    """
    r = np.tan(angle)
    c_vec_list = []
    for n in range(N):
        theta_ = 2*np.pi*(n/N)
        r_vec = x_axis*r*np.sin(theta_) + y_axis*r*np.cos(theta_)
        c_vec = z_axis + r_vec
        c_vec = get_normalized_vector(c_vec)
        c_vec_list.append(c_vec)
    return c_vec_list


def get_angles_on_map(vec):
    """
    Get the angles on a 2d map given the coordinates vec in the 3d space.
    """
    rn = 2*(np.random.random()-0.5)*0
    vec = vec + rn
    lat = np.arctan(vec[0]/(np.sqrt(vec[1]**2+vec[2]**2)))
    vec_yz = np.array([0, vec[1], vec[2]])
    vec_yz = get_normalized_vector(vec_yz)
    lon = np.sign(vec[1])*np.arccos(np.dot(vec_yz, np.array([0, 0, 1])))
    
    return np.array([lon, lat])


def get_projection_one_unit(angle_mc, angle_m, angle_r, N):
    """
    Get the 2d projection coordinates (longitudinal and lattitude angles) 
    given the angle of this unit angle_q and the central unit angle_qc.
    """
    x_axis = np.array([[1., 0, 0]])
    y_axis = np.array([[0, 1., 0]])
    z_axis = np.array([[0, 0, 1.]])
    x_axis = opsg.get_rotated_axes(angle_m, x_axis)
    y_axis = opsg.get_rotated_axes(angle_m, y_axis)
    z_axis = opsg.get_rotated_axes(angle_m, z_axis)
    c_vec_list = get_coords_circumf(x_axis, y_axis, z_axis, angle_r, N)
    
    angles_on_map = []
    a = 1e-3
    x_axis_rot = opsg.get_rotated_coordinates(-angle_mc, x_axis)
    y_axis_rot = opsg.get_rotated_coordinates(-angle_mc, y_axis)
    z_axis_rot = opsg.get_rotated_coordinates(-angle_mc, z_axis)
    angles_on_map.append(get_angles_on_map(a*x_axis_rot[0]+z_axis_rot[0]))
    angles_on_map.append(get_angles_on_map(a*y_axis_rot[0]+z_axis_rot[0]))
    angles_on_map.append(get_angles_on_map(z_axis_rot[0]))
    
    for c_vec in c_vec_list:
        c_vec_rot = opsg.get_rotated_coordinates(-angle_mc, c_vec)
        angles_on_map.append(get_angles_on_map(c_vec_rot[0]))
    c_vec_rot = opsg.get_rotated_coordinates(-angle_mc, c_vec_list[0])
    angles_on_map.append(get_angles_on_map(c_vec_rot[0]))
        
    return angles_on_map


def get_segments(angles_on_map):
    """
    Get the discontinued segments of the projected lines,
    due to the cut of the sphere.
    """
    L = len(angles_on_map)
    indL = 0
    indR = 0
    segments_list = []
    while indR <= L-1:
        valueR1 = angles_on_map[indR][0]
        if indR < L-1:
            valueR2 = angles_on_map[indR+1][0]
        if np.abs(valueR1-valueR2) >= np.pi or indR == L-1:
            break
        indR = indR+1
    # for a circle, stop now
    if indR == L-1: 
        segments_list.append(np.array(angles_on_map))
    # not a circle, but cut in half, loop toward the opposite direction
    else: 
        indF_1 = indR
        indR = 0
        while indR >= -(L-1):
            valueR1 = angles_on_map[indR][0]
            if indR > -(L-1):
                valueR2 = angles_on_map[indR-1][0]
            if np.abs(valueR1-valueR2) >= np.pi or indR == -(L-1):
                break
            indR = indR-1
        indF_2 = indR
        current_seg = np.concatenate((np.array(angles_on_map[indF_2:]), np.array(angles_on_map[:indF_1+1])))
        segments_list.append(current_seg)
        # check if the current_seg contains all the points, 
        # if not, this means there is one more segment.
        if len(current_seg) < L: 
            segments_list.append(np.array(angles_on_map[indF_1+1:indF_2]))
            
    return segments_list


def get_segments_ball(segments_list):
    """
    Get the discontinued segments of the projected lines,
    due to the cut of the sphere. For the ball.
    """
    
    # I will do something artificial, just for better illustrations.
    # first segment
    if len(segments_list) > 1:
        seg1 = segments_list[0]
        if np.abs(seg1[0][0] - seg1[-1][0]) <= 2 * np.pi / 20:
            x1_mean = (seg1[0][0] + seg1[-1][0]) / 2.
            seg1[0][0] = x1_mean
            seg1[-1][0] = x1_mean
        else:
            x1 = seg1[0][0]
            y1 = seg1[0][1]
            x2 = seg1[-1][0]
            y2 = seg1[-1][1]
            if np.abs(x1) > np.abs(x2):
                x_new = np.pi * np.sign(x1)
            else:
                x_new = np.pi * np.sign(x2)
            if np.abs(y1) > np.abs(y2):
                y_new = np.pi / 2 * np.sign(y1)
            else:
                y_new = np.pi / 2 * np.sign(y2)
            seg1 = np.concatenate((np.array([[x_new, y_new]]), seg1, np.array([[x_new, y_new]])))

        # second segment
        seg2 = segments_list[1]
        if np.abs(seg2[0][0] - seg2[-1][0]) <= 2 * np.pi / 20:
            x2_mean = (seg2[0][0] + seg2[-1][0]) / 2.
            seg2[0][0] = x2_mean
            seg2[-1][0] = x2_mean
        else:
            x1 = seg2[0][0]
            y1 = seg2[0][1]
            x2 = seg2[-1][0]
            y2 = seg2[-1][1]
            if np.abs(x1) > np.abs(x2):
                x_new = np.pi * np.sign(x1)
            else:
                x_new = np.pi * np.sign(x2)
            if np.abs(y1) > np.abs(y2):
                y_new = np.pi / 2 * np.sign(y1)
            else:
                y_new = np.pi / 2 * np.sign(y2)
            seg2 = np.concatenate((np.array([[x_new, y_new]]), seg2, np.array([[x_new, y_new]])))
        seg_list = [seg1, seg2]
    else:
        seg_list = segments_list
            
    return seg_list


# def get_segments(angles_on_map):
#     L = len(angles_on_map)
#     indL = 0
#     indR = 0
#     segments_list = []
#     while indR <= L-1:
#         valueR1 = angles_on_map[indR][0]
#         if indR < L-1:
#             valueR2 = angles_on_map[indR+1][0]
#         if np.abs(valueR1-valueR2) >= 3 or indR == L-1:
#             segments_list.append(np.array(angles_on_map[indL:indR+1]))
#             indL = indR+1
#         indR = indR+1
#     return segments_list


def get_upper_lower_bound(segment, center):
    segment_diff = np.diff(segment[:, 0])
    L = len(segment_diff)
    for ll in range(L):
        if ll <= L-2 and segment_diff[ll]*segment_diff[ll+1] <= 0:
            segment = np.roll(segment, -ll-1, axis=0)
            segment_diff = np.diff(segment[:, 0])
            for lll in range(L):
                if lll <= L-2 and segment_diff[lll]*segment_diff[lll+1] <= 0:
                    segment1 = segment[:lll+2, :]
                    segment2 = segment[lll+2:, :]
                    if segment[0, 1] >= segment[-1, 1]:
                        segment_upper = segment1
                        segment_lower = segment2
                        if segment_upper[:, 1].min() > segment_lower[:, 1].max():
                            segment_upper = np.concatenate((segment_upper, segment2[0:1, :]), axis=0)
                            segment_upper[-1, 1] = segment_lower[:, 1].max()
                    else:
                        segment_upper = segment2
                        segment_lower = segment1
                        if segment_upper[:, 1].min() > segment_lower[:, 1].max():
                            segment_lower = np.concatenate((segment_lower, segment2[0:1, :]), axis=0)
                            segment_lower[-1, 1] = segment_upper[:, 1].min()
            break
        if ll == L-1:
            if center[1] >= 0:
                segment_upper = np.ones_like(segment)*np.pi/2
                segment_upper[:, 0] = segment[:, 0]
                segment_lower = segment[:, :]
            else:
                segment_upper = segment[:, :]
                segment_lower = np.ones_like(segment)*np.pi/2
                segment_lower[:, 0] = segment[:, 0]

    return segment_lower, segment_upper


def get_response_on_map(M, N, res_T, t, res_m=1):
    lplc2_units = opsg.get_lplc2_units_xy_angles(M)
    mc = sorted_list[0]
    angle_mc = lplc2_units[mc]
    angle_r = np.pi/6
        
    R = 1
    x, y, z = traj[t*np.int(0.1/0.01), 0, :]
    angle_rb = opsg.get_angular_size(x, y, z, R)
    angle_b = opsg.get_xy_angles(x, y, z)
        
    fig = plt.figure()
    for m in range(M):
        angle_m = lplc2_units[sorted_list[m]]
        angles_on_map = get_projection_one_unit(angle_mc, angle_m, angle_r, N)
        plt.scatter(angles_on_map[0][0], angles_on_map[0][1], s=6+np.int(500*res_T[t, sorted_list[m]]/res_m), c='m', marker='o')
        segments_list = get_segments(angles_on_map[1:])
        for segment in segments_list:
            plt.plot(segment[:, 0], segment[:, 1], c='m', alpha=0.2)
    angles_on_map_b = get_projection_one_unit(angle_mc, angle_b, angle_rb, N)
    segments_list_b = get_segments(angles_on_map_b[1:])
    for segment in segments_list_b:
        plt.fill_between(segment[:, 0], segment[:, 1], facecolor=[0.5, 0.5, 0.5], edgecolor=None, alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('{} lplc2 units'.format(M))
    fig.savefig('../results/response_on_map_{}.png'.format(t+1))
    plt.show()
    
    
#############################
### Calculate the Hessian ###
#############################
        
def get_gradient_hessian(args, train_flow_snapshots, train_labels_snapshots, parameters_in):
    # inputs
    M = args['M']
    K = args['K']
    UV_flow = tf.compat.v1.placeholder(tf.float32, [None, None, M, K*K, 4], name = 'UV_flow')
    step_weights = tf.compat.v1.placeholder(tf.float32, [None, None], name='step_weights')
    labels = tf.compat.v1.placeholder(tf.float32, [None, None], name = 'labels')
    intensity = None
    if args['use_intensity']:
        intensity = tf.compat.v1.placeholder(tf.float32, 
            [None, K*K], name='intensity')

    # parameters
    weights_intensity = None
    tau_1 = args['tau']
    a = 1.
    
    parameters_in_tf = tf.convert_to_tensor(parameters_in)
    
    b = tf.slice(parameters_in_tf, [0], [1]) 
    intercept_e = tf.slice(parameters_in_tf, [1], [1]) 
    intercept_i = tf.slice(parameters_in_tf, [2], [1])  
    weights_e_raw = tf.slice(parameters_in_tf, [3], [72]) 
    weights_i_raw = tf.slice(parameters_in_tf, [75], [72])
    
    weights_e = expand_weight(weights_e_raw, (K+1)//2, K, K%2==0) 
    weights_i = expand_weight(weights_i_raw, (K+1)//2, K, K%2==0)

    # Regularization
    l1_l2_regu_we = tf.contrib.layers.l1_l2_regularizer(scale_l1=args["l1_regu_we"], scale_l2=args["l2_regu_we"], scope=None)
    l1_l2_regu_wi = tf.contrib.layers.l1_l2_regularizer(scale_l1=args["l1_regu_wi"], scale_l2=args["l2_regu_wi"], scope=None)
    l1_l2_regu_a = tf.contrib.layers.l1_l2_regularizer(scale_l1=args["l1_regu_a"], scale_l2=args["l2_regu_a"], scope=None)

    # loss and error function
    loss, error_step, error_trajectory, _, probabilities = \
    lplc2.loss_error_C_inhibitory2(\
         args, weights_e, weights_i, weights_intensity, intercept_e, intercept_i, UV_flow, intensity, \
         labels, tau_1, a, b, step_weights, l1_l2_regu_we, l1_l2_regu_wi, l1_l2_regu_a)

    # Calculate gradient
    Grad = flatten(tf.gradients(loss, parameters_in_tf))
#     # Calculate gradient of gradient
#     Grad_grad = tf.map_fn(lambda v: get_Hv_op(Grad, parameters_in_tf, v), \
#                           tf.eye(tf.shape(parameters_in_tf)[0], tf.shape(parameters_in_tf)[0]))
#     # Calculate hessian
#     Hess = tf.hessians(loss, parameters_in_tf)
    
    step_weights_samples = np.ones_like(train_labels_snapshots)
    with tf.compat.v1.Session() as sess:
        loss_res, grad_res, weights_e_out, weights_i_out = sess.run([loss, Grad, weights_e, weights_i], \
                 {UV_flow:train_flow_snapshots, labels:train_labels_snapshots, step_weights: step_weights_samples})

    return loss_res, grad_res, weights_e_out, weights_i_out


def get_Hv_op(grad, params, v):
    vprod = tf.math.multiply(grad, tf.stop_gradient(v))
    Hv_op = flatten(tf.gradients(vprod, params))
    
    return Hv_op


def flatten(params):
    """
    Flattens the list of tensor(s) into a 1D tensor

    Args:
        params: List of model parameters (List of tensor(s))

    Returns:
        A flattened 1D tensor
    """
    return tf.concat([tf.reshape(_params, [-1]) \
                      for _params in params], axis=0)


def expand_weight(weights, num_row, num_column, is_even):
    weights_reshaped = tf.reshape(weights, [num_row, num_column, 1])
    if is_even:
        assert(num_row*2 == num_column)
        weights_flipped = tf.concat([weights_reshaped, 
            tf.reverse(weights_reshaped, axis=[0])], axis=0)
    else:
        assert((num_row*2-1)  == num_column)
        weights_flipped = tf.concat([weights_reshaped, 
            tf.reverse(weights_reshaped[:-1], axis=[0])], axis=0)
    weights_reshaped_back = tf.reshape(weights_flipped, [num_column**2, 1])
    return weights_reshaped_back


#####################
### miscellaneous ###
#####################
def get_error_binomial(n_sample1, n_sample2):
    N = n_sample1+n_sample2
    p = n_sample1/N
    q = n_sample2/N
    error = np.sqrt(N*p*q)
    
    return error


def get_propagated_error(error, n_sample1, n_sample2):
    ratio = n_sample1/n_sample2
    ratio_error = ratio*np.sqrt((error/n_sample1)**2+(error/n_sample2)**2)
    
    return ratio_error