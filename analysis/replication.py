#!/usr/bin/env python


"""
This file generates various of replications of experiment.
"""

import sys
sys.path.append('../stimulus_core/')
sys.path.append('../models_core/')
sys.path.append('../helper/')

import os
import numpy as np

import dynamics_3d as dn3d
import flow_field as flfd
import helper_functions as hpfn
import optical_signal as opsg
import samples_generation_multi_units as smgnmu


figure_path = '/Volumes/Baohua/research/loom_detection/results/revision/' # where to store the results
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

data_folder =  '/Volumes/Baohua/data_on_hd/loom/Klapoetke_stimuli_L50_dt_0.01_exp/' # where the data are stored

####### Klapoetke #######
has_inhibition = True
if not os.path.exists(figure_path+'klap_replication/'):
    os.mkdir(figure_path+'klap_replication/')

figuretypes1 = ['2F', '3e', '3f', '4A', '4C', '4E', '4G']

figuretypes2 = \
[['2Fa', '2Fb', '2Fc', '2Fd', '2Fe', '2Ff', '2Fg'], \
 ['3e1', '3e2', '3e3', '3e4', '3e5', '3e6', '3e7', '3e8', '3e9', '3e10', '3e11', '3e12'], \
 ['3f1', '3f2', '3f3', '3f4', '3f5', '3f6', '3f7', '3f8', '3f9', '3f10', '3f11', '3f12'], \
 ['4Aa', '4Ab', '4Ac'], \
 ['4Ca', '4Cb', '4Cc', '4Cd', '4Ce'], \
 ['4E10', '4E20', '4E60'], \
 ['4Ga', '4Gb', '4Gc', '4Gd', '4Ge']]

labels_2F = np.zeros(7)
labels_2F[0] = 1

labels_3e = np.zeros(12)
labels_3f = np.zeros(12)

labels_4A = np.zeros(3)
labels_4A[0] = 1
labels_4C = np.zeros(5)
labels_4E = np.zeros(3)
labels_4G = np.zeros(5)

labels_T = [labels_2F, labels_3e, labels_3f, labels_4A, labels_4C, labels_4E, labels_4G]

ymin = -0.1
ymax_dict = {}
ymax_dict['3e'] = 0.4
ymax_dict['3f'] = 1.5
ymax_dict['4A'] = 1
ymax_dict['4C'] = 0.3
ymax_dict['4E'] = 0.6
ymax_dict['4G'] = 0.1


model_types = ['linear', 'rectified inhibition']
model_type = model_types[1]

filter_types = ['outward', 'inward']
filter_type = filter_types[0]

M = 256
args = {}
args['M'] = M
args["n"] = 0
args["dt"] = 0.01
args["use_intensity"] = False
args["symmetrize"] = True
args['K'] = 12
args['alpha_leak'] = 0.0
args['temporal_filter'] = False
args['tau_1'] = 1.

# save as '_M{}_1' if the inhibitory filter has the bulb on the right
# save as '_M{}_2' if the inhibitory filter does not have the bulb on the right
# save as '_M{}_3' if the inhibitory filter is missing

# 16 (0): outward without (with) the right bulb in inhibitory filter, delta, M = 256
# 17 (0): outward without (with) the right bulb in inhibitory filter, exponential, M = 256
# 5: outward without the inhibitory filter, exponential, M = 256
model_dict = {0:1, 17:2, 5:3}

for outward_model in [17, 0, 5]:
    saved_results_path = figure_path+'model_clustering/clusterings/'
    model_folders = np.load(saved_results_path+'model_folders_M{}.npy'.format(M), allow_pickle=True)
    model_path = model_folders[0][outward_model]+'/'
    print(f'The model is at {model_path}')
    data_path =  data_folder + 'Klapoetke_UV_flows/2F/Klapoetke_UV_flows_2Fa.npy'
    UV_flow = np.load(data_path, allow_pickle=True)
    _, res_T_2Fa, _, _ = hpfn.get_response_over_time(args, model_path, model_type, UV_flow, intensity=0)
    res_T_2Fa_max = (res_T_2Fa-res_T_2Fa[0]).max()

    if os.path.isdir(model_path):
        for ind, figuretype in enumerate(figuretypes1):
            data_path =  data_folder + 'Klapoetke_UV_flows/'+figuretype+'/'
            UV_flow_files = [data_path+'Klapoetke_UV_flows_'+file+'.npy' for file in figuretypes2[ind]]
            data_path = data_folder + 'Klapoetke_intensities_cg/'+figuretype+'/'
            intensity_files = [data_path+'Klapoetke_intensities_cg_'+file+'.npy' for file in figuretypes2[ind]]

            K = args['K']
            alpha_leak = args['alpha_leak']
            a = np.load(model_path + "trained_a.npy")
            b = np.load(model_path + "trained_b.npy")
            intercept_e = np.load(model_path + "trained_intercept_e.npy")
            tau_1 = np.load(model_path + "trained_tau_1.npy")
            weights_e = np.load(model_path + "trained_weights_e.npy")
            if model_type == 'rectified inhibition':
                weights_i = np.load(model_path + "trained_weights_i.npy")
                intercept_i = np.load(model_path + "trained_intercept_i.npy")
            weights_intensity = None
            if args["use_intensity"]:
                weights_intensity = np.load(model_path + "trained_weights_intensity.npy")
            if args['temporal_filter']:
                args['n'] = 1

            res_rest_all = []
            res_T_all = []
            Ie_T_all = []
            Ii_T_all = []
            Ii_T2_all = []
            for ind, UV_flow_file in enumerate(UV_flow_files):
                UV_flow = np.load(UV_flow_file)
                frame_intensity = None
                if args['use_intensity']:
                    frame_intensity = np.load(intensity_files[ind])
                    assert(len(frame_intensity) == len(UV_flow_))
                if model_type == 'linear':
                    res_rest, res_T = hpfn.get_response_linear(\
                        args, weights_e, intercept_e, UV_flow, weights_intensity, intensity=0)
                elif model_type == 'rectified inhibition':
                    res_rest, res_T, Ie_T, Ii_T, Ii_T2 = hpfn.get_response_with_rectified_inhibition(\
                        args, weights_e, weights_i, intercept_e, intercept_i, UV_flow, weights_intensity, intensity=0)
                res_rest_all.append(res_rest)
                res_T_all.append(res_T)
                Ie_T_all.append(Ie_T)
                Ii_T_all.append(Ii_T)
                Ii_T2_all.append(Ii_T2)
    #             print(Ie_T.max())
    #             print(Ii_T.max())

            # save as '_M{}_1' if the inhibitory filter has the bulb on the right
            # save as '_M{}_2' if the inhibitory filter does not have the bulb on the right
            filename = figure_path+'klap_replication/'+figuretype+f'_M{M}_{model_dict[outward_model]}'
            np.savez(filename, res_rest_all, res_T_all, Ie_T_all, Ii_T_all, Ii_T2_all)

    figuretype = figuretypes1[0]
    filename = figure_path+'klap_replication/'+figuretype+f'_M{M}_{model_dict[outward_model]}.npz'
    loaded_file = np.load(filename, allow_pickle=True)
    res_rest_all = loaded_file['arr_0']
    res_T_all = loaded_file['arr_1']
    Ie_T_all = loaded_file['arr_2']
    Ii_T_all = loaded_file['arr_3']
    Ii_T2_all = loaded_file['arr_4']
    N = len(res_T_all)
    res_max_2Fa = np.max(res_T_all[0])
    res_max_2Fa2 = np.max(Ie_T_all[0]-Ii_T_all[0])
    print(f'Max excitatory response of the looming disk is {np.max(Ie_T_all[0])}.')
    print(f'Max inhibitory response of the looming disk is {np.max(Ii_T_all[0])}.')
    print(f'Max excitatory-inhibitory response of the looming disk is {np.max(Ie_T_all[0]-Ii_T_all[0])}.')                

    
    
####### Linear law and size tuning #######
# responses, only use weights trained from the model with M = 256
M = 256
model_folders = np.load(figure_path + '/model_clustering/clusterings/model_folders_M{}.npy'.format(M), \
                        allow_pickle=True)
# model
in_or_out = 0 # 0: outward; 1: inward
args = {}
args["n"] = 0
args["dt"] = 0.03
args["use_intensity"] = False
args["symmetrize"] = True
args['K'] = 12
args['alpha_leak'] = 0.0
args['M'] = M
args['temporal_filter'] = False
args['tau_1'] = 1.
model_types = ['linear', 'rectified inhibition']
filter_types = ['outward', 'inward']
model_type = model_types[1]
filter_type = filter_types[in_or_out]

# 16 (0): outward without (with) the right bulb in inhibitory filter, delta
# 17 (0): outward without (with) the right bulb in inhibitory filter, exponential
model_dict2 = {17:'', 0:'_sup1', 5:'_sup2'} 
for outward_model in [17, 0, 5]:
    model_path = model_folders[in_or_out][outward_model]+'/' 
    data_type = 'hit'
    data_path = '/Volumes/Baohua/data_on_hd/loom/linear_law_exp/'+data_type+'/'

    # R_over_v_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, \
    #                  0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
    R_over_v_list = [0.01, 0.02, 0.04, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]

    for M in [1, 256]:
        res_T_all = []
        distance_T_all = []
        for R_over_v in R_over_v_list:
            file = data_path+f'UV_flow_sample_{R_over_v}_M{M}.npy'
            UV_flow = np.load(file, allow_pickle=True)
            res_rest, res_T, prob, prob_rest = hpfn.get_response_over_time(args, model_path, model_type, UV_flow, intensity=0)
            res_T_all.append(res_T.sum(axis=1))

            file = data_path+f'distance_{R_over_v}_M{M}.npy'
            distance = np.load(file, allow_pickle=True)
            distance_T_all.append(distance)

        save_path = figure_path + '/linear_law/' + data_type + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        np.save(save_path+f'distance_T_all_M{M}'+model_dict2[outward_model], distance_T_all)
        np.save(save_path+f'res_T_all_M{M}'+model_dict2[outward_model], res_T_all)
