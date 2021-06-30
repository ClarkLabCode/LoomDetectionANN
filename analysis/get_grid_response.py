#!/usr/bin/env python


"""
Here, we look at the responses of the trained models to a grid of incoming hit signals.
"""


import sys
sys.path.append('../stimulus_core/')
sys.path.append('../models_core/')
sys.path.append('../helper/')

import os
import numpy as np
import glob

import helper_functions as hpfn


figure_path = '/Volumes/Baohua/research/loom_detection/results/final_figures_for_paper_exp/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)
if not os.path.exists(figure_path+'grid_response/'):
    os.makedirs(figure_path+'grid_response/')
print(figure_path)
data_path = '/Volumes/Baohua/data_on_hd/loom/grid_conv2_M1_L100_par_exp/'

for M in [32, 256]:

    # response, outward filters
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
    model_types = ['excitatory', 'excitatory_WNR', 'inhibitory1', 'inhibitory2']
    filter_types = ['outward', 'inward']
    model_type = model_types[3]
    filter_type = filter_types[in_or_out]
    model_path = model_folders[in_or_out][2]+'/'
    print(model_path)

    res_rest_all = []
    res_T_all = []
    prob_all = []
    prob_rest_all = []
    for i in range(31):
        res_rest_tem = []
        res_T_tem = []
        prob_tem = []
        prob_rest_tem = []
        for j in range(72):
            file = glob.glob(data_path+'UV_flow_samples/'+'UV_flow_sample_{}_{}.npy'.format(i+1, j+1))
            UV_flow = np.load(file[0], allow_pickle=True)[0]
            res_rest, res_T, prob, prob_rest = hpfn.get_response_over_time(args, model_path, model_type, UV_flow, intensity=0)
            res_rest_tem.append(res_rest)
            res_T_tem.append(res_T)
            prob_tem.append(prob)
            prob_rest_tem.append(prob_rest)
        res_rest_all.append(res_rest_tem)
        res_T_all.append(res_T_tem)
        prob_all.append(prob_tem)
        prob_rest_all.append(prob_rest_tem)
    save_file = figure_path+'grid_response/'+'results'+'_'+filter_type+'_M{}'.format(M)
    np.savez(save_file, res_rest_all, res_T_all, prob_rest_all, prob_all)


    # response, inward filters
    model_folders = np.load(figure_path + '/model_clustering/clusterings/model_folders_M{}.npy'.format(M), \
                            allow_pickle=True)
    # model
    in_or_out = 1 # 0: outward; 1: inward
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
    model_types = ['excitatory', 'excitatory_WNR', 'inhibitory1', 'inhibitory2']
    filter_types = ['outward', 'inward']
    model_type = model_types[3]
    filter_type = filter_types[in_or_out]
    model_path = model_folders[in_or_out][2]+'/'

    res_rest_all = []
    res_T_all = []
    prob_all = []
    prob_rest_all = []
    for i in range(31):
        res_rest_tem = []
        res_T_tem = []
        prob_tem = []
        prob_rest_tem = []
        for j in range(72):
            file = glob.glob(data_path+'UV_flow_samples/'+'UV_flow_sample_{}_{}.npy'.format(i+1, j+1))
            UV_flow = np.load(file[0], allow_pickle=True)[0]
            res_rest, res_T, prob, prob_rest = hpfn.get_response_over_time(args, model_path, model_type, UV_flow, intensity=0)
            res_rest_tem.append(res_rest)
            res_T_tem.append(res_T)
            prob_tem.append(prob)
            prob_rest_tem.append(prob_rest)
        res_rest_all.append(res_rest_tem)
        res_T_all.append(res_T_tem)
        prob_all.append(prob_tem)
        prob_rest_all.append(prob_rest_tem)
    save_file = figure_path+'grid_response/'+'results'+'_'+filter_type+'_M{}'.format(M)
    np.savez(save_file, res_rest_all, res_T_all, prob_rest_all, prob_all)
        