#!/usr/bin/env python


"""
Here, we look at the predicted probability of hit given certain models.
"""

import sys
sys.path.append('../stimulus_core/')
sys.path.append('../models_core/')
sys.path.append('../helper/')

import os
import numpy as np

import helper_functions as hpfn


figure_path = '/Volumes/Baohua/research/loom_detection/results/final_figures_for_paper_exp/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)
if not os.path.exists(figure_path+'probability_of_hit/'):
    os.makedirs(figure_path+'probability_of_hit/')


for M in [1, 32, 256]:
    K = 12
    L = 4
    KK = K**2
    pad = 2*L
    NNs = 10
    set_number = np.int(1000+M)
    data_path = figure_path+'model_clustering/clusterings/'
    model_folders = np.load(data_path+'model_folders_M{}.npy'.format(M), allow_pickle=True)
    data_path = '/Volumes/Baohua/data_on_hd/loom/multi_lplc2_D5_L4_exp/set_{}/'.format(set_number)
    data_types = ['hit', 'miss', 'retreat', 'rotation']

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
    model_types = ['excitatory', 'excitatory_WNR', 'inhibitory1', 'inhibitory2', 'linear']
    model_type = model_types[3]
    filter_types = ['outward', 'inward']

    for ind, filter_type in enumerate(filter_types):
        model_path = model_folders[ind][2]+'/'
        print(filter_type, model_path)
        prob_mean_ = []
        prob_max_ = []
        prob_all_ = []
        res_T_all_ = []
        dist_min_ = []
        prob_rest_ = []
        for data_type in data_types:
            print(f'Now: M = {M}', filter_type, data_type)
            prob_mean_tem = []
            prob_max_tem = []
            prob_all_tem = []
            res_T_all_tem = []
            dist_min_tem = []
            if data_type == 'hit':
                snn1 = 0
                snn2 = 300
                pre_N = 1000
            elif data_type == 'miss':
                snn1 = 0
                snn2 = 150
                pre_N = 500
            elif data_type == 'retreat':
                snn1 = 0
                snn2 = 150
                pre_N = 500
            elif data_type == 'rotation':
                snn1 = 0
                snn2 = 600
                pre_N = 2000
            if M in [1, 2, 4]:
                snn2 = np.int(snn2 * 8 / M)
                pre_N = np.int(pre_N * 8 / M)
            for sn in range(snn1, snn2):
                # load data
                sample_number = sn
                sample_list_number = np.int((pre_N+sample_number)//NNs+1)
                path = data_path+'testing'+'/'+data_type+'/'
                UV_flow_sample_list = np.load(path+'UV_flow_samples/UV_flow_sample_list_{}.npy'.format(sample_list_number), \
                                              allow_pickle=True)
                UV_flow_sample = UV_flow_sample_list[sample_number%NNs-1]
                if data_type == 'miss':
                    path_dist = data_path+'other_info'+'/'+data_type+'/'+'distances/'
                    dist_list = np.load(path_dist+'distance_list_{}.npy'.format(sample_list_number), allow_pickle=True)
                    dist = dist_list[sample_number%NNs-1]
                    dist_min_tem.append(dist.min())
                elif data_type == 'hit':
                    dist_min_tem.append(0.)
                elif data_type == 'retreat':
                    dist_min_tem.append(7.)
                else:
                    dist_min_tem.append(6.)

                # get res and prob
                res_rest, res_T, prob, prob_rest = hpfn.get_response_over_time(args, model_path, model_type, UV_flow_sample)
                res_T_all_tem.append(res_T)
                prob_mean_tem.append(prob.mean())
                prob_max_tem.append(prob.max())
                prob_all_tem.append(prob)

            dist_min_.append(dist_min_tem)
            res_T_all_.append(res_T_all_tem)
            prob_mean_.append(prob_mean_tem)
            prob_max_.append(prob_max_tem)
            prob_all_.append(prob_all_tem)
            prob_rest_.append(prob_rest)

        save_file = figure_path+'probability_of_hit/'+'results'+'_'+filter_type+'_M{}'.format(M)
        np.savez(save_file, dist_min_, res_T_all_, prob_mean_, prob_max_, prob_all_, prob_rest_)


