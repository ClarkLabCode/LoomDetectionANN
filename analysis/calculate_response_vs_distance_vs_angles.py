#!/usr/bin/env python


"""
Here, we look at reponses of the trained models vs distance and angles.
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
if not os.path.exists(figure_path+'response_vs_distance/'):
    os.makedirs(figure_path+'response_vs_distance/')

for M in [32, 256]:
    data_path = figure_path+'model_clustering/clusterings/'
    model_folders = np.load(data_path+'model_folders_M{}.npy'.format(M), allow_pickle=True)

    # model
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

    set_number = np.int(1000+M)
    data_path = '/Volumes/Baohua/data_on_hd/loom/multi_lplc2_D5_L4_exp/set_{}/'.format(set_number)
    data_types = ['hit', 'miss', 'retreat', 'rotation']

    model_types = ['excitatory', 'excitatory_WNR', 'inhibitory1', 'inhibitory2']
    filter_types = ['outward', 'inward']
    model_type = model_types[3]

    dt = 0.01
    sample_dt = 0.03
    bin_size_a = 5
    bin_size_d = 0.2
    max_angle = 180
    type_dict = {'hit':100, 'miss':50, 'retreat':50, 'rotation':200}

    # outward
    filter_type = filter_types[0]
    model_path = model_folders[0][2]+'/'
    data_type = data_types[0]
    res_max_all_hit_n, res_T_all_hit_n, angles_all_hit_n, steps_hit_n, distances_all_hit_n = hpfn.generate_response_multiunit(args, model_path, model_type, data_path, data_type, type_dict)
    save_file = figure_path+'response_vs_distance/'+'results'+'_'+filter_type+'_'+data_type+'_M{}'.format(M)
    np.savez(save_file, res_max_all_hit_n, res_T_all_hit_n, angles_all_hit_n, steps_hit_n, distances_all_hit_n)
    data_type = data_types[1]
    res_max_all_miss_n, res_T_all_miss_n, angles_all_miss_n, steps_miss_n, distances_all_miss_n = hpfn.generate_response_multiunit(args, model_path, model_type, data_path, data_type, type_dict)
    save_file = figure_path+'response_vs_distance/'+'results'+'_'+filter_type+'_'+data_type+'_M{}'.format(M)
    np.savez(save_file, res_max_all_miss_n, res_T_all_miss_n, angles_all_miss_n, steps_miss_n, distances_all_miss_n)
    data_type = data_types[2]
    res_max_all_retreat_n, res_T_all_retreat_n, angles_all_retreat_n, steps_retreat_n, distances_all_retreat_n = hpfn.generate_response_multiunit(args, model_path, model_type, data_path, data_type, type_dict)
    save_file = figure_path+'response_vs_distance/'+'results'+'_'+filter_type+'_'+data_type+'_M{}'.format(M)
    np.savez(save_file, res_max_all_retreat_n, res_T_all_retreat_n, angles_all_retreat_n, steps_retreat_n, distances_all_retreat_n)

    res_heatmap_hit_n = hpfn.generate_heatmap_multiunit_dist(res_T_all_hit_n, angles_all_hit_n, distances_all_hit_n, bin_size_a, bin_size_d, max_angle, dt, sample_dt)
    res_heatmap_miss_n = hpfn.generate_heatmap_multiunit_dist(res_T_all_miss_n, angles_all_miss_n, distances_all_miss_n, bin_size_a, bin_size_d, max_angle, dt, sample_dt)
    res_heatmap_retreat_n = hpfn.generate_heatmap_multiunit_dist(res_T_all_retreat_n, angles_all_retreat_n, distances_all_retreat_n, bin_size_a, bin_size_d, max_angle, dt, sample_dt)
    save_file = figure_path+'response_vs_distance/'+'heatmap'+'_'+filter_type+'_M{}'.format(M)
    np.savez(save_file, res_heatmap_hit_n, res_heatmap_miss_n, res_heatmap_retreat_n)

    # inward
    filter_type = filter_types[1]
    model_path = model_folders[1][2]+'/'
    data_type = data_types[0]
    res_max_all_hit_r, res_T_all_hit_r, angles_all_hit_r, steps_hit_r, distances_all_hit_r = hpfn.generate_response_multiunit(args, model_path, model_type, data_path, data_type, type_dict)
    save_file = figure_path+'response_vs_distance/'+'results'+'_'+filter_type+'_'+data_type+'_M{}'.format(M)
    np.savez(save_file, res_max_all_hit_r, res_T_all_hit_r, angles_all_hit_r, steps_hit_r, distances_all_hit_r)
    data_type = data_types[1]
    res_max_all_miss_r, res_T_all_miss_r, angles_all_miss_r, steps_miss_r, distances_all_miss_r = hpfn.generate_response_multiunit(args, model_path, model_type, data_path, data_type, type_dict)
    save_file = figure_path+'response_vs_distance/'+'results'+'_'+filter_type+'_'+data_type+'_M{}'.format(M)
    np.savez(save_file, res_max_all_miss_r, res_T_all_miss_r, angles_all_miss_r, steps_miss_r, distances_all_miss_r)
    data_type = data_types[2]
    res_max_all_retreat_r, res_T_all_retreat_r, angles_all_retreat_r, steps_retreat_r, distances_all_retreat_r = hpfn.generate_response_multiunit(args, model_path, model_type, data_path, data_type, type_dict)
    save_file = figure_path+'response_vs_distance/'+'results'+'_'+filter_type+'_'+data_type+'_M{}'.format(M)
    np.savez(save_file, res_max_all_retreat_r, res_T_all_retreat_r, angles_all_retreat_r, steps_retreat_r, distances_all_retreat_r)

    res_heatmap_hit_r = hpfn.generate_heatmap_multiunit_dist(res_T_all_hit_r, angles_all_hit_r, distances_all_hit_r, bin_size_a, bin_size_d, max_angle, dt, sample_dt)
    res_heatmap_miss_r = hpfn.generate_heatmap_multiunit_dist(res_T_all_miss_r, angles_all_miss_r, distances_all_miss_r, bin_size_a, bin_size_d, max_angle, dt, sample_dt)
    res_heatmap_retreat_r = hpfn.generate_heatmap_multiunit_dist(res_T_all_retreat_r, angles_all_retreat_r, distances_all_retreat_r, bin_size_a, bin_size_d, max_angle, dt, sample_dt)
    save_file = figure_path+'response_vs_distance/'+'heatmap'+'_'+filter_type+'_M{}'.format(M)
    np.savez(save_file, res_heatmap_hit_r, res_heatmap_miss_r, res_heatmap_retreat_r)

