#!/usr/bin/env python
# coding: utf-8


"""
Generate auc curves for a specific model specified by M.
"""


import sys
sys.path.append('../stimulus_core/')
sys.path.append('../models_core/')
sys.path.append('../helper/')

import warnings
warnings.filterwarnings(action='ignore')

import os
import numpy as np
import tensorflow as tf
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
import seaborn as sns
from  matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R3d
import time
import importlib
import glob
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from scipy.spatial import distance_matrix, distance
from scipy.cluster import hierarchy

import dynamics_3d as dn3d
import optical_signal as opsg
import flow_field as flfd
import helper_functions as hpfn


has_inhibition = True
figure_path = '../results/FiguresForPaper/trained_model_and_response/'
if not os.path.exists(figure_path):
    os.mkdir(figure_path)
    
    
importlib.reload(hpfn)

M = 256

# model
data_path = '../results/FiguresForPaper/model_clustering/clusterings/'
subdir_all = np.load(data_path+'subdir_all_M{}.npy'.format(M), allow_pickle=True)

args = {}
args["n"] = 0
args["dt"] = 0.01
args["use_intensity"] = False
args["symmetrize"] = True
args['K'] = 12
args['alpha_leak'] = 0.0
args['M'] = M
args['temporal_filter'] = False
args['tau_1'] = 1.

model_types = ['excitatory', 'excitatory_WNR', 'inhibitory1', 'inhibitory2']


# sample data
set_number = np.int(1000+M)
sample_data_path = '../../data/loom/multi_lplc2_scal200_D5_5/'.format(set_number)
data_types = ['hit', 'miss', 'retreat', 'rotation']

y_scores_all = []
fpr_all = []
tpr_all = []
thres_roc_all = []
roc_auc_all = []
precision_all = []
recall_all = []
thres_pr_all = []
pr_auc_all = []
for ind, subdir in enumerate(subdir_all):
    if ind%1 == 0:
        if os.path.exists(subdir+'/trained_weights_e.npy'):
            model_path = subdir+'/'
            model_type = model_types[3]
            data_type = data_types[0]
            prob_hit = hpfn.get_probability2(args, model_path, model_type, sample_data_path, set_number, data_type)
            model_type = model_types[3]
            data_type = data_types[1]
            prob_miss = hpfn.get_probability2(args, model_path, model_type, sample_data_path, set_number, data_type)
            model_type = model_types[3]
            data_type = data_types[2]
            prob_retreat = hpfn.get_probability2(args, model_path, model_type, sample_data_path, set_number, data_type)
            model_type = model_types[3]
            data_type = data_types[3]
            prob_rot = hpfn.get_probability2(args, model_path, model_type, sample_data_path, set_number, data_type)

            y_scores = np.concatenate((prob_hit, prob_miss, prob_retreat, prob_rot))
            y_labels = np.zeros(len(y_scores))
            y_labels[:len(prob_hit)] = 1
            fpr, tpr, _ = metrics.roc_curve(y_labels, y_scores)
            precision, recall, _ = metrics.precision_recall_curve(y_labels, y_scores)
            roc_auc = metrics.roc_auc_score(y_labels, y_scores)
            pr_auc = metrics.average_precision_score(y_labels, y_scores)
            
            y_scores_all.append(y_scores)

            fpr_all.append(fpr)
            tpr_all.append(tpr)
            roc_auc_all.append(roc_auc)

            precision_all.append(precision)
            recall_all.append(recall)
            pr_auc_all.append(pr_auc)

        
np.save(figure_path+'y_scores_M{}'.format(M), y_scores_all)
np.save(figure_path+'fpr_all_M{}'.format(M), fpr_all)
np.save(figure_path+'tpr_all_M{}'.format(M), tpr_all)
np.save(figure_path+'roc_auc_all_M{}'.format(M), roc_auc_all)
np.save(figure_path+'precision_all_M{}'.format(M), precision_all)
np.save(figure_path+'recall_all_M{}'.format(M), recall_all)
np.save(figure_path+'pr_auc_all_M{}'.format(M), pr_auc_all)