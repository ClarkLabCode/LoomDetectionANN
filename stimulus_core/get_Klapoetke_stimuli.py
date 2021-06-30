#!/usr/bin/env python

"""
Get all the stimuli in Klapoetke et al's paper (Nature, 2017)
"""

import sys
sys.path.append('../stimulus_core/')
sys.path.append('../models_core/')
sys.path.append('../helper/')

import os
import numpy as np
import imageio
import importlib
import glob
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap
import time

import optical_signal as opsg
import flow_field as flfd


foldertypes1 = ['2F', '3e', '3f', '4A', '4C', '4E', '4G']

foldertypes2 = ['2F'] * 7 + ['3e'] * 12 + ['3f'] * 12 + ['4A'] * 3 + ['4C'] * 5 + ['4E'] * 3 + ['4G'] * 5

foldertypes3 = \
['2F/2Fa', '2F/2Fb', '2F/2Fc', '2F/2Fd', '2F/2Fe', '2F/2Ff', '2F/2Fg', \
 '3e/3e1', '3e/3e2', '3e/3e3', '3e/3e4', '3e/3e5', '3e/3e6', '3e/3e7', '3e/3e8', '3e/3e9', '3e/3e10', '3e/3e11', '3e/3e12', \
 '3f/3f1', '3f/3f2', '3f/3f3', '3f/3f4', '3f/3f5', '3f/3f6', '3f/3f7', '3f/3f8', '3f/3f9', '3f/3f10', '3f/3f11', '3f/3f12', \
 '4A/4Aa', '4A/4Ab', '4A/4Ac', \
 '4C/4Ca', '4C/4Cb', '4C/4Cc', '4C/4Cd', '4C/4Ce', \
 '4E/4E10', '4E/4E20', '4E/4E60', \
 '4G/4Ga', '4G/4Gb', '4G/4Gc', '4G/4Gd', '4G/4Ge']

filetypes1 = \
['2Fa', '2Fb', '2Fc', '2Fd', '2Fe', '2Ff', '2Fg', \
 '3e1', '3e2', '3e3', '3e4', '3e5', '3e6', '3e7', '3e8', '3e9', '3e10', '3e11', '3e12', \
 '3f1', '3f2', '3f3', '3f4', '3f5', '3f6', '3f7', '3f8', '3f9', '3f10', '3f11', '3f12', \
 '4Aa', '4Ab', '4Ac', \
 '4Ca', '4Cb', '4Cc', '4Cd', '4Ce', \
 '4E10', '4E20', '4E60', \
 '4Ga', '4Gb', '4Gc', '4Gd', '4Ge']

Klapoetke_intensities_names = ['Klapoetke_intensities_'+filetype+'.npy' for filetype in filetypes1]

Klapoetke_intensities_extended_names = ['Klapoetke_intensities_extended_'+filetype+'.npy' for filetype in filetypes1]

Klapoetke_intensities_cg_names = ['Klapoetke_intensities_cg_'+filetype+'.npy' for filetype in filetypes1]

Klapoetke_UV_flows_names = ['Klapoetke_UV_flows_'+filetype+'.npy' for filetype in filetypes1]

Klapoetke_intensities_frame_names = ['Klapoetke_intensities_'+filetype for filetype in filetypes1]

Klapoetke_intensities_extended_frame_names = ['Klapoetke_intensities_extended_'+filetype for filetype in filetypes1]

Klapoetke_intensities_cg_frame_names = ['Klapoetke_intensities_cg_'+filetype for filetype in filetypes1]

Klapoetke_U_flows_frame_names = ['Klapoetke_U_flows_'+filetype for filetype in filetypes1]

Klapoetke_V_flows_frame_names = ['Klapoetke_V_flows_'+filetype for filetype in filetypes1]

Klapoetke_intensities_movie_names = ['Klapoetke_intensities_'+filetype+'.mp4' for filetype in filetypes1]

Klapoetke_intensities_extended_movie_names = ['Klapoetke_intensities_extended_'+filetype+'.mp4' for filetype in filetypes1]

Klapoetke_intensities_cg_movie_names = ['Klapoetke_intensities_cg_'+filetype+'.mp4' for filetype in filetypes1]

Klapoetke_U_flows_movie_names = ['Klapoetke_U_flows_'+filetype+'.mp4' for filetype in filetypes1]

Klapoetke_V_flows_movie_names = ['Klapoetke_V_flows_'+filetype+'.mp4' for filetype in filetypes1]


def get_Klapoetke_intensities(K, L, pad, dt, savepaths):
    get_Klapoetke_intensities_2Fa(savepaths[0], K, L, pad, dt)
    get_Klapoetke_intensities_2Fb(savepaths[1], K, L, pad, dt)
    get_Klapoetke_intensities_2Fc(savepaths[2], K, L, pad, dt)
    get_Klapoetke_intensities_2Fd(savepaths[3], K, L, pad, dt)
    get_Klapoetke_intensities_2Fe(savepaths[4], K, L, pad, dt)
    get_Klapoetke_intensities_2Ff(savepaths[5], K, L, pad, dt)
    get_Klapoetke_intensities_2Fg(savepaths[6], K, L, pad, dt)

    get_Klapoetke_intensities_3e(savepaths[7:19], K, L, pad, dt)
    get_Klapoetke_intensities_3f(savepaths[19:31], K, L, pad, dt)
    
    get_Klapoetke_intensities_4Aa(savepaths[31], K, L, pad, dt)
    get_Klapoetke_intensities_4Ab(savepaths[32], K, L, pad, dt)
    get_Klapoetke_intensities_4Ac(savepaths[33], K, L, pad, dt)
    get_Klapoetke_intensities_4Ca(savepaths[34], K, L, pad, dt)
    get_Klapoetke_intensities_4Cb(savepaths[35], K, L, pad, dt)
    get_Klapoetke_intensities_4Cc(savepaths[36], K, L, pad, dt)
    get_Klapoetke_intensities_4Cd(savepaths[37], K, L, pad, dt)
    get_Klapoetke_intensities_4Ce(savepaths[38], K, L, pad, dt)
    get_Klapoetke_intensities_4E10(savepaths[39], K, L, pad, dt)
    get_Klapoetke_intensities_4E20(savepaths[40], K, L, pad, dt)
    get_Klapoetke_intensities_4E60(savepaths[41], K, L, pad, dt)
    get_Klapoetke_intensities_4Ga(savepaths[42], K, L, pad, dt)
    get_Klapoetke_intensities_4Gb(savepaths[43], K, L, pad, dt)
    get_Klapoetke_intensities_4Gc(savepaths[44], K, L, pad, dt)
    get_Klapoetke_intensities_4Gd(savepaths[45], K, L, pad, dt)
    get_Klapoetke_intensities_4Ge(savepaths[46], K, L, pad, dt)
    
    
def get_Klapoetke_intensities_extended(K, L, pad, dt, loadpaths, savepaths, names1, names2):
    leftup_corners = opsg.get_leftup_corners(K, L, pad)
    add_time = np.int(2/dt)
    for ind, name in enumerate(names1):
        frame = np.load(loadpaths[ind]+names1[ind])
        frame_ext = np.zeros((frame.shape[0]+2*add_time, frame.shape[1], frame.shape[2], frame.shape[3]))
        frame_ext[add_time:-add_time, :, :, :] = frame[:, :, :, :]
        for ii in range(add_time):
            frame_ext[ii, :, :, :] = frame[0, :, :, :]
            frame_ext[-ii-1, :, :, :] = frame[-1, :, :, :]
        np.save(savepaths[ind]+names2[ind], frame_ext)
    
    
def get_Klapoetke_intensities_cg(K, L, pad, loadpaths, savepaths, names1, names2):
    leftup_corners = opsg.get_leftup_corners(K, L, pad)
    for ind, name in enumerate(names1):
        frame = np.load(loadpaths[ind]+names1[ind])
        frame_cg = get_intensities_cg(K, L, pad, leftup_corners, frame)
        np.save(savepaths[ind]+names2[ind], frame_cg)
        
        
def get_Klapoetke_UV_flows(space_filter, K, L, pad, dt, delay_dt, loadpaths, savepaths, names1, names2):
    leftup_corners = opsg.get_leftup_corners(K, L, pad)
    for ind, name in enumerate(names1):
        frame = np.load(loadpaths[ind]+names1[ind])
        UV_flow = get_UV_flows_exp(space_filter, K, L, pad, dt, delay_dt, leftup_corners, frame)
        np.save(savepaths[ind]+names2[ind], UV_flow)


def plot_KLapoetke_intensities(dt, p, loadpaths, savepaths1, savepaths2, names1, names2, names3):
    for ind, name in enumerate(names1):
        intensities = np.load(loadpaths[ind]+names1[ind])
        intensities = np.squeeze(intensities)
        steps = intensities.shape[0]
        combined_movies_list = []
        for step in range(steps):
            if step % p == 0:
                save_intensity(intensities[step], step, savepaths2[ind]+names2[ind])
                combined_movies = imageio.imread(savepaths2[ind]+names2[ind]+'_{}.png'.format(step+1))
                combined_movies_list.append(combined_movies)
        movie_name = savepaths1[ind] + names3[ind]
        imageio.mimsave(movie_name, combined_movies_list, fps = np.int(1/dt))
        

def plot_KLapoetke_intensities_extended(dt, p, loadpaths, savepaths1, savepaths2, names1, names2, names3):
    for ind, name in enumerate(names1):
        intensities = np.load(loadpaths[ind]+names1[ind])
        intensities = np.squeeze(intensities)
        steps = intensities.shape[0]
        combined_movies_list = []
        for step in range(steps):
            if step%p == 0:
                save_intensity(intensities[step], step, savepaths2[ind]+names2[ind])
                combined_movies = imageio.imread(savepaths2[ind]+names2[ind]+'_{}.png'.format(step+1))
                combined_movies_list.append(combined_movies)
        movie_name = savepaths1[ind] + names3[ind]
        imageio.mimsave(movie_name, combined_movies_list, fps = np.int(1/dt))
        
        
def plot_KLapoetke_intensities_cg(K, dt, p, loadpaths, savepaths1, savepaths2, names1, names2, names3):
    for ind, name in enumerate(names1):
        intensities = np.load(loadpaths[ind]+names1[ind])
        intensities = get_reshape(K, intensities)
        steps = intensities.shape[0]
        combined_movies_list = []
        for step in range(steps):
            if step%p == 0:
                save_intensity(intensities[step], step, savepaths2[ind]+names2[ind])
                combined_movies = imageio.imread(savepaths2[ind]+names2[ind]+'_{}.png'.format(step+1))
                combined_movies_list.append(combined_movies)
        movie_name = savepaths1[ind] + names3[ind]
        imageio.mimsave(movie_name, combined_movies_list, fps = np.int(1/dt))


def plot_KLapoetke_UV_flows(K, L, pad, dt, p, loadpaths, savepaths1, savepath2, names1, names2, names3, names4, names5):
    leftup_corners = opsg.get_leftup_corners(K, L, pad)
    for ind, name in enumerate(names1):
        UV_flow = np.load(loadpaths[ind]+names1[ind])
        steps = UV_flow.shape[0]
        combined_movies_u_list = []
        combined_movies_v_list = []
        for step in range(steps):
            if step%p == 0:
                cf_u, cf_v = flfd.set_flow_fields_on_frame2(UV_flow[step, :, :, :], leftup_corners, K, L, pad)
                u_flow = cf_u[0, :, :]
                v_flow = cf_v[0, :, :]
                save_uv_flow(u_flow, step, savepaths2[ind]+names2[ind])
                save_uv_flow(v_flow, step, savepaths2[ind]+names3[ind])
                combined_movies_u = imageio.imread(savepaths2[ind]+names2[ind]+'_{}.png'.format(step+1))
                combined_movies_u_list.append(combined_movies_u)
                combined_movies_v = imageio.imread(savepaths2[ind]+names3[ind]+'_{}.png'.format(step+1))
                combined_movies_v_list.append(combined_movies_v)
        movie_name_u = savepaths1[ind] + names4[ind]
        imageio.mimsave(movie_name_u, combined_movies_u_list, fps = np.int(1/dt))
        movie_name_v = savepaths1[ind] + names5[ind]
        imageio.mimsave(movie_name_v, combined_movies_v_list, fps = np.int(1/dt))
        


def get_Klapoetke_intensities_2Fa(savepath, K, L, pad, dt):
    N = K * L
    ro = (N - 1) / 2.
    co = (N - 1) / 2.
    R = 2 * L
    v = 2 * L
    
    Klapoetke_intensities_2Fa = []
    while R <= (N + 1) / 2:
        one_disk = get_one_disk(K, L, pad, ro, co, R)
        Klapoetke_intensities_2Fa.append(one_disk)
        R = R + v * dt
    
    Klapoetke_intensities_2Fa = np.array(Klapoetke_intensities_2Fa)
    np.save(savepath+'Klapoetke_intensities_2Fa', Klapoetke_intensities_2Fa)
    
    
def get_Klapoetke_intensities_2Fb(savepath, K, L, pad, dt):
    N = K * L
    ro = (N - 1) / 2.
    co = -L
    theta_a = np.pi * 0.
    L1 = L * 1
    L2 = (N - 1) / 2.
    L3 = L * 1
    L4 = (N - 1) / 2.
    v = L * 4
    
    Klapoetke_intensities_2Fb = []
    while co <= N + L:
        one_bar = get_one_bar(K, L, pad, ro, co, theta_a, L1, L2, L3, L4)
        Klapoetke_intensities_2Fb.append(one_bar)
        co = co + v * dt
        
    Klapoetke_intensities_2Fb = np.array(Klapoetke_intensities_2Fb)
    np.save(savepath+'Klapoetke_intensities_2Fb', Klapoetke_intensities_2Fb)
    
    
def get_Klapoetke_intensities_2Fc(savepath, K, L, pad, dt):
    N = K * L
    ro = (N - 1) / 2.
    co = -L
    theta_a = np.pi * 0.
    L1 = 0
    L2 = (N - 1) / 2.
    L3 = 0
    L4 = (N - 1) / 2.
    v = L * 4
    
    Klapoetke_intensities_2Fc = []
    while L1 <= N - co:
        one_bar = get_one_bar(K, L, pad, ro, co, theta_a, L1, L2, L3, L4)
        Klapoetke_intensities_2Fc.append(one_bar)
        L1 = L1 + v * dt
        
    Klapoetke_intensities_2Fc = np.array(Klapoetke_intensities_2Fc)
    np.save(savepath+'Klapoetke_intensities_2Fc', Klapoetke_intensities_2Fc)
    
    
def get_Klapoetke_intensities_2Fd(savepath, K, L, pad, dt):
    N = K * L
    ro = -L
    co = (N - 1) / 2.
    theta_a = np.pi * 1.5
    L1 = L * 1
    L2 = (N - 1) / 2.
    L3 = L * 1
    L4 = (N - 1) / 2.
    v = L * 4
    
    Klapoetke_intensities_2Fd = []
    while ro <= N + L:
        one_bar = get_one_bar(K, L, pad, ro, co, theta_a, L1, L2, L3, L4)
        Klapoetke_intensities_2Fd.append(one_bar)
        ro = ro + v * dt
        
    Klapoetke_intensities_2Fd = np.array(Klapoetke_intensities_2Fd)
    np.save(savepath+'Klapoetke_intensities_2Fd', Klapoetke_intensities_2Fd)
    
    
def get_Klapoetke_intensities_2Fe(savepath, K, L, pad, dt):
    N = K * L
    ro = -L
    co = (N - 1) / 2.
    theta_a = np.pi * 1.5
    L1 = 0
    L2 = (N - 1) / 2.
    L3 = 0
    L4 = (N - 1) / 2.
    v = L * 4
    
    Klapoetke_intensities_2Fe = []
    while L1 <= N - ro:
        one_bar = get_one_bar(K, L, pad, ro, co, theta_a, L1, L2, L3, L4)
        Klapoetke_intensities_2Fe.append(one_bar)
        L1 = L1 + v * dt
        
    Klapoetke_intensities_2Fe = np.array(Klapoetke_intensities_2Fe)
    np.save(savepath+'Klapoetke_intensities_2Fe', Klapoetke_intensities_2Fe)
    
    
def get_Klapoetke_intensities_2Ff(savepath, K, L, pad, dt):
    T = 5
    k = 2. * np.pi / (4 * L)
    w = 2. * np.pi / 1.
    t = 0.
    
    Klapoetke_intensities_2Ff = []
    while t <= T:
        one_sin_wave = get_one_sin_wave1(K, L, pad, k, w, t)
        Klapoetke_intensities_2Ff.append(one_sin_wave)
        t = t + dt
        
    Klapoetke_intensities_2Ff = np.array(Klapoetke_intensities_2Ff)
    np.save(savepath+'Klapoetke_intensities_2Ff', Klapoetke_intensities_2Ff)
    
    
def get_Klapoetke_intensities_2Fg(savepath, K, L, pad, dt):
    T = 5
    k = 2. * np.pi / (4 * L)
    w = 2. * np.pi / 1.
    t = 0.
    
    Klapoetke_intensities_2Fg = []
    while t <= T:
        one_sin_wave = get_one_sin_wave2(K, L, pad, k, w, t)
        Klapoetke_intensities_2Fg.append(one_sin_wave)
        t = t + dt
        
    Klapoetke_intensities_2Fg = np.array(Klapoetke_intensities_2Fg)
    np.save(savepath+'Klapoetke_intensities_2Fg', Klapoetke_intensities_2Fg)
    
    
def get_Klapoetke_intensities_3e(savepaths, K, L, pad, dt):
    N = K * L
    ro = (N - 1) / 2.
    co = (N - 1) / 2.
    theta_as = np.pi * 0.5 * np.array([3./6., 4./6., 5./6., 6./6., 7./6., 8./6., 9./6., 10./6., 11./6., 0, 1./6., 2./6.])
    v = L * 4
    
    for ind, theta_a in enumerate(theta_as):
        L1 = L * 1
        L2 = L * 1
        L3 = L * 1
        L4 = L * 1
        Klapoetke_intensities_3e = []
        while L1 <= (N - 1) / 2.:
            one_bar = get_one_bar(K, L, pad, ro, co, theta_a, L1, L2, L3, L4)
            Klapoetke_intensities_3e.append(one_bar)
            L1 = L1 + v * dt
            L3 = L3 + v * dt

        Klapoetke_intensities_3e = np.array(Klapoetke_intensities_3e)
        np.save(savepaths[ind]+'Klapoetke_intensities_3e{}'.format(ind+1), Klapoetke_intensities_3e)
        
        
def get_Klapoetke_intensities_3f(savepaths, K, L, pad, dt):
    N = K * L
    ro = (N - 1) / 2.
    co = (N - 1) / 2.
    theta_as = np.pi * 0.5 * np.array([3./6., 4./6., 5./6., 6./6., 7./6., 8./6., 9./6., 10./6., 11./6., 0, 1./6., 2./6.])
    v = L * 4
    
    for ind, theta_a in enumerate(theta_as):
        L1 = L * 1
        L2 = (N - 1) / 2.
        L3 = L * 1
        L4 = (N - 1) / 2.
        Klapoetke_intensities_3f = []
        while L1 <= (N - 1) / 2.:
            one_bar = get_one_bar(K, L, pad, ro, co, theta_a, L1, L2, L3, L4)
            Klapoetke_intensities_3f.append(one_bar)
            L1 = L1 + v * dt
            L3 = L3 + v * dt

        Klapoetke_intensities_3f = np.array(Klapoetke_intensities_3f)
        np.save(savepaths[ind]+'Klapoetke_intensities_3f{}'.format(ind+1), Klapoetke_intensities_3f)

        
def get_Klapoetke_intensities_4Aa(savepath, K, L, pad, dt):
    N = K * L
    ro = (N - 1) / 2.
    co = (N - 1) / 2.
    R = 2 * L
    v = 2 * L
    
    Klapoetke_intensities_4Aa = []
    while R <= (N + 1) / 2:
        one_disk = get_one_disk(K, L, pad, ro, co, R)
        Klapoetke_intensities_4Aa.append(one_disk)
        R = R + v * dt
    
    Klapoetke_intensities_4Aa = np.array(Klapoetke_intensities_4Aa)
    np.save(savepath+'Klapoetke_intensities_4Aa', Klapoetke_intensities_4Aa)
        
    
def get_Klapoetke_intensities_4Ab(savepath, K, L, pad, dt):
    N = K * L
    ro1 = (N + 1) / 2
    co1 = (N + 1) / 2
    theta_a1 = np.pi * 0.
    L11 = L * 1
    L21 = L * 1
    L31 = L * 1
    L41 = L * 1
    v1 = L * 4
    ro2 = (N + 1) / 2
    co2 = (N + 1) / 2
    theta_a2 = np.pi * 0.5
    L12 = L * 1
    L22 = L * 1
    L32 = L * 1
    L42 = L * 1
    v2 = L * 4
    
    Klapoetke_intensities_4Ab = []
    while L11 <= (N - 1) / 2.:
        one_bar1 = get_one_bar(K, L, pad, ro1, co1, theta_a1, L11, L21, L31, L41)
        one_bar2 = get_one_bar(K, L, pad, ro2, co2, theta_a2, L12, L22, L32, L42)
        two_bar = one_bar1 + one_bar2
        two_bar[two_bar > 1.] = 1.
        Klapoetke_intensities_4Ab.append(two_bar)
        L11 = L11+ v1 * dt
        L31 = L31+ v1 * dt
        L12 = L12+ v2 * dt
        L32 = L32+ v2 * dt
        
    Klapoetke_intensities_4Ab = np.array(Klapoetke_intensities_4Ab)
    np.save(savepath+'Klapoetke_intensities_4Ab', Klapoetke_intensities_4Ab)
    
    
def get_Klapoetke_intensities_4Ac(savepath, K, L, pad, dt):
    N = K * L
    
    ro0 = (N - 1) / 2.
    co0 = (N - 1) / 2.
    theta_a0 = np.pi * 0.
    L10 = L * 1
    L20 = L * 1
    L30 = L * 1
    L40 = L * 1
    
    ro1 = ro0
    co1 = 2 * co0
    theta_a1 = np.pi * 1.
    L11 = L * 1
    L21 = L * 1
    L31 = L * 1
    L41 = L * 1
    v1 = L * 4
    
    ro2 = 0
    co2 = co0
    theta_a2 = np.pi * 1.5
    L12 = L * 1
    L22 = L * 1
    L32 = L * 1
    L42 = L * 1
    v2 = L * 4
    
    ro3 = ro0
    co3 = 0
    theta_a3 = np.pi * 0.
    L13 = L * 1
    L23 = L * 1
    L33 = L * 1
    L43 = L * 1
    v3 = L * 4
    
    ro4 = 2 * ro0
    co4 = co0
    theta_a4 = np.pi * 0.5
    L14 = L * 1
    L24 = L * 1
    L34 = L * 1
    L44 = L * 1
    v4 = L * 4
    
    Klapoetke_intensities_4Ac = []
    while L11 <= (N - 1) / 2. - L:
        one_bar0 = get_one_bar(K, L, pad, ro0, co0, theta_a0, L10, L20, L30, L40)
        one_bar1 = get_one_bar(K, L, pad, ro1, co1, theta_a1, L11, L21, L31, L41)
        one_bar2 = get_one_bar(K, L, pad, ro2, co2, theta_a2, L12, L22, L32, L42)
        one_bar3 = get_one_bar(K, L, pad, ro3, co3, theta_a3, L13, L23, L33, L43)
        one_bar4 = get_one_bar(K, L, pad, ro4, co4, theta_a4, L14, L24, L34, L44)
        five_bar = one_bar0 + one_bar1 + one_bar2 + one_bar3 + one_bar4
        five_bar[five_bar > 1.] = 1.
        Klapoetke_intensities_4Ac.append(five_bar)
        L11 = L11+ v1 * dt
        L12 = L12+ v2 * dt
        L13 = L13 + v3 * dt
        L14 = L14 + v4 * dt
        
    Klapoetke_intensities_4Ac = np.array(Klapoetke_intensities_4Ac)
    np.save(savepath+'Klapoetke_intensities_4Ac', Klapoetke_intensities_4Ac)
    
    
def get_Klapoetke_intensities_4Ca(savepath, K, L, pad, dt):
    N = K * L
    ro = (N - 1) / 2.
    co = (N - 1) / 2.
    theta_a = np.pi * 1.
    L1 = L * 1
    L2 = L * 1
    L3 = L * 1
    L4 = L * 1
    v = L * 4
    
    Klapoetke_intensities_4Ca = []
    while L1 <= (N - 1) / 2.:
        one_bar = get_one_bar(K, L, pad, ro, co, theta_a, L1, L2, L3, L4)
        Klapoetke_intensities_4Ca.append(one_bar)
        L1 = L1 + v * dt
        
    Klapoetke_intensities_4Ca = np.array(Klapoetke_intensities_4Ca)
    np.save(savepath+'Klapoetke_intensities_4Ca', Klapoetke_intensities_4Ca)
    
    
def get_Klapoetke_intensities_4Cb(savepath, K, L, pad, dt):
    N = K * L
    ro = (N - 1) / 2.
    co = (N - 1) / 2.
    theta_a = np.pi * 0.
    L1 = L * 1
    L2 = L * 1
    L3 = L * 1
    L4 = L * 1
    v = L * 4
    
    Klapoetke_intensities_4Cb = []
    while L1 <= (N - 1) / 2.:
        one_bar = get_one_bar(K, L, pad, ro, co, theta_a, L1, L2, L3, L4)
        Klapoetke_intensities_4Cb.append(one_bar)
        L1 = L1 + v * dt
        
    Klapoetke_intensities_4Cb = np.array(Klapoetke_intensities_4Cb)
    np.save(savepath+'Klapoetke_intensities_4Cb', Klapoetke_intensities_4Cb)
    
    
def get_Klapoetke_intensities_4Cc(savepath, K, L, pad, dt):
    N = K * L
    ro = (N - 1) / 2.
    co = (N - 1) / 2.
    theta_a = np.pi * 0.
    L1 = L * 1
    L2 = L * 1
    L3 = L * 1
    L4 = L * 1
    v = L * 4
    
    Klapoetke_intensities_4Cc = []
    while L1 <= (N - 1) / 2.:
        one_bar = get_one_bar(K, L, pad, ro, co, theta_a, L1, L2, L3, L4)
        Klapoetke_intensities_4Cc.append(one_bar)
        L1 = L1 + v * dt
        L3 = L3 + v * dt
        
    Klapoetke_intensities_4Cc = np.array(Klapoetke_intensities_4Cc)
    np.save(savepath+'Klapoetke_intensities_4Cc', Klapoetke_intensities_4Cc)
    
    
def get_Klapoetke_intensities_4Cd(savepath, K, L, pad, dt):
    N = K * L
    
    ro1 = (N - 1) / 2.
    co1 = (N - 1) / 2.
    theta_a1 = np.pi * 0.
    L11 = L * 1
    L21 = L * 1
    L31 = L * 1
    L41 = L * 1
    v1 = L * 4
    
    ro2 = (N - 1) / 2.
    co2 = (N - 1) / 2.
    theta_a2 = np.pi * 0.
    L12 = L * 1
    L22 = (N - 1) / 2.
    L32 = L * 1
    L42 = (N - 1) / 2.
    grey_level = 0.5
    
    Klapoetke_intensities_4Cd = []
    while L11 <= (N - 1) / 2.:
        one_bar1 = get_one_bar(K, L, pad, ro1, co1, theta_a1, L11, L21, L31, L41)
        one_bar2 = get_one_bar(K, L, pad, ro2, co2, theta_a2, L12, L22, L32, L42, grey_level)
        two_bar = one_bar1 + one_bar2
        two_bar[two_bar > 1.] = 1.
        Klapoetke_intensities_4Cd.append(two_bar)
        L11 = L11 + v1 * dt
        L31 = L31 + v1 * dt
        grey_level = grey_level + 0.5 * (v1 * dt / ((N - 1) / 2. - L))
        
    Klapoetke_intensities_4Cd = np.array(Klapoetke_intensities_4Cd)
    np.save(savepath+'Klapoetke_intensities_4Cd', Klapoetke_intensities_4Cd)
    
    
def get_Klapoetke_intensities_4Ce(savepath, K, L, pad, dt):
    N = K * L
    
    ro0 = (N - 1) / 2.
    co0 = (N - 1) / 2.
    theta_a0 = np.pi * 0.
    L10 = L * 1
    L20 = L * 1
    L30 = L * 1
    L40 = L * 1
    v0 = L * 4
    
    ro2 = 0
    co2 = co0
    theta_a2 = np.pi * 1.5
    L12 = L * 1
    L22 = L * 1
    L32 = L * 1
    L42 = L * 1
    v2 = L * 4
    
    ro4 = 2 * ro0
    co4 = co0
    theta_a4 = np.pi * 0.5
    L14 = L * 1
    L24 = L * 1
    L34 = L * 1
    L44 = L * 1
    v4 = L * 4
    
    Klapoetke_intensities_4Ce = []
    while L10 <= (N - 1) / 2. or L12 <= (N - 1) / 2. - L:
        one_bar0 = get_one_bar(K, L, pad, ro0, co0, theta_a0, L10, L20, L30, L40)
        one_bar2 = get_one_bar(K, L, pad, ro2, co2, theta_a2, L12, L22, L32, L42)
        one_bar4 = get_one_bar(K, L, pad, ro4, co4, theta_a4, L14, L24, L34, L44)
        three_bar = one_bar0 + one_bar2 + one_bar4
        three_bar[three_bar > 1.] = 1.
        Klapoetke_intensities_4Ce.append(three_bar)
        L10 = L10 + v0 * dt
        L30 = L30 + v0 * dt
        L12 = L12+ v2 * dt
        L14 = L14 + v4 * dt
        
    Klapoetke_intensities_4Ce = np.array(Klapoetke_intensities_4Ce)
    np.save(savepath+'Klapoetke_intensities_4Ce', Klapoetke_intensities_4Ce)
    
    
def get_Klapoetke_intensities_4E10(savepath, K, L, pad, dt):
    N = K * L
    ro = (N - 1) / 2.
    co = (N - 1) / 2.
    theta_a = np.pi * 0.25
    L1 = L * 2
    L2 = L * 1
    L3 = L * 2
    L4 = L * 1
    v = L * 4
    
    Klapoetke_intensities_4E10 = []
    while L1 <= (N - 1) / 2.:
        one_bar = get_one_bar(K, L, pad, ro, co, theta_a, L1, L2, L3, L4)
        Klapoetke_intensities_4E10.append(one_bar)
        L1 = L1 + v * dt
        L3 = L3 + v * dt
        
    Klapoetke_intensities_4E10 = np.array(Klapoetke_intensities_4E10)
    np.save(savepath+'Klapoetke_intensities_4E10', Klapoetke_intensities_4E10)
    
    
def get_Klapoetke_intensities_4E20(savepath, K, L, pad, dt):
    N = K * L
    ro = (N - 1) / 2.
    co = (N - 1) / 2.
    theta_a = np.pi * 0.25
    L1 = L * 2
    L2 = L * 2
    L3 = L * 2
    L4 = L * 2
    v = L * 4
    
    Klapoetke_intensities_4E20 = []
    while L1 <= (N - 1) / 2.:
        one_bar = get_one_bar(K, L, pad, ro, co, theta_a, L1, L2, L3, L4)
        Klapoetke_intensities_4E20.append(one_bar)
        L1 = L1 + v * dt
        L3 = L3 + v * dt
        
    Klapoetke_intensities_4E20 = np.array(Klapoetke_intensities_4E20)
    np.save(savepath+'Klapoetke_intensities_4E20', Klapoetke_intensities_4E20)
    
    
def get_Klapoetke_intensities_4E60(savepath, K, L, pad, dt):
    N = K * L
    ro = (N - 1) / 2.
    co = (N - 1) / 2.
    theta_a = np.pi * 0.25
    L1 = L * 2
    L2 = (N - 1) / 2.
    L3 = L * 2
    L4 = (N - 1) / 2.
    v = L * 4
    
    Klapoetke_intensities_4E60 = []
    while L1 <= (N - 1) / 2.:
        one_bar = get_one_bar(K, L, pad, ro, co, theta_a, L1, L2, L3, L4)
        Klapoetke_intensities_4E60.append(one_bar)
        L1 = L1 + v * dt
        L3 = L3 + v * dt
        
    Klapoetke_intensities_4E60 = np.array(Klapoetke_intensities_4E60)
    np.save(savepath+'Klapoetke_intensities_4E60', Klapoetke_intensities_4E60)
    
    
def get_Klapoetke_intensities_4Ga(savepath, K, L, pad, dt):
    N = K * L
    ro = (N - 1) / 2.
    co = (N - 1) / 2.
    theta_a = np.pi * 0.25
    L1 = L * 1
    L2 = L * 1
    L3 = L * 1
    L4 = L * 1
    v = L * 4
    
    Klapoetke_intensities_4Ga = []
    while L1 <= (N - 1) / 2.:
        one_bar = get_one_bar(K, L, pad, ro, co, theta_a, L1, L2, L3, L4)
        Klapoetke_intensities_4Ga.append(one_bar)
        L1 = L1 + v * dt
        L3 = L3 + v * dt
        
    Klapoetke_intensities_4Ga = np.array(Klapoetke_intensities_4Ga)
    np.save(savepath+'Klapoetke_intensities_4Ga', Klapoetke_intensities_4Ga)
    
    
def get_Klapoetke_intensities_4Gb(savepath, K, L, pad, dt):
    N = K * L
    
    ro1 = (N - 1) / 2.
    co1 = (N - 1) / 2.
    theta_a1 = np.pi * 0.25
    L11 = L * 1
    L21 = L * 1
    L31 = L * 1
    L41 = L * 1
    v1 = L * 4
    
    ro2 = ro1*(1.-1./np.sqrt(2.))
    co2 = co1*(1.-1./np.sqrt(2.))
    theta_a2 = np.pi * (-0.25)
    L12 = L * 1
    L22 = (N - 1) / 2.
    L32 = L * 1
    L42 = (N - 1) / 2.
    grey_level = 0.5
    
    ro3 = ro1*(1.+1./np.sqrt(2.))
    co3 = co1*(1.+1./np.sqrt(2.))
    theta_a3 = np.pi * (-0.25)
    L13 = L * 1
    L23 = (N - 1) / 2.
    L33 = L * 1
    L43 = (N - 1) / 2.
    grey_level = 0.5
    
    Klapoetke_intensities_4Gb = []
    while L11 <= (N - 1) / 2.:
        one_bar1 = get_one_bar(K, L, pad, ro1, co1, theta_a1, L11, L21, L31, L41)
        one_bar2 = get_one_bar(K, L, pad, ro2, co2, theta_a2, L12, L22, L32, L42, grey_level)
        one_bar3 = get_one_bar(K, L, pad, ro3, co3, theta_a3, L13, L23, L33, L43, grey_level)
        three_bar = one_bar1+one_bar2+one_bar3
        three_bar[three_bar>1.] = 1.
        Klapoetke_intensities_4Gb.append(three_bar)
        L11 = L11 + v1 * dt
        L31 = L31 + v1 * dt
        grey_level = grey_level + 0.5 * (v1 * dt / ((N - 1) / 2. - L))
        
    Klapoetke_intensities_4Gb = np.array(Klapoetke_intensities_4Gb)
    np.save(savepath+'Klapoetke_intensities_4Gb', Klapoetke_intensities_4Gb)
    
    
def get_Klapoetke_intensities_4Gc(savepath, K, L, pad, dt):
    N = K * L
    
    ro1 = (N - 1) / 2.
    co1 = (N - 1) / 2.
    theta_a1 = np.pi * 0.25
    L11 = L * 1
    L21 = L * 1
    L31 = L * 1
    L41 = L * 1
    v1 = L * 4
    
    ro2 = ro1 * (1. - 1. / np.sqrt(2.))
    co2 = co1 * (1. - 1. / np.sqrt(2.))
    theta_a2 = np.pi * 0.25
    L12 = L * 1
    L22 = L * 1
    L32 = L * 1
    L42 = L * 1
    v2 = L * 4
    
    ro3 = ro1 * (1. + 1. / np.sqrt(2.))
    co3 = co1 * (1. + 1. / np.sqrt(2.))
    theta_a3 = np.pi * 0.25
    L13 = L * 1
    L23 = L * 1
    L33 = L * 1
    L43 = L * 1
    v3 = L * 4
    
    Klapoetke_intensities_4Gc = []
    while L11 <= (N - 1) / 2.:
        one_bar1 = get_one_bar(K, L, pad, ro1, co1, theta_a1, L11, L21, L31, L41)
        one_bar2 = get_one_bar(K, L, pad, ro2, co2, theta_a2, L12, L22, L32, L42)
        one_bar3 = get_one_bar(K, L, pad, ro3, co3, theta_a3, L13, L23, L33, L43)
        three_bar = one_bar1 + one_bar2 + one_bar3
        three_bar[three_bar > 1.] = 1.
        Klapoetke_intensities_4Gc.append(three_bar)
        L11 = L11 + v1 * dt
        L31 = L31 + v1 * dt
        L12 = L12 + v2 * dt
        L32 = L32 + v2 * dt
        L13 = L13 + v3 * dt
        L33 = L33 + v3 * dt
        
    Klapoetke_intensities_4Gc = np.array(Klapoetke_intensities_4Gc)
    np.save(savepath+'Klapoetke_intensities_4Gc', Klapoetke_intensities_4Gc)
    
    
def get_Klapoetke_intensities_4Gd(savepath, K, L, pad, dt):
    N = K * L
    
    ro0 = (N - 1) / 2.
    co0 = (N - 1) / 2.
    theta_a0 = np.pi * 0.25
    L10 = L * 1
    L20 = L * 1
    L30 = L * 1
    L40 = L * 1
    v0 = L * 4
    
    ro1 = ro0 * (1 - np.sqrt(2))
    co1 = co0
    theta_a1 = np.pi * 1.25
    L11 = L * 1
    L21 = L * 1
    L31 = L * 1
    L41 = L * 1
    v1 = L * 4
    
    ro2 = ro0
    co2 = co0 * (1 - np.sqrt(2))
    theta_a2 = np.pi * 0.25
    L12 = L * 1
    L22 = L * 1
    L32 = L * 1
    L42 = L * 1
    v2 = L * 4
    
    ro3 = ro0 * (1 + np.sqrt(2))
    co3 = co0
    theta_a3 = np.pi * 0.25
    L13 = L * 1
    L23 = L * 1
    L33 = L * 1
    L43 = L * 1
    v3 = L * 4
    
    ro4 = ro0
    co4 = co0 * (1 + np.sqrt(2))
    theta_a4 = np.pi * 1.25
    L14 = L * 1
    L24 = L * 1
    L34 = L * 1
    L44 = L * 1
    v4 = L * 4
    
    Klapoetke_intensities_4Gd = []
    while L10 <= (N - 1) / 2. or L11 <= (N - 1) / 2.:
        one_bar0 = get_one_bar(K, L, pad, ro0, co0, theta_a0, L10, L20, L30, L40)
        one_bar1 = get_one_bar(K, L, pad, ro1, co1, theta_a1, L11, L21, L31, L41)
        one_bar2 = get_one_bar(K, L, pad, ro2, co2, theta_a2, L12, L22, L32, L42)
        one_bar3 = get_one_bar(K, L, pad, ro3, co3, theta_a3, L13, L23, L33, L43)
        one_bar4 = get_one_bar(K, L, pad, ro4, co4, theta_a4, L14, L24, L34, L44)
        five_bar = one_bar0 + one_bar1 + one_bar2 + one_bar3 + one_bar4
        five_bar[five_bar > 1.] = 1.
        Klapoetke_intensities_4Gd.append(five_bar)
        L10 = L10 + v0 * dt
        L30 = L30 + v0 * dt
        L11 = L11 + v1 * dt
        L12 = L12 + v2 * dt
        L13 = L13 + v3 * dt
        L14 = L14 + v4 * dt
        
    Klapoetke_intensities_4Gd = np.array(Klapoetke_intensities_4Gd)
    np.save(savepath+'Klapoetke_intensities_4Gd', Klapoetke_intensities_4Gd)
    
    
def get_Klapoetke_intensities_4Ge(savepath, K, L, pad, dt):
    N = K * L
    
    ro1 = (N - 1) / 2.
    co1 = (N - 1) / 2.
    
    ro2 = ro1 * (1. - 1. / np.sqrt(2.))
    co2 = co1 * (1. - 1. / np.sqrt(2.))
    theta_a2 = np.pi * 0.25
    L12 = L * 1
    L22 = L * 1
    L32 = L * 1
    L42 = L * 1
    v2 = L * 4
    
    ro3 = ro1 * (1. + 1. / np.sqrt(2.))
    co3 = co1 * (1. + 1. / np.sqrt(2.))
    theta_a3 = np.pi * 0.25
    L13 = L * 1
    L23 = L * 1
    L33 = L * 1
    L43 = L * 1
    v3 = L * 4
    
    Klapoetke_intensities_4Ge = []
    while L12 <= (N - 1) / 2.:
        one_bar2 = get_one_bar(K, L, pad, ro2, co2, theta_a2, L12, L22, L32, L42)
        one_bar3 = get_one_bar(K, L, pad, ro3, co3, theta_a3, L13, L23, L33, L43)
        two_bar = one_bar2+one_bar3
        two_bar[two_bar > 1.] = 1.
        Klapoetke_intensities_4Ge.append(two_bar)
        L12 = L12 + v2 * dt
        L32 = L32 + v2 * dt
        L13 = L13 + v3 * dt
        L33 = L33 + v3 * dt
        
    Klapoetke_intensities_4Ge = np.array(Klapoetke_intensities_4Ge)
    np.save(savepath+'Klapoetke_intensities_4Ge', Klapoetke_intensities_4Ge)


def get_UV_flows_exp(space_filter, K, L, pad, dt, delay_dt, leftup_corners, intensities):
    steps = intensities.shape[0]
    UV_flows = np.zeros((steps-1, 1, K*K, 4))
    signal_filtered_all = np.zeros((1, K*K, 4))
    cf0 = intensities[0]
    signal_filtered_all, signal_cur =\
        flfd.get_filtered_and_current(signal_filtered_all, cf0, leftup_corners, space_filter, K, L, pad, 1, 1)
    for step in range(1, steps):
        cf = intensities[step]
        signal_filtered_all, signal_cur =\
            flfd.get_filtered_and_current(signal_filtered_all, cf, leftup_corners, space_filter, K, L, pad, dt, delay_dt)
        UV_flow = flfd.get_flow_fields(signal_filtered_all, signal_cur, leftup_corners, K, L, pad)
        if UV_flow[0].shape[0] > 1:
            UV_flows[step-1, 0, :, :] = UV_flow[0][:, :]
        
    return UV_flows


# def get_UV_flows(space_filter, K, L, pad, dt, delay_dt, leftup_corners, intensities):
#     steps = intensities.shape[0]
#     delay_step = np.int(delay_dt / dt)
#     UV_flows = np.zeros((steps, 1, K * K, 4))
#     signal_filtered_all = np.zeros((1, K * K, 4))
#     for step in range(delay_step, steps):
#         # previous frame
#         cf1 = intensities[step - delay_step]
#         _, signal_cur1 = \
#             flfd.get_filtered_and_current(signal_filtered_all, cf1, \
#                                           leftup_corners, space_filter, K, L, pad, dt, delay_dt)
#         # current frame
#         cf = intensities[step]
#         _, signal_cur = \
#             flfd.get_filtered_and_current(signal_filtered_all, cf, \
#                                           leftup_corners, space_filter, K, L, pad, dt, delay_dt)
#         UV_flow = flfd.get_flow_fields(signal_cur1, signal_cur, leftup_corners, K, L, pad)
#         if UV_flow[0].shape[0] > 1:
#             UV_flows[step, 0, :, :] = UV_flow[0][:, :]
        
#     return UV_flows


def get_intensities_cg(K, L, pad, leftup_corners, intensities):
    steps = intensities.shape[0]
    intensities_cg = np.zeros((steps-1, 1, K*K))
    for step in range(1, steps):
        intensity_cg = opsg.get_intensity_cg(intensities[step, :, :, :], leftup_corners, K, L, pad)
        if intensity_cg[0].shape[0] > 1:
            intensities_cg[step-1, 0, :] = intensity_cg[0][:]
            
    return intensities_cg


def get_one_disk(K, L, pad, ro, co, R):
    N = K * L + 2 * pad
    ro = ro + pad
    co = co + pad
    coord_rp = np.arange(N)
    coord_cp = np.arange(N)
    coords_cp, coords_rp = np.meshgrid(coord_cp, coord_rp)
    dm = np.sqrt((coords_rp-ro)**2 + (coords_cp-co)**2)
    one_disk = np.zeros((1, N, N))
    con = dm <= R
    one_disk[0, :, :] = con
                
    return one_disk


def get_one_bar(K, L, pad, ro, co, theta_a, L1, L2, L3, L4, grey_level=1):
    """
    Get one bar with certain dimension.
    
    Args:
    K:
    L:
    pad:
    ro: row coordinate of the center of the bar
    co: column coordinate of the center of the bar
    theta_a: bar orientation
    L1:
    L2:
    L3:
    L4:
    
    Returns:
    one_bar
    """
    # total dimension
    N = K * L + 2 * pad 
    
    # adjust ro and co
    ro = ro + pad 
    co = co + pad
    
    # coordinates mesh
    coord_rp = np.arange(N)
    coord_cp = np.arange(N)
    coords_cp, coords_rp = np.meshgrid(coord_cp, coord_rp) 
    
    # define the four directions
    theta_a = - theta_a # account for the direction of the row indexing
    d1 = np.array([np.sin(theta_a), np.cos(theta_a)])
    d2 = np.array([np.cos(theta_a), -np.sin(theta_a)])
    d3 = np.array([-np.sin(theta_a), -np.cos(theta_a)])
    d4 = np.array([-np.cos(theta_a), np.sin(theta_a)]) 
    
    # get the four conditions
    dp = np.stack([coords_rp-ro, coords_cp-co], axis=-1)
    con1 = np.dot(dp, d1) <= L1
    con2 = np.dot(dp, d2) <= L2
    con3 = np.dot(dp, d3) <= L3
    con4 = np.dot(dp, d4) <= L4 
    
    one_bar = np.zeros((1, N, N))
    one_bar[0, :, :] = con1 * con2 * con3 * con4 * grey_level
                
    return one_bar


def get_one_sin_wave1(K, L, pad, k, w, t):
    N = K * L + 2 * pad
    ones = np.ones(N)
    one_sin_wave = np.zeros((1, N, N))
    col = np.arange(N)
    one_sin_wave[0, :, :] = np.outer(ones, np.sign(np.sin(k*col-w*t)))
        
    return one_sin_wave


def get_one_sin_wave2(K, L, pad, k, w, t):
    N = K * L + 2 * pad
    ones = np.ones(N)
    one_sin_wave = np.zeros((1, N, N))
    row = np.arange(N)
    one_sin_wave[0, :, :] = np.outer(np.sign(np.sin(k*row-w*t)), ones)
        
    return one_sin_wave


def get_reshape(K, intensities):
    steps = intensities.shape[0]
    intensities_re = intensities[:, 0, :].reshape((steps, K, K))
        
    return intensities_re


def save_intensity(intensity, step, save_name):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(intensity, cmap='gray_r', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
#         ax.vlines(np.arange(-0.55, N, step=N/K), -0.55, N-0.45, color='salmon', linewidth=.5)
#         ax.hlines(np.arange(-0.55, N, step=N/K), -0.55, N-0.45, color='salmon', linewidth=.5)
    fig.savefig(save_name+'_{}.png'.format(step+1))
    plt.close(fig)
    
    
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


if __name__ == "__main__":
    
    start_time = time.time()
    
    K = 12 # K * K is the total number of motion detectors
    L = 50 # dimension of each motion detector
    dt = 0.01 # simulation time step
    p = 1 # sample gaps
    pad = 2 * L # padding size
    delay_dt = 0.03 # delay time in the hrc model
    
    space_filter = flfd.get_space_filter(L/2, 4)
    folder_path = f'/Volumes/Baohua/data_on_hd/loom/Klapoetke_stimuli_L{L}_dt_{dt}_exp_2/'
    
    for ind, foldertype in enumerate(foldertypes1):
        os.makedirs(folder_path+'Klapoetke_intensities/'+foldertype)
        os.makedirs(folder_path+'Klapoetke_intensities_extended/'+foldertype)
        os.makedirs(folder_path+'Klapoetke_intensities_cg/'+foldertype)
        os.makedirs(folder_path+'Klapoetke_UV_flows/'+foldertype)
    
    for ind, foldertype in enumerate(foldertypes3):
        os.makedirs(folder_path+'Klapoetke_intensities_movies/'+foldertype)
        os.makedirs(folder_path+'Klapoetke_intensities_extended_movies/'+foldertype)
        os.makedirs(folder_path+'Klapoetke_intensities_cg_movies/'+foldertype)
        os.makedirs(folder_path+'Klapoetke_UV_flows_movies/'+foldertype)
    
    log_file = folder_path + 'log.txt'
    with open(log_file, 'w') as f:
        f.write('Simulation parameters:\n')
        f.write('----------------------------------------\n')
        f.write(f'K: {K}, K * K is the total number of motion detectors\n')
        f.write(f'L: {L}, dimension of each motion detector\n')
        f.write(f'dt: {dt}, simulation time step\n')
        f.write(f'p: {p}, sample gaps\n')
        f.write(f'pad: {pad}, padding size\n')
        f.write(f'delay_dt: {delay_dt}, delay time in the hrc model\n')
    
    
    
    savepaths = [folder_path+'Klapoetke_intensities/'+foldertype2+'/' for foldertype2 in foldertypes2]
    get_Klapoetke_intensities(K, L, pad, dt, savepaths)
    
    loadpaths = [folder_path+'Klapoetke_intensities/'+foldertype2+'/' for foldertype2 in foldertypes2]
    savepaths = [folder_path+'Klapoetke_intensities_extended/'+foldertype2+'/' for foldertype2 in foldertypes2]
    names1 = Klapoetke_intensities_names
    names2 = Klapoetke_intensities_extended_names
    get_Klapoetke_intensities_extended(K, L, pad, dt, loadpaths, savepaths, names1, names2)
    
    loadpaths = [folder_path+'Klapoetke_intensities_extended/'+foldertype2+'/' for foldertype2 in foldertypes2]
    savepaths = [folder_path+'Klapoetke_intensities_cg/'+foldertype2+'/' for foldertype2 in foldertypes2]
    names1 = Klapoetke_intensities_extended_names
    names2 = Klapoetke_intensities_cg_names
    get_Klapoetke_intensities_cg(K, L, pad, loadpaths, savepaths, names1, names2)
    
    loadpaths = [folder_path+'Klapoetke_intensities_extended/'+foldertype2+'/' for foldertype2 in foldertypes2]
    savepaths = [folder_path+'Klapoetke_UV_flows/'+foldertype2+'/' for foldertype2 in foldertypes2]
    names1 = Klapoetke_intensities_extended_names
    names2 = Klapoetke_UV_flows_names
    get_Klapoetke_UV_flows(space_filter, K, L, pad, dt, delay_dt, loadpaths, savepaths, names1, names2)

    loadpaths = [folder_path+'Klapoetke_intensities/'+foldertype2+'/' for foldertype2 in foldertypes2]
    savepaths1 = [folder_path+'Klapoetke_intensities_movies/'+foldertype2+'/' for foldertype2 in foldertypes2]
    savepaths2 = [folder_path+'Klapoetke_intensities_movies/'+foldertype3+'/' for foldertype3 in foldertypes3]
    names1 = Klapoetke_intensities_names
    names2 = Klapoetke_intensities_frame_names
    names3 = Klapoetke_intensities_movie_names
    plot_KLapoetke_intensities(dt, p, loadpaths, savepaths1, savepaths2, names1, names2, names3)
    
    loadpaths = [folder_path+'Klapoetke_intensities_extended/'+foldertype2+'/' for foldertype2 in foldertypes2]
    savepaths1 = [folder_path+'Klapoetke_intensities_extended_movies/'+foldertype2+'/' \
                 for foldertype2 in foldertypes2]
    savepaths2 = [folder_path+'Klapoetke_intensities_extended_movies/'+foldertype3+'/' \
                 for foldertype3 in foldertypes3]
    names1 = Klapoetke_intensities_extended_names
    names2 = Klapoetke_intensities_extended_frame_names
    names3 = Klapoetke_intensities_extended_movie_names
    plot_KLapoetke_intensities_extended(dt, p, loadpaths, savepaths1, savepaths2, names1, names2, names3)
    
    loadpaths = [folder_path+'Klapoetke_intensities_cg/'+foldertype2+'/' for foldertype2 in foldertypes2]
    savepaths1 = [folder_path+'Klapoetke_intensities_cg_movies/'+foldertype2+'/'\
                 for foldertype2 in foldertypes2]
    savepaths2 = [folder_path+'Klapoetke_intensities_cg_movies/'+foldertype3+'/'\
                 for foldertype3 in foldertypes3]
    names1 = Klapoetke_intensities_cg_names
    names2 = Klapoetke_intensities_cg_frame_names
    names3 = Klapoetke_intensities_cg_movie_names
    plot_KLapoetke_intensities_cg(K, dt, p, loadpaths, savepaths1, savepaths2, names1, names2, names3)
    
    loadpaths = [folder_path+'Klapoetke_UV_flows/'+foldertype2+'/' for foldertype2 in foldertypes2]
    savepaths1 = [folder_path+'Klapoetke_UV_flows_movies/'+foldertype2+'/' for foldertype2 in foldertypes2]
    savepaths2 = [folder_path+'Klapoetke_UV_flows_movies/'+foldertype3+'/' for foldertype3 in foldertypes3]
    names1 = Klapoetke_UV_flows_names
    names2 = Klapoetke_U_flows_frame_names
    names3 = Klapoetke_V_flows_frame_names
    names4 = Klapoetke_U_flows_movie_names
    names5 = Klapoetke_V_flows_movie_names
    plot_KLapoetke_UV_flows(K, L, pad, dt, p, loadpaths, savepaths1, savepaths2, names1, names2, names3, names4, names5)
    
    print('Time used: {}'.format(time.time()-start_time))
    
    