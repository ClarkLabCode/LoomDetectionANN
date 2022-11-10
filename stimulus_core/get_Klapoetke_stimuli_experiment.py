#!/usr/bin/env python

'''
Get all the stimuli in Klapoetke et al's paper (Nature, 2017)
'''

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


foldertypes1 = ['2F','3e','3f','4A','4C','4E','4G']

foldertypes2 = ['2F']*7+['3e']*12+['3f']*12+['4A']*3+['4C']*5+['4E']*3+['4G']*5

foldertypes3 = \
['2F/2Fa','2F/2Fb','2F/2Fc','2F/2Fd','2F/2Fe','2F/2Ff','2F/2Fg',\
 '3e/3e1','3e/3e2','3e/3e3','3e/3e4','3e/3e5','3e/3e6','3e/3e7','3e/3e8','3e/3e9','3e/3e10','3e/3e11','3e/3e12',\
 '3f/3f1','3f/3f2','3f/3f3','3f/3f4','3f/3f5','3f/3f6','3f/3f7','3f/3f8','3f/3f9','3f/3f10','3f/3f11','3f/3f12',\
 '4A/4Aa','4A/4Ab','4A/4Ac',\
 '4C/4Ca','4C/4Cb','4C/4Cc','4C/4Cd','4C/4Ce',\
 '4E/4E10','4E/4E20','4E/4E60',\
 '4G/4Ga','4G/4Gb','4G/4Gc','4G/4Gd','4G/4Ge']

filetypes1 = \
['2Fa','2Fb','2Fc','2Fd','2Fe','2Ff','2Fg',\
 '3e1','3e2','3e3','3e4','3e5','3e6','3e7','3e8','3e9','3e10','3e11','3e12',\
 '3f1','3f2','3f3','3f4','3f5','3f6','3f7','3f8','3f9','3f10','3f11','3f12',\
 '4Aa','4Ab','4Ac',\
 '4Ca','4Cb','4Cc','4Cd','4Ce',\
 '4E10','4E20','4E60',\
 '4Ga','4Gb','4Gc','4Gd','4Ge']

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


def get_Klapoetke_intensities(K,L,pad,dt,savepaths):
    get_Klapoetke_intensities_2Fa(savepaths[0],K,L,pad,dt)
    get_Klapoetke_intensities_2Fb(savepaths[1],K,L,pad,dt)
    get_Klapoetke_intensities_2Fc(savepaths[2],K,L,pad,dt)
    get_Klapoetke_intensities_2Fd(savepaths[3],K,L,pad,dt)
    get_Klapoetke_intensities_2Fe(savepaths[4],K,L,pad,dt)
    get_Klapoetke_intensities_2Ff(savepaths[5],K,L,pad,dt)
    get_Klapoetke_intensities_2Fg(savepaths[6],K,L,pad,dt)

    get_Klapoetke_intensities_3e(savepaths[7:19],K,L,pad,dt)
    get_Klapoetke_intensities_3f(savepaths[19:31],K,L,pad,dt)
    
    get_Klapoetke_intensities_4Aa(savepaths[31],K,L,pad,dt)
    get_Klapoetke_intensities_4Ab(savepaths[32],K,L,pad,dt)
    get_Klapoetke_intensities_4Ac(savepaths[33],K,L,pad,dt)
    get_Klapoetke_intensities_4Ca(savepaths[34],K,L,pad,dt)
    get_Klapoetke_intensities_4Cb(savepaths[35],K,L,pad,dt)
    get_Klapoetke_intensities_4Cc(savepaths[36],K,L,pad,dt)
    get_Klapoetke_intensities_4Cd(savepaths[37],K,L,pad,dt)
    get_Klapoetke_intensities_4Ce(savepaths[38],K,L,pad,dt)
    get_Klapoetke_intensities_4E10(savepaths[39],K,L,pad,dt)
    get_Klapoetke_intensities_4E20(savepaths[40],K,L,pad,dt)
    get_Klapoetke_intensities_4E60(savepaths[41],K,L,pad,dt)
    get_Klapoetke_intensities_4Ga(savepaths[42],K,L,pad,dt)
    get_Klapoetke_intensities_4Gb(savepaths[43],K,L,pad,dt)
    get_Klapoetke_intensities_4Gc(savepaths[44],K,L,pad,dt)
    get_Klapoetke_intensities_4Gd(savepaths[45],K,L,pad,dt)
    get_Klapoetke_intensities_4Ge(savepaths[46],K,L,pad,dt)
    
    
def get_Klapoetke_intensities_extended(K,L,pad,dt,loadpaths,savepaths,names1,names2):
    leftup_corners = opsg.get_leftup_corners(K,L,pad)
    add_time = np.int(2/dt)
    for ind,name in enumerate(names1):
        frame = np.load(loadpaths[ind]+names1[ind])
        frame_ext = np.zeros((frame.shape[0]+2*add_time,frame.shape[1],frame.shape[2],frame.shape[3]))
        frame_ext[add_time:-add_time,:,:,:] = frame[:,:,:,:]
        for ii in range(add_time):
            frame_ext[ii,:,:,:] = frame[0,:,:,:]
            frame_ext[-ii-1,:,:,:] = frame[-1,:,:,:]
        np.save(savepaths[ind]+names2[ind],frame_ext)
    
    
def get_Klapoetke_intensities_cg(K,L,pad,loadpaths,savepaths,names1,names2):
    leftup_corners = opsg.get_leftup_corners(K,L,pad)
    for ind,name in enumerate(names1):
        frame = np.load(loadpaths[ind]+names1[ind])
        frame_cg = get_intensities_cg(K,L,pad,leftup_corners,frame)
        np.save(savepaths[ind]+names2[ind],frame_cg)
        
        
def get_Klapoetke_UV_flows(space_filter,K,L,pad,dt,delay_dt,loadpaths,savepaths,names1,names2):
    leftup_corners = opsg.get_leftup_corners(K,L,pad)
    for ind,name in enumerate(names1):
        frame = np.load(loadpaths[ind]+names1[ind])
        UV_flow = get_UV_flows(space_filter,K,L,pad,dt,delay_dt,leftup_corners,frame)
        np.save(savepaths[ind]+names2[ind],UV_flow)


def plot_KLapoetke_intensities(dt,p,loadpaths,savepaths1,savepaths2,names1,names2,names3):
    for ind,name in enumerate(names1):
        intensities = np.load(loadpaths[ind]+names1[ind])
        intensities = np.squeeze(intensities)
        steps = intensities.shape[0]
        combined_movies_list = []
        for step in range(steps):
            if step%p == 0:
                save_intensity(intensities[step],step,savepaths2[ind]+names2[ind])
                combined_movies = imageio.imread(savepaths2[ind]+names2[ind]+'_{}.png'.format(step+1))
                combined_movies_list.append(combined_movies)
        movie_name = savepaths1[ind]+names3[ind]
        imageio.mimsave(movie_name, combined_movies_list, fps = np.int(1/dt))
        

def plot_KLapoetke_intensities_extended(dt,p,loadpaths,savepaths1,savepaths2,names1,names2,names3):
    for ind,name in enumerate(names1):
        intensities = np.load(loadpaths[ind]+names1[ind])
        intensities = np.squeeze(intensities)
        steps = intensities.shape[0]
        combined_movies_list = []
        for step in range(steps):
            if step%p == 0:
                save_intensity(intensities[step],step,savepaths2[ind]+names2[ind])
                combined_movies = imageio.imread(savepaths2[ind]+names2[ind]+'_{}.png'.format(step+1))
                combined_movies_list.append(combined_movies)
        movie_name = savepaths1[ind]+names3[ind]
        imageio.mimsave(movie_name, combined_movies_list, fps = np.int(1/dt))
        
        
def plot_KLapoetke_intensities_cg(K,dt,p,loadpaths,savepaths1,savepaths2,names1,names2,names3):
    for ind,name in enumerate(names1):
        intensities = np.load(loadpaths[ind]+names1[ind])
        intensities = get_reshape(K,intensities)
        steps = intensities.shape[0]
        combined_movies_list = []
        for step in range(steps):
            if step%p == 0:
                save_intensity(intensities[step],step,savepaths2[ind]+names2[ind])
                combined_movies = imageio.imread(savepaths2[ind]+names2[ind]+'_{}.png'.format(step+1))
                combined_movies_list.append(combined_movies)
        movie_name = savepaths1[ind]+names3[ind]
        imageio.mimsave(movie_name, combined_movies_list, fps = np.int(1/dt))


def plot_KLapoetke_UV_flows(K,L,pad,dt,p,loadpaths,savepaths1,savepath2,names1,names2,names3,names4,names5):
    leftup_corners = opsg.get_leftup_corners(K,L,pad)
    for ind,name in enumerate(names1):
        UV_flow = np.load(loadpaths[ind]+names1[ind])
        steps = UV_flow.shape[0]
        combined_movies_u_list = []
        combined_movies_v_list = []
        for step in range(steps):
            if step%p == 0:
                cf_u,cf_v = flfd.set_flow_fields_on_frame2(UV_flow[step,:,:,:],leftup_corners,K,L,pad)
                u_flow = cf_u[0,:,:]
                v_flow = cf_v[0,:,:]
                save_uv_flow(u_flow,step,savepaths2[ind]+names2[ind])
                save_uv_flow(v_flow,step,savepaths2[ind]+names3[ind])
                combined_movies_u = imageio.imread(savepaths2[ind]+names2[ind]+'_{}.png'.format(step+1))
                combined_movies_u_list.append(combined_movies_u)
                combined_movies_v = imageio.imread(savepaths2[ind]+names3[ind]+'_{}.png'.format(step+1))
                combined_movies_v_list.append(combined_movies_v)
        movie_name_u = savepaths1[ind]+names4[ind]
        imageio.mimsave(movie_name_u, combined_movies_u_list, fps = np.int(1/dt))
        movie_name_v = savepaths1[ind]+names5[ind]
        imageio.mimsave(movie_name_v, combined_movies_v_list, fps = np.int(1/dt))
            
    
def get_Klapoetke_intensities_2Ff(savepath,K,L,pad,dt):
    T = 5
    k = 2.*np.pi/(4*L)
    w = 2.*np.pi/1.
    t = 0.
    
    Klapoetke_intensities_2Ff = []
    while t <= T:
        one_sin_wave = get_one_sin_wave1(K,L,pad,k,w,t)
        # apply spacial filter
#         one_sin_wave = gaussian_filter(one_sin_wave,sigma=L/2.)
        Klapoetke_intensities_2Ff.append(one_sin_wave)
        t = t+dt
        
    Klapoetke_intensities_2Ff = np.array(Klapoetke_intensities_2Ff)
    np.save(savepath+'Klapoetke_intensities_2Ff',Klapoetke_intensities_2Ff)
    
    
def get_Klapoetke_intensities_2Fg(savepath,K,L,pad,dt):
    T = 5
    k = 2.*np.pi/(4*L)
    w = 2.*np.pi/1.
    t = 0.
    
    Klapoetke_intensities_2Fg = []
    while t <= T:
        one_sin_wave = get_one_sin_wave2(K,L,pad,k,w,t)
        # apply spacial filter
#         one_sin_wave = gaussian_filter(one_sin_wave,sigma=L/2.)
        Klapoetke_intensities_2Fg.append(one_sin_wave)
        t = t+dt
        
    Klapoetke_intensities_2Fg = np.array(Klapoetke_intensities_2Fg)
    np.save(savepath+'Klapoetke_intensities_2Fg',Klapoetke_intensities_2Fg)        


def get_UV_flows(space_filter,K,L,pad,dt,delay_dt,leftup_corners,intensities):
    lplc2_cells = np.array([[0,0]])
    steps = intensities.shape[0]
    UV_flows = np.zeros((steps-1,1,K*K,4))
    signal_filtered_all = np.zeros((1,K*K,4))
    cf0 = intensities[0]
    signal_filtered_all,signal_cur =\
        flfd.get_filtered_and_current(signal_filtered_all,cf0,leftup_corners,space_filter,K,L,pad,1,1)
    for step in range(1,steps):
        cf = intensities[step]
        signal_filtered_all,signal_cur =\
            flfd.get_filtered_and_current(signal_filtered_all,cf,leftup_corners,space_filter,K,L,pad,dt,delay_dt)
        UV_flow = flfd.get_flow_fields(signal_filtered_all,signal_cur,leftup_corners,K,L,pad)
        if UV_flow[0].shape[0] > 1:
            UV_flows[step-1,0,:,:] = UV_flow[0][:,:]
        
    return UV_flows


def get_UV_flows_r(space_filter,K,L,pad,dt,delay_dt,leftup_corners,intensities):
    lplc2_cells = np.array([[0,0]])
    steps = intensities.shape[0]
    UV_flows = np.zeros((steps-1,1,K*K,4))
    signal_filtered_all = np.zeros((1,K*K,4))
    cf0 = intensities[-1]
    signal_filtered_all,signal_cur =\
        flfd.get_filtered_and_current(signal_filtered_all,cf0,leftup_corners,space_filter,K,L,pad,1,1)
    for step in range(1,steps):
        cf = intensities[-step-1]
        signal_filtered_all,signal_cur =\
            flfd.get_filtered_and_current(signal_filtered_all,cf,leftup_corners,space_filter,K,L,pad,dt,delay_dt)
        UV_flow = flfd.get_flow_fields(signal_filtered_all,signal_cur,leftup_corners,K,L,pad)
        if UV_flow[0].shape[0] > 1:
            UV_flows[step-1,0,:,:] = UV_flow[0][:,:]
        
    return UV_flows


def get_UV_flows2(space_filter,K,L,pad,dt,delay_dt,leftup_corners,intensities):
    lplc2_cells = np.array([[0,0]])
    steps = intensities.shape[0]
    delay_step = np.int(delay_dt/dt)
    UV_flows = np.zeros((steps-delay_step,1,K*K,4))
    signal_filtered_all = np.zeros((1,K*K,4))
    for step in range(delay_step,steps):
        cf1 = intensities[step-delay_step]
        _,signal_cur1 =\
            flfd.get_filtered_and_current(signal_filtered_all,cf1,leftup_corners,space_filter,K,L,pad,dt,delay_dt)
        cf2 = intensities[step]
        _,signal_cur2 =\
            flfd.get_filtered_and_current(signal_filtered_all,cf2,leftup_corners,space_filter,K,L,pad,dt,delay_dt)
        UV_flow = flfd.get_flow_fields(signal_cur1,signal_cur2,leftup_corners,K,L,pad)
        if UV_flow[0].shape[0] > 1:
            UV_flows[step-delay_step,0,:,:] = UV_flow[0][:,:]
        
    return UV_flows


def get_UV_flows_r2(space_filter,K,L,pad,dt,delay_dt,leftup_corners,intensities):
    lplc2_cells = np.array([[0,0]])
    steps = intensities.shape[0]
    delay_step = np.int(delay_dt/dt)
    UV_flows = np.zeros((steps-delay_step,1,K*K,4))
    signal_filtered_all = np.zeros((1,K*K,4))
    for step in range(delay_step,steps):
        cf1 = intensities[-step+delay_step-1]
        _,signal_cur1 =\
            flfd.get_filtered_and_current(signal_filtered_all,cf1,leftup_corners,space_filter,K,L,pad,dt,delay_dt)
        cf2 = intensities[-step-1]
        _,signal_cur2 =\
            flfd.get_filtered_and_current(signal_filtered_all,cf2,leftup_corners,space_filter,K,L,pad,dt,delay_dt)
        UV_flow = flfd.get_flow_fields(signal_cur1,signal_cur2,leftup_corners,K,L,pad)
        if UV_flow[0].shape[0] > 1:
            UV_flows[step-delay_step,0,:,:] = UV_flow[0][:,:]
        
    return UV_flows


def get_intensities_cg(K,L,pad,leftup_corners,intensities):
    steps = intensities.shape[0]
    intensities_cg = np.zeros((steps-1,1,K*K))
    for step in range(1,steps):
        intensity_cg = opsg.get_frame_cg(intensities[step,:,:,:],leftup_corners,K,L,pad)
        if intensity_cg[0].shape[0] > 1:
            intensities_cg[step-1,0,:] = intensity_cg[0][:]
            
    return intensities_cg


def get_one_disk(K,L,pad,ro,co,R):
    N = K*L + 2*pad
    ro = ro + pad
    co = co + pad
    one_disk = np.zeros((1,N,N))
    
    for row in range(N):
        for col in range(N):
            con = np.sqrt((row-ro)**2+(col-co)**2) <= R
            if con:
                one_disk[0,row,col] = 1
                
    return one_disk


def get_one_bar(K,L,pad,ro,co,theta_a,L1,L2,L3,L4):
    N = K*L + 2*pad
    ro = ro + pad
    co = co + pad
    one_bar = np.zeros((1,N,N))
    theta_a = - theta_a
    d1 = np.array([np.sin(theta_a),np.cos(theta_a)])
    d2 = np.array([np.cos(theta_a),-np.sin(theta_a)])
    d3 = np.array([-np.sin(theta_a),-np.cos(theta_a)])
    d4 = np.array([-np.cos(theta_a),np.sin(theta_a)])
    
    for row in range(N):
        for col in range(N):
            dp = np.array([row-ro,col-co])
            con1 = np.dot(dp,d1) <= L1
            con2 = np.dot(dp,d2) <= L2
            con3 = np.dot(dp,d3) <= L3
            con4 = np.dot(dp,d4) <= L4
            if con1 and con2 and con3 and con4:
                one_bar[0,row,col] = 1
                
    return one_bar


def get_one_grey_bar(K,L,pad,ro,co,theta_a,L1,L2,L3,L4,grey_level):
    N = K*L + 2*pad
    ro = ro + pad
    co = co + pad
    one_bar = np.zeros((1,N,N))
    theta_a = - theta_a
    d1 = np.array([np.sin(theta_a),np.cos(theta_a)])
    d2 = np.array([np.cos(theta_a),-np.sin(theta_a)])
    d3 = np.array([-np.sin(theta_a),-np.cos(theta_a)])
    d4 = np.array([-np.cos(theta_a),np.sin(theta_a)])
    
    for row in range(N):
        for col in range(N):
            dp = np.array([row-ro,col-co])
            con1 = np.dot(dp,d1) <= L1
            con2 = np.dot(dp,d2) <= L2
            con3 = np.dot(dp,d3) <= L3
            con4 = np.dot(dp,d4) <= L4
            if con1 and con2 and con3 and con4:
                one_bar[0,row,col] = grey_level
                
    return one_bar


def get_one_sin_wave1(K,L,pad,k,w,t):
    N = K*L + 2*pad
    ones = np.ones(N)
    one_sin_wave = np.zeros((1,N,N))
    for col in range(N):
        one_sin_wave[0,:,col] = ones[:]*np.sign(np.sin(k*col-w*t))
        
    return one_sin_wave


def get_one_sin_wave2(K,L,pad,k,w,t):
    N = K*L + 2*pad
    ones = np.ones(N)
    one_sin_wave = np.zeros((1,N,N))
    for row in range(N):
        one_sin_wave[0,row,:] = ones[:]*np.sign(np.sin(k*row-w*t))
        
    return one_sin_wave


def get_reshape(K,intensities):
    steps = intensities.shape[0]
    intensities_re = np.zeros((steps,K,K))
    for step in range(steps):
        intensities_re[step,:,:] = intensities[step,0,:].reshape((K,K))
        
    return intensities_re


def save_intensity(intensity,step,save_name):
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    ax.imshow(intensity, cmap='gray_r', vmin=0, vmax=1)
#         ax.vlines(np.arange(-0.55, N, step=N/K), -0.55, N-0.45, color='salmon', linewidth=.5)
#         ax.hlines(np.arange(-0.55, N, step=N/K), -0.55, N-0.45, color='salmon', linewidth=.5)
    fig.savefig(save_name+'_{}.png'.format(step+1))
    plt.close(fig)
    
    
def save_uv_flow(flow,step,save_name,vmax = 0.08):
    myheat = LinearSegmentedColormap.from_list('br',["b", "w", "r"], N=256)
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    ax.imshow(flow, cmap=myheat, vmin=-vmax, vmax=vmax)
    PCM=ax.get_children()[9] #get the mappable, the 1st and the 2nd are the x and y axes
    plt.colorbar(PCM, ax=ax)
#     ax.vlines(np.arange(-0.55, N, step=N/K), -0.55, N-0.45, color='salmon', linewidth=.5)
#     ax.hlines(np.arange(-0.55, N, step=N/K), -0.55, N-0.45, color='salmon', linewidth=.5)
    fig.savefig(save_name+'_{}.png'.format(step+1))
    plt.close(fig)


if __name__ == "__main__":
    
    start_time = time.time()
    
    K = 12
    L = 30
    dt = 0.01
    p = 10
    pad = 2*L
    delay_dt = 0.03
    
    space_filter = flfd.get_space_filter(L/2,4)
    folder_path = '/Volumes/Baohua/data_on_hd/loom/Klapoetke_stimuli_test/'
    
    for ind,foldertype in enumerate(foldertypes1):
        os.makedirs(folder_path+'Klapoetke_intensities/'+foldertype)
        os.makedirs(folder_path+'Klapoetke_intensities_extended/'+foldertype)
        os.makedirs(folder_path+'Klapoetke_intensities_cg/'+foldertype)
        os.makedirs(folder_path+'Klapoetke_UV_flows/'+foldertype)
    
    for ind,foldertype in enumerate(foldertypes3):
        os.makedirs(folder_path+'Klapoetke_intensities_movies/'+foldertype)
        os.makedirs(folder_path+'Klapoetke_intensities_extended_movies/'+foldertype)
        os.makedirs(folder_path+'Klapoetke_intensities_cg_movies/'+foldertype)
        os.makedirs(folder_path+'Klapoetke_UV_flows_movies/'+foldertype)
    
    savepaths = [folder_path+'Klapoetke_intensities/'+foldertype2+'/' for foldertype2 in foldertypes2]
    get_Klapoetke_intensities(K,L,pad,dt,savepaths)
    
    loadpaths = [folder_path+'Klapoetke_intensities/'+foldertype2+'/' for foldertype2 in foldertypes2]
    savepaths = [folder_path+'Klapoetke_intensities_extended/'+foldertype2+'/' for foldertype2 in foldertypes2]
    names1 = Klapoetke_intensities_names
    names2 = Klapoetke_intensities_extended_names
    get_Klapoetke_intensities_extended(K,L,pad,dt,loadpaths,savepaths,names1,names2)
    
    loadpaths = [folder_path+'Klapoetke_intensities_extended/'+foldertype2+'/' for foldertype2 in foldertypes2]
    savepaths = [folder_path+'Klapoetke_intensities_cg/'+foldertype2+'/' for foldertype2 in foldertypes2]
    names1 = Klapoetke_intensities_extended_names
    names2 = Klapoetke_intensities_cg_names
    get_Klapoetke_intensities_cg(K,L,pad,loadpaths,savepaths,names1,names2)
    
    loadpaths = [folder_path+'Klapoetke_intensities_extended/'+foldertype2+'/' for foldertype2 in foldertypes2]
    savepaths = [folder_path+'Klapoetke_UV_flows/'+foldertype2+'/' for foldertype2 in foldertypes2]
    names1 = Klapoetke_intensities_extended_names
    names2 = Klapoetke_UV_flows_names
    get_Klapoetke_UV_flows(space_filter,K,L,pad,dt,delay_dt,loadpaths,savepaths,names1,names2)

    loadpaths = [folder_path+'Klapoetke_intensities/'+foldertype2+'/' for foldertype2 in foldertypes2]
    savepaths1 = [folder_path+'Klapoetke_intensities_movies/'+foldertype2+'/' for foldertype2 in foldertypes2]
    savepaths2 = [folder_path+'Klapoetke_intensities_movies/'+foldertype3+'/' for foldertype3 in foldertypes3]
    names1 = Klapoetke_intensities_names
    names2 = Klapoetke_intensities_frame_names
    names3 = Klapoetke_intensities_movie_names
    plot_KLapoetke_intensities(dt,p,loadpaths,savepaths1,savepaths2,names1,names2,names3)
    
    loadpaths = [folder_path+'Klapoetke_intensities_extended/'+foldertype2+'/' for foldertype2 in foldertypes2]
    savepaths1 = [folder_path+'Klapoetke_intensities_extended_movies/'+foldertype2+'/' \
                 for foldertype2 in foldertypes2]
    savepaths2 = [folder_path+'Klapoetke_intensities_extended_movies/'+foldertype3+'/' \
                 for foldertype3 in foldertypes3]
    names1 = Klapoetke_intensities_extended_names
    names2 = Klapoetke_intensities_extended_frame_names
    names3 = Klapoetke_intensities_extended_movie_names
    plot_KLapoetke_intensities_extended(dt,p,loadpaths,savepaths1,savepaths2,names1,names2,names3)
    
    loadpaths = [folder_path+'Klapoetke_intensities_cg/'+foldertype2+'/' for foldertype2 in foldertypes2]
    savepaths1 = [folder_path+'Klapoetke_intensities_cg_movies/'+foldertype2+'/'\
                 for foldertype2 in foldertypes2]
    savepaths2 = [folder_path+'Klapoetke_intensities_cg_movies/'+foldertype3+'/'\
                 for foldertype3 in foldertypes3]
    names1 = Klapoetke_intensities_cg_names
    names2 = Klapoetke_intensities_cg_frame_names
    names3 = Klapoetke_intensities_cg_movie_names
    plot_KLapoetke_intensities_cg(K,dt,p,loadpaths,savepaths1,savepaths2,names1,names2,names3)
    
    loadpaths = [folder_path+'Klapoetke_UV_flows/'+foldertype2+'/' for foldertype2 in foldertypes2]
    savepaths1 = [folder_path+'Klapoetke_UV_flows_movies/'+foldertype2+'/' for foldertype2 in foldertypes2]
    savepaths2 = [folder_path+'Klapoetke_UV_flows_movies/'+foldertype3+'/' for foldertype3 in foldertypes3]
    names1 = Klapoetke_UV_flows_names
    names2 = Klapoetke_U_flows_frame_names
    names3 = Klapoetke_V_flows_frame_names
    names4 = Klapoetke_U_flows_movie_names
    names5 = Klapoetke_V_flows_movie_names
    plot_KLapoetke_UV_flows(K,L,pad,dt,p,loadpaths,savepaths1,savepaths2,names1,names2,names3,names4,names5)
    
    print('Time used: {}'.format(time.time()-start_time))
    
    