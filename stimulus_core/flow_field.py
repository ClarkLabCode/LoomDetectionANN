#!/usr/bin/env python


"""
This module contains functions to extract the flow fields information from the optical signals of moving objects in an LPLC2's receptive field.
"""


import numpy as np
import optical_signal as opsg


# Calculate the flow field
def get_flow_fields(signal_filtered_all, signal_cur, leftup_corners, K, L, pad):
    """
    Args:
    signal_filtered_all: M by K*K by 4
    signal_cur: M by K*K by 4
    leftup_corners: indices of the left-up corner of each element on the frame
    
    Returns:
    UV_flow: flow field
    """
    u, v = get_flow_vector_hrc(signal_filtered_all, signal_cur)
    M = u.shape[0]
    K = np.int(np.sqrt(u.shape[1]))
    UV_flow = []  
    for m in range(M):
        if u[m, :].any() or v[m, :].any():
            UV_flow_tem = np.zeros((K*K, 4), np.float32)
            for counter, leftup_corner in enumerate(leftup_corners):
                if opsg.within_receptive(leftup_corner, K, L, pad):
                    if u[m, counter] > 0:
                        UV_flow_tem[counter, 0] = u[m, counter]
                    elif u[m, counter] < 0:
                        UV_flow_tem[counter, 1] = -u[m, counter]
                    if v[m, counter] > 0:
                        UV_flow_tem[counter, 2] = v[m, counter]
                    elif v[m, counter] < 0:
                        UV_flow_tem[counter, 3] = -v[m, counter]
            UV_flow.append(UV_flow_tem)
        else:
            UV_flow.append(np.array([-1.], np.float32))
        
    return UV_flow


# Get the flow vectors from a pair of consecutive frames using the Hessenstein-Reinhardt correlator
def get_flow_vector_hrc(signal_filtered_all, signal_cur):
    """
    Args:
    signal_filtered: filtered signal, with length of 4, 0: left, 1: right, 2: down, 3: up.
    signal_cur: current signal, with length of 4, 0: left, 1: right, 2: down, 3: up.
    
    Return:
    u: flow vector in the horizontal (y) direction
    v: flow vector in the vertical (x) direction
    """
    # Calculate U field
    signal_left_pre = signal_filtered_all[:, :, 0]
    signal_right_pre = signal_filtered_all[:, :, 1]
    signal_left_cur = signal_cur[:, :, 0]
    signal_right_cur = signal_cur[:, :, 1]
    u = signal_right_cur * signal_left_pre - signal_right_pre * signal_left_cur
    # Calculate V field
    signal_down_pre = signal_filtered_all[:, :, 2]
    signal_up_pre = signal_filtered_all[:, :, 3]
    signal_down_cur = signal_cur[:, :, 2]
    signal_up_cur = signal_cur[:, :, 3]
    v = signal_up_cur * signal_down_pre - signal_up_pre * signal_down_cur
    
    return u, v


# get filtered intensity and current intensity
def get_filtered_and_current(signal_filtered_all, intensity_sample, leftup_corners, space_filter, K, L, pad, dt, delay_dt):
    """
    Args:
    signal_filtered_all: filtered signal, M by K*K by 4. For the last axis, 0: left, 1: right, 2: down, 3: up.
    intensity_sample: current intensity, M by N by N
    leftup_corners: tuple, coordinates of the element centers
    K: K*K is the total # of elements
    L: element dimension
    pad: padding size
    dt: time step (sec)
    delay_dt: timescale of delay in the motion detector.
    
    Return:
    signal_filtered: filtered signal
    signal_cur: current signal
    """
    M = intensity_sample.shape[0]
    signal_cur = np.zeros((M, K*K, 4))
    for ind, leftup_corner in enumerate(leftup_corners):
        left_value, right_value, down_value, up_value = get_filtered_space(intensity_sample, leftup_corner, space_filter, L)
        signal_cur[:, ind, 0] = left_value
        signal_cur[:, ind, 1] = right_value
        signal_cur[:, ind, 2] = down_value
        signal_cur[:, ind, 3] = up_value

    signal_filtered_all = get_filtered(signal_filtered_all, signal_cur, dt, delay_dt)
    
    return signal_filtered_all, signal_cur


# Set flow fields on frame
def set_flow_fields_on_frame(UV_flow, leftup_corners, K, L, pad):
    """
    Args:
    UV_flow: a list with length M
    leftup_corners: indices of the left-up corner of each element on the frame
    K: K*K is the total # of elements
    L: element dimension
    pad: padding size
    
    Returns:
    cf_u: current frame with U field
    cf_v: current frame with V field
    """
    M = len(UV_flow)
    N = K*L
    cf_u = np.zeros((M, K, K))
    cf_v = np.zeros((M, K, K))
    for m in range(M):
        UV_flow_tem = UV_flow[m]
        if UV_flow_tem.shape[0] == 1:
            pass
        else:
            for counter, leftup_corner in enumerate(leftup_corners):
                row = np.int(counter/K)
                col = np.mod(counter, K)
                cf_u[m, row, col] = UV_flow_tem[counter, 0] - UV_flow_tem[counter, 1] 
                cf_v[m, row, col] = UV_flow_tem[counter, 2] - UV_flow_tem[counter, 3]
        
    return np.float32(cf_u), np.float32(cf_v)


# Set flow fields on frame
def set_flow_fields_on_frame2(UV_flow, leftup_corners, K, L, pad):
    """
    Args:
    UV_flow: M by K*K by 4 numpy array
    leftup_corners: indices of the left-up corner of each element on the frame
    K: K*K is the total # of elements
    L: element dimension
    pad: padding size
    
    Returns:
    cf_u: current frame with U field
    cf_v: current frame with V field
    """
    M = len(UV_flow)
    N = K*L
    cf_u = np.zeros((M, N, N))
    cf_v = np.zeros((M, N, N))
    
    for m in range(M):
        if UV_flow[m].shape[0] > 1:
            for counter, leftup_corner in enumerate(leftup_corners):
                leftup_corner = (leftup_corner[0] - pad, leftup_corner[1] - pad)
                row_range, col_range = opsg.get_element_range(leftup_corner, L)
                row_slice = slice(np.min(row_range), np.max(row_range)+1)
                col_slice = slice(np.min(col_range), np.max(col_range)+1)
                cf_u[m, row_slice, col_slice] = UV_flow[m][counter, 0] - UV_flow[m][counter, 1] 
                cf_v[m, row_slice, col_slice] = UV_flow[m][counter, 2] - UV_flow[m][counter, 3]
        
    return np.float32(cf_u), np.float32(cf_v)


# Get filtered signal
def get_filtered(signal_filtered_all, signal_cur, dt, delay_dt):
    """
    Args:
    signal_filtered_all: filtered signal
    signal_cur: current signal
    dt: simulation time step (sec)
    delay_dt: timescale of the filter (sec)
    
    Returns:
    signal_filtered_all: filtered signal
    """
    signal_filtered_all = np.exp(-dt/delay_dt) * signal_filtered_all + (1-np.exp(-dt/delay_dt)) * signal_cur
#     signal_filtered_all = (1-dt/delay_dt) * signal_filtered_all + (dt/delay_dt) * signal_cur
        
    return signal_filtered_all


# Get filtered in space
def get_filtered_space(intensity, leftup_corner, space_filter, L):
    """
    Args:
    intensity: M by N by N numpy array
    leftup_corners: indices of the left-up corner of each element on the intensity
    space_filter: 
    L: element dimension
    
    Returns:
    left_value, right_value, down_value, up_value: filtered values at the four points
    """
    rown = leftup_corner[0]
    coln = leftup_corner[1]
    left_boundary = np.array([[rown-L-L/2, rown+L+L+L/2], [coln-2*L, coln+2*L]]).astype(int)
    right_boundary = np.array([[rown-L-L/2, rown+L+L+L/2], [coln-L, coln+3*L]]).astype(int)
    down_boundary = np.array([[rown-L, rown+3*L], [coln-L-L/2, coln+L+L+L/2]]).astype(int)
    up_boundary = np.array([[rown-2*L, rown+2*L], [coln-L-L/2, coln+L+L+L/2]]).astype(int)
    left_field = intensity[:, left_boundary[0, 0]:left_boundary[0, 1], left_boundary[1, 0]:left_boundary[1, 1]]
    right_field = intensity[:, right_boundary[0, 0]:right_boundary[0, 1], right_boundary[1, 0]:right_boundary[1, 1]]
    down_field = intensity[:, down_boundary[0, 0]:down_boundary[0, 1], down_boundary[1, 0]:down_boundary[1, 1]]
    up_field = intensity[:, up_boundary[0, 0]:up_boundary[0, 1], up_boundary[1, 0]:up_boundary[1, 1]]
    left_value = np.multiply(space_filter, left_field).sum(axis=(1, 2))
    right_value = np.multiply(space_filter, right_field).sum(axis=(1, 2))
    down_value = np.multiply(space_filter, down_field).sum(axis=(1, 2))
    up_value = np.multiply(space_filter, up_field).sum(axis=(1, 2))
    
    return left_value, right_value, down_value, up_value
    
    
# Spacial filter
def get_space_filter(sigma_1, truncation):
    """
    Args:
    sigma_1: standard deviation of the Gaussian
    truncation: # of sigmas to truncate the Gaussian
    
    Returns:
    space_filter: space filter
    """
    half_range = sigma_1 * truncation
    coord_y = np.arange(-half_range+0.5, half_range+0.5)
    coord_x = np.arange(-half_range+0.5, half_range+0.5)
    coords_y, coords_x = np.meshgrid(coord_y, -coord_x) # coordinates of each point in the frame
    dm = np.sqrt(coords_y**2 + coords_x**2) # distance matrix squared
    
    space_filter = np.exp(-dm**2/(2*sigma_1**2))
    mask = np.logical_not(dm<=truncation*sigma_1)
    space_filter[mask] = 0
    space_filter = space_filter / space_filter.sum()
    
    return space_filter
    
    












