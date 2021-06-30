#!/usr/bin/env python


"""
This module contains functions to generate predefined weights for excitatory and inhibitory neurons.
"""


import numpy as np
import matplotlib.pyplot as plt
import optical_signal as opsg


# calculate the angle between the cell center and the cardinal directions.
def get_angle_from_cardinal(leftup_corner, cardinal, K, L):
    """
    Args:
    leftup_corner: the left-up corner of the current cell
    cardinal: string, 'up', 'down', 'right', or 'left'. The cardinal direction under consideration.
    
    Returns:
    theta_d: the angle between the cell center and the cardinal direction.
    """
    N = K * L
    frame_center = ((N - 1) / 2.,(N - 1) / 2.)
    cell_center = (leftup_corner[0] + (L - 1) / 2., leftup_corner[1] + (L - 1) / 2.)
    cell_center = tuple(np.subtract(cell_center, frame_center))
    if np.linalg.norm(cell_center) == 0:
        theta_d = 0.
    else:
        cell_center = cell_center / np.linalg.norm(cell_center)
        if cardinal == 'up':
            ax_vec = (-1, 0)
        elif cardinal == 'down':
            ax_vec = (1, 0)
        elif cardinal == 'right':
            ax_vec = (0, 1)
        elif cardinal == 'left':
            ax_vec = (0, -1)
        theta_d = np.arccos(np.dot(cell_center, ax_vec))
    
    return theta_d


# get the weights
def get_weights(leftup_corners, cardinal, theta_dt, K, L):
    """
    Args:
    leftup_corners: left-up corners of all cells
    cardinal: string, 'up', 'down', 'right', or 'left'. The cardinal direction under consideration.
    theta_dt: threshold value for the angle between cell center and cardinal directions.
    K: K*K is the total # of cells.
    L: cell dimension.
    
    Return:
    weights: K*K by 1, filter weight for the cardinal direction.
    """
    N = K * L
    weights = np.zeros(K*K)
    for counter, leftup_corner in enumerate(leftup_corners):
        theta_d = get_angle_from_cardinal(leftup_corner, cardinal, K, L)
        if theta_d <= np.deg2rad(theta_dt):
            frame_center = ((N - 1) / 2., (N - 1) / 2.)
            cell_center = (leftup_corner[0] + (L - 1) / 2., leftup_corner[1] + (L - 1) / 2.)
            cell_center = tuple(np.subtract(cell_center, frame_center))
            if np.linalg.norm(cell_center) <= int((N-1)/2.):
                weights[counter] = 1.
    
    return weights[:]


# Get weights for excitatory neurons
def get_all_weights_e(leftup_corners, theta_dt, K, L, scl=1):
    """
    Args:
    leftup_corners: left-up_corners of all cells
    theta_dt: degrees, threshold value for the angle between cell center and cardinal directions.
    K: K*K is the total # of cells.
    L: cell dimension.
    scl: scale
    
    Return:
    weights_e: K*K by 4, for axis=1, 0:right motion, 1:left motion, 2:up motion, 3:down motion
    """
    weights_e = np.zeros((K*K, 4))
    weights_e[:,0] = scl * get_weights(leftup_corners, 'right', theta_dt, K, L)
    weights_e[:,1] = scl * get_weights(leftup_corners, 'left', theta_dt, K, L)
    weights_e[:,2] = scl * get_weights(leftup_corners, 'up', theta_dt, K, L)
    weights_e[:,3] = scl * get_weights(leftup_corners, 'down', theta_dt, K, L)
    
    return weights_e.astype(np.float32)
    

# Get weights for inhibitory neurons
def get_all_weights_i(leftup_corners, theta_dt, K, L, scl=1):
    """
    Args:
    leftup_corners: left-up corners of all cells
    theta_dt: degrees, threshold value for the angle between cell center and cardinal directions.
    K: K*K is the total # of cells.
    L: cell dimension.
    scl: scale
    
    Return:
    weights_i: K*K by 4, for axis=1, 0:right motion, 1:left motion, 2:up motion, 3:down motion
    """
    weights_i = np.zeros((K*K, 4))
    weights_i[:,0] = scl*get_weights(leftup_corners, 'left', theta_dt, K, L)
    weights_i[:,1] = scl*get_weights(leftup_corners, 'right', theta_dt, K, L)
    weights_i[:,2] = scl*get_weights(leftup_corners, 'down', theta_dt, K, L)
    weights_i[:,3] = scl*get_weights(leftup_corners, 'up', theta_dt, K, L)
    
    return weights_i.astype(np.float32)


# plot weight on frame
def set_weights_on_frame(weights, leftup_corners, K, L):
    """
    Args:
    weights: filter weights.
    leftup_corners: left-up corners of all cells
    K: K*K is the total # of cells.
    L: cell dimension.
    
    Returns:
    weights_cf: weights on frame
    """
    N = K * L
    
    weights_cf = np.zeros((N, N))
    for counter, leftup_corner in enumerate(leftup_corners):
        row_range, col_range = opsg.get_cell_range(leftup_corner, L)
        row_slice = slice(np.min(row_range), np.max(row_range)+1)
        col_slice = slice(np.min(col_range), np.max(col_range)+1)
        weights_cf[row_slice, col_slice] = weights[counter]
    
    return weights_cf


    
