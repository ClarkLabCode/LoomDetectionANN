#!/usr/bin/env python


"""
This module contains functions to simulate the optical signals of moving objects.
"""


import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation as R3d
import dynamics_3d as dn3d


# Produce one of the intensity frames in the video
def get_one_intensity(M, pos, Rs, theta_r, theta_matrix, coord_matrix, K, L, pad, around_z_angles, sigma=0.0):
    """
    Args:
    M: # of lplc2 units
    pos: P by 3, the current positions of the centers of the P balls.
    Rs: len(Rs) = P, the radii of the P balls.
    theta_r: angular radius of the receptive field (rad).
    theta_matrix: theta matrix
    coord_matrix: coordinate matrix
    K: K*K is the total # of elements
    L: element dimension
    pad: padding size
    sigma: noise level on the intensity.
    
    Returns:
    cf_raw: M by N by N, raw current frame, slightly larger than the actual frame due to padding
    cf: M by N-2*pad by N-2*pad, current frame.
    hit: binary, whether hit or not
    """
    N = K * L + 2 * pad # Size of each frame, the additional 4*L is for spatial filtering and will be deleted afterwards.
    N_half = (N - 1) / 2.
    cf_raw = np.zeros((M, N, N)) # current raw frame
    cf = np.zeros((M, N-2*pad, N-2*pad)) # current frame
    P = pos.shape[0]
    Ds = dn3d.get_radial_distances(pos)
    hit = Ds <= Rs
    
    if M == 1:
        lplc2_units = np.zeros((1, 2))
    else:
        lplc2_units = get_lplc2_units_xy_angles(M)
        
    # Mask to remove signals outside the receptive field
    mask_2 = np.logical_not(theta_matrix<=theta_r)
    mask_2 = mask_2[pad:-pad, pad:-pad]
    for m in range(M):
        mask_1_T = np.zeros((N, N))
        angle = -lplc2_units[m]
        around_z = around_z_angles[m]
        pos_rot = get_rotated_coordinates(angle, pos, around_z)
        for p in range(P):
            x, y, z = pos_rot[p]
            R = Rs[p]
            D = dn3d.get_radial_distance(x, y, z)
            # Get the disk image of the ball on the frame
            theta_b = get_angular_size(x, y, z, R)
            angle_matrix_b = get_angle_matrix_b(coord_matrix, pos_rot[p])
            mask_1 = angle_matrix_b <= theta_b
            mask_1_T = np.logical_or(mask_1, mask_1_T)

        cf_raw[m, mask_1_T] = 1.0
        # Add noise to the signal
        noise = sigma * np.random.normal(size=N*N).reshape(N, N)
        cf_raw[m, :, :] = cf_raw[m,:,:] + noise
        # Crop the frame
        cf[m, :, :] = cf_raw[m, pad:-pad, pad:-pad]
        cf[m, mask_2] = 0.
    
    return np.float32(cf), np.float32(cf_raw), hit


# Get coarse-grained (cg) frame
def get_intensity_cg(intensity, leftup_corners, K, L, pad):
    """
    Args:
    intensity: one frame of optical flow, M by N by N
    leftup_corners: list of tuples, coordinates of the left-up corners of all the elements
    K: K*K is the total # of elements.
    L: element dimension.
    pad: padding size.
    
    Returns:
    intensity_cg: coarse-grained intensity, a list with length M, each element is a K*K vector.
    """
    M = intensity.shape[0]
    intensity_cg = []
    for m in range(M):
        intensity_cg_tem = np.zeros(K*K, np.float32)
        if intensity[m, :, :].any():
            for counter, leftup_corner in enumerate(leftup_corners):
                if within_receptive(leftup_corner, K, L, pad):
                    row_range, col_range = get_element_range(leftup_corner, L)
                    intensity_cg_tem[counter] = \
                        intensity[m, row_range.min():row_range.max()+1, col_range.min():col_range.max()+1].mean()
            intensity_cg.append(intensity_cg_tem)
        else:
            intensity_cg.append(np.array([-1.], np.float32))
        
    return intensity_cg


# Get the indices of the left-up corner of each element given the dimension of the frame N and # of element centers K*K
def get_leftup_corners(K, L, pad):
    """
    Args:
    K: K*K is the totoal # of elements
    L: dimension of each element
    pad: padding size
    
    Returns:
    leftup_corners: indices of the left-up corner of each element on the frame
    """
    leftup_corners = []
    for row in range(K):
        for col in range(K):
            row_value = row * L + pad
            col_value = col * L + pad
            leftup_corners.append([row_value, col_value])
            
    return np.array(leftup_corners)


# Get the whole element 
def get_element_range(leftup_corner, L):
    """
    Args:
    leftup_corner: tuple, indices of the left-up corner of the element under consideration
    L: element dimension
    
    Return:
    row_range: row range of the element
    col_range: column range of the element
    """
    row_range = np.arange(leftup_corner[0], leftup_corner[0]+L)
    col_range = np.arange(leftup_corner[1], leftup_corner[1]+L)
    
    return row_range, col_range


# Get the element center
def get_element_center(leftup_corner, L):
    """
    Args:
    leftup_corner: tuple, indices of the left-up corner of the element under consideration
    L: element dimension
    
    Return:
    element_center: indices of the element center
    """
    L_half = (L - 1) / 2.
    element_center = (leftup_corner[0] + L_half, leftup_corner[1] + L_half)
    
    return element_center


# Check whether within receptive field
def within_receptive(leftup_corner, K, L, pad):
    """
    Args:
    leftup_corner: tuple, indices of the left-up corner of the element under consideration
    K: K*K is the total # of elements
    L: element dimension
    pad: padding size
    
    Return:
    within_resep: whether the element indicated by the leftup corner is within the receptive field. True or False.
    """
    N = K * L + 2 * pad
    N_half = (N - 1) / 2.
    element_center = get_element_center(leftup_corner, L)
    d = np.sqrt((element_center[0]-N_half)**2 + (element_center[1]-N_half)**2)
    within_resep = d <= N_half - pad
    
    return within_resep


# Calculate the angular size of the ball with respective to the origin
def get_angular_size(x, y, z, R):
    """
    Args:
    (x, y, z): the current position of the center of the ball.
    R: the radius of the ball.
    
    Returns:
    theta_b: the half of the angular size of the ball.
    """
    D = dn3d.get_radial_distance(x, y, z)
    if D != 0:
        theta_b = np.arcsin(R/D)
        return theta_b
    else:
        print('Error: the radial distance is 0!')


# Calculate the angular size of the ball with respective to the origin
def get_angular_size_tan(x, y, z, R):
    """
    Args:
    (x, y, z): the current position of the center of the ball.
    R: the radius of the ball.
    
    Returns:
    theta_b: the half of the angular size of the ball.
    """
    D = dn3d.get_radial_distance(x, y, z)
    if D != 0:
        theta_b = np.arctan(R/D)
        return theta_b
    else:
        print('Error: the radial distance is 0!')
        
        
# Get a list of roughly evenly distributed lplc2_units on a sphere
# Adopted from https://stackoverflow.com/questions/9600801/evenly-distributing-n-lplc2_units-on-a-sphere
def get_lplc2_units(M, randomize=False):
    """
    Args:
    M: # of lplc2 units
    randomize: whether the distribution is randomized
    
    Returns:
    lplc2_units: M by 2, a list of angles that represent the centerlines of the lplc2 units
    """
    rnd = 1.
    if randomize:
        rnd = random.random() * M
    lplc2_units = []
    lplc2_units_coords = []
    offset = 2. / M
    increment = np.pi * (3. - np.sqrt(5.));
    for i in range(M):
        y = ((i * offset) - 1) + (offset / 2);
        r = np.sqrt(1 - pow(y, 2))
        phi = ((i + rnd) % M) * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        theta_s, phi_s = dn3d.get_spherical_angles(x, y, z)
        lplc2_units.append([phi_s, theta_s])
        lplc2_units_coords.append([x, y, z])

    return np.array(lplc2_units), np.array(lplc2_units_coords)


# Get a list of roughly evenly distributed lplc2_units on a sphere
# Adopted from https://stackoverflow.com/questions/9600801/evenly-distributing-n-lplc2_units-on-a-sphere
def get_lplc2_units_xy_angles(M,randomize=False):
    """
    Args:
    M: # of lplc2 units
    randomize: whether the distribution is randomized
    
    Returns:
    lplc2_units_xy: M by 2, a list of angles that represent the centerlines of the lplc2 units around x and y intrinsically
    """
    rnd = 1.
    if randomize:
        rnd = random.random() * M
    lplc2_units_xy = []
    offset = 2. / M
    increment = np.pi * (3. - np.sqrt(5.));
    for i in range(M):
        y = ((i * offset) - 1) + (offset / 2);
        r = np.sqrt(1 - pow(y, 2))
        phi = ((i + rnd) % M) * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        lplc2_units_xy.append(get_xy_angles(x, y, z))

    return np.array(lplc2_units_xy)


# Get angles that represent the rotations around x and y intrinsically to align z axis with this vector
def get_xy_angles(x, y, z):
    """
    Args:
    x, y, z: coordinates that represent the direction of a vector
    
    Returns:
    np.array([angle_x, angle_y]): angles that represent the rotations around x and y intrinsically to align z axis with this
    vector
    """
    if y <= 0: 
        if y == 0 and z == 0:
            angle_x = 0
        else:
            angle_x = np.arccos(z/(np.sqrt(y**2 + z**2)))
    else:
        angle_x = 2 * np.pi - np.arccos(z/(np.sqrt(y**2 + z**2)))
    yz = np.sqrt(y**2 + z**2)
    if x >= 0:
        angle_y = np.arccos(yz/(np.sqrt(x**2 + yz**2)))
    else:
        angle_y = 2 * np.pi - np.arccos(yz/(np.sqrt(x**2 + yz**2)))

    return np.array([angle_x, angle_y])


# Rotate intrinsically around x and y according to an angle
def get_rotated_coordinates(angle, pos, around_z=0):
    """
    Args:
    angle: [around_x, around_y] (rad).
    pos: P by 3, the current positions of the centers of the P balls.
    
    Returns:
    pos: the rotated positions of the centers of the balls.
    """
    around_x, around_y = angle
    r = R3d.from_euler('YXZ', [around_y, around_x, around_z], degrees=False)

    pos = r.apply(pos)
#     if around_x >= np.pi:
#         pos[:, 1] = -pos[:, 1]
    
    return pos


# Rotate intrinsically around x and y according to an angle reversed
def get_rotated_coordinates_rev(angle, pos, around_z=0):
    """
    Args:
    angle: [around_x, around_y] (rad).
    pos: P by 3, the current positions of the centers of the P balls.
    
    Returns:
    pos: the rotated positions of the centers of the balls.
    """
    around_x, around_y = angle
    r = R3d.from_euler('ZXY', [around_z, around_x, around_y], degrees=False)
    pos = r.apply(pos)
#     if around_x >= np.pi:
#         pos[:, 1] = -pos[:, 1]
    
    return pos


# Rotate the axis intrinsically according to an angle
def get_rotated_axes(angle, pos, rev=False, around_z=0):
    """
    Args:
    angle: (around_x, around_y) (rad).
    pos: P by 3, the current positions of the centers of the P balls.
    rev: reversed or not
    
    Returns:
    pos: the rotated positions of the centers of the balls.
    """
    around_x, around_y = angle
    r = R3d.from_euler('XYZ', [around_x, around_y, around_z], degrees=False)
    pos = r.apply(pos)
    if rev and around_x >= np.pi:
        pos[:, :] = -pos[:, :]
    
    return pos


# Calculate the angle between two vectors
def get_angle_two_vectors(vec_1, vec_2):
    """
    Args:
    vec_1: input vector
    vec_2: input vector
    
    Returns:
    angle: the angle between vec_1 and vec_2 (rad)
    """
    vec_1_norm = np.linalg.norm(vec_1)
    vec_2_norm = np.linalg.norm(vec_2)
    if vec_1_norm != 0 and vec_2_norm != 0:
        vec_1 = vec_1 / vec_1_norm
        vec_2 = vec_2 / vec_2_norm
        angle = np.arccos(np.clip(np.dot(vec_1, vec_2), -1., 1.))
        return angle
    else:
        print('Error: input vectors has lengths of 0!')
        
        
# Calculate the angle between two lplc2 units
def get_angle_between_lplc2(M, c1, c2):
    """
    Args:
    M: # of lplc2 units
    c1: index of unit 1
    c2: index of unit 2
    
    Returns:
    angle: the angle between two lplc2 units, in degree
    """
    _, lplc2_units_coords = get_lplc2_units(M)
    vec_1 = np.array(lplc2_units_coords[c1])
    vec_2 = np.array(lplc2_units_coords[c2])
    angle = get_angle_two_vectors(vec_1, vec_2)
    
    return angle * 180. / np.pi


# Calculate the angle between two lplc2 units
def get_angles_between_lplc2_and_vec(M, vec):
    """
    Args:
    M: # of lplc2 units
    vec: vector
    
    Returns:
    angles: the angles between lplc2 units and vec, in degree
    """
    angles = []
    if M > 1:
        _, lplc2_units_coords = get_lplc2_units(M)
    else:
        lplc2_units_coords = np.array([[0, 0, 1]])
    for m in range(M):
        lplc2_coord = lplc2_units_coords[m]
        angle = get_angle_two_vectors(lplc2_coord, vec)
        angles.append(angle*180./np.pi)
    
    return np.array(angles)


# Rotate a vector around the other vector clockwise
def rotate_vector_clockwise(vec_1, vec_2, angle):
    """
    This function rotates vec_1 around vec_2 clockwise with an angle angle.
    
    Args:
    vec_1: input vector
    vec_2: input vector, rotation axis
    angle: angle to rotate of vec_1 around vec_2 (rad)
    
    Returns:
    vec_1_new: rotated vec_1
    """
    vec_1_norm = np.linalg.norm(vec_1)
    vec_2_norm = np.linalg.norm(vec_2)
    if vec_1_norm != 0 and vec_2_norm != 0:
        vec_2 = vec_2 / vec_2_norm
        vec_1_para = np.dot(vec_1, vec_2) * vec_2
        vec_1_perp = vec_1 - vec_1_para
        vec_1_perp_norm = np.linalg.norm(vec_1_perp)
        vec_1_perp_hat = vec_1_perp / vec_1_perp_norm
        vec_cross_hat = np.cross(vec_1_perp_hat, vec_2)
        vec_1_new = vec_1_perp_norm * (vec_cross_hat * np.sin(angle) + vec_1_perp_hat * np.cos(angle)) + vec_1_para
        return vec_1_new
    else:
        print('Error: input vectors has lengths of 0!')


def get_angle_matrix(theta_r, coords_x, coords_y, dm, K, L):
    """    
    Args:
    theta_r: angular radius of the receptive field (rad).
    coords_x: coordinates of the frame in the vertical direction (x axis)
    coords_y: coordinates of the frame in the horizontal direction (y axis)
    dm: distance matrix calculated from coords_ud and coords_lr
    K: K*K is the total # of elements
    L: element dimension
    
    Returns:
    theta_matrix, phi_matrix: angle matrices
    """
    
    N = K * L
    N_half = (N - 1) / 2.
    theta_matrix = dm / N_half * theta_r
    phi_matrix = np.pi * (1 - np.sign(coords_y)) + np.multiply(np.sign(coords_y), np.arccos(np.divide(coords_x, dm)))
    
    return theta_matrix, phi_matrix


def get_coord_matrix(phi_matrix, theta_matrix, D=1.):
    """
    Args:
    theta_matrix, phi_matrix: angle matrices
    D: distance
    
    Returns:
    coord_matrix: coordinate matrix.
    """
    z = D * np.cos(theta_matrix)
    x = np.multiply(D*np.sin(theta_matrix), np.cos(phi_matrix))
    y = np.multiply(D*np.sin(theta_matrix), np.sin(phi_matrix))
    
    coord_matrix = np.stack((x, y, z))
    coord_matrix = np.swapaxes(coord_matrix, 0, 1)
    coord_matrix = np.swapaxes(coord_matrix, 1, 2)
    
    return coord_matrix


def get_angle_matrix_b(coord_matrix, ball_center):
    """
    Args:
    coord_matrix: coordinate matrix
    ball center: center of the ball
    
    Returns:
    angle_matrix_b: angle matrix with respect to the ball center.
    """
    ball_center_norm = np.linalg.norm(ball_center)
    if ball_center_norm != 0:
        ball_center = ball_center / ball_center_norm
        angle_matrix_b = np.arccos(np.clip(np.dot(coord_matrix, ball_center), -1., 1.))
        return angle_matrix_b
    else:
        print('Error: input vectors has lengths of 0!')
        return None
    

# Get disk mask
def get_disk_mask(K, L):
    """
    Args:
    K: K*K is the total # of elements
    L: element dimension
    
    Returns:
    disk_mask: boolean, disk mask
    """
    disk_mask = np.full(K*K, True)
    leftup_corners = get_leftup_corners(K, L, 0)
    for counter, leftup_corner in enumerate(leftup_corners):
        if within_receptive(leftup_corner, K, L, 0):
            disk_mask[counter] = False
    
    return disk_mask.reshape((K, K))

