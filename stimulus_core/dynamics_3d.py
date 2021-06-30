#!/usr/bin/env python


"""
This module contains functions to simulate the dynamics of moving objects.
"""


import numpy as np


# Calculate the radial distance of the ball with respect to the origin given the current position of the center
# of the ball
def get_radial_distance(x, y, z):
    """
    Args:
    (x, y, z): the current position of the center of the ball.
    
    Returns:
    D: radial distance.
    """
    D = np.sqrt(x**2 + y**2 + z**2)
    
    return D


# Calculate the radial distances of multiple balls with respect to the origin given the current positions of the centers
# of the balls
def get_radial_distances(pos):
    """
    Args:
    pos: P by 3, the current positions of the centers of the P balls.
    
    Returns:
    Ds: (P,), radial distances.
    """
    Ds = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + pos[:, 2]**2)
    
    return Ds


# Calculate the spherical angles of the ball with respect to the origin given the current position of the center
# of the ball
def get_spherical_angles(x, y, z):
    """
    Args:
    (x, y, z): the current position of the center of the ball.
    
    Returns:
    theta_s: the polar angle (rad).
    phi_s: the azimuthal angle (rad).
    """
    D = get_radial_distance(x, y, z)
    if D != 0:
        theta_s = np.arccos(z/D)
        if theta_s != 0:
            if y >= 0:
                ra = x / (D * np.sin(theta_s))
                ra = np.maximum(ra, -1)
                ra = np.minimum(ra, 1)
                phi_s = np.arccos(ra)
            else:
                ra = x / (D * np.sin(theta_s))
                ra = np.maximum(ra, -1)
                ra = np.minimum(ra, 1)
                phi_s = 2 * np.pi - np.arccos(ra)
        else:
            phi_s = 0 # phi_s is not defined if theta_s is zero, but set to zero for convenience.
        return theta_s, phi_s
    else:
        print('Error: the radial distance is 0!')
        
        
# Calculate the the current position of the center of the ball given spherical angles of the ball 
def get_coord(D, phi_s, theta_s):
    """
    Args:
    D: distance
    theta_s: the polar angle (rad).
    phi_s: the azimuthal angle (rad).
    
    Returns:
    (x, y, z): the current position of the center of the ball.
    """
    z = D * np.cos(theta_s)
    x = D * np.sin(theta_s) * np.cos(phi_s)
    y = D * np.sin(theta_s) * np.sin(phi_s)
    
    return [x, y, z]
        

# Udate the position of the center of the disk
def update_position(x, y, z, vx, vy, vz, dt):
    """
    Args:
    (x, y, z): the current position of the center of the ball
    (vx, vy, vz): the current velocity of the center of the ball (/sec)
    dt: time step for each frame (sec)
    
    Returns:
    (x, y, z): next position
    """
    x = x + vx * dt
    y = y + vy * dt
    z = z + vz * dt
    
    return x, y, z


# Udate the velocity of the center of the disk
def update_velocity(vx, vy, vz, ax, ay, az, dt):
    """
    Args:
    (vx, vy, vz): the current velocity of the center of the ball (/sec)
    (ax, ay, az): accelerations of the center of the ball (/sec^2)
    dt: time step for each frame (sec)
    
    Returns:
    (vx, vy, vz): next velocity (/sec)
    """
    vx = vx + ax * dt
    vy = vy + ay * dt
    vz = vz + az * dt
    
    return vx, vy, vz


# Predefined dynamics of the object when there is no field.
def dynamics_fun_zero_field(x, y, z, vx, vy, vz, eta_1, t, dt):
    """
    Args:
    (x, y, z): the current position of the center of the ball, not used here
    (vx, vy, vz): the current velocity of the center of the ball (/sec), not used here
    eta_1: some random acceleration imposed on the object (/sec^2)
    t: current time point, since the potential can be time dependent, not used here.
    dt: time step for each frame (sec)
    
    Returns:
    (ax,ay,az): accelerations of the center of the ball (/sec^2)
    """
    ax = np.random.normal(0, eta_1)
    ay = np.random.normal(0, eta_1)
    az = np.random.normal(0, eta_1)
    
    return ax, ay, az


# Predefined dynamics of the object when there is a uniform magnetic field along the z axis.
def dynamics_fun_uni_mag(x, y, z, vx, vy, vz, eta_1, t, dt):
    """
    Args:
    (x, y, z): the current position of the center of the ball, not used here
    (vx, vy, vz): the current velocity of the center of the ball (/sec)
    eta_1: some random acceleration imposed on the object (/sec^2)
    t: current time point, since the potential can be time dependent, not used here.
    dt: time step for each frame (sec)
    
    Returns:
    (ax, ay, az): accelerations of the center of the ball (/sec^2)
    """
    ax = -vy + np.random.normal(0, eta_1)
    ay = vx + np.random.normal(0, eta_1)
    az = np.random.normal(0, eta_1)
    
    return ax, ay, az


# Predefined dynamics of the object in critically damped harmonic oscillators
def dynamics_fun_damped_harmonic(x, y, z, vx, vy, vz, eta_1, t, dt):
    """
    Args:
    (x, y, z): the current position of the center of the ball, not used here
    (vx, vy, vz): the current velocity of the center of the ball (/sec), not used here
    eta_1: some random acceleration imposed on the object (/sec^2)
    t: current time point, since the potential can be time dependent, not used here.
    dt: time step for each frame (sec)
    
    Returns:
    (ax, ay, az): accelerations of the center of the ball (/sec^2)
    """
    k = 1. + 0.5 * eta_1
    ax = -k * x - 2 * np.sqrt(k) * vx + np.random.normal(0, eta_1)
    ay = -k * y - 2 * np.sqrt(k) * vy + np.random.normal(0, eta_1)
    az = np.random.normal(0, np.minimum(np.absolute(vz), eta_1))
    
    return ax, ay, az


# Predefined dynamics of the object in critically damped harmonic oscillators
def dynamics_fun_gravity(x, y, z, vx, vy, vz, eta_1, t, dt):
    """
    Args:
    (x, y, z): the current position of the center of the ball
    (vx, vy, vz): the current velocity of the center of the ball (/sec)
    eta_1: some random acceleration imposed on the object (/sec^2)
    t: current time point, since the potential can be time dependent, not used here.
    dt: time step for each frame (sec)
    
    Returns:
    (ax, ay, az): accelerations of the center of the ball (/sec^2)
    """
    vv = vx**2 + vy**2 + vz**2
    DD = x**2 + y**2 + z**2
    ax = -vv / DD * x + np.random.normal(0, eta_1)
    ay = -vv / DD * y + np.random.normal(0, eta_1)
    az = -vv / DD * z + np.random.normal(0, eta_1)
    
    return ax, ay, az


# Predefined dynamics of the object as a predator
def dynamics_fun_predator(x, y, z, vx, vy, vz, eta_1, t, dt):
    """
    Args:
    (x, y, z): the current position of the center of the ball, not used here
    (vx, vy, vz): the current velocity of the center of the ball (/sec), not used here
    eta_1: some random acceleration imposed on the object (/sec^2)
    t: current time point, since the potential can be time dependent, not used here.
    dt: time step for each frame (sec)
    
    Returns:
    (ax, ay, az): accelerations of the center of the ball (/sec^2)
    """
    r_vec = np.array([-x, -y, -z])
    v_vec = np.array([vx, vy, vz])
    
    if np.cross(r_vec, v_vec).sum() == 0. and np.dot(r_vec, v_vec) < 0.:
        ax = np.random.normal(0, 0.1)
        ay = np.random.normal(0, 0.1)
        az = np.random.normal(0, 0.1)
    else:
        angle = get_angle_two_vectors(r_vec, v_vec)
        if angle <= 5. * np.pi / 180.:
            a_vec = r_vec / np.linalg.norm(r_vec) * np.linalg.norm(v_vec) - v_vec
        else:
            v_vec_new = rotate_vector_clockwise(v_vec, np.cross(r_vec, v_vec), angle/5.)
            a_vec = v_vec_new - v_vec
 
        a_vec = a_vec / dt
        ax = a_vec[0] + np.random.normal(0, eta_1)
        ay = a_vec[1] + np.random.normal(0, eta_1)
        az = a_vec[2] + np.random.normal(0, eta_1)
      
    return ax, ay, az

