#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  21 15:38:09 2016
@author: pato
Modified by fran on Mon Dec 7 16:19:08 2019
"""
from math import cos, sin
import numpy as np
import recordtype
from tf.transformations import euler_from_quaternion, quaternion_from_euler
  
Vec3 = recordtype.recordtype('Vec3', 'x y z')
Sen = recordtype.recordtype('Sensor',
    [('sigma', 0.), 
    ('bias_sample', Vec3(0.,0.,0.)),
    ('bias', Vec3(0.,0.,0.)),
    ('bias_drift', Vec3(0.,0.,0.)),
    ('meas', Vec3(0.,0.,0.))])
GPSMeas = recordtype.recordtype('GPSMeas', 'speed track climb longitude latitude altitude')

def saturate(item, limits):
    """Returns a saturated value between limits"""
    try:
        retval = [min(max(value, limits[0]), limits[1]) for value in item]
    except TypeError:
        retval = min(max(item, limits[0]), limits[1])
    return retval

def wrap2pi(angle):
    """Return input angle wraped between [-np.pi, np.pi)"""
    return np.mod(angle + np.pi, 2*np.pi) - np.pi

def angular_wrap(vector, angulars):
    """Returns elements of vector, whose corresponding element in angulars is
    True, wraped between [-np.pi, np.pi)
    """
    return [wrap2pi(value) if is_angular else value for (value, is_angular) in zip(vector, angulars)]

def pose_split(pose):
    """Convierte una pose (de un mensaje) a dos listas.

    Arguments:
        pose        = objeto con miembros position y orientation
    Returns:
        position    = [pose.position.{x,y,z}]
        orientation = [pose.orientation.{x,y,z,w}]
    """
    position = [pose.position.x, pose.position.y, pose.position.z]
    orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

    return position, orientation

def pose2xyzY(pose):
    """Convierte una pose (de un mensaje) a xyzY (xyz yaw).

    Arguments:
        pose        = objeto con miembros position y orientation
    Returns:
        position    = [pose.position.{x,y,z} yaw]
    """
    position = [pose.position.x, pose.position.y, pose.position.z]
    orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    _, _, yaw = euler_from_quaternion(orientation)

    return [position[0], position[1], position[2], yaw]

def pose_tolist(pose):
    """Convierte una pose (de un mensaje) a una lista.

    Arguments:  pose = objeto con miembros position y orientation
    Returns:    v = [pose.position.{x,y,z} pose.orientation.{x,y,z,w}]
    """
    return [pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w]

def pose_toarray(pose, dtype='float'):
    """Convierte una pose (de un mensaje) a un array de numpy.

    Arguments:  pose = objeto con miembros position y orientation
    Returns:    a = array([pose.position.{x,y,z} pose.orientation.{x,y,z,w}])
    """
    return np.array([pose.position.x,
                     pose.position.y,
                     pose.position.z,
                     pose.orientation.x,
                     pose.orientation.y,
                     pose.orientation.z,
                     pose.orientation.w], dtype=dtype)

def world2body_matrix(yaw, dtype='float'):
    """Returns the rotation matrix needed to transform coordinates from world
    frame to body frame
    """
    return np.array([
        [cos(yaw), -sin(yaw), 0, 0],
        [sin(yaw), cos(yaw), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]], dtype=dtype)

def process_mocap_data(pose, orientation, C_px4_world, att_matrix_switch, xyz0=None):
    """Acondiciona los datos de pose (x, y, z) y orientation (qx, qy, qz, qw)
    generando una pose nueva (x, y, z, yaw)"""

    if xyz0 is None:
        xyz0 = [0, 0, 0, 0]

    roll, pitch, yaw = euler_from_quaternion(orientation)

    ## Esto sería get_yaw() de matlab...
    attitude = np.dot(att_matrix_switch, np.array([yaw, pitch, roll]))
    yaw_w = attitude[0] + 0 * attitude[2]
    ## hasta acá
    yaw_w += 0 # offset

    pose = np.dot(C_px4_world, np.array(pose) - np.array(xyz0[0:3]))

          #   X        Y        Z      YAW
    return pose[0], pose[1], pose[2], yaw_w

def fake_mocap_data(xyz_yaw, C_px4_world, att_matrix_switch, xyz0=None):
    """Fake MoCap data"""

    if xyz0 is None:
        xyz0 = [0, 0, 0, 0]

    position = np.dot(C_px4_world.transpose(), np.array(xyz_yaw[0:3]) - np.array(xyz0[0:3]))
    orientation = quaternion_from_euler(0, 0, xyz_yaw[-1] - xyz0[-1])

    return list(position), list(orientation)
