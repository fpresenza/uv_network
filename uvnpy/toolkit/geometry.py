#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def cross_product_matrix(vector):
    """The matrix cross_product_matrix(vector) that converts
    the cross product into a matrix product.


    Parameters
    ----------
    vector : numpy.ndarray
        A row stack of 3d vectors
    """
    x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]

    S = np.empty(vector.shape + (3,))
    S[..., 0, 0] = 0
    S[..., 0, 1] = -z
    S[..., 0, 2] = y
    S[..., 1, 0] = z
    S[..., 1, 1] = 0
    S[..., 1, 2] = -x
    S[..., 2, 0] = -y
    S[..., 2, 1] = x
    S[..., 2, 2] = 0

    return S


def rotation_matrix_from_vector(theta):
    """Rodrigues rotation formula.


    Parameters
    ----------
    theta : numqy.ndarray
        A row stack of unit vectors

    Returns
    ----------
        The rotation matrices asociated
    """
    angle = np.sqrt(np.sum(np.square(theta), axis=-1))
    axis = theta / angle[..., np.newaxis]

    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    x2 = x**2
    y2 = y**2
    z2 = z**2
    cos1 = 1 - np.cos(angle)
    sin = np.sin(angle)

    R = np.empty(np.shape(axis) + (3,), dtype=np.float64)
    R[..., 0, 0] = cos1 * (-y2 - z2) + 1.0
    R[..., 0, 1] = -z * sin + x * y * cos1
    R[..., 0, 2] = y * sin + x * z * cos1
    R[..., 1, 0] = z * sin + x * y * cos1
    R[..., 1, 1] = cos1 * (-x2 - z2) + 1.0
    R[..., 1, 2] = -x * sin + y * z * cos1
    R[..., 2, 0] = -y * sin + x * z * cos1
    R[..., 2, 1] = x * sin + y * z * cos1
    R[..., 2, 2] = cos1 * (-x2 - y2) + 1.0

    return R


def cone(apex, axis, hypot, fov, resolution=36):
    """

    args:
    -----
       apex       : (array) cone's apex
       axis       : (array) cone's normal axis (unit norm)
       fov        : (float) cone's half angle in radians
       hypot      : (float) cone's hypothenuse
       resolution : (int) number of circular sections
    """
    # get points
    r = hypot * np.sin(fov)
    angles = np.linspace(0, 2*np.pi, resolution)

    points = np.empty((resolution, 3), dtype=float)
    points[:, 0] = r * np.cos(angles)
    points[:, 1] = r * np.sin(angles)
    points[:, 2] = - hypot * np.cos(fov) * np.ones(angles.size)

    # get rotation matrix
    down = np.array([0.0, 0.0, -1.0])
    v = np.cross(down, axis)
    Sv = cross_product_matrix(v)
    s = np.sqrt(v.dot(v))
    c = down.dot(axis)
    if c == -1:
        R = c * np.eye(3) + s * Sv + (1 - c) * np.outer(v, v)
    else:
        R = np.eye(3) + Sv + Sv.dot(Sv) / (1 + c)

    # rotate points
    points = points.dot(R.T) + apex

    return [np.vstack([apex, points[i-1:i+1]]) for i in range(1, resolution)]
