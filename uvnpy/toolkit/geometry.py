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
