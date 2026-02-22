#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numba import njit


@njit
def cross_product_matrix(vector):
    """The matrix cross_product_matrix(vector) that converts
    the cross product into a matrix product.

    args:
        vector : (3, )-array
    """
    x, y, z = vector[0], vector[1], vector[2]

    S = np.empty((3, 3), dtype=float)
    S[0, 0] = 0
    S[0, 1] = -z
    S[0, 2] = y
    S[1, 0] = z
    S[1, 1] = 0
    S[1, 2] = -x
    S[2, 0] = -y
    S[2, 1] = x
    S[2, 2] = 0

    return S


def cross_product_matrix_multiple_axes(vector):
    """The matrix cross_product_matrix(vector) that converts
    the cross product into a matrix product.

    args:
        vector : (..., 3)-array
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


@njit
def rotation_matrix_from_vector(theta):
    """Rodrigues rotation formula.


    args:
        theta : nonzero vector (3, )-array

    returns:
        The rotation matrices asociated
    """
    angle = np.sqrt(np.sum(np.square(theta)))
    axis = theta / angle

    x, y, z = axis[0], axis[1], axis[2]
    x2 = x**2
    y2 = y**2
    z2 = z**2
    cos1 = 1 - np.cos(angle)
    sin = np.sin(angle)

    R = np.empty((3, 3), dtype=np.float64)
    R[0, 0] = cos1 * (-y2 - z2) + 1.0
    R[0, 1] = -z * sin + x * y * cos1
    R[0, 2] = y * sin + x * z * cos1
    R[1, 0] = z * sin + x * y * cos1
    R[1, 1] = cos1 * (-x2 - z2) + 1.0
    R[1, 2] = -x * sin + y * z * cos1
    R[2, 0] = -y * sin + x * z * cos1
    R[2, 1] = x * sin + y * z * cos1
    R[2, 2] = cos1 * (-x2 - y2) + 1.0

    return R


@njit
def rotation_matrix_from_quaternion(q):
    q0 = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]

    R = np.empty((3, 3), dtype=np.float64)

    R[0, 0] = 2 * (q0 * q0 + qx * qx) - 1
    R[0, 1] = 2 * (qx * qy - q0 * qz)
    R[0, 2] = 2 * (qx * qz + q0 * qy)

    R[1, 0] = 2 * (qx * qy + q0 * qz)
    R[1, 1] = 2 * (q0 * q0 + qy * qy) - 1
    R[1, 2] = 2 * (qy * qz - q0 * qx)

    R[2, 0] = 2 * (qx * qz - q0 * qy)
    R[2, 1] = 2 * (qy * qz + q0 * qx)
    R[2, 2] = 2 * (q0 * q0 + qz * qz) - 1

    return R


def rotation_matrix_from_vector_multiple_axes(theta):
    """Rodrigues rotation formula.

    args:
        theta : nonzero vector (.., 3)-array

    returns:
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


def triangle(center, heading, height, ratio=0.3):
    """Compute coordinates of a triangle.

    args:
        center  : coordinates of the base center | (2, )-array
        heading : orientation | float
        height  : length from center to apex | float
        ratio   : base / height | float
    """
    x = center[..., 0]
    y = center[..., 1]
    ct = height * np.cos(heading)
    st = height * np.sin(heading)

    vertices = np.empty((len(center), 3, 2), dtype=float)
    vertices[..., 0, 0] = x + ct
    vertices[..., 0, 1] = y + st
    vertices[..., 1, 0] = x - ratio * st
    vertices[..., 1, 1] = y + ratio * ct
    vertices[..., 2, 0] = x + ratio * st
    vertices[..., 2, 1] = y - ratio * ct

    return vertices


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
