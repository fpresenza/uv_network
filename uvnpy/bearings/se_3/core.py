#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar may 16 2025
"""
import numpy as np
from numba import njit

from uvnpy.toolkit.geometry import (
    cross_product_matrix,
    rotation_matrix_from_vector,
    rotation_matrix_from_vector_multiple_axes
)


THRESHOLD_EIG = 1e-10
THRESHOLD_SV = 1e-5


def bearing_matrix(x):
    """Matrix whose entries are bearings expressed
    in each robot local reference frame. That is,
    entry [i, j] contains:

        R(ang[i]).T @ (pos[j] - pos[i]) / dist[i, j]

    where
        pos[i] = x[i, :3]
        ang[i] = x[i, 3:]

    args:
        x : (..., n, 6) pose array
    """
    p = x[..., :3]
    a = x[..., 3:]
    R = rotation_matrix_from_vector_multiple_axes(a)

    r = p[..., np.newaxis, :, :] - p[..., np.newaxis, :]
    d = np.sqrt(np.square(r).sum(axis=-1))
    b = r / d[..., np.newaxis]

    return np.matmul(b, R)


def bearing_function(E, x):
    """Stack of vectors whose entries are bearings expressed
    in each robot local reference frame. That is,
    entry e = (i, j) contains:

        R(ang[i]).T @ (pos[j] - pos[i]) / dist[i, j]

    where
        pos[i] = x[i, :3]
        ang[i] = x[i, 3:]

    args:
        x : (..., n, 6) pose array
    """
    p = x[..., :3]
    a = x[..., E[:, 0], 3:]
    R = rotation_matrix_from_vector_multiple_axes(a)

    r = p[..., E[:, 1], :] - p[..., E[:, 0], :]
    d = np.sqrt(np.square(r).sum(axis=-1))
    b = r / d[..., np.newaxis]

    return np.matmul(b[..., np.newaxis, :], R).squeeze()


@njit
def rigidity_matrix(E, x):
    """Rigidity Matrix (jacobian of the bearing function)

    args:
        x : (n, 6) pose array
    """
    n = x.shape[0]
    m = E.shape[0]
    I3 = np.eye(3)
    R = np.zeros((m, 3, n, 6), dtype=float)
    for e, (i, j) in enumerate(E):
        a = x[i, 3:]
        Ct = rotation_matrix_from_vector(-a)

        r = x[j, :3] - x[i, :3]
        q = np.sqrt(np.dot(r, r))
        b = np.dot(Ct, r) / q
        P = I3 - np.dot(b.reshape(-1, 1), b.reshape(1, -1))
        M = np.dot(P, Ct) / q

        Sb = cross_product_matrix(b)
        Sa = cross_product_matrix(a)
        Q = np.dot(a.reshape(-1, 1), a.reshape(1, -1)) - np.dot(Ct.T - I3, Sa)
        N = np.dot(np.dot(Sb, Ct), Q) / np.dot(a, a)

        R[e, :, i, 0:3] = -M
        R[e, :, i, 3:6] = N
        R[e, :, j, 0:3] = M

    return R.reshape(3*m, 6*n)


def is_inf_rigid(E, x, threshold=THRESHOLD_SV):
    n = x.shape[0]
    R = rigidity_matrix(E, x)
    return np.linalg.matrix_rank(R, tol=threshold) == n*6 - 7
