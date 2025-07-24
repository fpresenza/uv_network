#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar may 16 2025
"""
import numpy as np
from numba import njit


THRESHOLD_EIG = 1e-10
THRESHOLD_SV = 1e-5


def bearing_matrix(x):
    """Matrix whose entries are bearings expressed
    in each robot local reference frame. That is,
    entry [i, j] contains:

        R(yaw).T @ (pos[j] - pos[i]) / dist[i, j]

    where
        pos[i] = x[i, :d]
        yaw[i] = x[i, d]

    args:
        x: (..., n, d + 1) position array
    """
    d = x.shape[-1] - 1
    p = x[..., :d]
    y = x[..., d]
    R = np.zeros(y.shape + (d, d), dtype=float)
    R[..., 0, 0] = np.cos(y)
    R[..., 0, 1] = -np.sin(y)
    R[..., 1, 0] = np.sin(y)
    R[..., 1, 1] = np.cos(y)
    R[..., 2:, 2:] = 1.0

    r = p[..., np.newaxis, :, :] - p[..., np.newaxis, :]
    d = np.sqrt(np.square(r).sum(axis=-1))
    b = r / d[..., np.newaxis]
    return np.matmul(b, R)


def bearing_function(E, x):
    """Stack of vectors whose entries are bearings expressed
    in each robot local reference frame. That is,
    entry e = (i, j) contains:

        R(yaw).T @ (pos[j] - pos[i]) / dist[i, j]

    where
        pos[i] = x[i, :d]
        yaw[i] = x[i, d]

    args:
        x
    """
    d = x.shape[-1] - 1
    p = x[..., :d]
    y = x[..., E[:, 0], d]
    R = np.zeros(y.shape + (d, d), dtype=float)
    R[..., 0, 0] = np.cos(y)
    R[..., 0, 1] = -np.sin(y)
    R[..., 1, 0] = np.sin(y)
    R[..., 1, 1] = np.cos(y)
    R[..., 2:, 2:] = np.eye(d - 2, dtype=float)

    r = p[..., E[:, 1], :] - p[..., E[:, 0], :]
    d = np.sqrt(np.square(r).sum(axis=-1))
    b = r / d[..., np.newaxis]

    return np.matmul(b[..., np.newaxis, :], R).squeeze()


@njit
def rigidity_matrix(E, x):
    """Rigidity Matrix (jacobian of the bearing function)

    args:
        x: (..., n, d + 1) position array
    """
    n = x.shape[0]
    s = x.shape[1]
    d = s - 1
    m = len(E)
    Id = np.eye(d)
    Ct = np.eye(d)
    R = np.zeros((m, d, n, s), dtype=float)
    for e, (i, j) in enumerate(E):
        y = x[i, d]
        Ct[0, 0] = np.cos(y)
        Ct[0, 1] = np.sin(y)
        Ct[1, 0] = -np.sin(y)
        Ct[1, 1] = np.cos(y)

        r = x[j, :d] - x[i, :d]
        q = np.sqrt(np.dot(r, r))
        b = np.dot(Ct, r) / q
        P = Id - np.dot(b.reshape(-1, 1), b.reshape(1, -1))
        M = np.dot(P, Ct) / q

        R[e, :, i, 0:d] = -M
        R[e, 0, i, d] = b[1]
        R[e, 1, i, d] = -b[0]
        R[e, :, j, 0:d] = M

    return R.reshape(d * m, s * n)


def is_inf_rigid(E, x, threshold=THRESHOLD_SV):
    n = x.shape[0]
    s = x.shape[1]
    R = rigidity_matrix(E, x)
    return np.linalg.matrix_rank(R, tol=threshold) == n*s - s - 1
