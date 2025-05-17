#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar may 16 2025
"""
import numpy as np
from numba import njit


THRESHOLD_EIG = 1e-6
THRESHOLD_SV = 1e-3


def bearing_matrix(x):
    """Matrix whose entries are bearings.

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
    B = bearing_matrix(x)
    b = B[..., E[:, 0], E[:, 1], :]
    *r, s, t = b.shape
    return b.reshape(*r, s * t, 1)


@njit
def rigidity_matrix(E, x):
    n = x.shape[0]
    s = x.shape[1]
    d = s - 1
    m = len(E)
    Id = np.eye(d)
    C = np.zeros((d, d), dtype=float)
    C[2:, 2:] = 1.0
    R = np.zeros((m*d, n*s))
    for e, (i, j) in enumerate(E):
        y = x[i, d]
        C[0, 0] = np.cos(y)
        C[0, 1] = np.sin(y)
        C[1, 0] = -np.sin(y)
        C[1, 1] = np.cos(y)
        r = x[j, :d] - x[i, :d]
        q = np.sqrt(np.dot(r, r))
        b = np.dot(C, r) / q
        P = Id - np.dot(b.reshape(-1, 1), b.reshape(1, -1))
        M = np.dot(P, C) / q
        si = s * i
        sj = s * j
        de = d * e
        R[de:de + d, si:si + d] = -M
        R[de, si + d] = b[1]
        R[de + 1, si + d] = -b[0]
        R[de:de + d, sj:sj + d] = M

    return R
