#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
"""
import numpy as np
from numba import njit

THRESHOLD_EIG = 1e-10
THRESHOLD_SV = 1e-5


@njit
def angle_set(n, E):
    """Compute the angles indices with the directed edges.
    An angle is a triple (i, j, k) where (i, j) and (i, k)
    are directed edges.

    args:
        E : edge set | (m, 2)-array

    returns:
        angle_set | (a, 3)-array
    """
    a = np.empty(shape=(0, 3), dtype=float)
    for i in range(n):
        S = E[:, 0] == i
        Ei = E[S]
        x, y = np.triu_indices(sum(S), k=1)
        ai = np.concatenate((Ei[x], Ei[y, 1, np.newaxis]), axis=1)
        a = np.concatenate((a, ai), axis=0)
    return a


@njit
def angle_function(E, p):
    """Compute the angles associated with the directed edges.
    An angle is a triple (i, j, k) where (i, j) and (i, k)
    are directed edges.

    args:
        E : edge_set | (m, 2)-array
        p : positions | (..., n, d)-array

    returns:
        angle_set | (..., a)-array
    """
    n = p.shape[-2]
    r = p[..., E[:, 1], :] - p[..., E[:, 0], :]
    d = np.sqrt(np.square(r).sum(axis=-1))
    b = r / d[..., np.newaxis]
    a = np.empty(shape=(0,), dtype=float)
    for i in range(n):
        S = E[:, 0] == i
        bi = b[S]
        x, y = np.triu_indices(sum(S), k=1)
        ai = np.sum(bi[x] * bi[y], axis=1)
        a = np.concatenate((a, ai))
    return a


@njit
def angle_rigidity_matrix(E, p):
    """Angle Rigidity matrix (jacobian of the bearing function)

    args:
        E: edge set | (m, 2)-array
        p: positions | (..., n, d)-array

    returns:
        angle rigidity matrix | (a, n*d)
    """
    n, d = p.shape
    Id = np.eye(d)

    r = p[..., E[:, 1], :] - p[..., E[:, 0], :]
    q = np.sqrt(np.square(r).sum(axis=-1))
    b = r / q[..., np.newaxis]
    R = np.empty(shape=(0, n*d), dtype=float)
    for i in range(n):
        S = E[:, 0] == i
        s = sum(S)
        bi = b[S]
        qi = q[S]
        Pi = Id - bi[..., :, np.newaxis] * bi[..., np.newaxis, :]
        Mi = Pi / qi[..., np.newaxis, np.newaxis]

        x, y = np.triu_indices(s, k=1)
        Nij = np.sum(bi[y, :, np.newaxis] * Mi[x], axis=1)
        Nik = np.sum(bi[x, :, np.newaxis] * Mi[y], axis=1)

        ds = int(s * (s - 1) / 2)
        Ri = np.zeros(shape=(ds, n, d), dtype=float)
        Ei = E[S]
        j, k = Ei[x, 1], Ei[y, 1]
        for a in range(ds):
            Ri[a, i] = - Nij[a] - Nik[a]
            Ri[a, j[a]] = Nij[a]
            Ri[a, k[a]] = Nik[a]
        R = np.concatenate((R, Ri.reshape(ds, n*d)))

    return R


def is_angle_rigid(E, p, threshold=THRESHOLD_SV):
    n, d = p.shape
    f = int(d * (d + 1)/2)
    R = angle_rigidity_matrix(E, p)
    return np.linalg.matrix_rank(R, tol=threshold) == n*d - f - 1
