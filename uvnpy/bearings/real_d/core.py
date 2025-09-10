#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
"""
import numpy as np
from numba import njit

from uvnpy.graphs.core import adjacency_from_geodesics, edges_from_adjacency


THRESHOLD_EIG = 1e-10
THRESHOLD_SV = 1e-5


def bearing_matrix(p):
    """Matrix whose entries are bearings.

    args:
        p: (..., n, d) position array
    """
    r = p[..., np.newaxis, :, :] - p[..., np.newaxis, :]
    d = np.sqrt(np.square(r).sum(axis=-1))
    return r / d[..., np.newaxis]


def bearing_function(E, p):
    """Stack of vectors whose entries are bearings expressed
    in a common reference frame. That is,
    entry e = (i, j) contains:

        (p[j] - p[i]) / dist[i, j]

    args:
        E: edge set (m, 2)-array
        p: positions (..., n, d)-array
    """
    r = p[..., E[:, 1], :] - p[..., E[:, 0], :]
    d = np.sqrt(np.square(r).sum(axis=-1))
    return r / d[..., np.newaxis]


@njit
def bearing_rigidity_matrix(E, p):
    """Bearing rigidity Matrix (jacobian of the bearing function)

    args:
        E: edge set (m, 2)-array
        p: positions (..., n, d)-array
    """
    n, d = p.shape
    m = E.shape[0]
    Id = np.eye(d)
    R = np.zeros((m, d, n, d), dtype=float)
    for e, (i, j) in enumerate(E):
        x = p[j] - p[i]
        q = np.dot(x, x)
        P = Id - np.dot(x.reshape(-1, 1), x.reshape(1, -1)) / q
        M = P / np.sqrt(q)

        R[e, :, i] = -M
        R[e, :, j] = M

    return R.reshape(m * d, n * d)


def bearing_rigidity_laplacian(A, p):
    """Bearing Laplacian / Stiffness matrix.

        S =  R^T W R

    args:
        A: weighted adjacency matrix (..., n, n)-array
        p: positions (..., n, d)-array

    returns:
        S: rigidity laplacian (..., n * d, n * d)-array
    """
    n, d = p.shape[-2:]
    In = np.eye(n, dtype=bool)
    Id = np.eye(d, dtype=float)
    r = p[..., np.newaxis, :] - p[..., np.newaxis, :, :]
    l2 = np.sum(r*r, axis=-1)
    R = r[..., np.newaxis] * r[..., np.newaxis, :]
    S = R / l2[..., np.newaxis, np.newaxis] - Id    # projection matrices
    S *= A[..., np.newaxis, np.newaxis]             # apply weights
    S[..., In, :, :] = 0.0
    S[..., In, :, :] -= S.sum(p.ndim - 1)
    S = S.swapaxes(-3, -2)
    return S.reshape(S.shape[:-4] + 2*(n*d,))


def is_inf_bearing_rigid(E, p, threshold=THRESHOLD_SV):
    n, d = p.shape
    R = bearing_rigidity_matrix(E, p)
    return np.linalg.matrix_rank(R, tol=threshold) == n*d - d - 1


def bearing_rigidity_eigenvalue(A, p):
    d = p.shape[1]
    S = bearing_rigidity_laplacian(A, p)
    eig = np.linalg.eigvalsh(S)
    return eig[d + 1]


def minimum_bearing_rigidity_extents(geodesics, p, threshold=THRESHOLD_SV):
    """
    requires:
        framework is rigid
    """
    n, d = p.shape
    extents = np.empty(n, dtype=int)
    A = adjacency_from_geodesics(geodesics)
    for i in range(n):
        minimum_found = False
        h = 0
        while not minimum_found:
            h += 1
            subset = geodesics[i] <= h
            Ei = edges_from_adjacency(
                A[np.ix_(subset, subset)], directed=False
            )
            pi = p[subset]
            minimum_found = is_inf_bearing_rigid(Ei, pi, threshold)
        extents[i] = h
    return extents
