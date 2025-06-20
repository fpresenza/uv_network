#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar feb 14 11:23:21 -03 2025
"""
import numpy as np
from numba import njit

from uvnpy.network import core


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
        E: (m, 2) edge array
        p: (..., n, d) position array
    """
    r = p[..., E[:, 1], :] - p[..., E[:, 0], :]
    d = np.sqrt(np.square(r).sum(axis=-1))
    return r / d[..., np.newaxis]


@njit
def rigidity_matrix(E, p):
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


@njit
def _rigidity_laplacian(A, p):
    n, d = p.shape
    Id = np.eye(d)
    L = np.zeros((n, n, d, d))
    edges = np.argwhere(np.triu(A) > 0)
    for i, j in edges:
        r = (p[j] - p[i]).reshape(d, 1)
        l2 = np.square(r).sum()
        L[i, j] = L[j, i] = (r.dot(r.T) / l2 - Id)
        L[i, i] -= L[i, j]
        L[j, j] -= L[i, j]

    return L


def rigidity_laplacian(A, p):
    L = _rigidity_laplacian(A, p)
    return L.swapaxes(1, 2).reshape(p.size, p.size)


def rigidity_laplacian_multiple_axes(A, p):
    """Matriz normalizada de rigidez.

        S =  R^T W R

    A[i, j] >= 0 respresenta el peso asociado a cada enlace.

    args:
        A: matriz de adyacencia (..., n, n)
        p: array de posiciones (..., n, d)

    returns
        S: laplaciano de rigidez (..., n * d, n * d)
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


def is_inf_rigid(E, p, threshold=THRESHOLD_SV):
    n, d = p.shape
    R = rigidity_matrix(E, p)
    return np.linalg.matrix_rank(R, tol=threshold) == n*d - d - 1


def rigidity_eigenvalue(A, p):
    d = p.shape[1]
    S = rigidity_laplacian(A, p)
    eig = np.linalg.eigvalsh(S)
    return eig[d + 1]


def minimum_rigidity_extents(geodesics, p, threshold=THRESHOLD_SV):
    """
    Requires:
    ---------
        framework is rigid
    """
    n, d = p.shape
    extents = np.empty(n, dtype=int)
    for i in range(n):
        minimum_found = False
        h = 0
        while not minimum_found:
            h += 1
            subset = geodesics[i] <= h
            M = geodesics[np.ix_(subset, subset)]
            Ei = core.edges_from_adjacency(M == 1.0)
            pi = p[subset]
            minimum_found = is_inf_rigid(Ei, pi, threshold)
        extents[i] = h
    return extents
