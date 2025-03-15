#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar feb 14 11:23:21 -03 2025
"""
import numpy as np
# from transformations import unit_vector
from numba import njit


THRESHOLD_EIG = 1e-6
THRESHOLD_SV = 1e-3


@njit
def rigidity_matrix(A, p):
    n, d = p.shape
    m = int(A.sum() / 2)
    Id = np.eye(d)
    R = np.zeros((m*d, n*d))
    e = 0
    for i in range(n):
        for j in range(i+1, n):
            if A[i, j] == 1:
                x = p[j] - p[i]
                q = np.dot(x, x)
                P = Id - np.dot(x.reshape(-1, 1), x.reshape(1, -1)) / q
                M = P / np.sqrt(q)
                di = d * i
                dj = d * j
                de = d * e
                R[de:de + d, di:di + d] = -M
                R[de:de + d, dj:dj + d] = M
                e += 1
    return R


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
    P = R / l2[..., np.newaxis, np.newaxis]
    S = P - Id                             # projection matrices
    S *= A[..., np.newaxis, np.newaxis]    # apply weights
    S[..., In, :, :] = 0.0
    S[..., In, :, :] -= S.sum(p.ndim - 1)
    S = S.swapaxes(-3, -2)
    s = list(S.shape)
    s[-4:] = n * d, n * d
    return S.reshape(s)


def is_inf_rigid(A, p, threshold=THRESHOLD_SV):
    n, d = p.shape
    R = rigidity_matrix(A, p)
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
    A = geodesics.copy()
    A[A > 1] = 0
    extents = np.empty(n, dtype=int)
    for i in range(n):
        minimum_found = False
        h = 0
        while not minimum_found:
            h += 1
            subset = geodesics[i] <= h
            Ai = A[np.ix_(subset, subset)]
            pi = p[subset]
            minimum_found = is_inf_rigid(Ai, pi, threshold)
        extents[i] = h
    return extents
