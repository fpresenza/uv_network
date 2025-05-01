#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar ene 26 18:33:51 -03 2021
"""
import numpy as np
from transformations import unit_vector
from numba import njit


THRESHOLD_EIG = 1e-6
THRESHOLD_SV = 1e-3


def distance_matrix(x):
    """ Devuelve matriz de distancias.

    args:
        x: array de posiciones (..., n, d)
    """
    r = x[..., None, :] - x[..., None, :, :]
    dist = np.sqrt(np.square(r).sum(axis=-1))
    return dist


def distance_matrix_from_edges(E, p):
    r = p[..., E[:, 0], :] - p[..., E[:, 1], :]
    dist = np.sqrt(np.square(r).sum(axis=-1))
    return dist


def distance_matrix_from_adjacency(A, p):
    r = p[..., None, :] - p[..., None, :, :]
    dist = np.sqrt(np.square(r).sum(axis=-1)) * A
    return dist[dist > 0]


def minimum_distance(p, axis=None):
    r = p[..., None, :] - p[..., None, :, :]
    d = np.sqrt(np.square(r).sum(axis=-1))
    d[..., np.eye(p.shape[-2], dtype=bool)] = np.nan
    return np.nanmin(d, axis=axis)


def sufficiently_dispersed_position(n, xlim, ylim, max_dist):
    """
    Generates a set of nodes positions where each node is away from
    the rest at least max_dist.
    """
    xl, xh = xlim
    yl, yh = ylim
    p = np.random.uniform((xl, yl), (xh, yh), (1, 2))
    for i in range(1, n):
        found = False
        while not found:
            q = np.random.uniform((xl, yl), (xh, yh), 2)
            dist2 = np.square(p - q).sum(axis=1)
            if np.all(dist2 > max_dist**2):
                found = True
                p = np.vstack([p, q])

    return p


def classic_rigidity_matrix_from_incidence(D, p):
    """Matriz de rigidez

    args:
        D: matriz de incidencia (n, m)
        p: array de posiciones (n, d)

    returns
        R: jacobiano (ne, n * d)
    """
    n, d = p.shape[-2:]
    Dt = D.T
    r = np.matmul(Dt, p)
    J = Dt[..., None] * r[..., None, :]
    return J.reshape(-1, n*d)


@njit
def classic_rigidity_matrix(A, p):
    n, d = p.shape
    num_edges = int(A.sum() / 2)
    R = np.zeros((num_edges, n*d))
    e = 0
    for i in range(n):
        for j in range(i+1, n):
            if A[i, j] == 1:
                dij = p[i] - p[j]
                R[e, d*i:d*(i+1)] = dij
                R[e, d*j:d*(j+1)] = -dij
                e += 1
    return R


def classic_rigidity_matrix_multiple_axes(D, p):
    """Matriz de rigidez

    args:
        D: matriz de incidencia (n, m)
        p: array de posiciones (n, d)

    returns
        R: jacobiano (ne, n * d)
    """
    n, d = p.shape[-2:]
    m = D.shape[-1]
    Dt = D.T
    r = np.matmul(Dt, p)
    J = Dt[..., None] * r[..., None, :]
    return J.reshape(-1, m, n*d)


def rigidity_matrix(D, p):
    """Matriz de rigidez

    args:
        D: matriz de incidencia (n, m)
        p: array de posiciones (n, d)

    returns
        R: jacobiano (ne, n * d)
    """
    n, d = p.shape[-2:]
    Dt = D.T
    r = unit_vector(np.matmul(Dt, p), axis=-1)
    J = Dt[..., None] * r[..., None, :]
    return J.reshape(-1, n*d)


def rigidity_matrix_multiple_axes(D, p):
    """Matriz de rigidez

    args:
        D: matriz de incidencia (n, m)
        p: array de posiciones (n, d)

    returns
        R: jacobiano (ne, n * d)
    """
    n, d = p.shape[-2:]
    m = D.shape[-1]
    Dt = D.T
    r = unit_vector(np.matmul(Dt, p), axis=-1)
    J = Dt[..., None] * r[..., None, :]
    return J.reshape(-1, m, n*d)


@njit
def _classic_rigidity_laplacian(A, p):
    n, d = p.shape
    L = np.zeros((n, n, d, d))
    edges = np.argwhere(np.triu(A) > 0)
    for i, j in edges:
        dji = p[j] - p[i]
        L[i, j] = L[j, i] = -dji.reshape(d, 1).dot(dji.reshape(1, d))
        L[i, i] -= L[i, j]
        L[j, j] -= L[i, j]

    return L


def classic_rigidity_laplacian(A, p):
    L = _classic_rigidity_laplacian(A, p)
    return L.swapaxes(1, 2).reshape(p.size, p.size)


@njit
def _rigidity_laplacian(A, p):
    n, d = p.shape
    L = np.zeros((n, n, d, d))
    edges = np.argwhere(np.triu(A) > 0)
    for i, j in edges:
        dij = p[i] - p[j]
        nij = np.square(dij).sum()
        L[i, j] = L[j, i] = dij.reshape(d, 1).dot(dij.reshape(1, d)) / -nij
        L[i, i] -= L[i, j]
        L[j, j] -= L[i, j]

    return L


def rigidity_laplacian(A, p):
    L = _rigidity_laplacian(A, p)
    return L.swapaxes(1, 2).reshape(p.size, p.size)


def classic_rigidity_laplacian_multiple_axes(A, p):
    """Matriz de rigidez.

        S =  R^T W R

    A[i, j] >= 0 respresenta el peso asociado a cada enlace.

    args:
        A: matriz de adyacencia (..., n, n)
        p: array de posiciones (..., n, d)

    returns
        S: laplaciano de rigidez (..., n * d, n * d)
    """
    n, d = p.shape[-2:]
    ii = np.eye(n, dtype=bool)
    r = p[..., None, :] - p[..., None, :, :]
    r[..., ii, :] = 0
    S = - r[..., None] * r[..., None, :]    # outer product
    S *= A[..., None, None]                 # aplicar pesos
    S[..., ii, :, :] -= S.sum(p.ndim - 1)
    S = S.swapaxes(-3, -2)
    s = list(S.shape)
    s[-4:] = n * d, n * d
    return S.reshape(s)


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
    ii = np.eye(n, dtype=bool)
    r = unit_vector(p[..., None, :] - p[..., None, :, :], axis=-1)
    r[..., ii, :] = 0
    S = - r[..., None] * r[..., None, :]    # outer product
    S *= A[..., None, None]                 # aplicar pesos
    S[..., ii, :, :] -= S.sum(p.ndim - 1)
    S = S.swapaxes(-3, -2)
    s = list(S.shape)
    s[-4:] = n * d, n * d
    return S.reshape(s)


def is_inf_rigid(A, p, threshold=THRESHOLD_SV):
    n, d = p.shape
    f = int(d * (d + 1)/2)
    R = classic_rigidity_matrix(A, p)
    return np.linalg.matrix_rank(R, tol=threshold) == n*d - f


def rigidity_eigenvalue(A, p):
    d = p.shape[1]
    f = int(d * (d + 1) / 2)
    S = rigidity_laplacian(A, p)
    eig = np.linalg.eigvalsh(S)
    return eig[f]


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


def minimum_rigidity_radius(A, p, threshold=THRESHOLD_SV, return_radius=False):
    """Add or delete edges to a framework until it is radius-wise minimally
    rigid."""
    A = A.copy()
    n, d = p.shape
    f = d * (d + 1) // 2
    dist = distance_matrix(p)

    if is_inf_rigid(A, p, threshold):
        B = A.copy()
        dist = dist * B

        rigid = True
        while rigid:
            A = B.copy()
            i, j = np.unravel_index(np.argmax(dist), (n, n))
            radius = dist[i, j]
            dist[i, j] = dist[j, i] = 0
            B[i, j] = B[j, i] = 0
            rigid = is_inf_rigid(B, p, threshold)
    else:
        B = np.eye(n) + A
        B[B == 1] = np.inf
        dist = dist + B

        rigid = False
        while not rigid:
            i, j = np.unravel_index(np.argmin(dist), (n, n))
            radius = dist[i, j]
            dist[i, j] = dist[j, i] = np.inf
            A[i, j] = A[j, i] = 1
            e = A.sum() // 2
            if e >= d*n - f:
                rigid = is_inf_rigid(A, p, threshold)

    if return_radius:
        return A, radius
    else:
        return A
