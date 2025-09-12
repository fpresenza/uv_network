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

from uvnpy.graphs.core import complete_edges, edge_set_diff


THRESHOLD_EIG = 1e-10
THRESHOLD_SV = 1e-5


def distance_matrix(x):
    """ Devuelve matriz de distancias.

    args:
        x: array de posiciones (..., n, d)
    """
    r = x[..., None, :] - x[..., None, :, :]
    dist = np.sqrt(np.square(r).sum(axis=-1))
    return dist


def distances_from_edges(E, p):
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


@njit
def distance_rigidity_matrix(E, p):
    """Distance rigidity Matrix (jacobian of the distance function)

    args:
        E : edge set | (m, 2)-array
        p : positions | (..., n, d)-array
    """
    n, d = p.shape
    m = E.shape[0]
    R = np.zeros((m, n*d), dtype=float)
    for e, (i, j) in enumerate(E):
        r = p[i] - p[j]
        b = r / np.sqrt(np.square(r).sum(axis=-1))
        R[e, d*i:d*(i+1)] = b
        R[e, d*j:d*(j+1)] = -b
    return R


def distance_rigidity_laplacian(A, p):
    """Distance Laplacian / Stiffness matrix.

        S =  R^T W R

    args:
        A : weighted adjacency matrix | (..., n, n)-array
        p : positions | (..., n, d)-array

    returns:
        S : rigidity laplacian | (..., n * d, n * d)-array
    """
    n, d = p.shape[-2:]
    ii = np.eye(n, dtype=bool)
    r = unit_vector(p[..., None, :] - p[..., None, :, :], axis=-1)
    r[..., ii, :] = 0
    S = - r[..., None] * r[..., None, :]    # outer product
    S *= A[..., None, None]                 # apply weights
    S[..., ii, :, :] -= S.sum(p.ndim - 1)
    S = S.swapaxes(-3, -2)
    s = list(S.shape)
    s[-4:] = n * d, n * d
    return S.reshape(s)


def is_distance_rigid(E, p, threshold=THRESHOLD_SV):
    n, d = p.shape
    f = int(d * (d + 1)/2)
    R = distance_rigidity_matrix(E, p)
    return np.linalg.matrix_rank(R, tol=threshold) == n*d - f


def distance_rigidity_eigenvalue(A, p):
    d = p.shape[1]
    f = int(d * (d + 1) / 2)
    S = distance_rigidity_laplacian(A, p)
    eig = np.linalg.eigvalsh(S)
    return eig[f]


def minimum_distance_rigidity_extents(E, G, p, threshold=THRESHOLD_SV):
    """
    requires:
        framework is rigid
    """
    n, d = p.shape
    extents = np.empty(n, dtype=int)
    remap = np.empty(n, dtype=int)
    for i in range(n):
        minimum_found = False
        h = 0
        while not minimum_found:
            h += 1
            subset = G[i] <= h
            Ei = E[subset[E].all(axis=1)]
            pi = p[subset]
            remap[subset] = np.arange(sum(subset))
            Ei = remap[Ei]
            minimum_found = is_distance_rigid(Ei, pi, threshold)
        extents[i] = h
    return extents


def minimum_distance_rigidity_radius(
        E, p, threshold=THRESHOLD_SV, return_radius=False):
    """Add or delete edges to a framework until it is radius-wise minimally
    rigid."""
    n, d = p.shape
    f = d * (d + 1) // 2

    if is_distance_rigid(E, p, threshold):
        dist = distances_from_edges(E, p)
        F = E.copy()
        rigid = True
        while rigid:
            E = F.copy()
            e = np.argmax(dist)
            radius = dist[e]
            dist = np.delete(dist, e)
            F = np.delete(F, e, axis=0)
            rigid = is_distance_rigid(F, p, threshold)
    else:
        K = complete_edges(n, directed=False)
        notE = edge_set_diff(K, E)
        dist = distances_from_edges(notE, p)

        rigid = False
        while not rigid:
            e = np.argmin(dist)
            radius = dist[e]
            E = np.vstack([E, notE[e]])
            dist = np.delete(dist, e)
            notE = np.delete(notE, e, axis=0)
            if len(E) >= d*n - f:
                rigid = is_distance_rigid(E, p, threshold)

    if return_radius:
        return E, radius
    else:
        return E
