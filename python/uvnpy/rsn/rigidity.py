#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mié ago 11 21:15:33 -03 2021
"""
import numpy as np
import scipy.linalg
import itertools
from transformations import unit_vector
from numba import njit

from uvnpy.network import subsets
from uvnpy.rsn import distances


THRESHOLD = 1e-4


def classic_matrix(D, p):
    """Matriz de rigidez clásica"""
    Dt = D.T
    r = Dt.dot(p)
    R = Dt[:, :, None] * r[:, None]
    return R.reshape(-1, p.size)


def matrix(D, p):
    """Matriz de rigidez normalizada

    args:
        D: matriz de incidencia (n, m)
        p: array de posiciones (n, d)

    returns
        R: jacobiano (ne, n * d)
    """
    Dt = D.T
    r = unit_vector(Dt.dot(p), axis=-1)
    J = Dt[:, :, None] * r[:, None]
    return J.reshape(-1, p.size)


def matrix_from_adjacency(A, p):
    """Matriz de rigidez normalizada

    args:
        A: matriz de adyacencia (n, n)
        p: array de posiciones (n, d)

    returns
        R: (ne, n * d)
    """
    n, d = p.shape
    r = unit_vector(p[:, None] - p, axis=-1)
    ii = np.eye(n, dtype=bool)
    r[ii] = 0
    r *= A[..., None]               # aplicar pesos
    E = np.argwhere(np.triu(A) != 0)
    Ef = np.flip(E, axis=1)
    ne = len(E)
    J = np.zeros((ne, n, d))
    i = np.arange(ne).reshape(-1, 1)
    J[i, E] = r[E, Ef]
    J = J.reshape(ne, n * d)
    return J


def matrix_from_edges(E, p, w=np.array([1.])):
    """Matriz de rigidez normalizada

    args:
        E: enlaces (m, 2)
        p: array de posiciones (n, d)

    returns
        R: (ne, n * d)
    """
    n, d = p.shape
    r = unit_vector(p[E[:, 0]] - p[E[:, 1]], axis=-1)
    r *= w.reshape(-1, 1)           # aplicar pesos
    ne = len(E)
    J = np.zeros((ne, n, d))
    i = np.arange(ne)
    J[i, E[:, 0]] = r[i]
    J[i, E[:, 1]] = -r[i]
    J = J.reshape(ne, n * d)
    return J


def classic_symmetric_matrix(A, p):
    """Matriz clasica de rigidez.

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


def fast_symmetric_matrix(A, p):
    L = _rigidity_laplacian(A, p)
    return L.swapaxes(1, 2).reshape(p.size, p.size)


def symmetric_matrix(A, p):
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


def complete_symmetric_matrix(p):
    n, d = p.shape[-2:]
    ii = np.eye(n, dtype=bool)
    r = unit_vector(p[..., None, :] - p[..., None, :, :], axis=-1)
    r[..., ii, :] = 0
    S = - r[..., None] * r[..., None, :]    # outer product
    S[..., ii, :, :] -= S.sum(p.ndim - 1)
    S = S.swapaxes(-3, -2)
    s = list(S.shape)
    s[-4:] = n * d, n * d
    return S.reshape(s)


def symmetric_matrix_diag(A, p):
    """Bloques diagonales del laplaciano de rigidez.

    args:
        A: matriz de adyacencia (..., n, n)
        p: array de posiciones (..., n, d)

    returns
        diag: bloques principales (..., n, d, d)
    """
    n, d = p.shape[-2:]
    ii = np.eye(n, dtype=bool)
    r = unit_vector(p[..., None, :] - p[..., None, :, :], axis=-1)
    r[..., ii, :] = 0
    S = r[..., None] * r[..., None, :]    # outer product
    S *= A[..., None, None]               # aplicar pesos
    diag = S.sum(p.ndim - 1)
    return diag


def local_symmetric_matrix(p, q, w=np.array(1.)):
    r = unit_vector(p[:, None] - q, axis=2)
    rw = r * w[..., None]
    S = r[..., None] * rw[..., None, :]
    Si = S.sum(1)
    return Si


def is_inf_rigid(A, p, threshold=THRESHOLD):
    d = p.shape[-1]
    f = int(d * (d + 1)/2)
    S = fast_symmetric_matrix(A, p)
    eig = np.linalg.eigvalsh(S)
    return eig[f] > threshold


def eigenvalue(A, p):
    d = p.shape[1]
    f = int(d * (d + 1)/2)
    S = fast_symmetric_matrix(A, p)
    eig = np.linalg.eigvalsh(S)
    return eig[f]


def trivial_motions(p):
    """Matriz cuyas columnas son una BON del espacio pose.

    args:
        p: array de posiciones (n, dof)

    returns
        M: matriz (n*dof, n*dof)
    """
    n = len(p)
    P = np.zeros((p.size, 3))
    r_cm = p - p.mean(0)

    P[::2, 0] = 1/np.sqrt(n)                    # dx
    P[1::2, 1] = 1/np.sqrt(n)                   # dy
    P[::2, 2] = -r_cm[:, 1]
    P[1::2, 2] = r_cm[:, 0]
    P[:, 2] /= np.sqrt(np.square(r_cm).sum())   # dt
    return P


def nontrivial_motions(p):
    T = trivial_motions(p)
    N = scipy.linalg.null_space(T.T)
    return N


def extents(A, p, threshold=THRESHOLD):
    if not is_inf_rigid(A, p, threshold):
        raise ValueError('Flexible Framework.')
    n = A.shape[0]
    d = p.shape[1]
    f = int(d * (d + 1)/2)
    hops = np.empty(n, dtype=int)
    for i in range(n):
        minimum_found = False
        h = 0
        while not minimum_found:
            h += 1
            Ai, xi = subsets.multihop_subframework(A, p, i, h)
            Si = symmetric_matrix(Ai, xi)
            re = np.linalg.eigvalsh(Si)[f]
            if re > threshold:
                minimum_found = True
        hops[i] = h
    return hops


def fast_extents(A, p, threshold=THRESHOLD):
    n, d = p.shape
    f = d * (d + 1) // 2
    geodesics = subsets.geodesics(A)
    hops = np.empty(n, dtype=int)
    for i in range(n):
        minimum_found = False
        h = 0
        while not minimum_found:
            h += 1
            subset = geodesics[i] <= h
            Ai = A[np.ix_(subset, subset)]
            pi = p[subset]
            Si = fast_symmetric_matrix(Ai, pi)
            re = np.linalg.eigvalsh(Si)[f]
            if re > threshold:
                minimum_found = True
        hops[i] = h
    return hops


def minimum_radius(A, p, threshold=THRESHOLD, return_radius=False):
    """Add or delete edges to a framework until it is radius-wise minimally
    rigid."""
    A = A.copy()
    n, d = p.shape
    f = d * (d + 1) // 2
    dist = distances.matrix(p)

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


def sparse_centers_full_search(
        A, p, extents, max_extent, metric, threshold=THRESHOLD):
    """Prueba todas las extensiones que no superen max(extents) y elige la de
    menor metric.

    Requires:
    ---------
        framework is rigid
    """
    n = len(p)
    geodesics = subsets.geodesics(A)
    search_space = itertools.product(
        *([0] + list(range(extents[i], max_extent + 1)) for i in range(n))
    )

    min_value = np.inf
    for h in list(search_space)[1:]:
        h = np.array(h)
        V = geodesics <= np.reshape(h, (-1, 1))
        C = V[h > 0]
        for i, c in enumerate(C):
            k = np.delete(C, i, axis=0)
            if np.all(~c + k, axis=1).any():
                continue
            Ai = A[c][:, c]
            pi = p[c]
            if not is_inf_rigid(Ai, pi, threshold):
                # not ridigly subgraph
                break
        new_value = metric(geodesics, h)
        if new_value < min_value:
            min_value = new_value
            h_opt = h

    return h_opt


def sparse_centers_binary_search(A, p, extents, metric, threshold=THRESHOLD):
    """Dada un conjunto de extensiones, elimina uno por uno los subframeworks
    alternadamente y elige la de menor metric.

    Requires:
    ---------
        framework is rigid
    """
    n = len(p)
    geodesics = subsets.geodesics(A)
    search_factors = itertools.product((0, 1), repeat=n)

    min_value = np.inf
    for f in list(search_factors)[1:]:
        h = np.multiply(extents, f)
        new_value = metric(geodesics, h)
        if new_value < min_value:
            min_value = new_value
            h_opt = h

    return h_opt


def sparse_centers_greedy_search(A, p, extents, metric, threshold=THRESHOLD):
    """Dada un conjunto de extensiones, elimina recursivamente el subframework
    de mayor extension, hasta que no puede eliminar mas.

    Requires:
    ---------
        framework is rigid
    """
    n = len(p)
    geodesics = subsets.geodesics(A)
    hops = extents.copy()
    remain = np.arange(n)
    terminate = False

    min_value = np.inf
    while not terminate:
        remove = None
        for i in remain:
            sparsed = hops.copy()
            sparsed[i] = 0
            new_value = metric(geodesics, sparsed)
            if new_value < min_value:
                min_value = new_value
                remove = i

        if remove is not None:
            hops[remove] = 0
            remain = np.delete(remain, remain == remove)
        else:
            terminate = True
    return hops
