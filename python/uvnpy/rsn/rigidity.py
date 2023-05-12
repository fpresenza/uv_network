#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mié ago 11 21:15:33 -03 2021
"""
import numpy as np
import scipy.linalg
from transformations import unit_vector

from uvnpy.network.core import geodesics
from uvnpy.network.subsets import multihop_subframework
from uvnpy.rsn.distances import matrix as distance_matrix


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


def algebraic_condition(A, p, threshold=1e-4):
    d = p.shape[-1]
    f = int(d * (d + 1)/2)
    S = symmetric_matrix(A, p)
    eig = np.linalg.eigvalsh(S)
    return eig[f] > threshold


def eigenvalue(A, p):
    d = p.shape[1]
    f = int(d * (d + 1)/2)
    S = symmetric_matrix(A, p)
    eig = np.linalg.eigvalsh(S)
    return eig[f]


def subframework_eigenvalue(A, p, i, h=1):
    d = p.shape[1]
    f = int(d * (d + 1)/2)
    Ai, xi = multihop_subframework(A, p, i, h)
    Si = symmetric_matrix(Ai, xi)
    eig = np.linalg.eigvalsh(Si)
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


def extents(A, p, threshold=1e-4):
    if not algebraic_condition(A, p, threshold):
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
            Ai, xi = multihop_subframework(A, p, i, h)
            Si = symmetric_matrix(Ai, xi)
            re = np.linalg.eigvalsh(Si)[f]
            if re > threshold:
                minimum_found = True
        hops[i] = h
    return hops


def minimum_radius(A, p, threshold=1e-4, return_radius=False):
    """Add or delete edges to a framework until it is radius-wise minimally
    rigid."""
    A = A.copy()
    n, d = p.shape
    f = d * (d + 1) // 2
    dist = distance_matrix(p)

    if algebraic_condition(A, p, threshold):
        B = A.copy()
        dist = dist * B

        rigid = True
        while rigid:
            A = B.copy()
            i, j = np.unravel_index(np.argmax(dist), (n, n))
            radius = dist[i, j]
            dist[i, j] = dist[j, i] = 0
            B[i, j] = B[j, i] = 0
            rigid = algebraic_condition(B, p, threshold)
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
                rigid = algebraic_condition(A, p, threshold)

    if return_radius:
        return A, radius
    else:
        return A


def subframework_based_rigidity(A, p, extents, threshold=1e-4):
    """Determines the sufficient condition for framework rigidity based on the
    rigidity of the subframeworks"""
    n, d = p.shape
    geo = geodesics(A)
    centers = np.where(extents > 0)[0]

    # check every vertex is covered
    covered = geo[centers] <= extents[centers].reshape(-1, 1)
    is_in = np.sum(covered, axis=0)
    if np.any(is_in == 0):
        raise ValueError('Decomposition don\'t cover all vertices.')

    # check separately if there is only one subframework
    if len(centers) == 1:
        return algebraic_condition(A, p, threshold), None

    # check if every subframework is rigid and
    # if each one has at least d binding vertices
    Ab = np.zeros((n, n))
    for i in centers:
        Vi = geo[i] <= extents[i]
        Ai = A[Vi][:, Vi]
        xi = p[Vi]
        if not algebraic_condition(Ai, xi, threshold):
            raise ValueError('Subframework {} is flexible.'.format(i))

        Bi = np.argwhere(np.logical_and(is_in > 1, Vi))
        if len(Bi) < d:
            raise ValueError(
                'Subframework {} hasn\'t got enough binding nodes.'.format(i))

        Ab[Bi.ravel(), Bi] = 1 - np.eye(len(Bi))

    # check if the binding graph is rigid
    B = is_in > 1
    if not algebraic_condition(Ab[B][:, B], p[B], threshold):
        raise ValueError('Binding framework is flexible.')

    return True, Ab


def sparse_centers(A, p, extents, metric, threshold=1e-4):
    """Dada un conjunto de extensiones, elimina subframeworks iterativamente
    hasta que no puede eliminar mas dado que se pierde rigidez
    """
    hops = extents.copy()
    for h in reversed(np.unique(hops)):
        max_extent = np.where(hops == h)[0]
        repeat = True
        while repeat:
            min_load = np.inf
            remove = None
            for i in max_extent:
                sparsed = hops.copy()
                sparsed[i] = 0
                try:
                    subframework_based_rigidity(A, p, sparsed, threshold)
                    new_load = metric(A, sparsed)
                    if new_load < min_load:
                        min_load = new_load
                        remove = i
                except ValueError:
                    pass

            if min_load < np.inf:
                hops[remove] = 0
                max_extent = np.delete(max_extent, max_extent == remove)
            else:
                repeat = False
    return hops
