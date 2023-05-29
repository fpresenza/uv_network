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

from uvnpy.network.subsets import (
    geodesics as fast_geodesics,
    multihop_subframework,
    multihop_subsets)
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


@njit
def _rigidity_laplacian(A, p):
    n, d = p.shape
    L = np.zeros((n, n, d, d))
    edges = np.argwhere(np.triu(A) > 0)
    for i, j in edges:
        dij = p[i] - p[j]
        nij = np.square(dij).sum()
        L[i, j] = L[j, i] = dij.reshape(2, 1).dot(dij.reshape(1, 2)) / -nij
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


def algebraic_condition(A, p, threshold=1e-4):
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


def fast_extents(A, p, threshold=1e-4):
    n, d = p.shape
    f = d * (d + 1) // 2
    geodesics = fast_geodesics(A)
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


def rigidly_linked(A, p, extents, threshold=1e-4):
    """Determines whether a set of subframeworks cover the whole framework and
    is rigidly linked

    Returns:
        Adjacency matrix of the link graph

    Raise:
        ValueError: whenever condition is not satisfied
    """
    n, d = p.shape
    centers = np.nonzero(extents)[0]
    subsets = multihop_subsets(A, centers, extents[centers])

    # check if union of subgraphs cover all vertices
    is_in_count = np.sum(subsets, axis=0)
    if np.any(is_in_count == 0):
        raise ValueError('Node(s) not covered.')

    # check if a single subgraph cover all vertices
    if len(centers) == 1:
        return 1 - np.eye(n)

    # computes the union between subgraphs
    Au = np.zeros((n, n))
    for subset in subsets:
        idx = np.ix_(subset, subset)
        Au[idx] = A[idx].copy()

    # link graph
    Al = A - Au
    # detect link nodes
    shared_nodes = is_in_count > 1
    link_edges = np.any(Al, axis=0)
    link_nodes = np.logical_or(shared_nodes, link_edges)
    for subset in subsets:
        link_subset = np.logical_and(link_nodes, subset)
        num_link = sum(link_subset)
        if num_link < d:
            raise ValueError('Subframework(s) hasn\'t got enough link nodes.')
        Al[np.ix_(link_subset, link_subset)] = 1 - np.eye(num_link)

    # check if the link graph is rigid
    if not algebraic_condition(
            Al[np.ix_(link_nodes, link_nodes)], p[link_nodes], threshold):
        raise ValueError('Link framework is flexible.')

    return Al


def rigidly_linked_by_vertices(A, p, extents, threshold=1e-4):
    """Determines whether a set of subframeworks cover the whole framework and
    is rigidly linked

    Returns:
        Adjacency matrix of the link graph

    Raise:
        ValueError: whenever condition is not satisfied
    """
    n, d = p.shape
    centers = np.nonzero(extents)[0]
    subsets = multihop_subsets(A, centers, extents[centers])

    # check if union of subgraphs cover all vertices
    is_in_count = np.sum(subsets, axis=0)
    if np.any(is_in_count == 0):
        raise ValueError('Node(s) not covered.')

    # check if a single subgraph cover all vertices
    if len(centers) == 1:
        return 1 - np.eye(n)

    # link graph
    Al = np.zeros((n, n))
    # detect link nodes
    link_nodes = is_in_count > 1
    for subset in subsets:
        link_subset = np.logical_and(link_nodes, subset)
        num_link = sum(link_subset)
        if num_link < d:
            raise ValueError('Subframework(s) hasn\'t got enough link nodes.')
        Al[np.ix_(link_subset, link_subset)] = 1 - np.eye(num_link)

    # check if the link graph is rigid
    if not algebraic_condition(
            Al[np.ix_(link_nodes, link_nodes)], p[link_nodes], threshold):
        raise ValueError('Link framework is flexible.')

    return Al


def sparse_centers_full_search(
        A, p, extents, metric, threshold=1e-4, vertices_only=False):
    """Dada un conjunto de extensiones, elimina subframeworks iterativamente
    hasta que no puede eliminar mas dado que se pierde rigidez o no se cubre
    todo el framework.

    Requires:
        framework is rigid
    """
    if vertices_only:
        is_linked = rigidly_linked_by_vertices
    else:
        is_linked = rigidly_linked

    n = len(p)
    geodesics = fast_geodesics(A)
    degree = A.sum(axis=0)
    max_hops = extents.max()
    search_space = itertools.product(range(max_hops + 1), repeat=n)

    min_value = np.inf
    for h in search_space:
        h = np.array(h)
        if np.any(np.logical_and(h > 0, h < extents)):
            continue
        try:
            is_linked(A, p, h, threshold)
            for i in np.nonzero(h)[0]:
                Vi = geodesics[i] <= h[i]
                Ai = A[Vi][:, Vi]
                pi = p[Vi]
                if not algebraic_condition(Ai, pi, threshold):
                    raise ValueError
            new_value = metric(degree, h, geodesics)
            if new_value < min_value:
                min_value = new_value
                h_opt = h
        except ValueError:
            pass

    return h_opt


def sparse_centers_binary_search(
        A, p, extents, metric, threshold=1e-4, vertices_only=False):
    """Dada un conjunto de extensiones, elimina subframeworks iterativamente
    hasta que no puede eliminar mas dado que se pierde rigidez o no se cubre
    todo el framework.

    Requires:
        framework is rigid
    """
    if vertices_only:
        is_linked = rigidly_linked_by_vertices
    else:
        is_linked = rigidly_linked

    n = len(p)
    geodesics = fast_geodesics(A)
    degree = A.sum(axis=0)
    search_factors = itertools.product((0, 1), repeat=n)

    min_value = np.inf
    for f in search_factors:
        h = np.multiply(extents, f)
        try:
            is_linked(A, p, h, threshold)
            new_value = metric(degree, h, geodesics)
            if new_value < min_value:
                min_value = new_value
                h_opt = h
        except ValueError:
            pass

    return h_opt


def sparse_centers(A, p, extents, metric, threshold=1e-4, vertices_only=False):
    """Dada un conjunto de extensiones, elimina subframeworks iterativamente
    hasta que no puede eliminar mas dado que se pierde rigidez o no se cubre
    todo el framework.

    Requires:
        framework is rigid
    """
    if vertices_only:
        is_linked = rigidly_linked_by_vertices
    else:
        is_linked = rigidly_linked

    n = len(p)
    geodesics = fast_geodesics(A)
    degree = A.sum(axis=0)
    hops = extents.copy()
    centers = np.arange(n)
    min_found = False
    while not min_found:
        min_value = np.inf
        remove = None
        for i in centers:
            sparsed = hops.copy()
            sparsed[i] = 0
            try:
                is_linked(A, p, sparsed, threshold)
                new_value = metric(degree, sparsed, geodesics)
                if new_value < min_value:
                    min_value = new_value
                    remove = i
            except ValueError:
                pass

        if min_value < np.inf:
            hops[remove] = 0
            centers = np.delete(centers, centers == remove)
        else:
            min_found = True
    return hops


def sparse_centers_two_steps(
        A, p, extents, metric, threshold=1e-4, vertices_only=False):
    """Dada un conjunto de extensiones, elimina subframeworks iterativamente
    hasta que no puede eliminar mas dado que se pierde rigidez o no se cubre
    todo el framework.

    Requires:
        framework is rigid
    """
    if vertices_only:
        is_linked = rigidly_linked_by_vertices
    else:
        is_linked = rigidly_linked

    geodesics = fast_geodesics(A)
    degree = A.sum(axis=0)
    hops = extents.copy()
    for h in reversed(np.unique(hops)):
        max_extent = np.where(hops == h)[0]
        min_found = False
        while not min_found:
            min_value = np.inf
            remove = None
            for i in max_extent:
                sparsed = hops.copy()
                sparsed[i] = 0
                try:
                    is_linked(A, p, sparsed, threshold)
                    new_value = metric(degree, sparsed, geodesics)
                    if new_value < min_value:
                        min_value = new_value
                        remove = i
                except ValueError:
                    pass

            if min_value < np.inf:
                hops[remove] = 0
                max_extent = np.delete(max_extent, max_extent == remove)
            else:
                min_found = True
    return hops
