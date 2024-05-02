#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date lun dic 14 15:37:42 -03 2020
"""
import numpy as np
from numba import njit


def complete_edges(n, directed=False):
    A = 1 - np.eye(n)
    if not directed:
        A = np.triu(A)
    return np.argwhere(A)


def complete_adjacency(n):
    A = 1 - np.eye(n)
    return A


def complete_laplacian(n):
    L = n * np.eye(n) - 1
    return L


def edges_from_adjacency(A, directed=False):
    """Devuelve array de enlaces."""
    if not directed:
        A = np.triu(A)
    E = np.argwhere(A > 0)
    return E


def adjacency_from_edges(n, E, w=1, directed=False):
    A = np.zeros((n, n))
    A[E[:, 0], E[:, 1]] = w
    if not directed:
        A[E[:, 1], E[:, 0]] = w
    return A


def laplacian_from_adjacency(A):
    n = A.shape[-1]
    ii = np.eye(n, dtype=bool)
    L = -A.copy()
    L[..., ii] += A.sum(axis=-1)
    return L


def algebraic_connectivity(A, return_value=True):
    L = laplacian_from_adjacency(A)
    a2 = np.linalg.eigvalsh(L)[1]
    if return_value:
        return a2
    return a2 > 1e-5


def degree(A):
    return A.sum(-1)


@njit
def geodesics(A):
    """Matrix of geodesic distances.

    Requires:
    ---------
        graph is connected
    """
    G = A.copy()
    As = np.eye(len(A)) + A
    h = 2
    while not np.all(As):
        Ah = np.linalg.matrix_power(A, h)
        for i, g in enumerate(G):
            idx = np.logical_and(Ah[i] > 0, As[i] == 0)
            g[idx] = h
        As += Ah
        h += 1
    return G


def diameter(geodesics):
    return np.max(geodesics)


def adjacency_from_geodesics(geodesics):
    A = geodesics.copy()
    A[A > 1] = 0
    return A
