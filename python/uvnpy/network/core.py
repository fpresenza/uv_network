#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date lun dic 14 15:37:42 -03 2020
"""
import numpy as np
import itertools


def undirected_edges(E):
    """Devuelve enlaces en un solo sentido."""
    Eu = np.unique(np.sort(E), axis=0)
    return Eu


def directed_edges(E):
    """Devuelve enlaces en ambos sentidos."""
    m = len(E)
    Ed = np.empty((2 * m, 2), dtype=np.int)
    Ed[:m] = E
    Ed[m:] = np.flip(E, axis=-1)
    return Ed


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


def complete_incidence(n):
    edges = itertools.combinations(range(n), 2)
    E = np.array(list(edges))
    m = len(E)
    D = np.zeros((n, m))
    e = range(m)
    D[E[:, 1], e] = -1
    D[E[:, 0], e] = 1
    return D


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


def incidence_from_edges(n, E):
    D = np.zeros((n, len(E)))
    e = range(len(E))
    D[E[:, 1], e] = -1
    D[E[:, 0], e] = 1
    return D


def laplacian_from_edges(n, E, w=1, directed=False):
    A = adjacency_from_edges(n, E, w, directed)
    Deg = np.diag(A.sum(axis=1))
    return Deg - A


def laplacian_from_adjacency(A):
    n = A.shape[-1]
    ii = np.eye(n, dtype=bool)
    L = -A.copy()
    L[..., ii] += A.sum(axis=-1)
    return L


def algebraic_connectivity(L):
    a2 = np.linalg.eigbalsh(L)[1]
    return a2 > 1e-5
