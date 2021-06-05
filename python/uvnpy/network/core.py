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


def complete_undirected_edges(V):
    edges = itertools.combinations(V, 2)
    return np.array(list(edges))


def complete_directed_edges(V):
    edges = itertools.permutations(V, 2)
    return np.array(list(edges))


def complete_adjacency(V):
    n = len(V)
    A = 1 - np.eye(n)
    return A


def edges_from_adjacency(A):
    """Devuelve array de enlaces."""
    E = np.argwhere(A > 0)
    return E


def incidence_from_edges(V, E):
    D = np.zeros((len(V), len(E)))
    e = range(len(E))
    D[E[:, 1], e] = -1
    D[E[:, 0], e] = 1
    return D


def adjacency_from_edges(V, E, w=1):
    n = len(V)
    A = np.zeros((n, n))
    A[E[:, 0], E[:, 1]] = w
    return A


def undirected_adjacency_from_edges(V, E, w=1):
    n = len(V)
    A = np.zeros((n, n))
    A[E[:, 0], E[:, 1]] = w
    A[E[:, 1], E[:, 0]] = w
    return A


def laplacian_from_edges(V, E, w=1):
    A = adjacency_from_edges(V, E, w)
    Deg = np.diag(A.sum(axis=1))
    return Deg - A


def undirected_laplacian_from_edges(V, E, w=1):
    A = undirected_adjacency_from_edges(V, E, w)
    Deg = np.diag(A.sum(axis=1))
    return Deg - A


def remove_one_edge_adjacency(A):
    E = np.argwhere(np.triu(A) > 0)
    m = len(E)
    Am = np.tile(A, (m, 1, 1))
    for e, (i, j) in enumerate(E):
        Am[e, i, j] = Am[e, j, i] = 0
    return Am
