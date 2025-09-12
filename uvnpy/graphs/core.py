#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date lun dic 14 15:37:42 -03 2020
"""
import numpy as np
from numba import njit
import networkx as nx


def adjacency_list(adjacency_matrix):
    """Transforms adjacency matrix to adjacency list

    Parameters
    ----------
    adjacency_matrix : numpy.ndarray

    Returns
    -------
    L : dict
        Adjacency list
    """
    return [np.nonzero(adj)[0] for adj in adjacency_matrix]


def geodesics_dict(adjacency_matrix):
    return dict(
        nx.shortest_path_length(
            nx.from_numpy_array(adjacency_matrix)
        )
    )


@njit
def geodesics(adjacency_matrix):
    """Matrix of geodesic distances.

    Requires:
    ---------
        graph is connected
    """
    A = adjacency_matrix
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


def as_undirected(adjacency_matrix):
    """Removes the direction of the edges"""
    return np.logical_or(adjacency_matrix, adjacency_matrix.swapaxes(-2, -1))


def as_oriented(adjacency_matrix):
    return np.triu(as_undirected(adjacency_matrix))


def complete_edges(n, directed=True):
    m = int(n*(n-1) // 2)
    edge_set = np.empty((m, 2), dtype=int)
    i, j = np.triu_indices(n, k=1)
    edge_set[:, 0] = i
    edge_set[:, 1] = j
    if directed:
        edge_set = np.vstack([edge_set, np.flip(edge_set, axis=1)])
    return edge_set


def edge_set_diff(E, F):
    return np.array([e for e in E if np.all(np.any(e != F, axis=1))])


def complete_adjacency(n):
    A = 1 - np.eye(n)
    return A.astype(bool)


def complete_laplacian(n):
    L = n * np.eye(n) - 1
    return L


def edges_from_adjacency(adjacency_matrix):
    return np.argwhere(adjacency_matrix)


def incidence_from_edges(n, E):
    D = np.zeros((n, len(E)))
    e = range(len(E))
    D[E[:, 1], e] = -1
    D[E[:, 0], e] = 1
    return D


def adjacency_matrix_from_edges(n, edge_set):
    A = np.zeros((n, n), dtype=bool)
    A[edge_set[:, 0], edge_set[:, 1]] = True
    return A


def adjacency_list_from_edges(vertex_set, edge_set, directed=True):
    if not directed:
        return [
            np.concatenate([
                edge_set[edge_set[:, 0] == i, 1],
                edge_set[edge_set[:, 1] == i, 0]
            ])
            for i in vertex_set
        ]
    else:
        return [edge_set[edge_set[:, 0] == i, 1] for i in vertex_set]


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


def diameter(geodesics):
    return np.max(geodesics)


def adjacency_from_geodesics(geodesics):
    A = np.full(geodesics.shape, False)
    A[geodesics == 1] = True
    return A
