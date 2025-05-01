#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar abr  6 20:52:31 -03 2021
"""
import numpy as np


class DiskGraph(object):
    def __init__(self, dmax):
        """
        Class representing a disk-like state dependant graph.

        args:
        -----
            dmax : max conextion range
        """
        self.dmax = np.reshape(dmax, (-1, 1))
        self._adj = None

    def adjacency_matrix(self):
        return self._adj.copy()

    def edge_set(self):
        return np.argwhere(self._adj)

    def is_edge(self, vertex_i, vertex_j):
        return self._adj[vertex_i, vertex_j]

    def share_edge(self, vertex_i, vertex_j):
        return self._adj[vertex_i, vertex_j] or self._adj[vertex_j, vertex_i]

    def out_neighbors(self, vertex):
        return np.where(self._adj[vertex])[0]

    def in_neighbors(self, vertex):
        return np.where(self._adj[:, vertex])[0]

    def update_adjacency_matrix(self, positions):
        """
        Disk graph adjacency matrix.

        args:
        -----
            positions : (n, d) vector array
        """
        r = positions - positions[:, np.newaxis]
        d = np.sqrt(np.square(r).sum(axis=-1))
        self._adj = d <= self.dmax
        self._adj[np.eye(len(positions), dtype=bool)] = False
        return self._adj.copy()


def edges_from_positions(p, dmax=np.inf):
    dmax = np.reshape(dmax, (-1, 1))
    r = p[:, np.newaxis] - p
    d2 = np.square(r).sum(axis=-1)
    A = d2 <= dmax**2
    A[np.eye(len(p), dtype=bool)] = 0.0
    return np.argwhere(A)


def adjacency_from_positions(p, dmax=np.inf):
    """
    Disk proximity matrix.

    args:
    -----
        p    : (n, d) state array
        dmax : maximum connectivity distance
    """
    r = p[:, None] - p
    A = np.square(r).sum(axis=-1)
    A[A > dmax**2] = 0
    A[A != 0] = 1
    return A
