#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar abr  6 20:52:31 -03 2021
"""
import numpy as np


class ConeGraph(object):
    def __init__(self, dmax, cmin=-1.0):
        """
        Class representing a cone-like state dependant graph.

        args:
        -----
            dmax : max conextion range
            cmin  : cosine of cone's half angle
        """
        self.dmax = np.reshape(dmax, (-1, 1))
        self.cmin = np.reshape(cmin, (-1, 1))
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

    def update_adjacency_matrix(self, positions, axes):
        """
        Cone graph adjacency matrix.

        args:
        -----
            positions : (n, d) vector array
            axes      : (n, d) unit vector array
        """
        r = positions - positions[:, np.newaxis]
        d = np.sqrt(np.square(r).sum(axis=-1))
        bearings = r / d[:, :, np.newaxis]
        cos = np.matmul(bearings, axes[:, :, np.newaxis]).squeeze()
        self._adj = np.logical_and(d <= self.dmax, cos >= self.cmin)
        self._adj[np.eye(len(positions), dtype=bool)] = False
        return self._adj.copy()


def adjacency_matrix(positions, axes, dmax=np.inf, cmin=-1.0):
    """
    Cone graph adjacency matrix.

    args:
    -----
        positions  : (n, d) vector array
        axes       : (n, d) unit vector array
        dmax       : max conextion range
        cmin       : cosine of cone's half angle
    """
    dmax = np.reshape(dmax, (-1, 1))
    r = positions - positions[:, np.newaxis]
    d = np.sqrt(np.square(r).sum(axis=-1))
    bearings = r / d[:, :, np.newaxis]
    cos = np.matmul(bearings, axes[:, :, np.newaxis]).squeeze()
    A = np.logical_and(d <= dmax, cos >= cmin)
    A[np.eye(len(positions), dtype=bool)] = 0.0
    return A
