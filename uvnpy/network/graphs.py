#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
"""
import numpy as np


class Graph(object):
    """
    Class representing a finite directed graph
    parametrized by a boolean adjacency matrix.
    """
    def __init__(self, adjacency_matrix=None):
        if adjacency_matrix is None:
            self._adj = None
        else:
            self._adj = adjacency_matrix.astype(bool)

    def vertex_set(self):
        return np.arange(len(self._adj))

    def adjacency_matrix(self):
        return self._adj.copy()

    def adjacency_list(self):
        return [np.where(adj)[0] for adj in self._adj]

    def edge_set(self):
        return np.argwhere(self._adj)

    def is_edge(self, vertex_i, vertex_j):
        return self._adj[vertex_i, vertex_j]

    def out_neighbors(self, vertex):
        return np.where(self._adj[vertex])[0]

    def in_neighbors(self, vertex):
        return np.where(self._adj[:, vertex])[0]

    def update(self, adjacency_matrix):
        """
        Update adjacency matrix.
        """
        self._adj = adjacency_matrix.astype(bool)

    def remove_vertex(self, vertex):
        self._adj = np.delete(
            np.delete(self._adj, vertex, axis=0), vertex, axis=1
        )

    def append_vertex(self, out_edges, in_edges):
        n = len(self._adj)
        adj = np.empty((n + 1, n + 1), dtype=bool)
        adj[:n, :n] = self._adj
        adj[n, :n] = out_edges
        adj[:n, n] = in_edges
        adj[n, n] = False
        self._adj = adj


class Framework(Graph):
    """
    Class representing a framework consisting of
    a finite directed graph parametrized by a
    boolean adjacency matrix and an array of states.
    """
    def __init__(self, adjacency_matrix=None, realization=None):
        """
        args:
        -----
            realization : (n, d) array
        """
        super().__init__(adjacency_matrix)
        if realization is None:
            self._real = None
        else:
            self._real = realization.astype(float)

    def realization(self):
        return self._real.copy()

    def update(self, adjacency_matrix, realization):
        super().update(adjacency_matrix)
        self._real = realization.astype(float)

    def append_vertex(self, out_edges, in_edges, realization):
        super().append_vertex(out_edges, in_edges)
        self._real = np.vstack([self.realization, realization])


class DiskGraph(Graph):
    """
    Class representing a disk-like state dependant graph.
    """
    def __init__(self, realization=None, dmax=np.inf):
        """
        args:
        -----
            realization : vertex positions in R^d
            dmax : max connection range
        """
        self.dmax = dmax
        if realization is None:
            self._adj = None
        else:
            self.update(realization)

    def update(self, realization):
        """
        Disk graph adjacency matrix.

        args:
        -----
            realization : (n, d) vector array
        """
        r = realization - realization[:, np.newaxis]
        d = np.sqrt(np.square(r).sum(axis=-1))
        adj = d <= self.dmax
        adj[np.eye(len(realization), dtype=bool)] = False
        super().update(adj)


class DiskFramework(DiskGraph):
    """
    Class representing a disk-like state dependant framework.
    """
    def __init__(self, position=None, dmax=np.inf):
        """
        args:
        -----
            position : disk centers in R^d
            dmax : max conextion range
        """
        super().__init__(position, dmax)
        if position is None:
            self._pos = None
        else:
            self._pos = position.astype(float)

    def position(self):
        return self._pos.copy()

    def update(self, position):
        super().update(position)
        self._pos = position

    def append_vertex(self, position):
        r = self._pos - position
        d = np.sqrt(np.square(r).sum(axis=-1))
        edges = d <= self.dmax
        super().append_vertex(edges, edges)
        self._pos = np.vstack([self._pos, position])


class ConeGraph(Graph):
    """
    Class representing a cone-like state dependant framework.
    """
    def __init__(self, positions=None, axes=None, dmax=np.inf, cmin=-1.0):
        """
        args:
        -----
            positions : cone apexes in R^d
            axes : cone axes in R^d
            dmax : max conextion range
            cmin : cosine of cone's half angle
        """
        self.dmax = dmax
        self.cmin = cmin
        if (positions is None) or (axes is None):
            self._adj = None
        else:
            self.update(positions, axes)

    def update(self, positions, axes):
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
        adj = np.logical_and(d <= self.dmax, cos >= self.cmin)
        adj[np.eye(len(positions), dtype=bool)] = False
        super().update(adj)


class ConeFramework(ConeGraph):
    """
    Class representing a cone-like state dependant framework.
    """
    def __init__(self, positions, axes, dmax=np.inf, cmin=-1.0):
        """
        args:
        -----
            positions : cone apexes in R^d
            axes : cone axes in R^d
            dmax : max conextion range
            cmin : cosine of cone's half angle
        """
        super().__init__(positions, axes, dmax, cmin)
        self._pos = positions
        self._axe = axes

    def position(self):
        return self._pos.copy()

    def axes(self):
        return self._axe.copy()

    def update(self, positions, axes):
        super().update(positions, axes)
        self.pos = positions
        self.axe = axes

    def append_vertex(self, position, axe):
        r = position - self._pos
        d = np.sqrt(np.square(r).sum(axis=-1))
        in_bearings = r / d[:, np.newaxis]
        in_cos = np.sum(in_bearings * self._axe, axis=-1)
        out_cos = - np.sum(in_bearings * axe, axis=-1)
        in_ball = d <= self.dmax

        in_edges = np.logical_and(in_ball, in_cos >= self.cmin)
        out_edges = np.logical_and(in_ball, out_cos >= self.cmin)
        super().append_vertex(out_edges, in_edges)

        self._pos = np.vstack([self._pos, position])
        self._axe = np.vstack([self._axe, axe])
