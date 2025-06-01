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


class ErdosRenyi(Graph):
    """
    Class representing a random graph based on the Erdos-Renyi model.
    """
    def __init__(self, n, p, undirected=False):
        """
        args:
        -----
            n : number of nodes
            p : probability of existence of each edge
        """
        self.n = n
        if undirected:
            self.p = 1 - np.sqrt(1 - p)
        else:
            self.p = p
        self.update()

    def update(self):
        adj = np.random.choice(
            [False, True], size=(self.n, self.n), p=(1 - self.p, self.p)
        )
        adj[np.eye(self.n, dtype=bool)] = False
        super().update(adj)


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

    def remove_vertex(self, vertex):
        super().remove_vertex(vertex)
        self._real = np.delete(self._real, vertex, axis=0)

    def append_vertex(self, out_edges, in_edges, realization):
        super().append_vertex(out_edges, in_edges)
        self._real = np.vstack([self._real, realization])


class DiskGraph(Framework):
    """
    Class representing a disk-like state dependant graph.
    """
    def __init__(self, realization=None, dmax=np.inf):
        """
        args:
        -----
            realization : disk centers in some realization space
            dmax : max connection distance
        """
        self.dmax = dmax
        if realization is not None:
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
        super().update(adj, realization)

    def append_vertex(self, realization):
        r = self._real - realization
        d = np.sqrt(np.square(r).sum(axis=-1))
        edges = d <= self.dmax
        super().append_vertex(edges, edges, realization)


class ConeGraph(Framework):
    """
    Class representing a cone-like state dependant graph.
    """
    def __init__(self, positions=None, axes=None, dmax=np.inf, cmin=-1.0):
        """
        args:
        -----
            position : (n, d) cone apexes in R^d
            axes : (n, d) cone axes in R^d)
            dmax : max connection distance
            cmin : cosine of cone's half angle
        """
        self.dmax = dmax
        self.cmin = cmin
        if (positions is not None) and (axes is not None):
            self.update(positions, axes)

    def positions(self, d):
        return self._real[:, :d].copy()

    def axes(self, d):
        return self._real[:, d:].copy()

    def update(self, positions, axes):
        """
        Cone graph adjacency matrix.

        args:
        -----
            positions : (n, d) vector array
            axes : (n, d) unit vector array
        """
        r = positions - positions[:, np.newaxis]
        d = np.sqrt(np.square(r).sum(axis=-1))
        bearings = r / d[:, :, np.newaxis]
        cos = np.matmul(bearings, axes[:, :, np.newaxis]).squeeze()
        adj = np.logical_and(d <= self.dmax, cos >= self.cmin)
        adj[np.eye(len(positions), dtype=bool)] = False
        realization = np.hstack([positions, axes])
        super().update(adj, realization)

    def append_vertex(self, position, axis):
        s = position.size
        r = position - self._real[:, :s]
        d = np.sqrt(np.square(r).sum(axis=-1))
        in_bearings = r / d[:, np.newaxis]
        in_cos = np.sum(in_bearings * self._real[:, s:], axis=-1)
        out_cos = - np.sum(in_bearings * axis, axis=-1)
        in_ball = d <= self.dmax

        in_edges = np.logical_and(in_ball, in_cos >= self.cmin)
        out_edges = np.logical_and(in_ball, out_cos >= self.cmin)
        realization = np.hstack([position, axis])
        super().append_vertex(out_edges, in_edges, realization)
