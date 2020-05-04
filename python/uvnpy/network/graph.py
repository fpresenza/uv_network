#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 12:41:07 2020
@author: fran
"""
import numpy as np
import networkx as nx

# MatrixTuple = recordclass.recordclass('MatrixTuple', 'D A W L',\
#     defaults=(np.empty(0), np.empty(0), np.empty(0), np.empty(0)))

# class Graph(object):
#     def __init__(self, *args, **kwargs):
#         self.V = kwargs.get('V', [1])
#         self.connect = kwargs.get('connect', lambda i,j: 0)
#         self.E = kwargs.get('E', [(i, j) for i in self.V for j in self.V if i!=j and self.connect(i,j)])
#         self.w = kwargs.get('w', [1 if kwargs.get('E') else self.connect(i,j) for (i,j) in self.E])
#         self.N = dict(zip(self.V, map(self.neighborhood, self.V)))
#         self.d = dict(zip(self.V, map(lambda v: len(self.N[v]), self.V)))

#         self.mat = MatrixTuple()
#         # self.mat.D = np.diag(self.d)
#         self.mat.A = np.array([[self.w[self.E.index((i,j))] if (i,j) in self.E else 0 for j in self.V] for i in self.V])
#         # self.mat.W = np.diag(self.w)
#         self.size = len(self.V)

#     # def degree(self, v, **kwargs):
#     #     weighted = kwargs.get('weighted', False)
#     #     return sum(map(lambda e: 1 if e[0]==v else 0, self.E)) 

#     def neighborhood(self, v, **kwargs):
#         e = list(filter(lambda e: 1 if e[0]==v else 0, self.E))
#         return list(map(lambda arg: arg[1], e))

#     def update(self, *args, **kwargs):
#         self.__init__(*args, **kwargs)

#     # def __call__(self):
#     #     return self.neighbors

#     # def __len__(self):
#     #     return len(self.neighbors)

#     # def incomplete(self):
#     #     return len(self.neighbors) < self.size

#     # def append(self, id):
#     #     self.neighbors.append(id)

# class Random(Graph):
#     def __init__(self, *args, **kwargs):
#         V = kwargs.get('V', [1])
#         w = dict([[(i,j), np.random.choice([1, 0])*np.random.randint(10)] for i in V for j in V])
#         kwargs['connect'] = lambda i,j: w[i,j]
#         super(Random, self).__init__(*args, **kwargs)


# class Complete(Graph):
#     def __init__(self, *args, **kwargs):
#         kwargs['connect'] = lambda i,j: 1
#         super(Complete, self).__init__(*args, **kwargs)


# class Path(Graph):
#     def __init__(self, *args, **kwargs):
#         kwargs['connect'] = lambda i,j: 1 if j==i+1 else 0
#         super(Path, self).__init__(*args, **kwargs)


# class Cycle(Graph):
#     def __init__(self, *args, **kwargs):
#         kwargs['connect'] = lambda i,j: 1 if abs(i-j)== 1%len(self.V) else 0
#         super(Cycle, self).__init__(*args, **kwargs)

class Graph(nx.Graph):
    def __init__(self, *args, **kwargs):
        super(Graph, self).__init__(*args, **kwargs)

    def __str__(self):
        return 'Graph instance:\nnodes: {}\nedges: {}'.format(self.nodes, self.edges)

    def __repr__(self):
        return 'Graph()'

    def get_nodes(self, *ids):
        if ids:
            subset = list(filter(lambda n: n.id in ids, self.nodes))
        else:
            subset = list(self.nodes)
        subset.sort(key=lambda n: n.id)
        return subset