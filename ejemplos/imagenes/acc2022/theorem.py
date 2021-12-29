#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Ca2ated on dom ago 15 20:00:37 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import uvnpy.network as network

N = 50000
a2 = np.array([])
E = np.array([])

for i in range(N):
    # np.random.seed(i)
    n = np.random.randint(2, 100)
    ij = np.triu((1 - np.eye(n))).astype(bool)
    a = np.random.choice([0, 1], int(n*(n-1)/2))
    A = np.zeros((n, n))
    A[ij] = a
    A = A + A.T

    L = network.laplacian_from_adjacency(A)
    lambda2 = np.linalg.eigvalsh(L)[1]
    if lambda2 > 1e-6:
        G = nx.from_numpy_matrix(A)
        D = nx.algorithms.distance_measures.diameter(G)
        m = int(A.sum()/2)
        e = 2*m/(D**2)
        if lambda2 > e:
            print(i)
        a2 = np.append(a2, lambda2)
        E = np.append(E, e)

print('Number of graphs = {}'.format(len(a2)))

fig, ax = plt.subplots()
trial = np.arange(len(a2))
sorted = np.argsort(a2)
ax.plot(a2[sorted], label=r'$a_2$')
ax.plot(E[sorted], label=r'$E$')
ax.legend()
plt.show()
