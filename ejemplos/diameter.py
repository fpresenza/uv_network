#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on lun jul  5 20:19:35 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt

import uvnpy.network as network
from uvnpy.network import disk_graph
from uvnpy.rsn import rigidity

n = 30
x = np.empty((n, 2), dtype=float)
x[::2, 0] = np.arange(int(n/2))
x[::2, 1] = 0
x[1::2, 0] = 1/2 + np.arange(int(n/2))
x[1::2, 1] = np.sqrt(3)/2

dmax = 1.1

# dmax = 10.
# np.random.seed(0)
# x = np.random.uniform(-15, 15, (n, 2))
# E = disk_graph.edges(x, dmax)
# A = network.adjacency_from_edges(n, E)
# print(rigidity.algebraic_condition(A, x))
# fig, ax = network.plot.figure()
# network.plot.graph(ax, x, E)
# plt.show()


p = x[:10]
q = x[10:]
Ep = disk_graph.edges(p, dmax)
Eq = disk_graph.edges(q, dmax)
Epq = np.array([
    [8, 10],
    [9, 10],
    [9, 11]])


fig, ax = plt.subplots(2, 1, figsize=(6, 3))
fig.subplots_adjust(bottom=0.14)

ax[0].tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False)   # labels along the bottom edge are off
ax[0].set_aspect('equal')
ax[0].grid(1)
ax[0].set_ylim(-1, 2)

network.plot.nodes(ax[0], p, color='b')
network.plot.edges(ax[0], p, Ep, color='k', alpha=0.6)
network.plot.nodes(ax[0], q, color='b', alpha=0.3)
network.plot.edges(ax[0], q, Eq, color='k', alpha=0.3)
network.plot.edges(ax[0], x, Epq, color='k', alpha=0.3)
ax[0].text(x[2, 0] - 0.6, x[2, 1] - 0.55, r'$n=3$')
ax[0].text(x[9, 0] - 0.6, x[9, 1] + 0.25, r'$n=10$')
ax[0].text(x[n - 1, 0] - 0.6, x[n - 1, 1] + 0.25, r'$n={}$'.format(n))


a2 = np.empty(n - 2)
a3 = np.empty(n - 2)
l4 = np.empty(n - 2)

for i in range(3, n + 1):
    p = x[:i]
    A = disk_graph.adjacency(p, dmax)
    E = network.edges_from_adjacency(A)
    D = network.incidence_from_edges(i, E)

    Lc = network.laplacian_from_adjacency(A)
    a2[i - 3] = np.linalg.eigvalsh(Lc)[1]
    a3[i - 3] = np.linalg.eigvalsh(Lc)[2]

    # R = rigidity.classic_matrix(D, p * 2)
    # Lr = R.T.dot(R)
    Lr = rigidity.laplacian(A, p)
    l4[i - 3] = np.linalg.eigvalsh(Lr)[3]

ax[1].set_xlabel('n')
ax[1].grid(1)
ax[1].plot(np.arange(3, n + 1), a2, label=r'$a_2$')
ax[1].plot(np.arange(3, n + 1), a3, label=r'$a_3$')
ax[1].plot(np.arange(3, n + 1), l4, label=r'$\lambda_4$')
ax[1].legend()
fig.savefig('/tmp/diameter.pdf', format='pdf')

plt.show()
