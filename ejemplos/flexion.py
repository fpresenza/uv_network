#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on dom ago 15 19:58:44 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt

import uvnpy.network as network
from uvnpy.network import disk_graph
from uvnpy.rsn import rigidity


def filter_edges(E):
    dels = []
    for k, e in enumerate(E):
        if (e % 2 == 0).all():
            dels.append(k)
    return np.delete(E, dels, axis=0)


fig, ax = plt.subplots(1, 2, figsize=(4, 2.5))

k = 10
mid = np.cumsum(4 * np.arange(50))
print(mid[k])
t = np.arange(-k, k+1, 1)
X, Y = np.meshgrid(t, t)
x = np.dstack([X, Y]).reshape(-1, 2)
print(x[mid[k]])
print('n = {}'.format(len(x)))

for _ax in ax:
    _ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False)   # labels along the bottom edge are off
    _ax.set_aspect('equal')
    _ax.grid(0)
    _ax.set_xlim(-k-1, k+1)
    _ax.set_ylim(-k-1, k+1)


dmax = 1.5
E = disk_graph.edges(x, dmax)
E = filter_edges(E)
A = network.adjacency_from_edges(len(x), E)
L = rigidity.laplacian(A, x)
e, V = np.linalg.eigh(L)
# print(V)
# print(e)

u = V[:, [3]].reshape(-1, 2)
# u -= u[0]
# print(u[12])

x_n = x + u

i = mid[k]
network.plot.nodes(ax[0], np.delete(x, i, 0), color='b', s=10, zorder=10)
network.plot.edges(ax[0], x, E, color='0.2', alpha=0.6, lw=0.8)
# network.plot.motions(ax[0], x, u, color='g', scale=2.5)

network.plot.nodes(
    ax[1], x_n[i], color='firebrick', marker='s', s=10, zorder=10)
network.plot.nodes(ax[1], np.delete(x_n, i, 0), color='b', s=10, zorder=10)
network.plot.edges(ax[1], x_n, E, color='0.2', alpha=0.6, lw=0.8)
network.plot.motions(ax[1], x_n, u, color='g', scale=2)

fig.savefig('/tmp/flexion.pdf', format='pdf')
