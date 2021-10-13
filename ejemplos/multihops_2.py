#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on vie sep 10 10:38:33 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx  # noqa

from uvnpy.network import disk_graph, subsets  # noqa
from uvnpy.rsn import rigidity  # noqa

# gpsic.plotting.core.set_pubstyle(style='sans-serif')
plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


def grid(n, sep):
    k = np.ceil(np.sqrt(n)) / 2
    nums = np.arange(-k, k) * sep
    g = np.meshgrid(nums, nums)
    return np.vstack(np.dstack(g))[:n]


# area = 50  # inverso del area que le corresponde a cada agente
# d_factor = np.array([1.75, 2.0, 2.5])
# dmax = d_factor / np.sqrt(density)
dmax = np.array([25.])
sep = dmax / 2

R = len(dmax)
N = 500 - 3
K = 10
rigidity_extent = np.zeros((N, K), dtype=np.ndarray)
nodes = np.zeros((N, K), dtype=int)
edges = np.zeros((N, K), dtype=int)
load = np.zeros((N, K))
max_rigidity_extent = np.zeros((N, K))

for i in range(N):
    k = 0
    while k < K:
        n = i + 3
        x = grid(n, sep) + np.random.uniform(-sep/2, sep/2, (n, 2))
        A = disk_graph.adjacency(x, dmax)
        # try:
        #     rigidity.algebraic_condition(A, x)
        #     rigidity_extent[i, k] = rigidity.minimum_hops(A, x)
        # #     max_rigidity_extent[i, k] = rigidity_extent[i, k].max()
        nodes[i, k] = n
        edges[i, k] = int(A.sum()/2)
        # #     load[i, k] = subsets.degree_load_std(A, rigidity_extent[i, k])
        # #     load[i, k] /= edges[i, k]
        #     print(i, k)
        k += 1
        # except ValueError as e:
        #     pass
        # except StopIteration as e:
        #     print(e)
        #     pass

print(nodes)
# print(rigidity_extent)
# print(max_rigidity_extent.max())
# versus number of nodes
fig, ax = plt.subplots(1, 2, figsize=(6, 3.5))
ax = ax.ravel()
fig.subplots_adjust(wspace=0.33, bottom=0.2)
for _ax in ax:
    _ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize='xx-small')
    _ax.grid(1, lw=0.4)
degree = 2*edges/nodes
ax[0].scatter(nodes[:, 1], degree.mean(1), s=3)
ax[0].set_ylim(0, 10)
# ax[1].plot(unique_nodes, np.repeat(p0[0], len(unique_nodes)))
# ax[1].set_ylim(0, 10)

plt.show()
