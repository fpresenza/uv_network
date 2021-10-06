#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on vie sep 10 10:38:33 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx  # noqa

from uvnpy.network import disk_graph, subsets
from uvnpy.rsn import rigidity

# gpsic.plotting.core.set_pubstyle(style='sans-serif')
plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'

area = 100  # inverso del area que le corresponde a cada agente
# d_factor = np.array([1.75, 2.0, 2.5])
# dmax = d_factor / np.sqrt(density)
dmax = np.array([25.])

R = len(dmax)
N = 100
rigid_hops = np.zeros((R, N), dtype=np.ndarray)
nodes = np.zeros((R, N), dtype=int)
edges = np.zeros((R, N), dtype=int)
load = np.zeros((R, N))
max_rigid_hops = np.zeros((R, N))

for i in range(R):
    k = 0
    while k < N:
        n = np.random.choice(np.arange(3, 100))
        hL = np.sqrt(n*area)/2
        x = np.random.uniform(-hL, hL, (n, 2))
        A = disk_graph.adjacency(x, dmax)
        try:
            rigid_hops[i, k] = rigidity.minimum_hops(A, x)
            max_rigid_hops[i, k] = rigid_hops[i, k].max()
            nodes[i, k] = n
            edges[i, k] = int(A.sum()/2)
            load[i, k] = subsets.degree_load_std(A, rigid_hops[i, k])
            load[i, k] /= edges[i, k]
            print(i, k)
            k += 1
        except ValueError as e:
            pass
        except StopIteration as e:
            print(e)
            pass

print(nodes)
print(rigid_hops)
print(max_rigid_hops.max())
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
mark = ['s', '^', 'o']
col = ['C2', 'C1', 'C0']
max_hops = int(max_rigid_hops.max())
for i in range(R):
    counts = np.zeros((100, max_hops))
    for k in range(N):
        n = nodes[i, k]
        counts[n] += np.bincount(rigid_hops[i, k], minlength=max_hops+1)[1:]

unique_nodes = np.unique(nodes.ravel())
counts = counts[unique_nodes]
counts /= counts.sum(axis=1).reshape(-1, 1)
counts *= 100
# print(edges)
for i in range(max_hops):
    ax[0].plot(unique_nodes, counts[:, i], markersize=3)
ax[1].scatter(nodes, 2*edges/nodes, s=3)
# ax[1].plot(unique_nodes, np.repeat(p0[0], len(unique_nodes)))
# ax[1].set_ylim(0, 10)

plt.show()
