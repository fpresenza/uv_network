#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on dom abr 11 18:34:34 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt

import uvnpy.network.core as core
import uvnpy.network.disk_graph as disk_graph


N = 16
V = np.arange(N)
L = 100
dmax = 65

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax in axes:
    ax.set_aspect('equal')
    ax.grid(1)
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.text(-31, -26, r'nodo-$i$', color='g')

np.random.seed(13)
p = np.random.uniform(-0.8 * L, 0.8 * L, (N, 2))
E = disk_graph.edges(p, dmax)
# print(p)
# print(E)

# q = disk_graph.local_neighbors(p[8], np.delete(p, 8, axis=0), dmax)
# pl = np.vstack([p[8], q])
Gi = disk_graph.local_subgraph(p, 8, dmax)
pl = p[Gi]
El = disk_graph.edges(pl, dmax)
Eaug = core.complete_undirected_edges(V[Gi])
# print(pl)
# print(El)
# print(Eaug)

# grafo
for ax in axes:
    core.plot_nodes(ax, p, color='b', marker='o')
    core.plot_edges(ax, p, E, color='0.2', lw=0.8)

# grafo-i
for ax in axes:
    core.plot_nodes(ax, pl, color='g', marker='o')
    core.plot_edges(ax, pl, El, color='g')
axes[0].text(-54, -11, r'$\mathcal{G}_i$', color='g', fontsize=15)
axes[0].set_title(r'Subgrafo local $i$')

core.plot_edges(axes[1], p, Eaug, color='g', ls='--')
axes[1].text(-62, 8, r'$\mathcal{G}^{\ast}_i$', color='g', fontsize=15)
axes[1].set_title(r'Subgrafo local completo $i$')


# plt.show()
plt.savefig('/tmp/subgrafos.png')
