#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on dom abr 11 18:34:34 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt

import uvnpy.network as network


E = E = np.empty((20, 2), dtype=int)
E[0] = 0, 4
E[1] = 0, 8
E[2] = 1, 4
E[3] = 1, 8
E[4] = 1, 5
E[5] = 1, 9
E[6] = 2, 5
E[7] = 2, 9
E[8] = 2, 6
E[9] = 2, 10
E[10] = 3, 6
E[11] = 3, 10
E[12] = 3, 7
E[13] = 3, 11
E[14] = 4, 5
E[15] = 5, 6
E[16] = 6, 7
E[17] = 8, 9
E[18] = 9, 10
E[19] = 10, 11

n = 12
i = 1
A = network.adjacency_from_edges(n, E)
Ni = (A[i] + np.eye(n)[i]).astype(bool)

p = np.empty((n, 2))
p[:4, 0] = 0, 2, 4, 6
p[:4, 1] = 0
p[4:8, 0] = 1, 3, 5, 7
p[4:8, 1] = 1
p[8:, 0] = 1, 3, 5, 7
p[8:, 1] = -1

fig, ax = network.plot.figure(1, 2)
ax[0].set_ylim(-1.4, 1.4)
ax[1].set_ylim(-1.4, 1.4)

pi = p[Ni]
qi = p[~Ni]
Ep = np.array([
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
    [1, 2],
    [3, 4]], dtype=int)
Eq = np.array([
    [1, 3],
    [1, 5],
    [2, 3],
    [2, 4],
    [2, 5],
    [2, 6],
    [3, 4],
    [5, 6]], dtype=int)
Epq = np.array([
    [0, 4],
    [0, 8],
    [5, 2],
    [5, 6],
    [9, 2],
    [9, 10]], dtype=int)

# ax 1 : subgrafo i
network.plot.nodes(ax[0], qi, color='b', alpha=0.6)
network.plot.edges(ax[0], qi, Eq, color='k', alpha=0.6, lw=0.8)
network.plot.edges(ax[0], p, Epq, color='k', alpha=0.6, lw=0.8)
network.plot.nodes(ax[0], pi, color='g')
network.plot.edges(ax[0], pi, Ep, color='g')
ax[0].text(p[i, 0] + 0.3, p[i, 1] - 0.1, r'$i$', color='g')
ax[0].text(
    p[i, 0] - 0.2, p[i, 1] + 0.5, r'$\mathcal{G}_i$', color='g', fontsize=15)
ax[0].set_title(r'Subgrafo local $i$')

# ax 2 : interacción i
# pni = np.delete(p, i, axis=0)
# Epni = disk_graph.edges(pni, dmax)
# Eini = disk_graph.inter_edges(p[[i]], pni, dmax)
# Eini[:, 1] += 1
# pNi = p[Ni]
network.plot.nodes(ax[1], qi, color='b', alpha=0.6)
network.plot.edges(ax[1], qi, Eq, color='k', alpha=0.6, lw=0.8)
network.plot.edges(ax[1], p, Epq, color='k', alpha=0.6, lw=0.8)
network.plot.nodes(ax[1], pi, color='g', s=10)
network.plot.edges(ax[1], pi, Ep, color='g')
ax[1].text(p[i, 0] + 0.3, p[i, 1] - 0.1, r'$i$', color='g')

for ni in pi:
    dpi = ni - p[i]
    if not np.allclose(dpi, 0):
        ax[1].arrow(
            p[i, 0], p[i, 1], dpi[0], dpi[1],
            color='g',
            head_width=.15, head_length=0.3, length_includes_head=True)
        ax[1].arrow(
            ni[0], ni[1], -dpi[0], -dpi[1],
            color='g',
            head_width=.15, head_length=0.3, length_includes_head=True)


ax[1].set_title(r'Interacción local $i$')


plt.show()
