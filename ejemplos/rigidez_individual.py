#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on lun jul  5 20:19:35 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt

import uvnpy.network as network
from uvnpy.rsn import distances
from gpsic.toolkit import linalg

fig, ax = network.plot.figure(1, 2)


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
i = 0
A = network.adjacency_from_edges(n, E)
Ni = (A[i] + np.eye(n)[i]).astype(bool)

Ep = np.array([
    [0, 1],
    [0, 2]], dtype=int)
Eq = np.array([
    [0, 3],
    [0, 6],
    [1, 3],
    [1, 6],
    [1, 4],
    [1, 7],
    [2, 4],
    [2, 7],
    [2, 5],
    [2, 8],
    [3, 4],
    [4, 5],
    [6, 7],
    [7, 8]], dtype=int)
Epq = np.array([
    [1, 4],
    [1, 8],
    [4, 5],
    [8, 9]], dtype=int)

# posicion original
p = np.empty((n, 2))
p[:4, 0] = 0, 2, 4, 6
p[:4, 1] = 0
p[4:8, 0] = 1, 3, 5, 7
p[4:8, 1] = 1
p[8:, 0] = 1, 3, 5, 7
p[8:, 1] = -1

pi = p[Ni]
qi = p[~Ni]

network.plot.nodes(ax[0], qi, color='b', alpha=0.6)
network.plot.edges(ax[0], qi, Eq, color='k', alpha=0.6, lw=0.8)
network.plot.edges(ax[0], p, Epq, color='k', alpha=0.6, lw=0.8)
network.plot.nodes(ax[0], pi, color='g')
network.plot.edges(ax[0], pi, Ep, color='g')
ax[0].text(p[i, 0] + 0.3, p[i, 1] - 0.1, r'$i$', color='g')
ax[0].text(
    p[i, 0] - 0.15, p[i, 1] + 0.6, r'$\mathcal{G}_i$', color='g', fontsize=15)
ax[0].set_ylim(-1.4, 1.4)

# posicion modificada
L = distances.laplacian(A, p)
_, V = np.linalg.eigh(L)
u = V[:, [3]]
p += u.reshape(-1, 2)
p -= p[0]
t = np.arctan2(p[3, 1] - p[0, 1], p[3, 0] - p[0, 0])
R = linalg.Rz(t)[:2, :2]
p = p.dot(R)

pi = p[Ni]
qi = p[~Ni]

network.plot.nodes(ax[1], qi, color='b', alpha=0.6)
network.plot.edges(ax[1], qi, Eq, color='k', alpha=0.6, lw=0.8)
network.plot.edges(ax[1], p, Epq, color='k', alpha=0.6, lw=0.8)
network.plot.nodes(ax[1], pi, color='g')
network.plot.edges(ax[1], pi, Ep, color='g')
ax[1].text(p[i, 0] + 0.3, p[i, 1] - 0.1, r'$i$', color='g')
ax[1].text(
    p[i, 0] - 0.15, p[i, 1] + 0.7, r'$\mathcal{G}_i$', color='g', fontsize=15)
ax[1].set_ylim(-1.4, 1.4)


plt.show()
