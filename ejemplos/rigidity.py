#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on lun jul  5 20:19:35 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt

import uvnpy.network as network
from uvnpy.rsn import rigidity
from gpsic.toolkit import linalg


p = np.array([
    [1., 0.],
    [0, 1],
    [-2, 0],
    [0, -1]])

fig, ax = plt.subplots(1, 3, figsize=(5, 1))
ax = ax.ravel()
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
    _ax.grid(1, lw=0.5)
    # _ax.set_aspect('equal')
    _ax.set_xlim(-2.5, 1.5)
    _ax.set_ylim(-1.5, 2)


E = network.complete_edges(len(p))

network.plot.nodes(ax[0], p, color='b')
network.plot.edges(ax[0], p, E, color='k', alpha=0.6)

network.plot.nodes(ax[1], p, color='b')
network.plot.edges(ax[1], p, E[[0, 2, 3, 4, 5]], color='k', alpha=0.6)

p[0, 0] = -1
network.plot.nodes(ax[1], p[0], color='b', alpha=0.3)
network.plot.edges(ax[1], p, E[[0, 2]], color='k', alpha=0.3)

p[0, 0] = 1
network.plot.nodes(ax[2], p, color='b')
network.plot.edges(ax[2], p, E[[0, 2, 3, 5]], color='k', alpha=0.6)

H = rigidity.matrix_from_edges(E[[0, 2, 3, 5]], p)
_, v = np.linalg.eigh(H.T.dot(H))
v4 = v[:, 3].reshape(-1, 2)
q = p + v4
q = q - q.mean(0) + p.mean(0)
t = np.arctan2(q[0, 1] - q[2, 1], q[0, 0] - q[2, 0])
Rz = linalg.Rz(t)[:2, :2]
q = q.dot(Rz)
network.plot.nodes(ax[2], q, color='b', alpha=0.3)
network.plot.edges(ax[2], q, E[[0, 2, 3, 5]], color='k', alpha=0.3)

plt.show()
fig.savefig('/tmp/rigid_flex.pdf', format='pdf')
