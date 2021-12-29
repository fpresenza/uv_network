#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar sep 14 12:01:21 -03 2021
"""
import numpy as np
import matplotlib.pyplot as plt

import uvnpy.network as network
from uvnpy.network import disk_graph, plot
from uvnpy.rsn import distances
from uvnpy.rsn.localization import distances_to_neighbors_kalman

np.random.seed(6)

dt = 0.05
tiempo = np.arange(0, 15, dt)
steps = list(enumerate(tiempo))

n = 7
nodes = np.arange(n)
lim = 20
dmax = 20
x = np.empty((len(tiempo), n, 2))
u = np.zeros((n, 2))
x[0] = np.random.uniform(-lim, lim, (n, 2))
A = disk_graph.adjacency(x[0], dmax)
E = network.edges_from_adjacency(A)

# np.random.seed(10)
p = 3
hatx = np.empty((len(tiempo), n, 2))
hatx[0] = x[0] + np.random.normal(0, p, (n, 2))

Pi = p**2 * np.eye(2)
hatP = np.empty((len(tiempo), n, 2, 2))
hatP[0] = Pi

q = 0.05
Q = q**2 * np.eye(2)
R = 3.
estimator = [distances_to_neighbors_kalman(
    hatx[0, i], Pi, Q * dt, R, tiempo[0]) for i in nodes]

d = np.empty((len(tiempo), len(E)))
d[0] = distances.from_edges(E, x[0])
z = np.empty((len(tiempo), len(E)))
z[0] = np.random.normal(d[0], R)

for k, t in steps[1:]:
    for i in nodes:
        Ni = A[i].astype(bool)
        xj = hatx[k-1, Ni]
        Pj = hatP[k-1, Ni]
        ei = np.logical_or(E[:, 0] == i, E[:, 1] == i)
        zi = z[k-1, ei]
        estimator[i].update_neighbors(xj, Pj)
        estimator[i].step(t, u[i], zi)
        hatx[k, i] = estimator[i].x
        hatP[k, i] = estimator[i].P

    u[:] = np.random.uniform(-3, 3, (n, 2))
    x[k] = x[k-1] + np.random.normal(u, q/np.sqrt(dt), (n, 2)) * dt
    d[k] = distances.from_edges(E, x[k])
    z[k] = np.random.normal(d[k], R)

hatz = distances.from_edges(E, hatx)

fig, ax = plot.figure()
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
plot.nodes(ax, x[:-1], color='b', marker='.', s=1)
plot.nodes(ax, x[-1], marker='x', color='b')
plot.edges(ax, x[0], A, color='k', alpha=0.7, lw=0.7)
plot.nodes(ax, hatx[0], marker='.', color='orange')
plot.nodes(ax, hatx[1:-1], marker='.', s=1, color='orange')
plot.nodes(ax, hatx[-1], marker='x', color='orange')

fig, ax = plt.subplots(2, 1)
fig.subplots_adjust(hspace=0.35)
ax[0].grid(1)
ax[0].set_title(r'$|d_{ij} - \hat{z}_{ij}|$')
ax[0].plot(tiempo, np.abs(d - hatz))

ax[1].grid(1)
ax[1].set_title(r'$|z_{obs} - \hat{z}|$')
ax[1].plot(tiempo, np.abs(z - hatz), label=r'$\tilde{z} - \hat{z}$')


plt.show()
