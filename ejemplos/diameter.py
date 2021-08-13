#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on lun jul  5 20:19:35 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import uvnpy.network as network
from uvnpy.network import disk_graph, subgraph
from uvnpy.rsn import rigidity

N = 30
x = np.empty((N, 2), dtype=float)
x[::2, 0] = np.arange(int(N/2))
x[::2, 1] = 0
x[1::2, 0] = 1/2 + np.arange(int(N/2))
x[1::2, 1] = np.sqrt(3)/2

dmax = 1.1

# dmax = 10.
# np.random.seed(0)
# x = np.random.uniform(-15, 15, (N, 2))
# E = disk_graph.edges(x, dmax)
# A = network.adjacency_from_edges(N, E)
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
ax[0].text(x[N - 1, 0] - 0.6, x[N - 1, 1] + 0.25, r'$n={}$'.format(N))


a = np.empty(N - 2)
l4 = np.empty(N - 2)

for i in range(3, N + 1):
    p = x[:i]
    A = disk_graph.adjacency(p, dmax)
    E = network.edges_from_adjacency(A)
    D = network.incidence_from_edges(i, E)

    Lc = network.laplacian_from_adjacency(A)
    a[i - 3] = np.linalg.eigvalsh(Lc)[1]

    # R = rigidity.classic_matrix(D, p * 2)
    # Lr = R.T.dot(R)
    Lr = rigidity.laplacian(A, p)
    l4[i - 3] = np.linalg.eigvalsh(Lr)[3]

ax[1].set_xlabel('n')
ax[1].grid(1)
ax[1].plot(np.arange(3, N + 1), a, label='Fiedler Eigenvalue')
ax[1].plot(np.arange(3, N + 1), l4, label='Rigidity Eigenvalue')
ax[1].legend()
fig.savefig('/tmp/diameter.pdf', format='pdf')

# plt.set_cmap(cm.coolwarm)
fig, ax = plt.subplots(figsize=(4, 2))
fig.subplots_adjust(left=0.17, bottom=0.22)
l4 = np.empty((16, N))
A = disk_graph.adjacency(x, dmax)
for h in range(1, 16):
    for i in range(N):
        Ni = subgraph.multihop_neighborhood(A, i, hops=h, inclusive=True)
        xi = x[Ni]
        Ai = A[Ni][:, Ni]
        L = rigidity.laplacian(Ai, xi)
        l4[h-1, i] = np.linalg.eigvalsh(L)[3]

    col = cm.coolwarm(int(320 - 20 * h))
    ax.semilogy(range(1, N+1), l4[h-1])

ax.set_xlabel('Vertex index')
ax.set_ylabel(r'$\eta$-Rigidity Eigenvalues')
ax.set_xlim(1, 30)
ax.grid(1, lw=0.4)
ax.minorticks_on()
fig.savefig('/tmp/eta_metrics.pdf', format='pdf')


fig, ax = plt.subplots(3, 1, figsize=(6, 4))
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
    _ax.grid(1)
    _ax.set_ylim(-1, 2)


A = disk_graph.adjacency(x, dmax)
E = disk_graph.edges(x, dmax)
L = rigidity.laplacian(A, x)
_, V = np.linalg.eigh(L)
# print(distances.matrix(x)[0])

u = V[:, [3]].reshape(-1, 2)
x_n = x + u
# print(distances.matrix(x_n)[0])
network.plot.nodes(ax[0], x_n, color='b', label=r'$p=4$')
network.plot.edges(ax[0], x_n, E, color='k', alpha=0.6)

u = V[:, [4]].reshape(-1, 2)
x_n = x + u
# print(distances.matrix(x_n)[0])
network.plot.nodes(ax[1], x_n, color='b', label=r'$p=5$')
network.plot.edges(ax[1], x_n, E, color='k', alpha=0.6)

u = V[:, [5]].reshape(-1, 2)
x_n = x + u
# print(distances.matrix(x_n)[0])
network.plot.nodes(ax[2], x_n, color='b', label=r'$p=6$')
network.plot.edges(ax[2], x_n, E, color='k', alpha=0.6)

ax[0].legend(
    loc='upper right', markerscale=0., handletextpad=0, handlelength=0)
ax[1].legend(
    loc='upper right', markerscale=0., handletextpad=0, handlelength=0)
ax[2].legend(
    loc='upper right', markerscale=0., handletextpad=0, handlelength=0)

fig.savefig('/tmp/flexion.pdf', format='pdf')

plt.show()
