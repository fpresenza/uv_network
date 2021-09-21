#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on lun jul  5 20:19:35 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx  # noqa

import uvnpy.network as network
from uvnpy.network import disk_graph
from uvnpy.rsn import rigidity

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


def filter_edges(E):
    dels = []
    for k, e in enumerate(E):
        if (e % 2 == 0).all():
            dels.append(k)
    return np.delete(E, dels, axis=0)


dmax = 1.5
K = np.arange(1, 10)
a2 = np.empty(len(K))
a3 = np.empty(len(K))
l4 = np.empty(len(K))
nodes = np.empty(len(K), dtype=int)
diam = np.empty(len(K), dtype=int)
edges = np.empty(len(K), dtype=int)
deg_max = np.empty(len(K), dtype=int)
energy = np.empty(len(K))
numerator = np.empty(len(K))
denominator = np.empty(len(K))

fig, axes = plt.subplots(3, 3)
for k, ax in zip(K, axes.ravel()):
    t = np.arange(-k, k+1, 1)
    X, Y = np.meshgrid(t, t)
    x = np.dstack([X, Y]).reshape(-1, 2)
    E = disk_graph.edges(x, dmax)
    E = filter_edges(E)
    A = network.adjacency_from_edges(len(x), E)
    n = len(x)
    nodes[k - 1] = n

    edges[k - 1] = len(E)

    L = network.laplacian_from_adjacency(A)
    a2[k - 1] = np.linalg.eigvalsh(L)[1]

    S = rigidity.laplacian(A, x)
    l4[k - 1] = np.linalg.eigvalsh(S)[3]

    G = nx.from_numpy_matrix(A)
    D = nx.diameter(G)
    diam[k - 1] = int(D)
    deg = G.degree()
    deg_max[k - 1] = max([d for _, d in deg])

    e = np.array(list(nx.eccentricity(G).values()))
    peripheral = np.argwhere(e == D).ravel()

    shortest_paths = [nx.single_source_shortest_path(G, p) for p in peripheral]
    vertices = [list(sp.keys()) for sp in shortest_paths]
    paths = [list(sp.values()) for sp in shortest_paths]
    hops = np.array([[len(path) - 1 for path in pivot] for pivot in paths])

    u = np.zeros((n, len(peripheral)))
    u[vertices, np.arange(len(peripheral)).reshape(-1, 1)] = hops/D
    v = u - u.mean(0)
    vTLv = (v * L.dot(v)).sum(0)
    vTv = (v * v).sum(0)
    imin = np.argmin(vTLv / vTv)
    # print(u[:, imin].mean(0))
    energy[k - 1] = vTLv[imin] / vTv[imin]
    numerator[k - 1] = vTLv[imin]
    denominator[k - 1] = vTv[imin]

    # ax.tick_params(
    #     axis='both',       # changes apply to the x-axis
    #     which='both',      # both major and mivTv ticks are affected
    #     bottom=False,      # ticks along the bottom edge are off
    #     top=False,         # ticks along the top edge are off
    #     left=False,
    #     right=False,
    #     labelbottom=False,
    #     labelleft=False)   # labels along the bottom edge are off
    # ax.set_aspect('equal')
    # ax.grid(1)
    # network.plot.nodes(ax, x)
    # network.plot.edges(ax, x, E)

alpha = np.linalg.pinv(nodes[:, None]).dot(denominator - 1/2)
deg_avg = 2*edges / nodes
# c = np.maximum(1/2, alpha*nodes)
c = 1/2 + alpha*nodes
gamma = np.sqrt(deg_max - 1) / deg_max

fig, ax = plt.subplots()
ax.set_xlabel('Number of nodes (n)')
ax.grid(1)
ax.semilogy(nodes, a2, label=r'$a$')
# ax.plot(K, l4, label=r'$\lambda_4$')
# ax.plot(nodes, energy, ls='--', label=r'$E$')
# ax.plot(nodes, edges/(diam**2)/c,
#   ls='--', label=r'$\frac{m/D^2}{1/2 + \alpha n}$')
ax.semilogy(
    nodes, deg_avg/(2*alpha*diam**2),
    ls='--', label=r'$\frac{\delta_{\mathrm{avg}}}{2\alpha D^2}$')
ax.semilogy(nodes, 2*edges/diam**2, ls='--', label=r'$\frac{2m}{D^2}$')
ax.semilogy(
    nodes, 1 - gamma + 2/diam * (1 + gamma),
    ls='--', label=r'$\gamma$')
ax.legend()

fig, ax = plt.subplots()
ax.set_xlabel('Number of nodes (n)')
ax.grid(1)
ax.plot(nodes, deg_avg, label=r'$\delta_{\mathrm{avg}}$')
# ax.plot(K, diam, label=r'$D$')
# ax.plot(K, edges/diam, label=r'$m/D$')
ax.legend()

fig, ax = plt.subplots()
ax.set_xlabel('Number of nodes (n)')
ax.grid(1)
ax.plot(nodes, numerator, color='b', label=r'$v^T L v$')
ax.plot(nodes, edges/(diam**2), color='b', ls='--', label=r'$m/D^2$')
ax.plot(nodes, denominator, color='g', label=r'$v^T v$')
ax.plot(nodes, c, color='g', ls='--', label=r'$1/2 + \alpha n$')
ax.hlines(1/2, nodes[0], nodes[-1], color='k', ls='--', label=r'$1/2$')
ax.legend()

plt.show()
