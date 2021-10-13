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

plt.rcParams["legend.borderpad"] = 0.1
plt.rcParams["legend.labelspacing"] = 0.3
plt.rcParams["legend.handlelength"] = 1.0
plt.rcParams["legend.columnspacing"] = 1.0

n = 60
x = np.empty((n, 2), dtype=float)
x[::2, 0] = np.arange(int(n/2))
x[::2, 1] = 0
x[1::2, 0] = 1/2 + np.arange(int(n/2))
x[1::2, 1] = np.sqrt(3)/2

dmax = 1.1
# E = disk_graph.edges(x, dmax)
A = disk_graph.adjacency(x, dmax)
x *= 2
cut = 20
fig, ax = plt.subplots(figsize=(4, 1))
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='xx-small')
ax.set_aspect('equal')
ax.grid(1, lw=0.3)
xticks = np.append(np.arange(0, cut, 2), 19)
ax.set_xticks(xticks)
ax.set_xticklabels(xticks)
ax.set_ylim(-0.5, 2.5)
ax.set_yticks([0, 2])
ax.set_yticklabels([0, 2])
network.plot.nodes(ax, x[:cut], color='b', s=12, zorder=10)
network.plot.edges(ax, x[:cut], A[:cut, :cut], color='k', lw=0.7, zorder=1)
fig.savefig('/tmp/slim.pdf', format='pdf')


a2 = np.empty(n - 2)
l4 = np.empty(n - 2)
diam = np.empty(n - 2, dtype=int)
edges = np.empty(n - 2, dtype=int)
deg_max = np.empty(n - 2, dtype=int)
# c = np.empty(n - 2)
energy = np.empty(n - 2)
numerator = np.empty(n - 2)
denominator = np.empty(n - 2)

for i in range(3, n + 1):
    p = x[:i]
    A = disk_graph.adjacency(p, dmax)
    E = network.edges_from_adjacency(A)
    edges[i - 3] = len(E)

    L = network.laplacian_from_adjacency(A)
    a2[i - 3] = np.linalg.eigvalsh(L)[1]

    S = rigidity.symmetric_matrix(A, p)
    l4[i - 3] = np.linalg.eigvalsh(S)[3]

    G = nx.from_numpy_matrix(A)
    _D = nx.diameter(G)
    diam[i - 3] = int(_D)
    deg = G.degree()
    deg_max[i - 3] = max([d for _, d in deg])

    e = np.array(list(nx.eccentricity(G).values()))
    peripheral = np.argwhere(e == _D).ravel()

    shortest_paths = [
        nx.single_source_shortest_path(G, per) for per in peripheral]
    vertices = [list(sp.keys()) for sp in shortest_paths]
    paths = [list(sp.values()) for sp in shortest_paths]
    hops = np.array([[len(path) - 1 for path in pivot] for pivot in paths])

    u = np.zeros((i, len(peripheral)))
    u[vertices, np.arange(len(peripheral)).reshape(-1, 1)] = hops/_D
    v = u - u.mean(0)
    qf = (v * L.dot(v)).sum(0)
    nor = (v * v).sum(0)
    imin = np.argmin(qf / nor)
    # print(u[:, imin].mean(0) - (i+1)/(4*_D+2))
    energy[i - 3] = qf[imin] / nor[imin]
    numerator[i - 3] = qf[imin]
    denominator[i - 3] = nor[imin]

nodes = np.arange(3, n + 1)
deg_avg = 2*edges / nodes
alpha = np.linalg.pinv(nodes[:, None]).dot(denominator - 1/2)
# c = np.maximum(1/2, alpha*nodes)
c = 1/2 + alpha*nodes
f = np.sqrt(deg_max - 1) / deg_max
b_0 = 1 - f + 2/diam * (1 + f)
print(b_0)

fig, ax = plt.subplots(figsize=(2, 1.5))
# fig.subplots_adjust(bottom=0.2, left=0.22)
fig.subplots_adjust(left=0.255, bottom=0.2)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='xx-small')
ax.grid(1, lw=0.4)
ax.set_xlabel('Diameter (D)', fontsize='x-small', labelpad=1)
ax.set_ylabel('Eigenvalue bounds', fontsize='x-small', labelpad=1)
ax.semilogy(diam, a2, lw=0.9, label=r'$a(\mathcal{G})$')
ax.semilogy(diam, l4, lw=0.9, label=r'$\rho(\mathcal{F})$')
ax.semilogy(
    diam, b_0,
    ls='dotted', label=r'$b_0$', color='k')
ax.semilogy(
    diam, 2*edges/diam**2,
    ls='--', lw=0.9, label=r'$b_1$')
ax.set_ylim(bottom=1e-5)
# ax.semilogy(
#     diam, deg_avg/(2*alpha*diam**2),
#     ls='--', lw=0.9,
#     label=r'$b_2$')
# ax.plot(nodes, l4, label=r'$\lambda_4$')
# ax.plot(nodes, energy, ls='--', label=r'$E$')
# ax.plot(nodes, edges/(diam**2)/c,
#   ls='--', label=r'$\frac{m/D^2}{1/2 + \alpha n}$')
ax.legend(
    fontsize='x-small',
    handletextpad=0.1,
    borderpad=0.2, ncol=2, columnspacing=0.2)
# bbox_to_anchor=(1., 1.03))
fig.savefig('/tmp/bounds.pdf', format='pdf')

fig, ax = plt.subplots()
ax.set_xlabel('Number of nodes (n)')
ax.grid(1)
ax.plot(nodes, deg_avg, label=r'$\delta_{\mathrm{avg}}$')
ax.plot(nodes, edges, label=r'$m$')
ax.scatter(nodes, 2*nodes - 3)
ax.plot(nodes, diam, label=r'$D$')
# ax.plot(nodes, edges/diam, label=r'$m/D$')
# ax.legend()

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
