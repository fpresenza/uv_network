#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on dom 23 ene 2022 22:37:55 -03
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import uvnpy.network as network
from uvnpy.network import disk_graph
from uvnpy.rsn import rigidity

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'

# Parte 1

hL = 50
L = 2 * hL
L2 = L**2
density = 1/100  # inverso del area que le corresponde a cada agente
n = int(density * L2)
d_factor = 1.7
dmax = d_factor / np.sqrt(density)
print('dmax = {}'.format(dmax))

for i in range(348, 2000):  # 348
    np.random.seed(i)
    x = np.random.uniform(-hL, hL, (n, 2))
    print('seed = {}'.format(i))
    A = disk_graph.adjacency(x, dmax)

    lambda4 = rigidity.eigenvalue(A, x)
    if lambda4 > 1e-3:
        G = nx.from_numpy_matrix(A)
        D = nx.diameter(G)
        extents = rigidity.extents(A, x)
        one_hop_rigid = extents == 1
        two_hop_rigid = extents == 2
        three_hop_rigid = extents == 3
        four_hop_rigid = extents == 4
        if sum(two_hop_rigid) > 2 and sum(three_hop_rigid) > 1 and sum(four_hop_rigid) == 0:   # noqa
            # print(len(two_hop_rigid), len(three_hop_rigid), len(four_hop_rigid))   # noqa
            # print('seed = ', i)
            # print(extents)
            break

print('diameter = {}'.format(D))

fig, ax = plt.subplots(1, 2, figsize=(4, 2))
fig.subplots_adjust(top=0.88, bottom=0.15, wspace=0.28)
ax[0].tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='xx-small')
ax[0].grid(1, lw=0.4)
ax[0].set_aspect('equal')
ax[0].set_xlim(-hL*1.1, hL*1.1)
ax[0].set_ylim(-hL*1.1, hL*1.4)
ax[0].set_xlabel(r'$\mathrm{x}$', fontsize='x-small', labelpad=0.6)
ax[0].set_ylabel(r'$\mathrm{y}$', fontsize='x-small', labelpad=0)
ax[0].set_xticks([-hL, 0, hL])
ax[0].set_yticks([-hL, 0, hL])
ax[0].set_xticklabels([-hL, 0, hL])
ax[0].set_yticklabels([-hL, 0, hL])

network.plot.nodes(
    ax[0], x[one_hop_rigid],
    marker='o', color='royalblue', s=7, zorder=10, label=r'$1$-hop')
network.plot.nodes(
    ax[0], x[two_hop_rigid],
    marker='D', color='chocolate', s=7, zorder=10, label=r'$2$-hop')
network.plot.nodes(
    ax[0], x[three_hop_rigid],
    marker='s', color='mediumseagreen', s=7, zorder=10, label=r'$3$-hop')
# network.plot.nodes(
#     ax[0], x[four_hop_rigid],
#     marker='^', color='lightcoral', s=7, zorder=10, label=r'$4$')
network.plot.edges(ax[0], x, A, color='0.6', lw=0.5)
ax[0].legend(
    fontsize='xx-small', handlelength=1, labelspacing=0.4,
    borderpad=0.2, handletextpad=0.2, framealpha=1.,
    ncol=3, columnspacing=0.2, loc='upper center')

# Parte 2

hL = 50
L = 2 * hL
L2 = L**2
density = 1/100  # inverso del area que le corresponde a cada agente
n = int(density * L2)
dmax = np.array([25, 20, 17.5])

R = len(dmax)
N = 250
rigidity_extent = np.zeros((R, N), dtype=np.ndarray)
max_rigidity_extent = np.zeros((R, N))
diam = np.zeros((R, N), dtype=int)

for i in range(R):
    k = 0
    while k < N:
        x = np.random.uniform(-hL, hL, (n, 2))
        A = disk_graph.adjacency(x, dmax[i])
        try:
            rigidity_extent[i, k] = rigidity.extents(A, x)
            max_rigidity_extent[i, k] = rigidity_extent[i, k].max()

            G = nx.from_numpy_matrix(A)
            diam[i, k] = nx.diameter(G)
            # geo = network.geodesics(A)
            print(i, k)
            k += 1
        except ValueError as e:
            # print(e)
            pass
        except StopIteration as e:
            print(e)
            pass

centralization_index = max_rigidity_extent / diam
print(centralization_index)
colors = ['C2', 'C1', 'C0']
labels = [r'$\Omega = {}$'.format(d) for d in dmax]
ax[1].hist(
    centralization_index.T,
    color=colors,
    bins=np.arange(0, 1.1, 0.1),
    # density=True,
    histtype='bar',
    stacked=True,
    label=labels)

ax[1].tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='xx-small')
ax[1].grid(1, lw=0.4)
# perc = np.array([20, 40, 60, 80])
# ax[1].set_yticks(perc * N / 100)
# ax[1].set_yticklabels(perc)
ax[1].set_xlabel(
    'Índice de centralización, ' + r'$\xi$',
    fontsize='x-small', labelpad=0.6)
ax[1].set_ylabel(r'Frecuencia ($\%$)', fontsize='x-small', labelpad=1)
ax[1].legend(
    fontsize='xx-small', handlelength=1.5, labelspacing=0.5, borderpad=0.2)

fig.savefig('/tmp/extents.png', format='png', dpi=300)

plt.show()
