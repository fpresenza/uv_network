#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on dom ago 15 20:00:37 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt

import uvnpy.network as network
from uvnpy.network import disk_graph
from uvnpy.rsn import rigidity

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


hL = 50
L = 2 * hL
L2 = L**2
density = 1/100  # inverso del area que le corresponde a cada agente
n = int(density * L2)
d_factor = 2.0
dmax = d_factor / np.sqrt(density)

for i in range(210, 2000):  # 1093
    np.random.seed(i)
    x = np.random.uniform(-hL, hL, (n, 2))
    A = disk_graph.adjacency(x, dmax)

    lambda4 = rigidity.algebraic_condition(A, x, return_value=True)
    if lambda4 > 1e-3:
        min_hops = rigidity.minimum_hops(A, x)
        one_hop_rigid = min_hops == 1
        two_hop_rigid = min_hops == 2
        # print(np.argwhere(two_hop_rigid))
        three_hop_rigid = min_hops == 3
        four_hop_rigid = min_hops == 4
        if sum(two_hop_rigid) > 2 and sum(three_hop_rigid) > 1 and sum(four_hop_rigid) > 1:   # noqa
            # print(len(two_hop_rigid), len(three_hop_rigid), len(four_hop_rigid))   # noqa
            print('seed = ', i)
            print(min_hops)
            break


fig, ax = plt.subplots(figsize=(2.5, 2))
# fig.subplots_adjust(wspace=0.24)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    labelsize='xx-small')

ax.set_aspect('equal')
ax.grid(1, lw=0.4)
ax.set_xlim(-hL*1.1, hL*1.1)
ax.set_ylim(-hL*1.1, hL*1.1)
ax.set_xticks([-50, 0, 50])
ax.set_yticks([-50, 0, 50])
ax.set_xticklabels([-50, 0, 50])
ax.set_yticklabels([-50, 0, 50])
# ax.set_xlabel('x', fontsize='x-small', labelpad=0.2)
# ax.set_ylabel('y', fontsize='x-small', labelpad=-2)

network.plot.nodes(
    ax, x[one_hop_rigid],
    marker='o', color='royalblue', s=7, zorder=10, label=r'$\eta_i = 1$')
network.plot.nodes(
    ax, x[two_hop_rigid],
    marker='D', color='orange', s=7, zorder=10, label=r'$\eta_i = 2$')
network.plot.nodes(
    ax, x[three_hop_rigid],
    marker='s', color='mediumseagreen', s=7, zorder=10, label=r'$\eta_i = 3$')
network.plot.nodes(
    ax, x[four_hop_rigid],
    marker='^', color='lightcoral', s=7, zorder=10, label=r'$\eta_i = 4$')
network.plot.edges(ax, x, A, color='0.6', lw=0.5)
ax.legend(
    fontsize='xx-small', handlelength=1, labelspacing=0.3,
    borderpad=0.2, handletextpad=0.2, loc='center right',
    bbox_to_anchor=(1.3, 0.5))

fig.savefig('/tmp/random_framework.pdf', format='pdf')

plt.show()
