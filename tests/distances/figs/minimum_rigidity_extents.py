#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from uvnpy.toolkit import plot
from uvnpy.graphs.core import geodesics, adjacency_matrix_from_edges
from uvnpy.graphs.models import DiskGraph
from uvnpy.distances.core import (
    minimum_distance_rigidity_extents,
    minimum_distance_rigidity_radius,
    sufficiently_dispersed_position
)

np.set_printoptions(suppress=True, precision=4, linewidth=250)

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-s', '--seed',
    default=0, type=int, help='numpy random seed')
arg = parser.parse_args()

n = 20
threshold = 1e-5

i = 0
np.random.seed(arg.seed)
while i < 10:
    # p = np.random.uniform((0, 0), (1, 0.9), (n, 2))
    p = sufficiently_dispersed_position(n, (0, 1), (0, 0.9), 0.1)
    E0 = DiskGraph(p, dmax=2/np.sqrt(n)).edge_set(directed=False)
    E, Rmin = minimum_distance_rigidity_radius(
        E0, p, threshold, return_radius=True
    )
    A = adjacency_matrix_from_edges(n, E, directed=False).astype(float)

    fig, ax = plt.subplots(figsize=(2.25, 2.25))
    # fig.subplots_adjust(top=0.88, bottom=0.15, wspace=0.28)
    G = geodesics(A)
    h = minimum_distance_rigidity_extents(E, G, p, threshold)
    D = np.max(G)

    if np.max(h) < 4 and np.max(h) != D:
        i += 1
        print('---{}---'.format(i))
        print(p)
    else:
        continue

    one_hop_rigid = h == 1
    two_hop_rigid = h == 2
    three_hop_rigid = h == 3

    ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,
        left=False,
        pad=1,
        labelsize='x-small'
    )
    # ax.grid(1, lw=0.4)
    ax.set_aspect('equal')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    # ax.set_xlabel(r'$\mathrm{x}$', fontsize='x-small', labelpad=0.6)
    # ax.set_ylabel(r'$\mathrm{y}$', fontsize='x-small', labelpad=0)
    ax.set_xticks(np.linspace(0, 1, 4, endpoint=True))
    ax.set_yticks(np.linspace(0, 1, 4, endpoint=True))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plot.bars(
        ax, p, E,
        lw=0.5, color=cm.coolwarm(20), zorder=0
    )

    for k in np.unique(h):
        c = h == k
        v_artist = ax.scatter(
            p[c, 0], p[c, 1],
            marker='o', s=(k+1) * 10,
            c=h[c], cmap=cm.coolwarm, vmin=-3, vmax=3,
            label=r'${}$'.format(k)
        )

    ax.legend(
        fontsize='x-small', handlelength=1, labelspacing=1.5,
        borderpad=0.2, handletextpad=0.2, framealpha=1.,
        ncol=4, columnspacing=0.4, loc='upper center'
    )
    fig.savefig(
        '/tmp/minimum_rigidity_extents_{}.png'.format(i), format='png', dpi=360
    )
