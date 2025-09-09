#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from uvnpy.toolkit import plot
from uvnpy.network.core import edges_from_adjacency
from uvnpy.network.graphs import DiskGraph
from uvnpy.distances.core import (
    distance_matrix,
    minimum_rigidity_extents,
    minimum_rigidity_radius,
    sufficiently_dispersed_position
)


np.set_printoptions(suppress=True, precision=4, linewidth=250)

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


n = 20
threshold = 1e-5

seed = 15
np.random.seed(seed)

# p = np.random.uniform((0, 0), (1, 0.9), (n, 2))
p = sufficiently_dispersed_position(n, (0, 1), (0, 0.9), 0.1)
A0 = DiskGraph(p, dmax=2/np.sqrt(n)).adjacency_matrix(float)
Rmax = distance_matrix(p).max()
A, Rmin = minimum_rigidity_radius(A0, p, threshold, return_radius=True)

# ------------------------------------------------------------------
# Fig 1
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(2.25, 2.25))
# fig.subplots_adjust(top=0.88, bottom=0.15, wspace=0.28)
h = minimum_rigidity_extents(A, p, threshold)

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
    ax, p, edges_from_adjacency(A, directed=False),
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
fig.savefig('/tmp/minimum_extents_versus_radius_1.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Fig 2
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(2.25, 2.25))
# fig.subplots_adjust(top=0.88, bottom=0.15, wspace=0.28)
A0 = A
A = DiskGraph(p, dmax=Rmin + 0.05 * (Rmax - Rmin)).adjacency_matrix(float)
h = minimum_rigidity_extents(A, p, threshold)

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
    ax, p, edges_from_adjacency(A0, directed=False),
    lw=0.5, color=cm.coolwarm(20), zorder=0
)
plot.bars(
    ax, p, edges_from_adjacency(A - A0, directed=False),
    lw=0.5, ls='--', color=cm.coolwarm(20), zorder=0
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
fig.savefig('/tmp/minimum_extents_versus_radius_2.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Fig 3
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(2.25, 2.25))
# fig.subplots_adjust(top=0.88, bottom=0.15, wspace=0.28)
A0 = A
A = DiskGraph(p, dmax=Rmin + 0.1 * (Rmax - Rmin)).adjacency_matrix(float)
h = minimum_rigidity_extents(A, p, threshold)

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
    ax, p, edges_from_adjacency(A0, directed=False),
    lw=0.5, color=cm.coolwarm(20), zorder=0
)
plot.bars(
    ax, p, edges_from_adjacency(A - A0, directed=False),
    lw=0.5, ls='--', color=cm.coolwarm(20), zorder=0
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
fig.savefig('/tmp/minimum_extents_versus_radius_3.png', format='png', dpi=360)
