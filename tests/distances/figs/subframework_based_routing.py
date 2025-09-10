#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from uvnpy.toolkit import plot
from uvnpy.graphs.models import DiskGraph
from uvnpy.graphs.core import geodesics, adjacency_matrix_from_edges
from uvnpy.distances.core import (
    distance_matrix,
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


n = 60
nodes = np.arange(n)
threshold = 1e-5

seed = 1
print('seed = {}'.format(seed))
np.random.seed(seed)

# p = np.random.uniform((0, 0), (1, 0.9), (n, 2))
p = sufficiently_dispersed_position(n, (0, 1), (0, 1), 0.1)
midpoint = np.mean(p, axis=0)
# print(p)
E0 = DiskGraph(p, dmax=2/np.sqrt(n)).edge_set(directed=False)
E = minimum_distance_rigidity_radius(E0, p, threshold)
A = adjacency_matrix_from_edges(n, E, directed=False).astype(float)
dist = distance_matrix(p)
Rmax = dist.max()

G = geodesics(A)
h = minimum_distance_rigidity_extents(G, p, threshold)
node_i = np.argmin(np.linalg.norm(p - midpoint, axis=1))
print('i = {}'.format(node_i))
super_centers = G[node_i] <= h

# ------------------------------------------------------------------
# Fig 1
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.5, 3.5))
# fig.subplots_adjust(top=0.88, bottom=0.15, wspace=0.28)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,
    left=False,
    pad=1,
    labelsize='x-small'
)
ax.grid(1, lw=0.4)
ax.set_aspect('equal')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.1)
# ax.set_xlabel(r'$x$', fontsize='small', labelpad=0.6)
# ax.set_ylabel(r'$y$', fontsize='small', labelpad=0)
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
fig.savefig('/tmp/subframework_based_routing_1.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Fig 2
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.5, 3.5))
# fig.subplots_adjust(top=0.88, bottom=0.15, wspace=0.28)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,
    left=False,
    pad=1,
    labelsize='x-small'
)
ax.grid(1, lw=0.4)
ax.set_aspect('equal')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.1)
# ax.set_xlabel(r'$x$', fontsize='small', labelpad=0.6)
# ax.set_ylabel(r'$y$', fontsize='small', labelpad=0)
ax.set_xticks(np.linspace(0, 1, 4, endpoint=True))
ax.set_yticks(np.linspace(0, 1, 4, endpoint=True))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.annotate(
    r'$i$',  xy=p[node_i], color='k',
    fontsize='small', weight='normal',
    horizontalalignment='center',
    verticalalignment='center', zorder=20
)
plot.points(
    ax, p[np.logical_not(super_centers)],
    marker='o', color='gray', s=11, zorder=10
)
plot.points(
    ax, p[super_centers],
    marker='o', color='lightblue', s=30, zorder=10,
    label=r'$j : \; i \in \mathcal{F}_j$'
)
plot.points(
    ax, p[node_i],
    marker='o', color='lightblue', s=70, zorder=10
)
plot.bars(
    ax, p, E,
    color='0.0', lw=0.4, alpha=0.5
)
ax.legend(
    fontsize='small', handlelength=1, labelspacing=0.4,
    borderpad=0.2, handletextpad=0.2, framealpha=1.,
    ncol=3, columnspacing=0.2, loc='upper center'
)
fig.savefig('/tmp/subframework_based_routing_2.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Fig 3
# ------------------------------------------------------------------
info = np.full(n, False)
for j in range(n):
    for c in np.where(super_centers)[0]:
        if G[j, c] <= h[c]:
            info[j] = True

fig, ax = plt.subplots(figsize=(3.5, 3.5))
# fig.subplots_adjust(top=0.88, bottom=0.15, wspace=0.28)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,
    left=False,
    pad=1,
    labelsize='x-small'
)
ax.grid(1, lw=0.4)
ax.set_aspect('equal')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.1)
# ax.set_xlabel(r'$x$', fontsize='small', labelpad=0.6)
# ax.set_ylabel(r'$y$', fontsize='small', labelpad=0)
ax.set_xticks(np.linspace(0, 1, 4, endpoint=True))
ax.set_yticks(np.linspace(0, 1, 4, endpoint=True))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.annotate(
    r'$i$',  xy=p[node_i], color='k',
    fontsize='small', weight='normal',
    horizontalalignment='center',
    verticalalignment='center', zorder=20
)
plot.points(
    ax, p[np.logical_not(info)],
    marker='o', color='gray', s=11, zorder=10
)
plot.points(
    ax, p[info],
    marker='o', color='mediumseagreen', s=30, zorder=10,
    label=r'$\cup \{\mathcal{F}_j : \; i \in \mathcal{F}_j\}$'
)
plot.points(
    ax, p[node_i],
    marker='o', color='mediumseagreen', s=70, zorder=10
)
plot.bars(
    ax, p, E,
    color='0.0', lw=0.4, alpha=0.5
)
ax.legend(
    fontsize='small', handlelength=1, labelspacing=0.4,
    borderpad=0.2, handletextpad=0.2, framealpha=1.,
    ncol=3, columnspacing=0.2, loc='upper center'
)
fig.savefig('/tmp/subframework_based_routing_3.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Fig 4
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.5, 3.5))
# fig.subplots_adjust(top=0.88, bottom=0.15, wspace=0.28)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,
    left=False,
    pad=1,
    labelsize='x-small'
)
ax.grid(1, lw=0.4)
ax.set_aspect('equal')
# ax.set_xlabel(r'$x$', fontsize='small', labelpad=0.6)
# ax.set_ylabel(r'$y$', fontsize='small', labelpad=0)
ax.set_xticks(np.linspace(0, 1, 4, endpoint=True))
ax.set_yticks(np.linspace(0, 1, 4, endpoint=True))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.annotate(
    r'$i$',  xy=p[node_i], color='k',
    fontsize='small', weight='normal',
    horizontalalignment='center',
    verticalalignment='center', zorder=20
)
plot.points(
    ax, p[np.logical_not(info)],
    marker='o', color='gray', s=11, zorder=10
)
plot.points(
    ax, p[info],
    marker='o', color='mediumseagreen', s=30, zorder=10,
    label=r'$\cup \{\mathcal{F}_j : \; j \in \mathcal{C}_i\}$'
)
centers_not_i = super_centers.copy()
centers_not_i[node_i] = False
plot.points(
    ax, p[centers_not_i],
    marker='o', color='mediumseagreen', edgecolor='k', s=100, zorder=10
)
plot.points(
    ax, p[node_i],
    marker='o', color='mediumseagreen', s=70, zorder=10
)
for k, c in enumerate(np.where(super_centers)[0]):
    if c != node_i:
        ax.annotate(
            r'$j_{}$'.format(k+1),  xy=p[c], color='k',
            fontsize='small', weight='normal',
            horizontalalignment='center',
            verticalalignment='center', zorder=20
        )
        for v in np.where(G[c] <= h[c])[0]:
            r = 0.92 * (p[c] - p[v])
            ax.quiver(
                p[v, 0], p[v, 1], r[0], r[1],
                color='mediumseagreen', angles='xy',
                scale_units='xy', scale=1, headwidth=3,
                headlength=5, headaxislength=4.5, linewidths=0.15,
                edgecolor='mediumseagreen', zorder=5
            )
plot.bars(
    ax, p, E,
    color='0.0', lw=0.4, alpha=0.5
)
ax.set_xlim(0.15, 0.85)
ax.set_ylim(0.15, 0.85)
fig.savefig('/tmp/subframework_based_routing_4.png', format='png', dpi=360)

# plt.show()
