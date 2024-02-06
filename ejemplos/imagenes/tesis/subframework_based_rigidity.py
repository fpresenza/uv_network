#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from uvnpy.rsn.rigidity import fast_extents, minimum_radius
from uvnpy.rsn.distances import matrix as distance_matrix
from uvnpy.rsn.distances import matrix_between as distance_between
from uvnpy.network.subsets import multihop_subsets
from uvnpy.network.disk_graph import adjacency as disk_adjacency
from uvnpy.network import plot

np.set_printoptions(suppress=True, precision=4, linewidth=250)

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


def generate_position(n, xlim, ylim, radius):
    """Genera una posicion donde los nodos estan a mas
    de cierto radio de separacion"""
    xl, xh = xlim
    yl, yh = ylim
    p = np.random.uniform((xl, yl), (xh, yh), 2)
    for i in range(1, n):
        found = False
        while not found:
            q = np.random.uniform((xl, yl), (xh, yh), 2)
            dist = distance_between(p, q)
            if np.all(dist > radius):
                found = True
                p = np.vstack([p, q])

    return p


n = 60
nodes = np.arange(n)
threshold = 1e-5

seed = 1
print('seed = {}'.format(seed))
np.random.seed(seed)

# p = np.random.uniform((0, 0), (1, 0.9), (n, 2))
p = generate_position(n, (0, 1), (0, 1), 0.1)
midpoint = np.mean(p, axis=0)
# print(p)
A0 = disk_adjacency(p, dmax=2/np.sqrt(n))
A, Rmin = minimum_radius(A0, p, threshold, return_radius=True)
dist = distance_matrix(p)
Rmax = dist.max()

# A = disk_adjacency(p, dmax=Rmin + 0.025 * (Rmax - Rmin))
h = fast_extents(A, p, threshold)
i = np.argmin(np.linalg.norm(p - midpoint, axis=1))
print('i = {}'.format(i))
graph = nx.from_numpy_array(A)
geo = nx.shortest_path_length(graph, source=i)
centers = np.full(n, False)
for j in geo.keys():
    if geo[j] <= h[j]:
        centers[j] = True

# PARTE 1
fig, ax = plt.subplots(figsize=(3.5, 3.5))
# fig.subplots_adjust(top=0.88, bottom=0.15, wspace=0.28)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,
    left=False,
    pad=1,
    labelsize='x-small')
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
for ext in np.unique(h):
    plot.nodes(
        ax, p[h == ext],
        s=11, zorder=10, label=r'$h_0={}$'.format(ext))
plot.edges(ax, p, A, color='0.0', lw=0.4, alpha=0.5)
ax.legend(
    fontsize='x-small', handlelength=1, labelspacing=0.4,
    borderpad=0.2, handletextpad=0.2, framealpha=1.,
    ncol=3, columnspacing=0.2, loc='upper center')
fig.savefig('/tmp/subframework_rigidity_extents.png', format='png', dpi=360)

# PARTE 2
fig, ax = plt.subplots(figsize=(3.5, 3.5))
# fig.subplots_adjust(top=0.88, bottom=0.15, wspace=0.28)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,
    left=False,
    pad=1,
    labelsize='x-small')
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
    r'$i$',  xy=p[i], color='k',
    fontsize='small', weight='normal',
    horizontalalignment='center',
    verticalalignment='center', zorder=20)
plot.nodes(
    ax, p[np.logical_not(centers)],
    marker='o', color='gray', s=11, zorder=10)
plot.nodes(
    ax, p[centers],
    marker='o', color='lightcoral', s=30, zorder=10,
    label=r'$j : \; i \in \mathcal{F}_j$')
plot.nodes(
    ax, p[i],
    marker='o', color='lightcoral', s=70, zorder=10)
plot.edges(ax, p, A, color='0.0', lw=0.4, alpha=0.5)
ax.legend(
    fontsize='small', handlelength=1, labelspacing=0.4,
    borderpad=0.2, handletextpad=0.2, framealpha=1.,
    ncol=3, columnspacing=0.2, loc='upper center')
fig.savefig('/tmp/subframework_rigidity_inclusion.png', format='png', dpi=360)

# PARTE 3
sub = multihop_subsets(A, np.where(centers)[0], h[centers])
info = np.any(sub, axis=0)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
# fig.subplots_adjust(top=0.88, bottom=0.15, wspace=0.28)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,
    left=False,
    pad=1,
    labelsize='x-small')
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
    r'$i$',  xy=p[i], color='k',
    fontsize='small', weight='normal',
    horizontalalignment='center',
    verticalalignment='center', zorder=20)
plot.nodes(
    ax, p[np.logical_not(info)],
    marker='o', color='gray', s=11, zorder=10)
plot.nodes(
    ax, p[info],
    marker='o', color='mediumseagreen', s=30, zorder=10,
    label=r'$\cup \{\mathcal{F}_j : \; i \in \mathcal{F}_j\}$')
plot.nodes(
    ax, p[i],
    marker='o', color='mediumseagreen', s=70, zorder=10)
plot.edges(ax, p, A, color='0.0', lw=0.4, alpha=0.5)
ax.legend(
    fontsize='small', handlelength=1, labelspacing=0.4,
    borderpad=0.2, handletextpad=0.2, framealpha=1.,
    ncol=3, columnspacing=0.2, loc='upper center')
fig.savefig('/tmp/subframework_rigidity_info.png', format='png', dpi=360)

# PARTE 4
fig, ax = plt.subplots(figsize=(3.5, 3.5))
# fig.subplots_adjust(top=0.88, bottom=0.15, wspace=0.28)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,
    left=False,
    pad=1,
    labelsize='x-small')
ax.grid(1, lw=0.4)
ax.set_aspect('equal')
# ax.set_xlabel(r'$x$', fontsize='small', labelpad=0.6)
# ax.set_ylabel(r'$y$', fontsize='small', labelpad=0)
ax.set_xticks(np.linspace(0, 1, 4, endpoint=True))
ax.set_yticks(np.linspace(0, 1, 4, endpoint=True))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.annotate(
    r'$i$',  xy=p[i], color='k',
    fontsize='small', weight='normal',
    horizontalalignment='center',
    verticalalignment='center', zorder=20)
plot.nodes(
    ax, p[np.logical_not(info)],
    marker='o', color='gray', s=11, zorder=10)
plot.nodes(
    ax, p[info],
    marker='o', color='mediumseagreen', s=30, zorder=10,
    label=r'$\cup \{\mathcal{F}_j : \; j \in \mathcal{C}_i\}$')
centers_not_i = centers.copy()
centers_not_i[i] = False
plot.nodes(
    ax, p[centers_not_i],
    marker='o', color='mediumseagreen', edgecolor='k', s=100, zorder=10)
plot.nodes(
    ax, p[i],
    marker='o', color='mediumseagreen', s=70, zorder=10)
for k, j in enumerate(np.where(centers)[0]):
    if j != i:
        ax.annotate(
            r'$j_{}$'.format(k+1),  xy=p[j], color='k',
            fontsize='small', weight='normal',
            horizontalalignment='center',
            verticalalignment='center', zorder=20)
        for v in np.where(sub[k])[0]:
            r = 0.92 * (p[j] - p[v])
            ax.quiver(
                p[v, 0], p[v, 1], r[0], r[1],
                color='mediumseagreen', angles='xy',
                scale_units='xy', scale=1, headwidth=3,
                headlength=5, headaxislength=4.5, linewidths=0.15,
                edgecolor='mediumseagreen', zorder=5)
plot.edges(ax, p, A, color='0.0', lw=0.4, alpha=0.5)
ax.set_xlim(0.15, 0.85)
ax.set_ylim(0.15, 0.85)
fig.savefig('/tmp/subframework_rigidity_routing_1.png', format='png', dpi=360)

# PARTE 5
fig, ax = plt.subplots(figsize=(7, 3.5))
# fig.subplots_adjust(top=0.88, bottom=0.15, wspace=0.28)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,
    left=False,
    pad=1,
    labelsize='x-small')
# ax.grid(1, lw=0.4)
ax.set_aspect('equal')
ax.axis('off')
# ax.set_xlabel(r'$x$', fontsize='small', labelpad=0.6)
# ax.set_ylabel(r'$y$', fontsize='small', labelpad=0)
ax.set_xticks(np.linspace(0, 1, 4, endpoint=True))
ax.set_yticks(np.linspace(0, 1, 4, endpoint=True))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.annotate(
    r'$i$',  xy=p[i], color='k',
    fontsize='small', weight='normal',
    horizontalalignment='center',
    verticalalignment='center', zorder=20)
plot.nodes(
    ax, p[np.logical_not(info)],
    marker='o', color='gray', s=11, zorder=10)
plot.nodes(
    ax, p[info],
    marker='o', color='mediumseagreen', s=30, zorder=10,
    label=r'$\cup \{\mathcal{F}_j : \; j \in \mathcal{C}_i\}$')
centers_not_i = centers.copy()
centers_not_i[i] = False
plot.nodes(
    ax, p[centers_not_i],
    marker='o', color='mediumseagreen', edgecolor='k', s=100, zorder=10)
plot.nodes(
    ax, p[i],
    marker='o', color='mediumseagreen', s=70, zorder=10)
for k, j in enumerate(np.where(centers)[0]):
    if j != i:
        ax.annotate(
            r'$j_{}$'.format(k+1),  xy=p[j], color='k',
            fontsize='x-small', weight='normal',
            horizontalalignment='center',
            verticalalignment='center', zorder=20)
        r = - 0.92 * (p[j] - p[i])
        ax.quiver(
            p[j, 0], p[j, 1], r[0], r[1],
            color='mediumseagreen', angles='xy',
            scale_units='xy', scale=1, headwidth=3,
            headlength=5, headaxislength=4.5, linewidths=0.15,
            edgecolor='mediumseagreen', zorder=5)
plot.edges(ax, p, A, color='0.0', lw=0.4, alpha=0.5)
ax.set_xlim(0.15 - 0.3, 0.85 + 0.3)
ax.set_ylim(0.15 - 0.3, 0.85 + 0.3)
fig.savefig('/tmp/subframework_rigidity_routing_2.png', format='png', dpi=360)

# plt.show()
