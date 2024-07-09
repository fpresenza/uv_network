#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import argparse
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

from uvnpy.network import plot
from uvnpy.network.disk_graph import adjacency_from_positions
from uvnpy.network.core import geodesics
from uvnpy.distances.core import (
    minimum_rigidity_radius,
    rigidity_extents,
    sufficiently_dispersed_position,
)
from uvnpy.network.subframeworks import (
    subframework_vertices,
    subframework_adjacencies,
    links,
    sparse_subframeworks_extended_greedy_search
)


np.set_printoptions(suppress=True, precision=4, linewidth=250)

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'
markers = ['o', '^', 'v', 's', 'd', '<', '>']


def num_repeated_edges(geodesics, extents):
    adjacency = subframework_adjacencies(geodesics, extents)
    return np.sum([np.sum(adj) for adj in adjacency]) / 2


def edge_freedom(geodesics, extents):
    S = subframework_vertices(geodesics, extents)
    n = np.sum(S, axis=1)
    return np.sum([(k-2)*(k-3)/2 for k in n if k > 1])


def decomposition_cost(geodesics, extents):
    num_rep_edges = num_repeated_edges(geodesics, extents)
    num_links = len(links(geodesics, extents))
    return num_rep_edges + 5 * num_links


def decomposition_cost2(geodesics, extents):
    num_rep_edges = num_repeated_edges(geodesics, extents)
    num_links = len(links(geodesics, extents))
    freedom = edge_freedom(geodesics, extents)
    return num_rep_edges + 5 * num_links - 0.5 * freedom


def decomposition_cost3(geodesics, extents, weight):
    _adj = subframework_adjacencies(geodesics, extents)
    _links = links(geodesics, extents)
    ball_sum = sum([weight(len(a)) * np.sum(a) / 2.0 for a in _adj])
    link_sum = weight(2) * len(_links)
    return ball_sum + link_sum


def links_adjacency(geodesics, extents):
    A = np.zeros(geodesics.shape)
    for i, j in links(geodesics, extents):
        A[i, j] = A[j, i] = 1
    return A


def weight(s):
    return (s)**(-1.5)


parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-s', '--seed',
    default=-1, type=int, help='seed'
)
arg = parser.parse_args()

n = 50
if arg.seed >= 0:
    np.random.seed(arg.seed)

p = sufficiently_dispersed_position(n, (0, 1), (0, 1), 0.1)
print(p)

A = adjacency_from_positions(p, dmax=2/np.sqrt(n))
A, Rmin = minimum_rigidity_radius(A, p, return_radius=True)

G = geodesics(A)
max_hops = 2
h_extended = rigidity_extents(G, p, max_hops)
print(h_extended)

h_sparsed = sparse_subframeworks_extended_greedy_search(
    G, h_extended, decomposition_cost
)
h_sparsed2 = sparse_subframeworks_extended_greedy_search(
    G, h_extended, decomposition_cost3, weight
)

print(h_sparsed, decomposition_cost(G, h_sparsed))
print(h_sparsed2, decomposition_cost3(G, h_sparsed2, weight))

fig, ax = plt.subplots(figsize=(2.25, 2.25))
# fig.subplots_adjust(top=0.88, bottom=0.15, wspace=0.28)
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
# ax.set_xticks(np.linspace(0, 1, 4, endpoint=True))
# ax.set_yticks(np.linspace(0, 1, 4, endpoint=True))
ax.set_xticklabels([])
ax.set_yticklabels([])

plot.edges(
    ax, p, A,
    lw=0.3, color='k', alpha=0.6, zorder=0
)

for i in range(n):
    for k in reversed(h_extended[i]):
        ax.scatter(
            p[i, 0], p[i, 1],
            marker='o', s=5*(k+1)**2,
            color='C{}'.format(k) if (k > 0) else 'k',
            # label=r'${}$'.format(k)
        )

# ax.legend(
#     fontsize='x-small', handlelength=1, labelspacing=1.5,
#     borderpad=0.2, handletextpad=0.2, framealpha=1.,
#     ncol=5, columnspacing=0.4, loc='upper center'
# )
fig.savefig(
    '/tmp/min_rigidity_extents.png', format='png', dpi=360
)


fig, ax = plt.subplots(figsize=(2.25, 2.25))
# fig.subplots_adjust(top=0.88, bottom=0.15, wspace=0.28)
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
# ax.set_xticks(np.linspace(0, 1, 4, endpoint=True))
# ax.set_yticks(np.linspace(0, 1, 4, endpoint=True))
ax.set_xticklabels([])
ax.set_yticklabels([])

leaders = h_sparsed > 0
followers = ~ leaders

ax.scatter(
    p[followers, 0], p[followers, 1],
    marker='o', s=10,
    color='0.7', zorder=10,
)

for j, i in enumerate(np.where(leaders)[0]):
    k = h_sparsed[i]
    q = p[G[i] <= k]
    cvx = ConvexHull(q)
    ax.fill(
        q[cvx.vertices, 0], q[cvx.vertices, 1],
        color='C{}'.format(j), alpha=0.15
    )
    ax.scatter(
        p[i, 0], p[i, 1],
        marker=markers[k], s=(k + 1) * 10,
        color='C{}'.format(j), zorder=10,
        # label=r'${}$'.format(k)
    )

for k in np.unique(h_sparsed):
    ax.scatter(
        -1, -1,
        marker=markers[k], s=(k + 1) * 10,
        color='k',
        label=r'${}$'.format(k)
    )

Aiso = links_adjacency(G, h_sparsed)
plot.edges(
    ax, p, A - Aiso,
    lw=0.3, color='k', alpha=0.6, zorder=0
)

plot.edges(
    ax, p, Aiso,
    lw=1.75, color='k', alpha=0.25, zorder=0
)

ax.legend(
    fontsize='x-small', handlelength=1, labelspacing=0.7,
    borderpad=0.2, handletextpad=0.2, framealpha=1.,
    ncol=5, columnspacing=0.8, loc='upper center'
)
fig.savefig(
    '/tmp/sparse_rigidity_extents.png', format='png', dpi=360
)


fig, ax = plt.subplots(figsize=(2.25, 2.25))
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
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
# ax.set_xlabel(r'$\mathrm{x}$', fontsize='x-small', labelpad=0.6)
# ax.set_ylabel(r'$\mathrm{y}$', fontsize='x-small', labelpad=0)
# ax.set_xticks(np.linspace(0, 1, 4, endpoint=True))
# ax.set_yticks(np.linspace(0, 1, 4, endpoint=True))
ax.set_xticklabels([])
ax.set_yticklabels([])

leaders = h_sparsed2 > 0
followers = ~ leaders

ax.scatter(
    p[followers, 0], p[followers, 1],
    marker='o', s=10,
    color='0.7', zorder=10,
)

for j, i in enumerate(np.where(leaders)[0]):
    k = h_sparsed2[i]
    q = p[G[i] <= k]
    cvx = ConvexHull(q)
    ax.fill(
        q[cvx.vertices, 0], q[cvx.vertices, 1],
        color='C{}'.format(j), alpha=0.15
    )
    ax.scatter(
        p[i, 0], p[i, 1],
        marker=markers[k], s=(k + 1) * 10,
        color='C{}'.format(j), zorder=10,
        # label=r'${}$'.format(k)
    )

for k in np.unique(h_sparsed2):
    ax.scatter(
        -1, -1,
        marker=markers[k], s=(k + 1) * 10,
        color='k',
        label=r'${}$'.format(k)
    )

Aiso = links_adjacency(G, h_sparsed2)
plot.edges(
    ax, p, A - Aiso,
    lw=0.3, color='k', alpha=0.6, zorder=0
)

plot.edges(
    ax, p, Aiso,
    lw=1.75, color='k', alpha=0.25, zorder=0
)

ax.legend(
    fontsize='x-small', handlelength=1, labelspacing=0.7,
    borderpad=0.2, handletextpad=0.2, framealpha=1.,
    ncol=5, columnspacing=0.8, loc='upper center'
)
fig.savefig(
    '/tmp/sparse_rigidity_extents2.png', format='png', dpi=360
)
