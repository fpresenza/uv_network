#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import argparse
import numpy as np
from numba import njit
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

from uvnpy.toolkit import plot
from uvnpy.graphs.models import DiskGraph
from uvnpy.graphs.core import (
    geodesics,
    edges_from_adjacency,
    adjacency_matrix_from_edges,
    edge_set_diff
)
from uvnpy.distances.core import (
    is_inf_distance_rigid,
    minimum_distance_rigidity_radius,
    sufficiently_dispersed_position,
)
from uvnpy.graphs.subframeworks import (
    valid_extents,
    isolated_links,
    sparse_subframeworks_greedy_search_by_expansion,
)


np.set_printoptions(suppress=True, precision=4, linewidth=250)

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'
markers = ['o', '^', 'v', 's', 'd', '<', '>']


def valid_ball(subset, adjacency, position):
    """
        A ball is considered valid if:
        it has zero radius
            or
        it is infinitesimally rigid
    """
    if sum(subset) == 1:
        return True

    A = adjacency[:, subset][subset]
    p = position[subset]
    E = edges_from_adjacency(A, directed=False)
    if is_inf_distance_rigid(E, p):
        return True

    return False


@njit
def decomposition_cost(extents, geodesics):
    """
    Computes the set of isolated links (edges not in any subframework).
    """
    n = len(extents)
    s = 0

    for i in range(n):
        for j in range(i + 1, n):
            if geodesics[i, j] == 1:
                in_ball = (geodesics[i] <= extents) * (geodesics[j] <= extents)
                # 1.0 if s > 2 else 5.0
                c = sum(in_ball)
                s += float(c) if c != 0 else 5.0
    return s


parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-s', '--seed',
    default=-1, type=int, help='seed'
)
arg = parser.parse_args()

n = 30
if arg.seed >= 0:
    np.random.seed(arg.seed)

p = sufficiently_dispersed_position(n, (0.0, 500.0), (0.0, 500.0), 30.0)

E0 = DiskGraph(p, dmax=2/np.sqrt(n)).edge_set(directed=False)
E = minimum_distance_rigidity_radius(E0, p)
A = adjacency_matrix_from_edges(n, E, directed=False).astype(float)

G = geodesics(A)
print("Graph diameter: {}".format(G.max()))
max_extent = 2
h_valid = valid_extents(
    G, condition=valid_ball, max_extent=max_extent,  args=(A, p)
)
print(h_valid)

h_sparsed = sparse_subframeworks_greedy_search_by_expansion(
    valid_extents=h_valid,
    metric=decomposition_cost,
    geodesics=G,
)

h_sparsed_dece = np.empty(n, dtype=int)
for i in range(n):
    S = G[i] <= 2 * max_extent
    Ai = A[:, S][S]
    pi = p[S]
    Gi = geodesics(Ai)
    h_valid_i = valid_extents(
        Gi, condition=valid_ball, max_extent=max_extent, args=(Ai, pi)
    )
    h_sparsed_i = sparse_subframeworks_greedy_search_by_expansion(
        valid_extents=h_valid_i,
        metric=decomposition_cost,
        geodesics=Gi,
    )
    idx = sum(S[:i])
    h_sparsed_dece[i] = h_sparsed_i[idx]

print(h_sparsed, decomposition_cost(h_sparsed, G))
print(h_sparsed_dece, decomposition_cost(h_sparsed_dece, G))

fig, ax = plt.subplots(figsize=(2.25, 2.25))
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,
    left=False,
    pad=1,
    labelsize='x-small'
)
ax.set_aspect('equal')
ax.set_xlim(0.0, 500.0)
ax.set_ylim(0.0, 500.0)
ax.set_xticklabels([])
ax.set_yticklabels([])

plot.bars(
    ax, p, E,
    lw=0.3, color='k', alpha=0.6, zorder=0
)

for i in range(n):
    for k in reversed(h_valid[i]):
        ax.scatter(
            p[i, 0], p[i, 1],
            marker='o', s=5*(k+1)**2,
            color='C{}'.format(k) if (k > 0) else 'k',
        )

fig.savefig(
    '/tmp/min_rigidity_extents.png', format='png', dpi=360
)

fig, ax = plt.subplots(figsize=(2.25, 2.25))
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
ax.set_xlim(0.0, 500.0)
ax.set_ylim(0.0, 500.0)
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
    )

for k in np.unique(h_sparsed):
    ax.scatter(
        -1, -1,
        marker=markers[k], s=(k + 1) * 10,
        color='k',
        label=r'${}$'.format(k)
    )

Eiso = isolated_links(G, h_sparsed)
print(E, Eiso)
plot.bars(
    ax, p, edge_set_diff(E, Eiso) if Eiso != [] else E,
    lw=0.3, color='k', alpha=0.6, zorder=0
)

plot.bars(
    ax, p, Eiso,
    lw=0.3, color='k', alpha=0.6, zorder=0
)

ax.legend(
    fontsize='xx-small', handlelength=1, labelspacing=0.7,
    borderpad=0.2, handletextpad=0.2, framealpha=1.,
    ncol=5, columnspacing=0.8, loc='upper center'
)
fig.savefig(
    '/tmp/sparse_rigidity_extents.png', format='png', dpi=360
)

fig, ax = plt.subplots(figsize=(2.25, 2.25))
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,
    left=False,
    pad=1,
    labelsize='x-small'
)
ax.set_aspect('equal')
ax.set_xlim(0.0, 500.0)
ax.set_ylim(0.0, 500.0)
ax.set_xticklabels([])
ax.set_yticklabels([])

leaders = h_sparsed_dece > 0
followers = ~ leaders

ax.scatter(
    p[followers, 0], p[followers, 1],
    marker='o', s=10,
    color='0.7', zorder=10,
)

for j, i in enumerate(np.where(leaders)[0]):
    k = h_sparsed_dece[i]
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
    )

for k in np.unique(h_sparsed_dece):
    ax.scatter(
        -1, -1,
        marker=markers[k], s=(k + 1) * 10,
        color='k',
        label=r'${}$'.format(k)
    )

Eiso = isolated_links(G, h_sparsed_dece)
plot.bars(
    ax, p, edge_set_diff(E, Eiso) if Eiso != [] else E,
    lw=0.3, color='k', alpha=0.6, zorder=0
)

plot.bars(
    ax, p, Eiso,
    lw=0.3, color='k', alpha=0.6, zorder=0
)

ax.legend(
    fontsize='x-small', handlelength=1, labelspacing=0.7,
    borderpad=0.2, handletextpad=0.2, framealpha=1.,
    ncol=5, columnspacing=0.8, loc='upper center'
)
fig.savefig(
    '/tmp/sparse_rigidity_extents_dece.png', format='png', dpi=360
)
