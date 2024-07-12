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
    is_inf_rigid,
    minimum_rigidity_radius,
    sufficiently_dispersed_position,
)
from uvnpy.network.subframeworks import (
    valid_extents,
    subframework_adjacencies,
    isolated_links,
    sparse_subframeworks_greedy_search,
    sparse_subframeworks_greedy_search_by_expansion,
    sparse_subframeworks_greedy_search_by_reduction
)


np.set_printoptions(suppress=True, precision=4, linewidth=250)

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'
markers = ['o', '^', 'v', 's', 'd', '<', '>']


def valid_ball(subset, adjacency, position, max_diam):
    """A ball is considered valid if:
        it has zero radius
            or
        (it does not exceeds the maximum allowed diameter
            and
        it is infinitesimally rigid)
    """
    if sum(subset) == 1:
        return True

    A = adjacency[:, subset][subset]
    if geodesics(A).max() > max_diam:
        return False

    p = position[subset]
    if not is_inf_rigid(A, p):
        return False

    return True


def weight(s):
    return 5.0 if s == 2 else 1.0


def decomposition_cost(geodesics, extents, weight):
    adj = subframework_adjacencies(geodesics, extents)
    num_links = len(isolated_links(geodesics, extents))
    ball_sum = sum([weight(len(a)) * np.sum(a) / 2.0 for a in adj])
    link_sum = weight(2) * num_links
    return ball_sum + link_sum


def links_adjacency(geodesics, extents):
    A = np.zeros(geodesics.shape)
    for i, j in isolated_links(geodesics, extents):
        A[i, j] = A[j, i] = 1
    return A


parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-s', '--seed',
    default=-1, type=int, help='seed'
)
arg = parser.parse_args()

n = 30
if arg.seed >= 0:
    np.random.seed(arg.seed)

p = sufficiently_dispersed_position(n, (0, 1), (0, 1), 0.1)

A = adjacency_from_positions(p, dmax=2/np.sqrt(n))
A, Rmin = minimum_rigidity_radius(A, p, return_radius=True)

G = geodesics(A)
max_diam = 4
h_valid = valid_extents(G, valid_ball, A, p, max_diam)
print(h_valid)

h_sparsed, count = sparse_subframeworks_greedy_search(
    geodesics=G,
    valid_extents=h_valid,
    metric=decomposition_cost,
    initial_guess=np.zeros(n, dtype=int),
    weight=weight
)
h_sparsed2, count2 = sparse_subframeworks_greedy_search_by_expansion(
    geodesics=G,
    valid_extents=h_valid,
    metric=decomposition_cost,
    weight=weight
)
h_sparsed3, count3 = sparse_subframeworks_greedy_search_by_reduction(
    geodesics=G,
    valid_extents=h_valid,
    metric=decomposition_cost,
    weight=weight
)


print(h_sparsed, decomposition_cost(G, h_sparsed, weight), count)
print(h_sparsed2, decomposition_cost(G, h_sparsed2, weight), count2)
print(h_sparsed3, decomposition_cost(G, h_sparsed3, weight), count3)

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
    for k in reversed(h_valid[i]):
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

leaders = h_sparsed3 > 0
followers = ~ leaders

ax.scatter(
    p[followers, 0], p[followers, 1],
    marker='o', s=10,
    color='0.7', zorder=10,
)

for j, i in enumerate(np.where(leaders)[0]):
    k = h_sparsed3[i]
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

for k in np.unique(h_sparsed3):
    ax.scatter(
        -1, -1,
        marker=markers[k], s=(k + 1) * 10,
        color='k',
        label=r'${}$'.format(k)
    )

Aiso = links_adjacency(G, h_sparsed3)
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
    '/tmp/sparse_rigidity_extents3.png', format='png', dpi=360
)
