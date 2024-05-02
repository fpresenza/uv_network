#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from uvnpy.network import plot
from uvnpy.network.disk_graph import adjacency_from_positions
from uvnpy.network.core import geodesics
from uvnpy.network.load import one_token_for_all, one_token_for_each
from uvnpy.distances.core import (
    minimum_rigidity_extents,
    minimum_rigidity_radius,
    sufficiently_dispersed_position
)
from uvnpy.network.subframeworks import (
    superframework_extents,
    isolated_edges,
    sparse_subframeworks_greedy_search
)


np.set_printoptions(suppress=True, precision=4, linewidth=250)

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


def metrics(geodesics, extents):
    action_load = one_token_for_each(geodesics, extents)
    super_extents = superframework_extents(geodesics, extents)
    state_load = one_token_for_all(geodesics, super_extents)
    n_isolated_edges = len(isolated_edges(geodesics, extents))
    return action_load + state_load, n_isolated_edges


def network_load(geodesics, extents):
    load, edges = metrics(geodesics, extents)
    return load + (1 + 0.5) * edges


parser = argparse.ArgumentParser(description='')
arg = parser.parse_args()

n = 20
threshold = 1e-5

p = sufficiently_dispersed_position(n, (0, 1), (0, 0.9), 0.1)
A0 = adjacency_from_positions(p, dmax=2/np.sqrt(n))
A, Rmin = minimum_rigidity_radius(A0, p, threshold, return_radius=True)

h = minimum_rigidity_extents(A, p, threshold)
G = geodesics(A)
D = np.max(G)

h_sparsed = sparse_subframeworks_greedy_search(G, h, network_load)

L = np.zeros(A.shape)
for i, j in isolated_edges(G, h_sparsed):
    L[i, j] = L[j, i] = 1

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
    lw=0.5, color=cm.coolwarm(20), zorder=0
)

hops = np.unique(h)
for k in hops:
    c = h == k
    print(h[c])
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
    '/tmp/sparse_rigidity_extents_a.png', format='png', dpi=360)


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

hops = np.unique(h_sparsed)
for k in hops:
    c = h_sparsed == k
    v_artist = ax.scatter(
        p[c, 0], p[c, 1],
        marker='o', s=(k+1) * 10,
        c=h_sparsed[c], cmap=cm.coolwarm, vmin=-3, vmax=3,
        label=r'${}$'.format(k)
    )

plot.edges(
    ax, p, A - L,
    lw=0.5,
    color=cm.coolwarm(20), zorder=0)

plot.edges(
    ax, p, L,
    lw=0.5, color=cm.coolwarm(20), ls='--', zorder=0)

ax.legend(
    fontsize='x-small', handlelength=1, labelspacing=0.7,
    borderpad=0.2, handletextpad=0.2, framealpha=1.,
    ncol=4, columnspacing=0.8, loc='upper center')
fig.savefig(
    '/tmp/sparse_rigidity_extents_b.png', format='png', dpi=360)
