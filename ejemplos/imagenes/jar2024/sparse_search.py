#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from uvnpy.rsn import rigidity
from uvnpy.network import core, disk_graph, plot, subsets


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
    p = np.random.uniform((xl, yl), (xh, yh), (1, 2))
    for i in range(1, n):
        found = False
        while not found:
            q = np.random.uniform((xl, yl), (xh, yh), 2)
            dist2 = np.square(p - q).sum(axis=1)
            if np.all(dist2 > radius**2):
                found = True
                p = np.vstack([p, q])

    return p


def metrics(geodesics, extents):
    degree = (geodesics == 1).sum(axis=1).astype(float)
    load = subsets.fast_degree_load_flat(degree, extents, geodesics)
    super_extents = subsets.supergraph_extent(geodesics, extents)
    super_load = subsets.fast_degree_load_flat(
        degree, super_extents, geodesics
    )
    _, L = subsets.kl_graphs(geodesics, extents)
    return load + super_load, L.sum()


def cost(geodesics, extents):
    return sum(metrics(geodesics, extents))


parser = argparse.ArgumentParser(description='')
arg = parser.parse_args()

n = 20
threshold = 1e-5

p = generate_position(n, (0, 1), (0, 0.9), 0.1)
A0 = disk_graph.adjacency(p, dmax=2/np.sqrt(n))
A, Rmin = rigidity.minimum_radius(A0, p, threshold, return_radius=True)

h = rigidity.fast_extents(A, p, threshold)
geo = subsets.geodesics(A)
D = np.max(geo)

h_sparsed = rigidity.sparse_centers_greedy_search(A, p, h, cost, threshold)
print(n, A.sum()//2, core.diameter(A))
print(h, 2*h.max(), metrics(geo, h))
print(h_sparsed, 2*h_sparsed.max(), metrics(geo, h_sparsed))
_, L = subsets.kl_graphs(geo, h_sparsed)

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

plot.edges(
    ax, p, A,
    lw=0.5, color=cm.coolwarm(20), zorder=0)

hops = np.unique(h)
for k in hops:
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
    ncol=4, columnspacing=0.4, loc='upper center')
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
