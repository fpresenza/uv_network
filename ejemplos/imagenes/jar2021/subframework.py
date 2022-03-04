#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on jue sep 16 22:41:17 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx  # noqa

import uvnpy.network as network
from uvnpy.network import disk_graph, subsets

np.random.seed(0)

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'

lim = 11
dmax = 10/1.6

x = np.array([
    [-3.64033641, -1.71474011],
    # [-5.31705007, 5.44944239],
    [1.33202908, -4.69221018],
    [0.46496107, -8.12118978],
    [2.7, 5.7],
    [-3.62862095, 3.3482076],
    [-4.21187814, -6.33617276],
    # [5.7302587, -9.59784908],
    # [7.57880058, -7.90609048],
    [4.55633074, -5.09984054],
    # [4.70388044, 9.2437709],
    [-5.02493713, 1.52314669],
    [2.34083863, 2.44503812],
    # [1.53836735, 10.05498023],
    [-1.05749243, 6.92817345],
    [6.27595639, -2.06988518],
    # [9., 1.62545745],
    [7.63470724, 3.8506318]])
n = len(x)
_x = x.copy()
x[:, 0] = _x[:, 1]
x[:, 1] = -_x[:, 0]

# hops necesarios para rigidez
E = disk_graph.edges(x, dmax)
E = np.delete(E, 7, axis=0)

fig, ax = plt.subplots(figsize=(1.65, 1.65))
plt.setp(ax.spines.values(), color='white')
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False)   # labels along the bottom edge are off
# ax.grid(0, lw=0.4)
# ax.set_aspect('equal')

network.plot.nodes(ax, x, color='lightblue', s=60, zorder=10)
network.plot.nodes(ax, x, color='0.5', facecolors='none', s=60, zorder=10)
network.plot.edges(ax, x, E[[0]], color='0.1', lw=0.7, zorder=1)
network.plot.edges(ax, x, E[1:], color='0.2', alpha=0.6, lw=0.7, zorder=1)

i, j = E[0]
ax.annotate(
    'i', xy=x[i], color='k',
    fontsize='x-small', weight='normal',
    horizontalalignment='center',
    verticalalignment='center', zorder=20)
ax.annotate(
    'j', xy=x[j], color='k',
    fontsize='x-small', weight='normal',
    horizontalalignment='center',
    verticalalignment='center', zorder=20)
ax.annotate(
    r'$e_k = \{i, j\}$', xy=x[[i, j]].mean(axis=0) + (3, 0),
    color='k',
    fontsize='x-small', weight='normal',
    horizontalalignment='center',
    verticalalignment='center', zorder=20)

fig.savefig('/tmp/framework.png', format='png', dpi=300)

# subframework
i = 0
h = 2
A = network.adjacency_from_edges(n, E)
Ai, xi = subsets.multihop_subframework(A, x, i, h)
yi = x[~np.isin(x[:, 0], xi[:, 0])]

fig, ax = plt.subplots(figsize=(1.65, 1.65))
plt.setp(ax.spines.values(), color='white')
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False)   # labels along the bottom edge are off
# ax.grid(0, lw=0.4)
# ax.set_aspect('equal')

network.plot.nodes(ax, yi, color='lightblue', s=60, zorder=10)
network.plot.nodes(ax, yi, color='0.5', facecolors='none', s=60, zorder=10)
network.plot.nodes(ax, xi, color='orange', s=60, zorder=10)
network.plot.edges(ax, x, E, color='0.2', alpha=0.6, lw=0.7, zorder=1)
network.plot.edges(ax, xi, Ai, color='orange', lw=1, zorder=1)

ax.annotate(
    'i', xy=x[i], color='k',
    fontsize='x-small', weight='normal',
    horizontalalignment='center',
    verticalalignment='center', zorder=20)
ax.annotate(
    r'$h_i = 2$', xy=x[[0, 1]].mean(axis=0) + (3, 0),
    color='k',
    fontsize='x-small', weight='normal',
    horizontalalignment='center',
    verticalalignment='center', zorder=20)

fig.savefig('/tmp/subframework.png', format='png', dpi=300)


plt.show()
