#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on dom abr 11 18:34:34 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt


import uvnpy.network as network
from uvnpy.rsn import rigidity

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


x = np.array([
    [2, 0],
    [1, 1],
    [3, 1],
    [1, -1],
    [3, -1],
    [0., 0],
    [4, 0],
    [5, 1],
    [5, -1],
    [6, 0],
    [7, 1],
    [7, -1],
    [8, 0]])

n = len(x)

hops = [r'$i$', 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4]

E = np.array([
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
    [1, 2],
    [1, 3],
    [3, 4],
    [1, 5],
    [2, 6],
    [3, 5],
    [4, 6],
    [4, 8],
    [2, 7],
    [6, 7],
    [6, 8],
    [7, 9],
    [7, 10],
    [8, 9],
    [8, 11],
    [9, 10],
    [9, 11],
    # [10, 11],
    [10, 12],
    [11, 12]], dtype=int)

# definicion sub-frameworks
fig, ax = plt.subplots(figsize=(2.5, 1.125))
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
ax.set_aspect('equal')
ax.set_xlim(-0.6, 8.6)
ax.set_ylim(-1.6, 1.6)
network.plot.nodes(ax, x[0], color='0.7', s=60, zorder=10)
network.plot.nodes(ax, x[1:5], color='skyblue', s=60, zorder=10)
network.plot.nodes(ax, x[5:9], color='orange', s=60, zorder=10)
network.plot.nodes(ax, x[9:12], color='mediumseagreen', s=60, zorder=10)
network.plot.nodes(ax, x[12:], color='lightpink', s=60, zorder=10)

network.plot.edges(ax, x, E[:6], color='skyblue', lw=0.8, zorder=1)
network.plot.edges(ax, x, E[6:14], color='orange', lw=0.8, zorder=1)
network.plot.edges(ax, x, E[14:21], color='mediumseagreen', lw=0.8, zorder=1)
network.plot.edges(ax, x, E[21:], color='lightpink', lw=0.8, zorder=1)

# loop through each x,y pair
for i, xi in enumerate(x):
    ax.annotate(
        hops[i],  xy=xi, color='k',
        fontsize='x-small', weight='normal',
        horizontalalignment='center',
        verticalalignment='center', zorder=20)

fig.savefig('/tmp/subframework.pdf', format='pdf')


# hops necesarios para rigidez
A = network.adjacency_from_edges(n, E)
min_hops = rigidity.minimum_hops(A, x)

fig, ax = plt.subplots(figsize=(2.5, 1.125))
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
ax.set_aspect('equal')
ax.set_xlim(-0.6, 8.6)
ax.set_ylim(-1.6, 1.6)
network.plot.nodes(ax, x[min_hops == 1], color='skyblue', s=60, zorder=10)
network.plot.nodes(ax, x[min_hops == 2], color='orange', s=60, zorder=10)
network.plot.nodes(
    ax, x[min_hops == 3],
    color='mediumseagreen', s=60, zorder=10)
network.plot.nodes(ax, x[min_hops == 4], color='lightcoral', s=60, zorder=10)
network.plot.edges(ax, x, E, color='0.2', alpha=0.6, lw=0.7, zorder=1)

# loop through each x,y pair
for i, xi in enumerate(x):
    ax.annotate(
        min_hops[i],  xy=xi, color='k',
        fontsize='x-small', weight='normal',
        horizontalalignment='center',
        verticalalignment='center', zorder=20)

fig.savefig('/tmp/minimum_hops_1.pdf', format='pdf')


# agregando un enlace
E = np.vstack([E, [10, 11]])
A = network.adjacency_from_edges(n, E)
min_hops = rigidity.minimum_hops(A, x)

fig, ax = plt.subplots(figsize=(2.5, 1.125))
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
ax.set_aspect('equal')
ax.set_xlim(-0.6, 8.6)
ax.set_ylim(-1.6, 1.6)
network.plot.nodes(ax, x[min_hops == 1], color='skyblue', s=60, zorder=10)
network.plot.nodes(ax, x[min_hops == 2], color='orange', s=60, zorder=10)
network.plot.nodes(
    ax, x[min_hops == 3],
    color='mediumseagreen', s=60, zorder=10)
network.plot.nodes(ax, x[min_hops == 4], color='lightcoral', s=60, zorder=10)
network.plot.edges(ax, x, E, color='0.2', alpha=0.6, lw=0.7, zorder=1)

# loop through each x,y pair
for i, xi in enumerate(x):
    ax.annotate(
        min_hops[i],  xy=xi, color='k',
        fontsize='x-small', weight='normal',
        horizontalalignment='center',
        verticalalignment='center', zorder=20)

fig.savefig('/tmp/minimum_hops_2.pdf', format='pdf')
