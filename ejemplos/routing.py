#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on jue sep 16 22:41:17 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx  # noqa
from transformations import unit_vector

import uvnpy.network as network
from uvnpy.network import disk_graph, subsets
from uvnpy.rsn import rigidity

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'

np.random.seed(0)

lim = 11
dmax = 10/1.6

# rigid= False
# k = 1
# while not rigid:
#     x = np.random.uniform(-lim, lim, (n, 2))
#     A = disk_graph.adjacency(x, dmax)
#     rigid = rigidity.algebraic_condition(A, x)
#     print(k)
#     k += 1
# print(x)

x = np.array([
    [-3.64033641, -1.71474011],
    [-5.31705007, 5.44944239],
    [1.33202908, -4.69221018],
    [0.46496107, -8.12118978],
    [2.7, 5.7],
    [-3.62862095, 3.3482076],
    [-4.21187814, -6.33617276],
    [5.7302587, -9.59784908],
    [7.57880058, -7.90609048],
    [4.55633074, -5.09984054],
    [4.70388044, 9.2437709],
    [-5.02493713, 1.52314669],
    [2.34083863, 2.44503812],
    [1.53836735, 10.05498023],
    [-1.05749243, 6.92817345],
    [6.27595639, -2.06988518],
    [9., 1.62545745],
    [7.63470724, 3.8506318]])
n = len(x)
_x = x.copy()
x[:, 0] = _x[:, 1]
x[:, 1] = -_x[:, 0]


# hops necesarios para rigidez
A = disk_graph.adjacency(x, dmax)
G = nx.from_numpy_matrix(A)
diam = nx.algorithms.distance_measures.diameter(G)
print(diam)
min_hops = rigidity.minimum_hops(A, x)

fig, ax = plt.subplots(figsize=(1.65, 1.65))
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

network.plot.nodes(ax, x[min_hops == 1], color='skyblue', s=60, zorder=10)
network.plot.nodes(ax, x[min_hops == 2], color='orange', s=60, zorder=10)
network.plot.nodes(
    ax, x[min_hops == 3],
    color='mediumseagreen', s=60, zorder=10)
network.plot.nodes(ax, x[min_hops == 4], color='lightcoral', s=60, zorder=10)
network.plot.edges(ax, x, A, color='0.2', alpha=0.6, lw=0.7, zorder=1)

# loop through each x,y pair
for i, xi in enumerate(x):
    ax.annotate(
        min_hops[i],  xy=xi, color='k',
        fontsize='x-small', weight='normal',
        horizontalalignment='center',
        verticalalignment='center', zorder=20)
fig.savefig('/tmp/routing_1.png', format='png', dpi=300)

# sub-framework i=0
fig, ax = plt.subplots(figsize=(1.65, 1.65))
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

N1 = subsets.multihop_neighborhood(A, 1)
N2 = subsets.multihop_neighborhood(A, 2)
N3 = subsets.multihop_neighborhood(A, 3)
N4 = subsets.multihop_neighborhood(A, 4)

network.plot.nodes(ax, x[0], color='mediumseagreen', s=60, zorder=10)
network.plot.nodes(ax, x[N1[0]], color='mediumseagreen', s=60, zorder=10)
network.plot.nodes(ax, x[N2[0]], color='mediumseagreen', s=60, zorder=10)
network.plot.nodes(ax, x[N3[0]], color='0.6', s=20, zorder=10)
network.plot.nodes(ax, x[N4[0]], color='0.6', s=20, zorder=10)
network.plot.edges(ax, x, A, color='0.2', alpha=0.6, lw=0.7, zorder=1)

ax.annotate(
    r'$i$',  xy=x[0], color='k',
    fontsize='x-small', weight='normal',
    horizontalalignment='center',
    verticalalignment='center', zorder=20)

tags = np.vstack(
    [0, np.argwhere(N1[0]), np.argwhere(N2[0])]).ravel()
print(tags)
# for i, xi in enumerate(x[N1[0]]):
#     ax.annotate(
#         r'$j_{}$'.format(i+1), xy=xi, color='k',
#         fontsize='x-small', weight='normal',
#         horizontalalignment='center',
#         verticalalignment='center', zorder=20)
# for k, xk in enumerate(x[N2[0]][:-1]):
#     ax.annotate(
#         r'$j_{}$'.format(k+i+2), xy=xk, color='k',
#         fontsize='x-small', weight='normal',
#         horizontalalignment='center',
#         verticalalignment='center', zorder=20)
# ax.annotate(
#     r'$j_{10}$', xy=x[N2[0]][-1], color='k',
#     fontsize='x-small', weight='normal',
#     horizontalalignment='center',
#     verticalalignment='center', zorder=20)

# A2, x2 = subsets.multihop_subframework(A, x, i=0, hops=2)
# E2 = network.edges_from_adjacency(A2)
R = np.array([
    [0, tags[3]],
    [tags[3], tags[6]],
    [0, tags[1]],
    [tags[1], tags[7]],
    [tags[1], tags[10]],
    [0, tags[4]],
    [0, tags[2]],
    [tags[4], tags[5]],
    [tags[2], tags[8]],
    [tags[2], tags[9]]])
R = np.vstack([R, np.flip(R, axis=1)])
print(R)
r = x[R[:, 0]] - x[R[:, 1]]
r = r - 0.6 * unit_vector(r, axis=1)
ax.quiver(
    x[R[:, 1], 0], x[R[:, 1], 1], r[:, 0], r[:, 1],
    color='mediumseagreen', angles='xy',
    scale_units='xy', scale=1, headwidth=6,
    headlength=6, headaxislength=5, linewidths=0.25,
    edgecolor='mediumseagreen')
fig.savefig('/tmp/routing_3.png', format='png', dpi=300)

# membership i=0
fig, ax = plt.subplots(figsize=(1.65, 1.65))
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

M = N1[0]
M[0] = True
M[15] = True
network.plot.nodes(ax, x[M], color='lightcoral', s=60, zorder=10)
network.plot.nodes(ax, x[np.logical_not(M)], color='0.6', s=20, zorder=10)
network.plot.edges(ax, x, A, color='0.2', alpha=0.6, lw=0.7, zorder=1)

ax.annotate(
    r'$i$',  xy=x[0], color='k',
    fontsize='x-small', weight='normal',
    horizontalalignment='center',
    verticalalignment='center', zorder=20)
# ax.annotate(
#     r'$j_1$', xy=x[2], color='k',
#     fontsize='x-small', weight='normal',
#     horizontalalignment='center',
#     verticalalignment='center', zorder=20)
# ax.annotate(
#     r'$j_2$', xy=x[5], color='k',
#     fontsize='x-small', weight='normal',
#     horizontalalignment='center',
#     verticalalignment='center', zorder=20)
# ax.annotate(
#     r'$j_3$', xy=x[6], color='k',
#     fontsize='x-small', weight='normal',
#     horizontalalignment='center',
#     verticalalignment='center', zorder=20)
# ax.annotate(
#     r'$j_4$', xy=x[11], color='k',
#     fontsize='x-small', weight='normal',
#     horizontalalignment='center',
#     verticalalignment='center', zorder=20)
# ax.annotate(
#     r'$j_{10}$', xy=x[15], color='k',
#     fontsize='x-small', weight='normal',
#     horizontalalignment='center',
#     verticalalignment='center', zorder=20)
E = np.array([
    [0, 2],
    [0, 5],
    [0, 6],
    [0, 11],
    [2, 15]])
E = np.vstack([E, np.flip(E, axis=1)])
r = x[E[:, 0]] - x[E[:, 1]]
r = r - 0.65 * unit_vector(r, axis=1)
ax.quiver(
    x[E[:, 1], 0], x[E[:, 1], 1], r[:, 0], r[:, 1],
    color='lightcoral', angles='xy', scale_units='xy', scale=1, headwidth=6,
    headlength=6, headaxislength=5, linewidths=0.25, edgecolor='lightcoral')
fig.savefig('/tmp/routing_2.png', format='png', dpi=300)

plt.show()
