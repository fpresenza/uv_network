#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on dom abr 11 18:34:34 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt


import uvnpy.network as network


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


p = np.array([
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

hops = [r'$i$', 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4]

E = np.array([
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
    [1, 2],
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
    [10, 11],
    [10, 12],
    [11, 12]], dtype=int)


network.plot.nodes(ax, p[0], color='0.7', s=60, zorder=10)
# ax.text(
#     p[0, 0] + 0.25, p[0, 1] - 0.2, r'$=i$', color='0.7', fontsize='x-small')
# ax.text(
#     p[5, 0] + 0.25, p[5, 1] - 0.2, r'$=j$',
#     color='orange', fontsize='x-small')
network.plot.nodes(ax, p[1:5], color='skyblue', s=60, zorder=10)
network.plot.nodes(ax, p[5:9], color='orange', s=60, zorder=10)
network.plot.nodes(ax, p[9:12], color='mediumseagreen', s=60, zorder=10)
network.plot.nodes(ax, p[12:], color='lightpink', s=60, zorder=10)

network.plot.edges(ax, p, E[:6], color='skyblue', lw=0.8, zorder=1)
network.plot.edges(ax, p, E[6:14], color='orange', lw=0.8, zorder=1)
network.plot.edges(ax, p, E[14:21], color='mediumseagreen', lw=0.8, zorder=1)
network.plot.edges(ax, p, E[21:], color='lightpink', lw=0.8, zorder=1)

# loop through each x,y pair
for i, pi in enumerate(p):
    ax.annotate(
        hops[i],  xy=pi, color='k',
        fontsize='x-small', weight='normal',
        horizontalalignment='center',
        verticalalignment='center', zorder=20)

plt.show()
fig.savefig('/tmp/subframework.pdf', format='pdf')
