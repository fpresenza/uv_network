#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on mar jul 20 19:06:45 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches  # noqa


import uvnpy.network as network
from uvnpy.toolkit.calculus import circle2d

fig, ax = plt.subplots(1, 2, figsize=(4.5, 1.125))
fig.subplots_adjust(wspace=0.14)
ax = ax.ravel()
for _ax in ax:
    _ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False)   # labels along the bottom edge are off
    # _ax.grid(1, lw=0.4)
    _ax.set_aspect('equal')

x = np.array([
    [1.25, 1.5],
    [0.625, 0.],
    [1.875, 0.],
    [3.125, 0.],
    [4.375, 0.],
    [5.625, 0.],
    [6.875, 0.],
    [8.125, 0.],
    [9.375, 0.],
    [8.75, -1.5]])


y = x
# np.array([
# [2., 3],
# [1, 1],
# [3, 1],
# [3, -1],
# [5, 1],
# [5, -1],
# [7, 1],
# [7, -1],
# [9, -1],
# [8, -3]])

E = np.array([
    [0, 1],
    [0, 2],
    [1, 2],    # [1, 3],
    [2, 3],    # [2, 4],
    [3, 4],    # [3, 5],
    [4, 5],    # [4, 6],
    [5, 6],    # [5, 7],
    [6, 7],    # [6, 8],
    [7, 8],
    [7, 9],
    [8, 9]], dtype=int)


network.plot.nodes(ax[0], x[:-1], color='skyblue', s=60, zorder=10)
network.plot.nodes(ax[0], x[-1], color='0.7', s=60, zorder=10)
network.plot.edges(ax[0], x, E, color='gray', alpha=0.6, lw=0.8, zorder=1)
network.plot.nodes(ax[1], y[1:], color='orange', s=60, zorder=10)
network.plot.nodes(ax[1], y[0], color='0.7', s=60, zorder=10)
network.plot.edges(ax[1], y, E, color='gray', alpha=0.6, lw=0.8, zorder=1)

# loop through each x,y pair
for i, p in enumerate(x):
    ax[0].annotate(
        i + 1,  xy=p, color='k',
        fontsize='x-small', weight='normal',
        horizontalalignment='center',
        verticalalignment='center', zorder=20)

for i, p in enumerate(y):
    ax[1].annotate(
        i + 1,  xy=p, color='k',
        fontsize='x-small', weight='normal',
        horizontalalignment='center',
        verticalalignment='center', zorder=20)

c = circle2d(R=2.5, n=360)[243:298]
for _ax in ax:
    _ax.plot(
        c[:, 0] + x[2, 0], c[:, 1] + 2.1, ls='--',
        color='gray', alpha=0.8, lw=0.8, zorder=1)
    _ax.plot(
        c[:, 0] + x[3, 0], -c[:, 1] - 2.1, ls='--',
        color='gray', alpha=0.8, lw=0.8, zorder=1)
    _ax.plot(
        c[:, 0] + x[4, 0], c[:, 1] + 2.1, ls='--',
        color='gray', alpha=0.8, lw=0.8, zorder=1)
    _ax.plot(
        c[:, 0] + x[5, 0], -c[:, 1] - 2.1, ls='--',
        color='gray', alpha=0.8, lw=0.8, zorder=1)
    _ax.plot(
        c[:, 0] + x[6, 0], c[:, 1] + 2.1, ls='--',
        color='gray', alpha=0.8, lw=0.8, zorder=1)
    _ax.plot(
        c[:, 0] + x[7, 0], -c[:, 1] - 2.1, ls='--',
        color='gray', alpha=0.8, lw=0.8, zorder=1)

# # specify the location of (left,bottom),width,height
# Nj = patches.Rectangle((-0., -1), 10.5, 4.5,
#                         fill = True,
#                         color = 'skyblue',
#                         alpha=0.25,
#                         linewidth=3)
# ax[0].add_patch(Nj)
# Ni = patches.Rectangle((-0., -3.5), 10.5, 4.5,
#                         fill = True,
#                         color = 'orange',
#                         alpha=0.25,
#                         linewidth=3)
# ax[0].add_patch(Ni)

# ax[0].plot(l1[:, 0], l1[:, 1], ls='--', color='k', lw=0.8)
# l2 = np.vstack([p[2], q[0]])
# ax[0].plot(l2[:, 0], l2[:, 1], ls='--', color='k', lw=0.8)
# l3 = np.vstack([p[8], q[6]])
# ax[0].plot(l3[:, 0], l3[:, 1], ls='--', color='k', lw=0.8)

ax[0].set_xlim(0, 10)
ax[0].set_ylim(-2, 2)

ax[1].set_xlim(0, 10)
ax[1].set_ylim(-2, 2)

plt.show()
fig.savefig('/tmp/colinearity.pdf', format='pdf')
