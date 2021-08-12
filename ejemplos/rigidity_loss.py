#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on mar jul 20 19:06:45 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt


import uvnpy.network as network


fig, ax = plt.subplots(figsize=(3, 1.125))
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False)   # labels along the bottom edge are off
ax.grid(1, lw=0.4)
ax.set_aspect('equal')

p = np.array([
    [0., 0],
    [2, 0],
    [4, 0],
    [1, 1],
    [3, 1],
    [5, 1],
    [1, -1],
    [3, -1],
    [5, -1]])

Ep = np.array([
    [0, 3],
    [0, 6],
    [1, 3],
    [1, 4],
    [1, 6],
    [1, 7],
    [2, 4],
    [2, 5],
    [2, 7],
    [2, 8],
    [3, 4],
    [4, 5],
    [6, 7],
    [7, 8]], dtype=int)

q = p + np.array([6, 0])
Eq = Ep

network.plot.nodes(ax, p[2:], color='b', s=30, alpha=0.6, zorder=50)
network.plot.nodes(ax, p[:2], color='orange', s=30, marker='s', zorder=50)
network.plot.nodes(
    ax, q[[0, 1, 2, 3, 4, 6, 7]], color='b', s=30, alpha=0.6, zorder=50)
network.plot.nodes(ax, q[[5, 8]], color='green', s=30, marker='^', zorder=50)
network.plot.edges(ax, p, Ep, color='k', alpha=0.6, lw=0.8, zorder=5)
network.plot.edges(
    ax, p[:2], np.array([[0, 1]]), color='orange', lw=0.8, zorder=5)
network.plot.edges(ax, q, Eq, color='k', alpha=0.6, lw=0.8, zorder=5)
network.plot.edges(
    ax, q[[5, 8]], np.array([[0, 1]]), color='green', lw=0.8, zorder=5)

l1 = np.vstack([p[5], q[3]])
ax.plot(l1[:, 0], l1[:, 1], ls='--', color='k', lw=0.8)
l2 = np.vstack([p[2], q[0]])
ax.plot(l2[:, 0], l2[:, 1], ls='--', color='k', lw=0.8)
l3 = np.vstack([p[8], q[6]])
ax.plot(l3[:, 0], l3[:, 1], ls='--', color='k', lw=0.8)

ax.set_xlim(-0.6, 11.6)
ax.set_ylim(-1.6, 1.6)

plt.show()
fig.savefig('/tmp/rigidity_loss.pdf', format='pdf')
