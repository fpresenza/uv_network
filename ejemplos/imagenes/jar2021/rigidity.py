#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on lun jul  5 20:19:35 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt

import uvnpy.network as network
from uvnpy.rsn import rigidity
# from gpsic.toolkit import linalg


p = np.array([
    [1., 0.],
    [0, 1],
    [-2, 0],
    [0, -1]])
E = network.complete_edges(len(p))

fig, ax = plt.subplots(figsize=(2, 1.25))
ax.tick_params(
   axis='both',       # changes apply to the x-axis
   which='both',      # both major and minor ticks are affected
   bottom=False,      # ticks along the bottom edge are off
   top=False,         # ticks along the top edge are off
   left=False,
   right=False,
   labelbottom=False,
   labelleft=False)   # labels along the bottom edge are off
ax.grid(1, lw=0.5)
# ax.set_aspect('equal')
ax.set_xlim(-2.5, 1.5)
ax.set_ylim(-1.5, 2)

network.plot.nodes(ax, p[0], color='b')
network.plot.nodes(ax, p[1], color='r')
network.plot.nodes(ax, p[2], color='g')
network.plot.nodes(ax, p[3], color='orange')
network.plot.edges(ax, p, E, color='k', alpha=0.6)
fig.savefig('/tmp/rigidity_a.png', format='png')

fig, ax = plt.subplots(figsize=(2, 1.25))
ax.tick_params(
   axis='both',       # changes apply to the x-axis
   which='both',      # both major and minor ticks are affected
   bottom=False,      # ticks along the bottom edge are off
   top=False,         # ticks along the top edge are off
   left=False,
   right=False,
   labelbottom=False,
   labelleft=False)   # labels along the bottom edge are off
ax.grid(1, lw=0.5)
# ax.set_aspect('equal')
ax.set_xlim(-2.5, 1.5)
ax.set_ylim(-1.5, 2)

network.plot.nodes(ax, p[0], color='b')
network.plot.nodes(ax, p[1], color='r')
network.plot.nodes(ax, p[2], color='g')
network.plot.nodes(ax, p[3], color='orange')
network.plot.edges(ax, p, E[[0, 2, 3, 4, 5]], color='k', alpha=0.6)
fig.savefig('/tmp/rigidity_b.png', format='png')

fig, ax = plt.subplots(figsize=(2, 1.25))
ax.tick_params(
   axis='both',       # changes apply to the x-axis
   which='both',      # both major and minor ticks are affected
   bottom=False,      # ticks along the bottom edge are off
   top=False,         # ticks along the top edge are off
   left=False,
   right=False,
   labelbottom=False,
   labelleft=False)   # labels along the bottom edge are off
ax.grid(1, lw=0.5)
# ax.set_aspect('equal')
ax.set_xlim(-2.5, 1.5)
ax.set_ylim(-1.5, 2)

p[0, 0] = -1
network.plot.nodes(ax, p[0], color='b')
network.plot.nodes(ax, p[1], color='r')
network.plot.nodes(ax, p[2], color='g')
network.plot.nodes(ax, p[3], color='orange')
network.plot.edges(ax, p, E[[0, 2, 3, 4, 5]], color='k', alpha=0.6)
fig.savefig('/tmp/rigidity_c.png', format='png')

fig, ax = plt.subplots(figsize=(2, 1.25))
ax.tick_params(
   axis='both',       # changes apply to the x-axis
   which='both',      # both major and minor ticks are affected
   bottom=False,      # ticks along the bottom edge are off
   top=False,         # ticks along the top edge are off
   left=False,
   right=False,
   labelbottom=False,
   labelleft=False)   # labels along the bottom edge are off
ax.grid(1, lw=0.5)
# ax.set_aspect('equal')
ax.set_xlim(-2.5, 1.5)

p[0, 0] = 1
network.plot.nodes(ax, p[0], color='b')
network.plot.nodes(ax, p[1], color='r')
network.plot.nodes(ax, p[2], color='g')
network.plot.nodes(ax, p[3], color='orange')
network.plot.edges(ax, p, E[[0, 2, 3, 5]], color='k', alpha=0.6)

H = rigidity.matrix_from_edges(E[[0, 2, 3, 5]], p)
_, v = np.linalg.eigh(H.T.dot(H))
v4 = v[:, 3].reshape(-1, 2)
q = p + v4
q = q - q.mean(0) + p.mean(0)
t = np.arctan2(q[0, 1] - q[2, 1], q[0, 0] - q[2, 0])
# Rz = linalg.Rz(t)[:2, :2]
Rz = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

q = q.dot(Rz)
network.plot.nodes(ax, q[0], color='b', alpha=0.3)
network.plot.nodes(ax, q[1], color='r', alpha=0.3)
network.plot.nodes(ax, q[2], color='g', alpha=0.3)
network.plot.nodes(ax, q[3], color='orange', alpha=0.3)
network.plot.edges(ax, q, E[[0, 2, 3, 5]], color='k', alpha=0.3)
fig.savefig('/tmp/rigidity_d.png', format='png')

plt.show()
