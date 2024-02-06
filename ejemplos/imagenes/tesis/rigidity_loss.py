#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt

from uvnpy.rsn.rigidity import extents
from uvnpy.rsn.distances import matrix as distance_matrix
from uvnpy.network import disk_graph
from uvnpy import network

np.set_printoptions(suppress=True, precision=4, linewidth=250)

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'

n = 20
threshold = 1e-5

p = np.array([
    [0.8488, 0.161],
    [0.0544, 0.3254],
    [0.2754, 0.477],
    [0.3059, 0.274],
    [0.1117, 0.2249],
    [0.9176, 0.2377],
    [0.7178, 0.7791],
    [0.1672, 0.042],
    [0.9985, 0.3355],
    [0.7605, 0.4261],
    [0.5097, 0.8505],
    [0.1094, 0.4186],
    [0.6656, 0.3279],
    [0.623, 0.5785],
    [0.42, 0.3629],
    [0.0783, 0.6261],
    [0.1954, 0.6156],
    [0.3511, 0.6939],
    [0.2688, 0.0215],
    [0.3713, 0.088]])

print(p)
Rmin = 0.2961289583948183
Rmax = distance_matrix(p).max()
A = disk_graph.adjacency(p, dmax=Rmin)
A[1, 2] = A[2, 15] = A[3, 7] = A[3, 18] = 0
A[2, 1] = A[15, 2] = A[7, 2] = A[18, 3] = 0

# PARTE 1
fig, ax = plt.subplots(figsize=(2.25, 2.25))
# fig.subplots_adjust(top=0.88, bottom=0.15, wspace=0.28)
h = extents(A, p, threshold)

ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,
    left=False,
    pad=1,
    labelsize='x-small')
ax.grid(1, lw=0.4)
ax.set_aspect('equal')
# ax.set_xlim(-0.05, 1.05)
# ax.set_ylim(-0.05, 1.05)
# ax.set_xlabel(r'$\mathrm{x}$', fontsize='x-small', labelpad=0.6)
# ax.set_ylabel(r'$\mathrm{y}$', fontsize='x-small', labelpad=0)
ax.set_xticks(np.linspace(0, 1, 4, endpoint=True))
ax.set_yticks(np.linspace(0, 1, 4, endpoint=True))
ax.set_xticklabels([])
ax.set_yticklabels([])

network.plot.nodes(
    ax, p,
    marker='o', color='royalblue', s=11, zorder=10)
network.plot.edges(ax, p, A, color='0.0', lw=0.4)
fig.savefig('/tmp/random_framework.png', format='png', dpi=360)

# LOSS
A[13, 14] = 0
A[14, 13] = 0

fig, ax = plt.subplots(figsize=(2.25, 2.25))
# fig.subplots_adjust(top=0.88, bottom=0.15, wspace=0.28)

ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,
    left=False,
    pad=1,
    labelsize='x-small')
ax.grid(1, lw=0.4)
ax.set_aspect('equal')
# ax.set_xlim(-0.05, 1.05)
# ax.set_ylim(-0.05, 1.05)
# ax.set_xlabel(r'$\mathrm{x}$', fontsize='x-small', labelpad=0.6)
# ax.set_ylabel(r'$\mathrm{y}$', fontsize='x-small', labelpad=0)
ax.set_xticks(np.linspace(0, 1, 4, endpoint=True))
ax.set_yticks(np.linspace(0, 1, 4, endpoint=True))
ax.set_xticklabels([])
ax.set_yticklabels([])

network.plot.nodes(
    ax, p,
    marker='o', color='royalblue', s=11, zorder=10)
network.plot.edges(ax, p, A, color='0.0', lw=0.4)
fig.savefig('/tmp/random_framework_loss.png', format='png', dpi=360)


p = np.array([
    [0.8488, 0.161],
    [0.0544, 0.3254],
    [0.2754, 0.477],
    [0.3059, 0.274],
    [0.1117, 0.2249],
    [0.9176, 0.2377],
    [0.7178, 0.7791],
    [0.1672, 0.042],
    [0.9985, 0.3355],
    [0.7605, 0.4261],
    [0.5097, 0.8505],
    [0.1094, 0.4186],
    [0.6656, 0.3279],
    [0.623, 0.5785],
    [0.42, 0.3629],
    [0.0783, 0.6261],
    [0.1954, 0.6156],
    [0.3511, 0.6939],
    [0.2688, 0.0215],
    [0.3713, 0.088]])
