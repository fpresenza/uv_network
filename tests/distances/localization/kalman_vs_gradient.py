#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from uvnpy.network import plot
from uvnpy.network.core import edges_from_adjacency, incidence_from_edges
from uvnpy.network.disk_graph import adjacency_from_positions
from uvnpy.distances.core import distance_matrix_from_edges, is_inf_rigid
from uvnpy.distances.localization import (
    DistanceBasedGradientFilter,
    DistanceBasedKalmanFilter
)

np.set_printoptions(precision=3, suppress=True, linewidth=200)
plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'

# Común
##################
np.random.seed(6)

dt = 0.05
tiempo = np.arange(0, 20, dt)
steps = range(len(tiempo))

n = 7
nodes = np.arange(n)
lim = 20.0
dmax = 20.0
x = np.random.uniform(-lim, lim, (n, 2))
A = adjacency_from_positions(x, dmax)
if not is_inf_rigid(A, x):
    raise ValueError('Flexible Framework.')
E = edges_from_adjacency(A)
D = incidence_from_edges(n, E)

# np.random.seed(10)
p = 3.0
hatx = np.empty((len(tiempo), n, 2))
hatx[0] = np.random.normal(x, p)

R = 3.0
G = 5.0
d = distance_matrix_from_edges(E, x)
z = np.array([np.random.normal(d, R) for _ in tiempo])
y = np.array([np.random.normal(x, G) for _ in tiempo])
# d2 = d**2
# z2 = z**2


fig, ax = plt.subplots(figsize=(3.5, 2.5))
fig.subplots_adjust(bottom=0.2, left=0.21, right=0.91)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='small')
ax.grid(lw=0.4)
ax.set_xlabel(r'Step ($k$)', fontsize=10)
ax.set_ylabel(r'RMS error $[meters]$', fontsize=10)

# GD 1
##################
stepsize = 0.025
W = np.eye(2)
estimator = [DistanceBasedGradientFilter(
    hatx[0, i], stepsize, W, tiempo[0]) for i in nodes]

for k in steps[1:]:
    for i in nodes:
        Ni = A[i].astype(bool)
        xj = hatx[k-1, Ni]
        ei = np.logical_or(E[:, 0] == i, E[:, 1] == i)
        zi = z[k-1, ei]
        estimator[i].batch_range_step(zi, xj)
        if i == 0 or i == 1:
            estimator[i].gps_step(y[k-1, i])
        hatx[k, i] = estimator[i].state()

# hatz = distance_matrix_from_edges(E, hatx)

ax.plot(
    # steps, np.abs(d - hatz).sum(axis=1) / len(E),
    # steps, np.abs(x - hatx).sum(axis=1).sum(axis=1) / n,
    steps, np.sqrt(np.mean(np.square(x - hatx).sum(axis=2), axis=1)),
    label=r'GD $(\alpha = {})$'.format(stepsize), lw=1
)

fig2, ax2 = plot.figure(figsize=(2.5, 2.5))
ax2.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,
    left=False,
    pad=1,
    labelsize='small')
ax2.set_xlim(-lim, lim)
ax2.set_ylim(-lim, lim)
ax2.grid(lw=0.4)
plot.nodes(ax2, x, color='0.4', marker='o', s=15, zorder=5)
plot.edges(ax2, x, A, color='k', alpha=0.7, lw=0.7)
plot.nodes(
    ax2, hatx[0],
    marker='o', s=15, color='C0', facecolor='none', label=r'$k=0$')
plot.nodes(
    ax2, hatx[1:-1:2], marker='.', s=1, alpha=0.3, zorder=1, color='C0')
plot.nodes(
    ax2, hatx[-1],
    marker='x', s=15, color='C0', zorder=10, label=r'$k=200$')
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.set_xticks(np.linspace(-20, 20, 5, endpoint=True))
ax2.set_yticks(np.linspace(-20, 20, 5, endpoint=True))
ax2.legend(
    fontsize='small', handlelength=1.5,
    labelspacing=0.5, borderpad=0.2, loc='upper right')
fig2.savefig('/tmp/kfa_gd_1.png', format='png', dpi=360)

# KGD
##################
Pi = p**2 * np.eye(2)
hatP = np.empty((len(tiempo), n, 2, 2))
hatP[0] = Pi

q = 0.0
Q = q**2 * np.eye(2)
estimator = [DistanceBasedKalmanFilter(
    hatx[0, i], Pi, Q, R, G**2 * np.eye(2), tiempo[0]) for i in nodes]

for k in steps[1:]:
    for i in nodes:
        Ni = A[i].astype(bool)
        xj = hatx[k-1, Ni]
        Pj = hatP[k-1, Ni]
        ei = np.logical_or(E[:, 0] == i, E[:, 1] == i)
        zi = z[k-1, ei]
        estimator[i].batch_range_step(zi, xj, Pj)
        if i == 0 or i == 1:
            estimator[i].gps_step(y[k-1, i])
        hatx[k, i] = estimator[i].state()
        hatP[k, i] = estimator[i].covariance()

# hatz = distance_matrix_from_edges(E, hatx)

ax.plot(
    # steps, np.abs(d - hatz).sum(axis=1) / len(E),
    # steps, np.abs(x - hatx).sum(axis=1).sum(axis=1) / n,
    steps, np.sqrt(np.mean(np.square(x - hatx).sum(axis=2), axis=1)),
    label=r'KGD', lw=1
)

fig2, ax2 = plot.figure(figsize=(2.5, 2.5))
ax2.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,
    left=False,
    pad=1,
    labelsize='small')
ax2.set_xlim(-lim, lim)
ax2.set_ylim(-lim, lim)
ax2.grid(lw=0.4)
plot.nodes(ax2, x, color='0.4', marker='o', s=15, zorder=5)
plot.edges(ax2, x, A, color='k', alpha=0.7, lw=0.7)
plot.nodes(
    ax2, hatx[0],
    marker='o', s=15, color='C2', facecolor='none', label=r'$k=0$')
plot.nodes(
    ax2, hatx[1:-1:2], marker='.', s=1, alpha=0.3, zorder=1, color='C2')
plot.nodes(
    ax2, hatx[-1],
    marker='x', s=15, color='C2', zorder=10, label=r'$k=200$')
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.set_xticks(np.linspace(-20, 20, 5, endpoint=True))
ax2.set_yticks(np.linspace(-20, 20, 5, endpoint=True))
ax2.legend(
    fontsize='small', handlelength=1.5,
    labelspacing=0.5, borderpad=0.2, loc='upper right')
fig2.savefig('/tmp/kfa_gd_kf.png', format='png', dpi=360)

# Común
##################
ax.legend(
    fontsize='small', handlelength=1.5,
    labelspacing=0.5, borderpad=0.2, loc='upper right')
# ax.set_ylim(bottom=1e-2)
fig.savefig('/tmp/kfa_gd.png', format='png', dpi=360)

# plt.show()
