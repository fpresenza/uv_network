#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar sep 14 12:01:21 -03 2021
"""
import numpy as np
import matplotlib.pyplot as plt

import uvnpy.network as network
from uvnpy.network import disk_graph, plot
from uvnpy.rsn import distances
from uvnpy.rsn.localization import (
    distances_to_neighbors_gradient,
    distances_to_neighbors_kalman)
from uvnpy.rsn import rigidity

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
tiempo = np.arange(0, 10, dt)
steps = range(len(tiempo))

n = 7
nodes = np.arange(n)
lim = 20
dmax = 20
x = np.empty((len(tiempo), n, 2))
u = np.zeros(2)
x[0] = np.random.uniform(-lim, lim, (n, 2))
A = disk_graph.adjacency(x[0], dmax)
E = network.edges_from_adjacency(A)

# np.random.seed(10)
p = 3
hatx = np.empty((len(tiempo), n, 2))
hatx[0] = x[0] + np.random.normal(0, p, (n, 2))


fig, ax = plt.subplots(figsize=(4, 2.5))
fig.subplots_adjust(bottom=0.2, left=0.21, right=0.91)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='small')
ax.grid(lw=0.4)
ax.set_xlabel('Step ($k$)', fontsize=10)
ax.set_ylabel(r'Error ($\xi / |E|$) [$m^2$]', fontsize=10)

# GD 1
##################
R = 3.
stepsize = 0.1
W = np.eye(2)
estimator = [distances_to_neighbors_gradient(
    hatx[0, i], tiempo[0], stepsize, W) for i in nodes]

d = np.empty((len(tiempo), len(E)))
d[0] = distances.from_edges(E, x[0])
z = np.empty((len(tiempo), len(E)))
z[0] = np.random.normal(d[0], R)

for k in steps[1:]:
    for i in nodes:
        Ni = A[i].astype(bool)
        xj = hatx[k-1, Ni]
        ei = np.logical_or(E[:, 0] == i, E[:, 1] == i)
        zi = z[k-1, ei]
        estimator[i].update_neighbors(xj, 1/2)
        estimator[i].step(tiempo[k], u, zi)
        hatx[k, i] = estimator[i].x

    x[k] = x[k-1] + u
    d[k] = distances.from_edges(E, x[k])
    z[k] = np.random.normal(d[k], R)

hatz = distances.from_edges(E, hatx)

ax.semilogy(
    steps, np.square(d - hatz).sum(axis=1) / 2 / len(E),
    label=r'GD $(\alpha = 0.15)$', lw=1, marker='o', markersize=3)

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
plot.nodes(ax2, x[0], color='0.4', marker='o', s=15, zorder=5)
plot.edges(ax2, x[0], A, color='k', alpha=0.7, lw=0.7)
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

# GD 2
##################
# R = 3.
# stepsize = 0.05
# W = np.eye(2)
# estimator = [distances_to_neighbors_gradient(
#     hatx[0, i], tiempo[0], stepsize, W) for i in nodes]

# d = np.empty((len(tiempo), len(E)))
# d[0] = distances.from_edges(E, x[0])
# z = np.empty((len(tiempo), len(E)))
# z[0] = np.random.normal(d[0], R)

# for k in steps[1:]:
#     for i in nodes:
#         Ni = A[i].astype(bool)
#         xj = hatx[k-1, Ni]
#         ei = np.logical_or(E[:, 0] == i, E[:, 1] == i)
#         zi = z[k-1, ei]
#         estimator[i].update_neighbors(xj, 1/2)
#         estimator[i].step(tiempo[k], u, zi)
#         hatx[k, i] = estimator[i].x

#     x[k] = x[k-1]
#     d[k] = distances.from_edges(E, x[k])
#     z[k] = np.random.normal(d[k], R)

# hatz = distances.from_edges(E, hatx)

# ax.semilogy(
#     steps, np.square(d - hatz).sum(axis=1) / 2 / len(E),
#     label=r'GD $(\alpha = 0.05)$', lw=1)

# fig2, ax2 = plot.figure(figsize=(2.5, 2.5))
# ax2.tick_params(
#     axis='both',       # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,
#     left=False,
#     pad=1,
#     labelsize='small')
# ax2.set_xlim(-lim, lim)
# ax2.set_ylim(-lim, lim)
# ax2.grid(lw=0.4)
# plot.nodes(ax2, x[0], color='0.4', marker='o', s=15, zorder=5)
# plot.edges(ax2, x[0], A, color='k', alpha=0.7, lw=0.7)
# plot.nodes(
#     ax2, hatx[0],
#     marker='o', s=15, color='C1', facecolor='none', label=r'$k=0$')
# plot.nodes(
#     ax2, hatx[1:-1:2], marker='.', s=1, alpha=0.3, zorder=1, color='C1')
# plot.nodes(
#     ax2, hatx[-1],
#     marker='x', s=15, color='C1', zorder=10, label=r'$k=200$')
# ax2.set_xlabel('')
# ax2.set_ylabel('')
# ax2.set_xticks(np.linspace(-20, 20, 5, endpoint=True))
# ax2.set_yticks(np.linspace(-20, 20, 5, endpoint=True))
# ax2.legend(
#     fontsize='small', handlelength=1.5,
#     labelspacing=0.5, borderpad=0.2, loc='upper right')
# fig2.savefig('/tmp/kfa_gd_1.png', format='png', dpi=360)

# GD 3
##################
R = 3.
stepsize = 0.1
W = np.eye(2)
estimator = [distances_to_neighbors_gradient(
    hatx[0, i], tiempo[0], stepsize, W) for i in nodes]

d = np.empty((len(tiempo), len(E)))
d[0] = distances.from_edges(E, x[0])
z = np.empty((len(tiempo), len(E)))
z[0] = np.random.normal(d[0], R)

for k in steps[1:]:
    for i in nodes:
        Ni = A[i].astype(bool)
        xj = hatx[k-1, Ni]
        ei = np.logical_or(E[:, 0] == i, E[:, 1] == i)
        zi = z[k-1, ei]
        estimator[i].update_neighbors(xj, 1/2)
        estimator[i].step(tiempo[k], u, zi, stepsize)
        hatx[k, i] = estimator[i].x

    x[k] = x[k-1] + u
    d[k] = distances.from_edges(E, x[k])
    z[k] = np.random.normal(d[k], R)

    """ Barzilai-Borwein method"""
    step_x = np.ravel(hatx[k] - hatx[k-1])

    Rmat = rigidity.matrix_from_adjacency(A, hatx[k])
    Rmat_last = rigidity.matrix_from_adjacency(A, hatx[k-1])
    hat_z = distances.from_edges(E, hatx[k])
    hat_z_last = distances.from_edges(E, hatx[k-1])
    grad = Rmat.T.dot(hat_z - z[k])
    grad_last = Rmat.T.dot(hat_z_last - z[k-1])
    step_g = grad - grad_last

    stepsize = 1.75 * step_x.dot(step_x) / step_x.dot(step_g)
    stepsize = np.clip(stepsize, 1e-5,  np.inf)
    # print(stepsize)

hatz = distances.from_edges(E, hatx)

ax.semilogy(
    steps, np.square(d - hatz).sum(axis=1) / 2 / len(E),
    label=r'GD $(\alpha_{BB})$', lw=1, marker='s', markersize=3)

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
plot.nodes(ax2, x[0], color='0.4', marker='o', s=15, zorder=5)
plot.edges(ax2, x[0], A, color='k', alpha=0.7, lw=0.7)
plot.nodes(
    ax2, hatx[0],
    marker='o', s=15, color='C1', facecolor='none', label=r'$k=0$')
plot.nodes(
    ax2, hatx[1:-1:2], marker='.', s=1, alpha=0.3, zorder=1, color='C1')
plot.nodes(
    ax2, hatx[-1],
    marker='x', s=15, color='C1', zorder=10, label=r'$k=200$')
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.set_xticks(np.linspace(-20, 20, 5, endpoint=True))
ax2.set_yticks(np.linspace(-20, 20, 5, endpoint=True))
ax2.legend(
    fontsize='small', handlelength=1.5,
    labelspacing=0.5, borderpad=0.2, loc='upper right')
fig2.savefig('/tmp/kfa_gd_bb.png', format='png', dpi=360)

# FKA
##################
Pi = p**2 * np.eye(2)
hatP = np.empty((len(tiempo), n, 2, 2))
hatP[0] = Pi

q = 0.0
Q = q**2 * np.eye(2)
R = 3.
estimator = [distances_to_neighbors_kalman(
    hatx[0, i], Pi, Q * dt, R**2, tiempo[0]) for i in nodes]

d = np.empty((len(tiempo), len(E)))
d[0] = distances.from_edges(E, x[0])
z = np.empty((len(tiempo), len(E)))
z[0] = np.random.normal(d[0], R)

for k in steps[1:]:
    for i in nodes:
        Ni = A[i].astype(bool)
        xj = hatx[k-1, Ni]
        Pj = hatP[k-1, Ni]
        ei = np.logical_or(E[:, 0] == i, E[:, 1] == i)
        zi = z[k-1, ei]
        estimator[i].update_neighbors(xj, Pj)
        estimator[i].step(tiempo[k], u, zi)
        hatx[k, i] = estimator[i].x
        hatP[k, i] = estimator[i].P

    x[k] = x[k-1] + u
    d[k] = distances.from_edges(E, x[k])
    z[k] = np.random.normal(d[k], R)

hatz = distances.from_edges(E, hatx)

ax.semilogy(
    steps, np.square(d - hatz).sum(axis=1) / 2 / len(E),
    label=r'DKF', lw=1, marker='x', markersize=3)

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
plot.nodes(ax2, x[0], color='0.4', marker='o', s=15, zorder=5)
plot.edges(ax2, x[0], A, color='k', alpha=0.7, lw=0.7)
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
ax.set_ylim(bottom=1e-2)
fig.savefig('/tmp/kfa_gd.png', format='png', dpi=360)

# plt.show()
