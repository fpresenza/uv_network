#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt


np.set_printoptions(suppress=True, precision=4, linewidth=250)
plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
nodes = np.loadtxt('/tmp/nodes.csv', delimiter=',', dtype=int)
max_dist = np.loadtxt('/tmp/max_dist.csv', delimiter=',', dtype=float)
dist = np.loadtxt('/tmp/dist.csv', delimiter=',', dtype=float)
eccen = np.loadtxt('/tmp/eccen.csv', delimiter=',', dtype=int)

nmin = nodes.min()
nmax = nodes.max()
max_dist = max_dist.reshape(nmax - nmin + 1, -1, 1)
dist = dist.reshape(nmax - nmin + 1, -1, nmax)
eccen = eccen.reshape(nmax - nmin + 1, -1, nmax)

rep = len(max_dist[0])

# print(max_dist)
# print(dist.shape)
# print(eccen.shape)

# print(max_dist[:, :, np.newaxis] - dist)
# rho = max_dist[:, :, np.newaxis] - dist
# T = rho / 2 * vmax

tau = 0.100
vmax = 100.0 / 50.0

M = np.empty((nmax - nmin + 1, rep), dtype=float)
for k, n in enumerate(range(nmin, nmax + 1)):
    rho = max_dist[k] - dist[k, :, :n]
    T = rho / (2 * vmax)
    # print(eccen[k, :, :n] * tau)
    # print(T)
    # print(rho)
    # print(eccen[k, :, :n] * tau / T)
    M[k] = np.max(eccen[k, :, :n] * tau / T, axis=1)

# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
# Delay and Diameter
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3, 2))
fig.subplots_adjust(bottom=0.2)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='small'
)
ax.grid(lw=0.4)
ax.set_xlabel(r'Number of nodes ($|\mathcal{V}|$)', fontsize=10)
ax.set_xticks(np.arange(20, nmax + 1, 20))
ax.set_xticklabels(np.arange(20, nmax + 1, 20))
ax.semilogy(
    nodes, np.median(M, axis=1),
    label=r'$M$', lw=1
)
# ax.fill_between(
#     nodes,
#     np.quantile(M, 0.25, axis=1),
#     np.quantile(M, 0.75, axis=1),
#     alpha=0.3
# )
# diam_ticks = np.arange(0, 18, 4)
# ax.set_yticks(diam_ticks)
# ax.set_yticklabels(diam_ticks)
# ax.set_ylim(0, 12)
ax.legend(
    fontsize='small', handlelength=1.5,
    labelspacing=0.5, borderpad=0.2, loc='upper left'
)
fig.savefig('/tmp/m.png', format='png', dpi=360)
# plt.show()
