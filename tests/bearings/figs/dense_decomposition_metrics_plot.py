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
nodes = np.loadtxt('/tmp/nodes.csv', delimiter=',')
diam = np.loadtxt('/tmp/diam.csv', delimiter=',')
hmax_d = np.loadtxt('/tmp/hmax_d.csv', delimiter=',')
hmax_b = np.loadtxt('/tmp/hmax_b.csv', delimiter=',')

# nmax = int(nodes.max())

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
ax.set_xlabel(r'Number of nodes ($n$)', fontsize=10)
# ax.set_xticks(np.arange(20, nmax + 1, 20))
# ax.set_xticklabels(np.arange(20, nmax + 1, 20))
ax.plot(
    nodes, np.median(hmax_d, axis=1),
    label=r'$\eta_{\mathrm{dist}}$', lw=1
)
# ax.fill_between(
#     nodes,
#     2 * np.quantile(hmax[0], 0.25, axis=1),
#     2 * np.quantile(hmax[0], 0.75, axis=1),
#     alpha=0.3
# )
# ax.plot(
#     nodes, np.median(diam[0], axis=1),
#     label=r'$D$', lw=1
# )
# ax.fill_between(
#     nodes,
#     np.quantile(diam[0], 0.25, axis=1),
#     np.quantile(diam[0], 0.75, axis=1),
#     alpha=0.3
# )
# diam_ticks = np.arange(0, 18, 4)
# ax.set_yticks(diam_ticks)
# ax.set_yticklabels(diam_ticks)
# ax.set_ylim(0, 12)
ax.legend(
    fontsize='small', handlelength=1.5,
    labelspacing=0.5, borderpad=0.2, loc='upper left')
fig.savefig('/tmp/distance_extents.png', format='png', dpi=360)
# plt.show()

fig, ax = plt.subplots(figsize=(3, 2))
fig.subplots_adjust(bottom=0.2)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='small')
ax.grid(lw=0.4)
ax.set_xlabel(r'Number of nodes ($n$)', fontsize=10)
# ax.set_xticks(np.arange(20, nmax + 1, 20))
# ax.set_xticklabels(np.arange(20, nmax + 1, 20))
ax.plot(
    nodes, np.median(hmax_b, axis=1),
    label=r'$\eta_{\mathrm{bear}}$', lw=1
)
# ax.fill_between(
#     nodes,
#     2 * np.quantile(hmax[1], 0.25, axis=1),
#     2 * np.quantile(hmax[1], 0.75, axis=1),
#     alpha=0.3
# )
# ax.plot(
#     nodes, np.median(diam[1], axis=1),
#     label=r'$D$', lw=1
# )
# ax.fill_between(
#     nodes,
#     np.quantile(diam[1], 0.25, axis=1),
#     np.quantile(diam[1], 0.75, axis=1),
#     alpha=0.3)
# diam_ticks = np.arange(0, 18, 4)
# ax.set_yticks(diam_ticks)
# ax.set_yticklabels(diam_ticks)
# ax.set_ylim(0, 12)
ax.legend(
    fontsize='small', handlelength=1.5,
    labelspacing=0.5, borderpad=0.2, loc='upper left'
)
fig.savefig('/tmp/bearing_extents.png', format='png', dpi=360)
