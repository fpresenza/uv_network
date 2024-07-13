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


def reshape(data):
    if data.ndim == 1:
        return data.reshape(-1, 1)
    else:
        return data


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

nodes = np.loadtxt('/tmp/nodes.csv', delimiter=',')
cost_dense = reshape(
    np.loadtxt('/tmp/cost_dense.csv', delimiter=',')
)
cost_sparse = reshape(
    np.loadtxt('/tmp/cost_sparse.csv', delimiter=',')
)

d = 2
nmin = d + 2
nmax = int(nodes.max())

# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
# Cost Dense
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
ax.set_xlabel(r'Number of nodes ($|V|)$', fontsize=10)
ax.set_xticks(np.arange(20, nmax + 1, 20))
ax.set_xticklabels(np.arange(20, nmax + 1, 20))
# plot delay
ax.plot(
    nodes, 2 * np.median(cost_dense, axis=-1),
    label=r'$\mathscr{C}(r_{\mathrm{dense}})$', lw=1
)
ax.fill_between(
    nodes,
    2 * np.quantile(cost_dense, 0.25, axis=-1),
    2 * np.quantile(cost_dense, 0.75, axis=-1),
    alpha=0.3
)
# ax.set_ylim(0, 10)
ax.set_ylim(bottom=0)
ax.legend(
    fontsize='small', handlelength=1.5,
    labelspacing=0.5, borderpad=0.2, loc='upper left'
)

# plot delay
ax.plot(
    nodes, 2 * np.median(cost_sparse, axis=-1),
    label=r'$\mathscr{C}(r_{\mathrm{sparse}})$', lw=1
)
ax.fill_between(
    nodes,
    2 * np.quantile(cost_sparse, 0.25, axis=-1),
    2 * np.quantile(cost_sparse, 0.75, axis=-1),
    alpha=0.3
)
# ax.set_ylim(0, 10)
ax.set_ylim(bottom=0)
ax.legend(
    fontsize='small', handlelength=1.5,
    labelspacing=0.5, borderpad=0.2, loc='upper left'
)
# save figure
fig.savefig('/tmp/cost_dense_vs_sparse.png', format='png', dpi=360)
