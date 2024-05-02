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
diam = reshape(np.loadtxt('/tmp/diam.csv', delimiter=','))
sparse_hmax = reshape(
    np.loadtxt('/tmp/sparse_hmax_subopt.csv', delimiter=',')
)
sparse_load = reshape(
    np.loadtxt('/tmp/sparse_load_subopt.csv', delimiter=',')
)
edges = reshape(np.loadtxt('/tmp/edges.csv', delimiter=','))
isolated_edges = reshape(
    np.loadtxt('/tmp/isolated_edges.csv', delimiter=',')
)

d = 2
nmin = d + 2
nmax = int(nodes.max())

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
ax.set_xlabel(r'Number of nodes ($|\mathcal{V}|)$', fontsize=10)
ax.set_xticks(np.arange(20, nmax + 1, 20))
ax.set_xticklabels(np.arange(20, nmax + 1, 20))
# plot delay
ax.plot(
    nodes, 2 * np.median(sparse_hmax, axis=-1),
    label=r'$\mathscr{D}$', lw=1
)
ax.fill_between(
    nodes,
    2 * np.quantile(sparse_hmax, 0.25, axis=-1),
    2 * np.quantile(sparse_hmax, 0.75, axis=-1),
    alpha=0.3
)
# plot diameter
ax.plot(
    nodes, np.median(diam, axis=-1),
    label=r'$D$', lw=1
)
ax.fill_between(
    nodes,
    np.quantile(diam, 0.25, axis=-1),
    np.quantile(diam, 0.75, axis=-1),
    alpha=0.3
)
diam_ticks = np.arange(0, 18, 4)
ax.set_yticks(diam_ticks)
ax.set_yticklabels(diam_ticks)
# ax.set_ylim(0, 10)
ax.set_ylim(bottom=0)
ax.legend(
    fontsize='small', handlelength=1.5,
    labelspacing=0.5, borderpad=0.2, loc='upper left'
)
# save figure
fig.savefig('/tmp/sparse_delay_vs_diam.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Load and Average Degree
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
ax.set_xlabel(r'Number of nodes ($|\mathcal{V}|)$', fontsize=10)
ax.set_xticks(np.arange(20, nmax + 1, 20))
ax.set_xticklabels(np.arange(20, nmax + 1, 20))
# plot load
ax.plot(
    nodes, np.median(sparse_load, axis=-1) / nodes,
    label=r'$\mathscr{L} / |\mathcal{V}|$', lw=1
)
ax.fill_between(
    nodes,
    np.quantile(sparse_load, 0.25, axis=-1) / nodes,
    np.quantile(sparse_load, 0.75, axis=-1) / nodes,
    alpha=0.3
)
# plot average degree
ax.plot(
    nodes,
    2 * np.median(edges, axis=-1) / nodes,
    label=r'$2 |\mathcal{E}| / |\mathcal{V}|$', lw=1
)
ax.fill_between(
    nodes,
    2 * np.quantile(edges, 0.25, axis=-1) / nodes,
    2 * np.quantile(edges, 0.75, axis=-1) / nodes,
    alpha=0.3
)
# plot number of isolated edges
ax.plot(
    nodes,
    2 * np.median(isolated_edges, axis=-1) / nodes,
    label=r'$2 |\mathcal{E}_{I}| / |\mathcal{V}|$', lw=1
)
ax.fill_between(
    nodes,
    2 * np.quantile(isolated_edges, 0.25, axis=-1) / nodes,
    2 * np.quantile(isolated_edges, 0.75, axis=-1) / nodes,
    alpha=0.3
)
ax.legend(
    fontsize='small', handlelength=1.5, ncol=4, columnspacing=0.8,
    labelspacing=0.5, borderpad=0.2, loc='upper left'
)
# save figure
fig.savefig('/tmp/sparse_load_vs_edges.png', format='png', dpi=360)
