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
rmin = reshape(np.loadtxt('/tmp/rmin.csv', delimiter=','))
rmax = reshape(np.loadtxt('/tmp/rmax.csv', delimiter=','))
alpha = reshape(np.loadtxt('/tmp/alpha.csv', delimiter=','))

nmax = int(nodes.max())
n, r = rmin.shape

diam = np.empty((3, n, r))
hmax = np.empty((3, n, r))
load = np.empty((3, n, r))
edges = np.empty((3, n, r))

for i in range(3):
    diam[i] = reshape(np.loadtxt('/tmp/diam_{}.csv'.format(i), delimiter=','))
    hmax[i] = reshape(np.loadtxt('/tmp/hmax_{}.csv'.format(i), delimiter=','))
    load[i] = reshape(np.loadtxt('/tmp/load_{}.csv'.format(i), delimiter=','))
    edges[i] = reshape(
        np.loadtxt('/tmp/edges_{}.csv'.format(i), delimiter=',')
    )

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
ax.plot(
    nodes, 2 * np.median(hmax[0], axis=1),
    label=r'$\mathscr{D}$', lw=1
)
ax.fill_between(
    nodes,
    2 * np.quantile(hmax[0], 0.25, axis=1),
    2 * np.quantile(hmax[0], 0.75, axis=1),
    alpha=0.3
)
ax.plot(
    nodes, np.median(diam[0], axis=1),
    label=r'$D$', lw=1
)
ax.fill_between(
    nodes,
    np.quantile(diam[0], 0.25, axis=1),
    np.quantile(diam[0], 0.75, axis=1),
    alpha=0.3
)
diam_ticks = np.arange(0, 18, 4)
ax.set_yticks(diam_ticks)
ax.set_yticklabels(diam_ticks)
# ax.set_ylim(0, 12)
ax.legend(
    fontsize='small', handlelength=1.5,
    labelspacing=0.5, borderpad=0.2, loc='upper left')
fig.savefig('/tmp/delay_vs_diam_1.png', format='png', dpi=360)
# plt.show()

fig, ax = plt.subplots(figsize=(3, 2))
fig.subplots_adjust(bottom=0.2)
# ax.set_title(r'$\alpha = 0.1$')
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='small')
ax.grid(lw=0.4)
ax.set_xlabel(r'Number of nodes ($|\mathcal{V}|$)', fontsize=10)
ax.set_xticks(np.arange(20, nmax + 1, 20))
ax.set_xticklabels(np.arange(20, nmax + 1, 20))
ax.plot(
    nodes, 2 * np.median(hmax[1], axis=1),
    label=r'$\mathscr{D}$', lw=1)
ax.fill_between(
    nodes,
    2 * np.quantile(hmax[1], 0.25, axis=1),
    2 * np.quantile(hmax[1], 0.75, axis=1),
    alpha=0.3)
ax.plot(
    nodes, np.median(diam[1], axis=1),
    label=r'$D$', lw=1)
ax.fill_between(
    nodes,
    np.quantile(diam[1], 0.25, axis=1),
    np.quantile(diam[1], 0.75, axis=1),
    alpha=0.3)
diam_ticks = np.arange(0, 18, 4)
ax.set_yticks(diam_ticks)
ax.set_yticklabels(diam_ticks)
# ax.set_ylim(0, 12)
ax.legend(
    fontsize='small', handlelength=1.5,
    labelspacing=0.5, borderpad=0.2, loc='upper left')
fig.savefig('/tmp/delay_vs_diam_2.png', format='png', dpi=360)

fig, ax = plt.subplots(figsize=(3, 2))
fig.subplots_adjust(bottom=0.2)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='small')
ax.grid(lw=0.4)
ax.set_xlabel(r'Number of nodes ($|\mathcal{V}|$)', fontsize=10)
ax.set_xticks(np.arange(20, nmax + 1, 20))
ax.set_xticklabels(np.arange(20, nmax + 1, 20))
ax.plot(
    nodes, 2 * np.median(hmax[2], axis=1),
    label=r'$\mathscr{D}$', lw=1)
ax.fill_between(
    nodes,
    2 * np.quantile(hmax[2], 0.25, axis=1),
    2 * np.quantile(hmax[2], 0.75, axis=1),
    alpha=0.3)
ax.plot(
    nodes, np.median(diam[2], axis=1),
    label=r'$D$', lw=1)
ax.fill_between(
    nodes,
    np.quantile(diam[2], 0.25, axis=1),
    np.quantile(diam[2], 0.75, axis=1),
    alpha=0.3)
diam_ticks = np.arange(0, 18, 4)
ax.set_yticks(diam_ticks)
ax.set_yticklabels(diam_ticks)
# ax.set_ylim(0, 12)
ax.legend(
    fontsize='small', handlelength=1.5,
    labelspacing=0.5, borderpad=0.2, loc='upper left')
fig.savefig('/tmp/delay_vs_diam_3.png', format='png', dpi=360)

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
ax.set_xlabel(r'Number of nodes ($|\mathcal{V}|$)', fontsize=10)
ax.set_xticks(np.arange(20, nmax + 1, 20))
ax.set_xticklabels(np.arange(20, nmax + 1, 20))
ax.plot(
    nodes, np.median(load[0], axis=1) / nodes,
    label=r'$\mathscr{L} / |\mathcal{V}|$', lw=1
)
ax.fill_between(
    nodes,
    np.quantile(load[0], 0.25, axis=1) / nodes,
    np.quantile(load[0], 0.75, axis=1) / nodes,
    alpha=0.3
)
ax.plot(
    nodes, 2 * np.median(edges[0], axis=1) / nodes,
    label=r'$2 |\mathcal{E}| / |\mathcal{V}|$', lw=1
)
ax.fill_between(
    nodes,
    2 * np.quantile(edges[0], 0.25, axis=1) / nodes,
    2 * np.quantile(edges[0], 0.75, axis=1) / nodes,
    alpha=0.3
)
load_ticks = np.arange(0, 80, 20)
ax.set_yticks(load_ticks)
ax.set_yticklabels(load_ticks)
ax.set_ylim(0, 60)
ax.legend(
    fontsize='small', handlelength=1.5,
    labelspacing=0.5, borderpad=0.2, loc='upper left'
)
fig.savefig('/tmp/load_vs_edges_1.png', format='png', dpi=360)

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
ax.plot(
    nodes, np.median(load[1], axis=1) / nodes,
    label=r'$\mathscr{L} / |\mathcal{V}|$', lw=1
)
ax.fill_between(
    nodes,
    np.quantile(load[1], 0.25, axis=1) / nodes,
    np.quantile(load[1], 0.75, axis=1) / nodes,
    alpha=0.3
)
ax.plot(
    nodes, 2 * np.median(edges[1], axis=1) / nodes,
    label=r'$2 |\mathcal{E}| / |\mathcal{V}|$', lw=1
)
ax.fill_between(
    nodes,
    2 * np.quantile(edges[1], 0.25, axis=1) / nodes,
    2 * np.quantile(edges[1], 0.75, axis=1) / nodes,
    alpha=0.3
)
load_ticks = np.arange(0, 80, 20)
ax.set_yticks(load_ticks)
ax.set_yticklabels(load_ticks)
ax.set_ylim(0, 60)
ax.legend(
    fontsize='small', handlelength=1.5,
    labelspacing=0.5, borderpad=0.2, loc='upper left'
)
fig.savefig('/tmp/load_vs_edges_2.png', format='png', dpi=360)

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
ax.plot(
    nodes, np.median(load[2], axis=1) / nodes,
    label=r'$\mathscr{L} / |\mathcal{V}|$', lw=1
)
ax.fill_between(
    nodes,
    np.quantile(load[2], 0.25, axis=1) / nodes,
    np.quantile(load[2], 0.75, axis=1) / nodes,
    alpha=0.3
)
ax.plot(
    nodes, 2 * np.median(edges[2], axis=1) / nodes,
    label=r'$2 |\mathcal{E}| / |\mathcal{V}|$', lw=1
)
ax.fill_between(
    nodes,
    2 * np.quantile(edges[2], 0.25, axis=1) / nodes,
    2 * np.quantile(edges[2], 0.75, axis=1) / nodes,
    alpha=0.3
)
load_ticks = np.arange(0, 80, 20)
ax.set_yticks(load_ticks)
ax.set_yticklabels(load_ticks)
ax.set_ylim(0, 60)
ax.legend(
    fontsize='small', handlelength=1.5,
    labelspacing=0.5, borderpad=0.2, loc='upper left'
)
fig.savefig('/tmp/load_vs_edges_3.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Min max radius
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3, 2))
fig.subplots_adjust(bottom=0.2, left=0.23)
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
ax.plot(
    nodes, np.median(rmin, axis=1) / np.sqrt(2),
    label=r'$\rho_0 / \sqrt{2}$', lw=1
)
ax.fill_between(
    nodes,
    np.quantile(rmin, 0.25, axis=1) / np.sqrt(2),
    np.quantile(rmin, 0.75, axis=1) / np.sqrt(2),
    alpha=0.3
)
ax.plot(
    nodes, np.median(rmax, axis=1) / np.sqrt(2),
    label=r'$\rho_1 / \sqrt{2}$', lw=1
)
ax.fill_between(
    nodes,
    np.quantile(rmax, 0.25, axis=1) / np.sqrt(2),
    np.quantile(rmax, 0.75, axis=1) / np.sqrt(2),
    alpha=0.3
)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.])
ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.])
ax.legend(
    fontsize='small', handlelength=1.5,
    labelspacing=0.5, borderpad=0.2
)
fig.savefig('/tmp/min_max_radius.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Minimum alpha
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3, 2))
fig.subplots_adjust(bottom=0.2, left=0.23)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='small'
)
ax.grid(lw=0.4)
ax.set_xlabel(r'Number of nodes ($|\mathcal{V}|$)', fontsize=10)
ax.set_ylabel(r'$\alpha_{\min}$', fontsize=10)
ax.set_xticks(np.arange(20, nmax + 1, 20))
ax.set_xticklabels(np.arange(20, nmax + 1, 20))
ax.plot(nodes, np.median(alpha, axis=1), label=r'$\alpha_{\min}$', lw=1)
ax.fill_between(
    nodes,
    np.quantile(alpha, 0.25, axis=1),
    np.quantile(alpha, 0.75, axis=1),
    alpha=0.3
)
ax.legend(
    fontsize='small', handlelength=1.5,
    labelspacing=0.5, borderpad=0.2, loc='upper left'
)
fig.savefig('/tmp/minimum_alpha.png', format='png', dpi=360)

# plt.show()
