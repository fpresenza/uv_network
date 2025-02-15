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
d = 2
nmin = d + 2
nmax = int(nodes.max())

fig, ax = plt.subplots(figsize=(4, 2))
fig.subplots_adjust(bottom=0.2, left=0.15)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='small'
)
ax.grid(lw=0.4)
ax.set_ylabel(r'$(\mathscr{C}(\hat{r}) - \mathscr{C}(r))\%$', fontsize=10)
ax.set_xlabel(r'Number of nodes ($|V|)$', fontsize=10)
ax.set_xticks(np.arange(10, nmax + 1, 10))
ax.set_xticklabels(np.arange(10, nmax + 1, 10))

for delta in [4, 5]:
    # cost_dense = reshape(
    #     np.loadtxt('/tmp/cost_dense.csv', delimiter=',')
    # )
    cost_sparse = reshape(
        np.loadtxt(
            '/tmp/cost_sparse_delta_{}.csv'.format(delta),
            delimiter=','
        )
    )
    cost_sparse_dece = reshape(
        np.loadtxt(
            '/tmp/cost_sparse_dece_delta_{}.csv'.format(delta),
            delimiter=','
        )
    )

    # plot cost comparison
    cost_diff = 100 * (cost_sparse_dece - cost_sparse) / cost_sparse
    ax.plot(
        nodes, 2 * np.median(cost_diff, axis=-1),
        label=r'$\delta={}$'.format(delta), lw=1
    )
    ax.fill_between(
        nodes,
        2 * np.quantile(cost_diff, 0.25, axis=-1),
        2 * np.quantile(cost_diff, 0.75, axis=-1),
        alpha=0.3
    )
    # ax.set_ylim(0, 10)
    ax.set_ylim(bottom=0)
    ax.legend(
        fontsize='small', handlelength=1.5,
        labelspacing=0.5, borderpad=0.2, loc='upper left'
    )
# save figure
fig.savefig('/tmp/cost_sparse.png', format='png', dpi=360)
