#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
import progressbar

from uvnpy.toolkit import data
from uvnpy.network import plot

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


# ------------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-i', '--init',
    default=0.0, type=float, help='init time in milli seconds'
)
parser.add_argument(
    '-e', '--end',
    default=0.0, type=float, help='end time in milli seconds'
)
parser.add_argument(
    '-j', '--jump',
    default=1, type=int, help='numbers of frames jumped'
)
arg = parser.parse_args()

# ------------------------------------------------------------------
# Read simulated data
# ------------------------------------------------------------------
t = np.loadtxt('data/log_time.csv', delimiter=',')
arg.end = t[-1] if (arg.end == 0) else arg.end

# slices
k_i = int(np.argmin(np.abs(t - arg.init)))
k_i_jump = int(k_i / arg.jump)
k_e = int(np.argmin(np.abs(t - arg.end))) + 1

x = data.read_csv(
    'data/robots_position.csv',
    rows=(k_i, k_e),
    jump=arg.jump,
    dtype=float,
    shape=(-1, 2)
)
n = len(x[0])

edge_list = data.read_csv(
    'data/edge_list.csv',
    rows=(k_i, k_e),
    jump=arg.jump,
    dtype=int,
    shape=(-1, 2)
)
len(edge_list)

targets = data.read_csv(
    'data/targets_position.csv',
    rows=(k_i, k_e),
    jump=arg.jump,
    dtype=float,
    shape=(-1, 3)
)

N = len(x)

n = len(x[0])
nodes = np.arange(n)

# ------------------------------------------------------------------
# Plot snapshots
# ------------------------------------------------------------------
bar = progressbar.ProgressBar(maxval=N).start()

for k in range(N):
    tk = t[k_i_jump + k * arg.jump]

    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize='x-small'
    )
    ax.grid(1, lw=0.4)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x \ (\mathrm{m})$', fontsize='x-small', labelpad=0.6)
    ax.set_ylabel(r'$y \ (\mathrm{m})$', fontsize='x-small', labelpad=0.6)

    lim = 100.0
    ax.set_xlim(0.0, lim)
    ax.set_ylim(0.0, lim)

    ax.text(
        0.05, 0.01, r't = {:.3f}s'.format(tk),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes, color='r', fontsize='x-small'
    )

    for i in nodes:
        plot.nodes(
            ax, x[k][i],
            color='b',
            marker='o',
            s=15,
            lw=0.2
        )
    plot.edges(
        ax,
        x[k],
        edge_list[k],
        color='0.0',
        alpha=0.5,
        lw=0.3,
        zorder=0
    )

    untracked = targets[k][:, 2].astype(bool)
    ax.scatter(
        targets[k][untracked, 0],
        targets[k][untracked, 1],
        color='k',
        marker='d',
        s=15,
        lw=0.2
    )

    fig.savefig('xy_snapshots/{}.png'.format(k), bbox_inches='tight', dpi=600)
    plt.close()
    bar.update(k)

bar.finish()
