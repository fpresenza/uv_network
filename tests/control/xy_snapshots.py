#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on mié 29 dic 2021 16:41:13 -03
@author: fran
"""
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
    default=1, type=float, help='init time in milli seconds'
)
parser.add_argument(
    '-e', '--end',
    default=1, type=float, help='end time in milli seconds'
)
parser.add_argument(
    '-j', '--jump',
    default=1, type=int, help='numbers of frames jumped'
)
arg = parser.parse_args()

# ------------------------------------------------------------------
# Read simulated data
# ------------------------------------------------------------------
t = np.loadtxt('data/t.csv', delimiter=',')

# slices
k_i = int(np.argmin(np.abs(t - arg.init)))
k_e = int(np.argmin(np.abs(t - arg.end)))

x = data.read_csv(
    'data/position.csv',
    rows=(k_i, k_e), jump=arg.jump, dtype=float, shape=(-1, 2)
)
n = len(x[0])

A = data.read_csv(
    'data/adjacency.csv',
    rows=(k_i, k_e), jump=arg.jump, dtype=float, shape=(n, n)
)
targets = data.read_csv(
    'data/targets.csv',
    rows=(k_i, k_e), jump=arg.jump, dtype=float, shape=(-1, 3)
)
action_extents = data.read_csv(
    'data/action_extents.csv', rows=(k_i, k_e), jump=arg.jump, dtype=float
)

N = len(x)

n = len(x[0])
nodes = np.arange(n)

# ------------------------------------------------------------------
# Plot snapshots
# ------------------------------------------------------------------
bar = progressbar.ProgressBar(maxval=N).start()

for k in range(N):
    tk = t[k_i + k]
    lim = min(1500.0 + tk, 4000.0)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize='x-small')
    ax.grid(1, lw=0.4)
    ax.set_aspect('equal')
    # ax.set_xlabel(r'$\mathrm{x}$', fontsize='x-small', labelpad=0.6)
    # ax.set_ylabel(r'$\mathrm{y}$', fontsize='x-small', labelpad=0)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    ax.text(
            0.05, 0.01, r't = {:.3f}s'.format(tk),
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes, color='r', fontsize=8)

    for i in nodes:
        plot.nodes(
            ax, x[k][i],
            color='k',
            marker=f'${i}$',
            s=20,
            lw=0.2
        )
        circle = plt.Circle(
            x[k][i],
            color='C{}'.format(int(action_extents[k][i])),
            radius=30.0, alpha=0.3)
        ax.add_artist(circle)
    plot.edges(
        ax, x[k], A[k], color='C0', lw=0.5, zorder=0
    )

    untracked = targets[k][:, 2].astype(bool)
    tracked = np.logical_not(untracked)
    ax.scatter(
        targets[k][untracked, 0], targets[k][untracked, 1],
        marker='s', s=4, color='0.6')
    ax.scatter(
        targets[k][tracked, 0], targets[k][tracked, 1],
        marker='s', s=4, color='green')

    fig.savefig('data/snapshots/{}.png'.format(k), format='png', dpi=360)
    plt.close()
    bar.update(k)

bar.finish()
