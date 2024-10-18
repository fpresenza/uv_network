#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on mié 29 dic 2021 16:41:13 -03
@author: fran
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import progressbar

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
    '-s', '--skip',
    default=1, type=int, help='communication step in milli seconds'
)
arg = parser.parse_args()

# ------------------------------------------------------------------
# Read simulated data
# ------------------------------------------------------------------
t = np.loadtxt('data/t.csv', delimiter=',')
x = np.loadtxt('data/position.csv', delimiter=',')
A = np.loadtxt('data/adjacency.csv', delimiter=',')
targets = np.loadtxt('data/targets.csv', delimiter=',')

n = int(len(x[0])/2)
n_targets = int(len(targets[0]) / 3)
nodes = np.arange(n)

# slices
t = t[::arg.skip]
x = x[::arg.skip]
A = A[::arg.skip]
targets = targets[::arg.skip]

# reshapes
x = x.reshape(-1, n, 2)
A = A.reshape(-1, n, n)
targets = targets.reshape(-1, n_targets, 3)

# ------------------------------------------------------------------
# Plot snapshots
# ------------------------------------------------------------------
lim = 1000
bar = progressbar.ProgressBar(maxval=t[-1]).start()

for k, tk in enumerate(t):
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
            0.05, 0.01, r't = {:.2f}s'.format(tk),
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes, color='r', fontsize=8)

    for i in nodes:
        plot.nodes(
            ax, x[k, i],
            color='k',
            marker=f'${i}$',
            s=20,
            lw=0.2
        )
        circle = plt.Circle(x[k, i], 30.0, alpha=0.3)
        ax.add_artist(circle)
    plot.edges(ax, x[k], A[k], color=cm.coolwarm(20), lw=0.5, zorder=0)

    untracked = targets[k, :, 2].astype(bool)
    tracked = np.logical_not(untracked)
    ax.scatter(
        targets[k, untracked, 0], targets[k, untracked, 1],
        marker='s', s=4, color='0.6')
    ax.scatter(
        targets[k, tracked, 0], targets[k, tracked, 1],
        marker='s', s=4, color='green')

    fig.savefig('data/snapshots/{}.png'.format(k), format='png', dpi=360)
    plt.close()
    bar.update(tk)

bar.finish()
