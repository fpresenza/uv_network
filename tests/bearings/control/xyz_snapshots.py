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
t = np.loadtxt('data/t.csv', delimiter=',')
arg.end = t[-1] if (arg.end == 0) else arg.end

# slices
k_i = int(np.argmin(np.abs(t - arg.init)))
k_e = int(np.argmin(np.abs(t - arg.end)))

t = t[k_i:k_e:arg.jump]

x = data.read_csv(
    'data/position.csv',
    rows=(k_i, k_e), jump=arg.jump, dtype=float, shape=(-1, 3)
)
n = len(x[0])

A = data.read_csv(
    'data/adjacency.csv',
    rows=(k_i, k_e), jump=arg.jump, dtype=float, shape=(n, n)
)
targets = data.read_csv(
    'data/targets.csv',
    rows=(k_i, k_e), jump=arg.jump, dtype=float, shape=(-1, 4)
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
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    fig.tight_layout()
    ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize='x-small')
    # ax.grid(1, lw=0.4)
    ax.set_aspect('equal')
    # ax.set_xlabel(r'$\mathrm{x}$', fontsize='x-small', labelpad=0.6)
    # ax.set_ylabel(r'$\mathrm{y}$', fontsize='x-small', labelpad=0)

    ax.set_xlim3d(0, 100.0)
    ax.set_ylim3d(0, 100.0)
    ax.set_zlim3d(0, 100.0)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])
    # ax.view_init(elev=27.5, azim=-17.5)

    ax.text(
            5.0, 5.0, 15.0, r't = {:.3f}s'.format(tk),
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes, color='green', fontsize=6)

    plot.nodes(
        ax, x[k],
        color='b',
        marker='x',
        s=20,
        lw=1
    )
    plot.edges(
        ax, x[k], A[k], color='k', lw=0.35, zorder=0
    )

    untracked = targets[k][:, 3].astype(bool)
    tracked = np.logical_not(untracked)
    ax.scatter(
        targets[k][untracked, 0],
        targets[k][untracked, 1],
        targets[k][untracked, 2],
        marker='o',
        s=4,
        color='0.6'
    )
    ax.scatter(
        targets[k][tracked, 0],
        targets[k][tracked, 1],
        targets[k][tracked, 2],
        marker='o',
        s=4,
        color='green'
    )
    fig.savefig(
        'data/snapshots/{}.png'.format(k),
        format='png',
        dpi=360,
        bbox_inches="tight",
        transparent=True
    )
    plt.close()
    bar.update(k)

bar.finish()
