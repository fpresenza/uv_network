#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import matplotlib.pyplot as plt
import progressbar
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

from uvnpy.toolkit import data, geometry, plot
from uvnpy.graphs import core

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
k_i_jump = int(k_i / arg.jump)
k_e = int(np.argmin(np.abs(t - arg.end)))

t = t[k_i:k_e:arg.jump]

x = data.read_csv(
    'data/pose.csv',
    rows=(k_i, k_e),
    jump=arg.jump,
    dtype=float,
    shape=(-1, 4)
)
n = len(x[0])

A = data.read_csv(
    'data/sens_adj.csv',
    rows=(k_i, k_e),
    jump=arg.jump,
    dtype=float,
    shape=(n, n)
)
targets = data.read_csv(
    'data/targets.csv',
    rows=(k_i, k_e),
    jump=arg.jump,
    dtype=float,
    shape=(-1, 4)
)

N = len(x)

n = len(x[0])

# ------------------------------------------------------------------
# Plot snapshots
# ------------------------------------------------------------------
bar = progressbar.ProgressBar(maxval=N).start()

for k in range(N):
    tk = t[k]
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")
    # fig.subplots_adjust(right=0.5)
    # fig.set_size_inches(10, 8)  # Width = 10, Height = 8
    # fig.tight_layout()
    ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=-2,
        labelsize='x-small'
    )
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x \ (\mathrm{m})$', fontsize='small', labelpad=0.5)
    ax.set_ylabel(r'$y \ (\mathrm{m})$', fontsize='small', labelpad=0.5)
    ax.set_zlabel(r'$z \ (\mathrm{m})$', fontsize='small', labelpad=-8.0)
    # ax.zaxis.labelpad = 0

    ax.set_xlim3d(0, 100.0)
    ax.set_ylim3d(0, 100.0)
    ax.set_zlim3d(0.0, 50.0)
    ax.set_xticks([0.0, 25.0, 50.0, 75.0, 100.0])
    ax.set_yticks([0.0, 25.0, 50.0, 75.0, 100.0])
    ax.set_zticks([0.0, 25.0, 50.0])
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])
    ax.view_init(elev=40.0, azim=-30.0)

    ax.text(
        120.0, 100.0, 150.0, r't = {:.3f}s'.format(tk),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes, color='g', fontsize=8
    )

    p = x[k][:, :3]
    psi = x[k][:, 3]
    for i in range(n):
        axis = np.array([np.cos(psi[i]), np.sin(psi[i]), 0.0])
        cone = geometry.cone(p[i], axis, 3.0, np.pi/3)
        ax.add_collection3d(art3d.Poly3DCollection(cone, alpha=0.5))

    plot.points(
        ax, p,
        facecolor='b',
        edgecolor='none',
        marker='o',
        s=10,
        # lw=1,
        zorder=10,
        # alpha=1
    )
    plot.bars(
        ax,
        p,
        core.edges_from_adjacency(A[k]),
        color='0.0',
        alpha=0.5,
        lw=0.5,
        zorder=0
    )

    untracked = targets[k][:, 3].astype(bool)
    tracked = np.logical_not(untracked)
    ax.scatter(
        targets[k][untracked, 0],
        targets[k][untracked, 1],
        targets[k][untracked, 2],
        marker='d',
        # linewidth=2,
        edgecolor='black',
        facecolor='black',
        s=6,
    )
    ax.scatter(
        targets[k][tracked, 0],
        targets[k][tracked, 1],
        targets[k][tracked, 2],
        marker='d',
        linewidth=0.2,
        edgecolor='black',
        facecolor='none',
        s=3,
    )
    ax.xaxis._axinfo['grid'].update(
        color='0.5',
        linestyle='-',
        linewidth=0.25,
        alpha=0.5
    )
    ax.yaxis._axinfo['grid'].update(
        color='0.5',
        linestyle='-',
        linewidth=0.25,
        alpha=0.5
    )
    ax.zaxis._axinfo['grid'].update(
        color='0.5',
        linestyle='-',
        linewidth=0.25,
        alpha=0.5
    )

    # Shorten the Z-axis (flatten effect)
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.55, 1]))
    ax.set_box_aspect(None, zoom=1.0)

    fig.savefig(
        'xyz_snapshots/frame{}.png'.format(str(k + k_i_jump).zfill(3)),
        format='png',
        dpi=200,
        bbox_inches="tight",
        # transparent=True
    )
    plt.close()
    bar.update(k)

bar.finish()
