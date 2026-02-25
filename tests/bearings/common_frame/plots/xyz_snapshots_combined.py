#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import matplotlib.pyplot as plt
import progressbar
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

from uvnpy.toolkit import data, geometry, plot
from uvnpy.graphs.core import edges_from_adjacency, as_undirected

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
    '-s', '--snaps',
    nargs='+', type=int, help='snapshots'
)
arg = parser.parse_args()

# ------------------------------------------------------------------
# Read simulated data
# ------------------------------------------------------------------
t = np.loadtxt('data/t.csv', delimiter=',')
T = len(t)
# slices
N = len(arg.snaps)
ks = [int(np.argmin(np.abs(t - s))) for s in arg.snaps]

t = t[ks]

x = data.read_rows_csv(
    'data/pose.csv',
    rows=ks,
    dtype=float,
    shape=(-1, 4),
    asarray=True
)
n = len(x[0])
As = data.read_rows_csv(
    'data/sens_adj.csv',
    rows=ks,
    dtype=float,
    shape=(n, n),
    asarray=True
)
Ac = data.read_rows_csv(
    'data/comm_adj.csv',
    rows=ks,
    dtype=float,
    shape=(n, n),
    asarray=True
)
targets = data.read_rows_csv(
    'data/targets.csv',
    rows=ks,
    dtype=float,
    shape=(-1, 4),
    asarray=True
)
# ------------------------------------------------------------------
# Plot snapshots
# ------------------------------------------------------------------
bar = progressbar.ProgressBar(maxval=N).start()

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(projection="3d")
fig.subplots_adjust(right=0.5)
# fig.set_size_inches(10, 8)  # Width = 10, Height = 8
fig.tight_layout()
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=0,
    labelsize='xx-small'
)
ax.set_aspect('equal')
ax.set_xlabel(
    r'$x \ (\mathrm{m})$', fontsize='xx-small', labelpad=-12.0
)
ax.set_ylabel(
    r'$y \ (\mathrm{m})$', fontsize='xx-small', labelpad=-12.0
)
ax.set_zlabel(
    r'$z \ (\mathrm{m})$', fontsize='xx-small', labelpad=-12.0
)
# ax.zaxis.labelpad = 0

ax.set_xlim3d(0, 100.0)
ax.set_ylim3d(0, 100.0)
ax.set_zlim3d(0, 50.0)
ax.set_xticks([0.0, 50.0, 100.0])
ax.set_yticks([0.0, 50.0, 100.0])
ax.set_zticks([0.0, 25.0, 50.0])
ax.set_xticklabels(['0.0', '', '100.0'])
ax.set_yticklabels(['0.0', '', '100.0'])
ax.set_zticklabels(['0.0', '', '50.0'])
ax.view_init(elev=5.0, azim=-85.0)

for k, s in enumerate(arg.snaps):
    p = x[k][:, :3]
    psi = x[k][:, 3]

    plot.arrows(
        ax,
        p,
        edges_from_adjacency(As[k].astype(int)),
        color='0.3',
        # alpha=1.0,
        lw=0.4,
        # zorder=1
    )
    plot.bars(
        ax,
        p,
        edges_from_adjacency(Ac[k] - as_undirected(As[k]).astype(int)),
        color='0.3',
        alpha=1.0,
        ls='--',
        lw=0.6,
        zorder=1
    )
    plot.points(
        ax, p,
        facecolor='C{}'.format(k+1),
        edgecolor='k',
        marker='o',
        s=9,
        lw=0.2,
        zorder=2,
        alpha=1
    )
    for i in range(n):
        rotation = np.array(
            [np.cos(psi[i]), -np.sin(psi[i]), 0.0],
            [np.sin(psi[i]), np.cos(psi[i]), 0.0],
            [0.0, 0.0, 1.0],
        )
        cone = geometry.draw_cone(p[i], rotation, 3.0, np.pi/3)
        ax.add_collection3d(art3d.Poly3DCollection(cone, alpha=0.6))
    bar.update(k)

bar.finish()

untracked = targets[k][:, 3].astype(bool)
tracked = np.logical_not(untracked)
ax.scatter(
    targets[k][untracked, 0],
    targets[k][untracked, 1],
    targets[k][untracked, 2],
    marker='d',
    linewidth=0,
    edgecolor='black',
    facecolor='black',
    alpha=1,
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
ax.set_box_aspect(None, zoom=1.1)

fig.savefig('xyz_snapshots/xyz_snapshots_combined.pdf', bbox_inches='tight')
