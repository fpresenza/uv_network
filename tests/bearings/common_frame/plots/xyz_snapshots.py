#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot bearing-control snapshots from ``common_frame/control/simu_data``.

Example (from the repository root)::

    python tests/bearings/common_frame/plots/xyz_snapshots.py -j 1000 \
        --view 20 -70 --view 30 60 --view 90 0

Repeat ``--view ELEV AZIM`` to create one subplot per pair, in degrees.
The default views are (20, -70) and (20, -10). Start/end times are in seconds.
Frames are saved under ``common_frame/control/xyz_snapshots``. Both input and
output paths are relative to this script, independently of the working directory.
"""

import argparse
from pathlib import Path
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
parser = argparse.ArgumentParser(
    description='Plot bearing-control simulation snapshots.'
)
parser.add_argument(
    '-i', '--init',
    default=0.0, type=float, help='start time in seconds'
)
parser.add_argument(
    '-e', '--end',
    default=0.0, type=float,
    help='end time in seconds (excluded); 0 selects the last time'
)
parser.add_argument(
    '-j', '--jump',
    default=1, type=int, help='numbers of frames jumped'
)
parser.add_argument(
    '-v', '--view', nargs=2, type=float, action='append', metavar=('ELEV', 'AZIM'),
    help='camera elevation and azimuth in degrees; repeat for multiple subplots '
         '(default: 20 -70 and 20 -10)'
)
arg = parser.parse_args()
if arg.jump < 1:
    parser.error('--jump must be positive.')
views = arg.view or [(20.0, -70.0), (20.0, -10.0)]
if not np.isfinite(views).all():
    parser.error('--view angles must be finite numbers.')

simulation_dir = Path(__file__).resolve().parents[1] / 'control'
data_dir = simulation_dir / 'simu_data'
output_dir = simulation_dir / 'xyz_snapshots'

# ------------------------------------------------------------------
# Read simulated data
# ------------------------------------------------------------------
t = np.atleast_1d(np.loadtxt(data_dir / 't.csv', delimiter=','))
arg.end = t[-1] if (arg.end == 0) else arg.end

# slices
k_i = int(np.argmin(np.abs(t - arg.init)))
k_i_jump = int(k_i / arg.jump)
k_e = int(np.argmin(np.abs(t - arg.end)))
if k_e <= k_i:
    parser.error('the selected time window must contain at least one logged sample.')

t = t[k_i:k_e:arg.jump]

# Slice all per-sample logs from the same starting row as the timestamps.
x = data.read_csv_numpy(
    data_dir / 'pose.csv', start=k_i, stop=k_e, jump=arg.jump
).reshape(len(t), -1, 4)
n = x.shape[1]

A = data.read_csv_numpy(
    data_dir / 'sens_adj.csv', start=k_i, stop=k_e, jump=arg.jump
).reshape(len(t), n, n)
targets = data.read_csv_numpy(
    data_dir / 'targets.csv', start=k_i, stop=k_e, jump=arg.jump
).reshape(len(t), -1).astype(bool)

targets_positions = np.loadtxt(
    data_dir / 'targets_positions.csv',
    delimiter=','
).reshape(-1, 3)

N = len(x)

# ------------------------------------------------------------------
# Plot snapshots
# ------------------------------------------------------------------
output_dir.mkdir(parents=True, exist_ok=True)
bar = progressbar.ProgressBar(maxval=N).start()

for k in range(N):
    tk = t[k]
    fig, axes = plt.subplots(
        1, len(views), subplot_kw={"projection": "3d"},
        figsize=(6 * len(views), 6), squeeze=False
    )
    axes = axes[0]
    fig.subplots_adjust(bottom=0.0, wspace=0.0, right=1.0, top=1.0)
    # fig.set_size_inches(10, 8)  # Width = 10, Height = 8
    # fig.tight_layout()

    for view_index, ax in enumerate(axes):
        ax.tick_params(
            axis='both',       # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            pad=-2,
            labelsize='small'
        )
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x \ (\mathrm{m})$', fontsize='small', labelpad=0.5)
        ax.set_ylabel(r'$y \ (\mathrm{m})$', fontsize='small', labelpad=0.5)
        ax.set_zlabel(r'$z \ (\mathrm{m})$', fontsize='small', labelpad=-8.0)
        # ax.zaxis.labelpad = 0

        xy_lim = 100.0
        z_lim = xy_lim / 2
        ax.set_xlim3d(0, xy_lim)
        ax.set_ylim3d(0, xy_lim)
        ax.set_zlim3d(0.0, z_lim)
        ax.set_xticks(np.linspace(0.0, xy_lim, num=5, endpoint=True))
        ax.set_yticks(np.linspace(0.0, xy_lim, num=5, endpoint=True))
        ax.set_zticks(np.linspace(0.0, z_lim, num=3, endpoint=True))

        ax.view_init(elev=views[view_index][0], azim=views[view_index][1])
        # Preserve the existing flattened projection, binding each axis
        # separately so multiple camera views use their own projection.
        z_scale = 0.50 if view_index == 0 else 0.55
        ax.get_proj = lambda ax=ax, z_scale=z_scale: np.dot(
            Axes3D.get_proj(ax), np.diag([1, 1, z_scale, 1])
        )
        ax.set_box_aspect(None, zoom=1.0)

    axes[0].text(
        120.0, 100.0, 150.0, r't = {:.3f}s'.format(tk),
        verticalalignment='bottom', horizontalalignment='left',
        transform=axes[0].transAxes, color='g', fontsize=10
    )

    p = x[k][:, :3]
    psi = x[k][:, 3]
    for i in range(n):
        rotation = np.array([
            [np.cos(psi[i]), -np.sin(psi[i]), 0.0],
            [np.sin(psi[i]), np.cos(psi[i]), 0.0],
            [0.0, 0.0, 1.0],
        ])
        cone = geometry.draw_cone(p[i], rotation, 3.0, np.pi/3)
        for ax in axes:
            ax.add_collection3d(art3d.Poly3DCollection(cone, alpha=0.5))

    for ax in axes:
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
        active_targets = targets_positions[targets[k]]
        ax.scatter(
            active_targets[:, 0],
            active_targets[:, 1],
            active_targets[:, 2],
            marker='d',
            # linewidth=2,
            edgecolor='black',
            facecolor='black',
            s=6,
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

    fig.savefig(
        output_dir / 'frame{}.png'.format(str(k + k_i_jump).zfill(3)),
        format='png',
        dpi=300,
        bbox_inches="tight",
        # transparent=True
    )
    plt.close(fig)
    bar.update(k)

bar.finish()
