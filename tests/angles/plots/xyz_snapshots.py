#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot snapshots from an angle-based simulation's logged data.

Examples (from the repository root)::

    python tests/angles/plots/xyz_snapshots.py arm -j 1000
    python tests/angles/plots/xyz_snapshots.py anl -j 1000
    python tests/angles/plots/xyz_snapshots.py arm -j 1000 --view 10 -45 --view 30 60

Read ``tests/angles/<directory>/simu_data`` and save frames under
``tests/angles/<directory>/xyz_snapshots``, independently of the working directory.
Repeat ``--view ELEV AZIM`` to create one subplot per pair, in degrees.
Without ``--view``, the views are (10, -45) and (10, 45).
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import progressbar
import mpl_toolkits.mplot3d.art3d as art3d

from uvnpy.toolkit.data import read_csv_numpy, read_json_file
from uvnpy.toolkit.geometry import draw_cone
from uvnpy.toolkit import plot
from uvnpy.graphs.core import edges_from_adjacency

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


# ------------------------------------------------------------------
# Argument parse
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Plot angle-based simulation snapshots.')
parser.add_argument(
    'directory', choices=('arm', 'anl'),
    help='simulation directory under tests/angles'
)
parser.add_argument(
    '-j', '--jump',
    default=1, type=int, help='simulation step jumped'
)
parser.add_argument(
    '-s', '--snaps',
    default=False, action='store_true', help='Whether to plot as snapshots.'
)
parser.add_argument(
    '-m', '--multijump',
    default=1, type=int, help='Multiplicative jump'
)
parser.add_argument(
    '-v', '--view', nargs=2, type=float, action='append', metavar=('ELEV', 'AZIM'),
    help='camera elevation and azimuth in degrees; repeat for multiple subplots '
         '(default: 10 -45 and 10 45)'
)
arg = parser.parse_args()
if arg.jump < 1 or arg.multijump < 1 or arg.jump % arg.multijump != 0:
    parser.error('--jump must be a positive multiple of --multijump (also positive).')
views = arg.view or [(10.0, -45.0), (10.0, 45.0)]
if not np.isfinite(views).all():
    parser.error('--view angles must be finite numbers.')

simulation_dir = Path(__file__).resolve().parents[1] / arg.directory
data_dir = simulation_dir / 'simu_data'
output_dir = simulation_dir / 'xyz_snapshots'

# ------------------------------------------------------------------
# Read simulated data
# ------------------------------------------------------------------
t = np.atleast_1d(read_csv_numpy(data_dir / 't.csv', jump=arg.jump))
log_num_steps = len(t)

position = read_csv_numpy(
    data_dir / 'position.csv', jump=arg.jump
).reshape(log_num_steps, -1, 3)
n = position.shape[1]

orientation = read_csv_numpy(
    data_dir / 'orientation.csv', jump=arg.jump
).reshape(log_num_steps, n, 3, 3)

adjacency = read_csv_numpy(
    data_dir / 'adjacency.csv', jump=arg.jump
).reshape(-1, n, n)
# Some localization runs log only the initial graph. Dynamic graphs require
# one adjacency row per logged pose; a single matrix is reused for all frames.
if len(adjacency) == 1:
    adjacency = np.broadcast_to(adjacency, (log_num_steps, n, n))
elif len(adjacency) != log_num_steps:
    parser.error('adjacency.csv must contain one graph or one per logged pose.')

targets_ids = []
target_position = None
if (data_dir / 'targets.jsonlog').exists() and \
        (data_dir / 'target_position.csv').exists():
    targets = read_json_file(data_dir / 'targets.jsonlog')
    targets_ids = targets['ids']
    if targets_ids:
        target_position = read_csv_numpy(
            data_dir / 'target_position.csv', jump=arg.jump // arg.multijump
        ).reshape(-1, len(targets_ids), 3)
targets_style = ['d', 'P', 's', '*', 'X', '^']

# ------------------------------------------------------------------
# Plot snapshots
# ------------------------------------------------------------------
output_dir.mkdir(parents=True, exist_ok=True)
bar = progressbar.ProgressBar(maxval=log_num_steps).start()

fig, axes = plt.subplots(
    1, len(views), subplot_kw={"projection": "3d"},
    figsize=(6 * len(views), 6), squeeze=False
)
axes = axes[0]
# fig.subplots_adjust(bottom=0.0, wspace=0.0, right=1.0, top=1.0)
# fig.set_size_inches(10, 8)  # Width = 10, Height = 8
fig.tight_layout()
for k in range(log_num_steps):
    tk = t[k]

    for ax, (elev, azim) in zip(axes, views):
        ax.tick_params(
            axis='both',       # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            pad=-2,
            labelsize='medium'
        )
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x \ (\mathrm{m})$', fontsize='medium', labelpad=0.5)
        ax.set_ylabel(r'$y \ (\mathrm{m})$', fontsize='medium', labelpad=0.5)
        ax.set_zlabel(r'$z \ (\mathrm{m})$', fontsize='medium', labelpad=-9.0)
        # ax.zaxis.labelpad = 0

        xy_lim = 100.0
        z_lim = xy_lim
        ax.set_xlim3d(0.0, xy_lim)
        ax.set_ylim3d(0.0, xy_lim)
        ax.set_zlim3d(0.0, z_lim)
        ax.set_xticks(np.linspace(0.0, xy_lim, num=4, endpoint=False))
        ax.set_yticks(np.linspace(0.0, xy_lim, num=4, endpoint=False))
        ax.set_zticks(np.linspace(0.0, z_lim, num=4, endpoint=False))

        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect(None, zoom=1.0)

    axes[0].text(
        100, 100.0, 100.0, r't = {:.3f}s'.format(tk),
        verticalalignment='bottom', horizontalalignment='left',
        transform=axes[0].transAxes, color='g', fontsize=12
    )

    # --- position and orientation--- #
    p = position[k]
    R = orientation[k]

    # --- plot --- #
    for i in range(n):
        # axis = R[i, :, 0]    # first column (x-axis)
        cone = draw_cone(p[i], R[i], xy_lim/15, np.pi/3)

        colors = [f'C{i}'] * len(cone)
        # colors[0:2] = ['r'] * 3    # mark one sector
        for ax in axes:
            poly = art3d.Poly3DCollection(cone, alpha=0.5)
            poly.set_facecolor(colors)
            ax.add_collection3d(poly)

    for ax in axes:
        # plot.points(
        #     ax, q,
        #     facecolor='r',
        #     edgecolor='none',
        #     marker='s',
        #     s=10,
        #     # lw=1,
        #     zorder=10,
        #     # alpha=1
        # )
        plot.points(
            ax, p,
            facecolor='0.3',
            edgecolor='none',
            marker='o',
            s=15,
            # lw=1,
            zorder=10,
            # alpha=1
        )
        plot.arrows(
            ax,
            p,
            edges_from_adjacency(adjacency[k]),
            color='0.0',
            alpha=0.5,
            lw=0.75,
            zorder=0,
            length=0.45,
            arrow_length_ratio=0.3
        )
        for m, i in enumerate(targets_ids):
            mk = arg.multijump * k
            ax.scatter(
                target_position[mk, m, 0],
                target_position[mk, m, 1],
                target_position[mk, m, 2],
                marker=targets_style[m],
                # linewidth=2,
                edgecolor='k',
                facecolor='k',
                s=20,
                alpha=1.0,
                zorder=0,
            )
            if arg.snaps:
                ax.scatter(
                    target_position[max(0, mk-60):mk:6, m, 0],
                    target_position[max(0, mk-60):mk:6, m, 1],
                    target_position[max(0, mk-60):mk:6, m, 2],
                    marker=targets_style[m],
                    # linewidth=2,
                    edgecolor='0.0',
                    facecolor='0.0',
                    s=3,
                    alpha=0.5,
                    zorder=10
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
        output_dir / 'frame{}.png'.format(str(k).zfill(3)),
        format='png',
        dpi=300,
        # bbox_inches="tight",
        # transparent=True
    )
    for ax in axes:
        ax.clear()
    bar.update(k)

bar.finish()
plt.close(fig)
