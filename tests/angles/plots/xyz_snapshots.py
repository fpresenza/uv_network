#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import progressbar
import mpl_toolkits.mplot3d.art3d as art3d

from uvnpy.toolkit.data import read_csv_numpy
from uvnpy.toolkit.geometry import draw_cone
from uvnpy.toolkit import plot
from uvnpy.graphs.core import edges_from_adjacency

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


# ------------------------------------------------------------------
# Read simulated data
# ------------------------------------------------------------------
t = read_csv_numpy('data/t.csv')
log_num_steps = len(t)

position = read_csv_numpy('data/position.csv').reshape(log_num_steps, -1, 3)
n = len(position[0])
orientation = read_csv_numpy(
    'data/orientation.csv'
).reshape(log_num_steps, n, 3, 3)

adjacency = read_csv_numpy('data/adjacency.csv')
edge_set = edges_from_adjacency(adjacency.reshape(n, n))

# ------------------------------------------------------------------
# Plot snapshots
# ------------------------------------------------------------------
bar = progressbar.ProgressBar(maxval=log_num_steps).start()

fig, axes = plt.subplots(
    1, 2, subplot_kw={"projection": "3d"}, figsize=(12, 6)
)
fig.subplots_adjust(bottom=0.0, wspace=0.0, right=1.0, top=1.0)
# fig.set_size_inches(10, 8)  # Width = 10, Height = 8
# fig.tight_layout()
for k in range(log_num_steps):
    tk = t[k]

    for ax in axes:
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

        xy_lim = 1.0
        z_lim = xy_lim
        ax.set_xlim3d(0.0, xy_lim)
        ax.set_ylim3d(0.0, xy_lim)
        ax.set_zlim3d(0.0, z_lim)
        ax.set_xticks(np.linspace(0.0, xy_lim, num=5, endpoint=True))
        ax.set_yticks(np.linspace(0.0, xy_lim, num=5, endpoint=True))
        ax.set_zticks(np.linspace(0.0, z_lim, num=3, endpoint=True))

    axes[0].view_init(elev=20.0, azim=-70.0)
    axes[0].set_box_aspect(None, zoom=1.0)

    axes[1].view_init(elev=40.0, azim=10.0)
    axes[1].set_box_aspect(None, zoom=1.0)

    axes[0].text(
        1.2, 1.0, 1.5, r't = {:.3f}s'.format(tk),
        verticalalignment='bottom', horizontalalignment='left',
        transform=axes[0].transAxes, color='g', fontsize=10
    )

    # --- position and orientation--- #
    p = position[k]
    R = orientation[k]

    # --- plot --- #
    for i in range(n):
        axis = R[i, :, 0]    # first column (x-axis)
        cone = draw_cone(p[i], axis, xy_lim / 20, np.pi/3)
        axes[0].add_collection3d(art3d.Poly3DCollection(cone, alpha=0.5))
        axes[1].add_collection3d(art3d.Poly3DCollection(cone, alpha=0.5))

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
            facecolor='b',
            edgecolor='none',
            marker='o',
            s=10,
            # lw=1,
            zorder=10,
            # alpha=1
        )
        plot.arrows(
            ax,
            p,
            edge_set,
            color='0.0',
            alpha=0.5,
            lw=0.75,
            zorder=0,
            length=0.9,
            arrow_length_ratio=0.15
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
        'xyz_snapshots/frame{}.png'.format(str(k).zfill(3)),
        format='png',
        dpi=300,
        bbox_inches="tight",
        # transparent=True
    )
    axes[0].clear()
    axes[1].clear()
    plt.close()
    bar.update(k)

bar.finish()
