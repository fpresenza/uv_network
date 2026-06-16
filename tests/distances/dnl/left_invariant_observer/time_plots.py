#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from uvnpy.toolkit.plot import bars
from uvnpy.toolkit.data import read_csv_numpy

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'

# ------------------------------------------------------------------
# Argument parse
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-t', '--targets',
    default=False, action='store_true', help='Whether there are targets.'
)
arg = parser.parse_args()

# ------------------------------------------------------------------
# Read simulated data
# ------------------------------------------------------------------
t = read_csv_numpy('simu_data/t.csv')
log_num_steps = len(t)

p = read_csv_numpy('simu_data/position.csv').reshape(log_num_steps, -1, 3)
n = len(p[0])
R = read_csv_numpy('simu_data/orientation.csv').reshape(-1, n, 3, 3)
hatq = read_csv_numpy('simu_data/estimated_position.csv').reshape(-1, n - 1, 3)
hatQ = read_csv_numpy('simu_data/estimated_orientation.csv').reshape(-1, 3, 3)

cov_matrix = read_csv_numpy('simu_data/covariance.csv').reshape(-1, 3*n, 3*n)

control_u = read_csv_numpy('simu_data/control_u.csv').reshape(-1, n, 3)
control_w = read_csv_numpy('simu_data/control_w.csv').reshape(-1, n, 3)

if arg.targets:
    targets_positions = np.loadtxt(
        'simu_data/targets_positions.csv', delimiter=','
    ).reshape(-1, 3)
    nt = len(targets_positions)
    active_targets = read_csv_numpy(
        'simu_data/targets.csv'
    ).astype(bool).reshape(log_num_steps, nt)


a = 0
neighbors = np.setdiff1d(np.arange(n), a)

# change of basis
q = np.matmul(p[:, neighbors] - p[:, np.newaxis, a], R[:, a])

# ------------------------------------------------------------------
# Plot left-invariant position
# ------------------------------------------------------------------
fig, ax = plt.subplots(3, 1, figsize=(9.0, 6.0))
fig.subplots_adjust(
    bottom=0.215,
    top=0.925,
    wspace=0.33,
    right=0.975,
    left=0.18
)

for k, d in enumerate(['x', 'y', 'z']):
    ax[k].tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize=9
    )

    ax[k].set_xlabel(r'$t\ (\mathrm{s})$', fontsize=10)
    ax[k].set_ylabel(fr'$p_{{ij, {d}}}, \hat{{p}}_{{ij, {d}}} \ (\rm m)$', fontsize=10)
    ax[k].grid(1)
    # ax[k].set_ylim(-10.0, 50.0)

    ax[k].plot(
        t,
        q[:, :, k],
        lw=1.0,
        ds='steps-post',
    )
    ax[k].plot(
        t,
        hatq[:, :, k],
        lw=0.8,
        color='0.5',
        ls='--',
        ds='steps-post',
    )

fig.savefig('time_plots/position.pdf', bbox_inches='tight')

# -------------------------------------------------------
# Plot orientations
# ------------------------------------------------------------------
fig, ax = plt.subplots(3, 1, figsize=(9.0, 6.0))
fig.subplots_adjust(
    bottom=0.215,
    top=0.925,
    wspace=0.33,
    right=0.975,
    left=0.18
)

# obtain euler angles
euler_angles = np.empty((log_num_steps, n, 3), dtype=np.float64)
hat_euler_angles = np.empty((log_num_steps, 3), dtype=np.float64)
for k in range(log_num_steps):
    euler_angles[k] = Rotation.from_matrix(
        R[k]
    ).as_euler('ZYX', degrees=False)
    hat_euler_angles[k] = Rotation.from_matrix(
        hatQ[k]
    ).as_euler('ZYX', degrees=False)

for k, d in enumerate(['yaw', 'pitch', 'roll']):
    ax[k].tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize=9
    )

    ax[k].set_xlabel(r'$t\ (\mathrm{s})$', fontsize=10)
    ax[k].set_ylabel(fr'${d} \ (\rm rad)$', fontsize=10)
    ax[k].set_ylim(-np.pi, np.pi)
    ax[k].grid(1)

    ax[k].plot(t, euler_angles[:, :, k], lw=1.0, ds='steps-post')

fig.savefig('time_plots/euler_angles.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot left-invariant pose error
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8.5, 2.5))
fig.tight_layout()
fig.subplots_adjust(wspace=0.25)

for ax in axes:
    ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize=12
    )
    ax.grid(1)

axes[0].set_xlabel(r'$t\ (\mathrm{s})$', fontsize=13, labelpad=2)
axes[0].set_ylabel(r'$\|\delta x_i\|^2$', fontsize=13, labelpad=5)
# axes[0].set_yticks([0, 2, 4, 6])
# axes[0].set_yticklabels(['0.0', '2.0', '4.0', '6.0'])
for j, _ in enumerate(neighbors):
    axes[0].semilogy(
        t,
        np.square(hatq[:, j] - q[:, j]).sum(axis=-1),
        lw=2.0,
        ds='steps-post',
        label=fr'$p_{{ij_{j+1}}}$'
    )

E = np.matmul(R[:, a].swapaxes(1, 2), hatQ)
delta_theta = np.arccos((np.trace(E, axis1=1, axis2=2) - 1)/2)
# ax.set_ylabel(
#     r'$\mathrm{tr}\left(I - \tilde{Q}_i\right) / 2$',
#     fontsize=15
# )
axes[0].semilogy(
    t,
    delta_theta**2,
    lw=2.0,
    ds='steps-post',
    label=r'$\theta_i$'
)
axes[0].set_ylim(1e-5, 1e2)

axes[0].legend(
    fontsize=12,
    ncols=5,
    labelspacing=0.2,
    handlelength=0.8,
    handletextpad=0.3,
    columnspacing=0.6,
    loc='upper right'
)

axes[1].set_xlabel(r'$t\ (\mathrm{s})$', fontsize=13)
axes[1].set_ylabel(r'$\mathrm{tr}(P_i)$', fontsize=13)
# axes[1].set_yticks([0, 1, 2, 3])
# axes[1].set_yticklabels(['0.0', '1.0', '2.0', '3.0'])

cov_diag = cov_matrix[:, np.eye(3*n, 3*n).astype(bool)]
cov_diag_pij = cov_diag[:, :-3].reshape(-1, n - 1, 3)
cov_diag_Ri = cov_diag[:, -3:]

for j, _ in enumerate(neighbors):
    axes[1].semilogy(
        t,
        cov_diag_pij[:, j].sum(axis=-1),
        lw=2.0,
        ds='steps-post',
        label=fr'$p_{{ij_{j+1}}}$'
    )
axes[1].semilogy(
    t,
    cov_diag_Ri.reshape(-1, 1, 3).sum(axis=-1),
    lw=2.0,
    ds='steps-post',
    label=r'$\theta_i$'
)
axes[1].set_ylim(1e-5, 1e2)

axes[1].legend(
    fontsize=12,
    ncols=5,
    labelspacing=0.2,
    handlelength=0.8,
    handletextpad=0.3,
    columnspacing=0.6,
    loc='upper right'
)

fig.savefig('time_plots/pose_error.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot control
# ------------------------------------------------------------------
fig, ax = plt.subplots(3, 2, figsize=(18.0, 6.0))
fig.subplots_adjust(
    bottom=0.215,
    top=0.925,
    wspace=0.33,
    right=0.975,
    left=0.18
)
fig.tight_layout()

for k, d in enumerate(['x', 'y', 'z']):
    ax[k, 0].tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize=9
    )

    ax[k, 0].set_xlabel(r'$t\ (\mathrm{s})$', fontsize=10)
    ax[k, 0].set_ylabel(fr'$u_{{i, {d}}} \ (\rm m / s)$', fontsize=10)
    ax[k, 0].set_ylim(-2.0, 2.0)
    ax[k, 0].grid(1)

    ax[k, 0].plot(t, control_u[:, :, k], lw=1.0, ds='steps-post')

    ax[k, 1].tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize=9
    )

    ax[k, 1].set_xlabel(r'$t\ (\mathrm{s})$', fontsize=10)
    ax[k, 1].set_ylabel(fr'$w^i_{{i, {d}}} \ (\rm m / s)$', fontsize=10)
    ax[k, 1].set_ylim(-2.0, 2.0)
    ax[k, 1].grid(1)

    ax[k, 1].plot(t, control_w[:, :, k], lw=1.0, ds='steps-post')

fig.savefig('time_plots/control.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot 3d trajectories
# ------------------------------------------------------------------
for k_i, k in enumerate([0, 29999]):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(4, 4))
    # fig.tight_layout()
    fig.subplots_adjust(
        bottom=0.0,
        top=1.0,
        right=0.85,
        left=0.0
    )
    ax.tick_params(
        axis='x',       # changes apply to the x-axis
        which='major',      # both major and minor ticks are affected
        pad=-5,
        labelsize='10'
    )
    ax.tick_params(
        axis='y',       # changes apply to the x-axis
        which='major',      # both major and minor ticks are affected
        pad=1,
        labelsize='10'
    )
    ax.tick_params(
        axis='z',       # changes apply to the x-axis
        which='major',      # both major and minor ticks are affected
        pad=-3,
        labelsize='10'
    )
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x \ (\mathrm{m})$', fontsize='10', labelpad=-5.0)
    ax.set_ylabel(r'$y \ (\mathrm{m})$', fontsize='10', labelpad=0.5)
    ax.set_zlabel(r'$z \ (\mathrm{m})$', fontsize='10', labelpad=-8.0)

    xy_lim = 30.0
    z_lim = xy_lim
    ax.set_xlim3d(0.0, xy_lim)
    ax.set_ylim3d(0.0, xy_lim)
    ax.set_zlim3d(0.0, z_lim)
    ax.set_xticks(np.linspace(0.0, xy_lim, num=3, endpoint=True))
    ax.set_yticks(np.linspace(0.0, xy_lim, num=3, endpoint=True))
    ax.set_zticks(np.linspace(0.0, z_lim, num=3, endpoint=True))

    ax.view_init(elev=5.0, azim=-20.0)
    ax.set_box_aspect(None, zoom=1.0)

    for i in np.arange(n):
        if i == a:
            ax.scatter(
                p[0, i, 0], p[0, i, 1], p[0, i, 2],
                marker='s', s=30, color='k', facecolor='none', zorder=10
            )
            if k_i != 0:
                ax.scatter(
                    p[k, i, 0], p[k, i, 1], p[k, i, 2],
                    marker='s', s=30, color='k', zorder=10
                )
        else:
            ax.scatter(
                p[0, i, 0], p[0, i, 1], p[0, i, 2],
                marker='o', s=20, color='k', facecolor='none', zorder=10
            )
            if k_i != 0:
                ax.scatter(
                    p[k, i, 0], p[k, i, 1], p[k, i, 2],
                    marker='o', s=20, color='k', zorder=10
                )
        ax.plot(p[0:k:400, i, 0], p[0:k:400, i, 1], p[0:k:400, i, 2], ls='--', zorder=0)

    if arg.targets:
        ax.scatter(
            targets_positions[active_targets[k], 0],
            targets_positions[active_targets[k], 1],
            targets_positions[active_targets[k], 2],
            marker='d', s=15, color='k', facecolor='none', zorder=10
        )

    bars(
        ax,
        p[a],
        [[a, j] for j in neighbors],
        color='0.0',
        alpha=0.5,
        lw=1.0,
        zorder=0,
    )
    fig.savefig(f'time_plots/trajectory_{k_i}.pdf')

# plt.show()
