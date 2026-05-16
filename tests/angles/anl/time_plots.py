#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from uvnpy.toolkit.data import read_csv_numpy
from uvnpy.toolkit import plot
from uvnpy.angles.local_frame.core import (
    angle_function, angle_indices, angle_rigidity_matrix
)
from uvnpy.graphs.core import edges_from_adjacency

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
    '-c', '--coupled',
    default=False, action='store_true', help='Coupled or uncoupled estimation.'
)
arg = parser.parse_args()

# ------------------------------------------------------------------
# Read simulated data
# ------------------------------------------------------------------
t = read_csv_numpy('simu_data/t.csv')
log_num_steps = len(t)

p = read_csv_numpy('simu_data/position.csv').reshape(log_num_steps, -1, 3)
n = len(p[0])

R = read_csv_numpy(
    'simu_data/orientation.csv'
).reshape(log_num_steps, n, 3, 3)

hatq = read_csv_numpy(
    'simu_data/estimated_position.csv'
).reshape(log_num_steps, n, 3)

gradient_q = read_csv_numpy('simu_data/gradient_q.csv').reshape(log_num_steps, n, 3)

control_u = read_csv_numpy('simu_data/control_u.csv').reshape(log_num_steps, n, 3)
control_w = read_csv_numpy('simu_data/control_w.csv').reshape(log_num_steps, n, 3)

correction_u = read_csv_numpy('simu_data/correction_u.csv').reshape(log_num_steps, n, 3)

if arg.coupled:
    hatQ = read_csv_numpy(
        'simu_data/estimated_orientation.csv'
    ).reshape(log_num_steps, n, 3, 3)
    correction_w = read_csv_numpy(
        'simu_data/correction_w.csv'
    ).reshape(log_num_steps, n, 3)

adjacency = read_csv_numpy('simu_data/adjacency.csv').reshape(n, n)

edge_set = edges_from_adjacency(adjacency)
angle_set = angle_indices(np.arange(n), edge_set).astype(int)
a, b, c = 0, 1, 2

# change of basis
hatp = np.matmul(hatq, R[:, a].swapaxes(1, 2)) + p[:, np.newaxis, a]
q = np.matmul(p - p[:, np.newaxis, a], R[:, a])
if arg.coupled:
    Q = np.matmul(R[:, a, np.newaxis].swapaxes(2, 3), R)

# Hessian
A = [angle_rigidity_matrix(angle_set, pk) for pk in p]
eb = np.eye(n)
S = np.kron(np.diag(eb[a] + eb[b] + eb[c]), np.eye(3))

# ------------------------------------------------------------------
# Plot position
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
    ax[k].set_ylabel(fr'$p_{{i, {d}}}, \hat{{p}}_{{i, {d}}} \ (\rm m)$', fontsize=10)
    ax[k].grid(1)
    # ax[k].set_ylim(-10.0, 50.0)

    ax[k].plot(
        t,
        p[:, :, k],
        lw=1.0,
        ds='steps-post',
    )
    ax[k].plot(
        t,
        hatp[:, :, k],
        lw=0.8,
        color='0.5',
        ls='--',
        ds='steps-post',
    )

fig.savefig('time_plots/position.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot position error
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
    # ax[k].set_ylabel(fr'$\hat{{p}}_{{i, {d}}} - p_{{i, {d}}} \ (\rm m)$', fontsize=10)
    # ax[k].grid(1)

    # ax[k].plot(
    #     t,
    #     hatp[:, :, k] - p[:, :, k],
    #     lw=1.0,
    #     ds='steps-post'
    # )
    ax[k].set_ylabel(fr'$\hat{{q}}_{{i, {d}}} - q_{{i, {d}}} \ (\rm m)$', fontsize=10)
    ax[k].grid(1)

    ax[k].plot(
        t,
        hatq[:, :, k] - q[:, :, k],
        lw=1.0,
        ds='steps-post'
    )

# ------------------------------------------------------------------
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
for k in range(log_num_steps):
    euler_angles[k] = Rotation.from_matrix(
        R[k]
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
# Plot pose error
# ------------------------------------------------------------------
if arg.coupled:
    fig, axes = plt.subplots(2, 1, figsize=(4.0, 3.5))
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.45)

    for ax in axes:
        ax.tick_params(
            axis='both',       # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            pad=1,
            labelsize=12
        )
        ax.grid(1)

    axes[0].set_xlabel(r'$t\ (\mathrm{s})$', fontsize=12, labelpad=2)
    axes[0].set_ylabel(r'$\|\tilde{{q}}_{i}\| (\rm m)$', fontsize=14, labelpad=5)
    axes[0].set_yticks([0.0, 5.0])
    axes[0].set_yticklabels(['0.0', '5.0'])
    axes[0].plot(
        t,
        np.sqrt(np.square(hatq[:, :2] - q[:, :2]).sum(axis=-1)),
        lw=2.0,
        ls='-',
        ds='steps-post'
    )
    axes[0].plot(
        t,
        np.sqrt(np.square(hatq[:, 2:] - q[:, 2:]).sum(axis=-1)),
        lw=2.0,
        ls='--',
        ds='steps-post'
    )
    axes[0].plot(0.0, 0.0, color='k', ls='-', label='lead')
    axes[0].plot(0.0, 0.0, color='k', ls='--', label='foll')
    axes[0].legend(fontsize=12)

    E = np.matmul(Q.swapaxes(2, 3), hatQ)
    axes[1].set_xlabel(r'$t\ (\mathrm{s})$', fontsize=12, labelpad=2)
    # ax.set_ylabel(
    #     r'$\mathrm{tr}\left(I - \tilde{Q}_i\right) / 2$',
    #     fontsize=15
    # )
    axes[1].set_ylabel(r'$\|\psi_i\| \ (\rm rad)$', fontsize=14, labelpad=5)
    axes[1].set_yticks([0.0, 0.5])
    axes[1].set_yticklabels(['0.0', '0.5'])
    axes[1].plot(
        t,
        np.arccos(1 - 0.5 * (3 - np.trace(E[:, :2], axis1=2, axis2=3))),
        lw=2.0,
        ls='-',
        ds='steps-post'
    )
    axes[1].plot(
        t,
        np.arccos(1 - 0.5 * (3 - np.trace(E[:, 2:], axis1=2, axis2=3))),
        lw=2.0,
        ls='--',
        ds='steps-post'
    )
    axes[1].plot(0.0, 0.0, color='k', ls='-', label='lead')
    axes[1].plot(0.0, 0.0, color='k', ls='--', label='foll')
    axes[1].legend(fontsize=12)

    fig.savefig('time_plots/pose_error.pdf', bbox_inches='tight')

else:
    fig, ax = plt.subplots(figsize=(9.0, 2.0))
    fig.tight_layout()

    ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize=15
    )

    ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=15)
    ax.set_ylabel(r'$\|\tilde{{q}}_{i}\| \ (\rm m)$', fontsize=15)
    ax.grid(1)

    ax.plot(
        t,
        np.sqrt(np.square(hatq - q).sum(axis=-1)),
        lw=2.0,
        ds='steps-post'
    )

    fig.savefig('time_plots/position_error_norm.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot gradient
# ------------------------------------------------------------------
fig, ax = plt.subplots(3, 1, figsize=(9.0, 6.0))
fig.tight_layout()

for k, d in enumerate(['x', 'y', 'z']):
    ax[k].tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize=9
    )

    ax[k].set_xlabel(r'$t\ (\mathrm{s})$', fontsize=10)
    ax[k].set_ylabel(fr'$\zeta_{{i, {d}}} \ (\rm m / s)$', fontsize=10)
    # ax[k].set_ylim(-2.0, 2.0)
    ax[k].grid(1)

    ax[k].plot(t, gradient_q[:, :, k], lw=1.0, ds='steps-post')

fig.savefig('time_plots/gradient_q.pdf', bbox_inches='tight')

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
# Plot correction
# ------------------------------------------------------------------
if arg.coupled:
    fig, ax = plt.subplots(3, 2, figsize=(18.0, 6.0))
    fig.subplots_adjust(
        bottom=0.215,
        top=0.925,
        wspace=0.33,
        right=0.975,
        left=0.18
    )

    for k, d in enumerate(['x', 'y', 'z']):
        ax[k, 0].tick_params(
            axis='both',       # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            pad=1,
            labelsize=9
        )

        ax[k, 0].set_xlabel(r'$t\ (\mathrm{s})$', fontsize=10)
        ax[k, 0].set_ylabel(
            fr'$\delta \hat{{p}}_{{i, {d}}} \ (\rm m / s)$', fontsize=10
        )
        # ax[k, 0].set_ylim(-1e-4, 1e-4)
        ax[k, 0].grid(1)

        ax[k, 0].plot(t, correction_u[:, :, k], lw=1.0, ds='steps-post')

        ax[k, 1].tick_params(
            axis='both',       # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            pad=1,
            labelsize=9
        )

        ax[k, 1].set_xlabel(r'$t\ (\mathrm{s})$', fontsize=10)
        ax[k, 1].set_ylabel(
            fr'$\delta \hat{{Q}}_{{i, {d}}} \ (\rm m / s)$', fontsize=10
        )
        # ax[k, 1].set_ylim(-1e-4, 1e-4)
        ax[k, 1].grid(1)

        ax[k, 1].plot(t, correction_w[:, :, k], lw=1.0, ds='steps-post')

    fig.savefig('time_plots/correction.pdf', bbox_inches='tight')

else:

    fig, ax = plt.subplots(3, figsize=(9.0, 6.0))
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
        ax[k].set_ylabel(fr'$\delta \hat{{p}}_{{i, {d}}} \ (\rm m / s)$', fontsize=10)
        # ax[k].set_ylim(-1e-4, 1e-4)
        ax[k].grid(1)

        ax[k].plot(t, correction_u[:, :, k], lw=1.0, ds='steps-post')

    fig.savefig('time_plots/correction.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot angle error
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.0, 6.0))
fig.subplots_adjust(
    bottom=0.215,
    top=0.925,
    wspace=0.33,
    right=0.975,
    left=0.18
)

ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize=9
)

ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=10)
ax.set_ylabel(r'$|\hat{a}_{ijk} - a_{ijk}|$', fontsize=10)
ax.grid(1)

ax.plot(
    t,
    [
        np.abs(angle_function(edge_set, hatpk) - angle_function(edge_set, pk))
        for hatpk, pk in zip(hatp, p)
    ],
    lw=1.0, ds='steps-post'
)
fig.savefig('time_plots/angle_error.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot distance error
# ------------------------------------------------------------------
distance = np.linalg.norm(
    p[:, np.newaxis, :, :] - p[:, :, np.newaxis, :],
    axis=-1
)
estimated_distance = np.linalg.norm(
    hatq[:, np.newaxis, :, :] - hatq[:, :, np.newaxis, :],
    axis=-1
)

fig, ax = plt.subplots(figsize=(9.0, 6.0))
fig.subplots_adjust(
    bottom=0.215,
    top=0.925,
    wspace=0.33,
    right=0.975,
    left=0.18
)

ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize=9
)

ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=10)
ax.set_ylabel(r'$|\hat{d}_{ij} - d_{ij}|$', fontsize=10)
ax.grid(1)

ax.plot(
    t,
    np.unique(
        np.abs(estimated_distance - distance).reshape(log_num_steps, -1), axis=-1
    )[:, 1:],
    lw=1.0,
    ds='steps-post'
)
fig.savefig('time_plots/distance_error.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot Hessian eigenvalues
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.0, 6.0))
fig.subplots_adjust(
    bottom=0.215,
    top=0.925,
    wspace=0.33,
    right=0.975,
    left=0.18
)

ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize=9
)

ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=10)
ax.set_ylabel(r'$\lambda(H(p))$', fontsize=10)
ax.grid(1)

ax.semilogy(
    t,
    np.linalg.eigvalsh([A[k].T.dot(A[k]) + S for k in range(log_num_steps)]),
    lw=1.0,
    ds='steps-post'
)
fig.savefig('time_plots/hessian_eigenvalues.pdf', bbox_inches='tight')


# ------------------------------------------------------------------
# Plot 3d trajectories
# ------------------------------------------------------------------
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(4, 4))
fig.tight_layout()
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

xy_lim = 20.0
z_lim = xy_lim
ax.set_xlim3d(0.0, xy_lim)
ax.set_ylim3d(0.0, xy_lim)
ax.set_zlim3d(0.0, z_lim)
ax.set_xticks(np.linspace(0.0, xy_lim, num=3, endpoint=True))
ax.set_yticks(np.linspace(0.0, xy_lim, num=3, endpoint=True))
ax.set_zticks(np.linspace(0.0, z_lim, num=3, endpoint=True))

ax.view_init(elev=10.0, azim=-15.0)
ax.set_box_aspect(None, zoom=1.0)

for i in range(2):
    ax.scatter(
        p[0, i, 0], p[0, i, 1], p[0, i, 2],
        marker='o', s=12, color='k', zorder=10
    )
    ax.scatter(
        p[-1, i, 0], p[-1, i, 1], p[-1, i, 2],
        marker='x', s=14, color='k', zorder=10
    )
    ax.plot(p[1::400, i, 0], p[1::400, i, 1], p[1::400, i, 2], ls='-', zorder=0)

for i in range(2, n):
    ax.scatter(
        p[0, i, 0], p[0, i, 1], p[0, i, 2],
        marker='o', s=12, color='k', zorder=10
    )
    ax.scatter(
        p[-1, i, 0], p[-1, i, 1], p[-1, i, 2],
        marker='x', s=14, color='k', zorder=10
    )
    ax.plot(p[1::400, i, 0], p[1::400, i, 1], p[1::400, i, 2], ls='--', zorder=0)

plot.arrows(
    ax,
    p[0],
    edge_set,
    color='0.0',
    alpha=0.5,
    lw=0.75,
    zorder=0,
    length=0.45,
    arrow_length_ratio=0.3
)
fig.savefig('time_plots/trajectory.pdf', bbox_inches='tight')

# plt.show()
