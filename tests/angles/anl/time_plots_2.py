#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from uvnpy.toolkit.data import read_csv_numpy
from uvnpy.toolkit import plot
from uvnpy.angles.local_frame.core import angle_function, angle_indices
from uvnpy.graphs.core import edges_from_adjacency

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'

# ------------------------------------------------------------------
# Read simulated data
# ------------------------------------------------------------------
t = read_csv_numpy('simu_data/t.csv')
log_num_steps = len(t)

p = read_csv_numpy('simu_data/position.csv').reshape(log_num_steps, -1, 3)
n = len(p[0])

R = read_csv_numpy('simu_data/orientation.csv').reshape(log_num_steps, n, 3, 3)

hatp = read_csv_numpy(
    'simu_data/estimated_position.csv'
).reshape(log_num_steps, n, 3)

control_u = read_csv_numpy('simu_data/control_u.csv').reshape(log_num_steps, n, 3)
control_w = read_csv_numpy('simu_data/control_w.csv').reshape(log_num_steps, n, 3)

hatR = read_csv_numpy(
    'simu_data/estimated_orientation.csv').reshape(log_num_steps, n, 3, 3)

adjacency = read_csv_numpy('simu_data/adjacency.csv').reshape(n, n)

edge_set = edges_from_adjacency(adjacency)
angle_set = angle_indices(np.arange(n), edge_set).astype(int)
leaders = np.unique(angle_set[:, 0])
followers = np.setdiff1d(np.arange(n), leaders)
a, b, c = 0, 1, 2

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
fig, axes = plt.subplots(2, 1, figsize=(4.0, 3.5))
fig.tight_layout()
fig.subplots_adjust(hspace=0.35)

for ax in axes:
    ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize=12
    )
    ax.grid(1)

axes[0].set_xlabel(r'$t\ (\mathrm{s})$', fontsize=12, labelpad=-2)
axes[0].set_ylabel(r'$\|\hat{p}_i - p_i\| (\rm m)$', fontsize=14, labelpad=5)
axes[0].set_yticks([0.0, 5.0])
axes[0].set_yticklabels(['0.0', '5.0'])
axes[0].plot(
    t,
    np.sqrt(np.square(hatp - p).sum(axis=-1)),
    lw=2.0,
    ls='-',
    ds='steps-post'
)

E = np.matmul(R.swapaxes(2, 3), hatR)
phi = np.arccos((np.trace(E, axis1=2, axis2=3) - 1)/2)
axes[1].set_xlabel(r'$t\ (\mathrm{s})$', fontsize=12, labelpad=-2)
axes[1].set_ylabel(r'$\|\psi_i\| \ (\rm rad)$', fontsize=14, labelpad=5)
axes[1].set_yticks([0.0, 0.5])
axes[1].set_yticklabels(['0.0', '0.5'])
axes[1].plot(
    t,
    phi,
    lw=2.0,
    ls='-',
    ds='steps-post'
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
    hatp[:, np.newaxis, :, :] - hatp[:, :, np.newaxis, :],
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

for i in range(n):
    ax.scatter(
        p[0, i, 0], p[0, i, 1], p[0, i, 2],
        marker='o', s=25, color='k', zorder=10
    )
    ax.scatter(
        p[-1, i, 0], p[-1, i, 1], p[-1, i, 2],
        marker='x', s=25, color='k', zorder=10
    )
    ax.plot(p[1::400, i, 0], p[1::400, i, 1], p[1::400, i, 2], ls='-', zorder=0)

plot.arrows(
    ax,
    p[0],
    edge_set,
    color='0.0',
    alpha=0.5,
    lw=0.75,
    zorder=0,
    length=0.4,
    arrow_length_ratio=0.2
)
fig.savefig('time_plots/trajectory.pdf', bbox_inches='tight')

plt.show()
