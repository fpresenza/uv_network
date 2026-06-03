#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

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
R = read_csv_numpy('simu_data/orientation.csv').reshape(-1, n, 3, 3)
hatq = read_csv_numpy('simu_data/estimated_position.csv').reshape(-1, n - 1, 3)
hatQ = read_csv_numpy('simu_data/estimated_orientation.csv').reshape(-1, 3, 3)

cov_matrix = read_csv_numpy('simu_data/covariance.csv').reshape(-1, 3*n, 3*n)

control_u = read_csv_numpy('simu_data/control_u.csv').reshape(-1, n, 3)
control_w = read_csv_numpy('simu_data/control_w.csv').reshape(-1, n, 3)

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
axes[0].set_ylabel(r'$\|\hat{p}_{ij} - p_{ij}\| (\rm m)$', fontsize=14, labelpad=5)
axes[0].plot(
    t,
    np.sqrt(np.square(hatq - q).sum(axis=-1)),
    lw=2.0,
    ls='-',
    ds='steps-post'
)

E = np.matmul(R[:, a].swapaxes(1, 2), hatQ)
delta_theta = np.arccos((np.trace(E, axis1=1, axis2=2) - 1)/2)
axes[1].set_xlabel(r'$t\ (\mathrm{s})$', fontsize=12, labelpad=2)
# ax.set_ylabel(
#     r'$\mathrm{tr}\left(I - \tilde{Q}_i\right) / 2$',
#     fontsize=15
# )
axes[1].set_ylabel(r'$\|\delta \theta_i\| \ (\rm rad)$', fontsize=14, labelpad=5)
axes[1].plot(
    t,
    delta_theta,
    lw=2.0,
    ls='-',
    ds='steps-post'
)

fig.savefig('time_plots/pose_error.pdf', bbox_inches='tight')


# ------------------------------------------------------------------
# Plot left-invariant covariance
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.5, 4.5))

ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize=9
)

ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=10)
ax.set_ylabel(r'$\mathrm{tr}(P)$', fontsize=10)
ax.grid(1)
# ax[k].set_ylim(-10.0, 50.0)

cov_diag = cov_matrix[:, np.eye(3*n, 3*n).astype(bool)]
cov_diag_pij = cov_diag[:, :-3]
cov_diag_Ri = cov_diag[:, -3:]

ax.plot(
    t,
    cov_diag_pij.reshape(-1, n-1, 3).sum(axis=-1),
    lw=1.0,
    ds='steps-post',
)
ax.plot(
    t,
    cov_diag_Ri.reshape(-1, 1, 3).sum(axis=-1),
    lw=1.0,
    ls='--',
    ds='steps-post',
)

fig.savefig('time_plots/covariance.pdf', bbox_inches='tight')

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

plt.show()
