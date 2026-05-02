#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from uvnpy.toolkit.data import read_csv_numpy
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

gradient = read_csv_numpy('simu_data/gradient.csv').reshape(log_num_steps, n, 3)

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

fig.savefig('time_plots/position_error.pdf', bbox_inches='tight')

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
# Plot orientation error
# ------------------------------------------------------------------
# position reconstruction
# Q = R
# hatR = np.matmul(hatQ[:, a, np.newaxis], hatQ)

if arg.coupled:
    E = np.matmul(Q.swapaxes(2, 3), hatQ)
    e = 0.5 * (3 - np.trace(E, axis1=2, axis2=3))

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
    ax.set_ylabel(
        r'$\frac{1}{2} \  \mathrm{tr}\left(I - Q_i^\top \hat{Q}_{i}\right) \ (\rm m)$',
        fontsize=10
    )
    ax.grid(1)

    ax.plot(
        t,
        e,
        lw=1.0,
        ds='steps-post'
    )

    fig.savefig('time_plots/orientation_error.pdf', bbox_inches='tight')

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

    ax[k].plot(t, gradient[:, :, k], lw=1.0, ds='steps-post')

fig.savefig('time_plots/gradient.pdf', bbox_inches='tight')

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

plt.show()

plt.show()
