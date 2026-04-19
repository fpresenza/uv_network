#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt

from uvnpy.toolkit.data import read_csv_numpy
from uvnpy.angles.local_frame.core import angle_function
# from uvnpy.graphs.core import edges_from_adjacency
from uvnpy.graphs.core import complete_edges
from uvnpy.toolkit.geometry import optimal_rigid_transform

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
t = read_csv_numpy('input_data/t.csv')
log_num_steps = len(t)

position = read_csv_numpy(
    'input_data/position.csv'
).reshape(log_num_steps, -1, 3)

n = position.shape[1]
nodes = np.arange(n)

orientation = read_csv_numpy(
    'input_data/orientation.csv'
).reshape(log_num_steps, n, 3, 3)

estimated_position = read_csv_numpy(
    'simu_data/estimated_position.csv'
).reshape(log_num_steps, n, 3)

control_u = read_csv_numpy('input_data/control_u.csv').reshape(log_num_steps - 1, n, 3)
control_w = read_csv_numpy('input_data/control_w.csv').reshape(log_num_steps - 1, n, 3)

correction_u = read_csv_numpy('simu_data/correction_u.csv').reshape(log_num_steps, n, 3)

if arg.coupled:
    estimated_orientation = read_csv_numpy(
        'simu_data/estimated_orientation.csv'
    ).reshape(log_num_steps, n, 3, 3)
    correction_w = read_csv_numpy(
        'simu_data/correction_w.csv'
    ).reshape(log_num_steps, n, 3)

adjacency = read_csv_numpy(
    'input_data/adjacency.csv'
).reshape(log_num_steps, n, n)


# edge_set = [edges_from_adjacency(adj) for adj in adjacency]
edge_set = complete_edges(n)

a = 0

# position reconstruction
# rotated_position = np.matmul(
#     orientation[:, a], estimated_position.swapaxes(1, 2)
# ).swapaxes(1, 2)
# reconstructed_position = position[:, np.newaxis, a] + rotated_position
reconstructed_position = np.empty(shape=estimated_position.shape)
for k in range(log_num_steps):
    U, v = optimal_rigid_transform(estimated_position[k], position[k])
    reconstructed_position[k] = estimated_position[k].dot(U.T) + v

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
        position[:, :, k],
        lw=1.0,
        ds='steps-post',
    )
    ax[k].plot(
        t,
        reconstructed_position[:, :, k],
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
    ax[k].set_ylabel(fr'$\hat{{p}}_{{i, {d}}} - p_{{i, {d}}} \ (\rm m)$', fontsize=10)
    ax[k].grid(1)

    ax[k].plot(
        t,
        reconstructed_position[:, :, k] - position[:, :, k],
        lw=1.0,
        ds='steps-post'
    )

fig.savefig('time_plots/position_error.pdf', bbox_inches='tight')

fig, ax = plt.subplots(figsize=(9.0, 2.0))
fig.tight_layout()

ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize=15
)

ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=15)
ax.set_ylabel(r'$\| \hat{{p}}_{i} - p_{i} \| \ (\rm m)$', fontsize=15)
ax.grid(1)

ax.plot(
    t,
    np.sqrt(np.square(reconstructed_position - position).sum(axis=-1)),
    lw=2.0,
    ds='steps-post'
)

fig.savefig('time_plots/position_error_agg.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot orientation error
# ------------------------------------------------------------------
# position reconstruction
# Q = orientation
# hatR = np.matmul(estimated_orientation[:, a, np.newaxis], estimated_orientation)

if arg.coupled:
    Q = np.matmul(orientation[:, a, np.newaxis].swapaxes(2, 3), orientation)
    hatQ = estimated_orientation

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

    ax[k, 0].plot(t[1:], control_u[:, :, k], lw=1.0, ds='steps-post')

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

    ax[k, 1].plot(t[1:], control_w[:, :, k], lw=1.0, ds='steps-post')

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
        np.square(angle_function(edge_set, hatp) - angle_function(edge_set, p))
        for hatp, p in zip(estimated_position, position)
    ],
    lw=1.0, ds='steps-post'
)
fig.savefig('time_plots/angle_error.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot distance error
# ------------------------------------------------------------------
distance = np.linalg.norm(
    position[:, np.newaxis, :, :] - position[:, :, np.newaxis, :],
    axis=-1
)
estimated_distance = np.linalg.norm(
    estimated_position[:, np.newaxis, :, :] - estimated_position[:, :, np.newaxis, :],
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

plt.show()
