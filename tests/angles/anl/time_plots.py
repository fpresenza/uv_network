#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from uvnpy.toolkit.data import read_csv_numpy
from uvnpy.angles.local_frame.core import angle_function
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

position = read_csv_numpy('simu_data/position.csv').reshape(log_num_steps, -1, 3)
n = len(position[0])

estimated_position = read_csv_numpy(
    'simu_data/estimated_position.csv'
).reshape(log_num_steps, -1, 3)

control_u = read_csv_numpy('simu_data/control_u.csv').reshape(log_num_steps, n, 3)

correction = read_csv_numpy('simu_data/correction.csv').reshape(log_num_steps, n, 3)

adjacency = read_csv_numpy('simu_data/adjacency.csv').reshape(n, n)

edge_set = edges_from_adjacency(adjacency)
a = 0

# ------------------------------------------------------------------
# Plot positions
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
    ax[k].set_ylabel(fr'$\hat{{p}}_{{i, {d}}} \ (\rm m)$', fontsize=10)
    ax[k].grid(1)

    ax[k].plot(
        t,
        (
            estimated_position[:, :, k] -
            position[:, :, k] +
            position[:, a, np.newaxis, k]
        ),
        lw=1.0,
        ds='steps-post'
    )

fig.savefig('time_plots/estimated_position.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot control
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
    ax[k].set_ylabel(fr'$u_{{i, {d}}} \ (\rm m / s)$', fontsize=10)
    # ax[k].set_ylim(-1e-4, 1e-4)
    ax[k].grid(1)

    ax[k].plot(t, control_u[:, :, k], lw=1.0, ds='steps-post')

fig.savefig('time_plots/control_u.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot correction
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
    ax[k].set_ylabel(fr'$\delta \hat{{p}}_{{i, {d}}} \ (\rm m / s)$', fontsize=10)
    # ax[k].set_ylim(-1e-4, 1e-4)
    ax[k].grid(1)

    ax[k].plot(t, correction[:, :, k], lw=1.0, ds='steps-post')

fig.savefig('time_plots/correction.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot angle errors
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
        np.abs(angle_function(edge_set, hatp) - angle_function(edge_set, p))
        for hatp, p in zip(estimated_position, position)
    ],
    lw=1.0, ds='steps-post'
)
fig.savefig('time_plots/angle_errors.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot distance
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
fig.savefig('time_plots/distance.pdf', bbox_inches='tight')

plt.show()
