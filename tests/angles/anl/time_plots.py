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

estimated_position = read_csv_numpy(
    'simu_data/estimated_position.csv'
).reshape(log_num_steps, -1, 3)
n = len(estimated_position[0])
position = read_csv_numpy('simu_data/position.csv').reshape(-1, 3)
adjacency = read_csv_numpy('simu_data/adjacency.csv').reshape(n, n)

edge_set = edges_from_adjacency(adjacency)

angles = angle_function(edge_set, position)
angles = [angles for _ in t]

kappa, ell = edge_set[0]
distance = np.linalg.norm(position[kappa] - position[ell])
distance = [distance for _ in t]

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
    ax[k].set_ylim(-1.5, 1.5)
    ax[k].grid(1)

    ax[k].plot(t, estimated_position[:, :, k], lw=1.0, ds='steps-post')

fig.savefig('time_plots/estimated_position.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot angles
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
ax.set_ylabel(r'$a_{ijk}$', fontsize=10)
ax.set_ylim(-1.0, 1.0)
ax.grid(1)

ax.plot(
    t,
    [angle_function(edge_set, p) for p in estimated_position],
    lw=1.0, ds='steps-post'
)
ax.plot([], [], color='k', label='estimated')
ax.set_prop_cycle(None)    # resets color counter
ax.plot(t, angles, lw=0.8, ls='--')
ax.plot([], [], color='k', ls='--', label='real')
ax.legend(
    fontsize=10, handlelength=1.5, labelspacing=0.4,
    borderpad=0.2, handletextpad=0.2, framealpha=1.,
    ncol=2, columnspacing=1
)

fig.savefig('time_plots/angles.pdf', bbox_inches='tight')

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
ax.set_ylim(0.0, None)
ax.grid(1)

ax.plot(
    t,
    [
        np.abs(angle_function(edge_set, estimated_position[k]) - angles[k])
        for k in range(log_num_steps)
    ],
    lw=1.0, ds='steps-post'
)
fig.savefig('time_plots/angle_errors.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot distance
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
ax.set_ylabel(r'$d_{\kappa \ell}$', fontsize=10)
ax.set_ylim(-1.0, 1.0)
ax.grid(1)

ax.plot(
    t,
    [np.linalg.norm(p[kappa] - p[ell]) for p in estimated_position],
    lw=1.0, ds='steps-post'
)
ax.plot([], [], color='k', label='estimated')
ax.set_prop_cycle(None)    # resets color counter
ax.plot(t, distance, lw=0.8, ls='--')
ax.plot([], [], color='k', ls='--', label='real')
ax.legend(
    fontsize=10, handlelength=1.5, labelspacing=0.4,
    borderpad=0.2, handletextpad=0.2, framealpha=1.,
    ncol=2, columnspacing=1
)

fig.savefig('time_plots/distance.pdf', bbox_inches='tight')

plt.show()
