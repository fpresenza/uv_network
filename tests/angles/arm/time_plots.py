#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from transformations import unit_vector
from scipy.spatial.transform import Rotation

from uvnpy.toolkit.data import read_csv_numpy, read_json_file

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
n = position.shape[1]

orientation = read_csv_numpy(
    'simu_data/orientation.csv'
).reshape(log_num_steps, n, 3, 3)

# velocity = read_csv_numpy('simu_data/velocity.csv').reshape(log_num_steps, n, 3)

control_u = read_csv_numpy('simu_data/control_u.csv').reshape(log_num_steps - 1, n, 3)
control_w = read_csv_numpy('simu_data/control_w.csv').reshape(log_num_steps - 1, n, 3)

rigidity_val = read_csv_numpy('simu_data/rigidity_val.csv')

targets = read_json_file('simu_data/targets.jsonlog')
tracking_nodes = targets['tracking_nodes']
targets_ids = targets['alloc']

target_position = read_csv_numpy(
    'simu_data/target_position.csv'
).reshape(log_num_steps, -1, 3)

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
    ax[k].set_ylabel(fr'$p_{{i, {d}}} \ (\rm m)$', fontsize=10)
    ax[k].set_ylim(0.0, 100.0)
    ax[k].grid(1)

    ax[k].plot(t, position[:, :, k], lw=1.0, ds='steps-post')

fig.savefig('time_plots/position.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot velocities
# ------------------------------------------------------------------
# fig, ax = plt.subplots(3, 1, figsize=(9.0, 6.0))
# fig.subplots_adjust(
#     bottom=0.215,
#     top=0.925,
#     wspace=0.33,
#     right=0.975,
#     left=0.18
# )

# for k, d in enumerate(['x', 'y', 'z']):
#     ax[k].tick_params(
#         axis='both',       # changes apply to the x-axis
#         which='both',      # both major and minor ticks are affected
#         pad=1,
#         labelsize=9
#     )

#     ax[k].set_xlabel(r'$t\ (\mathrm{s})$', fontsize=10)
#     ax[k].set_ylabel(fr'$v_{{i, {d}}} \ (\rm m / s)$', fontsize=10)
#     ax[k].set_ylim(-0.5, 0.5)
#     ax[k].grid(1)

#     ax[k].plot(t, velocity[:, :, k], lw=1.0, ds='steps-post')

# fig.savefig('time_plots/velocity.pdf', bbox_inches='tight')

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
        orientation[k]
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
    ax[k].set_ylabel(fr'$u_{{i, {d}}} \ (\mathrm{{m s}}^{{-1}})$', fontsize=10)
    # ax[k].set_ylim(-1e-4, 1e-4)
    ax[k].grid(1)

    ax[k].plot(t[1:], control_u[:, :, k], lw=1.0, ds='steps-post')

fig.savefig('time_plots/control_u.pdf', bbox_inches='tight')

fig, ax = plt.subplots(3, 1, figsize=(9.0, 6.0))
fig.subplots_adjust(
    bottom=0.215,
    top=0.925,
    wspace=0.33,
    right=0.975,
    left=0.18
)

# convert angular velocity to local frame
for k in range(log_num_steps - 1):
    control_w[k] = np.squeeze(np.matmul(
        control_w[k, :, np.newaxis, :], orientation[k]
    ))

for k, d in enumerate(['x', 'y', 'z']):
    ax[k].tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize=9
    )

    ax[k].set_xlabel(r'$t\ (\mathrm{s})$', fontsize=10)
    ax[k].set_ylabel(fr'$\omega_{{i, {d}}} \ (\rm rad / s)$', fontsize=10)
    ax[k].set_ylim(-2.5, 2.5)
    ax[k].grid(1)

    ax[k].plot(t[1:], control_w[:, :, k], lw=1.0, ds='steps-post')

fig.savefig('time_plots/control_w.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot target tracking metrics
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.0, 3.0))
fig.tight_layout()

ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize=3
)

ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=15)
ax.set_ylabel(r'$d_{i \tau_i} \ (\rm m)$', fontsize=15)
ax.set_ylim(0.0, 100.0)
ax.set_xticks(np.linspace(t[0], t[-1], 5))
ax.grid(1)

ax.plot(
    t, np.sqrt(np.square(
        target_position[:, targets_ids] - position[:, tracking_nodes]
    ).sum(axis=-1)),
    lw=2.0, ds='steps-post'
)
# ax.hlines(30.0, t[0], t[-1], ls='--', color='k')
fig.savefig('time_plots/tracking_distance.pdf', bbox_inches='tight')


fig, ax = plt.subplots(figsize=(9.0, 3.0))
fig.tight_layout()

ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize=15
)

ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=15)
ax.set_ylabel(r'$\arccos(\zeta_{i \tau_i})$', fontsize=15)
# ax.set_ylim(bottom=0.0)
ax.set_xticks(np.linspace(t[0], t[-1], 5))
ax.set_yticks(np.linspace(0, 360, 6, endpoint=False))
ax.grid(1)

ax.plot(
    t,
    np.rad2deg(np.arccos(np.sum(
        unit_vector(
            target_position[:, targets_ids] - position[:, tracking_nodes],
            axis=-1
        ) * orientation[:, tracking_nodes, :, 0].swapaxes(0, 1),
        axis=-1
    ))),
    lw=2.0, ds='steps-post'
)
ax.hlines(60.0, t[0], t[-1], ls='--', color='k', lw=1.0)
ax.text(90.0, 63.0, r'$60\degree$', size=15)

fig.savefig('time_plots/tracking_aiming.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot rigidity eigenvalue
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.0, 3.0))
fig.tight_layout()

ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize=15
)

ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=15)
# ax.set_ylabel(r'$\lambda$', fontsize=10)
# ax.set_ylim(-1e-4, 1e-4)
ax.set_xticks(np.linspace(t[0], t[-1], 5))
ax.grid(1)

ax.semilogy(
    t[1:], rigidity_val[:, 0],
    lw=2.0, ds='steps-post', color='C0', label=r'$\lambda_8$', ls='--'
)
ax.semilogy(
    t[1:], rigidity_val[:, 1],
    lw=2.0, ds='steps-post', color='C0', label=r'$\widetilde{\lambda}_8$'
)
# ax.semilogy(
#     t[1:], rigidity_val[:, 2],
#     lw=1.0, ds='steps-post', color='C1', label=r'$\widetilde{\lambda}_9$'
# )
ax.legend(
    fontsize=15, handlelength=1.5, labelspacing=0.4,
    borderpad=0.2, handletextpad=0.2, framealpha=1.,
    ncol=1, columnspacing=1
)

fig.savefig('time_plots/rigidity_val.pdf', bbox_inches='tight')

plt.show()
