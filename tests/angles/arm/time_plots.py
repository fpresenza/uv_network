#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
control = read_csv_numpy('simu_data/control.csv').reshape(log_num_steps - 1, n, 6)
rigidity_val = read_csv_numpy('simu_data/rigidity_val.csv')

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
euler_angles = np.empty((log_num_steps, 3, 3), dtype=np.float64)
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
    ax[k].set_ylabel(fr'$u_{{i, {d}}} \ (\rm m / s^2)$', fontsize=10)
    # ax[k].set_ylim(-1e-4, 1e-4)
    ax[k].grid(1)

    ax[k].plot(t[1:], control[:, :, k], lw=1.0, ds='steps-post')

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
    w = control[k, :, 3:]
    control[k, :, 3:] = np.squeeze(np.matmul(w[:, np.newaxis, :], orientation[k]))

for k, d in enumerate(['x', 'y', 'z']):
    ax[k].tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize=9
    )

    ax[k].set_xlabel(r'$t\ (\mathrm{s})$', fontsize=10)
    ax[k].set_ylabel(fr'$\omega_{{i, {d}}} \ (\rm rad / s)$', fontsize=10)
    ax[k].set_ylim(-1.5, 1.5)
    ax[k].grid(1)

    ax[k].plot(t[1:], control[:, :, 3 + k], lw=1.0, ds='steps-post')

fig.savefig('time_plots/control_w.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot rigidity eigenvalue
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
# ax.set_ylabel(r'$\lambda$', fontsize=10)
# ax.set_ylim(-1e-4, 1e-4)
ax.grid(1)

ax.plot(
    t[1:], rigidity_val[:, 0],
    lw=1.0, ds='steps-post', color='C0', label=r'$\lambda_8$', ls='--'
)
ax.plot(
    t[1:], rigidity_val[:, 1],
    lw=1.0, ds='steps-post', color='C0', label=r'$\widehat{\lambda}_8$'
)
ax.plot(
    t[1:], rigidity_val[:, 2],
    lw=1.0, ds='steps-post', color='C1', label=r'$\widehat{\lambda}_9$'
)
ax.legend(
    fontsize=10, handlelength=1.5, labelspacing=0.4,
    borderpad=0.2, handletextpad=0.2, framealpha=1.,
    ncol=1, columnspacing=1
)

fig.savefig('time_plots/rigidity_val.pdf', bbox_inches='tight')

plt.show()
