#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot bearing-control logs, including position and wrapped yaw errors.

Read ``common_frame/control/simu_data`` and save figures under
``common_frame/control/time_plot``, independently of the working directory.
The pose-error figure follows the angle tests' two-panel layout and includes
first-second insets. It is skipped when ``est_pose.csv`` is absent.
Poses are already logged in the same global frame.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import shortest_path

from uvnpy.toolkit import data
from uvnpy.graphs.core import geodesics, as_undirected
from uvnpy.bearings.common_frame.core import bearing_rigidity_eigenvalue
from uvnpy.distances.core import minimum_distance

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


def pose_errors(pose, estimated_pose):
    """Return position norms and yaw errors in [0, pi] for (..., n, 4) poses."""
    position_error = np.linalg.norm(estimated_pose[..., :3] - pose[..., :3], axis=-1)
    delta_yaw = estimated_pose[..., 3] - pose[..., 3]
    orientation_error = np.abs(np.arctan2(np.sin(delta_yaw), np.cos(delta_yaw)))
    return position_error, orientation_error


# ------------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Plot bearing-control simulation logs.')
parser.add_argument(
    '-i', '--init',
    default=0.0, type=float, help='start time in seconds'
)
parser.add_argument(
    '-e', '--end',
    default=0.0, type=float,
    help='end time in seconds (inclusive); 0 selects the last time'
)
parser.add_argument(
    '-j', '--jump',
    default=1, type=int, help='numbers of frames jumped'
)
parser.add_argument(
    '-s', '--subset',
    default=None, type=int, nargs='+', help='subset of nodes to plot (default: all)'
)
arg = parser.parse_args()
if arg.jump < 1:
    parser.error('--jump must be positive.')

# ------------------------------------------------------------------
# Read simulated data
# ------------------------------------------------------------------
simulation_dir = Path(__file__).resolve().parents[1] / 'control'
data_dir = simulation_dir / 'simu_data'
output_dir = simulation_dir / 'time_plot'
output_dir.mkdir(parents=True, exist_ok=True)

all_t = np.atleast_1d(np.loadtxt(data_dir / 't.csv', delimiter=','))
arg.end = all_t[-1] if arg.end == 0 else arg.end
k_i = int(np.argmin(np.abs(all_t - arg.init)))
k_e = int(np.argmin(np.abs(all_t - arg.end))) + 1
if k_e <= k_i:
    parser.error('the selected time window must contain at least one logged sample.')
t = all_t[k_i:k_e:arg.jump]

# Every per-sample log uses the same starting row and sampling stride.
x = data.read_csv_numpy(
    data_dir / 'pose.csv', start=k_i, stop=k_e, jump=arg.jump
).reshape(len(t), -1, 4)
n = x.shape[1]
nodes = np.arange(n)
subset = nodes if arg.subset is None or arg.subset == [-1] else np.array(arg.subset)
if np.any((subset < 0) | (subset >= n)):
    parser.error('--subset robot indices must be between 0 and n - 1.')

D = data.read_csv_numpy(
    data_dir / 'sens_adj.csv', start=k_i, stop=k_e, jump=arg.jump
).reshape(len(t), n, n)
targets = data.read_csv_numpy(
    data_dir / 'targets.csv', start=k_i, stop=k_e, jump=arg.jump
).reshape(len(t), -1).astype(bool)
targets_positions = np.loadtxt(
    data_dir / 'targets_positions.csv', delimiter=','
).reshape(-1, 3)
u_t = data.read_csv_numpy(
    data_dir / 'target_action.csv', start=k_i, stop=k_e, jump=arg.jump
).reshape(len(t), n, 4)
u_c = data.read_csv_numpy(
    data_dir / 'collision_action.csv', start=k_i, stop=k_e, jump=arg.jump
).reshape(len(t), n, 4)
u_r = data.read_csv_numpy(
    data_dir / 'rigidity_action.csv', start=k_i, stop=k_e, jump=arg.jump
).reshape(len(t), n, 4)
action_extents = data.read_csv_numpy(
    data_dir / 'action_extents.csv', start=k_i, stop=k_e, jump=arg.jump
).reshape(len(t), n)
# Estimation logs are optional; all other figures use the true poses.
has_estimated_pose = (data_dir / 'est_pose.csv').is_file()
if has_estimated_pose:
    hatx = data.read_csv_numpy(
        data_dir / 'est_pose.csv', start=k_i, stop=k_e, jump=arg.jump
    ).reshape(len(t), n, 4)

    # These poses already use the same global frame. The fourth component is yaw;
    # its wrapped absolute difference is the rotation-angle error for yaw rotations.
    position_error, orientation_error = pose_errors(x, hatx)

    # Keep the initial transient at full log resolution, even when --jump > 1.
    zoom_stop = min(k_e, int(np.searchsorted(all_t, 1.0, side='right')))
    zoom_t = all_t[k_i:zoom_stop]
    if zoom_t.size:
        zoom_x = data.read_csv_numpy(
            data_dir / 'pose.csv', start=k_i, stop=zoom_stop
        ).reshape(len(zoom_t), n, 4)
        zoom_hatx = data.read_csv_numpy(
            data_dir / 'est_pose.csv', start=k_i, stop=zoom_stop
        ).reshape(len(zoom_t), n, 4)
        zoom_position_error, zoom_orientation_error = pose_errors(zoom_x, zoom_hatx)
else:
    print('Skipping pose-error plot: est_pose.csv was not found.')

A = as_undirected(D).astype(float)
fre = np.empty(len(t))
re = np.empty((len(t), n))
fdiam = np.empty(len(t))
diam = np.empty((len(t), n))
for k, (adj, pos, ext) in enumerate(zip(A, x[:, :, :3], action_extents)):
    geo = shortest_path(adj, unweighted=True)
    fre[k] = bearing_rigidity_eigenvalue(adj, pos)
    fdiam[k] = np.max(geo)
    for i in nodes:
        Vi = geo[i] <= ext[i]
        if sum(Vi) > 1:
            Ai = adj[np.ix_(Vi, Vi)]
            qi = pos[Vi]
            re[k, i] = bearing_rigidity_eigenvalue(Ai, qi)
            diam[k, i] = np.max(geodesics(Ai))
        else:
            re[k, i] = 0.0
            diam[k, i] = np.inf

# tc = np.loadtxt(data_dir / 'tc.csv', delimiter=',')
# cov = np.loadtxt(data_dir / 'covariance.csv', delimiter=',')

# v = np.loadtxt(data_dir / 'vel_meas_err.csv', delimiter=',')
# g = np.loadtxt(data_dir / 'gps_meas_err.csv', delimiter=',')
# r = np.loadtxt(data_dir / 'range_meas_err.csv', delimiter=',')
# state_extents = np.loadtxt(data_dir / 'state_extents.csv', delimiter=',')


# ------------------------------------------------------------------
# Plot target control action
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.0, 4.0))
fig.subplots_adjust(
    bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small')

ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=12)
ax.set_ylabel(r'$\Vert u_t \Vert \ (\mathrm{m}/\mathrm{s})$', fontsize=12)
ax.grid(1)
ax.plot(
    t,
    np.sqrt(np.sum(u_t[:, subset]**2, axis=2)),
    lw=0.8,
    ds='steps-post'
)
fig.savefig(output_dir / 'target_control.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot collision control action
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.0, 4.0))
fig.subplots_adjust(
    bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small')

ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=12)
ax.set_ylabel(r'$\Vert u_c \Vert \ (\mathrm{m}/\mathrm{s})$', fontsize=12)
ax.grid(1)
ax.plot(
    t,
    np.sqrt(np.sum(u_c[:, subset]**2, axis=2)),
    lw=0.8, ds='steps-post'
)
fig.savefig(output_dir / 'collision_control.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot rigidity control action
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.0, 4.0))
fig.subplots_adjust(
    bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small')

ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=12)
ax.set_ylabel(r'$\Vert u_r \Vert \ (\mathrm{m}/\mathrm{s})$', fontsize=12)
ax.grid(1)
ax.plot(
    t,
    np.sqrt(np.sum(u_r[:, subset]**2, axis=2)),
    lw=0.8, ds='steps-post'
)
fig.savefig(output_dir / 'rigidity_control.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot control action composition
# ------------------------------------------------------------------
u = u_t + u_c + u_r
fig, ax = plt.subplots(figsize=(4.0, 4.0))
fig.subplots_adjust(
    bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small')

ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=12)
ax.set_ylabel(r'$\Vert u \Vert \ (\mathrm{m}/\mathrm{s})$', fontsize=12)
ax.grid(1)
ax.plot(
    t,
    np.sqrt(np.sum(u[:, subset]**2, axis=2)),
    lw=0.8, ds='steps-post'
)
fig.savefig(output_dir / 'control.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot state extents
# ------------------------------------------------------------------
# fig, ax = plt.subplots(figsize=(4.0, 2.0))
# fig.subplots_adjust(
#     bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
# ax.tick_params(
#     axis='both',       # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     pad=1,
#     labelsize='x-small')

# ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=12)
# ax.set_ylabel('state extents', fontsize=12)
# ax.grid(1)
# ax.plot(t, state_extents, lw=0.8, marker='.', ds='steps-post')
# fig.savefig(output_dir / 'state_extents.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot velocity measurement error
# ------------------------------------------------------------------
# e_vel = np.sqrt(v[:, subset, 0]**2 + v[:, subset, 1]**2)
# fig, ax = plt.subplots(figsize=(4.0, 2.0))
# fig.subplots_adjust(
#     bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
# ax.tick_params(
#     axis='both',       # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     pad=1,
#     labelsize='x-small')

# ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=12)
# ax.set_ylabel(
#   r'$\Vert e_{\mathrm{vel}} \Vert \ (\mathrm{m}/\mathrm{s})$',
#   fontsize=12
# )
# ax.grid(1)
# ax.plot(
#     t, np.nanmedian(e_vel, axis=1),
#     lw=0.8, label='median', ds='steps-post'
# )
# ax.fill_between(
#     t,
#     np.nanmin(e_vel, axis=1),
#     np.nanmax(e_vel, axis=1),
#     alpha=0.3
# )
# ax.set_ylim(bottom=0.0)
# ax.legend(
#     fontsize=12, handlelength=1, labelspacing=0.4,
#     borderpad=0.2, handletextpad=0.2, framealpha=1.,
#     ncol=2, columnspacing=1)
# fig.savefig(output_dir / 'vel_meas_err.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot gps measurement error
# ------------------------------------------------------------------
# e_gps = np.sqrt(g[:, subset, 0]**2 + g[:, subset, 1]**2)
# fig, ax = plt.subplots(figsize=(4.0, 2.0))
# fig.subplots_adjust(
#     bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
# ax.tick_params(
#     axis='both',       # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     pad=1,
#     labelsize='x-small')

# ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=12)
# ax.set_ylabel(
#     r'$\Vert e_{\mathrm{gps}} \Vert \ (\mathrm{m}/\mathrm{s})$',
#     fontsize=12
# )
# ax.grid(1)
# ax.plot(
#     t, np.nanmedian(e_gps, axis=1),
#     lw=0.8, label='median', ds='steps-post'
# )
# ax.fill_between(
#     t,
#     np.nanmin(e_gps, axis=1),
#     np.nanmax(e_gps, axis=1),
#     alpha=0.3
# )
# ax.set_ylim(bottom=0.0)
# ax.legend(
#     fontsize=12, handlelength=1, labelspacing=0.4,
#     borderpad=0.2, handletextpad=0.2, framealpha=1.,
#     ncol=2, columnspacing=1)
# fig.savefig(output_dir / 'gps_meas_err.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot range measurement error
# ------------------------------------------------------------------
# e_range = np.abs(r)
# fig, ax = plt.subplots(figsize=(4.0, 2.0))
# fig.subplots_adjust(
#     bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
# ax.tick_params(
#     axis='both',       # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     pad=1,
#     labelsize='x-small')

# ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=12)
# ax.set_ylabel(r'$\Vert e_{\mathrm{range}} \Vert$ [$m$]', fontsize=12)
# ax.grid(1)
# ax.plot(
#     t, np.nanmedian(e_range, axis=1),
#     lw=0.8, label='median', ds='steps-post'
# )
# ax.fill_between(
#     t,
#     np.nanmin(e_range, axis=1),
#     np.nanmax(e_range, axis=1),
#     alpha=0.3
# )
# ax.set_ylim(bottom=0.0)
# ax.legend(
#     fontsize=12, handlelength=1, labelspacing=0.4,
#     borderpad=0.2, handletextpad=0.2, framealpha=1.,
#     ncol=2, columnspacing=1)
# fig.savefig(output_dir / 'range_meas_err.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot position x
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(2.75, 1.75))
fig.subplots_adjust(
    bottom=0.215,
    top=0.925,
    wspace=0.33,
    right=0.975,
    left=0.2
)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small'
)
ax.grid(1, lw=0.4)

ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=12)
ax.set_ylabel(r'$x_i\ (\mathrm{m})$', fontsize=12)
ax.plot(t, x[:, subset, 0], lw=0.8, ds='steps-post')
fig.savefig(output_dir / 'pos_x.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot position y
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(2.75, 1.75))
fig.subplots_adjust(
    bottom=0.215,
    top=0.925,
    wspace=0.33,
    right=0.975,
    left=0.2
    )
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small'
)
ax.grid(1, lw=0.4)

ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=12)
ax.set_ylabel(r'$y_i\ (\mathrm{m})$', fontsize=12)
ax.plot(t, x[:, subset, 1], lw=0.8, ds='steps-post')
fig.savefig(output_dir / 'pos_y.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot position z
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(2.75, 1.75))
fig.subplots_adjust(
    bottom=0.215,
    top=0.925,
    wspace=0.33,
    right=0.975,
    left=0.2
    )
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small'
)
ax.grid(1, lw=0.4)

ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=12)
ax.set_ylabel(r'$z_i\ (\mathrm{m})$', fontsize=12)
ax.plot(t, x[:, subset, 2], lw=0.8, ds='steps-post')
fig.savefig(output_dir / 'pos_z.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot eigenvalues
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 2))
fig.tight_layout()
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='medium')
ax.grid(1, lw=0.4)

ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=12)

ax.semilogy(
    t,
    fre,
    lw=0.8,
    ls='--',
    color='k',
    ds='steps-post',
    # label=r'$F$'
)
ax.semilogy(
    t,
    re,
    lw=0.5,
    color='C0',
    # marker='.',
    ds='steps-post',
    # label=r'$F_{{{}}}$'.format(i)
)
ax.fill_between(t, np.min(re, axis=1), np.max(re, axis=1), alpha=0.3)
ax.set_ylim(bottom=1e-4, top=10)
# ax.legend(
#     fontsize=12, handlelength=1, labelspacing=0.4,
#     borderpad=0.2, handletextpad=0.2, framealpha=1.,
#     ncol=4, columnspacing=1)
fig.savefig(output_dir / 'eigenvalues.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot minimum distance between agents
# ------------------------------------------------------------------
mindist = minimum_distance(x[:, :, :3], axis=1)

fig, ax = plt.subplots(figsize=(2.5, 1.5))
fig.tight_layout()
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small'
)
ax.grid(1, lw=0.4)
ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=12)
ax.plot(
    t,
    mindist,
    lw=0.5,
    color='C0',
    ds='steps-post'
)
ax.fill_between(t, np.min(mindist, axis=1), np.max(mindist, axis=1), alpha=0.3)
# ax.set_ylim(bottom=0.0)
# ax.legend(
#     fontsize=12,
#     handlelength=1,
#     labelspacing=0.4,
#     borderpad=0.2,
#     handletextpad=0.2,
#     framealpha=1.,
#     ncol=2,
#     columnspacing=1
# )
ax.hlines(1.0, xmin=t.min(), xmax=t.max(), ls='--', lw=0.8, color='k')
ax.hlines(1.0, xmin=t.min(), xmax=t.max(), ls='--', lw=0.8, color='k')
ax.set_yticks([0.0, 5.0, 10.0, 15.0, 20.0])
fig.savefig(output_dir / 'min_dist.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot pose error (same two-panel format as the angle tests)
# ------------------------------------------------------------------
if has_estimated_pose:
    fig, axes = plt.subplots(2, 1, figsize=(4.0, 3.5))
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.35)

    for ax in axes:
        ax.tick_params(axis='both', which='both', pad=1, labelsize=12)
        ax.locator_params(axis='x', nbins=5)
        ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=12, labelpad=-2)
        ax.grid(1)

    axes[0].set_ylabel(r'$\|\hat{p}_i - p_i\| (\rm m)$', fontsize=14, labelpad=5)
    lines = axes[0].plot(
        t, position_error[:, subset], lw=2.0, ls='-', ds='steps-post',
        label=[str(i) for i in subset]
    )
    axes[1].set_ylabel(r'$\|\psi_i\| \ (\rm rad)$', fontsize=14, labelpad=5)
    axes[1].plot(
        t, orientation_error[:, subset], lw=2.0, ls='-', ds='steps-post',
        label=[str(i) for i in subset]
    )
    for ax, error in zip(axes, (position_error, orientation_error)):
        ax.set_ylim(bottom=0.0)
        if not np.any(error[:, subset]):
            ax.set_ylim(top=1.0)

    if len(subset) <= 5:
        for ax in axes:
            ax.legend(fontsize=12, ncols=len(subset), loc='upper right')
    else:
        # A shared legend leaves the error curves and zoom boxes visible for
        # larger networks (the bearing-control test currently has 30 robots).
        fig.set_size_inches(6.0, 4.5)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.85))
        fig.subplots_adjust(hspace=0.35)
        fig.legend(
            lines, [str(i) for i in subset], loc='upper center', fontsize=7,
            ncols=min(10, len(subset)), handlelength=1.0, columnspacing=0.8
        )

    if zoom_t.size:
        for ax, error in zip(axes, (zoom_position_error, zoom_orientation_error)):
            inset = ax.inset_axes([0.24, 0.28, 0.70, 0.49])
            inset.plot(zoom_t, error[:, subset], lw=0.9, ds='steps-post')
            inset.set_xlim(0.0, 1.0)
            ymax = np.max(error[:, subset])
            inset.set_ylim(0.0, 1.08 * ymax if ymax > 0 else 1.0)
            inset.set_xticks([0.0, 0.5, 1.0])
            inset.locator_params(axis='y', nbins=3)
            inset.tick_params(axis='both', labelsize=6, pad=1)
            inset.text(
                0.97, 0.94, 'First second', transform=inset.transAxes,
                ha='right', va='top', fontsize=7,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1)
            )
            inset.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=6, labelpad=0)
            inset.grid(True, alpha=0.35)

    fig.savefig(output_dir / 'pose_error.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot covariance
# ------------------------------------------------------------------
# fig, ax = plt.subplots(figsize=(4.0, 2.0))
# fig.subplots_adjust(
#     bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
# ax.tick_params(
#     axis='both',       # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     pad=1,
#     labelsize='x-small')
# ax.grid(1, lw=0.4)
# ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=12)
# ax.set_ylabel(
#     r'$\mathrm{tr}(\mathrm{cov}(e_{\mathrm{pos}}))$ [$m^2$]', fontsize=12
# )
# # ax.plot(
#     t,
#     np.sqrt(cov[:, subset, 0]**2 + cov[:, subset, 1]**2),
#     lw=0.8, ds='steps-post'
# )
# ax.semilogy(
#     t, np.median(cov[:, subset, 0] + cov[:, subset, 1], axis=1),
#     lw=0.8, label='median', ds='steps-post'
# )
# ax.fill_between(
#     t,
#     np.min(cov[:, subset, 0] + cov[:, subset, 1], axis=1),
#     np.max(cov[:, subset, 0] + cov[:, subset, 1], axis=1),
#     alpha=0.3
# )
# ax.set_ylim(bottom=1e-2)
# ax.legend(
#     fontsize=12, handlelength=1, labelspacing=0.4,
#     borderpad=0.2, handletextpad=0.2, framealpha=1.,
#     ncol=2, columnspacing=1)
# fig.savefig(output_dir / 'pos_cov.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot target collection
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(2.5, 1.5))
fig.tight_layout()
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small'
)
ax.grid(1, lw=0.4)
ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=12)
# ax.set_ylabel(r'$\ell_{ij} \ (\mathrm{m})$', fontsize=12, labelpad=-2.0)
ax.plot(
    t,
    targets.shape[1] - np.sum(targets, axis=1),
    lw=0.5,
    color='C0',
    # label='median',
    ds='steps-post'
)
# ax.set_ylim(bottom=0.0)
# ax.legend(
#     fontsize=12,
#     handlelength=1,
#     labelspacing=0.4,
#     borderpad=0.2,
#     handletextpad=0.2,
#     framealpha=1.,
#     ncol=2,
#     columnspacing=1
# )
# ax.hlines(1.0, xmin=0.0, xmax=200.0, ls='--', lw=0.8, color='k')
ax.set_yticks([0.0, 50.0, 100.0])
fig.savefig(output_dir / 'targets.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot communication metrics
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 2.0))
fig.tight_layout()
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='medium'
)
ax.grid(1, lw=0.4)
ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=12)
ax.set_yticks([1, 2, 3, 4, 5])
# ax.set_ylabel(r'$\ell_{ij} \ (\mathrm{m})$', fontsize=12, labelpad=-2.0)
ax.plot(
    t,
    fdiam,
    lw=0.8,
    ls='--',
    color='k',
    # label='median',
    ds='steps-post'
)
ax.plot(
    t,
    diam,
    lw=0.5,
    color='C0',
    # label='median',
    ds='steps-post'
)
ax.fill_between(t, np.min(diam, axis=1), np.max(diam, axis=1), alpha=0.3)
# ax.set_ylim(bottom=0.0)
# ax.legend(
#     fontsize=12,
#     handlelength=1,
#     labelspacing=0.4,
#     borderpad=0.2,
#     handletextpad=0.2,
#     framealpha=1.,
#     ncol=2,
#     columnspacing=1
# )
fig.savefig(output_dir / 'diameters.pdf', bbox_inches='tight')

plt.show()
