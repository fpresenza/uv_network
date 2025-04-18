#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import matplotlib.pyplot as plt

from uvnpy.toolkit import data
from uvnpy.network.core import geodesics
from uvnpy.bearings.core import rigidity_eigenvalue
from uvnpy.distances.core import distance_matrix

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'

# ------------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-i', '--init',
    default=0.0, type=float, help='init time in milli seconds'
)
parser.add_argument(
    '-e', '--end',
    default=0.0, type=float, help='end time in milli seconds'
)
parser.add_argument(
    '-j', '--jump',
    default=1, type=int, help='numbers of frames jumped'
)
parser.add_argument(
    '-s', '--subset',
    default=-1, type=int, nargs='+', help='subset of nodes to plot'
)
arg = parser.parse_args()

# ------------------------------------------------------------------
# Read simulated data
# ------------------------------------------------------------------
t = np.loadtxt('data/t.csv', delimiter=',')
arg.end = t[-1] if (arg.end == 0) else arg.end

# slices
k_i = int(np.argmin(np.abs(t - arg.init)))
k_e = int(np.argmin(np.abs(t - arg.end))) + 1

t = t[k_i:k_e:arg.jump]

p = data.read_csv(
    'data/position.csv',
    rows=(k_i, k_e),
    jump=arg.jump,
    dtype=float,
    shape=(-1, 3),
    asarray=True
)
n = len(p[0])
nodes = np.arange(n)
if (arg.subset == -1):
    subset = nodes
else:
    subset = arg.subset


A = data.read_csv(
    'data/adjacency.csv',
    rows=(k_i, k_e), jump=arg.jump, dtype=float, shape=(n, n), asarray=True
)
targets = data.read_csv(
    'data/targets.csv',
    rows=(k_i, k_e), jump=arg.jump, dtype=float, shape=(-1, 4), asarray=True
)
u_t = data.read_csv(
    'data/target_action.csv',
    rows=(k_i, k_e),
    jump=arg.jump,
    dtype=float,
    shape=(n, 3),
    asarray=True
)
u_c = data.read_csv(
    'data/collision_action.csv',
    rows=(k_i, k_e),
    jump=arg.jump,
    dtype=float,
    shape=(n, 3),
    asarray=True
)
u_r = data.read_csv(
    'data/rigidity_action.csv',
    rows=(k_i, k_e),
    jump=arg.jump,
    dtype=float,
    shape=(n, 3),
    asarray=True
)
action_extents = data.read_csv(
    'data/action_extents.csv',
    rows=(k_i, k_e),
    jump=arg.jump,
    dtype=float,
    asarray=True
)

hatp = data.read_csv(
    'data/est_position.csv',
    rows=(k_i, k_e),
    jump=arg.jump,
    dtype=float,
    shape=(-1, 3),
    asarray=True
)

fre = np.empty(len(t))
re = np.empty((len(t), n))
fdiam = np.empty(len(t))
diam = np.empty((len(t), n))
for k, (adj, pos, ext) in enumerate(zip(A, p, action_extents)):
    geo = geodesics(adj)
    fre[k] = rigidity_eigenvalue(adj, pos)
    fdiam[k] = np.max(geo)
    for i in nodes:
        Vi = geo[i] <= ext[i]
        Ai = adj[np.ix_(Vi, Vi)]
        qi = pos[Vi]
        re[k, i] = rigidity_eigenvalue(Ai, qi)
        diam[k, i] = np.max(geodesics(Ai))

# tc = np.loadtxt('data/tc.csv', delimiter=',')
# cov = np.loadtxt('data/covariance.csv', delimiter=',')

# v = np.loadtxt('data/vel_meas_err.csv', delimiter=',')
# g = np.loadtxt('data/gps_meas_err.csv', delimiter=',')
# r = np.loadtxt('data/range_meas_err.csv', delimiter=',')
# state_extents = np.loadtxt('data/state_extents.csv', delimiter=',')


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

ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=8)
ax.set_ylabel(r'$\Vert u_t \Vert \ (\mathrm{m}/\mathrm{s})$', fontsize=8)
ax.grid(1)
ax.plot(
    t,
    np.sqrt(np.sum(u_t[:, subset]**2, axis=2)),
    lw=0.8,
    ds='steps-post'
)
fig.savefig('data/time/target_control.png', format='png', dpi=360)

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

ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=8)
ax.set_ylabel(r'$\Vert u_c \Vert \ (\mathrm{m}/\mathrm{s})$', fontsize=8)
ax.grid(1)
ax.plot(
    t,
    np.sqrt(np.sum(u_c[:, subset]**2, axis=2)),
    lw=0.8, ds='steps-post'
)
fig.savefig('data/time/collision_control.png', format='png', dpi=360)

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

ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=8)
ax.set_ylabel(r'$\Vert u_r \Vert \ (\mathrm{m}/\mathrm{s})$', fontsize=8)
ax.grid(1)
ax.plot(
    t,
    np.sqrt(np.sum(u_r[:, subset]**2, axis=2)),
    lw=0.8, ds='steps-post'
)
fig.savefig('data/time/rigidity_control.png', format='png', dpi=360)

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

ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=8)
ax.set_ylabel(r'$\Vert u \Vert \ (\mathrm{m}/\mathrm{s})$', fontsize=8)
ax.grid(1)
ax.plot(
    t,
    np.sqrt(np.sum(u[:, subset]**2, axis=2)),
    lw=0.8, ds='steps-post'
)
fig.savefig('data/time/control.png', format='png', dpi=360)

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

# ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=8)
# ax.set_ylabel('state extents', fontsize=8)
# ax.grid(1)
# ax.plot(t, state_extents, lw=0.8, marker='.', ds='steps-post')
# fig.savefig('data/time/state_extents.png', format='png', dpi=400)

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

# ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=8)
# ax.set_ylabel(
#   r'$\Vert e_{\mathrm{vel}} \Vert \ (\mathrm{m}/\mathrm{s})$',
#   fontsize=8
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
#     fontsize=8, handlelength=1, labelspacing=0.4,
#     borderpad=0.2, handletextpad=0.2, framealpha=1.,
#     ncol=2, columnspacing=1)
# fig.savefig('data/time/vel_meas_err.png', format='png', dpi=400)

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

# ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=8)
# ax.set_ylabel(
#     r'$\Vert e_{\mathrm{gps}} \Vert \ (\mathrm{m}/\mathrm{s})$',
#     fontsize=8
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
#     fontsize=8, handlelength=1, labelspacing=0.4,
#     borderpad=0.2, handletextpad=0.2, framealpha=1.,
#     ncol=2, columnspacing=1)
# fig.savefig('data/time/gps_meas_err.png', format='png', dpi=400)

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

# ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=8)
# ax.set_ylabel(r'$\Vert e_{\mathrm{range}} \Vert$ [$m$]', fontsize=8)
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
#     fontsize=8, handlelength=1, labelspacing=0.4,
#     borderpad=0.2, handletextpad=0.2, framealpha=1.,
#     ncol=2, columnspacing=1)
# fig.savefig('data/time/range_meas_err.png', format='png', dpi=400)

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

ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=8)
ax.set_ylabel(r'$x_i\ (\mathrm{m})$', fontsize=8)
ax.plot(t, p[:, subset, 0], lw=0.8, ds='steps-post')
fig.savefig('data/time/pos_x.png', format='png', dpi=360)

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

ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=8)
ax.set_ylabel(r'$y_i\ (\mathrm{m})$', fontsize=8)
ax.plot(t, p[:, subset, 1], lw=0.8, ds='steps-post')
fig.savefig('data/time/pos_y.png', format='png', dpi=360)

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

ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=8)
ax.set_ylabel(r'$z_i\ (\mathrm{m})$', fontsize=8)
ax.plot(t, p[:, subset, 2], lw=0.8, ds='steps-post')
fig.savefig('data/time/pos_z.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Plot eigenvalues
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.0, 2.0))
fig.subplots_adjust(
    bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small')
ax.grid(1, lw=0.4)

ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=8)

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
#     fontsize=8, handlelength=1, labelspacing=0.4,
#     borderpad=0.2, handletextpad=0.2, framealpha=1.,
#     ncol=4, columnspacing=1)
fig.savefig('data/time/eigenvalues.png', format='png', dpi=400)

# ------------------------------------------------------------------
# Plot minimum distance between agents
# ------------------------------------------------------------------
dist = distance_matrix(p)
dist[..., np.eye(20).astype(bool)] = np.nan
mindist = np.nanmin(dist, axis=1)

fig, ax = plt.subplots(figsize=(2.5, 1.5))
fig.tight_layout()
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='xx-small'
)
ax.grid(1, lw=0.4)
ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=8)
ax.set_ylabel(r'$\ell_{ij} \ (\mathrm{m})$', fontsize=8, labelpad=-2.0)
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
#     fontsize=8,
#     handlelength=1,
#     labelspacing=0.4,
#     borderpad=0.2,
#     handletextpad=0.2,
#     framealpha=1.,
#     ncol=2,
#     columnspacing=1
# )
ax.hlines(1.0, xmin=0.0, xmax=200.0, ls='--', lw=0.8, color='k')
ax.set_yticks([0.0, 5.0, 10.0, 15.0])
fig.savefig('data/time/min_dist.png', format='png', dpi=400)

# ------------------------------------------------------------------
# Plot position error
# ------------------------------------------------------------------
# hatp = np.empty((len(t), n, 3))
# hatp[0] = np.random.normal(p[0], scale=1.0)
# for k in range(len(t)):
err = np.sqrt(np.square(p - hatp).sum(axis=-1))
fig, ax = plt.subplots(figsize=(4.0, 2.0))
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
    labelsize='x-small'
)
ax.grid(1, lw=0.4)
ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=8)
ax.set_ylabel(r'$\Vert p - \hat{p} \Vert \ (\mathrm{m})$', fontsize=8)
ax.plot(
    t,
    np.max(err, axis=1),
    lw=0.8,
    # label='median',
    ds='steps-post'
)
# ax.fill_between(t, np.min(err, axis=1), np.max(err, axis=1), alpha=0.3)
# ax.set_ylim(bottom=0.0)
# ax.legend(
#     fontsize=8,
#     handlelength=1,
#     labelspacing=0.4,
#     borderpad=0.2,
#     handletextpad=0.2,
#     framealpha=1.,
#     ncol=2,
#     columnspacing=1
# )
fig.savefig('data/time/pos_error.png', format='png', dpi=400)

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
# ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=8)
# ax.set_ylabel(
#     r'$\mathrm{tr}(\mathrm{cov}(e_{\mathrm{pos}}))$ [$m^2$]', fontsize=8
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
#     fontsize=8, handlelength=1, labelspacing=0.4,
#     borderpad=0.2, handletextpad=0.2, framealpha=1.,
#     ncol=2, columnspacing=1)
# fig.savefig('data/time/pos_cov.png', format='png', dpi=400)

# ------------------------------------------------------------------
# Plot target collection
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(2.5, 1.5))
fig.tight_layout()
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='xx-small'
)
ax.grid(1, lw=0.4)
ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=8)
# ax.set_ylabel(r'$\ell_{ij} \ (\mathrm{m})$', fontsize=8, labelpad=-2.0)
ax.plot(
    t,
    np.sum(targets[0, :, 3]) - np.sum(targets[:, :, 3], axis=1),
    lw=0.5,
    color='C0',
    # label='median',
    ds='steps-post'
)
# ax.set_ylim(bottom=0.0)
# ax.legend(
#     fontsize=8,
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
fig.savefig('data/time/targets.png', format='png', dpi=400)

# ------------------------------------------------------------------
# Plot communication metrics
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(2.5, 1.5))
fig.tight_layout()
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='xx-small'
)
ax.grid(1, lw=0.4)
ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=8)
# ax.set_ylabel(r'$\ell_{ij} \ (\mathrm{m})$', fontsize=8, labelpad=-2.0)
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
#     fontsize=8,
#     handlelength=1,
#     labelspacing=0.4,
#     borderpad=0.2,
#     handletextpad=0.2,
#     framealpha=1.,
#     ncol=2,
#     columnspacing=1
# )
fig.savefig('data/time/diameter.png', format='png', dpi=400)

plt.show()
