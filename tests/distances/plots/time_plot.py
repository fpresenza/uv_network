#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import matplotlib.pyplot as plt

from uvnpy.toolkit import data
from uvnpy.network.core import geodesics
from uvnpy.distances.core import distance_rigidity_eigenvalue

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

x = data.read_csv(
    'data/position.csv',
    rows=(k_i, k_e),
    jump=arg.jump,
    dtype=float,
    shape=(-1, 2),
    asarray=True
)

n = len(x[0])
nodes = np.arange(n)
if (arg.subset == -1):
    subset = nodes
else:
    subset = arg.subset

A = data.read_csv(
    'data/adjacency.csv',
    rows=(k_i, k_e),
    jump=arg.jump,
    dtype=float,
    shape=(n, n),
    asarray=True
)
targets = data.read_csv(
    'data/targets.csv',
    rows=(k_i, k_e),
    jump=arg.jump,
    dtype=float,
    shape=(-1, 3),
    asarray=True
)
u_t = data.read_csv(
    'data/target_action.csv',
    rows=(k_i, k_e),
    jump=arg.jump,
    dtype=float,
    shape=(n, 2),
    asarray=True
)
u_c = data.read_csv(
    'data/collision_action.csv',
    rows=(k_i, k_e),
    jump=arg.jump,
    dtype=float,
    shape=(n, 2),
    asarray=True
)
u_r = data.read_csv(
    'data/rigidity_action.csv',
    rows=(k_i, k_e),
    jump=arg.jump,
    dtype=float,
    shape=(n, 2),
    asarray=True
)
action_extents = data.read_csv(
    'data/action_extents.csv',
    rows=(k_i, k_e),
    jump=arg.jump,
    dtype=float,
    asarray=True
)
hatx = data.read_csv(
    'data/est_position.csv',
    rows=(k_i, k_e),
    jump=arg.jump,
    dtype=float,
    shape=(n, 2),
    asarray=True
)

fre = np.empty(len(t))
re = np.empty((len(t), n))
fdiam = np.empty(len(t))
diam = np.empty((len(t), n))
for k, (adj, pos, ext) in enumerate(zip(A, x, action_extents)):
    geo = geodesics(adj)
    fre[k] = distance_rigidity_eigenvalue(adj, pos)
    fdiam[k] = np.max(geo)
    for i in nodes:
        Vi = geo[i] <= ext[i]
        if sum(Vi) > 1:
            Ai = adj[np.ix_(Vi, Vi)]
            qi = pos[Vi]
            re[k, i] = distance_rigidity_eigenvalue(Ai, qi)
            diam[k, i] = np.max(geodesics(Ai))
        else:
            re[k, i] = 0.0
            diam[k, i] = np.inf

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
fig.savefig('data/time/target_control.pdf', bbox_inches='tight')

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
fig.savefig('data/time/collision_control.pdf', bbox_inches='tight')

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
fig.savefig('data/time/rigidity_control.pdf', bbox_inches='tight')

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
fig.savefig('data/time/control.pdf', bbox_inches='tight')

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
# fig.savefig('data/time/state_extents.pdf', bbox_inches='tight')

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
ax.plot(t, x[:, subset, 0], lw=0.8, ds='steps-post')
fig.savefig('data/time/pos_x.pdf', bbox_inches='tight')

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
ax.plot(t, x[:, subset, 1], lw=0.8, ds='steps-post')
fig.savefig('data/time/pos_y.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot eigenvalues
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(2.5, 1.5))
fig.tight_layout()
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
fig.savefig('data/time/eigenvalues.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot position error
# ------------------------------------------------------------------
err = np.sqrt(np.square(x - hatx).sum(axis=-1))
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
ax.set_ylabel(r'$\Vert x - \hat{x} \Vert \ (\mathrm{m})$', fontsize=8)
ax.plot(
    t,
    err[:, subset],
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
fig.savefig('data/time/pos_error.pdf', bbox_inches='tight')

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
# ax.set_xlabel(r'$t$ [$s$]', fontsize=8)
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
# fig.savefig('data/time/pos_cov.pdf', bbox_inches='tight')

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
ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=8)
ax.plot(
    t,
    np.sum(targets[0, :, 2]) - np.sum(targets[:, :, 2], axis=1),
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
# ax.set_yticks([0.0, 50.0, 100.0])
fig.savefig('data/time/targets.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot communication metrics
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
ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=8)
ax.set_yticks([1, 2, 3, 4, 5])

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
fig.savefig('data/time/diameters.pdf', bbox_inches='tight')

plt.show()
