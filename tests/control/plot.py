#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on mié 29 dic 2021 16:41:13 -03
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from uvnpy.network import plot

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


# ------------------------------------------------------------------
# Read simulated data
# ------------------------------------------------------------------
t = np.loadtxt('data/t.csv', delimiter=',')
x = np.loadtxt('data/position.csv', delimiter=',')
hatx = np.loadtxt('data/est_position.csv', delimiter=',')
cov = np.loadtxt('data/covariance.csv', delimiter=',')
u = np.loadtxt('data/action.csv', delimiter=',')
v = np.loadtxt('data/vel_meas_err.csv', delimiter=',')
g = np.loadtxt('data/gps_meas_err.csv', delimiter=',')
r = np.loadtxt('data/range_meas_err.csv', delimiter=',')
fre = np.loadtxt('data/fre.csv', delimiter=',')
re = np.loadtxt('data/re.csv', delimiter=',')
A = np.loadtxt('data/adjacency.csv', delimiter=',')
state_extents = np.loadtxt('data/state_extents.csv', delimiter=',')
targets = np.loadtxt('data/targets.csv', delimiter=',')

n = int(len(x[0])/2)
nodes = np.arange(n)
state_extents = state_extents.astype(int)

# reshapes
x = x.reshape(len(t), n, 2)
hatx = hatx.reshape(len(t), n, 2)
cov = cov.reshape(len(t), n, 2)
u = u.reshape(len(t), n, 2)
v = v.reshape(len(t), n, 2)
g = g.reshape(len(t), n, 2)
re = re.reshape(len(t), -1)
A = A.reshape(len(t), n, n)
targets = targets.reshape(len(t), -1, 3)

comm_events = np.arange(0, t[-1], 0.25)

# slice
# ki = np.argmin(np.abs(t - 0))
# kf = np.argmin(np.abs(t - 25))
# t = t[:kf]
# x = x[:kf]
# hatx = hatx[:kf]
# u = u[:kf]
# fre = fre[:kf]
# re = re[:kf]
# A = A[:kf]
# state_extents = state_extents[:kf]

# calculos
edges = A.sum(-1).sum(-1)/2

# ------------------------------------------------------------------
# Plot control action
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.0, 2.0))
fig.subplots_adjust(
    bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small')

ax.set_xlabel('$t$ [$s$]', fontsize=8)
ax.set_ylabel(r'$\Vert u \Vert$ [$m/s$]', fontsize=8)
ax.grid(1)
ax.plot(t, np.sqrt(u[..., 0]**2 + u[..., 1]**2), lw=0.8, ds='steps-post')
fig.savefig('data/control.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Plot state extents
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.0, 2.0))
fig.subplots_adjust(
    bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small')

ax.set_xlabel('$t$ [$s$]', fontsize=8)
ax.set_ylabel('state extents', fontsize=8)
ax.grid(1)
ax.plot(t, state_extents, lw=0.8, marker='.', ds='steps-post')
fig.savefig('data/state_extents.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Plot velocity measurement error
# ------------------------------------------------------------------
# e_vel = np.sqrt(v[..., 0]**2 + v[..., 1]**2)
# fig, ax = plt.subplots(figsize=(4.0, 2.0))
# fig.subplots_adjust(
#     bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
# ax.tick_params(
#     axis='both',       # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     pad=1,
#     labelsize='x-small')

# ax.set_xlabel('$t$ [$s$]', fontsize=8)
# ax.set_ylabel(r'$\Vert e_{\mathrm{vel}} \Vert$ [$m/s$]', fontsize=8)
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
# fig.savefig('data/vel_meas_err.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Plot gps measurement error
# ------------------------------------------------------------------
# e_gps = np.sqrt(g[..., 0]**2 + g[..., 1]**2)
# fig, ax = plt.subplots(figsize=(4.0, 2.0))
# fig.subplots_adjust(
#     bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
# ax.tick_params(
#     axis='both',       # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     pad=1,
#     labelsize='x-small')

# ax.set_xlabel('$t$ [$s$]', fontsize=8)
# ax.set_ylabel(r'$\Vert e_{\mathrm{gps}} \Vert$ [$m/s$]', fontsize=8)
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
# fig.savefig('data/gps_meas_err.png', format='png', dpi=360)

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

# ax.set_xlabel('$t$ [$s$]', fontsize=8)
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
# fig.savefig('data/range_meas_err.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Plot position x
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(2.75, 1.75))
fig.subplots_adjust(
    bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.2)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small')
ax.grid(1, lw=0.4)

ax.set_xlabel(r'$t$ [$s$]', fontsize=8)
ax.set_ylabel('position-$x$ [$m$]', fontsize=8)
ax.plot(t, x[..., 0], lw=0.8, ds='steps-post')
# fig.savefig('data/pos_x.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Plot position y
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(2.75, 1.75))
fig.subplots_adjust(
    bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.2)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small')
ax.grid(1, lw=0.4)

ax.set_xlabel(r'$t$ [$s$]', fontsize=8)
ax.set_ylabel('position-$y$ [$m$]', fontsize=8)
ax.plot(t, x[..., 1], lw=0.8, ds='steps-post')
# fig.savefig('data/pos_y.png', format='png', dpi=360)

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

ax.set_xlabel(r'$t$ [$s$]', fontsize=8)
ax.set_ylabel('Autovalores \n de Rigidez', fontsize=8)
ax.semilogy(
    t, np.median(re, axis=1),
    lw=0.8,
    marker='.',
    ds='steps-post',
    label='median'
)
ax.semilogy(
    t, fre,
    lw=0.8,
    ls='--',
    color='k',
    ds='steps-post',
    label=r'framework'
)
ax.set_ylim(bottom=1e-4, top=3)
ax.fill_between(
    t,
    np.min(re, axis=1),
    np.max(re, axis=1),
    alpha=0.3
)
ax.vlines(comm_events, 1e-4, 1, lw=0.75, color='g', alpha=0.5)
ax.set_ylim(bottom=1e-4)
ax.legend(
    fontsize=8, handlelength=1, labelspacing=0.4,
    borderpad=0.2, handletextpad=0.2, framealpha=1.,
    ncol=2, columnspacing=1)
fig.savefig('data/eigenvalues.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Plot position error
# ------------------------------------------------------------------
err = np.sqrt(np.square(x - hatx).sum(axis=-1))
fig, ax = plt.subplots(figsize=(4.0, 2.0))
fig.subplots_adjust(
    bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small')
ax.grid(1, lw=0.4)
ax.set_xlabel(r'$t$ [$s$]', fontsize=8)
ax.set_ylabel(r'$\Vert e_{\mathrm{pos}} \Vert$ [$m$]', fontsize=8)
ax.plot(t, np.median(err, axis=1), lw=0.8, label='median', ds='steps-post')
ax.fill_between(
    t,
    np.min(err, axis=1),
    np.max(err, axis=1),
    alpha=0.3
)
ax.set_ylim(bottom=0.0)
ax.legend(
    fontsize=8, handlelength=1, labelspacing=0.4,
    borderpad=0.2, handletextpad=0.2, framealpha=1.,
    ncol=2, columnspacing=1)
fig.savefig('data/pos_error.png', format='png', dpi=360)

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
#     t, np.sqrt(cov[..., 0]**2 + cov[..., 1]**2), lw=0.8, ds='steps-post')
# ax.semilogy(
#     t, np.median(cov[..., 0] + cov[..., 1], axis=1),
#     lw=0.8, label='median', ds='steps-post'
# )
# ax.fill_between(
#     t,
#     np.min(cov[..., 0] + cov[..., 1], axis=1),
#     np.max(cov[..., 0] + cov[..., 1], axis=1),
#     alpha=0.3
# )
# ax.set_ylim(bottom=1e-2)
# ax.legend(
#     fontsize=8, handlelength=1, labelspacing=0.4,
#     borderpad=0.2, handletextpad=0.2, framealpha=1.,
#     ncol=2, columnspacing=1)
# fig.savefig('data/pos_cov.png', format='png', dpi=360)

plt.show()

# ------------------------------------------------------------------
# Plot snapshots
# ------------------------------------------------------------------
lim = 1000

for tk in comm_events:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize='x-small')
    ax.grid(1, lw=0.4)
    ax.set_aspect('equal')
    # ax.set_xlabel(r'$\mathrm{x}$', fontsize='x-small', labelpad=0.6)
    # ax.set_ylabel(r'$\mathrm{y}$', fontsize='x-small', labelpad=0)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    k = np.argmin(np.abs(t - tk))
    ax.text(
            0.05, 0.01, r't = {:.2f}s'.format(tk),
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes, color='r', fontsize=8)

    for i in nodes:
        plot.nodes(
            ax, x[k, i],
            color='k',
            marker=f'${i}$',
            s=20,
            lw=0.2
        )
        circle = plt.Circle(x[k, i], 30.0, alpha=0.3)
        ax.add_artist(circle)
    plot.edges(ax, x[k], A[k], color=cm.coolwarm(20), lw=0.5, zorder=0)

    untracked = targets[k, :, 2].astype(bool)
    tracked = np.logical_not(untracked)
    ax.scatter(
        targets[k, untracked, 0], targets[k, untracked, 1],
        marker='s', s=4, color='0.6')
    ax.scatter(
        targets[k, tracked, 0], targets[k, tracked, 1],
        marker='s', s=4, color='green')

    fig.savefig('data/snapshots/{}.png'.format(k), format='png', dpi=360)
    plt.close()
