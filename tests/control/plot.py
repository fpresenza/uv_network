#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on mié 29 dic 2021 16:41:13 -03
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from uvnpy.network import plot
from uvnpy.network.core import geodesics

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
extents = np.loadtxt('data/extents.csv', delimiter=',')
targets = np.loadtxt('data/targets.csv', delimiter=',')

n = int(len(x[0])/2)
nodes = np.arange(n)
extents = extents.astype(int)

# reshapes
x = x.reshape(len(t), n, 2)
hatx = hatx.reshape(len(t), n, 2)
cov = cov.reshape(len(t), n, 2)
u = u.reshape(len(t), n, 2)
v = v.reshape(len(t), n, 2)
g = g.reshape(len(t), n, 2)
A = A.reshape(len(t), n, n)
targets = targets.reshape(len(t), -1, 3)

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
# extents = extents[:kf]

# calculos
edges = A.sum(-1).sum(-1)/2
G = [geodesics(adj) for adj in A]

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
# Plot velocity measurement error
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
ax.set_ylabel(r'$\Vert e_{\mathrm{vel}} \Vert$ [$m/s$]', fontsize=8)
ax.grid(1)
ax.plot(
    t, np.nanmedian(np.sqrt(v[..., 0]**2 + v[..., 1]**2), axis=1),
    lw=0.8, label='median', ds='steps-post'
)
ax.fill_between(
    t,
    np.nanmin(np.sqrt(v[..., 0]**2 + v[..., 1]**2), axis=1),
    np.nanmax(np.sqrt(v[..., 0]**2 + v[..., 1]**2), axis=1),
    alpha=0.3
)
ax.set_ylim(bottom=0.0)
ax.legend(
    fontsize=8, handlelength=1, labelspacing=0.4,
    borderpad=0.2, handletextpad=0.2, framealpha=1.,
    ncol=2, columnspacing=1)
fig.savefig('data/vel_meas_err.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Plot gps measurement error
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
ax.set_ylabel(r'$\Vert e_{\mathrm{gps}} \Vert$ [$m/s$]', fontsize=8)
ax.grid(1)
ax.plot(
    t, np.nanmedian(np.sqrt(g[..., 0]**2 + g[..., 1]**2), axis=1),
    lw=0.8, label='median', ds='steps-post'
)
ax.fill_between(
    t,
    np.nanmin(np.sqrt(g[..., 0]**2 + g[..., 1]**2), axis=1),
    np.nanmax(np.sqrt(g[..., 0]**2 + g[..., 1]**2), axis=1),
    alpha=0.3
)
ax.set_ylim(bottom=0.0)
ax.legend(
    fontsize=8, handlelength=1, labelspacing=0.4,
    borderpad=0.2, handletextpad=0.2, framealpha=1.,
    ncol=2, columnspacing=1)
fig.savefig('data/gps_meas_err.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Plot range measurement error
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
ax.set_ylabel(r'$\Vert e_{\mathrm{range}} \Vert$ [$m$]', fontsize=8)
ax.grid(1)
ax.plot(
    t, np.nanmedian(np.abs(r), axis=1),
    lw=0.8, label='median', ds='steps-post'
)
ax.fill_between(
    t,
    np.nanmin(np.abs(r), axis=1),
    np.nanmax(np.abs(r), axis=1),
    alpha=0.3
)
ax.set_ylim(bottom=0.0)
ax.legend(
    fontsize=8, handlelength=1, labelspacing=0.4,
    borderpad=0.2, handletextpad=0.2, framealpha=1.,
    ncol=2, columnspacing=1)
fig.savefig('data/range_meas_err.png', format='png', dpi=360)

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
ax.semilogy(t, np.median(re, axis=1), lw=0.8, label='median')
ax.semilogy(t, fre, ls='--', color='k', lw=0.8, label=r'framework')
ax.set_ylim(bottom=1e-4, top=3)
ax.fill_between(
    t,
    np.min(re, axis=1),
    np.max(re, axis=1),
    alpha=0.3
)
ax.set_ylim(bottom=1e-2)
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
ax.set_ylabel(r'$\Vert e_{\mathrm{ekf}} \Vert$ [$m$]', fontsize=8)
ax.semilogy(t, np.median(err, axis=1), lw=0.8, label='median', ds='steps-post')
ax.fill_between(
    t,
    np.min(err, axis=1),
    np.max(err, axis=1),
    alpha=0.3
)
ax.set_ylim(bottom=1e-2)
ax.legend(
    fontsize=8, handlelength=1, labelspacing=0.4,
    borderpad=0.2, handletextpad=0.2, framealpha=1.,
    ncol=2, columnspacing=1)
fig.savefig('data/pos_error.png', format='png', dpi=360)

# ------------------------------------------------------------------
# Plot covariance
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
ax.set_ylabel(
    r'$\mathrm{tr}(\mathrm{cov}(e_{\mathrm{ekf}}))$ [$m^2$]', fontsize=8
)
# ax.plot(t, np.sqrt(cov[..., 0]**2 + cov[..., 1]**2), lw=0.8, ds='steps-post')
ax.semilogy(
    t, np.median(cov[..., 0] + cov[..., 1], axis=1),
    lw=0.8, label='median', ds='steps-post'
)
ax.fill_between(
    t,
    np.min(cov[..., 0] + cov[..., 1], axis=1),
    np.max(cov[..., 0] + cov[..., 1], axis=1),
    alpha=0.3
)
ax.set_ylim(bottom=1e-2)
ax.legend(
    fontsize=8, handlelength=1, labelspacing=0.4,
    borderpad=0.2, handletextpad=0.2, framealpha=1.,
    ncol=2, columnspacing=1)
fig.savefig('data/pos_cov.png', format='png', dpi=360)
# ------------------------------------------------------------------
# Plot snapshots
# ------------------------------------------------------------------
snapshots = np.array([0, 40, 125, 250, 300, 400])
# lim = np.abs(x).max()
lim = 1000

for i, tk in enumerate(snapshots):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize='x-small')
    ax.grid(1, lw=0.4)
    ax.set_aspect('equal')
    # ax.set_xlabel(r'$\mathrm{x}$', fontsize='x-small', labelpad=0.6)
    # ax.set_ylabel(r'$\mathrm{y}$', fontsize='x-small', labelpad=0)

    a = int(lim)
    b = int(0.5 * lim)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    k = np.argmin(np.abs(t - tk))
    ax.text(
            0.05, 0.01, r't = {:.2f}s'.format(tk),
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes, color='r', fontsize=8)

    one_hop_rigid = extents[k] == 1
    two_hop_rigid = extents[k] == 2
    three_hop_rigid = extents[k] == 3

    plot.nodes(
        ax, x[k, one_hop_rigid],
        marker='o', color='royalblue', s=8, zorder=20, label=r'$h_0=1$')
    plot.nodes(
        ax, x[k, two_hop_rigid],
        marker='D', color='chocolate', s=8, zorder=20, label=r'$h_0=2$')
    plot.nodes(
        ax, x[k, three_hop_rigid],
        marker='s', color='mediumseagreen', s=8, zorder=20, label=r'$h_0=3$')
    plot.edges(ax, x[k], A[k], color='k', lw=0.5)

    untracked = targets[k, :, 2].astype(bool)
    tracked = np.logical_not(untracked)
    ax.scatter(
        targets[k, untracked, 0], targets[k, untracked, 1],
        marker='s', s=4, color='0.6')
    ax.scatter(
        targets[k, tracked, 0], targets[k, tracked, 1],
        marker='s', s=4, color='0.2')

    plot.nodes(
        ax, x[max(0, k-150):k+1:5, one_hop_rigid],
        marker='.', color='royalblue', s=1, zorder=1, lw=0.5)
    plot.nodes(
        ax, x[max(0, k-150):k+1:5, two_hop_rigid],
        marker='.', color='chocolate', s=1, zorder=1, lw=0.5)
    plot.nodes(
        ax, x[max(0, k-150):k+1:5, three_hop_rigid],
        marker='.', color='mediumseagreen', s=1, zorder=1, lw=0.5)

    # fig.savefig(
    #     'data/snapshots_{}.png'.format(int(tk)),
    #     format='png', dpi=360)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
fig.subplots_adjust(left=0.15)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small')
ax.grid(1, lw=0.4)
ax.set_aspect('equal')

a = int(40)
b = int(20)
ax.set_xlabel(r'$x$ [m]', fontsize=9, labelpad=0)
ax.set_ylabel(r'$y$ [m]', fontsize=9, labelpad=0)
ax.set_xlim(-a, a)
ax.set_ylim(-a, a)
ax.set_xticks([-a, -b, 0, b, a])
ax.set_yticks([-a, -b, 0, b, a])
ax.set_xticklabels([-a, -b, 0, b, a])
ax.set_yticklabels([-a, -b, 0, b, a])

one_hop_rigid = extents[k] == 1
two_hop_rigid = extents[k] == 2
three_hop_rigid = extents[k] == 3

plot.nodes(
    ax, x[0, one_hop_rigid],
    marker='o', color='royalblue', s=7, zorder=20, label=r'$h_0=1$')
plot.nodes(
    ax, x[0, two_hop_rigid],
    marker='D', color='chocolate', s=7, zorder=20, label=r'$h_0=2$')
plot.nodes(
    ax, x[0, three_hop_rigid],
    marker='s', color='mediumseagreen', s=7, zorder=20, label=r'$h_0=3$')
plot.edges(ax, x[0], A[0], color='k', lw=0.5)

show = np.full(targets.shape[1], True)
show[[2, 6, 19]] = False
# print(targets.shape)
ax.scatter(
    targets[0, show, 0], targets[0, show, 1],
    marker='s', s=4, color='0.6', alpha=0.5)

circle = Circle(x[0, 6], 1.8, facecolor='None', linewidth=0.5, edgecolor='red')
ax.add_artist(circle)
circle = Circle(x[0, 8], 1.8, facecolor='None', linewidth=0.5, edgecolor='red')
ax.add_artist(circle)

ax.legend(
    fontsize='6',
    handletextpad=0.0,
    borderpad=0.2,
    ncol=4, columnspacing=0.15,
    loc='upper center')

# fig.savefig('data/snapshots_init.png', format='png', dpi=360)

# fig, ax = plt.subplots(figsize=(2.5, 2.5))
# fig.subplots_adjust(left=0.15)
# ax.tick_params(
#     axis='both',       # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     pad=1,
#     labelsize='x-small')
# ax.grid(1, lw=0.4)
# ax.set_aspect('equal')
# ax.set_xlabel(r'$x$ [m]', fontsize=9, labelpad=0)
# ax.set_ylabel(r'$y$ [m]', fontsize=9, labelpad=0)
# ax.set_xticks([-100, 0, 100])
# ax.set_yticks([-100, 0, 100])
# ax.set_xticklabels([-100, 0, 100])
# ax.set_yticklabels([-100, 0, 100])
# ax.set_xlim(-100, 100)
# ax.set_ylim(-100, 115)

# plot.nodes(
#     ax, x[0, one_hop_rigid],
#     marker='o', color='royalblue', s=7, zorder=20, label=r'$h_0 = 1$')
# plot.nodes(
#     ax, x[0, two_hop_rigid],
#     marker='D', color='chocolate', s=7, zorder=20, label=r'$h_0 = 2$')
# plot.nodes(
#     ax, x[0, three_hop_rigid],
#     marker='s', color='mediumseagreen', s=7, zorder=20, label=r'$h_0 = 3$')
# plot.nodes(
#     ax, x[0, four_hop_rigid],
#     marker='^', color='purple', s=7, zorder=10, label=r'$h_0 = 4$')
# plot.edges(ax, x[0], A[0], color='k', lw=0.5)

# circle = Circle(x[0, 15], 5, facecolor='None', linewidth=1, edgecolor='red')
# ax.add_artist(circle)
# circle = Circle(x[0, 41], 5, facecolor='None', linewidth=1, edgecolor='red')
# ax.add_artist(circle)

# ax.legend(
#     fontsize='6',
#     handletextpad=0.0,
#     borderpad=0.2,
#     ncol=4, columnspacing=0.15,
#     loc='upper center')
# fig.savefig('data/snapshots_init.png', format='png', dpi=360)

# plt.show()
