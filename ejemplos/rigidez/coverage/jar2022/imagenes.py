#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on mié 29 dic 2021 16:41:13 -03
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt

from uvnpy import network
from uvnpy.network import subsets   # noqa


plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


# extraigo datos
t = np.loadtxt('/tmp/t.csv', delimiter=',')
x = np.loadtxt('/tmp/position.csv', delimiter=',')
hatx = np.loadtxt('/tmp/est_position.csv', delimiter=',')
u = np.loadtxt('/tmp/action.csv', delimiter=',')
fre = np.loadtxt('/tmp/fre.csv', delimiter=',')
re = np.loadtxt('/tmp/re.csv', delimiter=',')
A = np.loadtxt('/tmp/adjacency.csv', delimiter=',')
extents = np.loadtxt('/tmp/extents.csv', delimiter=',')
targets = np.loadtxt('/tmp/targets.csv', delimiter=',')

n = int(len(x[0])/2)
nodes = np.arange(n)
extents = extents.astype(int)

# reshapes
x = x.reshape(len(t), n, 2)
hatx = hatx.reshape(len(t), n, 2)
# print(x[0], hatx[0])
u = u.reshape(len(t), n, 2)
A = A.reshape(len(t), n, n)
targets = targets.reshape(len(t), -1, 3)

# slice
ki = np.argmin(np.abs(t - 0))
kf = np.argmin(np.abs(t - 25))
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
load = np.array([subsets.degree_load_std(a, h) for a, h in zip(A, extents)])

""" Extents """
diam = np.array([network.diameter(adj) for adj in A])
centralization_index = extents.max(axis=-1) / diam

fig, ax = plt.subplots(figsize=(4.0, 1.75))
fig.subplots_adjust(
    bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='small')
ax.grid(1, lw=0.4)

ax.plot(t, extents, lw=0.8, ds='steps-post')
ax.set_xlabel(r'$t$ [$sec$]', fontsize='10')
ax.set_ylabel(r'Extents', fontsize='10')
fig.savefig('/tmp/extents.png', format='png', dpi=360)

""" Control """
fig, axes = plt.subplots(2, 2, figsize=(10, 4))

axes[0, 0].set_xlabel('$t$ [$seg$]')
axes[0, 0].set_ylabel('$x_i$ [$m$]')
axes[0, 0].grid(1)
axes[0, 0].plot(t, x[..., 0])
plt.gca().set_prop_cycle(None)

axes[0, 1].set_xlabel('$t$ [$seg$]')
axes[0, 1].set_ylabel('$y_i$ [$m$]')
axes[0, 1].grid(1)
axes[0, 1].plot(t, x[..., 1])
plt.gca().set_prop_cycle(None)

axes[1, 0].set_xlabel('$t$ [$seg$]')
axes[1, 0].set_ylabel('$u_x$ [$m/s$]')
axes[1, 0].grid(1)
axes[1, 0].plot(t, u[..., 0], ds='steps-post')
plt.gca().set_prop_cycle(None)

axes[1, 1].set_xlabel('$t$ [$seg$]')
axes[1, 1].set_ylabel('$u_y$ [$m/s$]')
axes[1, 1].grid(1)
axes[1, 1].plot(t, u[..., 1], ds='steps-post')
fig.savefig('/tmp/control.png', format='png', dpi=360)

""" Position """
""" x """
fig, ax = plt.subplots(figsize=(2.75, 1.75))
fig.subplots_adjust(
    bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.2)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='small')
ax.grid(1, lw=0.4)

ax.set_xlabel(r'$t$ [$sec$]', fontsize='10')
ax.set_ylabel('posición-$x$ [$m$]', fontsize='10')
ax.plot(t[ki:kf], x[ki:kf, :, 0], lw=0.9)
fig.savefig('/tmp/pos_x.png', format='png', dpi=360)

""" y """
fig, ax = plt.subplots(figsize=(2.75, 1.75))
fig.subplots_adjust(
    bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.2)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='small')
ax.grid(1, lw=0.4)

ax.set_xlabel(r'$t$ [$sec$]', fontsize='10')
ax.set_ylabel('posición-$y$ [$m$]', fontsize='10')
ax.plot(t[ki:kf], x[ki:kf, :, 1], lw=0.9)
fig.savefig('/tmp/pos_y.png', format='png', dpi=360)

""" Metrics """
""" eigenvalues """
fig, ax = plt.subplots(figsize=(4.0, 1.75))
fig.subplots_adjust(
    bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='small')
ax.grid(1, lw=0.4)

ax.set_xlabel(r'$t$ [$sec$]', fontsize='10')
ax.set_ylabel('Autovalores \n de Rigidez', fontsize='10')
ax.semilogy(t, re.min(axis=1), lw=0.9, label='min')
ax.semilogy(t, re.mean(axis=1), lw=0.9, label='medio')
ax.semilogy(t, re.max(axis=1), lw=0.9, label='max')
ax.semilogy(t, fre, ls='--', color='k', lw=0.9, label=r'framework')
ax.set_ylim(bottom=1e-2)
ax.legend(
    fontsize=8, handlelength=1, labelspacing=0.4,
    borderpad=0.2, handletextpad=0.2, framealpha=1.,
    ncol=2, columnspacing=1)
fig.savefig('/tmp/eigenvalues.png', format='png', dpi=360)

""" load """
fig, ax = plt.subplots(figsize=(4.0, 1.75))
fig.subplots_adjust(
    bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='small')
ax.grid(1, lw=0.4)
ax.set_xlabel(r'$t$ [$sec$]', fontsize=10)
ax.set_ylabel(r'Carga ($\mathcal{L} / \bar{n}$)', fontsize=10)
# ax[1.plot(t, edges, lw=0.9)
ax.plot(t, load/2/edges[0], lw=0.9)
ax.hlines(1, t[0], t[-1], color='k', ls='--', lw=0.9)
ax.set_ylim(bottom=0)
fig.savefig('/tmp/load.png', format='png', dpi=360)

""" pos error """
e2 = np.square(x - hatx).sum(axis=-1)
fig, ax = plt.subplots(figsize=(4.0, 1.75))
fig.subplots_adjust(
    bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='small')
ax.grid(1, lw=0.4)
ax.set_xlabel(r'$t$ [$sec$]', fontsize=10)
ax.set_ylabel('Error Cuadrático \n [$m^2$]', fontsize=10)
ax.semilogy(t, e2.min(axis=1), lw=0.9, label='mín')
ax.semilogy(t, e2.mean(axis=1), lw=0.9, label='medio')
ax.semilogy(t, e2.max(axis=1), lw=0.9, label='max')
ax.set_ylim(bottom=1e-5)
ax.legend(
    fontsize=8, handlelength=1, labelspacing=0.4,
    borderpad=0.2, handletextpad=0.2, framealpha=1.,
    ncol=2, columnspacing=1)
fig.savefig('/tmp/pos_error.png', format='png', dpi=360)

""" INSTANTS """
instants = np.array([0., 5.5, 7, 40., 125., 250])
# lim = np.abs(x).max()
lim = 40

for i, tk in enumerate(instants):
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
    ax.set_xlim(-a, a)
    ax.set_ylim(-a, a)
    ax.set_xticks([-a, -b, 0, b, a])
    ax.set_yticks([-a, -b, 0, b, a])
    ax.set_xticklabels([-a, -b, 0, b, a])
    ax.set_yticklabels([-a, -b, 0, b, a])

    k = np.argmin(np.abs(t - tk))
    ax.text(
            0.05, 0.01, r't = {:.2f}s'.format(tk),
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes, color='r', fontsize=8)
    '''
    if i == 0:
        ax.set_xlim(-0.6*lim, 0.6*lim)
        ax.set_ylim(-0.6*lim, 0.6*lim)
        ax.set_xticks([-0.4 * lim, 0, -0.4 * lim])
        ax.set_yticks([-0.4 * lim, 0, -0.4 * lim])
        ax.set_xticklabels([-0.4 * lim, 0, -0.4 * lim])
        ax.set_yticklabels([-0.4 * lim, 0, -0.4 * lim])
    '''

    one_hop_rigid = extents[k] == 1
    two_hop_rigid = extents[k] == 2
    three_hop_rigid = extents[k] == 3

    network.plot.nodes(
        ax, x[k, one_hop_rigid],
        marker='o', color='royalblue', s=8, zorder=20, label=r'$h_0=1$')
    network.plot.nodes(
        ax, x[k, two_hop_rigid],
        marker='D', color='chocolate', s=8, zorder=20, label=r'$h_0=2$')
    network.plot.nodes(
        ax, x[k, three_hop_rigid],
        marker='s', color='mediumseagreen', s=8, zorder=20, label=r'$h_0=3$')
    network.plot.edges(ax, x[k], A[k], color='k', lw=0.5)

    untracked = targets[k, :, 2].astype(bool)
    tracked = np.logical_not(untracked)
    ax.scatter(
        targets[k, untracked, 0], targets[k, untracked, 1],
        marker='s', s=4, color='0.6')
    ax.scatter(
        targets[k, tracked, 0], targets[k, tracked, 1],
        marker='s', s=4, color='0.2')

    network.plot.nodes(
        ax, x[max(0, k-150):k+1:5, one_hop_rigid],
        marker='.', color='royalblue', s=1, zorder=1, lw=0.5)
    network.plot.nodes(
        ax, x[max(0, k-150):k+1:5, two_hop_rigid],
        marker='.', color='chocolate', s=1, zorder=1, lw=0.5)
    network.plot.nodes(
        ax, x[max(0, k-150):k+1:5, three_hop_rigid],
        marker='.', color='mediumseagreen', s=1, zorder=1, lw=0.5)

    if i == 0:
        ax.legend(
            fontsize='6',
            handletextpad=0.0,
            borderpad=0.2,
            ncol=4, columnspacing=0.15,
            loc='upper center')

    fig.savefig('/tmp/instants_{}.png'.format(int(tk)), format='png', dpi=360)

# plt.show()
