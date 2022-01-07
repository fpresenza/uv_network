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


class CoverageAnimate(network.plot.Animate):
    def __init__(self, *args, **kwargs):
        super(CoverageAnimate, self).__init__(*args, **kwargs)

    def _update_extra_artists(self, frame):
        P = frame[1]
        T = frame[3]
        n = int(len(P)/2)
        for i in range(n):
            self._extra_artists[i].center = P[i]
        untracked = T[:, 2].astype(bool)
        Tt = T[~untracked]
        Tu = T[untracked]
        self._extra_artists[-2].set_data(Tt[:, 0], Tt[:, 1])
        self._extra_artists[-1].set_data(Tu[:, 0], Tu[:, 1])


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
# kf = np.argmin(np.abs(t - 200))
# t = t[:kf]
# x = x[:kf]
# hatx = hatx[:kf]
# u = u[:kf]
# fre = fre[:kf]
# re = re[:kf]
# A = A[:kf]

# calculos
edges = A.sum(-1).sum(-1)/2
# load = np.array([subsets.degree_load_std(a, h) for a, h in zip(A, extents)])

fig, ax = plt.subplots(figsize=(3, 1.5))
fig.subplots_adjust(bottom=0.2)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='xx-small')
ax.grid(1, lw=0.4)
ax.plot(t, extents, lw=0.8, ds='steps-post')
ax.set_xlabel(r'$t$ (sec)', fontsize='x-small', labelpad=0.6)
ax.set_ylabel(r'Extents', fontsize='x-small', labelpad=0.6)

fig, axes = plt.subplots(2, 2, figsize=(10, 4))

axes[0, 0].set_xlabel(r'$t [seg]$')
axes[0, 0].set_ylabel('x [m]')
axes[0, 0].grid(1)
axes[0, 0].plot(t, x[..., 0], ds='steps-post')
plt.gca().set_prop_cycle(None)

axes[0, 1].set_xlabel(r'$t [seg]$')
axes[0, 1].set_ylabel('y [m]')
axes[0, 1].grid(1)
axes[0, 1].plot(t, x[..., 1], ds='steps-post')
plt.gca().set_prop_cycle(None)

axes[1, 0].set_xlabel(r'$t [seg]$')
axes[1, 0].set_ylabel(r'$u_x [m/s]$')
axes[1, 0].grid(1)
axes[1, 0].plot(t, u[..., 0], ds='steps-post')
plt.gca().set_prop_cycle(None)

axes[1, 1].set_xlabel(r'$t [seg]$')
axes[1, 1].set_ylabel(r'$u_y [m/s]$')
axes[1, 1].grid(1)
axes[1, 1].plot(t, u[..., 1], ds='steps-post')
fig.savefig('/tmp/control.png', format='png', dpi=300)

fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.25))
fig.subplots_adjust(bottom=0.215, top=0.925, wspace=0.33, right=0.975)
for ax in axes:
    ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize='xx-small')
    ax.grid(1, lw=0.4)

axes[0].set_xlabel(r'$t$ (sec)', fontsize='x-small', labelpad=0.6)
axes[0].set_ylabel(r'Rigidity eigenvalues', fontsize='x-small', labelpad=0.6)
axes[0].semilogy(t, re.min(axis=1), lw=0.8, label='min')
axes[0].semilogy(t, re.mean(axis=1), lw=0.8, label='mean')
axes[0].semilogy(t, re.max(axis=1), lw=0.8, label='max')
axes[0].semilogy(t, fre, ls='--', color='k', lw=0.8, label='whole')
axes[0].set_ylim(bottom=1e-3)
axes[0].legend(
    fontsize='xx-small', handlelength=1, labelspacing=0.4,
    borderpad=0.2, handletextpad=0.2, framealpha=1.,
    ncol=2, columnspacing=1)
plt.gca().set_prop_cycle(None)

axes[1].set_xlabel(r'$t$ (sec)', fontsize='x-small', labelpad=0.6)
axes[1].set_ylabel(
    r'Std. Load', fontsize='x-small', labelpad=0.6)
# axes[1].plot(t, edges, lw=0.8)
# axes[1].plot(t, load/2/edges, lw=0.8)
# axes[1].hlines(2*n-3, t[0], t[-1], color='k', ls='--', lw=0.8)
axes[1].set_ylim(bottom=1)
fig.savefig('/tmp/simu_metrics.png', format='png', dpi=300)

# instantes
instants = np.array([0., 200])
# lim = np.abs(x).max()
lim = 50
one_hop_rigid = extents[0] == 1
two_hop_rigid = extents[0] == 2
three_hop_rigid = extents[0] == 3
four_hop_rigid = extents[0] == 4

fig, axes = plt.subplots(1, 2, figsize=(3.4, 2))
fig.subplots_adjust(wspace=0.215, left=0.11, right=0.975)
axes = axes.ravel()
for i, ax in enumerate(axes):
    ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize='xx-small')
    ax.grid(1, lw=0.4)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$\mathrm{x}$', fontsize='x-small', labelpad=0.6)
    if i % 2 == 0:
        ax.set_ylabel(r'$\mathrm{y}$', fontsize='x-small', labelpad=0)
    ax.set_xticks([-100, 0, 100])
    ax.set_yticks([-100, 0, 100])
    ax.set_xticklabels([-100, 0, 100])
    ax.set_yticklabels([-100, 0, 100])

for tk, ax in zip(instants, axes):
    k = np.argmin(np.abs(t - tk))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.text(
            0.05, 0.01, r't = {:.2f}s'.format(tk),
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes, color='k', fontsize=5)
    network.plot.nodes(
        ax, x[k, one_hop_rigid],
        marker='o', color='royalblue', s=7, zorder=20, label=r'$1$')
    network.plot.nodes(
        ax, x[k, two_hop_rigid],
        marker='D', color='chocolate', s=7, zorder=20, label=r'$2$')
    network.plot.nodes(
        ax, x[k, three_hop_rigid],
        marker='s', color='mediumseagreen', s=7, zorder=20, label=r'$3$')
    network.plot.nodes(
        ax, x[k, four_hop_rigid],
        marker='^', color='purple', s=7, zorder=10, label=r'$4$')
    network.plot.edges(ax, x[k], A[k], color='k', lw=0.5)

network.plot.nodes(
    axes[1], x[::20, one_hop_rigid],
    marker='.', color='royalblue', s=1, zorder=1, lw=0.5)
network.plot.nodes(
    axes[1], x[::20, two_hop_rigid],
    marker='.', color='chocolate', s=1, zorder=1, lw=0.5)
network.plot.nodes(
    axes[1], x[::20, three_hop_rigid],
    marker='.', color='mediumseagreen', s=1, zorder=1, lw=0.5)
network.plot.nodes(
    axes[1], x[::20, four_hop_rigid],
    marker='.', color='purple', s=1, zorder=1, lw=0.5)

axes[0].legend(
    fontsize='xx-small',
    handletextpad=0.0,
    borderpad=0.2,
    ncol=4, columnspacing=0.2,
    loc='upper center')

fig.savefig('/tmp/instants.png', format='png', dpi=300)


# animacion
timestep = np.diff(t).mean()
frames = np.empty((t.size, 4), dtype=np.ndarray)
# E = network.edges_from_adjacency(A[0])
steps = list(enumerate(t))
for k, tk in steps:
    E = network.edges_from_adjacency(A[k])
    X = np.vstack([x[k], hatx[k]])
    T = targets[k]
    frames[k] = tk, X, E, T


fig, ax = plt.subplots()
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='xx-small')
ax.set_aspect('equal')
ax.grid(1, lw=0.4)
ax.set_xlabel(r'$x$', fontsize='x-small', labelpad=0.6)
ax.set_ylabel(r'$y$', fontsize='x-small', labelpad=0.6)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
# anim = network.plot.Animate(fig, ax, timestep/2, frames, maxlen=50)
anim = CoverageAnimate(fig, ax, timestep/2, frames, maxlen=50)

one_hop_rigid = extents[0] == 1
two_hop_rigid = extents[0] == 2
three_hop_rigid = extents[0] == 3
four_hop_rigid = extents[0] == 4
anim.set_teams(
    {'ids': nodes[one_hop_rigid], 'tail': True,
        'style': {'color': 'royalblue', 'marker': 'o', 'markersize': 5}},
    {'ids': nodes[two_hop_rigid], 'tail': True,
        'style': {'color': 'chocolate', 'marker': 'D', 'markersize': 5}},
    {'ids': nodes[three_hop_rigid], 'tail': True,
        'style': {'color': 'mediumseagreen', 'marker': 's', 'markersize': 5}},
    {'ids': nodes[four_hop_rigid], 'tail': True,
        'style': {'color': 'purple', 'marker': '^', 'markersize': 5}},
    {'ids': nodes + nodes[-1] + 1, 'tail': False,
        'style': {'color': 'red', 'marker': '+', 'markersize': 5}})  # noqa
anim.set_edgestyle(color='0.4', alpha=0.6, lw=0.8)


# circles = [plt.Circle(pi, 3., alpha=0.4) for pi in x[0]]
circles = []
for p in x[0]:
    circle = plt.Circle(p, 3., alpha=0.4)
    circles.append(circle)
    ax.add_artist(circle)
tracked = ax.plot([], [], ls='', marker='s', markersize=3, color='y')
untracked = ax.plot(
    targets[0, :, 0], targets[0, :, 1],
    ls='', marker='s', markersize=3, color='0.5')
extras = circles + tracked + untracked
anim.set_extra_artists(*extras)

# anim.ax.legend(ncol=5)
anim.run()
# anim.run('/tmp/multihop.mp4')

plt.show()
