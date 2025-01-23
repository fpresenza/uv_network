#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on jue sep 23 16:05:13 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt

from uvnpy import network
from uvnpy.network import subsets


plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'

# extraigo datos
t = np.loadtxt('/tmp/t.csv', delimiter=',')
x = np.loadtxt('/tmp/x.csv', delimiter=',')
hatx = np.loadtxt('/tmp/hatx.csv', delimiter=',')
u = np.loadtxt('/tmp/u.csv', delimiter=',')
fre = np.loadtxt('/tmp/fre.csv', delimiter=',')
re = np.loadtxt('/tmp/re.csv', delimiter=',')
A = np.loadtxt('/tmp/adjacency.csv', delimiter=',')
hops = np.loadtxt('/tmp/hops.csv', delimiter=',')
n = int(len(x[0])/2)
nodes = np.arange(n)
hops = hops.astype(int)

# reshapes
x = x.reshape(len(t), n, 2)
hatx = hatx.reshape(len(t), n, 2)
u = u.reshape(len(t), n, 2)
A = A.reshape(len(t), n, n)


# slice
kf = np.argmin(np.abs(t - 200))
t = t[:kf]
x = x[:kf]
hatx = hatx[:kf]
u = u[:kf]
fre = fre[:kf]
re = re[:kf]
A = A[:kf]

# calculos
edges = A.sum(-1).sum(-1)/2
load = np.array(
    [subsets.fast_degree_load_flat(a, hops, subsets.geodesics(a)) for a in A]
)
diam = np.array([network.diameter(a) for a in A])

fig, axes = plt.subplots(2, 2, figsize=(10, 4))

axes[0, 0].set_xlabel(r'$t [seg]$')
axes[0, 0].set_ylabel('x [m]')
axes[0, 0].grid(1)
axes[0, 0].plot(t, x[..., 0])
plt.gca().set_prop_cycle(None)

axes[0, 1].set_xlabel(r'$t [seg]$')
axes[0, 1].set_ylabel('y [m]')
axes[0, 1].grid(1)
axes[0, 1].plot(t, x[..., 1])
plt.gca().set_prop_cycle(None)

axes[1, 0].set_xlabel(r'$t [seg]$')
axes[1, 0].set_ylabel(r'$u_x [m/s]$')
axes[1, 0].grid(1)
axes[1, 0].plot(t, u[..., 0])
plt.gca().set_prop_cycle(None)

axes[1, 1].set_xlabel(r'$t [seg]$')
axes[1, 1].set_ylabel(r'$u_y [m/s]$')
axes[1, 1].grid(1)
axes[1, 1].plot(t, u[..., 1])
fig.savefig('/tmp/control.png', format='png', dpi=300)

fig, ax = plt.subplots(figsize=(4, 2.5))
fig.subplots_adjust(bottom=0.2, left=0.15)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='medium'
)
ax.grid(lw=0.4)
ax.set_xlabel(r'Time [$s$]', fontsize=10)
# ax.set_ylabel(r'Rigidity Metrics', fontsize=10)
ax.semilogy(t, fre, color='k', label=r'$\lambda_{D+1}$')
ax.fill_between(t, re.min(axis=1), re.max(axis=1), alpha=0.5)
# ax.semilogy(t, re.max(axis=1), lw=0.8, label='max')
ax.set_ylim(bottom=1e-3)
ax.legend(
    fontsize='medium', handlelength=1.5,
    labelspacing=0.5, borderpad=0.2)
fig.savefig('/tmp/eigenvalues.png', format='png', dpi=300)

fig, ax = plt.subplots(figsize=(4, 2.5))
fig.subplots_adjust(bottom=0.2, left=0.15)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='medium'
)
ax.grid(lw=0.4)
ax.set_xlabel(r'Time [$s$]', fontsize=10)
# ax.set_ylabel(r'Communication Metrics', fontsize=10)
ax.plot(
    t, np.full(len(t), hops.max() * 2),
    label=r'$\mathrm{RTD}$', marker='o', markersize=3, markevery=200, lw=0.7)
ax.plot(
    t, diam,
    label=r'$\mathrm{diam}$', marker='s', markersize=3, markevery=200, lw=0.7)
ax.plot(
    t, load / n,
    label=r'$\mathrm{CL}$', marker='x', markersize=4, markevery=200, lw=0.7)
ax.plot(
    t, 2 * edges / n,
    label=r'$\mathrm{deg}$', marker='v', markersize=3, markevery=200, lw=0.7)

ax.legend(
    fontsize='medium', handlelength=1.5,
    labelspacing=0.5, borderpad=0.2)
fig.savefig('/tmp/load.png', format='png', dpi=300)

# instantes
hops = hops[0]
instants = np.array([0., 10, 20, 50, 100, 200])
lim = np.abs(x).max()
one_hop_rigid = hops == 1
two_hop_rigid = hops == 2
three_hop_rigid = hops == 3
four_hop_rigid = hops == 4

for i, tk in enumerate(instants):
    fig, ax = plt.subplots(figsize=(2.2, 2))
    ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize='xx-small')
    ax.grid(1, lw=0.4)
    ax.set_aspect('equal')

    a = 10 * (lim // 10 + 1)
    b = a // 2
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
            transform=ax.transAxes, color='k', fontsize=5)

    network.plot.nodes(
        ax, x[k, one_hop_rigid],
        marker='o', color='royalblue', s=7, zorder=20, label=r'$h=1$')
    network.plot.nodes(
        ax, x[k, two_hop_rigid],
        marker='D', color='chocolate', s=7, zorder=20, label=r'$h=2$')
    network.plot.nodes(
        ax, x[k, three_hop_rigid],
        marker='s', color='mediumseagreen', s=7, zorder=20, label=r'$h=3$')
    network.plot.nodes(
        ax, x[k, four_hop_rigid],
        marker='^', color='purple', s=7, zorder=10, label=r'$h=4$')
    network.plot.edges(ax, x[k], A[k], color='k', lw=0.5)

    network.plot.nodes(
        ax, x[max(0, k-600):k:20, one_hop_rigid],
        marker='.', color='royalblue', s=1, zorder=1, lw=0.5)
    network.plot.nodes(
        ax, x[max(0, k-600):k:20, two_hop_rigid],
        marker='.', color='chocolate', s=1, zorder=1, lw=0.5)
    network.plot.nodes(
        ax, x[max(0, k-600):k:20, three_hop_rigid],
        marker='.', color='mediumseagreen', s=1, zorder=1, lw=0.5)
    network.plot.nodes(
        ax, x[max(0, k-600):k:20, four_hop_rigid],
        marker='.', color='purple', s=1, zorder=1, lw=0.5)

    if i == 0:
        ax.legend(
            fontsize='xx-small',
            handletextpad=0.0,
            borderpad=0.2,
            ncol=4, columnspacing=0.2,
            loc='upper center')

    fig.savefig('/tmp/instants_{}.png'.format(int(tk)), format='png', dpi=300)


# animacion
timestep = np.diff(t).mean()
frames = np.empty((t.size, 3), dtype=np.ndarray)
E = network.edges_from_adjacency(A[0])
steps = list(enumerate(t))
for k, tk in steps:
    E = network.edges_from_adjacency(A[k])
    X = np.vstack([x[k], hatx[k]])
    frames[k] = tk, X, E


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
anim = network.plot.Animate(fig, ax, timestep/2, frames, maxlen=50)
one_hop_rigid = hops == 1
two_hop_rigid = hops == 2
three_hop_rigid = hops == 3
four_hop_rigid = hops == 4
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
# anim.ax.legend(ncol=5)
# anim.run()
# anim.run('/tmp/multihop.mp4')

# plt.show()
