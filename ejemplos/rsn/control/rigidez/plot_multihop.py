#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on jue sep 23 16:05:13 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from uvnpy import network
from uvnpy.network import subsets
# from uvnpy.rsn import rigidity


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
# A2 = np.loadtxt('/tmp/adjacency2.csv', delimiter=',')
hops = np.loadtxt('/tmp/hops.csv', delimiter=',')

n = int(len(x[0])/2)
nodes = np.arange(n)
hops = hops.astype(int)

# reshapes
N = len(t)
x = x.reshape(N, n, 2)
hatx = hatx.reshape(N, n, 2)
# print(x[0], hatx[0])
u = u.reshape(N, n, 2)
A = A.reshape(N, n, n)
# A2 = A2.reshape(N, n, n)

# slice
kf = np.argmin(np.abs(t - 200))
t = t[:kf]
x = x[:kf]
hatx = hatx[:kf]
u = u[:kf]
fre = fre[:kf]
re = re[:kf]
A = A[:kf]
N = len(t)

# calculos
edges = A.sum(-1).sum(-1)/2

load = np.array([subsets.degree_load_std(a, hops) for a in A])
# load2 = np.empty(N, dtype=np.ndarray)
# for k in range(N):
#     h = rigidity.minimum_hops(A2[k], x[k])
#     load2[k] = subsets.degree_load_std(A2[k], h)

""" Control """
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
fig.savefig('/tmp/control.png', format='png', dpi=360)


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

ax.set_xlabel(r'$t$ [$sec$]', fontsize=10)
ax.set_ylabel(
    'Autovalores \n de Rigidez',
    fontsize=10)
ax.semilogy(t, re.min(axis=1), lw=0.9, label='mín')
ax.semilogy(t, re.mean(axis=1), lw=0.9, label='medio')
ax.semilogy(t, re.max(axis=1), lw=0.9, label='max')
ax.semilogy(t, fre, ls='--', color='k', lw=0.9, label='framework')
ax.set_ylim(bottom=1e-3)
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
ax.plot(t, load/2/edges[0], lw=0.9)
# ax.plot(t, load/2/edges[0], lw=0.9, ls='--', label='$h$ fijo')
# ax.plot(t, load2/2/edges[0], lw=0.9, label='$h$ variable')
ax.hlines(1, t[0], t[-1], color='k', ls='--', lw=0.9)
ax.set_ylim(bottom=0)
# ax.legend(
#     fontsize=8, handlelength=1, labelspacing=0.4,
#     borderpad=0.2, handletextpad=0.2, framealpha=1.)
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
one_hop_rigid = hops == 1
two_hop_rigid = hops == 2
three_hop_rigid = hops == 3
four_hop_rigid = hops == 4

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small')
ax.grid(1, lw=0.4)
ax.set_aspect('equal')
# ax.set_xlabel(r'$\mathrm{x}$', fontsize=10, labelpad=0.6)
# if i % 2 == 0:
#     ax.set_ylabel(r'$\mathrm{y}$', fontsize=10, labelpad=0)
ax.set_xticks([-100, 0, 100])
ax.set_yticks([-100, 0, 100])
ax.set_xticklabels([-100, 0, 100])
ax.set_yticklabels([-100, 0, 100])
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 115)

network.plot.nodes(
    ax, x[0, one_hop_rigid],
    marker='o', color='royalblue', s=7, zorder=20, label=r'$h_0 = 1$')
network.plot.nodes(
    ax, x[0, two_hop_rigid],
    marker='D', color='chocolate', s=7, zorder=20, label=r'$h_0 = 2$')
network.plot.nodes(
    ax, x[0, three_hop_rigid],
    marker='s', color='mediumseagreen', s=7, zorder=20, label=r'$h_0 = 3$')
network.plot.nodes(
    ax, x[0, four_hop_rigid],
    marker='^', color='purple', s=7, zorder=10, label=r'$h_0 = 4$')
network.plot.edges(ax, x[0], A[0], color='k', lw=0.5)

circle = Circle(x[0, 15], 5, facecolor='None', linewidth=1, edgecolor='red')
ax.add_artist(circle)
circle = Circle(x[0, 41], 5, facecolor='None', linewidth=1, edgecolor='red')
ax.add_artist(circle)

ax.legend(
    fontsize='6',
    handletextpad=0.0,
    borderpad=0.2,
    ncol=4, columnspacing=0.15,
    loc='upper center')
fig.savefig('/tmp/instants_init.png', format='png', dpi=360)

instants = np.array([0., 10, 20, 50, 100, 200])
lim = 160   # 1.1 * np.abs(x).max()

for i, tk in enumerate(instants):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize='x-small')
    ax.grid(1, lw=0.4)
    ax.set_aspect('equal')
    # ax.set_xlabel(r'$\mathrm{x}$', fontsize=10, labelpad=0.6)
    # if i % 2 == 0:
    #     ax.set_ylabel(r'$\mathrm{y}$', fontsize=10, labelpad=0)
    ax.set_xticks([-100, 0, 100])
    ax.set_yticks([-100, 0, 100])
    ax.set_xticklabels([-100, 0, 100])
    ax.set_yticklabels([-100, 0, 100])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    k = np.argmin(np.abs(t - tk))
    ax.text(
            0.05, 0.01, r't = {:.2f}s'.format(tk),
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes, color='r', fontsize=8)
    network.plot.nodes(
        ax, x[k, one_hop_rigid],
        marker='o', color='royalblue', s=7, zorder=20, label=r'$h_0 = 1$')
    network.plot.nodes(
        ax, x[k, two_hop_rigid],
        marker='D', color='chocolate', s=7, zorder=20, label=r'$h_0 = 2$')
    network.plot.nodes(
        ax, x[k, three_hop_rigid],
        marker='s', color='mediumseagreen', s=7, zorder=20, label=r'$h_0 = 3$')
    network.plot.nodes(
        ax, x[k, four_hop_rigid],
        marker='^', color='purple', s=7, zorder=10, label=r'$h_0 = 4$')
    network.plot.edges(ax, x[k], A[k], color='k', lw=0.5)

    tail = 150 * i
    network.plot.nodes(
        ax, x[max(0, k-tail):k+1:5, one_hop_rigid],
        marker='.', color='royalblue', s=1, zorder=1, lw=0.5)
    network.plot.nodes(
        ax, x[max(0, k-tail):k+1:5, two_hop_rigid],
        marker='.', color='chocolate', s=1, zorder=1, lw=0.5)
    network.plot.nodes(
        ax, x[max(0, k-tail):k+1:5, three_hop_rigid],
        marker='.', color='mediumseagreen', s=1, zorder=1, lw=0.5)
    network.plot.nodes(
        ax, x[max(0, k-tail):k+1:5, four_hop_rigid],
        marker='.', color='purple', s=1, zorder=1, lw=0.5)

    fig.savefig('/tmp/instants_{}.png'.format(int(tk)), format='png', dpi=360)
