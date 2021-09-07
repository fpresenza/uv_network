#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on dom ago 15 20:00:37 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt

import uvnpy.network as network
from uvnpy.network import disk_graph, subsets
from uvnpy.rsn import rigidity


plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


def scale(x):
    return x**(1/3)


L = 100
n = 80
nodes = np.arange(n)
np.random.seed(1)
x = np.random.uniform(-0.9*L, 0.9*L, (n, 2))

dmax = 0.4*L
E = disk_graph.edges(x, dmax)
A = disk_graph.adjacency(x, dmax)
print('is rigid: {}'.format(rigidity.algebraic_condition(A, x)))

hops = np.arange(1, 9)

fig, ax = plt.subplots(
    1, 2, figsize=(4, 2.5), gridspec_kw={'width_ratios': [1.5, 2]})
fig.subplots_adjust(wspace=0.24)
ax[0].tick_params(
   axis='both',       # changes apply to the x-axis
   which='both',      # both major and minor ticks are affected
   labelsize='xx-small')
ax[1].tick_params(
   axis='both',       # changes apply to the x-axis
   which='both',      # both major and minor ticks are affected
   labelsize='xx-small')

ax[0].set_aspect('equal')
ax[0].grid(1, lw=0.4)
ax[0].set_xlim(-L, L)
ax[0].set_ylim(-L, L)
ax[0].set_xlabel('x', fontsize='x-small', labelpad=0.2)
ax[0].set_ylabel('y', fontsize='x-small', labelpad=-2)

l4 = np.empty((n, len(hops)))
for h in hops:
    for i in nodes:
        Ai, xi = subsets.multihop_subframework(A, x, i, h)
        Li = rigidity.laplacian(Ai, xi)
        l4[i, h-1] = np.linalg.eigvalsh(Li)[3]

# print(l4[:, 2].min(), l4[:, 2].mean(), l4[:, 2].max())
i = np.argwhere(l4[:, 0] < 1e-3).ravel()
j = np.argwhere(l4[:, 1] < 1e-3).ravel()
k = np.argwhere(l4[:, 2] < 1e-3).ravel()
l = np.argwhere(l4[:, 3] < 1e-3).ravel()  # noqa
print(i)

network.plot.nodes(
    ax[0], np.delete(x, np.hstack([i, j, k, l]), axis=0),
    color='b', s=6, zorder=10)
network.plot.nodes(ax[0], x[i], color='orange', s=6, zorder=10)
network.plot.nodes(ax[0], x[j], color='purple', s=6, zorder=10)
network.plot.nodes(ax[0], x[k], color='green', s=6, zorder=10)
network.plot.nodes(ax[0], x[l], color='chocolate', s=6, zorder=10)
network.plot.edges(ax[0], x, E, color='0.2', alpha=0.6, lw=0.8)

# ax[1].plot(
#     hops, scale(l4.max(axis=0)),
#     color='k', lw=0.7, ds='steps-post', label=r'$\max {}_i \; \rho_{i, h}$')
for z in i:
    ax[1].plot(
        hops, scale(l4[z]),
        color='orange', lw=0.5,  ds='steps-post')
for z in j:
    ax[1].plot(
        hops, scale(l4[z]),
        color='purple', lw=0.5,  ds='steps-post')
for z in k:
    ax[1].plot(
        hops, scale(l4[z]),
        color='green', lw=0.5,  ds='steps-post')
for z in l:
    ax[1].plot(
        hops, scale(l4[z]),
        color='chocolate', lw=0.5,  ds='steps-post')
ax[1].plot(
    hops, scale(l4.min(axis=0)),
    color='0.6', lw=0.7, ls='--', ds='steps-post',
    label=r'$\min {}_i \; \rho_{i, h}$')

ax[1].set_aspect(4.8)
ax[1].grid(1, lw=0.4)
ax[1].set_xticks(hops)
ax[1].set_xticklabels(hops)
ax[1].set_yticks(scale(np.arange(0, 2, 0.5)))
ax[1].set_yticklabels(np.arange(0, 2, 0.5))
ax[1].set_xlabel('$h$ (reach)', fontsize='x-small')
# ax[1].set_ylabel(r'$\lambda_4$', fontsize='small', labelpad=0.05)
ax[1].legend(
    fontsize='x-small', handlelength=1, labelspacing=0.3, borderpad=0.2)

fig.savefig('/tmp/h_metrics.pdf', format='pdf')


# rho = l4.copy()
# rho[rho > 1e-6] **= -1
# fig, ax = plt.subplots(figsize=(2.5, 1.125))
# ax.semilogy(hops[2:], rho[:, 2:].max(axis=0), color='r', ds='steps-post')
# ax.semilogy(hops[2:], gmean(rho[:, 2:], axis=0), color='b', ds='steps-post')
# ax.semilogy(hops[2:], rho[:, 2:].sum(axis=0), color='g', ds='steps-post')
# ax.grid()

plt.show()
