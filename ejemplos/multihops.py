#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on vie sep 10 10:38:33 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx  # noqa

from uvnpy.network import disk_graph
from uvnpy.rsn import rigidity


# gpsic.plotting.core.set_pubstyle(style='sans-serif')
plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'

hL = 50
L = 2 * hL
L2 = L**2
density = 1/100  # inverso del area que le corresponde a cada agente
n = int(density * L2)
# d_factor = np.array([1.75, 2.0, 2.5])
# dmax = d_factor / np.sqrt(density)
dmax = np.array([17.5, 20, 25])

R = len(dmax)
N = 250
min_hops = np.empty((R, N, n))
# max_min_hops = np.empty((R, N))
# diam = np.empty((R, N))

for i in range(R):
    k = 0
    while k < N:
        x = np.random.uniform(-hL, hL, (n, 2))
        A = disk_graph.adjacency(x, dmax[i])
        try:
            min_hops[i, k] = rigidity.minimum_hops(A, x)
            # G = nx.from_numpy_matrix(A)
            # diam[i, k] = nx.algorithms.distance_measures.diameter(G)
            # max_min_hops[i, k] = min_hops[i, k].max()
            # print(k, max_min_hops[i, k], diam[i, k])
            print(k)
            k += 1
        except ValueError as e:
            # print(e)
            pass
        except StopIteration as e:
            print(e)
            pass

print(min_hops)
# print(max_min_hops)
# print(diam)

fig, ax = plt.subplots(1, 2, figsize=(4, 2))
fig.subplots_adjust(bottom=0.225, left=0.15, wspace=0.33, top=0.81)
ax[0].tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='xx-small')
ax[0].grid(1, lw=0.4)
ax[1].tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='xx-small')
ax[1].grid(1, lw=0.4)

# porcentajes
hops = np.arange(1, 10).astype(int)
freq = np.empty((R, len(hops)))
mark = ['s', '^', 'o']
for i in range(R):
    for k, hop in enumerate(hops):
        freq[i, k] = (min_hops[i].max(axis=1) == hop).mean() * 100
    ax[0].plot(
        hops, freq[i],
        marker=mark[i], markersize=3, lw=0.7,
        label=r'$\Omega = {}$'.format(dmax[i]))
    ax[1].plot(
        hops, freq[i].cumsum(),
        marker=mark[i], markersize=3, lw=0.7,
        label=r'$\Omega / L = {}$'.format(dmax[i]/L))

print(freq)
print(freq.cumsum(axis=1))
ax[0].set_xticks(hops)
ax[0].set_xticklabels(hops)
ax[0].set_xlabel(r'Maximum extent ($\eta$)', fontsize='x-small')
ax[0].set_ylabel(r'$\%$ of networks', fontsize='x-small', labelpad=1)
ax[0].legend(
    fontsize='xx-small', handlelength=2, labelspacing=0.5,
    borderpad=0.2, ncol=3, columnspacing=1,
    loc='upper center', bbox_to_anchor=(1.15, 1.3))

ax[1].set_xticks(hops)
ax[1].set_xticklabels(hops)
ax[1].set_xlabel(r'Maximum extent ($\eta$)', fontsize='x-small')
ax[1].set_ylabel(r'Cumulative $\%$', fontsize='x-small', labelpad=1)
# ax[1].legend(
#     fontsize='xx-small', handlelength=1, labelspacing=0.3, borderpad=0.2)

fig.savefig('/tmp/multihops.pdf', format='pdf')


# hops por nodo
fig, ax = plt.subplots(figsize=(1.5, 1.5))
fig.subplots_adjust(bottom=0.225, left=0.23)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='xx-small')
ax.grid(1, lw=0.4)

hops = np.arange(1, 10).astype(int)
freq = np.empty((R, len(hops)))
mark = ['s', '^', 'o']
for i in range(R):
    for k, h in enumerate(hops):
        freq[i, k] = (min_hops[i] == h).mean() * 100
    ax.plot(
        hops, freq[i],
        marker=mark[i], markersize=3, lw=0.7,
        label=r'$\Omega = {}$'.format(dmax[i]))
print(freq.cumsum(axis=1))
ax.set_xticks(hops[::2])
ax.set_xticklabels(hops[::2])
ax.set_xlabel(r'Extent ($\eta_i$)', fontsize='x-small')
ax.set_ylabel(r'$\%$ of nodes', fontsize='x-small', labelpad=1)
ax.legend(
    fontsize='xx-small', handlelength=1,
    labelspacing=0.3, borderpad=0.2)

fig.savefig('/tmp/multihops_nodes.pdf', format='pdf')


# hist de hops / diam
# fig, ax = plt.subplots(1, 2, figsize=(4, 1.5))
# fig.subplots_adjust(bottom=0.225, left=0.15)
# ax[0].tick_params(
#     axis='both',       # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     labelsize='xx-small')
# ax[1].tick_params(
#     axis='both',       # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     labelsize='xx-small')

# ax[0].grid(1, lw=0.4)
# ax[0].set_xlim(0, 1)
# ax[0].set_xticks(np.arange(0, 1.1, 0.1))
# ax[0].set_xticklabels(np.round(np.arange(0, 1.1, 0.1), 1))
# ax[0].set_yticks(np.arange(0, 9, 2))
# ax[0].set_yticklabels(np.arange(0, 90, 20))

# mark = ['s', '^', 'o']
# for i in range(R):
#     freq, hops = np.histogram(
#         max_min_hops[i], bins=10, range=(0, 1), density=True)
#     bars = np.cumsum(np.diff(hops)) - 0.05
#     ax[0].plot(
#         bars, freq,
#         marker=mark[i], markersize=3, lw=0.7,
#         label=r'$\Omega / L= {}$'.format(dmax[i]/L))

#     # ax.hist(
#     #     max_min_hops[i]/diam[i],
#     #     bins=10, range=(0, 1), histtype='bar', density=True, stacked=True,
#     #     label=r'$\Omega / L = {}$'.format(d_factor[i]))

# ax[0].set_xlabel(r'hops required / diameter  ($\xi$)', fontsize='x-small')
# ax[0].set_ylabel(r'$\%$ of networks', fontsize='x-small')
# ax[0].legend(
#     fontsize='xx-small', handlelength=1, labelspacing=0.3, borderpad=0.2)

# # ax[1].hist(diam.T, histtype='bar', density=True, stacked=True)
# fig.savefig('/tmp/multihops_diam.pdf', format='pdf')


plt.show()
