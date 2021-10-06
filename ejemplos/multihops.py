#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on vie sep 10 10:38:33 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx  # noqa

from uvnpy.network import disk_graph, subsets
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
dmax = np.array([25, 20, 17.5])

R = len(dmax)
N = 250
# min_hops = np.zeros((R, N, n), dtype=int)
edges = np.zeros((R, N), dtype=int)
load = np.zeros((R, N))
max_min_hops = np.zeros((R, N))
diam = np.zeros((R, N))

for i in range(R):
    k = 0
    while k < N:
        x = np.random.uniform(-hL, hL, (n, 2))
        A = disk_graph.adjacency(x, dmax[i])
        try:
            min_hops = rigidity.minimum_hops(A, x)
            edges[i, k] = int(A.sum()/2)
            load[i, k] = subsets.degree_load_flat(A, min_hops)
            load[i, k] /= edges[i, k]
            G = nx.from_numpy_matrix(A)
            diam[i, k] = nx.diameter(G)
            max_min_hops[i, k] = min_hops.max()
            # print(k, max_min_hops[i, k], diam[i, k])
            # print(k)
            print(i, k)
            k += 1
        except ValueError as e:
            # print(e)
            pass
        except StopIteration as e:
            print(e)
            pass

# print(min_hops.max())
# print(max_min_hops)
# print(diam)
print(load)


fig, ax = plt.subplots(1, 2, figsize=(4, 1.5))
ax = ax.ravel()
fig.subplots_adjust(wspace=0.33, bottom=0.2)
for _ax in ax:
    _ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize='xx-small')
    _ax.grid(1, lw=0.4)

# porcentajes
hops = np.arange(1, 10).astype(int)
freq = np.zeros((R, len(hops)))
# mean_edges = np.zeros((R, len(hops)))
mean_load = np.zeros((R, len(hops)))
lower_load = np.zeros((R, len(hops)))
upper_load = np.zeros((R, len(hops)))
# K = n*(n - 1)/2
mark = ['s', '^', 'o']
col = ['C2', 'C1', 'C0']
for i in range(R):
    for k, hop in enumerate(hops):
        hop_rigid = max_min_hops[i] == hop
        freq[i, k] = hop_rigid.mean() * 100
        # mean_edges[i, k] = edges[i, hop_rigid].mean()
        mean_load[i, k] = load[i, hop_rigid].mean()
        # sd = np.sqrt(load[i, hop_rigid].var())
        # lower_load[i, k] = 2*sd
        # upper_load[i, k] = 2*sd
    ax[0].plot(
        hops, freq[i],
        color=col[i], marker=mark[i], markersize=3, lw=0.7,
        label=r'$\Omega = {}$'.format(dmax[i]))
    # ax[0].plot(
    #     hops, freq[i].cumsum(),
    #     color=col[i], marker=mark[i], markersize=3, lw=0.7,
    #     label=r'$\Omega = {}$'.format(dmax[i]))
    # ax[2].plot(
    #     hops, mean_edges[i]/K,
    #     color=col[i], marker=mark[i], markersize=3, lw=0.7)
    ax[1].semilogy(
        hops, mean_load[i],
        color=col[i], marker=mark[i], markersize=3, lw=0.7)
    # ax[1].errorbar(
    #     hops, mean_load[i], xerr=None,
    #     yerr=np.vstack([lower_load[i], upper_load[i]]), capsize=3,
    #     color=col[i], marker=mark[i], markersize=3, lw=0.7)

# print(freq)
# print(freq.cumsum(axis=1))
ax[0].set_xticks(hops)
ax[0].set_xticklabels(hops)
# ax[0].set_ylim(bottom=0)
ax[0].set_xlabel(r'Maximum extent ($\eta$)', fontsize='x-small', labelpad=0.6)
ax[0].set_ylabel(r'$\%$ of networks', fontsize='x-small', labelpad=1)
ax[0].legend(
    fontsize='xx-small', handlelength=2, labelspacing=0.5, borderpad=0.2)

# ax[1].set_xticks(hops)
# ax[1].set_xticklabels(hops)
# ax[1].set_ylim(bottom=0)
# ax[1].set_xlabel(
#     r'Maximum extent ($\eta$)', fontsize='x-small', labelpad=0.6)
# ax[1].set_ylabel(r'Cumulative $\%$', fontsize='x-small', labelpad=1)
# ax[1].legend(
#     fontsize='xx-small', handlelength=1, labelspacing=0.3, borderpad=0.2)

# ax[2].set_xticks(hops)
# ax[2].set_xticklabels(hops)
# ax[2].set_ylim(bottom=0)
# ax[2].set_xlabel(
#     r'Maximum extent ($\eta$)', fontsize='x-small', labelpad=0.6)
# ax[2].set_ylabel(r'Average density', fontsize='x-small', labelpad=1)

ax[1].set_xticks(hops)
ax[1].set_xticklabels(hops)
ax[1].set_yticks([1, 2, 4, 10])
ax[1].set_yticklabels([1, 2, 4, 10])
# ax[1].set_ylim(bottom=1)
ax[1].set_xlabel(r'Maximum extent ($\eta$)', fontsize='x-small', labelpad=0.6)
ax[1].set_ylabel(
    r'Normalized load ($\ell / m$)', fontsize='x-small', labelpad=1)

fig.savefig('/tmp/multihops.pdf', format='pdf')


# hops por nodo
# fig, ax = plt.subplots(figsize=(1.5, 1.5))
# fig.subplots_adjust(bottom=0.225, left=0.23)
# ax.tick_params(
#     axis='both',       # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     pad=1,
#     labelsize='xx-small')
# ax.grid(1, lw=0.4)

# hops = np.arange(1, 10).astype(int)
# freq = np.zeros((R, len(hops)))
# mark = ['s', '^', 'o']
# for i in range(R):
#     for k, h in enumerate(hops):
#         freq[i, k] = (min_hops[i] == h).mean() * 100
#     ax.plot(
#         hops, freq[i],
#         marker=mark[i], markersize=3, lw=0.7,
#         label=r'$\Omega = {}$'.format(dmax[i]))
# print(freq.cumsum(axis=1))
# ax.set_xticks(hops[::2])
# ax.set_xticklabels(hops[::2])
# ax.set_xlabel(r'Extent ($\eta_i$)', fontsize='x-small')
# ax.set_ylabel(r'$\%$ of nodes', fontsize='x-small', labelpad=1)
# ax.legend(
#     fontsize='xx-small', handlelength=1,
#     labelspacing=0.3, borderpad=0.2)

# fig.savefig('/tmp/multihops_nodes.pdf', format='pdf')


# hist de hops / diam
fig, ax = plt.subplots(figsize=(2, 1.5))
fig.subplots_adjust(bottom=0.25, left=0.2)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    labelsize='xx-small')

ax.grid(1, lw=0.4)
ax.set_xlim(0, 1)
ax.set_xticks(np.arange(0, 1.1, 0.1))
ax.set_xticklabels(np.round(np.arange(0, 1.1, 0.1), 1))
ax.set_yticks(np.arange(0, 9, 2))
ax.set_yticklabels(np.arange(0, 90, 20))

mark = ['s', '^', 'o']
col = ['C2', 'C1', 'C0']
for i in range(R):
    freq, hops = np.histogram(
        max_min_hops[i]/diam[i], bins=10, range=(0, 1), density=True)
    bars = np.cumsum(np.diff(hops)) - 0.05
    ax.plot(
        bars, freq,
        marker=mark[i], color=col[i], markersize=3, lw=0.7,
        label=r'$\Omega= {}$'.format(dmax[i]))

    # ax.hist(
    #     max_min_hops[i]/diam[i], color=col[i],
    #     bins=10, range=(0, 1), histtype='bar', density=True, stacked=True)

ax.set_xlabel(r'Max. extent ($\eta$) / diameter (D)', fontsize='x-small')
ax.set_ylabel(r'$\%$ of networks', fontsize='x-small')
ax.legend(
    fontsize='xx-small', handlelength=1, labelspacing=0.3, borderpad=0.2)

fig.savefig('/tmp/multihops_diam.pdf', format='pdf')


plt.show()
