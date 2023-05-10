#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on vie sep 10 10:38:33 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx  # noqa

import uvnpy.network as network  # noqa
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
rigidity_extent = np.zeros((R, N), dtype=np.ndarray)
edges = np.zeros((R, N), dtype=int)
load = np.zeros((R, N))
maxload = np.zeros((R, N))
max_rigidity_extent = np.zeros((R, N))
diam = np.zeros((R, N), dtype=int)

for i in range(R):
    k = 0
    while k < N:
        x = np.random.uniform(-hL, hL, (n, 2))
        A = disk_graph.adjacency(x, dmax[i])
        try:
            rigidity_extent[i, k] = rigidity.extents(A, x)
            max_rigidity_extent[i, k] = rigidity_extent[i, k].max()
            edges[i, k] = int(A.sum()/2)
            load[i, k] = subsets.degree_load_std(A, rigidity_extent[i, k])
            load[i, k] /= 2*edges[i, k]

            G = nx.from_numpy_array(A)
            diam[i, k] = nx.diameter(G)
            eccen = np.array(list(nx.eccentricity(G).values()))
            # central = np.argmin(ecc)
            # geo = network.geodesics(A)
            # coeff = (ecc[central] - geo[central]).clip(min=0)
            # maxload[i, k] = coeff.dot(A.sum(1))
            maxload[i, k] = subsets.degree_load_std(A, eccen)
            maxload[i, k] /= 2*edges[i, k]

            # print(k, max_rigidity_extent[i, k], diam[i, k])
            # print(k)
            print(i, k)
            k += 1
        except ValueError as e:
            # print(e)
            pass
        except StopIteration as e:
            print(e)
            pass

# print(rigidity_extent.max())
# print(max_rigidity_extent)
# print(diam)
print(diam)
# print(load)
# print(maxload)

# porcentajes
fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.25))
fig.subplots_adjust(bottom=0.3, top=0.925, wspace=0.33, left=0.12, right=0.975)
for ax in axes:
    ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize='xx-small')
    ax.grid(1, lw=0.4)

mark = ['s', '^', 'o']
col = ['C2', 'C1', 'C0']
hops = np.arange(1, 10).astype(int)
freq = np.zeros((R, len(hops)))
mean_load = np.zeros((R, len(hops)))
mean_maxload = np.zeros((R, len(hops)))
for i in range(R):
    for k, hop in enumerate(hops):
        hop_rigid = max_rigidity_extent[i] == hop
        freq[i, k] = hop_rigid.mean() * 100
        mean_load[i, k] = load[i, hop_rigid].mean()
        mean_maxload[i, k] = maxload[i, hop_rigid].mean()
    axes[0].plot(
        hops, freq[i],
        color=col[i], marker=mark[i], markersize=3, lw=0.7,
        label=r'$\Omega = {}$'.format(dmax[i]))
    # diam_lims = np.arange(diam[i].min(), diam[i].max() + 1)
    # diam_count = np.bincount(diam[i])
    # axes[1].plot(
    #     diam_lims, diam_count[diam_lims] / diam[i].size * 100,
    #     color=col[i], marker=mark[i], markersize=3, lw=0.7)
    # axes[1].xaxis_date()
    axes[1].semilogy(
        hops, mean_load[i],
        color=col[i], marker=mark[i], markersize=3, lw=0.7,
        label=r'$\Omega = {}$'.format(dmax[i]))
    axes[1].semilogy(
        hops, mean_maxload[i],
        color=col[i], lw=0.7, ls='--',
        label=r'$\Omega = {}$'.format(dmax[i]))

print(freq.sum(axis=0)[:5].sum()/3)

axes[0].set_xticks(hops)
axes[0].set_xticklabels(hops)
axes[0].set_xlabel(
    'Worst-case \n rigidity extent, ' + r'$\eta$',
    fontsize='x-small', labelpad=0.6)
axes[0].set_ylabel(r'Frequency ($\%$)', fontsize='x-small', labelpad=1)
axes[0].legend(
    fontsize='xx-small', handlelength=1.5, labelspacing=0.5, borderpad=0.2)

axes[1].set_xticks(hops)
axes[1].set_xticklabels(hops)
# axes[2].set_yticks([1, 4, 10, 20, 30])
# axes[2].set_yticklabels([1, 4, 10, 20, 30])
axes[1].set_xlabel(
    'Worst-case \n rigidity extent, ' + r'$\eta$',
    fontsize='x-small', labelpad=0.6)
axes[1].set_ylabel(
    r'Avg. Std. Load', fontsize='x-small', labelpad=1)
# axes[2].legend(
#     fontsize='xx-small', handlelength=2, labelspacing=0.5, borderpad=0.2)

fig.savefig('/tmp/extents.png', format='png', dpi=300)

plt.show()
