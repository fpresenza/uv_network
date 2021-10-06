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
rigidity_extent = np.zeros((R, N), dtype=np.ndarray)
edges = np.zeros((R, N), dtype=int)
load = np.zeros((R, N))
max_rigidity_extent = np.zeros((R, N))
diam = np.zeros((R, N), dtype=int)

for i in range(R):
    k = 0
    while k < N:
        x = np.random.uniform(-hL, hL, (n, 2))
        A = disk_graph.adjacency(x, dmax[i])
        try:
            rigidity_extent[i, k] = rigidity.minimum_hops(A, x)
            edges[i, k] = int(A.sum()/2)
            load[i, k] = subsets.degree_load_std(A, rigidity_extent[i, k])
            load[i, k] /= edges[i, k]
            G = nx.from_numpy_matrix(A)
            diam[i, k] = nx.diameter(G)
            max_rigidity_extent[i, k] = rigidity_extent[i, k].max()
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

# porcentajes
figs = [None, None, None]
axes = [None, None, None]
figs[0], axes[0] = plt.subplots(figsize=(2, 1.5))
figs[1], axes[1] = plt.subplots(figsize=(2, 1.5))
figs[2], axes[2] = plt.subplots(figsize=(2, 1.5))
figs[0].subplots_adjust(left=0.18, bottom=0.2, right=0.9)
figs[1].subplots_adjust(left=0.225, bottom=0.2, right=0.8)
figs[2].subplots_adjust(left=0.225, bottom=0.2, right=0.8)

for ax in axes:
    ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize='xx-small')
    ax.grid(1, lw=0.4)
mark = ['s', '^', 'o']
col = ['C2', 'C1', 'C0']
width = 0.27
hops = np.arange(1, 10).astype(int)
freq = np.zeros((R, len(hops)))
mean_load = np.zeros((R, len(hops)))
for i in range(R):
    for k, hop in enumerate(hops):
        hop_rigid = max_rigidity_extent[i] == hop
        freq[i, k] = hop_rigid.mean() * 100
        mean_load[i, k] = load[i, max_rigidity_extent[i] == hop].mean()
    axes[0].bar(
        hops + width*(i-1), freq[i], width,
        color=col[i],
        label=r'$\Omega = {}$'.format(dmax[i]))
    diam_lims = np.arange(diam[i].min(), diam[i].max() + 1)
    diam_count = np.bincount(diam[i])
    axes[1].bar(
        diam_lims + width*(i-1), diam_count[diam_lims] / diam[i].size * 100,
        width, color=col[i])
    # axes[1].xaxis_date()
    axes[2].semilogy(
        hops, mean_load[i],
        color=col[i], marker=mark[i], markersize=3, lw=0.7,
        label=r'$\Omega = {}$'.format(dmax[i]))

axes[0].set_xticks(hops)
axes[0].set_xticklabels(hops)
axes[0].set_xlabel(
    r'Max. rigidity extent ($\eta$)', fontsize='x-small', labelpad=0.6)
axes[0].set_ylabel(r'Frequency ($\%$)', fontsize='x-small', labelpad=1)
axes[0].legend(
    fontsize='xx-small', handlelength=1.5, labelspacing=0.5, borderpad=0.2)
figs[0].savefig('/tmp/rigidity_extent.pdf', format='pdf')

axes[1].set_xticks(np.arange(diam.min(), diam.max() + 1))
axes[1].set_xticklabels(np.arange(diam.min(), diam.max() + 1))
axes[1].set_xlabel(r'Diameter (D)', fontsize='x-small', labelpad=0.6)
axes[1].set_ylabel(r'Frequency ($\%$)', fontsize='x-small', labelpad=1)
figs[1].savefig('/tmp/diameter.pdf', format='pdf')

axes[2].set_xticks(hops)
axes[2].set_xticklabels(hops)
axes[2].set_yticks([1, 2, 4, 10, 20])
axes[2].set_yticklabels([1, 2, 4, 10, 20])
axes[2].set_xlabel(
    r'Max. rigidity extent ($\eta$)', fontsize='x-small', labelpad=0.6)
axes[2].set_ylabel(
    r'Load ($\ell / m$)', fontsize='x-small', labelpad=1)
axes[2].legend(
    fontsize='xx-small', handlelength=2, labelspacing=0.5, borderpad=0.2)

figs[2].savefig('/tmp/load.pdf', format='pdf')

plt.show()
