#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Ca2ated on dom ago 15 20:00:37 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import uvnpy.network as network

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'

N = 250
probabilities = np.array([0.2, 0.5, 0.7])

fig, ax = plt.subplots(figsize=(1.8, 1.5))
fig.subplots_adjust(bottom=0.16, left=0.2)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='xx-small')
ax.grid(1, lw=0.4)
ax.set_xlabel('Number of nodes (n)', fontsize='x-small', labelpad=1)
ax.set_ylabel(r'$\sum_i (u_i - \mu)^2$', fontsize='x-small', labelpad=1)

fig1, ax1 = plt.subplots(figsize=(1.8, 1.5))
fig1.subplots_adjust(bottom=0.16, left=0.2)
ax1.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='xx-small')
ax1.grid(1, lw=0.4)
ax1.set_xlabel('Number of nodes (n)', fontsize='x-small', labelpad=1)
ax1.set_ylabel(r'$\delta_{\mathrm{avg}}$', fontsize='x-small', labelpad=1)
# ax1.set_ylabel(
#   r'$\sum_k\sigma_k^2 / \delta_{\mathrm{avg}}$',
#   fontsize='x-small', labelpad=1)

markers = ['o', 'D', '^']

for k, prob in enumerate(probabilities):
    nodes = np.array([])
    edges = np.array([])
    diam = np.array([])
    vTLv = np.array([])
    vTv = np.array([])
    alpha = np.array([])

    for i in range(N):
        n = np.random.randint(2, 100)
        ij = np.triu((1 - np.eye(n))).astype(bool)
        a = np.random.choice([0, 1], int(n*(n-1)/2), p=[1-prob, prob])
        A = np.zeros((n, n))
        A[ij] = a
        A = A + A.T

        L = network.laplacian_from_adjacency(A)
        lambda2 = np.linalg.eigvalsh(L)[1]
        if lambda2 > 1e-6:
            G = nx.from_numpy_matrix(A)
            D = nx.diameter(G)
            print(D)
            m = int(A.sum()/2)

            e = np.array(list(nx.eccentricity(G).values()))
            peripheral = np.argwhere(e == D).ravel()

            shortest_paths = [
                nx.single_source_shortest_path(G, per) for per in peripheral]
            vertices = [list(sp.keys()) for sp in shortest_paths]
            paths = [list(sp.values()) for sp in shortest_paths]
            hops = np.array([
                [len(path) - 1 for path in pivot] for pivot in paths])

            u = np.zeros((n, len(peripheral)))
            u[vertices, np.arange(len(peripheral)).reshape(-1, 1)] = hops/D
            v = u - u.mean(0)

            vTLv = np.append(vTLv, max((v * L.dot(v)).sum(0)))
            vTv = np.append(vTv, min((v * v).sum(0)))
            nodes = np.append(nodes, n)
            edges = np.append(edges, m)
            diam = np.append(diam, D)

    print('Number of graphs = {}'.format(len(vTv)))

    ax.scatter(
        nodes, vTv,
        marker=markers[k], s=2,
        label=r'$p={:.1f}$'.format(prob))
    # ax1.scatter(nodes, 2*edges/(nodes*(nodes-1)),
    #     marker=markers[k], s=3,
    #     label=r'$p={:.1f}$'.format(prob))
    # alpha = vTv / nodes
    # density = 2*edges/(nodes*(nodes-1))
    degree = 2*edges / nodes
    ax1.scatter(
        nodes, degree,
        marker=markers[k], s=2,
        label=r'$p={:.1f}$'.format(prob))
    ax1.plot(nodes, prob*(nodes-1), ls='--')
    # ax1.scatter(nodes, edges/(diam**2), color='k', ls='--', lw=0.7)

ax.hlines(1/2, 1, 100, color='k', ls='--', lw=0.7)
ax.legend(
    fontsize='xx-small', handlelength=1,
    labelspacing=0.3, borderpad=0.2)
ax.set_xlim(left=1)
ax.set_ylim(bottom=0)


ax1.legend(
    fontsize='xx-small', handlelength=1,
    labelspacing=0.3, borderpad=0.2)
# ax1.set_ylim(0, 8)
fig1.savefig('/tmp/numerator.pdf', format='pdf')
fig.savefig('/tmp/denominator.pdf', format='pdf')

plt.show()
