#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from uvnpy.network.subsets import (
    degree_load_std,
    multihop_subframework,
    subgraph_union)
from uvnpy.network import plot
from uvnpy.rsn.rigidity import (
    extents,
    minimum_radius,
    rigidly_linked,
    rigidly_linked_by_vertices,
    sparse_centers,
    sparse_centers_two_steps)
from uvnpy.rsn.distances import matrix_between as distance_between
from uvnpy.network.disk_graph import adjacency as disk_adjacency

rigidly_linked


def generate_position(n, xlim, ylim, radius):
    """Genera una posicion donde los nodos estan a mas
    de cierto radio de separacion"""
    xl, xh = xlim
    yl, yh = ylim
    p = np.random.uniform((xl, yl), (xh, yh), 2)
    for i in range(1, n):
        found = False
        while not found:
            q = np.random.uniform((xl, yl), (xh, yh), 2)
            dist = distance_between(p, q)
            if np.all(dist > radius):
                found = True
                p = np.vstack([p, q])

    return p


def isolated_edges(A, h):
    """Cantidad de enlaces que no pertenecen a ningún subframework"""
    As = subgraph_union(A, h)
    return np.sum(A - As) / len(A)


def multiobjetive(A, h, alpha=5):
    return degree_load_std(A, h) + alpha * isolated_edges(A, h)


metrics = (
    (degree_load_std, r'$\mathcal{L}(h)$'),
    (multiobjetive, r'$\mathcal{L}(h) + \alpha 2 |\mathcal{E}_A| / n$'),
    (degree_load_std, r'$2$ step$')
)

# p = np.array([
#     [-0.8967,  0.4359],
#     [-0.1932, -0.2075],
#     [ 0.6964,  0.9842],
#     [-0.9011, -0.4148],
#     [ 0.1151, -0.0142],
#     [ 0.0797, -0.688 ],
#     [ 0.562 ,  0.0812],
#     [ 0.4342, -0.215 ],
#     [-0.273 , -0.7416],
#     [ 0.3137,  0.6364]])

# A = np.array([
#     [0., 0., 0., 1., 1., 0., 0., 0., 0., 0.],
#     [0., 0., 0., 1., 1., 1., 1., 1., 1., 0.],
#     [0., 0., 0., 0., 0., 0., 1., 0., 0., 1.],
#     [1., 1., 0., 0., 0., 0., 0., 0., 1., 0.],
#     [1., 1., 0., 0., 0., 1., 1., 1., 1., 1.],
#     [0., 1., 0., 0., 1., 0., 0., 1., 1., 0.],
#     [0., 1., 1., 0., 1., 0., 0., 1., 0., 1.],
#     [0., 1., 0., 0., 1., 1., 1., 0., 0., 0.],
#     [0., 1., 0., 1., 1., 1., 0., 0., 0., 0.],
#     [0., 0., 1., 0., 1., 0., 1., 0., 0., 0.]])

# A = np.array([
#     [0., 1., 1., 0., 0., 0., 0., 0., 1., 1.],
#     [1., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
#     [1., 1., 0., 1., 0., 1., 0., 0., 0., 0.],
#     [0., 0., 1., 0., 1., 1., 0., 0., 0., 0.],
#     [0., 0., 0., 1., 0., 1., 1., 0., 0., 0.],
#     [0., 0., 1., 1., 1., 0., 1., 1., 1., 0.],
#     [0., 0., 0., 0., 1., 1., 0., 1., 0., 0.],
#     [0., 0., 0., 0., 0., 1., 1., 0., 1., 0.],
#     [1., 0., 0., 0., 0., 1., 0., 1., 0., 1.],
#     [1., 0., 0., 0., 0., 0., 0., 0., 1., 0.]])
# A[2, 5] = A[5, 2] = 0
# A[7, 9] = A[9, 7] = 1

# p = np.array([
#     [ 2.921961,  0.962442],
#     [ 3.479602,  1.951268],
#     [ 2.496763,  2.044489],
#     [-0.05499 ,  1.955132],
#     [-1.120506,  0.903233],
#     [ 0.856702,  0.878715],
#     [-0.533504, -0.472975],
#     [ 0.975142, -0.536853],
#     [ 1.392686,  0.203473],
#     [ 2.188298,  0.004104]])

# A = np.array([
#     [0., 1., 1., 1., 1., 1., 1.],
#     [1., 0., 1., 1., 0., 0., 0.],
#     [1., 1., 0., 1., 0., 0., 0.],
#     [1., 1., 1., 0., 0., 0., 1.],
#     [1., 0., 0., 0., 0., 1., 1.],
#     [1., 0., 0., 0., 1., 0., 1.],
#     [1., 0., 0., 1., 1., 1., 0.]])

# p = np.array([
#     [0.5, 0.],
#     [1., 0.],
#     [0.75, 0.25],
#     [0.75, 0.5],
#     [0., 0.],
#     [0.25, 0.25],
#     [0.25, 0.5]])

# A = np.array([
#     [0., 1., 0., 0., 0., 1., 0., 1., 0., 1.],
#     [1., 0., 0., 0., 0., 1., 0., 1., 0., 0.],
#     [0., 0., 0., 1., 1., 0., 0., 0., 1., 0.],
#     [0., 0., 1., 0., 0., 0., 1., 1., 1., 0.],
#     [0., 0., 1., 0., 0., 0., 0., 0., 1., 1.],
#     [1., 1., 0., 0., 0., 0., 0., 1., 0., 1.],
#     [0., 0., 0., 1., 0., 0., 0., 1., 0., 0.],
#     [1., 1., 0., 1., 0., 1., 1., 0., 0., 1.],
#     [0., 0., 1., 1., 1., 0., 0., 0., 0., 1.],
#     [1., 0., 0., 0., 1., 1., 0., 1., 1., 0.]])

# p = np.array([
#     [0.42512982, 0.10862431],
#     [0.08390347, 0.13320194],
#     [0.96102407, 0.71678363],
#     [0.42391391, 0.73690224],
#     [0.94214822, 0.40892907],
#     [0.41717728, 0.07954662],
#     [0.30690858, 0.66746692],
#     [0.31661759, 0.20757877],
#     [0.90928967, 0.63609389],
#     [0.73370665, 0.12652372]])


np.random.seed(3)
for k in range(10):
    print('---{}---'.format(k))
    threshold = 1e-4
    n = np.random.randint(10, 11)
    p = generate_position(n, (0, 1), (0, 1), 0.15)
    A0 = disk_adjacency(p, dmax=2/np.sqrt(n))
    A = minimum_radius(A0, p, threshold)

    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    for a in ax:
        a.grid(1)
        a.set_aspect('equal')
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        plot.graph(a, p, A, color='k', lw=0.4)

    for m, (metric, name) in enumerate(metrics):
        print('--{}--'.format(name))
        h0 = extents(A, p, threshold)
        print(h0, metric(A, h0))

        if m < 2:
            hopt = sparse_centers(
                A, p, h0, metric, threshold, vertices_only=True)
        else:
            hopt = sparse_centers_two_steps(
                A, p, h0, metric, threshold, vertices_only=True)
        print(hopt, metric(A, hopt))

        centers = np.where(hopt > 0)[0]

        try:
            Al = rigidly_linked_by_vertices(A, p, hopt, threshold)
        except ValueError as e:
            print(e)
            print(A, p, hopt)

        ax[m].set_title(name)
        for i in range(len(p)):
            ax[m].text(
                p[i, 0] + 0.01, p[i, 1] + 0.01, '{}'.format(i), size=5)
        for i, c in enumerate(centers):
            Ac, pc = multihop_subframework(A, p, c, hopt[c])
            plot.graph(ax[m], pc, Ac, color='C{}'.format(i))
            ax[m].scatter(
                p[c, 0], p[c, 1],
                color='C{}'.format(i), s=120, label=r'$h={}$'.format(hopt[c]))
        ax[m].legend(fontsize=6, markerscale=0.5)

        links = np.any(Al > 0, axis=1)
        plot.nodes(ax[m], p[links], color='k', marker='s', s=60)
        plot.edges(
            ax[m], p[links], Al[links][:, links],
            color='k', ls='--', lw=2, alpha=0.4)

    fig.savefig(
        '/tmp/sparse_centers{}.png'.format(k), format='png', dpi=360)

# plt.show()
