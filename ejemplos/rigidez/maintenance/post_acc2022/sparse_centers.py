#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from uvnpy.network.subsets import degree_load_std, multihop_subframework
from uvnpy.network import plot
from uvnpy.rsn.rigidity import (
    extents, subframework_based_rigidity, sparse_centers)
from uvnpy.rsn.distances import matrix_between as distance_between
from uvnpy.rsn.rigidity import minimum_radius
from uvnpy.network.disk_graph import adjacency as disk_adjacency


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


metrics = [
    (degree_load_std, r'$\mathcal{L}(h)$'),
    (lambda A, h: np.max(h), r'$\max_i \; h_i$')
]


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

# n, d = 10, 2
# found_rigid = False
# while not found_rigid:
#     p = np.random.uniform(-1, 1, (n, d))
#     A = disk_adjacency(p, 0.8)
#     found_rigid = rigidity.algebraic_condition(A, p)

# print(A, p)

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


for k in range(4):
    print(k)
    threshold = 1e-4
    n = np.random.randint(10, 11)
    p = generate_position(n, (0, 1), (0, 1), 0.02)
    A0 = disk_adjacency(p, dmax=2/np.sqrt(n))
    A = minimum_radius(A0, p, threshold)

    for m, (metric, name) in enumerate(metrics):
        hops = extents(A, p, threshold)
        # print('----', name, '----')
        # print(hops, metric(A, hops))

        hops = sparse_centers(A, p, hops, metric, threshold)

        # print(hops, metric(A, hops))
        centers = np.where(hops > 0)[0]

        try:
            is_rigid, B = subframework_based_rigidity(
                A, p, hops, threshold)
        except ValueError as e:
            print(e)
            print(A, p, hops)

        fig, ax = plt.subplots()
        fig.suptitle(name)
        ax.grid(1)
        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plot.edges(ax, p, A, color='k', lw=0.4)

        for i in range(len(p)):
            ax.text(
                p[i, 0] + 0.01, p[i, 1] + 0.01, '{}'.format(i), size=5)
        for i, c in enumerate(centers):
            Ac, pc = multihop_subframework(A, p, c, hops[c])
            plot.graph(ax, pc, Ac, color='C{}'.format(i))
            ax.scatter(
                p[c, 0], p[c, 1],
                color='C{}'.format(i), s=120, label=r'$h={}$'.format(hops[c]))
        ax.legend()

        if len(centers) > 1:
            binding = np.nonzero(np.sum(B, axis=1))[0]
            plot.nodes(ax, p[binding], color='k', marker='s', s=60)
            plot.edges(
                ax, p[binding], B[binding][:, binding],
                color='k', ls='--', lw=0.5)

        fig.savefig(
            '/tmp/sparse_centers{}{}.png'.format(k, m), format='png', dpi=360)

# plt.show()
