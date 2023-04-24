#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from uvnpy.network import subsets, disk_graph, plot
from uvnpy.rsn import rigidity

check_rigidity = rigidity.subframework_based_rigidity

# metric = subsets.degree_load_std
# metric = subsets.degree_load_flat
# metric = lambda A, h : subsets.degree_load_std(A, h) + 60*np.max(h)
# metric = lambda A, h : np.sum(h * h)
metrics = [
    subsets.degree_load_std,
    subsets.degree_load_flat,
    lambda A, h: np.max(h)
]


# x = np.array([
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

n, d = 10, 2
found_rigid = False
while not found_rigid:
    x = np.random.uniform(-1, 1, (n, d))
    A = disk_graph.adjacency(x, 0.8)
    found_rigid = rigidity.algebraic_condition(A, x)

print(A, x)

for metric in metrics:
    extents = rigidity.extents(A, x)
    print(extents, metric(A, extents))

    for h in reversed(np.unique(extents)):
        max_extent = np.where(extents == h)[0]
        repeat = True
        while repeat:
            min_load = np.inf
            remove = None
            for i in max_extent:
                sparsed = extents.copy()
                sparsed[i] = 0
                if check_rigidity(A, x, sparsed) is True:
                    new_load = metric(A, sparsed)
                    if new_load < min_load:
                        min_load = new_load
                        remove = i

            if min_load < np.inf:
                extents[remove] = 0
                max_extent = np.delete(max_extent, max_extent == remove)
            else:
                repeat = False

    print(extents, metric(A, extents))
    centers = np.where(extents > 0)[0]

    if len(centers) == 1:
        is_rigid = check_rigidity(A, x, extents, return_binding=True)
    else:
        is_rigid, B = check_rigidity(A, x, extents, return_binding=True)

    if not is_rigid:
        print('algo salio mal.')

    fig, ax = plt.subplots()
    ax.grid(1)
    ax.set_aspect('equal')
    ax.set_xlim(x[:, 0].min() * 1.15, x[:, 0].max() * 1.15)
    ax.set_ylim(x[:, 1].min() * 1.15, x[:, 1].max() * 1.15)

    for i in range(len(x)):
        ax.text(x[i, 0] + 0.02, x[i, 1] + 0.02, '{}'.format(i))
    for i, c in enumerate(centers):
        Ac, xc = subsets.multihop_subframework(A, x, c, extents[c])
        plot.graph(ax, xc, Ac, color='C{}'.format(i))
        ax.scatter(
            x[c, 0], x[c, 1], color='C{}'.format(i), facecolor='none', s=100)

    if len(centers) > 1:
        binding = np.nonzero(np.sum(B, axis=1))[0]
        plot.nodes(ax, x[binding], color='k', marker='s', s=60)
        plot.edges(
            ax, x[binding], B[binding][:, binding], color='k', ls='--', lw=0.5)

plt.show()
