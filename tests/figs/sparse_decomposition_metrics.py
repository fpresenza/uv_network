#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import argparse
import collections
import numpy as np
import progressbar

from uvnpy.network.core import geodesics
from uvnpy.network.disk_graph import adjacency_from_positions
from uvnpy.distances.core import (
    is_inf_rigid,
    minimum_rigidity_extents,
    minimum_rigidity_radius,
)
from uvnpy.network.subframeworks import (
    valid_extents,
    subframework_adjacencies,
    isolated_links,
    sparse_subframeworks_greedy_search_by_expansion,
)
# ------------------------------------------------------------------
# Definición de variables, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple(
    'Logs',
    'nodes \
    cost_dense \
    cost_sparse'
)


def valid_ball(subset, adjacency, position, max_diam):
    """A ball is considered valid if:
        it has zero radius
            or
        (it does not exceeds the maximum allowed diameter
            and
        it is infinitesimally rigid)
    """
    if sum(subset) == 1:
        return True

    A = adjacency[:, subset][subset]
    if geodesics(A).max() > max_diam:
        return False

    p = position[subset]
    if not is_inf_rigid(A, p):
        return False

    return True


def weight(s):
    return 5.0 if s == 2 else 1.0


def decomposition_cost(extents, geodesics, weight):
    adj = subframework_adjacencies(geodesics, extents)
    num_links = len(isolated_links(geodesics, extents))
    ball_sum = sum([weight(len(a)) * np.sum(a) / 2.0 for a in adj])
    link_sum = weight(2) * num_links
    return ball_sum + link_sum


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------
def run(d, nmin, nmax, logs, rep):
    bar.start()

    for k, n in enumerate(range(nmin, nmax)):
        bar.update(k)
        logs.nodes[k] = n
        # print(n)

        for r in range(rep):
            # print(r)
            # p = sufficiently_dispersed_position(n, (0, 1), (0, 0.9), 0.1)
            p = np.random.uniform((0, 0), (1, 0.9), (n, 2))
            A = adjacency_from_positions(p, dmax=2/np.sqrt(n))
            A, Rmin = minimum_rigidity_radius(A, p, return_radius=True)

            G = geodesics(A)
            h_valid = valid_extents(G, valid_ball, A, p, max_diam)
            h_dense = minimum_rigidity_extents(G, p)
            h_sparse = sparse_subframeworks_greedy_search_by_expansion(
                valid_extents=h_valid,
                metric=decomposition_cost,
                geodesics=G,
                weight=weight
            )

            logs.cost_dense[k, r] = decomposition_cost(h_dense, G, weight)
            logs.cost_sparse[k, r] = decomposition_cost(h_sparse, G, weight)

    bar.finish()
    return logs


# ------------------------------------------------------------------
# Parseo de argumentos
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-n', '--nodes',
    default=50, type=int, help='number of nodes')
parser.add_argument(
    '-r', '--rep',
    default=1, type=int, help='number of repetitions')
parser.add_argument(
    '-s', '--save',
    default=True, action='store_true', help='flag to store data')

arg = parser.parse_args()

# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------
d = 2
nmin = d + 2
nmax = arg.nodes + 1
size = nmax - nmin
max_diam = 4
rep = arg.rep

logs = Logs(
    nodes=np.empty(size, dtype=int),
    cost_dense=np.empty((size, rep), dtype=float),
    cost_sparse=np.empty((size, rep), dtype=float),
)

# ------------------------------------------------------------------
# Simulación
# ------------------------------------------------------------------
bar = progressbar.ProgressBar(maxval=size)

logs = run(d, nmin, nmax, logs, rep)

np.savetxt('/tmp/nodes.csv', logs.nodes, delimiter=',')
np.savetxt(
    '/tmp/cost_dense.csv', logs.cost_dense, delimiter=','
)
np.savetxt(
    '/tmp/cost_sparse.csv', logs.cost_sparse, delimiter=','
)
