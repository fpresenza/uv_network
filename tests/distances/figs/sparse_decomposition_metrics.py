#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import argparse
import collections
import numpy as np
from numba import njit
import progressbar

from uvnpy.network.core import geodesics
from uvnpy.network.graphs import DiskGraph
from uvnpy.distances.core import (
    is_inf_rigid,
    minimum_rigidity_radius,
)
from uvnpy.network.subframeworks import (
    valid_extents,
    sparse_subframeworks_greedy_search_by_expansion,
)

# ------------------------------------------------------------------
# Definición de variables, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple(
    'Logs',
    'nodes \
    cost_sparse \
    cost_sparse_dece'
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


@njit
def decomposition_cost(extents, geodesics):
    """
    Computes the set of isolated links (edges not in any subframework).
    """
    n = len(extents)
    s = 0

    for i in range(n):
        for j in range(i + 1, n):
            if geodesics[i, j] == 1:
                in_ball = (geodesics[i] <= extents) * (geodesics[j] <= extents)
                # 1.0 if s > 2 else 5.0
                c = sum(in_ball)
                s += float(c) if c != 0 else 5.0
    return s


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------
def run(d, nmin, nmax, logs, rep):
    bar.start()

    for k, n in enumerate(range(nmin, nmax)):
        bar.update(k)
        logs.nodes[k] = n

        for r in range(rep):
            p = np.random.uniform((0, 0), (1, 0.9), (n, 2))
            A = DiskGraph(p, dmax=2/np.sqrt(n)).adjacency_matrix(float)
            A, Rmin = minimum_rigidity_radius(A, p, return_radius=True)

            G = geodesics(A)
            h_valid = valid_extents(G, valid_ball, args=(A, p, max_diam))
            h_sparse = sparse_subframeworks_greedy_search_by_expansion(
                valid_extents=h_valid,
                metric=decomposition_cost,
                geodesics=G,
            )
            h_sparse_dece = np.empty(n, dtype=int)
            for i in range(n):
                S = G[i] <= max_diam
                Ai = A[:, S][S]
                pi = p[S]
                Gi = geodesics(Ai)
                h_valid_i = valid_extents(
                    Gi, valid_ball, args=(Ai, pi, max_diam)
                )
                h_sparse_i = sparse_subframeworks_greedy_search_by_expansion(
                    valid_extents=h_valid_i,
                    metric=decomposition_cost,
                    geodesics=Gi,
                )
                idx = sum(S[:i])
                h_sparse_dece[i] = h_sparse_i[idx]

            logs.cost_sparse[k, r] = decomposition_cost(h_sparse, G)
            logs.cost_sparse_dece[k, r] = decomposition_cost(h_sparse_dece, G)

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
max_diam = 5
rep = arg.rep

logs = Logs(
    nodes=np.empty(size, dtype=int),
    cost_sparse=np.empty((size, rep), dtype=float),
    cost_sparse_dece=np.empty((size, rep), dtype=float)
)

# ------------------------------------------------------------------
# Simulación
# ------------------------------------------------------------------
bar = progressbar.ProgressBar(maxval=size)

logs = run(d, nmin, nmax, logs, rep)

np.savetxt('/tmp/nodes.csv', logs.nodes, delimiter=',')
np.savetxt(
    '/tmp/cost_sparse_delta_{}.csv'.format(max_diam),
    logs.cost_sparse,
    delimiter=','
)
np.savetxt(
    '/tmp/cost_sparse_dece_delta_{}.csv'.format(max_diam),
    logs.cost_sparse_dece,
    delimiter=','
)
