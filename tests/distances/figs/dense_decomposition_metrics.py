#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import argparse
import collections
import numpy as np
import progressbar

from uvnpy.graphs.core import (
    geodesics,
    as_undirected,
    adjacency_matrix_from_edges
)
from uvnpy.graphs.models import DiskGraph
from uvnpy.graphs.subframeworks import superframework_extents
from uvnpy.graphs.load import one_token_for_all_sum, one_token_for_each_sum
from uvnpy.distances.core import (
    distance_matrix,
    minimum_distance_rigidity_extents,
    minimum_distance_rigidity_radius
)

# ------------------------------------------------------------------
# Definición de variables, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple(
    'Logs', 'nodes diam hmax load edges rmin rmax alpha')


def minimum_alpha(E, A, G, h, p, dist, Rmin, Rmax, threshold=1e-5):
    """Calcula el alpha necesario para que todos los subframeworks
    tengan alcance unitario"""
    A = A.copy()
    if np.all(h == 1):
        return 0.0
    else:
        radius = np.unique(dist[dist > Rmin])
        for R in radius:
            A[dist == R] = 1
            G = geodesics(A)
            h = minimum_distance_rigidity_extents(E, G, p, threshold)
            if np.all(h == 1):
                return (R - Rmin) / (Rmax - Rmin)


def network_load(geodesics, extents):
    action_load = one_token_for_each_sum(geodesics, extents)
    super_extents = superframework_extents(geodesics, extents)
    state_load = one_token_for_all_sum(geodesics, super_extents)
    return action_load + state_load


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(d, nmin, nmax, logs, threshold, rep):
    bar.start()

    for k, n in enumerate(range(nmin, nmax)):
        bar.update(k)
        logs.nodes[k] = n

        for r in range(rep):
            p = np.random.uniform(0, 1, (n, d))
            E0 = DiskGraph(p, dmax=2/np.sqrt(n)).edge_set(as_oriented=True)
            dist = distance_matrix(p)
            Rmax = dist.max()
            E, Rmin = minimum_distance_rigidity_radius(
                E0, p, threshold, return_radius=True
            )
            A = as_undirected(adjacency_matrix_from_edges(n, E)).astype(float)

            logs.rmax[k, r] = Rmax
            logs.rmin[k, r] = Rmin

            G = geodesics(A)
            h = minimum_distance_rigidity_extents(E, G, p, threshold)
            logs.diam[0, k, r] = np.max(G)
            logs.hmax[0, k, r] = np.max(h)
            logs.load[0, k, r] = network_load(G, h)
            logs.edges[0, k, r] = np.sum(A) / 2

            Alpha = minimum_alpha(E, A, G, h, p, dist, Rmin, Rmax, threshold)
            logs.alpha[k, r] = Alpha

            graph = DiskGraph(
                p, dmax=Rmin + 0.05 * (Rmax - Rmin)
            )
            A = graph.adjacency_matrix(float)
            E = graph.edge_set(as_oriented=True)
            G = geodesics(A)
            h = minimum_distance_rigidity_extents(E, G, p, threshold)
            logs.diam[1, k, r] = np.max(G)
            logs.hmax[1, k, r] = np.max(h)
            logs.load[1, k, r] = network_load(G, h)
            logs.edges[1, k, r] = np.sum(A) / 2

            graph = DiskGraph(
                p, dmax=Rmin + 0.1 * (Rmax - Rmin)
            )
            A = graph.adjacency_matrix(float)
            E = graph.edge_set(as_oriented=True)
            G = geodesics(A)
            h = minimum_distance_rigidity_extents(E, G, p, threshold)
            logs.diam[2, k, r] = np.max(G)
            logs.hmax[2, k, r] = np.max(h)
            logs.load[2, k, r] = network_load(G, h)
            logs.edges[2, k, r] = np.sum(A) / 2

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

arg = parser.parse_args()

# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------
d = 2
nmin = d + 2
nmax = arg.nodes + 1
threshold = 1e-5
size = nmax - nmin
rep = arg.rep
logs = Logs(
    nodes=np.empty(size, dtype=int),
    rmin=np.empty((size, rep)),
    rmax=np.empty((size, rep)),
    alpha=np.empty((size, rep)),
    diam=np.empty((3, size, rep)),
    hmax=np.empty((3, size, rep)),
    load=np.empty((3, size, rep)),
    edges=np.empty((3, size, rep)))

# ------------------------------------------------------------------
# Simulación
# ------------------------------------------------------------------
bar = progressbar.ProgressBar(maxval=size)

logs = run(d, nmin, nmax, logs, threshold, rep)

np.savetxt('/tmp/nodes.csv', logs.nodes, delimiter=',')
np.savetxt('/tmp/rmin.csv', logs.rmin, delimiter=',')
np.savetxt('/tmp/rmax.csv', logs.rmax, delimiter=',')
np.savetxt('/tmp/alpha.csv', logs.alpha, delimiter=',')

for i in range(3):
    np.savetxt('/tmp/diam_{}.csv'.format(i), logs.diam[i], delimiter=',')
    np.savetxt('/tmp/hmax_{}.csv'.format(i), logs.hmax[i], delimiter=',')
    np.savetxt('/tmp/load_{}.csv'.format(i), logs.load[i], delimiter=',')
    np.savetxt('/tmp/edges_{}.csv'.format(i), logs.edges[i], delimiter=',')
