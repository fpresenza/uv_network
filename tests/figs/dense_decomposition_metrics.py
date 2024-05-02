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
from uvnpy.network.subframeworks import superframework_extents
from uvnpy.network.load import one_token_for_all, one_token_for_each
from uvnpy.distances.core import (
    distance_matrix,
    minimum_rigidity_extents,
    minimum_rigidity_radius
)

# ------------------------------------------------------------------
# Definición de variables, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple(
    'Logs', 'nodes diam hmax load edges rmin rmax alpha')


def minimum_alpha(A, p, dist, Rmin, Rmax, threshold=1e-5):
    """Calcula el alpha necesario para que todos los subframeworks
    tengan alcance unitario"""
    A = A.copy()
    h = minimum_rigidity_extents(A, p, threshold)
    if np.all(h == 1):
        return 0.
    else:
        radius = np.unique(dist[dist > Rmin])
        for R in radius:
            A[dist == R] = 1
            h = minimum_rigidity_extents(A, p, threshold)
            if np.all(h == 1):
                return (R - Rmin) / (Rmax - Rmin)


def network_load(geodesics, extents):
    action_load = one_token_for_each(geodesics, extents)
    super_extents = superframework_extents(geodesics, extents)
    state_load = one_token_for_all(geodesics, super_extents)
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
            A0 = adjacency_from_positions(p, dmax=2/np.sqrt(n))
            dist = distance_matrix(p)
            Rmax = dist.max()
            A, Rmin = minimum_rigidity_radius(
                A0, p, threshold, return_radius=True
            )
            Alpha = minimum_alpha(A, p, dist, Rmin, Rmax, threshold)

            logs.rmax[k, r] = Rmax
            logs.rmin[k, r] = Rmin
            logs.alpha[k, r] = Alpha

            h = minimum_rigidity_extents(A, p, threshold)
            G = geodesics(A)
            logs.diam[0, k, r] = np.max(G)
            logs.hmax[0, k, r] = np.max(h)
            logs.load[0, k, r] = network_load(G, h)
            logs.edges[0, k, r] = np.sum(A) / 2

            A = adjacency_from_positions(p, dmax=Rmin + 0.05 * (Rmax - Rmin))
            h = minimum_rigidity_extents(A, p, threshold)
            G = geodesics(A)
            logs.diam[1, k, r] = np.max(G)
            logs.hmax[1, k, r] = np.max(h)
            logs.load[1, k, r] = network_load(G, h)
            logs.edges[1, k, r] = np.sum(A) / 2

            A = adjacency_from_positions(p, dmax=Rmin + 0.1 * (Rmax - Rmin))
            h = minimum_rigidity_extents(A, p, threshold)
            G = geodesics(A)
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
nodes = logs.nodes
diam = logs.diam
hmax = logs.hmax
load = logs.load
edges = logs.edges
rmin = logs.rmin
rmax = logs.rmax
alpha = logs.alpha

np.savetxt('/tmp/nodes.csv', logs.nodes, delimiter=',')
np.savetxt('/tmp/rmin.csv', logs.rmin, delimiter=',')
np.savetxt('/tmp/rmax.csv', logs.rmax, delimiter=',')
np.savetxt('/tmp/alpha.csv', logs.alpha, delimiter=',')

for i in range(3):
    np.savetxt('/tmp/diam_{}.csv'.format(i), logs.diam[i], delimiter=',')
    np.savetxt('/tmp/hmax_{}.csv'.format(i), logs.hmax[i], delimiter=',')
    np.savetxt('/tmp/load_{}.csv'.format(i), logs.load[i], delimiter=',')
    np.savetxt('/tmp/edges_{}.csv'.format(i), logs.edges[i], delimiter=',')
