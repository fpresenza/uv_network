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
    distance_matrix,
    minimum_rigidity_radius
)

# ------------------------------------------------------------------
# Definición de variables, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple(
    'Logs', 'nodes max_dist dist eccen')

# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(d, side_length, nmin, nmax, logs, threshold, rep):
    bar.start()

    for k, n in enumerate(range(nmin, nmax + 1)):
        bar.update(k)
        logs.nodes[k] = n

        for r in range(rep):
            p = np.random.uniform(0, side_length, (n, d))
            A0 = adjacency_from_positions(p, dmax=2/np.sqrt(n))
            dist = distance_matrix(p)
            Rmax = dist.max()
            A, Rmin = minimum_rigidity_radius(
                A0, p, threshold, return_radius=True
            )
            alpha = 0.1
            R = Rmin + alpha * (Rmax - Rmin)
            A = adjacency_from_positions(p, dmax=R)
            G = geodesics(A)

            logs.max_dist[k, r] = R
            logs.dist[k, r*nmax:r*nmax + n] = np.max(A * dist, axis=0)
            logs.eccen[k, r*nmax:r*nmax + n] = np.max(G, axis=0)

    bar.finish()
    return logs


# ------------------------------------------------------------------
# Parseo de argumentos
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-n', '--nodes',
    default=10, type=int, help='number of nodes')
parser.add_argument(
    '-r', '--rep',
    default=1, type=int, help='number of repetitions')

arg = parser.parse_args()

# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------
d = 2
side_length = 100.0
nmin = d + 2
nmax = arg.nodes
threshold = 1e-5
size = nmax - nmin + 1
rep = arg.rep
logs = Logs(
    nodes=np.empty(size, dtype=int),
    max_dist=np.empty((size, rep), dtype=float),
    dist=np.zeros((size, nmax * rep), dtype=float),
    eccen=np.zeros((size, nmax * rep), dtype=int)
)
# ------------------------------------------------------------------
# Simulación
# ------------------------------------------------------------------
bar = progressbar.ProgressBar(maxval=size)

logs = run(d, side_length, nmin, nmax, logs, threshold, rep)

np.savetxt('/tmp/nodes.csv', logs.nodes, delimiter=',', fmt='%d')
np.savetxt('/tmp/max_dist.csv', logs.max_dist, delimiter=',', fmt='%.5f')
np.savetxt('/tmp/dist.csv', logs.dist, delimiter=',', fmt='%.5f')
np.savetxt('/tmp/eccen.csv', logs.eccen, delimiter=',', fmt='%d')
