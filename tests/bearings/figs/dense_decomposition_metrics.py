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
from uvnpy.network.random_graph import erdos_renyi
import uvnpy.distances.core as distances
import uvnpy.bearings.core as bearings

# ------------------------------------------------------------------
# Definición de variables, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple(
    'Logs', 'nodes diam hmax_d hmax_b')


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(d, nmin, nmax, degree, rep, logs):
    bar.start()

    for k, n in enumerate(range(nmin, nmax)):
        bar.update(k)
        logs.nodes[k] = n

        p = np.random.uniform(0, 1, (n, d))
        r = 0
        while r < rep:
            A = erdos_renyi(n, degree / (n - 1))
            if distances.is_inf_rigid(A, p):
                G = geodesics(A)
                hd = distances.minimum_rigidity_extents(G, p)
                hb = bearings.minimum_rigidity_extents(G, p)
                logs.diam[k, r] = np.max(G)
                logs.hmax_d[k, r] = np.max(hd)
                logs.hmax_b[k, r] = np.max(hb)
                r += 1

    bar.finish()
    return logs


# ------------------------------------------------------------------
# Parseo de argumentos
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-n', '--nodes',
    default=50, type=int, help='number of nodes'
)
parser.add_argument(
    '-g', '--degree',
    default=1.0, type=float, help='average vertex degree'
)
parser.add_argument(
    '-r', '--rep',
    default=1, type=int, help='number of repetitions'
)

arg = parser.parse_args()

# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------
d = 3
degree = arg.degree
nmin = int(degree) + 1
nmax = arg.nodes + 1
size = nmax - nmin
rep = arg.rep
logs = Logs(
    nodes=np.empty(size, dtype=int),
    diam=np.empty((size, rep)),
    hmax_d=np.empty((size, rep)),
    hmax_b=np.empty((size, rep))
)

# ------------------------------------------------------------------
# Simulación
# ------------------------------------------------------------------
bar = progressbar.ProgressBar(maxval=size)

logs = run(d, nmin, nmax, degree, rep, logs)

np.savetxt('/tmp/nodes.csv', logs.nodes, delimiter=',')

np.savetxt('/tmp/diam.csv', logs.diam, delimiter=',')
np.savetxt('/tmp/hmax_d.csv', logs.hmax_d, delimiter=',')
np.savetxt('/tmp/hmax_b.csv', logs.hmax_b, delimiter=',')
