#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import argparse
import collections
import numpy as np
import progressbar

from uvnpy.toolkit.data import write_csv
from uvnpy.network.core import geodesics
from uvnpy.network.random_graph import erdos_renyi
import uvnpy.distances.core as distances
import uvnpy.bearings.core as bearings

# ------------------------------------------------------------------
# Definición de variables, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple(
    'Logs',
    'nodes diam diam_count hd hd_count hb hb_count'
)


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(d, nmin, nmax, degree, rep, logs):
    bar.start()

    for k, n in enumerate(range(nmin, nmax)):
        bar.update(k)
        logs.nodes[k] = n

        p = np.random.uniform(0, 1, (n, d))
        diam = np.array([], dtype=int)
        hd = np.array([], dtype=int)
        hb = np.array([], dtype=int)
        r = 0
        while r < rep:
            A = erdos_renyi(n, degree / (n - 1))
            if distances.is_inf_rigid(A, p):
                G = geodesics(A)
                diam = np.append(diam, np.max(G).astype(int))
                hd = np.append(hd, distances.minimum_rigidity_extents(G, p))
                hb = np.append(hb, bearings.minimum_rigidity_extents(G, p))
                r += 1

        logs.diam[k], logs.diam_count[k] = np.unique(diam, return_counts=True)
        logs.hd[k], logs.hd_count[k] = np.unique(hd, return_counts=True)
        logs.hb[k], logs.hb_count[k] = np.unique(hb, return_counts=True)

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
    diam=np.empty(size, dtype=np.ndarray),
    diam_count=np.empty(size, dtype=np.ndarray),
    hd=np.empty(size, dtype=np.ndarray),
    hd_count=np.empty(size, dtype=np.ndarray),
    hb=np.empty(size, dtype=np.ndarray),
    hb_count=np.empty(size, dtype=np.ndarray),
)

# ------------------------------------------------------------------
# Simulación
# ------------------------------------------------------------------
bar = progressbar.ProgressBar(maxval=size)

logs = run(d, nmin, nmax, degree, rep, logs)

write_csv('/tmp/nodes.csv', logs.nodes, one_row=True)
write_csv('/tmp/diam.csv', logs.diam)
write_csv('/tmp/diam_count.csv', logs.diam_count)
write_csv('/tmp/hd.csv', logs.hd)
write_csv('/tmp/hd_count.csv', logs.hd_count)
write_csv('/tmp/hb.csv', logs.hb)
write_csv('/tmp/hb_count.csv', logs.hb_count)
