#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import argparse
from dataclasses import dataclass
import numpy as np
import progressbar

from uvnpy.toolkit.data import write_csv
from uvnpy.graphs.core import geodesics, as_undirected
from uvnpy.graphs.models import ErdosRenyi
import uvnpy.distances.core as distances
import uvnpy.bearings.real_d.core as bearings

# ------------------------------------------------------------------
# Definición de variables, funciones y clases
# ------------------------------------------------------------------


@dataclass
class Logs(object):
    nodes: np.ndarray
    hd: np.ndarray
    hd_count: np.ndarray
    hb: np.ndarray
    hb_count: np.ndarray


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(d, nmin, nmax, degree, rep, logs):
    bar.start()

    for k, n in enumerate(range(nmin, nmax)):
        bar.update(k)
        logs.nodes[k] = n

        p = np.random.uniform(0, 1, (n, d))
        hd = np.array([], dtype=int)
        hb = np.array([], dtype=int)
        r = 0
        prob = degree / (n - 1)
        while r < rep:
            graph = ErdosRenyi(n, prob, directed=False)
            E = graph.edge_set(directed=False)
            if distances.is_distance_rigid(E, p):
                A = as_undirected(graph.adjacency_matrix()).astype(float)
                G = geodesics(A)
                hd = np.append(
                    hd, distances.minimum_distance_rigidity_extents(E, G, p)
                )
                hb = np.append(
                    hb, bearings.minimum_bearing_rigidity_extents(E, G, p)
                )
                r += 1

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
nmin = max(3, int(degree) + 1)
nmax = arg.nodes + 1
size = nmax - nmin
rep = arg.rep
logs = Logs(
    nodes=np.empty(size, dtype=int),
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
write_csv('/tmp/hd.csv', logs.hd)
write_csv('/tmp/hd_count.csv', logs.hd_count)
write_csv('/tmp/hb.csv', logs.hb)
write_csv('/tmp/hb_count.csv', logs.hb_count)
