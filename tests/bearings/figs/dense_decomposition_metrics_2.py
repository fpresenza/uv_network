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
from uvnpy.network.core import geodesics
from uvnpy.network.load import one_token_for_each_per_node
from uvnpy.network.subframeworks import subframework_diameters
from uvnpy.network.random_graph import erdos_renyi
import uvnpy.bearings.core as bearings

# ------------------------------------------------------------------
# Definición de variables, funciones y clases
# ------------------------------------------------------------------


@dataclass
class Logs(object):
    nodes: np.ndarray
    diam: np.ndarray
    diam_count: np.ndarray
    compl: np.ndarray
    compl_count: np.ndarray


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
        compl = np.array([], dtype=int)
        r = 0
        prob = degree / (n - 1)
        while r < rep:
            A = erdos_renyi(n, prob)
            if bearings.is_inf_rigid(A, p):
                G = geodesics(A)
                h = bearings.minimum_rigidity_extents(G, p)
                diam_ratio = np.divide(subframework_diameters(G, h), G.max())
                diam = np.append(diam, diam_ratio)
                compl_ratio = [
                    one_token_for_each_per_node(Gi, hi) / np.sum(Ai)
                    for Ai, Gi, hi in zip(A, G, h)
                ]
                compl = np.append(compl, compl_ratio)
                r += 1

        logs.diam[k], logs.diam_count[k] = np.unique(diam, return_counts=True)
        logs.compl[k], logs.compl_count[k] = np.unique(
            compl, return_counts=True
        )

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
np.random.seed(0)
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
    compl=np.empty(size, dtype=np.ndarray),
    compl_count=np.empty(size, dtype=np.ndarray)
)

# ------------------------------------------------------------------
# Simulación
# ------------------------------------------------------------------
bar = progressbar.ProgressBar(maxval=size)

logs = run(d, nmin, nmax, degree, rep, logs)

write_csv('/tmp/nodes.csv', logs.nodes, one_row=True)
write_csv('/tmp/diam.csv', logs.diam)
write_csv('/tmp/diam_count.csv', logs.diam_count)
write_csv('/tmp/compl.csv', logs.compl)
write_csv('/tmp/compl_count.csv', logs.compl_count)
