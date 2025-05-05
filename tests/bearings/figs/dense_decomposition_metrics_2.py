#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import argparse
import progressbar
from dataclasses import dataclass
import numpy as np
import transformations

from uvnpy.toolkit.data import write_csv
from uvnpy.network.core import geodesics, as_undirected
from uvnpy.network.load import one_token_for_each_sum
from uvnpy.network.subframeworks import subframework_diameters
import uvnpy.bearings.core as bearings
from uvnpy.network import cone_graph

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


def run(d, nmin, nmax, sens_range, rep, logs):
    bar.start()

    sens_cos = np.cos(np.deg2rad(120.0 / 2))

    for k, n in enumerate(range(nmin, nmax)):
        bar.update(k+1)
        logs.nodes[k] = n

        diam = np.array([], dtype=int)
        compl = np.array([], dtype=int)
        r = 0

        while r < rep:
            p = np.random.uniform(0.0, 1.0, (n, d))
            baricenter = np.mean(p, axis=0)
            axes = transformations.unit_vector(baricenter - p, axis=1)
            D = cone_graph.adjacency_matrix(p, axes, sens_range, sens_cos)
            A = as_undirected(D).astype(float)
            if bearings.is_inf_rigid(A, p):
                G = geodesics(A)
                h = bearings.minimum_rigidity_extents(G, p)
                diam_ratio = np.mean(subframework_diameters(G, h)) / G.max()
                diam = np.append(diam, diam_ratio)
                compl_ratio = one_token_for_each_sum(G, h) / A.sum()
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
    '-s', '--sens_range',
    default=1.0, type=float, help='sensing range'
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
sens_range = arg.sens_range
nmin = 2
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
bar = progressbar.ProgressBar(maxval=size + 1)

logs = run(d, nmin, nmax, sens_range, rep, logs)

write_csv('/tmp/nodes.csv', logs.nodes, one_row=True)
write_csv('/tmp/diam_{}.csv'.format(sens_range), logs.diam)
write_csv('/tmp/diam_count_{}.csv'.format(sens_range), logs.diam_count)
write_csv('/tmp/compl_{}.csv'.format(sens_range), logs.compl)
write_csv('/tmp/compl_count_{}.csv'.format(sens_range), logs.compl_count)
