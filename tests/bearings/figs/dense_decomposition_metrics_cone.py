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
from uvnpy.network.subframeworks import superframework_geodesics
from uvnpy.bearings.real_d.core import is_inf_rigid, minimum_rigidity_extents
from uvnpy.network import cone_graph

# ------------------------------------------------------------------
# Definición de variables, funciones y clases
# ------------------------------------------------------------------


@dataclass
class Logs(object):
    nodes: np.ndarray
    delay: np.ndarray
    delay_count: np.ndarray
    compl: np.ndarray
    compl_count: np.ndarray


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(d, nmin, nmax, sens_range, rep, logs):
    bar.start()

    sens_cos = np.cos(np.deg2rad(120.0 / 2))

    for k, n in enumerate(range(nmin, nmax)):
        bar.update(k)
        logs.nodes[k] = n

        delay = np.array([], dtype=int)
        compl = np.array([], dtype=float)
        r = 0

        while r < rep:
            p = np.random.uniform(0.0, 1.0, (n, d))
            baricenter = np.mean(p, axis=0)
            axes = transformations.unit_vector(baricenter - p, axis=1)
            D = cone_graph.adjacency_matrix(p, axes, sens_range, sens_cos)
            A = as_undirected(D).astype(float)
            if is_inf_rigid(A, p):
                G = geodesics(A)
                h = minimum_rigidity_extents(G, p)
                s = superframework_geodesics(G, h)
                n_state = [
                    len([j for j in range(n) if G[i, j] < s[j]])
                    for i in range(n)
                ]
                n_action = [
                    sum(
                        len([k for k in range(n) if G[i, j] < G[j, k] <= h[j]])
                        for j in range(n)
                    )
                    for i in range(n)
                ]
                delay_ratio = 2 * h / G.max()
                compl_ratio = np.add(n_state, n_action) / (1 + A.sum(axis=1))
                delay = np.append(delay, delay_ratio)
                compl = np.append(compl, compl_ratio)
                r += 1

        logs.delay[k], logs.delay_count[k] = np.unique(
            delay, return_counts=True
        )
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
    delay=np.empty(size, dtype=np.ndarray),
    delay_count=np.empty(size, dtype=np.ndarray),
    compl=np.empty(size, dtype=np.ndarray),
    compl_count=np.empty(size, dtype=np.ndarray)
)

# ------------------------------------------------------------------
# Simulación
# ------------------------------------------------------------------
bar = progressbar.ProgressBar(maxval=size)

logs = run(d, nmin, nmax, sens_range, rep, logs)

write_csv('/tmp/nodes.csv', logs.nodes, one_row=True)
write_csv('/tmp/delay.csv', logs.delay)
write_csv('/tmp/delay_count.csv', logs.delay_count)
write_csv('/tmp/compl.csv', logs.compl)
write_csv('/tmp/compl_count.csv', logs.compl_count)
