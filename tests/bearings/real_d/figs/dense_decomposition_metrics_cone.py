#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import argparse
import progressbar
from dataclasses import dataclass
import numpy as np

from uvnpy.toolkit.data import write_csv
from uvnpy.network.core import geodesics, as_undirected
from uvnpy.network.subframeworks import superframework_geodesics
from uvnpy.bearings.real_d.core import is_inf_rigid, minimum_rigidity_extents
from uvnpy.network.graphs import ConeGraph

# ------------------------------------------------------------------
# Definición de variables, funciones y clases
# ------------------------------------------------------------------


@dataclass
class Logs(object):
    nodes: np.ndarray
    delay: np.ndarray
    compl: np.ndarray


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(nmin, nmax, size, rep, logs):
    bar.start()

    sens_cos = np.cos(np.deg2rad(120.0 / 2))
    sens_range = 1.0

    for r in range(rep):
        for k, n in enumerate(range(nmin, nmax)):
            bar.update(size * r + k)
            side_length = np.cbrt(n)  # side_length = np.cbrt(vol = n)
            rigid_graph = False

            while not rigid_graph:
                if n == nmin:
                    positions = np.random.uniform(0.0, side_length, (nmin, 3))
                    angles = np.random.uniform(-np.pi, np.pi, nmin)
                    axes = np.empty((nmin, 3), dtype=float)
                    axes[:nmin, 0] = np.cos(angles)
                    axes[:nmin, 1] = np.sin(angles)
                    axes[:nmin, 2] = 0.0
                    cone_graph = ConeGraph(
                        positions, axes, dmax=sens_range, cmin=sens_cos
                    )
                else:
                    position = np.random.uniform(0.0, side_length, 3)
                    angle = np.random.uniform(-np.pi, np.pi)
                    axis = np.empty(3, dtype=float)
                    axis[0] = np.cos(angle)
                    axis[1] = np.sin(angle)
                    axis[2] = 0.0
                    cone_graph.append_vertex(position, axis)

                E = cone_graph.edge_set()
                p = cone_graph.positions(d=3)
                if is_inf_rigid(E, p):
                    rigid_graph = True
                    A = as_undirected(
                        cone_graph.adjacency_matrix()
                    ).astype(float)
                    G = geodesics(A)
                    h = minimum_rigidity_extents(G, p)
                    s = superframework_geodesics(G, h)
                    n_state = np.sum([
                        len([j for j in range(n) if G[i, j] < s[j]])
                        for i in range(n)
                    ])
                    n_action = np.sum([
                        sum(
                            len([
                                k for k in range(n)
                                if G[i, j] < G[j, k] <= h[j]
                            ])
                            for j in range(n)
                        )
                        for i in range(n)
                    ])
                    d_comm = np.max([
                        max([
                            h[j] + G[i, j] for j in range(n)
                            if G[i, j] <= h[j]
                        ])
                        for i in range(n)
                    ])
                    logs.delay[r, k] = d_comm / G.max()
                    logs.compl[r, k] = (n_state + n_action) / A.sum()
                else:
                    cone_graph.remove_vertex(-1)

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
    '-r', '--rep',
    default=1, type=int, help='number of repetitions'
)

arg = parser.parse_args()

# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------
nmin = 2
nmax = arg.nodes + 1
size = nmax - nmin
rep = arg.rep
logs = Logs(
    nodes=np.arange(nmin, nmax),
    delay=np.empty((rep, size), dtype=np.ndarray),
    compl=np.empty((rep, size), dtype=np.ndarray),
)

# ------------------------------------------------------------------
# Simulación
# ------------------------------------------------------------------
bar = progressbar.ProgressBar(maxval=size * rep)

logs = run(nmin, nmax, size, rep, logs)

write_csv('/tmp/nodes.csv', logs.nodes, one_row=True)
write_csv('/tmp/delay.csv', logs.delay)
write_csv('/tmp/compl.csv', logs.compl)
