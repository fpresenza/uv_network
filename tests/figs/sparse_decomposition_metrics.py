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
from uvnpy.network.load import one_token_for_all, one_token_for_each
from uvnpy.network.subframeworks import (
    superframework_extents,
    isolated_edges,
    sparse_subframeworks_greedy_search
)
from uvnpy.distances.core import (
    minimum_rigidity_extents,
    minimum_rigidity_radius
)

# ------------------------------------------------------------------
# Definición de variables, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple(
    'Logs',
    'nodes \
    diam \
    hmax \
    sparse_hmax_subopt \
    sparse_load_subopt \
    edges \
    isolated_edges')


def metrics(geodesics, extents):
    action_load = one_token_for_each(geodesics, extents)
    super_extents = superframework_extents(geodesics, extents)
    state_load = one_token_for_all(geodesics, super_extents)
    n_isolated_edges = len(isolated_edges(geodesics, extents))
    return action_load + state_load, n_isolated_edges


def network_load(geodesics, extents):
    load, edges = metrics(geodesics, extents)
    return load + (1 + 0.5) * edges


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------
def run(d, nmin, nmax, logs, rep, dense):
    bar.start()

    for k, n in enumerate(range(nmin, nmax)):
        bar.update(k)
        logs.nodes[k] = n

        for r in range(rep):
            p = np.random.uniform(0, 1, (n, d))
            A0 = adjacency_from_positions(p, dmax=2/np.sqrt(n))
            A, Rmin = minimum_rigidity_radius(A0, p, return_radius=True)

            G = geodesics(A)
            h = minimum_rigidity_extents(G, p)
            degree = A.sum(axis=1)
            logs.diam[k, r] = np.max(G)
            logs.hmax[k, r] = np.max(h)
            logs.edges[k, r] = np.sum(degree) / 2

            if dense:
                sparse_h_subopt = h
            else:
                sparse_h_subopt = sparse_subframeworks_greedy_search(
                    G, h, network_load
                )

            logs.sparse_hmax_subopt[k, r] = np.max(sparse_h_subopt)
            load, n_isolated_edges = metrics(G, sparse_h_subopt)
            logs.sparse_load_subopt[k, r] = load
            logs.isolated_edges[k, r] = n_isolated_edges / 2

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
parser.add_argument(
    '-s', '--save',
    default=True, action='store_true', help='flag to store data')
parser.add_argument(
    '-d', '--dense',
    default=False, action='store_true', help='select dense or sparse mode')

arg = parser.parse_args()

# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------
d = 2
nmin = d + 2
nmax = arg.nodes + 1
size = nmax - nmin
rep = arg.rep
dense = arg.dense
logs = Logs(
    nodes=np.empty(size, dtype=int),
    diam=np.empty((size, rep)),
    hmax=np.empty((size, rep)),
    sparse_hmax_subopt=np.empty((size, rep)),
    sparse_load_subopt=np.empty((size, rep)),
    edges=np.empty((size, rep)),
    isolated_edges=np.empty((size, rep))
)

# ------------------------------------------------------------------
# Simulación
# ------------------------------------------------------------------
bar = progressbar.ProgressBar(maxval=size)

logs = run(d, nmin, nmax, logs, rep, dense)

np.savetxt('/tmp/nodes.csv', logs.nodes, delimiter=',')
np.savetxt('/tmp/diam.csv', logs.diam, delimiter=',')
np.savetxt('/tmp/hmax.csv', logs.hmax, delimiter=',')
np.savetxt(
    '/tmp/sparse_hmax_subopt.csv', logs.sparse_hmax_subopt, delimiter=',')
np.savetxt(
    '/tmp/sparse_load_subopt.csv', logs.sparse_load_subopt, delimiter=',')
np.savetxt('/tmp/edges.csv', logs.edges, delimiter=',')
np.savetxt('/tmp/isolated_edges.csv', logs.isolated_edges, delimiter=',')
