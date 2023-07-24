#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import argparse
import collections
import numpy as np
import progressbar

from uvnpy.network.subsets import geodesics, fast_degree_load_std
from uvnpy.rsn.rigidity import (
    fast_extents,
    minimum_radius,
    sparse_centers_binary_search,
    sparse_centers_two_steps)
from uvnpy.network.disk_graph import adjacency as disk_adjacency


# ------------------------------------------------------------------
# Definición de variables, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple(
    'Logs',
    'nodes \
    diam \
    hmax \
    sparse_hmax \
    sparse_hmax_subopt \
    sparse_load \
    sparse_load_subopt \
    edges')


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------
def run(d, nmin, nmax, cutoff, logs, threshold, rep):
    bar.start()

    for k, n in enumerate(range(nmin, nmax)):
        bar.update(k)
        logs.nodes[k] = n

        for r in range(rep):
            p = np.random.uniform(0, 1, (n, d))
            A0 = disk_adjacency(p, dmax=2/np.sqrt(n))
            A, Rmin = minimum_radius(A0, p, threshold, return_radius=True)

            h = fast_extents(A, p, threshold)
            geo = geodesics(A)
            deg = A.sum(axis=1)
            logs.diam[k, r] = np.max(geo)
            logs.hmax[k, r] = np.max(h)
            logs.edges[k, r] = np.sum(deg) / 2

            if n <= nmin + cutoff:
                sparse_h = sparse_centers_binary_search(
                    A, p, h, fast_degree_load_std, threshold,
                    vertices_only=True)
                logs.sparse_hmax[k, r] = np.max(sparse_h)
                logs.sparse_load[k, r] = fast_degree_load_std(
                    deg, sparse_h, geo) / n

            sparse_h_subopt = sparse_centers_two_steps(
                A, p, h, fast_degree_load_std, threshold, vertices_only=True)
            logs.sparse_hmax_subopt[k, r] = np.max(sparse_h_subopt)
            logs.sparse_load_subopt[k, r] = fast_degree_load_std(
                deg, sparse_h_subopt, geo) / n

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
    '-f', '--full',
    default=4, type=int, help='full search')
parser.add_argument(
    '-r', '--rep',
    default=1, type=int, help='number of repetitions')
parser.add_argument(
    '-s', '--save',
    default=True, action='store_true', help='flag to store data')

arg = parser.parse_args()

# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------
d = 2
nmin = d + 2
nmax = arg.nodes + 1
cutoff = arg.full - nmin
threshold = 1e-5
size = nmax - nmin
rep = arg.rep
logs = Logs(
    nodes=np.empty(size, dtype=int),
    diam=np.empty((size, rep)),
    hmax=np.empty((size, rep)),
    sparse_hmax=np.empty((size, rep)),
    sparse_hmax_subopt=np.empty((size, rep)),
    sparse_load=np.empty((size, rep)),
    sparse_load_subopt=np.empty((size, rep)),
    edges=np.empty((size, rep)))

# ------------------------------------------------------------------
# Simulación
# ------------------------------------------------------------------
bar = progressbar.ProgressBar(maxval=size)

logs = run(d, nmin, nmax, cutoff, logs, threshold, rep)

np.savetxt('/tmp/nodes.csv', logs.nodes, delimiter=',')
np.savetxt('/tmp/diam.csv', logs.diam, delimiter=',')
np.savetxt('/tmp/hmax.csv', logs.hmax, delimiter=',')
np.savetxt('/tmp/sparse_hmax.csv', logs.sparse_hmax, delimiter=',')
np.savetxt(
    '/tmp/sparse_hmax_subopt.csv', logs.sparse_hmax_subopt, delimiter=',')
np.savetxt('/tmp/sparse_load.csv', logs.sparse_load, delimiter=',')
np.savetxt(
    '/tmp/sparse_load_subopt.csv', logs.sparse_load_subopt, delimiter=',')
np.savetxt('/tmp/edges.csv', logs.edges, delimiter=',')
