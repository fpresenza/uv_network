#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import argparse
import collections
import numpy as np
import progressbar

from uvnpy.network.subsets import (
    geodesics,
    fast_degree_load_flat,
    supergraph_extent,
    kl_graphs
)
from uvnpy.rsn.rigidity import (
    fast_extents,
    minimum_radius,
    sparse_centers_greedy_search
)
from uvnpy.network.disk_graph import adjacency as disk_adjacency


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
    links')


def metrics(geodesics, extents):
    degree = (geodesics == 1).sum(axis=1).astype(float)
    load = fast_degree_load_flat(degree, extents, geodesics)
    super_extents = supergraph_extent(geodesics, extents)
    super_load = fast_degree_load_flat(degree, super_extents, geodesics)
    _, L = kl_graphs(geodesics, extents)
    return load + super_load, L.sum()


def cost(geodesics, extents):
    return sum(metrics(geodesics, extents))


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------
def run(d, nmin, nmax, logs, threshold, rep, dense):
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

            if dense:
                sparse_h_subopt = h
            else:
                sparse_h_subopt = sparse_centers_greedy_search(
                    A, p, h, cost, threshold
                )

            logs.sparse_hmax_subopt[k, r] = np.max(sparse_h_subopt)
            load, m_links = metrics(geo, sparse_h_subopt)
            logs.sparse_load_subopt[k, r] = load / n
            logs.links[k, r] = m_links

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
threshold = 1e-5
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
    links=np.empty((size, rep)))

# ------------------------------------------------------------------
# Simulación
# ------------------------------------------------------------------
bar = progressbar.ProgressBar(maxval=size)

logs = run(d, nmin, nmax, logs, threshold, rep, dense)

np.savetxt('/tmp/nodes.csv', logs.nodes, delimiter=',')
np.savetxt('/tmp/diam.csv', logs.diam, delimiter=',')
np.savetxt('/tmp/hmax.csv', logs.hmax, delimiter=',')
np.savetxt(
    '/tmp/sparse_hmax_subopt.csv', logs.sparse_hmax_subopt, delimiter=',')
np.savetxt(
    '/tmp/sparse_load_subopt.csv', logs.sparse_load_subopt, delimiter=',')
np.savetxt('/tmp/edges.csv', logs.edges, delimiter=',')
np.savetxt('/tmp/links.csv', logs.links, delimiter=',')
