#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import argparse
import numpy as np
import progressbar
import networkx as nx
import matplotlib.pyplot as plt

from uvnpy.graphs.models import ErdosRenyi
from uvnpy.angles.local_frame.core import is_angle_rigid
from uvnpy.toolkit.plot import point_arrow_framework

# ------------------------------------------------------------------
# Definición de variables, funciones y clases
# ------------------------------------------------------------------


def is_rooted_out_branching(n, E):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(E)
    is_root = [len(nx.descendants(G, i)) == (n - 1) for i in range(n)]
    return np.any(is_root)


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(d, nmin, nmax, degree, rep):
    bar.start()

    for k, n in enumerate(range(nmin, nmax)):
        bar.update(k)
        p = np.random.uniform(0, 1, (n, d))
        for deg in degree:
            r = 0
            prob = deg / (n - 1)
            while r < rep:
                graph = ErdosRenyi(n, prob)
                E = graph.edge_set()
                is_iar = is_angle_rigid(E, p, threshold=1e-10)
                is_rooted = is_rooted_out_branching(n, E)
                if is_iar and not is_rooted:
                    print(p.ravel())
                    print(E.ravel())
                    fig, ax = plt.subplots()
                    point_arrow_framework(ax, p, E)
                    plt.show()
                    raise ValueError
                r += 1

    bar.finish()


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
d = 2
degree = range(2, int(arg.degree) + 1)
nmin = max(3, int(arg.degree) + 1)
nmax = arg.nodes + 1
size = nmax - nmin
rep = arg.rep

# ------------------------------------------------------------------
# Simulación
# ------------------------------------------------------------------
bar = progressbar.ProgressBar(maxval=size)
run(d, nmin, nmax, degree, rep)
