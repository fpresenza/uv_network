#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import argparse
import numpy as np
import progressbar

from uvnpy.network.subsets import degree_load_std
from uvnpy.rsn.rigidity import (
    fast_extents,
    minimum_radius,
    sparse_centers_full_search,
    sparse_centers_binary_search)
from uvnpy.network.disk_graph import adjacency as disk_adjacency


# ------------------------------------------------------------------
# Definición de variables, funciones y clases
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------
def run(d, nmin, nmax, threshold, rep):
    bar.start()

    for k, n in enumerate(range(nmin, nmax)):
        bar.update(k)

        for _ in range(rep):
            p = np.random.uniform(0, 1, (n, d))
            A0 = disk_adjacency(p, dmax=2/np.sqrt(n))
            A = minimum_radius(A0, p, threshold)

            h = fast_extents(A, p, threshold)
            full_h = sparse_centers_full_search(
                A, p, h, degree_load_std, threshold, vertices_only=False)
            binary_h = sparse_centers_binary_search(
                A, p, h, degree_load_std, threshold, vertices_only=False)

            if degree_load_std(A, full_h) != degree_load_std(A, binary_h):
                print(A)
                print(p)
                print(h)
                print(full_h, degree_load_std(A, full_h))
                print(binary_h, degree_load_std(A, binary_h))
                raise ValueError

    bar.finish()


if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Parseo de argumentos
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-n', '--nodes',
        default=20, type=int, help='number of nodes')
    parser.add_argument(
        '-r', '--rep',
        default=10, type=int, help='number of repetitions')

    arg = parser.parse_args()

    # ------------------------------------------------------------------
    # Configuración
    # ------------------------------------------------------------------
    d = 2
    nmin = d + 2
    nmax = arg.nodes
    rep = arg.rep
    threshold = 1e-4
    size = (nmax - nmin)

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    bar = progressbar.ProgressBar(maxval=size)
    run(d, nmin, nmax, threshold, rep)
