#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import argparse
import numpy as np
import progressbar

from uvnpy.graphs.models import ErdosRenyi
from uvnpy.bearings.local_frame.core import is_bearing_rigid
from uvnpy.angles.local_frame.core import is_angle_rigid, angle_indices

# ------------------------------------------------------------------
# Definición de variables, funciones y clases
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(d, nmin, nmax, degree, rep):
    bar.start()

    for k, n in enumerate(range(nmin, nmax)):
        bar.update(k)
        p = np.random.uniform(0, 1, (n, d))
        psi = np.random.uniform(-1, 1, (n, d))
        x = np.concatenate((p, psi), axis=1)
        for deg in degree:
            r = 0
            prob = deg / (n - 1)
            while r < rep:
                graph = ErdosRenyi(n, prob)
                E = graph.edge_set()
                A = angle_indices(np.arange(n), E).astype(int)
                b = is_bearing_rigid(E, x, threshold=1e-10)
                a = is_angle_rigid(A, p, threshold=1e-10)
                if b and not a:
                    print(p.ravel())
                    print(E.ravel())
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
d = 3
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
