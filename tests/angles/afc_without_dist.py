#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import progressbar
import numpy as np

from uvnpy.angles.local_frame.core import angle_indices, angle_function


# ------------------------------------------------------------------
# Functions, Classes and Configurations
# ------------------------------------------------------------------
np.set_printoptions(suppress=True, precision=10)

# ------------------------------------------------------------------
# Simulation loop inner functions
# ------------------------------------------------------------------


def simu_step():
    p = np.random.uniform(-1, 1, (n, 3))

    N = len(angle_set)
    A = np.empty(shape=(N, 3*n), dtype=np.float64)
    B = np.empty(shape=(N, 3*n), dtype=np.float64)
    a = np.empty(shape=(N,), dtype=np.float64)

    for m in range(N):
        i, j, k = angle_set[m]

        rij = p[j] - p[i]
        dij = np.sqrt(np.sum(rij**2))
        bij = rij / dij
        Pij = np.eye(3) - np.outer(bij, bij)

        rik = p[k] - p[i]
        dik = np.sqrt(np.sum(rik**2))
        bik = rik / dik
        Pik = np.eye(3) - np.outer(bik, bik)

        sijk = Pij.dot(bik)
        qijk = sijk / dij
        sikj = Pik.dot(bij)
        qikj = sikj / dik

        A[m, 0:3] = - qijk - qikj
        A[m, 3:6] = qijk
        A[m, 6:9] = qikj

        B[m, 0:3] = - sijk - sikj
        B[m, 3:6] = sijk
        B[m, 6:9] = sikj

        a[m] = bij.dot(bik)

    e = a - desired_angles
    M = A.dot(B.T)
    # print(e, M)
    x = e.dot(M).dot(e)
    if x <= 0.0:
        print(x, p)


# ------------------------------------------------------------------
# Argument parse
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-r', '--rep',
    default=1, type=int, help='number of repetitions'
)
arg = parser.parse_args()

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
# --- simulation parameters --- #
simu_num_steps = arg.rep
np.random.seed(0)

# --- world parameters --- #
t = 0.0
n = 3
nodes = np.arange(n)
edge_set = np.array([
    [0, 1],
    [0, 2],
    [1, 0],
    [1, 2],
])
angle_set = angle_indices(n, edge_set).astype(int)

desired_position = np.random.uniform(0.0, 1.0, (n, 3))
desired_angles = angle_function(edge_set, desired_position)

# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------

simu_counter = 0
bar = progressbar.ProgressBar(maxval=simu_num_steps).start()

while simu_counter < simu_num_steps:
    simu_step()

    simu_counter += 1

    bar.update(simu_counter)

bar.finish()
