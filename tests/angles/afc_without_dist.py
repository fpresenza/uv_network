#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import progressbar
import numpy as np
from numba import njit

from uvnpy.angles.local_frame.core import (
    angle_indices, angle_function, angle_rigidity_matrix
)


# ------------------------------------------------------------------
# Functions, Classes and Configurations
# ------------------------------------------------------------------
np.set_printoptions(suppress=True, precision=10, linewidth=200)


@njit
def angle_rigidity_matrix_scale_free(E, p):
    """Angle Rigidity matrix (jacobian of the bearing function)

    args:
        E: edge set | (m, 2)-array
        p: positions | (..., n, d)-array

    returns:
        scale free angle rigidity matrix | (a, n*d)
    """
    n, d = p.shape
    Id = np.eye(d)

    r = p[..., E[:, 1], :] - p[..., E[:, 0], :]
    q = np.sqrt(np.square(r).sum(axis=-1))
    b = r / q[..., np.newaxis]
    R = np.empty(shape=(0, n*d), dtype=float)
    for i in range(n):
        S = E[:, 0] == i
        s = sum(S)
        bi = b[S]
        Pi = Id - bi[..., :, np.newaxis] * bi[..., np.newaxis, :]

        x, y = np.triu_indices(s, k=1)
        Nij = np.sum(bi[y, :, np.newaxis] * Pi[x], axis=1)
        Nik = np.sum(bi[x, :, np.newaxis] * Pi[y], axis=1)

        ds = int(s * (s - 1) / 2)
        Ri = np.zeros(shape=(ds, n, d), dtype=float)
        Ei = E[S]
        j, k = Ei[x, 1], Ei[y, 1]
        for a in range(ds):
            Ri[a, i] = - Nij[a] - Nik[a]
            Ri[a, j[a]] = Nij[a]
            Ri[a, k[a]] = Nik[a]
        R = np.concatenate((R, Ri.reshape(ds, n*d)))

    return R

# ------------------------------------------------------------------
# Simulation loop inner functions
# ------------------------------------------------------------------


def simu_step():
    p = np.random.uniform(-1, 1, (n, 3))

    A = angle_rigidity_matrix(edge_set, p)
    B = angle_rigidity_matrix_scale_free(edge_set, p)
    a = angle_function(edge_set, p)

    e = a - desired_angles
    M = A.dot(B.T)
    # print(e)
    x = e.dot(M).dot(e)
    # y = np.linalg.eigvals(M + M.T).min()
    if x <= 0.0:
        print(x, p)
    # if y < -1e-9:
    #     print(y, p)


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
n = 4
nodes = np.arange(n)
edge_set = np.array([
    [0, 1],
    [0, 2],
    [1, 0],
    [1, 2],
    [0, 3],
    [1, 3]
])
angle_set = angle_indices(n, edge_set).astype(int)

desired_position = np.random.uniform(0.0, 1.0, (n, 3))
desired_angles = angle_function(edge_set, desired_position)
print(desired_position)

# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------

simu_counter = 0
bar = progressbar.ProgressBar(maxval=simu_num_steps).start()

while simu_counter < simu_num_steps:
    simu_step()

    simu_counter += 1

    # bar.update(simu_counter)

bar.finish()
