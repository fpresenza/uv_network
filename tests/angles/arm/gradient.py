#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from transformations import unit_vector

from uvnpy.angles.local_frame.core import (
    angle_indices,
    is_angle_rigid,
    angle_rigidity_matrix
)

# ------------------------------------------------------------------
# Functions, Classes and Configurations
# ------------------------------------------------------------------


def A(p):
    X = angle_rigidity_matrix(edge_set, p)
    return X.T.dot(X)


def lambda8(p):
    return np.linalg.eigvalsh(A(p))[7]


def Aijk(p, i, j, k):
    dij = np.sqrt(np.sum((p[j] - p[i])**2))
    bij = unit_vector(p[j] - p[i])
    Pij = np.eye(3) - np.outer(bij, bij)

    dik = np.sqrt(np.sum((p[k] - p[i])**2))
    bik = unit_vector(p[k] - p[i])
    Pik = np.eye(3) - np.outer(bik, bik)

    qijk = Pij.dot(bik) / dij
    qikj = Pik.dot(bij) / dik

    aijk = np.zeros((n, 3), dtype=np.float64)
    aijk[i] = - qijk - qikj
    aijk[j] = qijk
    aijk[k] = qikj

    return aijk.ravel()


def dAijk_l(p, i, j, k, m):
    dij = np.sqrt(np.sum((p[j] - p[i])**2))
    bij = unit_vector(p[j] - p[i])
    Pij = np.eye(3) - np.outer(bij, bij)

    dik = np.sqrt(np.sum((p[k] - p[i])**2))
    bik = unit_vector(p[k] - p[i])
    Pik = np.eye(3) - np.outer(bik, bik)

    Dijk = Pij.dot(np.outer(bik, bij))
    Dijk += np.outer(bij, bik).dot(Pij)
    Dijk += bij.dot(bik) * Pij
    Dijk /= dij**2

    Dikj = Pik.dot(np.outer(bij, bik))
    Dikj += np.outer(bik, bij).dot(Pik)
    Dikj += bik.dot(bij) * Pik
    Dikj /= dik**2

    Eijk = Pij.dot(Pik) / (dij * dik)
    Eikj = Pik.dot(Pij) / (dik * dij)

    daijk = np.zeros((n, 3, 3), dtype=np.float64)
    if m == i:
        daijk[i] = - Dijk - Dikj + Eijk + Eikj
        daijk[j] = Dijk - Eikj
        daijk[k] = Dikj - Eijk
    elif m == j:
        daijk[i] = Dijk - Eijk
        daijk[j] = - Dijk
        daijk[k] = Eijk
    elif m == k:
        daijk[i] = Dikj - Eikj
        daijk[j] = Eikj
        daijk[k] = - Dikj

    return np.hstack(daijk)


def e(i, j):
    x = np.zeros((n, 3))
    x[i, j] = 1.0
    return x


def complete_angle_set(out_neighbors):
    i, j = np.triu_indices(out_neighbors.size, k=1)
    return np.column_stack([out_neighbors[i], out_neighbors[j]])


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
# --- simulation parameters --- #
np.set_printoptions(suppress=True, precision=6, linewidth=250)
np.random.seed(0)

# --- world parameters --- #
n = 4
edge_set = np.array([
    [0, 1],
    [0, 2],
    [1, 0],
    [1, 2],
    [0, 3],
    [1, 3]
])
angle_set = angle_indices(n, edge_set).astype(int)
# print(angle_set)
p = np.random.uniform(0.0, 1.0, (n, 3))

if not is_angle_rigid(edge_set, p):
    raise ValueError('The initial framework is not IAR.')

rigidity_laplacian = A(p)
evals, evecs = np.linalg.eigh(rigidity_laplacian)
v = evecs[:, 7].reshape(n, 3)
# print(evals[7], v.ravel().dot(rigidity_laplacian).dot(v.ravel()), '\n')

# --- compute derivatives numerically --- #
h = 1e-9    # step
dlambda1 = np.zeros((n, 3), dtype=np.float64)
for m in range(n):
    for d in range(3):
        dlambda1[m, d] = (lambda8(p + h*e(m, d)) - lambda8(p - h*e(m, d))) / (2*h)

print(dlambda1)

# --- compute derivatives numerically (using formula) --- #
h = 1e-9    # step
dlambda2 = np.zeros((n, 3), dtype=np.float64)
for m in range(n):
    for d in range(3):
        dA = (A(p + h * e(m, d)) - A(p - h * e(m, d))) / (2 * h)
        dlambda2[m, d] = v.ravel().dot(dA).dot(v.ravel())

print(dlambda2)

# --- compute derivatives numerically (using formula 2) --- #
h = 1e-9    # step
dlambda3 = np.zeros((n, 3), dtype=np.float64)
for m in range(n):
    for d in range(3):
        s = 0.0
        for i, j, k in angle_set:
            dAijk = Aijk(p + h*e(m, d), i, j, k) - Aijk(p - h*e(m, d), i, j, k)
            dAijk /= (2*h)
            s += 2.0 * Aijk(p, i, j, k).dot(v.ravel()) * dAijk.dot(v.ravel())
        dlambda3[m, d] = s

print(dlambda3)

# --- compute derivatives numerically by closed formula --- #
h = 1e-9    # step
dlambda4 = np.zeros((n, 3), dtype=np.float64)
for m in range(n):
    s = np.zeros(3, dtype=np.float64)
    for i, j, k in angle_set:
        dAijk = dAijk_l(p, i, j, k, m)
        s += 2.0 * Aijk(p, i, j, k).dot(v.ravel()) * dAijk.dot(v.ravel())
    dlambda4[m] = s

print(dlambda4)

# --- compute derivatives by closed formula 2 --- #
dlambda5 = np.zeros((n, 3), dtype=np.float64)
for i in range(n):
    out_neighbors = edge_set[:, 1][edge_set[:, 0] == i]

    distances = {
        j: np.sqrt(np.sum((p[j] - p[i])**2)) for j in out_neighbors
    }
    bearings = {
        j: unit_vector(p[j] - p[i]) for j in out_neighbors
    }

    for j, k in complete_angle_set(out_neighbors):
        dij = distances[j]
        bij = bearings[j]
        Pij = np.eye(3) - np.outer(bij, bij)

        dik = distances[k]
        bik = bearings[k]
        Pik = np.eye(3) - np.outer(bik, bik)

        qijk = Pij.dot(bik) / dij
        qikj = Pik.dot(bij) / dik

        sijk = qijk.dot(v[j] - v[i]) + qikj.dot(v[k] - v[i])

        Dijk = Pij.dot(np.outer(bik, bij))
        Dijk += np.outer(bij, bik).dot(Pij)
        Dijk += bij.dot(bik) * Pij
        Dijk /= dij**2

        Dikj = Pik.dot(np.outer(bij, bik))
        Dikj += np.outer(bik, bij).dot(Pik)
        Dikj += bik.dot(bij) * Pik
        Dikj /= dik**2

        Eijk = Pij.dot(Pik) / (dij * dik)
        Eikj = Pik.dot(Pij) / (dik * dij)

        nijk_i = (Dijk - Eikj).dot(v[j] - v[i]) + (Dikj - Eijk).dot(v[k] - v[i])
        nijk_j = - Dijk.dot(v[j] - v[i]) + Eijk.dot(v[k] - v[i])
        nijk_k = Eikj.dot(v[j] - v[i]) - Dikj.dot(v[k] - v[i])

        dlambda5[i] += 2 * sijk * nijk_i
        dlambda5[j] += 2 * sijk * nijk_j
        dlambda5[k] += 2 * sijk * nijk_k

print(dlambda5)

print(np.allclose(dlambda1, dlambda2, atol=1e-4))
print(np.allclose(dlambda1, dlambda3, atol=1e-4))
print(np.allclose(dlambda1, dlambda4, atol=1e-4))
print(np.allclose(dlambda1, dlambda5, atol=1e-4))
