#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from transformations import unit_vector

from uvnpy.toolkit.functions import cosine_activation, cosine_activation_derivative
from uvnpy.toolkit.geometry import (
    cross_product_matrix,
    rotation_matrix_from_vector
)

# ------------------------------------------------------------------
# Functions, Classes and Configurations
# ------------------------------------------------------------------


def cross_product_matrix_inverse(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def w(p, R):
    ei = R[:, 0]

    dij = np.linalg.norm(p[1] - p[0])
    bij = unit_vector(p[1] - p[0])
    dik = np.linalg.norm(p[2] - p[0])
    bik = unit_vector(p[2] - p[0])

    wrij = 1 - s_r(dij)
    wfij = s_f(ei.dot(bij))
    wrik = 1 - s_r(dik)
    wfik = s_f(ei.dot(bik))

    return wrij * wfij * wrik * wfik


def f(p, R):
    ei = R[:, 0]
    bij = unit_vector(p[1] - p[0])
    return ei.dot(bij)


def s_r(x):
    return cosine_activation(np.array([x]), 0.8, 1.0)


def ds_r(x):
    return cosine_activation_derivative(np.array([x]), 0.8, 1.0)


def s_f(x):
    return cosine_activation(np.array([x]), 0.5, 0.7)


def ds_f(x):
    return cosine_activation_derivative(np.array([x]), 0.5, 0.7)


def e(i, j):
    x = np.zeros((3, 3))
    x[i, j] = 1.0
    return x


def A(i):
    x = np.zeros(3)
    x[i] = 1.0
    return cross_product_matrix(x)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
# --- simulation parameters --- #
np.set_printoptions(suppress=True, precision=6, linewidth=250)
np.random.seed(0)

# --- world parameters --- #
p = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
    [0.6, 0.7, -0.1]
])
R = rotation_matrix_from_vector(np.random.uniform(-0.1, 0.1, 3))
S = cross_product_matrix
e1 = np.array([1.0, 0.0, 0.0])

# --- position part --- #
# --- compute derivatives numerically --- #
h = 1e-9    # step
dw1 = np.zeros((3, 3), dtype=np.float64)
for m in range(3):
    for d in range(3):
        dw1[m, d] = ((w(p + h*e(m, d), R) - w(p - h*e(m, d), R)) / (2*h)).item()

print(dw1)

# --- compute derivatives by closed formula --- #
dw2 = np.zeros((3, 3), dtype=np.float64)

ei = R[:, 0]

dij = np.linalg.norm(p[1] - p[0])
bij = unit_vector(p[1] - p[0])
Pij = np.eye(3) - np.outer(bij, bij)
dik = np.linalg.norm(p[2] - p[0])
bik = unit_vector(p[2] - p[0])
Pik = np.eye(3) - np.outer(bik, bik)

nij = ei.dot(bij)
nik = ei.dot(bik)

wrij = 1 - s_r(dij)
wfij = s_f(nij)
wrik = 1 - s_r(dik)
wfik = s_f(nik)

dw2[0] = wfij*wfik*(wrik*ds_r(dij) * bij + wrij*ds_r(dik) * bik)
dw2[0] -= wrij*wrik*(wfik*ds_f(nij) * Pij/dij + wfij*ds_f(nik) * Pik/dik).dot(ei)
dw2[1] = - wfij*wfik*wrik*ds_r(dij) * bij + wrij*wrik*wfik*ds_f(nij) * Pij.dot(ei)/dij
dw2[2] = - wfij*wfik*wrij*ds_r(dik) * bik + wrij*wrik*wfij*ds_f(nik) * Pik.dot(ei)/dik

print(dw2)

print(np.allclose(dw1, dw2, atol=1e-4))

# --- orientation part --- #
# --- compute derivatives numerically --- #
h = 1e-9    # step
dw1 = np.zeros(3, dtype=np.float64)
for m in range(3):
    for d in range(3):
        dw1[d] = ((w(p, R + h*A(d).dot(R)) - w(p, R - h*A(d).dot(R))) / (2*h)).item()

print(dw1)

# --- compute derivatives by closed formula --- #
ei = R[:, 0]

dij = np.linalg.norm(p[1] - p[0])
bij = unit_vector(p[1] - p[0])
# Pij = np.eye(3) - np.outer(bij, bij)
dik = np.linalg.norm(p[2] - p[0])
bik = unit_vector(p[2] - p[0])
# Pik = np.eye(3) - np.outer(bik, bik)

wrij = 1 - s_r(dij)
wfij = s_f(ei.dot(bij))
wrik = 1 - s_r(dik)
wfik = s_f(ei.dot(bik))

nij = ei.dot(bij)
nik = ei.dot(bik)

dw2 = 0.5 * wrij*wrik*S(ei).dot(wfik*ds_f(nij) * bij + wfij*ds_f(nik) * bik)

# dw2 = cross_product_matrix_inverse(
#     np.outer(bij, ei) - np.outer(ei, bij)
# )

print(dw2)

print(np.allclose(dw1/2, dw2, atol=1e-4))
