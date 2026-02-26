#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from transformations import unit_vector

from uvnpy.toolkit.functions import cosine_activation, cosine_activation_derivative

# ------------------------------------------------------------------
# Functions, Classes and Configurations
# ------------------------------------------------------------------


def w(p, R):
    ei = R[0][:, 0]

    dij = np.linalg.norm(p[1] - p[0])
    bij = unit_vector(p[1] - p[0])
    dik = np.linalg.norm(p[2] - p[0])
    bik = unit_vector(p[2] - p[0])

    wrij = 1 - s_r(dij)
    wfij = s_f(ei.dot(bij))
    wrik = 1 - s_r(dik)
    wfik = s_f(ei.dot(bik))

    return wrij * wfij * wrik * wfik


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
R = np.array([np.eye(3)])

# --- compute derivatives numerically --- #
h = 1e-9    # step
dw1 = np.zeros((3, 3), dtype=np.float64)
for m in range(3):
    for d in range(3):
        dw1[m, d] = ((w(p + h*e(m, d), R) - w(p - h*e(m, d), R)) / (2*h)).item()

print(dw1)

# # --- compute derivatives by closed formula --- #
dw2 = np.zeros((3, 3), dtype=np.float64)

ei = R[0][:, 0]

dij = np.linalg.norm(p[1] - p[0])
bij = unit_vector(p[1] - p[0])
Pij = np.eye(3) - np.outer(bij, bij)
dik = np.linalg.norm(p[2] - p[0])
bik = unit_vector(p[2] - p[0])
Pik = np.eye(3) - np.outer(bik, bik)

wrij = 1 - s_r(dij)
wfij = s_f(ei.dot(bij))
wrik = 1 - s_r(dik)
wfik = s_f(ei.dot(bik))

nij = ei.dot(bij)
nik = ei.dot(bik)

dw2[0] = wfij*wfik*(wrik*ds_r(dij) * bij + wrij*ds_r(dik) * bik)
dw2[0] -= wrij*wrik*(wfik*ds_f(nij) * Pij/dij + wfij*ds_f(nik) * Pik/dik).dot(ei)
dw2[1] = - wfij*wfik*wrik*ds_r(dij) * bij + wrij*wrik*wfik*ds_f(nij) * Pij.dot(ei)/dij
dw2[2] = - wfij*wfik*wrij*ds_r(dik) * bik + wrij*wrik*wfij*ds_f(nik) * Pik.dot(ei)/dik

print(dw2)

print(np.allclose(dw1, dw2, atol=1e-4))
