#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar feb 14 11:23:21 -03 2025
"""
from numba import njit


THRESHOLD_EIG = 1e-6
THRESHOLD_SV = 1e-3


@njit
def rigidity_matrix(A, p):
    return


def rigidity_matrix_multiple_axes(D, p):
    """Matriz de rigidez

    args:
        D: matriz de incidencia (n, m)
        p: array de posiciones (n, d)

    returns
        R: jacobiano (ne, n * d)
    """
    return


@njit
def _rigidity_laplacian(A, p):
    return


def rigidity_laplacian(A, p):
    return


def rigidity_laplacian_multiple_axes(A, p):
    """Matriz de rigidez.

        S =  R^T W R

    A[i, j] >= 0 respresenta el peso asociado a cada enlace.

    args:
        A: matriz de adyacencia (..., n, n)
        p: array de posiciones (..., n, d)

    returns
        S: laplaciano de rigidez (..., n * d, n * d)
    """
    return


def is_inf_rigid(A, p, threshold=THRESHOLD_SV):
    return


def rigidity_eigenvalue(A, p):
    return


def minimum_rigidity_extents(geodesics, p, threshold=THRESHOLD_SV):
    """
    Requires:
    ---------
        framework is rigid
    """
    return
