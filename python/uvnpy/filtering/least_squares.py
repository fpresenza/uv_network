#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date jue oct 22 14:18:35 -03 2020
"""
import numpy as np
from numpy.linalg import multi_dot, inv, pinv  # noqa

from . import normalizar_pesos


def wlstsq(A, b, Q=None):
    """ Cuadrados mínimos con pesos.

    La matriz de coeficientes A debe tener columnas
    linealmente independientes
    """
    if Q is None:
        Q = np.identity(b.size)
    P = inv(multi_dot([A.T, Q, A]))
    A_pinv = multi_dot([P, A.T, Q])
    x = np.matmul(A_pinv, b)
    return x, P


def ajustar_gaussiana(muestras, pesos=None):
    """ Estimador de media y covarianza

    A partir de una secuencia de muestras independientes,
    estima media y covarianza.

    Argumentos:
        muestras = (m_1, ..., m_n)
        pesos = (p_1, ..., p_n)
    """
    N = len(muestras)
    if pesos is None:
        pesos = np.ones(len(muestras))
    pesos = normalizar_pesos(pesos)
    media = np.matmul(pesos, muestras)
    error = np.subtract(muestras, media)
    cov = sum([p * np.outer(e, e) for p, e in zip(pesos, error)]) * N / (N - 1)
    return media, cov
