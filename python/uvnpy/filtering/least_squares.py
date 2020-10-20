#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date jue oct 22 14:18:35 -03 2020
"""
import numpy as np
from numpy.linalg import multi_dot, inv  # noqa


def normalizar_pesos(pesos):
    """ Devuelve un vector de pesos que suman 1. """
    return np.divide(pesos, sum(pesos))


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


def weigthed_least_squares(muestras, pesos):
    """ Estimador de cuadrados mínimos de una variable aleatoria

    A partir de una secuencia de muestras independientes
    con pesos, obtiene media y covarianza estimada.

    Argumentos:
        muestras = (m_1, ..., m_n)
        pesos = (W_1, ..., W_n)
    """
    inv_sum_Wi = inv(sum(pesos))
    sum_Wi_xi = sum([np.matmul(W, x) for x, W in zip(muestras, pesos)])
    media = np.matmul(inv_sum_Wi, sum_Wi_xi)
    error = np.subtract(muestras, media)
    sum_Wi_ei2 = sum(
      [np.matmul(W, np.outer(e, e)) for e, W in zip(error, pesos)]
    )
    cov = np.matmul(inv_sum_Wi, sum_Wi_ei2)
    return media, cov
