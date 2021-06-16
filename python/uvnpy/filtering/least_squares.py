#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date jue oct 22 14:18:35 -03 2020
"""
import numpy as np


def wlstsq(A, b, Q=None):
    """Cuadrados mínimos con pesos.

    A * x = b

    La matriz de coeficientes A debe tener columnas
    linealmente independientes
    """
    if Q is None:
        Q = np.identity(b.size)
    At = A.T
    P = np.linalg.inv(At.dot(Q).dot(A))
    A_pinv = P.dot(At).dot(Q)
    x = A_pinv.dot(b)
    return x, P
