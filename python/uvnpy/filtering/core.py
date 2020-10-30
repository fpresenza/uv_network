#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date vie oct 30 09:37:21 -03 2020
"""
import numpy as np
from numpy.linalg import inv


def normalizar_pesos(pesos):
    """ Devuelve un vector de pesos que suman 1. """
    return np.divide(pesos, sum(pesos))


def similaridad(x, P):
    """ Transformación de similaridad.

    Transforma un vector y una matriz p.d. entre el espacio
    de los estados y el espacio de la información, en ambos
    sentidos.
    """
    Pinv = inv(P)
    y = np.matmul(Pinv, x)
    return y, Pinv


def ajustar_sigma(dic):
    """ Ajusta la desv. std. a la frecuencia de muestreo. """
    f = dic.pop('freq')
    sigma = dic.get('sigma')
    if f is not None:
        dic.update(sigma=np.multiply(f**0.5, sigma))
    return dic
