#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar feb 16 11:17:50 -03 2021
"""
import numpy as np


def repulsion(p):
    """Costo de repulsión entre agentes puntuales.

    El costo es igual a la suma de los cuadrados de la
    inversa de las distancias entre agentes.
    """
    r = p[..., None, :] - p[..., None, :, :]
    d2 = np.square(r).sum(axis=-1)
    d2[d2 != 0] **= -1
    return d2.sum()
