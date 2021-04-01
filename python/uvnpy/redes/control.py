#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mié mar 31 20:46:03 -03 2021
"""
import numpy as np
from transformations import unit_vector


def edge_distance_potencial_gradient(A, p):
    """Gradiente de un potencial función de la distancia de los enlaces.

    Devuelve el gradiente de

            V = sum_{ij in E} V_{ij},
            V_{ij} es función de d_{ij} = ||x_i - x_j||.

    A es la matriz de adyacencia donde cada componente
    A[i, j] = partial{V_{ij}} / partial{d_{ij}} es la derivada
    de V_{ij} respecto de la distancia. Si A[i, j] = 0, los nodos
    (i, j) no están conectados.

    args:
        A: matriz de adyacencia (nv, nv)
        p: array de posiciones (nv, dof)
    """
    nv, dof = p.shape
    r = unit_vector(p[:, None] - p, axis=-1)
    ii = np.diag([True] * nv)
    r[ii] = 0
    r *= A[..., None]               # aplicar pesos
    grad = r.sum(1)
    return grad
