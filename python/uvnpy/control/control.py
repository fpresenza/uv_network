#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue Aug 12 14:25:16 2020
@author: fran
"""
from gpsic.controladores.mpc import MPC, Control
from uvnpy.sensor.rango import Rango


class MPCInformativo(MPC):
    """
    Esta clase sirve para implementar un controlador MPC para maximizar la
    ganancia de información de un filtro extendido de Kalman, minimizando:
        - esfuerzo de control: u y delta_u
        - determinante de la matriz de información
     """
    def __init__(self, lamda, Q, modelo, **kwargs):
        costos = (
            {'fun': Control.effort,    'lamda': lamda[0], 'Q': Q[0]},
            {'fun': Control.slew,      'lamda': lamda[1], 'Q': Q[1]},
            {'fun': Rango.collection_matrix_det,  'lamda': lamda[2], 'Q': Q[2]}
        )
        super(MPCInformativo, self).__init__(costos, modelo, **kwargs)

    def update(self, x, t, args, ineq_args=(), eq_args=(), **kwargs):
        u = super(MPCInformativo, self).update(
          x, t, ([], [self.u], args), ineq_args=(), eq_args=(), **kwargs
        )
        return u
