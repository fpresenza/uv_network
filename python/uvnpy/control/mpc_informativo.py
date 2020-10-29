#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue Aug 12 14:25:16 2020
@author: fran
"""
from numpy.linalg import det

from gpsic.controladores.mpc import MPC, Control

from uvnpy.sensores import rango


def det_delta_informacion(u, y_p, Q, landmarks, sigma):
    cm_set = [rango.delta_informacion_sum(y, landmarks, sigma) for y in y_p]
    return sum([det(cm) for cm in cm_set])


class controlador(MPC):
    def __init__(self, metrica, lamda, Q, modelo, **kwargs):
        """Controlador MPC

        Costos:
            esfuerzo de control.
            métrica del filtro.
        """
        costos = (
            {'fun': Control.effort,    'lamda': lamda[0], 'Q': Q[0]},
            {'fun': Control.slew,      'lamda': lamda[1], 'Q': Q[1]},
            {'fun': metrica,  'lamda': lamda[2], 'Q': Q[2]}
        )
        super(controlador, self).__init__(costos, modelo, **kwargs)

    def update(self, x, t, args, ineq_args=(), eq_args=(), **kwargs):
        u = super(controlador, self).update(
          x, t, ([], [self.u], args), ineq_args=(), eq_args=(), **kwargs
        )
        return u
