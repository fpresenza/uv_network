#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue Aug 12 14:25:16 2020
@author: fran
"""
from gpsic.controladores.mpc import MPC, esfuerzo_control, incremento_control


class minimizar(MPC):
    def __init__(self, metrica, matriz, modelo, Q, **kwargs):
        self.metrica = metrica
        self.matriz = matriz
        costos = (
            {'fun': esfuerzo_control, 'Q': Q[0]},
            {'fun': incremento_control, 'Q': Q[1]},
            {'fun': self.informacion, 'Q': Q[2]}
        )
        super(minimizar, self).__init__(modelo, costos, **kwargs)

    def informacion(self, u, x_p, Q, *args):
        M = sum([self.matriz(x, *args) for x in x_p])
        return Q * self.metrica(M)

    def update(self, x, t, args, ineq_args=(), eq_args=(), **kwargs):
        u = super(minimizar, self).update(
          x, t, ([], [self.u], args), ineq_args=(), eq_args=(), **kwargs
        )
        return u
