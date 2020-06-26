#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue May 06 11:15:10 2020
@author: fran
"""
import numpy as np
from uvnpy.model.discrete import DiscreteModel

class VelocityModel(DiscreteModel):
    """ Esta clase implementa un modelo de cinemática de control
    de velocidad de un robot. """
    def __init__(self, ti=0., dof=1, **kwargs):
        """ Para inicializar una instancia se debe ingresar:
        dof: nro. de grados de libertad del robot
        ctrl_gain: controlador interno del modelo de movimiento
        Los parametros del modelo de error de la dinamica: 
        sigma[:dof]: std. dev. en la accion de los actuadores
        sigma[dof:]: std. dev. en la medicion interna de la velocidad
        """
        pi = kwargs.get('pi',  np.zeros(dof))
        vi = kwargs.get('vi',  np.zeros(dof))
        super(VelocityModel, self).__init__(ti=ti, xi=np.hstack([pi, vi]))
        self.dof = dof
        self.a = np.zeros(dof)
        self.sigma = kwargs.get('sigma', np.zeros(2*dof))
        I = np.eye(dof)
        Z = np.zeros_like(I)
        #   controller matrix
        K = np.copy(kwargs.get('ctrl_matrix', I))
        #   state transition matrix
        self.F_x = np.block([[Z,  I],
                             [Z, -K]])
        #   input transition matrix
        self.F_r = np.block([[Z],
                             [K]])
        #   disturbances
        self.B = np.block([[Z,  Z],
                           [I, -K]])
        #   noise matrix
        self.Q = np.diag(np.square(self.sigma))

    def __str__(self):
        return 'VelocityModel(dof={})'.format(self.dof)

    def f(self, x, r):
        """ This function represents the dynamics of the closed loop model """
        dot_x = np.matmul(self.F_x, x) + np.matmul(self.F_r, r)
        return dot_x, self.F_x, self.B, self.Q
        
    def dot_x(self, x, r, t):
        """ This function takes reference r and returns
        a state derivative noisy sample """
        e = np.random.normal(0, self.sigma)
        dot_x = np.matmul(self.F_x, x) + np.matmul(self.F_r, r) + np.matmul(self.B, e)
        self.a = dot_x[self.dof:]
        return dot_x


class VelocityRandomWalk(DiscreteModel):
    """ This class emulates a Wiener proccess dynamic
    """
    def __init__(self, ti=0., dim=1, sigma=1.):
        self.dim = dim
        self.sigma = np.ones(dim)*sigma
        super(VelocityRandomWalk, self).__init__(ti=ti, xi=np.zeros(2*dim))
        self.Q = np.diag(np.square(self.sigma))
        I = np.eye(dim)
        Z = np.zeros_like(I)
        self.F_x = np.block([[Z, I],
                             [Z, Z]])
        self.B = np.block([[Z],
                           [I]])

    def dot_x(self, x, u, t):
        #        self.F_x @ x + self.B @ e
        e = np.random.normal(0, self.sigma)
        return np.hstack([x[self.dim:], e])