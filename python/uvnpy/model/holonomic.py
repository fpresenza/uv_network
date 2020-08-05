#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue May 06 11:15:10 2020
@author: fran
"""
import numpy as np
from gpsic.modelos.discreto import SistemaDiscreto

class Integrador(SistemaDiscreto):
    """ Esta clase implementa un modelo de cinemática de posición y 
    velocidad, con control por realimentación de estados. """
    def __init__(self, ti=0., dof=1, **kwargs):
        """ Para inicializar una instancia se debe ingresar:
        dof: nro. de grados de libertad del robot
        Los parametros del modelo de error de la dinamica: 
        sigma: std. dev. en la entrada del integrador
        """
        pi = kwargs.get('pi',  np.zeros(dof))
        vi = kwargs.get('vi',  np.zeros(dof))
        self._v = vi
        self.sigma = kwargs.get('sigma', np.zeros(dof))
        super(Integrador, self).__init__(ti=ti, xi=pi)
        self.dof = dof
        #   matrices del sistema
        self.G_x = np.zeros([dof, dof])
        self.G_r = self.G_z = np.identity(dof)
        #   matriz de covarianza del ruido
        self.Q = np.diag(np.square(self.sigma))
    
    @property
    def p(self):
        return self._x.copy()
    
    @property
    def v(self):
        return self._v.copy()
    
    def dinamica(self, x, t, r):
        z = np.random.normal(0, self.sigma)
        dot_x = np.matmul(self.G_r, r) + np.matmul(self.G_z, z)
        self._v = dot_x
        return dot_x

    def salida(self, x, t, r):
        return x


class ControlVelocidad(SistemaDiscreto):
    """ Esta clase implementa un modelo de cinemática de posición y 
    velocidad, con control por realimentación de estados. """
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
        super(ControlVelocidad, self).__init__(ti=ti, xi=np.hstack([pi, vi]))
        self.dof = dof
        self._v = vi
        self.a = np.zeros(dof)
        self.sigma = kwargs.get('sigma', np.zeros(2*dof))
        I = np.identity(dof)
        O = np.zeros_like(I)
        #   controller matrix
        K = np.copy(kwargs.get('ctrl_matrix', I))
        #   matrices del sistema
        self.G_x = np.block([[O,  I],
                             [O, -K]])
        self.G_r = np.block([[O],
                             [K]])
        self.G_z = np.block([[O,  O],
                             [I, -K]])
        #   matriz de covarianza del ruido
        self.Q = np.diag(np.square(self.sigma))

    @property
    def p(self):
        return self._x[:2].copy()
    
    @property
    def v(self):
        return self._x[2:].copy()
        
    def dinamica(self, x, t, r):
        z = np.random.normal(0, self.sigma)
        dot_x = np.matmul(self.G_x, x) + np.matmul(self.G_r, r) + np.matmul(self.G_z, z)
        self.a = dot_x[self.dof:]
        return dot_x

    def salida(self, x, t, r):
        return x
        

class VelocityRandomWalk(SistemaDiscreto):
    """ Esta clase simula un proceso aleatorio de Wiener
    """
    def __init__(self, ti=0., dof=1, **kwargs):
        pi = kwargs.get('pi', np.zeros(dof))
        vi = kwargs.get('vi', np.zeros(dof))
        self.dof = dof
        self.p = pi
        self.v = vi
        self.sigma = kwargs.get('sigma', np.zeros(dof))
        I = np.identity(dof)
        O = np.zeros_like(I)
        #   matrices del sistema
        self.G_x = np.block([[O, I],
                             [O, O]])
        self.G_z = np.block([[O],
                             [I]])
        #   matriz de covarianza del ruido e
        self.Q = np.diag(np.square(self.sigma))

    def dinamica(self, x, t, u):
        #        self.G_x @ x + self.G_z @ e
        e = np.random.normal(0, self.sigma)
        return np.hstack([x[self.dof:], e])

    def salida(self, x, t, u):
        self.p, self.v = x[:self.dof], x[self.dof:]
        return x

    def f(self, x, t, u):
        """ Esta función retorna las variables necesarias para el paso de 
        predicción de un EKF """
        dot_x = np.matmul(self.G_x, x)
        return dot_x, self.G_x, self.G_z, self.Q