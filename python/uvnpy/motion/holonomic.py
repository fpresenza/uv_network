#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue May 06 11:15:10 2020
@author: fran
"""
import numpy as np
from uvnpy.motion.dynamic import DynamicModel

class VelocityModel(DynamicModel):
    """ Esta clase implementa un modelo de cinemática de control
    de velocidad de un robot. """
    def __init__(self, *args, **kwargs):
        """ Para inicializar una instancia se debe ingresar:
        dof: nro. de grados de libertad del robot
        ctrl_gain: controlador interno del modelo de movimiento
        Los parametros del modelo de error de la dinamica: 
        sigma[0]: std. dev. en la accion de los actuadores
        sigma[1]: std. dev. en la medicion interna de la velocidad
        """
        self.dof = kwargs.get('dof', 1)
        pi = kwargs.get('pi',  np.zeros((self.dof, 1)))
        vi = kwargs.get('vi',  np.zeros((self.dof, 1)))
        kwargs['xi'] = np.vstack((vi, pi))
        kwargs['r'] = np.zeros((self.dof, 1))
        kwargs['u'] = np.zeros((self.dof, 1))
        super(VelocityModel, self).__init__(*args, **kwargs)
        self.ctrl_gain = kwargs.get('ctrl_gain', 1.)
        I = np.eye(self.dof)
        Z = np.zeros_like(I)
        #   controller matrix
        self.K = np.diag(np.broadcast_to(self.ctrl_gain, self.dof))
        #   state transition matrix
        self.F_x = np.block([[-self.K, Z],
                             [      I, Z]])
        #   input transition matrix
        self.F_r = np.block([[self.K],
                             [     Z]])
        #   disturbances
        self.B = np.block([[I, -self.K],
                           [Z,       Z]])
        #   noise matrix
        self.Q = np.diag(np.square(self.sigma))

    def __str__(self):
        return 'VelocityModel(dof={})'.format(self.dof)

    def fmat(self, x, r, **kwargs):
        return self.F_x, self.B, self.Q

    def f(self, x, r, **kwargs):
        """ This function represents the dynamics of the closed loop model """
        self.r = r
        v_meas = x[:3] + self.e[3:]
        self.u = np.dot(self.K, r-v_meas)
        return np.dot(self.F_x, x) + np.dot(self.F_r, r)
        
    def dot_x(self, x, r, **kwargs):
        """ This function takes reference r and returns
        a state derivative noisy sample """
        return self.f(self.x, r, **kwargs) + np.dot(self.B, self.e)

    def accel(self):
        return self.u + self.e[:3]


class VRW(object):
    """ This class emulates a Wiener proccess dynamic
    """
    def __init__(self, **kwargs):
        self.rate = kwargs.get('rate', 1.)
        NSD = kwargs.get('NSD', 0.)
        self.agents = kwargs.get('agents', 1)
        self.dim = kwargs.get('dim', 1)
        self.N = self.agents * self.dim  
        BW = 0.5 * self.rate
        self.sigma = NSD * np.sqrt(2*BW) * 0.01
        self.Q = np.diag(np.repeat(self.sigma**2, self.N))
    
    def __len__(self):
        return 2*self.N

    def f(self, mean, u):
        u = mean[:self.N]
        dot_v = np.random.normal(0., self.sigma * np.ones_like(u))
        dot_x = u
        dot_x = np.vstack((dot_v, dot_x)) 
        I = np.eye(self.N)
        Z = np.zeros_like(I)
        F_x = np.block([[Z, Z],
                        [I, Z]])
        B = np.block([[I],
                      [Z]])
        return dot_x, F_x, B, self.Q