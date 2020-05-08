#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue May 06 11:15:10 2020
@author: fran
"""
import numpy as np

class VelocityModel(object):
    """ Esta clase implementa un modelo de cinemática de control
    de velocidad de un robot. """
    def __init__(self, *args, **kwargs):
        """ Para inicializar una instancia se debe ingresar:
        dof: nro. de grados de libertad del robot
        ctrl_gain: controlador interno del modelo de movimiento
        Los parametros del modelo de error de la dinamica: 
        alpha[0]: std. dev. en la accion de los actuadores
        alpha[1]: std. dev. en la medicion interna de la velocidad
        """
        self.dof = kwargs.get('dof', 3)
        self.ti = kwargs.get('ti', 0.)
        self.time = self.ti
        self.vi = kwargs.get('vi', np.zeros((self.dof, 1)))
        self.xi = kwargs.get('xi', np.zeros((self.dof, 1)))
        self.X = np.vstack((self.vi, self.xi))
        self.ctrl_gain = kwargs.get('ctrl_gain', 1.)
        self.alphas = kwargs.get('alphas', [[0.,0.,0.],[0.,0.,0.]])
        I = np.eye(self.dof)
        Z = np.zeros_like(I)
        #   controller matrix
        self.K = np.diag(np.broadcast_to(self.ctrl_gain, self.dof))
        #   state transition matrix
        self.G_X = np.block([[-self.K, Z],
                             [      I, Z]])
        #   input transition matrix
        self.G_v = np.block([[self.K],
                             [     Z]])
        #   disturbances
        self.B = np.block([[I, -self.K],
                           [Z,       Z]])
        #   noise matrix
        self.Q = np.diag(np.square(self.alphas[0]+self.alphas[1]))

    def __str__(self):
        return 'VelocityModel(dof={})'.format(self.dof)

    def restart(self):
        """ Return all values to initial conditions. """
        self.time = self.ti
        self.X = np.vstack((self.vi, self.xi))
        
    def step(self, v, t):
        """Implementa un sistema de doble integrador con lazo 
        cerrado en velocidad. Como parametros toma comandos de velocidad,
        el tiempo actual en segundos. Devuelve una muestra de la futura 
        posicion y velocidad en base al modelo de perturbacion.
        """
        Ts = t - self.time
        self.time = t
        #   noise in control actuators
        n1 = np.random.normal(0., np.array(self.alphas[0]).reshape(-1,1))
        #   noise in internal velocity sensing 
        n2 = np.random.normal(0., np.array(self.alphas[1]).reshape(-1,1))
        #   Discrete state matrix
        Ad = np.eye(*self.G_X.shape) + Ts*self.G_X
        #   Discrete state-input matrix
        Bd = Ts*np.block([self.G_v, self.B])
        #   system input
        u = np.block([[ v],
                      [n1],
                      [n2]])
        #   New estimate
        self.X = np.dot(Ad, self.X) + np.dot(Bd, u)        
        v_meas = self.X[:3] + n2
        accel = np.dot(self.K, v-v_meas) + n1
        return accel

    def f(self, mean, v):
        """ This function takes mean and covariance of a gaussian
        proccess, an input and generates values needed for prediction."""
        dot_X = np.dot(self.G_X, mean) + np.dot(self.G_v, v) 
        return dot_X, self.G_X, self.B, self.Q


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
        v = mean[:self.N]
        dot_v = np.random.normal(0., self.sigma * np.ones_like(v))
        dot_x = v
        dot_X = np.vstack((dot_v, dot_x)) 
        I = np.eye(self.N)
        Z = np.zeros_like(I)
        F_X = np.block([[Z, Z],
                        [I, Z]])
        B = np.block([[I],
                      [Z]])
        return dot_X, F_X, B, self.Q