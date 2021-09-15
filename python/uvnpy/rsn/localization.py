#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date lun sep 13 20:27:17 -03 2021
"""
import numpy as np


class decentralized_localization(object):
    def __init__(self, xi, ti=0.):
        """Solo toma un numpy.array como vector de estados x"""
        self.init(xi, ti)

    @property
    def x(self):
        return self._x.copy()

    def init(self, xi, ti=0.):
        self.t = ti
        self._x = xi.copy()
        self.xj = None
        self.wj = None

    def update_neighbors(self, xj, wj):
        """actualizar estado y pesos asociado a los vecinos"""
        self.xj = xj
        self.wj = wj


class distances_to_neighbors_gradient(decentralized_localization):
    def __init__(self, xi, ti, stepsize, W):
        """Filtro de gradiente descendiente.

        xi: estado inicial del nodo i
        ti: tiempo inicial
        stepsize: stepsize gradient descent
        W: peso relativo de la dinamica

        Funcion objetivo:
        V(x) = sum_{edges} (wij/2)*(||xi-xj|| - z_obs)^2 +
            (1/2)*sum_{nodes} ||x - f(x, u)||_{W_i}^2
        """
        super(distances_to_neighbors_gradient, self).__init__(xi, ti)
        self.alpha = stepsize
        self.W = W

    def dynamic_model(self, dt, u):
        """Modelo integrador

        dt: paso de tiempo
        u: accion de control
        w: peso de imporancia relativa
        """
        x = self._x + u * dt
        return x

    def observation_model(self, z):
        """Observacion de distancia con vecinos

        z: mediciones
        xj: posicion de vecinos
        wj: peso de imporancia asociado a cada medicion
        """
        r = self._x - self.xj
        d = np.sqrt(np.square(r).sum(axis=1))
        m = self.wj * (z - d)/d
        g = sum(r * m[:, None])
        return g

    def step(self, t, u, z, dyn_args=(), obs_args=()):
        """Paso del gradiente descendiente.

        t: instante de tiempo
        u: accion de control
        z: observaciones
        """
        dt = t - self.t
        self.t = t
        x_dyn = self.dynamic_model(dt, u, *dyn_args)
        g_obs = self.observation_model(z, *obs_args)
        self._x += self.alpha * (g_obs - self.W.dot(self._x - x_dyn))


class distances_to_neighbors_kalman(decentralized_localization):
    def __init__(self, xi, Pi, Q, R, ti=0.):
        """Filtro de kalman sin correlación cruzada."""
        super(distances_to_neighbors_kalman, self).__init__(xi, ti)
        self._P = Pi.copy()
        self.Q = Q
        self.R = R

    @property
    def P(self):
        return self._P.copy()

    def dynamic_model(self, dt, u):
        """Modelo integrador

        dt: paso de tiempo
        u: accion de control
        w: peso de imporancia relativa
        """
        x = self._x + u * dt
        F = np.eye(len(x))
        return x, F, self.Q

    def observation_model(self, z):
        """Observacion de distancia con vecinos

        z: mediciones
        xj: posicion de vecinos
        wj: peso de imporancia asociado a cada medicion
        """
        r = self._x - self.xj
        d = np.sqrt(np.square(r).sum(axis=1))
        dz = z - d
        H = r / d[:, None]
        Rj = np.matmul(H[:, None], np.matmul(self.wj, H[:, :,  None]))
        R = np.diag((Rj + self.R).ravel())
        return dz, H, R

    def step(self, t, u, z, dyn_args=(), obs_args=()):
        """Paso del gradiente descendiente.

        t: instante de tiempo
        u: accion de control
        z: observaciones
        """
        dt = t - self.t
        self.t = t

        # prediccion
        x, F, Q = self.dynamic_model(dt, u, *dyn_args)
        self._x = x
        self._P = F.dot(self._P).dot(F.T) + Q

        dz, H, R = self.observation_model(z, *obs_args)
        Pz = H.dot(self._P).dot(H.T) + R
        K = self._P.dot(H.T).dot(np.linalg.inv(Pz))

        self._x += K.dot(dz)
        self._P -= K.dot(H).dot(self._P)
