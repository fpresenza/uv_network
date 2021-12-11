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
    def position(self):
        return self._x.copy()

    def init(self, xi, ti=0.):
        self.t = ti
        self._x = xi.copy()


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

    def observation_model(self, z, xj, Pj):
        """Observacion de distancia con vecinos

        z: mediciones
        xj: posicion de vecinos
        wj: peso de imporancia asociado a cada medicion
        """
        r = self._x - xj
        d = np.sqrt(np.square(r).sum(axis=1))
        m = Pj * (z - d)/d
        g = sum(r * m[:, None])
        return g

    def prediction_step(self, t, u, *args):
        """Paso del gradiente descendiente.

        t: instante de tiempo
        u: accion de control
        z: observaciones
        """
        dt = t - self.t
        self.t = t
        x = self.dynamic_model(dt, u, *args)
        self._x -= self.W.dot(self._x - x)

    def correction_step(self, z, xj, Pj, *args):
        g_obs = self.observation_model(z, xj, Pj, *args)
        self._x += self.alpha * g_obs


class distances_to_neighbors_kalman(decentralized_localization):
    def __init__(self, xi, Pi, Qu, Rz, ti=0.):
        """Filtro de kalman sin correlacion cruzada."""
        super(distances_to_neighbors_kalman, self).__init__(xi, ti)
        self._P = Pi.copy()
        self.ctrl_cov = Qu
        self.range_cov = Rz

    @property
    def covariance(self):
        return self._P.copy()

    def dynamic_model(self, dt, u):
        """Modelo integrador

        dt: paso de tiempo
        u: accion de control
        """
        x = self._x + u * dt
        F = np.eye(len(x))
        Q = self.ctrl_cov * dt
        return x, F, Q

    def observation_model(self, z, xj, Pj):
        """Observacion de distancia con vecinos

        z: mediciones
        """
        r = self._x - xj
        d = np.sqrt(np.square(r).sum(axis=-1, keepdims=True))
        d = d.ravel()
        dz = z.ravel() - d
        H = r / d[:, None]
        Rj = np.matmul(H[:, None], np.matmul(Pj, H[:, :,  None]))
        R = np.diag((Rj + self.range_cov).ravel())
        return dz, H, R

    def prediction_step(self, t, u, *args):
        """
        t: instante de tiempo
        u: accion de control
        """
        dt = t - self.t
        self.t = t

        x, F, Q = self.dynamic_model(dt, u, *args)
        self._x[:] = x
        self._P[:] = F.dot(self._P).dot(F.T) + Q

    def correction_step(self, z, xj, Pj, *args):
        """
        z: mediciones
        """
        dz, H, R = self.observation_model(z, xj, Pj, *args)
        Pz = H.dot(self._P).dot(H.T) + R
        K = self._P.dot(H.T).dot(np.linalg.inv(Pz))

        self._x += K.dot(dz)
        self._P -= K.dot(H).dot(self._P)
