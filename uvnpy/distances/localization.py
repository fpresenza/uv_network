#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date lun sep 13 20:27:17 -03 2021
"""
import numpy as np

np.set_printoptions(precision=10)


class DecentralizedLocalization(object):
    def __init__(self, xi, ti=0.):
        self.init(xi, ti)
        self.dim = len(xi)

    @property
    def position(self):
        return self._x.copy()

    def init(self, xi, ti=0.):
        self.t = ti
        self._x = xi.copy()


class GradientBasedFilter(DecentralizedLocalization):
    def __init__(self, xi, ti, stepsize, W):
        """
        Gradient Descent of the cost functoin:
            V(x) = sum_{ij in E} (wij/2)*(||xi-xj|| - z_obs)^2 +
                (1/2)*sum_{i in V} ||x_i - f(x_i, u_i)||_{W_i}^2

        args:
        -----
            xi       : initial state
            ti       : initial time
            stepsize : stepsize gradient descent
            W        : dynamic model weigth
        """
        super(GradientBasedFilter, self).__init__(xi, ti)
        self.alpha = stepsize
        self.W = W

    def dynamic_model(self, dt, u):
        """
        Integrator.

        args:
        -----
            dt : time elapsed since last update
            u  : control action
        """
        x = self._x + u * dt
        return x

    def distances_model(self, z, xj):
        """
        Distance with neighbors.

        args:
        -----
            z  : measurements
            xj : neighbors states
        """
        r = self._x - xj
        d = np.sqrt(np.square(r).sum(axis=1))
        m = (z - d)/d
        g = sum(r * m[:, None])
        return g

    def square_distances_model(self, z, xj):
        """
        Square distance with neighbors.

        args:
        -----
            z  : measurements
            xj : neighbors states
        """
        r = self._x - xj
        d2 = np.square(r).sum(axis=1)
        m = (z - d2)
        g = sum(r * m[:, None])
        return g

    def dynamic_step(self, t, u, *args):
        """
        Dynamic model step.

        args:
        -----
            t : current time
            u : control action
        """
        dt = t - self.t
        self.t = t
        x = self.dynamic_model(dt, u, *args)
        self._x -= self.W.dot(self._x - x)

    def distances_step(self, z, xj, *args):
        """
        Measurement model step.

        args:
        -----
            z  : measurements
            xj : neighbors states

        """
        g_obs = self.distances_model(z, xj, *args)
        self._x += self.alpha * g_obs

    def square_distances_step(self, z, xj, *args):
        """
        Measurement model step.

        args:
        -----
            z  : measurements
            xj : neighbors states

        """
        g_obs = self.square_distances_model(z, xj, *args)
        self._x += self.alpha * g_obs

    def gps_step(self, z):
        """
        Position measurements model step.

        args:
        -----
            z : measurements
        """
        self._x += self.alpha * (z - self._x)


class KalmanBasedFilter(DecentralizedLocalization):
    def __init__(self, xi, Pi, Qu, Rd, Rp, ti=0.):
        """
        Kalman Filter without cross-correlations.

        args:
            xi : intial state
            Pi : initial covariance
            Qu : control step covariance
            Rd : distance measurement variance
            Rp : gps covariance
        """
        super(KalmanBasedFilter, self).__init__(xi, ti)
        self._P = Pi.copy()
        self.ctrl_cov = Qu
        self.range_cov = Rd
        self.square_range_cov = Rd**2
        self.gps_cov = Rp * np.eye(self.dim)

    @property
    def covariance(self):
        return self._P.copy()

    def dynamic_model(self, dt, u):
        """
        Integrator.

        args:
        -----
            dt : time elapsed since last update
            u  : control action
        """
        x = self._x + u * dt
        F = np.eye(len(x))
        Q = self.ctrl_cov * dt
        return x, F, Q

    def distances_model(self, z, xj, Pj):
        """
        Distance with neighbors.

        args:
        -----
            z  : measurements
            xj : neighbors states
            Pj : covariance associated to each neighbor
        """
        r = self._x - xj
        d = np.sqrt(np.square(r).sum(axis=-1, keepdims=True))
        d = d.ravel()
        dz = z.ravel() - d
        H = r / d[:, None]
        Rj = np.matmul(H[:, None], np.matmul(Pj, H[:, :,  None]))
        R = np.diag((Rj + self.range_cov).ravel())
        return dz, H, R

    def square_distances_model(self, z, xj, Pj):
        """
        Square distance with neighbors.

        args:
        -----
            z  : measurements
            xj : neighbors states
            Pj : covariance associated to each neighbor
        """
        r = self._x - xj
        d2 = np.square(r).sum(axis=-1, keepdims=True)
        d2 = d2.ravel()
        dz = z.ravel() - d2
        H = 2 * r
        Rj = np.matmul(H[:, None], np.matmul(Pj, H[:, :,  None]))
        R = np.diag((Rj + self.square_range_cov).ravel())
        return dz, H, R

    def dynamic_step(self, t, u, *args):
        """
        Dynamic model step.

        args:
        -----
            t : current time
            u : control action
        """
        dt = t - self.t
        self.t = t

        x, F, Q = self.dynamic_model(dt, u, *args)
        self._x[:] = x
        self._P[:] = F.dot(self._P).dot(F.T) + Q

    def distances_step(self, z, xj, Pj, *args):
        """
        Distance measurements model step.

        args:
        -----
            z  : measurements
            xj : neighbors states
            Pj : weigth associated to each neighbor

        """
        dz, H, R = self.distances_model(z, xj, Pj, *args)
        Pz = H.dot(self._P).dot(H.T) + R
        K = self._P.dot(H.T).dot(np.linalg.inv(Pz))

        self._x += K.dot(dz)
        self._P -= K.dot(H).dot(self._P)

    def square_distances_step(self, z, xj, Pj, *args):
        """
        Distance measurements model step.

        args:
        -----
            z  : measurements
            xj : neighbors states
            Pj : weigth associated to each neighbor

        """
        dz, H, R = self.square_distances_model(z, xj, Pj, *args)
        Pz = H.dot(self._P).dot(H.T) + R
        K = self._P.dot(H.T).dot(np.linalg.inv(Pz))

        self._x += K.dot(dz)
        self._P -= K.dot(H).dot(self._P)

    def gps_step(self, z):
        """
        Position measurements model step.

        args:
        -----
            z : measurements
        """
        dz = z - self._x
        Pz = self._P + self.gps_cov
        K = self._P.dot(np.linalg.inv(Pz))
        self._x += K.dot(dz)
        self._P -= K.dot(self._P)
