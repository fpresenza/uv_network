#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Tue Jan 14 16:15:16 2020
@author: fran
"""
import numpy as np
import collections
import matplotlib.pyplot as plt


Logs = collections.namedtuple('Logs', 't mean cov innovation')


def integrator(dt, x, u, Q):
    x = x + u * dt
    F = np.identity(len(x))
    return x, F, Q


def gps_model(x, R):
    hat_z = x
    H = np.identity(len(x))
    return hat_z, H, R


def covariance_prediction(P, F, Q):
    return F.dot(P).dot(F.T) + Q


def covariance_correction(P, H, R):
    Pz_inv = np.linalg.inv(H.dot(P).dot(H.T) + R)
    K = P.dot(H.T).dot(Pz_inv)
    return P - K.dot(H).dot(P)


def two_step_covariance(P, F, Q, H, R):
    P = F.dot(P).dot(F.T) + Q
    Pz_inv = np.linalg.inv(H.dot(P).dot(H.T) + R)
    K = P.dot(H.T).dot(Pz_inv)
    return P - K.dot(H).dot(P)


class kalman(object):
    def __init__(self, xi, Pi, ti=0.):
        """Filtros de Kalman"""
        self.init(xi, Pi, ti)
        self.logs = Logs(
            t=[],
            mean=[],
            cov=[],
            innovation={}
        )
        self.save_data()

    @property
    def x(self):
        return self._x.copy()

    @property
    def P(self):
        return self._P.copy()

    def init(self, xi, Pi, ti=0.):
        self.t = ti
        self._x = xi.copy().reshape(-1, 1)
        self._P = Pi.copy()

    def save_data(self):
        self.logs.t.append(self.t)
        self.logs.mean.append(self.x)
        self.logs.cov.append(self.P)

    def summary(self):
        _innovation = self.logs.innovation.copy()
        logs = Logs(
            t=np.array(self.logs.t),
            mean=np.array(self.logs.mean),
            cov=np.array(self.logs.cov),
            innovation=np.empty((len(_innovation), 3), dtype=np.ndarray)
        )
        for i, (key, val) in enumerate(_innovation.items()):
            logs.innovation[i, :2] = key
            logs.innovation[i, 2] = val

        return logs

    def prediction(self, f, t, *args):
        """Paso de prediccion

        args:
            t: tiempo
        """
        dt = t - self.t
        self.t = t
        x, F, Q = f(dt, self._x, *args)
        self._x = x
        self._P = F.dot(self._P).dot(F.T) + Q


class KF(kalman):
    def __init__(self, xi, Pi, ti=0.):
        """Filtro de Kalman en forma clasica. """
        super(KF, self).__init__(xi, Pi, ti)

    def correction(self, h, z, *args):
        """Paso de correccion

        args:
            h: modelo de sensor
            z: medicion
        """
        hat_z, H, R = h(self._x, *args)
        dz = z - hat_z
        Pz = H.dot(self._P).dot(H.T) + R
        Pz_inv = np.linalg.inv(Pz)
        K = self._P.dot(H.T).dot(Pz_inv)

        self._x = self._x + K.dot(dz)
        self._P = self._P - K.dot(H).dot(self._P)

        self.logs.innovation[self.t, h.__name__] = dz.copy()


class KFi(kalman):
    def __init__(self, xi, Pi, ti=0.):
        """Filtro de Kalman en forma alternativa. """
        super(KFi, self).__init__(xi, Pi, ti)

    def correction(self, dy, Y):
        """Paso de correccion

        args:

            dy: innovacón en espacio de informacion
            Y: matriz de innovacion
        """
        x, P = self._x, self._P
        I_prior = np.linalg.inv(P)
        self._P = np.linalg.inv(I_prior + Y)
        self._x = x + P.dot(dy)


class KCFE(kalman):
    def __init__(self, xi, Pi, ti=0.):
        """Filtro de Kalman por Consenso en Estimaciones

        Ver:
            Olfati-Saber,
            ''Kalman-Consensus Filter: Optimality
              Stability and Performance'',
            IEEE Conference on Decision and Control (2009).
        """
        super(KCFE, self).__init__(xi, Pi, ti)
        self.t_a = ti

    def correction(self, t, dy, Y, x_j, c=30.):
        """Paso de correccion

        args:

            t: tiempo
            dy: innovacion en espacio de informacion
            Y: matriz de innovacion
            x_j: tupla de estimados
        """
        dt = t - self.t_a
        self.t_a = t
        x, P = self._x, self._P

        I_prior = np.linalg.inv(P)
        P = np.linalg.inv(I_prior + Y)

        d_i = len(x_j)
        suma = np.sum(x_j, axis=0) - d_i * x
        norm_P = np.linalg.norm(P, 'fro')
        c *= dt / (norm_P + 1)

        self._x = x + P.dot(dy + c*suma)
        self._P = P


class IF(object):
    def __init__(self, xi, Pi, ti=0.):
        """Filtro de Informacion."""
        self.init(xi, Pi, ti)
        self.motion_model = None

    @property
    def v(self):
        return self._v.copy()

    @property
    def Fisher(self):
        return self._I.copy()

    def init(self, xi, Pi, ti=0., f=None):
        self.t = ti
        self._I = np.linalg.inv(Pi)
        self._v = self._I.dot(xi)

    def transform(self, u, M):
        Minv = np.linalg.inv(M)
        v = Minv.dot(u)
        return v, Minv

    def prediction(self, t, *args):
        """Paso de predicción

        args:

            t: tiempo
        """
        dt = t - self.t
        self.t = t
        x, P = self.transform(self._v, self._I)
        x, F, Q = self.motion_model(dt, self._x, *args)
        P = F.dot(P).dot(F.T) + Q
        self._v, self._I = self.transform(x, P)

    def correction(self, y, Y):
        """Paso de correccion

        args:

            y: contribuciones en espacio de informacion
            Y: matriz de contribución
        """
        self._v = self._v + y
        self._I = self._I + Y


def plot(logs, ground_truth=None, basis=None):
    t = logs.t
    mean = logs.mean
    cov = logs.cov
    innovation = logs.innovation
    gt = ground_truth
    if basis is not None:
        m = basis.shape[-1]
        B = basis
        BT = B.swapaxes(-1, -2)
        mean = np.matmul(BT, mean).reshape(-1, m)
        cov = np.matmul(BT, np.matmul(cov, B))
    else:
        mean = mean.reshape(len(t), -1)

    figs = []

    f, axes = plt.subplots(2, 2)
    f.suptitle('Metrics')
    figs.append(f)

    lines = axes[0, 0].plot(t, mean)
    axes[0, 0].set_title('Estimate')
    axes[0, 0].grid(1)
    axes[0, 0].minorticks_on()
    axes[0, 0].set_xlabel(r'$t\,[sec]$')
    axes[0, 0].set_ylabel(r'$x(t)$')

    if gt is not None:
        if basis is not None:
            gt = np.matmul(BT, gt).reshape(-1, m)
        gt = gt.reshape(len(t), -1)

        # plt.gca().set_prop_cycle(None)
        for i, l in enumerate(lines):
            axes[0, 0].plot(t, gt[:, i], color=l.get_color(), ls='dotted')

        axes[1, 0].set_title('Error')
        axes[1, 0].plot(t, gt - mean)
        axes[1, 0].grid(1)
        axes[1, 0].minorticks_on()
        axes[1, 0].set_xlabel(r'$t\,[sec]$')
        axes[1, 0].set_ylabel(r'$x(t) - \hat{x}(t)$')

        sigma = np.sqrt(cov.diagonal(axis1=1, axis2=2))
        plt.gca().set_prop_cycle(None)
        axes[1, 0].plot(t, sigma, ls='dashed')
        plt.gca().set_prop_cycle(None)
        axes[1, 0].plot(t, -sigma, ls='dashed')

    axes[0, 1].set_title('Covariance eigenvalues')
    axes[0, 1].plot(t, np.sqrt(np.linalg.eigvalsh(cov)))
    axes[0, 1].grid(1)
    axes[0, 1].minorticks_on()
    axes[0, 1].set_xlabel(r'$t\,[sec]$')
    axes[0, 1].set_ylabel(r'$\sqrt{\lambda(P)(t)}$')

    axes[1, 1].set_title('Covariance metrics')
    axes[1, 1].plot(t, cov.trace(axis1=1, axis2=2), label=r'$\rm{tr}(P)$')
    axes[1, 1].plot(t, np.log(np.linalg.det(cov)), label=r'$\rm{logdet}(P)$')
    axes[1, 1].grid(1)
    axes[1, 1].minorticks_on()
    axes[1, 1].set_xlabel(r'$t\,[sec]$')
    axes[1, 1].legend()

    sensors = np.unique(innovation[:, 1])
    s = len(sensors)
    f, axes = plt.subplots(s, 2)
    axes = axes.reshape(s, 2)
    axes[0, 0].set_title('Measurements')
    axes[0, 1].set_title('Auto-Correlation')
    figs.append(f)

    for i, s in enumerate(sensors):

        match = innovation[:, 1] == s
        t = innovation[match, 0]
        dz = np.array(innovation[match, 2].tolist())
        dz = dz.reshape(len(t), -1)
        axes[i, 0].plot(t, dz, ls='', marker='.', markersize=2)
        axes[i, 0].grid(1)
        axes[i, 0].minorticks_on()
        axes[i, 0].set_xlabel(r'$t\,[sec]$')
        axes[i, 0].set_ylabel(r'$\delta z$')
        axes[i, 0].set_xlim(logs.t[0], None)

        for dz_i in dz.T:
            axes[i, 1].acorr(dz_i, usevlines=False, linestyle='-')
        axes[i, 1].grid(1)
        axes[i, 1].minorticks_on()
        axes[i, 1].set_xlabel(r'$\Delta t\,[sec]$')

    return figs
