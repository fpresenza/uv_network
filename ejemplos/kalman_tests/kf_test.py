#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date vie nov  6 15:24:28 -03 2020
"""
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from gpsic.toolkit import linalg
import gpsic.plotting.planar as plotting

from uvnpy.modelos import integrador
from uvnpy.filtering import kalman
from uvnpy.sensores import rango

GPS = np.random.multivariate_normal


def xBee(p, landmarks, R):
    d = [norm(np.subtract(p, lm)) for lm in landmarks]
    return np.random.normal(d, R)


class kf_test(kalman.KF):
    def __init__(self, x, dx, Q, R_gps, R_xbee, ti=0.):
        super(kf_test, self).__init__(x, dx, ti=0.)
        self.Id = np.identity(2)
        self.Q = Q
        self.R_gps = R_gps
        self.R_xbee = R_xbee

    @property
    def p(self):
        return self.x

    def f(self, dt, x, u):
        Phi = self.Id
        x_prior = x + np.multiply(dt, u)
        Q = self.Q * (dt**2)
        return x_prior, Phi, Q

    def modelo_gps(self):
        H = self.Id
        R = self.R_gps
        return H, R

    def modelo_xbee(self, landmarks):
        p = self.x
        hat_z = [linalg.dist(p, lm) for lm in landmarks]
        H = np.vstack([rango.gradiente(p, lm) for lm in landmarks])
        R = np.diag([self.R_xbee for _ in landmarks])
        return H, R, hat_z


if __name__ == '__main__':
    pi = [5., 10.]
    dpi = [3., 3.]
    Q = 1. * np.eye(2)
    R_gps = 9 * np.eye(2)
    R_xbee = 9.

    landmarks = [(0., 0.), (0., 50.), (50., 0.)]

    v = integrador(pi, Q=Q)

    kf = kf_test(pi, dpi, Q, R_gps, R_xbee)
    tiempo = np.arange(0.1, 50, 0.1)
    p = [pi]
    f_p = [pi]
    for t in tiempo:
        # u = [np.cos(t), np.sin(t)]
        u = [0.5, 1.]
        v.step(t, u)
        kf.prediccion(t, u)

        # z_gps = GPS(v.p, R_gps)
        # kf.actualizacion(z_gps, *kf.modelo_gps())
        z_xbee = xBee(v.p, landmarks, R_xbee)
        kf.actualizacion(z_xbee, *kf.modelo_xbee(landmarks))
        p.append(v.p)
        f_p.append(kf.p)

    t = np.hstack([0, tiempo])
    p = np.vstack(p)
    f_p = np.vstack(f_p)

    fig = plt.figure()
    gs = fig.add_gridspec(1, 1)
    # posición
    pax = plotting.agregar_ax(
        gs[0, 0],
        title='Pos. (verdadero vs. estimado)', title_kw={'fontsize': 11},
        xlabel='t [seg]', ylabel='posición [m]', label_kw={'fontsize': 10})
    plotting.agregar_linea(pax, t, p[:, 0], color='r', label='$p_x$')
    plotting.agregar_linea(pax, t, p[:, 1], color='g', label='$p_y$')
    plotting.agregar_linea(pax, t, f_p[:, 0], color='r', ls='dotted')
    plotting.agregar_linea(pax, t, f_p[:, 1], color='g', ls='dotted')

    plt.show()
