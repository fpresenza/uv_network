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

import gpsic.plotting.planar as plotting

from uvnpy.modelos import integrador
from uvnpy.filtering import kalman


GPS = np.random.multivariate_normal


def xBee(p, landmarks, R):
    d = [norm(np.subtract(p, lm)) for lm in landmarks]
    return np.random.normal(d, R)


class kfi_test(kalman.KFi):
    def __init__(self, x, dx, Q, R_gps, R_xbee, ti=0.):
        super(kfi_test, self).__init__(x, dx, ti=0.)
        self.Id = np.identity(2)
        self.Q = Q
        self.Rinv_gps = np.linalg.inv(R_gps)
        self.Rinv_xbee = 1./R_xbee

    @property
    def p(self):
        return self.x

    def f(self, dt, x, u):
        Phi = self.Id
        x_prior = x + np.multiply(dt, u)
        Q = self.Q * (dt**2)
        return x_prior, Phi, Q

    def modelo_gps(self, z):
        Rinv = self.Rinv_gps
        y = np.matmul(Rinv, z)
        Y = Rinv
        return y, Y

    def modelo_xbee_i(self, z_i, p, lm):
        Rinv = self.Rinv_xbee
        diff = np.subtract(p, lm)
        dist = norm(diff)
        h_i = diff / dist
        HtRinv = Rinv * h_i
        y = HtRinv * z_i
        Y = np.outer(HtRinv, h_i)
        hat_y = HtRinv * dist
        return y, Y, hat_y

    def modelo_xbee(self, z, landmarks):
        p = self.x
        xbee_i = self.modelo_xbee_i
        info = [xbee_i(z_i, p, lm_i) for z_i, lm_i in zip(z, landmarks)]
        y, Y, hat_y = list(zip(*info))
        return sum(y), sum(Y), sum(hat_y)


if __name__ == '__main__':
    pi = [5., 10.]
    dpi = [3., 3.]
    Q = 1. * np.eye(2)
    R_gps = 9 * np.eye(2)
    R_xbee = 9.

    landmarks = [(0., 0.), (0., 50.), (50., 0.)]

    v = integrador(pi, Q=Q)

    kfi = kfi_test(pi, dpi, Q, R_gps, R_xbee)
    tiempo = np.arange(0.1, 50, 0.1)
    p = [pi]
    f_p = [pi]
    for t in tiempo:
        # u = [np.cos(t), np.sin(t)]
        u = [0.5, 1.]
        v.step(t, u)
        kfi.prediccion(t, u)

        # z_gps = GPS(v.p, R_gps)
        # kfi.actualizacion(*kfi.modelo_gps(z_gps))
        z_xbee = xBee(v.p, landmarks, R_xbee)
        kfi.actualizacion(*kfi.modelo_xbee(z_xbee, landmarks))
        p.append(v.p)
        f_p.append(kfi.p)

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
