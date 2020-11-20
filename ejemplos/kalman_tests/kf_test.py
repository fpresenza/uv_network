#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date vie nov  6 15:24:28 -03 2020
"""
import numpy as np

import gpsic.plotting.planar as plotting
from uvnpy.modelos.lineal import integrador
from uvnpy.filtering import kalman

plt = plotting.matplotlib.pyplot
GPS = wgn = np.random.multivariate_normal


def norma(v):
    sqr_sum = np.multiply(v, v).sum(1)
    return np.sqrt(sqr_sum)


def xBee(p, landmarks, sigma):
    d = norma(np.subtract(p, landmarks))
    return np.random.normal(d, sigma)


class kf_test(kalman.KF):
    def __init__(self, x, dx, Q, R_gps, R_xbee, ti=0.):
        super(kf_test, self).__init__(x, dx, ti=0.)
        self.Id = np.identity(2)
        self.Q = Q
        self.R_gps = R_gps
        self.R_xbee = R_xbee
        self.dz = None

    def prior(self, dt, u):
        self._x = self._x + np.multiply(dt, u)
        Q = self.Q * (dt**2)
        self._P = self._P + Q

    def modelo_gps(self, z):
        dz = z - self._x
        H = self.Id
        R = self.R_gps
        return dz, H, R

    def modelo_xbee(self, z, landmarks):
        diff = np.subtract(self.x, landmarks)
        hat_z = norma(diff)
        dz = z - hat_z
        self.dz = dz
        H = diff / hat_z.reshape(-1, 1)
        R = self.R_xbee * np.eye(len(landmarks))
        return dz, H, R


if __name__ == '__main__':
    hat_pi = [5., 10.]
    dpi = [3., 3.]
    pi = wgn(hat_pi, np.diag(dpi))
    Q = 0.5 * np.eye(2)
    R_gps = 9 * np.eye(2)
    sigma_xbee = 3.
    R_xbee = sigma_xbee ** 2

    landmarks = [(0., 0.), (0., 50.), (50., 0.)]

    fig = plt.figure()
    gs = fig.add_gridspec(2, 1)
    # posición
    pax = plotting.agregar_ax(
        gs[0, 0],
        title='Pos. (verdadero vs. estimado)', title_kw={'fontsize': 11},
        xlabel='t [seg]', ylabel='posición [m]', label_kw={'fontsize': 10})

    iax = plotting.agregar_ax(
        gs[1, 0],
        title='Innovación', title_kw={'fontsize': 11},
        xlabel='t [seg]', ylabel='posición [m]', label_kw={'fontsize': 10})

    for i in range(10):
        v = integrador(pi)

        kf = kf_test(hat_pi, dpi, Q, R_gps, R_xbee)
        tiempo = np.arange(0, 50, 0.1)
        p = [pi]
        f_p = [hat_pi]
        f_dz = [np.zeros(3)]
        for t in tiempo[1:]:
            u = [0.5, 1.]

            v.step(t, wgn(u, Q))
            x = v.x

            kf.prediccion(t, u)
            # z_gps = GPS(x, R_gps)
            # kf.actualizacion(*kf.modelo_gps(z_gps))
            z_xbee = xBee(x, landmarks, sigma_xbee)
            kf.actualizacion(*kf.modelo_xbee(z_xbee, landmarks))
            p.append(x)
            f_p.append(kf.x)
            f_dz.append(kf.dz)

        t = tiempo
        p = np.vstack(p)
        f_p = np.vstack(f_p)

        plotting.agregar_linea(pax, t, p[:, 0] - f_p[:, 0], color='r')
        plotting.agregar_linea(pax, t, p[:, 1] - f_p[:, 1], color='g')
        plotting.agregar_linea(iax, t, f_dz, color='0.5')

    plt.show()
