#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date vie nov  6 15:24:28 -03 2020
"""
import numpy as np

import gpsic.plotting.planar as plotting
from uvnpy.modelos import integrador
from uvnpy.filtering import kalman

plt = plotting.matplotlib.pyplot
GPS = wgn = np.random.multivariate_normal


def norma(v):
    sqr_sum = np.multiply(v, v).sum(1)
    return np.sqrt(sqr_sum)


def xBee(p, landmarks, sigma):
    d = norma(np.subtract(p, landmarks))
    return np.random.normal(d, sigma)


class kfi_test(kalman.KFi):
    def __init__(self, x, dx, Q, R_gps, R_xbee, ti=0.):
        super(kfi_test, self).__init__(x, dx, ti=0.)
        self.Id = np.identity(2)
        self.Q = Q
        self.Rinv_gps = np.linalg.inv(R_gps)
        self.Rinv_xbee = 1./R_xbee
        self.dz = None

    def prior(self, dt, u):
        self._x = self._x + np.multiply(dt, u)
        Q = self.Q * (dt**2)
        self._P = self._P + Q

    def modelo_gps(self, z):
        p = self.x
        Rinv = self.Rinv_gps
        dz = np.subtract(z, p)
        dy = np.matmul(Rinv, dz)
        Y = Rinv
        return dy, Y

    def modelo_xbee(self, z, landmarks):
        p = self.x
        Rinv = self.Rinv_xbee
        diff = np.subtract(p, landmarks)
        hat_z = norma(diff)
        H = diff / hat_z.reshape(-1, 1)
        dz = np.subtract(z, hat_z)
        self.dz = dz
        dy = Rinv * np.matmul(H.T, dz)
        Y = Rinv * np.matmul(H.T, H)
        return dy, Y


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

        kfi = kfi_test(hat_pi, dpi, Q, R_gps, R_xbee)
        tiempo = np.arange(0, 50, 0.1)
        p = [pi]
        f_p = [hat_pi]
        f_dz = [np.zeros(3)]
        for t in tiempo[1:]:
            u = [0.5, 1.]

            v.step(t, wgn(u, Q))
            x = v.x

            kfi.prediccion(t, u)
            # z_gps = GPS(x, R_gps)
            # kfi.actualizacion(*kfi.modelo_gps(z_gps))
            z_xbee = xBee(x, landmarks, sigma_xbee)
            kfi.actualizacion(*kfi.modelo_xbee(z_xbee, landmarks))
            p.append(x)
            f_p.append(kfi.x)
            f_dz.append(kfi.dz)

        t = tiempo
        p = np.vstack(p)
        f_p = np.vstack(f_p)

        plotting.agregar_linea(pax, t, p[:, 0] - f_p[:, 0], color='r')
        plotting.agregar_linea(pax, t, p[:, 1] - f_p[:, 1], color='g')
        plotting.agregar_linea(iax, t, f_dz, color='0.5')

    plt.show()
