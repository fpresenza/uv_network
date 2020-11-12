#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mié nov 11 20:10:02 -03 2020
"""
import numpy as np

import gpsic.plotting.planar as plotting
import gpsic.cluster.r3dof4.kinematic_functions as r3dof4
from uvnpy.modelos import integrador
from uvnpy.filtering import kalman

plt = plotting.matplotlib.pyplot
GPS = wgn = np.random.multivariate_normal
Kfor = r3dof4.forward_position_kinematics
Kinv = r3dof4.inverse_position_kinematics


def h(r_c, i):
    x_c, y_c, z_c, Roll, Pitch, Yaw, phi_1, phi_2, phi_3, p, q, beta = r_c

    Ry = np.array([[1., 0., 0.],
                   [0., np.cos(Pitch), -np.sin(Pitch)],
                   [0., np.sin(Pitch), np.cos(Pitch)]])

    Rx = np.array([[np.cos(Roll), 0., np.sin(Roll)],
                   [0., 1., 0.],
                   [-np.sin(Roll), 0., np.cos(Roll)]])

    Rz = np.array([[np.cos(Yaw), -np.sin(Yaw), 0.],
                   [np.sin(Yaw), np.cos(Yaw), 0.],
                   [0., 0., 1.]])

    r = np.sqrt(p**2. + q**2. + 2. * p * q * np.cos(beta))

    x1_2d = 0.
    y1_2d = (1. / 3.) * r
    z1_2d = 0.

    if i == 1:
        r1_o = np.array([x1_2d, y1_2d, z1_2d])
        r1_r = np.dot(Rx, r1_o.T)
        r1_rp = np.dot(Ry, r1_r.T)
        r1_rpy = np.dot(Rz, r1_rp.T)

        x_1 = x_c + r1_rpy[0]
        y_1 = y_c + r1_rpy[1]
        z_1 = z_c + r1_rpy[2]
        yaw_1 = Yaw + phi_1 + np.pi/2.
        return np.array([x_1, y_1, z_1, yaw_1])

    alpha_1 = np.arctan2(-y1_2d, -x1_2d)
    alpha_2 = np.arctan2(q * np.sin(beta), q * np.cos(beta) + p)
    alpha = alpha_2 - alpha_1

    if i == 2:
        x2_2d = x1_2d + p * np.cos(alpha)
        y2_2d = y1_2d - p * np.sin(alpha)
        z2_2d = 0.

        r2_o = np.array([x2_2d, y2_2d, z2_2d])
        r2_r = np.dot(Rx, r2_o.T)
        r2_rp = np.dot(Ry, r2_r.T)
        r2_rpy = np.dot(Rz, r2_rp.T)

        x_2 = x_c + r2_rpy[0]
        y_2 = y_c + r2_rpy[1]
        z_2 = z_c + r2_rpy[2]
        yaw_2 = Yaw + phi_2 + np.pi/2.
        return np.array([x_2, y_2, z_2, yaw_2])

    if i == 3:
        x3_2d = x1_2d + q * np.cos(beta - alpha)
        y3_2d = y1_2d + q * np.sin(beta - alpha)
        z3_2d = 0.

        r3_o = np.array([x3_2d, y3_2d, z3_2d])
        r3_r = np.dot(Rx, r3_o.T)
        r3_rp = np.dot(Ry, r3_r.T)
        r3_rpy = np.dot(Rz, r3_rp.T)

        x_3 = x_c + r3_rpy[0]
        y_3 = y_c + r3_rpy[1]
        z_3 = z_c + r3_rpy[2]
        yaw_3 = Yaw + phi_3 + np.pi/2.
        return np.array([x_3, y_3, z_3, yaw_3])


class r3dof4_kcf(kalman.KFi):
    def __init__(self, x, dx, Q, R, ti=0.):
        super(r3dof4_kcf, self).__init__(x, dx, ti=0.)
        self.Id = np.identity(12)
        self.Q = Q
        self.Rinv = np.linalg.inv(R)

    def f(self, dt, x, u):
        Phi = self.Id
        x_prior = x
        Q = self.Q * (dt**2)
        return x_prior, Phi, Q

    def observacion(self, z):
        # x = self.x
        Rinv = self.Rinv
        dz = np.subtract(z, p)
        dy = np.matmul(Rinv, dz)
        Y = Rinv
        return dy, Y

    # def modelo_xbee(self, z, landmarks):
    #     p = self.x
    #     Rinv = self.Rinv_xbee
    #     diff = np.subtract(p, landmarks)
    #     hat_z = norma(diff)
    #     H = diff / hat_z.reshape(-1, 1)
    #     dz = np.subtract(z, hat_z)
    #     dy = Rinv * np.matmul(H.T, dz)
    #     Y = Rinv * np.matmul(H.T, H)
    #     return dy, Y


if __name__ == '__main__':
    ci = [
        5., 10., 7.,         # x, y, z
        0., 0.,  0.,         # r, p, y
        0., 0.,  0.,         # phi_i
        4., 5.,  np.pi/3]    # p, q, beta
    dci = [
        1., 1., 1.,         # x, y, z
        0., 0., 0.,         # r, p, y
        0., 0., 0.,         # phi_i
        1., 1., 0.]         # p, q, beta
    Q = np.diag([
        1.,  1.,  1.,         # x, y, z
        0.2, 0.2, 0.2,        # r, p, y
        0.2, 0.2, 0.2,        # phi_i
        1.,  1.,  0.2])       # p, q, beta
    R = np.diag([
        4., 4., 4., 0.25,     # x, y, z, yaw
        4., 4., 4., 0.25,     # x, y, z, yaw
        4., 4., 4., 0.25      # x, y, z, yaw
        ])

    # fig = plt.figure()
    # gs = fig.add_gridspec(1, 1)
    # # posición
    # pax = plotting.agregar_ax(
    #     gs[0, 0],
    #     title='Pos. (verdadero vs. estimado)', title_kw={'fontsize': 11},
    #     xlabel='t [seg]', ylabel='posición [m]', label_kw={'fontsize': 10})

    ri = Kinv(ci)
    uav = [
        integrador(ri[:4]),
        integrador(ri[4:8]),
        integrador(ri[8:])]

    kcf = r3dof4_kcf(ci, dci, Q, R)
    tiempo = np.arange(0.1, 50, 0.1)
    p = [ci]
    f_p = [kcf.x]
    # for t in tiempo:
    #     u = [0.5, 1.]
    #     u = wgn(u, Q)
    #     v.step(t, u)
    #     x = v.x

    #     kcf.prediccion(t, u)
    #     # z_gps = GPS(x, R_gps)
    #     # kcf.actualizacion(*kcf.modelo_gps(z_gps))
    #     z_xbee = xBee(x, landmarks, R_xbee)
    #     kcf.actualizacion(*kcf.modelo_xbee(z_xbee, landmarks))
    #     p.append(x)
    #     f_p.append(kcf.p)

    # t = np.hstack([0, tiempo])
    # p = np.vstack(p)
    # f_p = np.vstack(f_p)
    # plotting.agregar_linea(pax, t, p[:, 0] - f_p[:, 0], color='r')
    # plotting.agregar_linea(pax, t, p[:, 1] - f_p[:, 1], color='g')

    plt.show()
