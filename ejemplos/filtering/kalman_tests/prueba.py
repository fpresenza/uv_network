#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mié nov 25 19:09:08 -03 2020
"""
import argparse
import numpy as np

import gpsic.plotting.planar as plotting
import gpsic.cluster.r3dof4.kinematic_functions as r3dof4
from uvnpy.modelos.lineal import integrador
from uvnpy.filtering import kalman

plt = plotting.matplotlib.pyplot
GPS = wgn = np.random.multivariate_normal
Kfor = r3dof4.forward_position_kinematics
Kinv = r3dof4.inverse_position_kinematics
rad2deg = 180 / np.pi


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

    if i == 0:
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

    if i == 1:
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

    if i == 2:
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


def h_jac(r_c, i):
    J_inv = r3dof4.inverse_jacobian(r_c)
    r = 4 * i
    return J_inv[r:r+4, :]


class r3dof4_kfi(kalman.KFi):
    def __init__(self, i, x, dx, Q, R, ti=0.):
        super(r3dof4_kfi, self).__init__(x, dx, ti=0.)
        self.Q = Q
        self.Rinv = np.linalg.inv(R)
        self.In = np.eye(12)
        self.i = i

    def prior(self, dt, u):
        self._x = self._x + np.multiply(dt, u)
        Q = self.Q * (dt**2)
        self._P = self._P + Q

    def innovacion(self, z, y_j, I_j):
        x = self._x
        Rinv = self.Rinv
        y = Rinv.dot(z) + sum(y_j)
        Y = Rinv + sum(I_j)
        dy = y - Y.dot(x)
        return dy, Y

    def informacion(self, z):
        # I_i = np.linalg.inv(self._P)
        # y_i = I_i.dot(self._x)
        Rinv = self.Rinv
        y = Rinv.dot(z)
        Y = Rinv
        return y, Y


if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Parseo de argumentos
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-s', '--step',
        dest='h', default=100e-3, type=float, help='paso de simulación')
    parser.add_argument(
        '-e', '--tf',
        default=1.0, type=float, help='tiempo final')
    parser.add_argument(
        '-g', '--save',
        default=False, action='store_true',
        help='flag para guardar')

    arg = parser.parse_args()

    # ------------------------------------------------------------------
    # Configuración
    # ------------------------------------------------------------------
    hat_ci = [
        0., 0., 0.,         # x, y, z
        0., 0., 0.,         # r, p, y
        0., 0., 0.,         # phi_i
        4., 5., np.pi/3]    # p, q, beta
    dci = [
        1., 1., 1.,         # x, y, z
        0.5, 0.5, 0.5,         # r, p, y
        0.3, 0.3, 0.3,         # phi_i
        1., 1., 0.3]         # p, q, beta
    # ci = wgn(hat_ci, np.diag(dci))
    ci = hat_ci
    Q = np.diag([
        1., 1., 1.,           # x, y, z
        0.1, 0.1, 0.1,        # r, p, y
        0.1, 0.1, 0.1,        # phi_i
        1.5, 1.5, 0.1])      # p, q, beta
    R = np.diag([
        2., 2., 2.,           # x, y, z
        0.3, 0.3, 0.3,        # r, p, y
        0.3, 0.3, 0.3,        # phi_i
        2.5, 2.5, 0.3])      # p, q, beta
    cluster = integrador(ci)
    ref = integrador(ci)

    kfi = [
        r3dof4_kfi(0, hat_ci, dci, Q, R),
        r3dof4_kfi(1, hat_ci, dci, Q, R),
        r3dof4_kfi(2, hat_ci, dci, Q, R)]

    tiempo = np.arange(0, arg.tf, arg.h)
    c = [ci]
    r = [ci]
    f_c = [
        [kfi[0].x],
        [kfi[1].x],
        [kfi[2].x]]
    z = [None, None, None]
    dy = [None, None, None]
    Y = [None, None, None]
    y_j = [None, None, None]
    I_j = [None, None, None]

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------

    for t in tiempo[1:]:
        seno = np.sin(0.5 * t)
        dc = np.array([
            seno, 0., 0.5,                  # x, y, z
            -0.05, 0., 0.1 * seno,          # r, p, y
            0., 0., 0.3 * seno,             # phi_i
            0.5, 0., 0.])                   # p, q, beta
        if t < 15:
            u = dc
        elif 30. < t < 40:
            u = -dc
        else:
            u = np.zeros(12)
        cluster.step(t, u)
        ref.step(t, u)

        r_c = cluster.x
        z[0] = GPS(r_c, R)
        z[1] = GPS(r_c, R)
        z[2] = GPS(r_c, R)

        kfi[0].prediccion(t, u)
        kfi[1].prediccion(t, u)
        kfi[2].prediccion(t, u)

        y_j[0], I_j[0] = kfi[0].informacion(z[0])
        y_j[1], I_j[1] = kfi[1].informacion(z[1])
        y_j[2], I_j[2] = kfi[2].informacion(z[2])

        dy[0], Y[0] = kfi[0].innovacion(z[0], [y_j[1]], [I_j[1]])
        dy[1], Y[1] = kfi[1].innovacion(z[1], [y_j[0]], [I_j[0]])
        dy[2], Y[2] = kfi[2].innovacion(z[2], [], [])

        kfi[0].actualizacion(dy[0], Y[0])
        kfi[1].actualizacion(dy[1], Y[1])
        kfi[2].actualizacion(dy[2], Y[2])

        c.append(cluster.x)
        r.append(ref.x)
        f_c[0].append(kfi[0].x)
        f_c[1].append(kfi[1].x)
        f_c[2].append(kfi[2].x)

    t = tiempo
    c = np.vstack(c)
    r = np.vstack(r)
    f_c = np.hstack(f_c)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    pfig = plt.figure(figsize=(12, 5))
    gs = pfig.add_gridspec(3, 2)
    xax = plotting.agregar_ax(
        gs[0, 0],
        xlabel='t [seg]', ylabel='[m]', label_kw={'fontsize': 10})
    xax.set_ylim(-10, 10)
    yax = plotting.agregar_ax(
        gs[1, 0],
        xlabel='t [seg]', ylabel='[m]', label_kw={'fontsize': 10})
    yax.set_ylim(-10, 10)
    zax = plotting.agregar_ax(
        gs[2, 0],
        xlabel='t [seg]', ylabel='[m]', label_kw={'fontsize': 10})
    zax.set_ylim(-10, 10)

    pax = plotting.agregar_ax(
        gs[0, 1],
        xlabel='t [seg]', ylabel='[m]', label_kw={'fontsize': 10})
    pax.set_ylim(0, 15)

    qax = plotting.agregar_ax(
        gs[1, 1],
        xlabel='t [seg]', ylabel='[m]', label_kw={'fontsize': 10})
    qax.set_ylim(0, 15)

    plotting.agregar_linea(xax, t, c[:, 0], color='r', label='$x$')
    # plotting.agregar_linea(xax, t, r[:, 0],
    #   color='r', label=r'$x_{r}$', ls='--')
    plotting.agregar_linea(xax, t, f_c[:, [0, 12, 24]], ls='dotted')

    plotting.agregar_linea(yax, t, c[:, 1], color='b', label='$y$')
    # plotting.agregar_linea(yax, t, r[:, 1],
    #   color='b', label=r'$y_{r}$', ls='--')
    plotting.agregar_linea(yax, t, f_c[:, [1, 13, 25]], ls='dotted')

    plotting.agregar_linea(zax, t, c[:, 2], color='g', label='$z$')
    # plotting.agregar_linea(zax, t, r[:, 2],
    #   color='g', label=r'$z_{r}$', ls='--')
    plotting.agregar_linea(zax, t, f_c[:, [2, 14, 26]], ls='dotted')

    plotting.agregar_linea(pax, t, c[:, 9], color='y', label='$p$')
    # plotting.agregar_linea(pax, t, r[:, 9],
    #   color='y', label=r'$p_{r}$', ls='--')
    plotting.agregar_linea(pax, t, f_c[:, [9, 21, 33]], ls='dotted')

    plotting.agregar_linea(qax, t, c[:, 10], color='purple', label='$q$')
    # plotting.agregar_linea(qax, t, r[:, 10],
    #   color='purple', label=r'$q_{r}$', ls='--')
    plotting.agregar_linea(qax, t, f_c[:, [10, 22, 34]], ls='dotted')

    ofig = plt.figure(figsize=(12, 5))
    gs = ofig.add_gridspec(2, 2)
    rax = plotting.agregar_ax(
        gs[0, 0],
        xlabel='t [seg]', ylabel='roll [deg]', label_kw={'fontsize': 10})
    rax.set_ylim(-90, 90)
    pax = plotting.agregar_ax(
        gs[0, 1],
        xlabel='t [seg]', ylabel='pitch [deg]', label_kw={'fontsize': 10})
    pax.set_ylim(-90, 90)
    yax = plotting.agregar_ax(
        gs[1, 0],
        xlabel='t [seg]', ylabel='yaw [deg]', label_kw={'fontsize': 10})
    yax.set_ylim(-90, 90)
    bax = plotting.agregar_ax(
        gs[1, 1],
        xlabel='t [seg]', ylabel='beta [deg]', label_kw={'fontsize': 10})
    bax.set_ylim(-90, 90)

    plotting.agregar_linea(
        rax, t, c[:, 3] * rad2deg, color='r', label=r'$\phi$')
    # plotting.agregar_linea(rax, t, r[:, 3] * rad2deg,
    #   color='r', label=r'$\phi_{r}$', ls='--')
    plotting.agregar_linea(rax, t, f_c[:, [3, 15, 27]] * rad2deg, ls='dotted')

    plotting.agregar_linea(
        pax, t, c[:, 4] * rad2deg, color='b', label=r'$\theta$')
    # plotting.agregar_linea(pax, t, r[:, 4] * rad2deg,
    #   color='b', label=r'$\theta_{r}$', ls='--')
    plotting.agregar_linea(pax, t, f_c[:, [4, 16, 28]] * rad2deg, ls='dotted')

    plotting.agregar_linea(
        yax, t, c[:, 5] * rad2deg, color='g', label=r'$\psi$')
    # plotting.agregar_linea(yax, t, r[:, 5] * rad2deg,
    #   color='g', label=r'$\psi_{r}$', ls='--')
    plotting.agregar_linea(yax, t, f_c[:, [5, 17, 29]] * rad2deg, ls='dotted')

    plotting.agregar_linea(
        bax, t, c[:, 11] * rad2deg, color='y', label=r'$\beta$')
    # plotting.agregar_linea(bax, t, r[:, 11] * rad2deg,
    #   color='y', label=r'$\beta_{r}$', ls='--')
    plotting.agregar_linea(bax, t, f_c[:, [11, 23, 35]] * rad2deg, ls='dotted')

    hfig = plt.figure(figsize=(6, 5))
    gs = hfig.add_gridspec(3, 1)
    vax_1 = plotting.agregar_ax(
        gs[0, 0],
        xlabel='t [seg]', ylabel='yaw [deg]', label_kw={'fontsize': 10})
    vax_1.set_ylim(-90, 90)

    vax_2 = plotting.agregar_ax(
        gs[1, 0],
        xlabel='t [seg]', ylabel='yaw [deg]', label_kw={'fontsize': 10})
    vax_2.set_ylim(-90, 90)

    vax_3 = plotting.agregar_ax(
        gs[2, 0],
        xlabel='t [seg]', ylabel='yaw [deg]', label_kw={'fontsize': 10})
    vax_3.set_ylim(-90, 90)

    plotting.agregar_linea(
        vax_1, t, c[:, 6] * rad2deg, color='r', label=r'$\varphi_1$')
    # plotting.agregar_linea(vax_1, t, r[:, 6] * rad2deg,
    #   color='r', label=r'$\varphi_{1r}$', ls='--')
    plotting.agregar_linea(
        vax_1, t, f_c[:, [6, 18, 30]] * rad2deg, ls='dotted')
    plotting.agregar_linea(
        vax_2, t, c[:, 7] * rad2deg, color='b', label=r'$\varphi_2$')
    # plotting.agregar_linea(vax_2, t, r[:, 7] * rad2deg,
    #   color='b', label=r'$\varphi_{2r}$', ls='--')
    plotting.agregar_linea(
        vax_2, t, f_c[:, [7, 19, 31]] * rad2deg, ls='dotted')
    plotting.agregar_linea(
        vax_3, t, c[:, 8] * rad2deg, color='g', label=r'$\varphi_3$')
    # plotting.agregar_linea(vax_3, t, r[:, 8] * rad2deg,
    #   color='g', label=r'$\varphi_{3r}$', ls='--')
    plotting.agregar_linea(
        vax_3, t, f_c[:, [8, 20, 32]] * rad2deg, ls='dotted')

    if arg.save:
        pfig.savefig('/tmp/cluster_r3dof4_kfi_p.pdf', format='pdf')
        ofig.savefig('/tmp/cluster_r3dof4_kfi_o.pdf', format='pdf')
        hfig.savefig('/tmp/cluster_r3dof4_kfi_h.pdf', format='pdf')
    else:
        plt.show()
