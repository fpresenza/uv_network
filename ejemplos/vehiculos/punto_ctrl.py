#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Thu Ago 5 11:00:29 2020
@author: fran
"""
import argparse
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt

import gpsic.plotting.planar as plotting
from gpsic.analisis.core import cargar_yaml
from uvnpy.modelos import vehiculo, integrador
from uvnpy.filtering import kalman, metricas
from uvnpy.control.informativo import minimizar
from uvnpy.sensores import rango


class integrador_ruidoso(integrador):
    def __init__(self, xi, Q, ti=0.):
        super(integrador_ruidoso, self).__init__(xi, ti)
        self.Q = np.asarray(Q)

    def dinamica(self, x, t, u):
        self._dx = np.random.multivariate_normal(u, self.Q)
        return self._dx


class ekf(kalman.KF):
    def __init__(self, x, dx, Q, R, ti=0.):
        """Filtro de localización. """
        super(ekf, self).__init__(x, dx, ti=0.)
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)
        self.logs = SimpleNamespace(
            t=[self.t],
            x=[self.x],
            dvst=[metricas.sqrt_diagonal(self.P)],
            eigs=[metricas.eigvalsh(self.P)])

    @property
    def p(self):
        return self.x

    def prior(self, dt, u):
        self._x = self._x + np.multiply(dt, u)
        self._P = self._P + self.Q * (dt**2)

    def modelo_rango(self, z, landmarks):
        p = self.x
        hat_z = rango.distancia(p, landmarks)
        dz = z - hat_z
        H = rango.jacobiano(p, landmarks)
        R = self.R * np.eye(len(landmarks))
        return dz, H, R

    def update(self, t, u, rangos, landmarks):
        self.prediccion(t, u)
        dz, H, R = self.modelo_rango(rangos, landmarks)
        self.actualizacion(dz, H, R)

    def guardar(self):
        """Guarda los últimos datos. """
        self.logs.t.append(self.t)
        self.logs.x.append(self.x)
        self.logs.dvst.append(
            metricas.sqrt_diagonal(self.P))
        self.logs.eigs.append(
            metricas.eigvalsh(self.P))


det_matriz_innovacion = {
    'metrica': metricas.det,
    'matriz': rango.matriz_innovacion,
    'modelo': integrador,
    'Q': (np.eye(2), 4.5*np.eye(2), -100),
    'dim': 2,
    'horizonte': np.linspace(0.1, 1, 10)
}


def run(tiempo, puntos, landmarks):
    # logs
    r = puntos[0]
    x = [r.din.x]
    u = [r.control.u]
    P = dict([(punto.id, [punto.din.x]) for punto in puntos])

    for t in tiempo[1:]:
        for punto in puntos:
            p = punto.din.x
            hat_p = punto.filtro.p
            u_cmd = punto.control.update(
                hat_p, t,
                (landmarks, punto.rango.sigma))
            punto.din.step(t, u_cmd)
            rangos = punto.rango(p, landmarks)
            punto.filtro.update(t, u_cmd, rangos, landmarks)
            P[punto.id].append(p)
        x.append(r.din.x)
        u.append(r.control.u)
        r.filtro.guardar()

    return P, x, u, r.filtro.logs


if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Parseo de argumentos
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-s', '--step',
        dest='h', default=50e-3, type=float, help='paso de simulación')
    parser.add_argument(
        '-t', '--ti',
        metavar='T0', default=0.0, type=float, help='tiempo inicial')
    parser.add_argument(
        '-e', '--tf',
        default=1.0, type=float, help='tiempo final')
    parser.add_argument(
        '-r', '--arxiv',
        default='/tmp/punto.yaml', type=str, help='arhivo de configuración')
    parser.add_argument(
        '-g', '--save',
        default=False, action='store_true',
        help='flag para guardar los videos')
    parser.add_argument(
        '-a', '--animate',
        default=False, action='store_true',
        help='flag para generar animacion')
    parser.add_argument(
        '-n', '--agents',
        default=1, type=int, help='cantidad de agentes')

    arg = parser.parse_args()

    # ------------------------------------------------------------------
    # Configuración
    # ------------------------------------------------------------------
    # archivo de configuración
    config = cargar_yaml(arg.arxiv)
    Q = config['din']['Q']
    sigma = config['rango']['sigma']
    R = sigma**2

    # landmarks
    landmarks = [(0, -10), (0, 10)]
    # landmarks = []
    # landmarks = [[-15.90949736, 11.74311878],
    #              [-5.21570337, -6.41701965],
    #              [-13.76694731, -2.34360965],
    #              [-3.2733689, 18.90361114]]

    puntos = []
    for i in range(arg.agents):
        pi = np.random.uniform(-20, 20, 2)
        p = vehiculo(
            i,
            din=integrador_ruidoso(pi, Q),
            rango=rango.sensor(sigma),
            filtro=ekf(pi, np.ones(2), Q, R),
            control=minimizar(**det_matriz_innovacion))
        puntos.append(p)

    tiempo = np.arange(arg.ti, arg.tf, arg.h)

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    P, x, u, f = run(tiempo, puntos, landmarks)
    t = tiempo

    # variables
    x = np.vstack(x)
    f_x = np.vstack(f.x)
    f_dvst = np.vstack(f.dvst)
    f_eigs = np.vstack(f.eigs)
    u = np.vstack(u)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(13, 5))
    fig.subplots_adjust(hspace=0.5, wspace=0.25)
    gs = fig.add_gridspec(2, 2)
    # posición
    ax_pos = plotting.agregar_ax(
        gs[0, 0],
        title='Pos. (verdadero vs. estimado)', title_kw={'fontsize': 11},
        xlabel='t [seg]', ylabel='[m]', label_kw={'fontsize': 10})
    plotting.agregar_linea(ax_pos, t, x[:, 0], color='r', label='$p_x$')
    plotting.agregar_linea(ax_pos, t, x[:, 1], color='g', label='$p_y$')
    plotting.agregar_linea(ax_pos, t, f_x[:, 0], color='r', ls='dotted')
    plotting.agregar_linea(ax_pos, t, f_x[:, 1], color='g', ls='dotted')

    # control
    ax_ctr = plotting.agregar_ax(
        gs[0, 1],
        title='Esfuerzo de control', title_kw={'fontsize': 11},
        xlabel='t [seg]', ylabel=r'[m/s]', label_kw={'fontsize': 10})
    plotting.agregar_linea(ax_ctr, t, u[:, 0], color='r', label='$u_x$')
    plotting.agregar_linea(ax_ctr, t, u[:, 1], color='g', label='$u_y$')

    # error posición
    ax_ep = plotting.agregar_ax(
        gs[1, 0],
        title='Pos. (error vs. std. dev.)', title_kw={'fontsize': 11},
        xlabel='t [seg]', ylabel='[m]', label_kw={'fontsize': 10})
    plotting.agregar_linea(
        ax_ep, t, x[:, 0] - f_x[:, 0],
        color='r', label='$e_x$')
    plotting.agregar_linea(
        ax_ep, t, x[:, 1] - f_x[:, 1],
        color='g', label='$e_y$')
    plotting.agregar_linea(ax_ep, t, f_dvst[:, 0], color='r', ls='dotted')
    plotting.agregar_linea(ax_ep, t, f_dvst[:, 1], color='g', ls='dotted')
    plotting.agregar_linea(ax_ep, t, -f_dvst[:, 0], color='r', ls='dotted')
    plotting.agregar_linea(ax_ep, t, -f_dvst[:, 1], color='g', ls='dotted')

    # Valores singulares
    ax_sv = plotting.agregar_ax(
        gs[1, 1],
        title=r'Valores singulares: $\sigma(P_x)$', title_kw={'fontsize': 11},
        xlabel='t [seg]', ylabel='[m]', label_kw={'fontsize': 10})
    plotting.agregar_linea(
        ax_sv, t, np.sqrt(f_eigs[:, 1]), color='m',
        label=r'$\sqrt{\sigma_{\rm{max}}}$')
    plotting.agregar_linea(
        ax_sv, t, np.sqrt(f_eigs[:, 0]),
        color='c', label=r'$\sqrt{\sigma_{\rm{min}}}$')

    if arg.save:
        fig.savefig('/tmp/punto.pdf', format='pdf')
    else:
        plt.show()

    E = [[] for _ in t]
    graph_plotter = plotting.GraphPlotter(
        t, P, E, save=arg.save, landmarks=landmarks)
    if arg.animate:
        graph_plotter.animation2d(
            step=1, plot_kw={'xlim': [-40, 40], 'ylim': [-40, 40]})
