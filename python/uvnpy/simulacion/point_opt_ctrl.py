#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on Thu Ago 5 11:00:29 2020
@author: fran
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

import gpsic.plotting.planar as plotting

from uvnpy.modelos import point


def run(tiempo, points, landmarks):
    # logs
    x = [points[0].c]
    u = [points[0].control.u]
    f_t, f_x, f_dvst, f_eig = points[0].filtro.logs()
    f = ([f_x], [f_dvst], [np.flip(f_eig)])
    P = dict([(point.id, [point.kin.p]) for point in points])

    for t in tiempo[1:]:
        for point in points:
            point.kin_step(t)
            point.control_step(t, landmarks=landmarks)
            P[point.id].append(point.kin.p)
            if point.id == points[0].id:
                x.append(point.kin.x)
                u.append(point.control.u)
                f_t, f_x, f_dvst, f_eig = point.filtro.logs()
                f[0].append(f_x)
                f[1].append(f_dvst)
                f[2].append(np.flip(f_eig))

    return P, x, u, f


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
      '-g', '--save',
      default=False, action='store_true', help='flag para guardar los videos')
    parser.add_argument(
      '-a', '--animate',
      default=False, action='store_true', help='flag para generar animacion')
    parser.add_argument(
      '-n', '--agents',
      default=1, type=int, help='cantidad de agentes')

    arg = parser.parse_args()

    # ------------------------------------------------------------------
    # Configuración
    # ------------------------------------------------------------------
    # landmarks
    landmarks = [(0, -10), (0, 10)]
    # landmarks = []
    # landmarks = [[-15.90949736, 11.74311878],
    #              [-5.21570337, -6.41701965],
    #              [-13.76694731, -2.34360965],
    #              [-3.2733689, 18.90361114]]

    def make_kin_kw():
        # return {'pi': np.random.uniform(-10, 10, 2), 'freq': arg.h**(-1)}
        return {'pi': [0.1, 0.], 'freq': arg.h**(-1)}
    points = [point(i, kin_kw=make_kin_kw()) for i in range(arg.agents)]

    tiempo = np.arange(arg.ti, arg.tf, arg.h)

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    P, x, u, f = run(tiempo, points, landmarks)
    t = tiempo

    # variables
    x = np.vstack(x).T
    f_x = np.vstack(f[0]).T
    f_dvst = np.vstack(f[1]).T
    f_eig = np.vstack(f[2]).T
    u = np.vstack(u).T

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(13, 5))
    fig.subplots_adjust(hspace=0.5, wspace=0.25)
    gs = fig.add_gridspec(2, 3)
    # posición
    ax_pos = plotting.agregar_ax(
        gs[0, 0],
        title='Pos. (verdadero vs. estimado)', title_kw={'fontsize': 11},
        xlabel='t [seg]', ylabel='posición [m]', label_kw={'fontsize': 10})
    plotting.agregar_linea(ax_pos, t, x[0], color='r', label='$p_x$')
    plotting.agregar_linea(ax_pos, t, x[1], color='g', label='$p_y$')
    plotting.agregar_linea(ax_pos, t, f_x[0], color='r', ls='dotted')
    plotting.agregar_linea(ax_pos, t, f_x[1], color='g', ls='dotted')

    # velocidad
    ax_vel = plotting.agregar_ax(
        gs[0, 1],
        title='Vel. (verdadero vs. estimado)', title_kw={'fontsize': 11},
        xlabel='t [seg]', ylabel=r'velocidad [m/s]', label_kw={'fontsize': 10})
    plotting.agregar_linea(ax_vel, t, x[2], color='r', label='$v_x$')
    plotting.agregar_linea(ax_vel, t, x[3], color='g', label='$v_y$')
    plotting.agregar_linea(ax_vel, t, f_x[2], color='r', ls='dotted')
    plotting.agregar_linea(ax_vel, t, f_x[3], color='g', ls='dotted')

    # control
    ax_ctr = plotting.agregar_ax(
        gs[0, 2],
        title='Esfuerzo de control', title_kw={'fontsize': 11},
        xlabel='t [seg]', ylabel=r'velocidad [m/s]', label_kw={'fontsize': 10})
    plotting.agregar_linea(ax_ctr, t, u[0], color='r', label='$v_x$')
    plotting.agregar_linea(ax_ctr, t, u[1], color='g', label='$v_y$')

    # error posición
    ax_ep = plotting.agregar_ax(
        gs[1, 0],
        title='Pos. (error vs. std. dev.)', title_kw={'fontsize': 11},
        xlabel='t [seg]', ylabel='posición [m]', label_kw={'fontsize': 10})
    plotting.agregar_linea(ax_ep, t, x[0] - f_x[0], color='r', label='$e_x$')
    plotting.agregar_linea(ax_ep, t, x[1] - f_x[1], color='g', label='$e_y$')
    plotting.agregar_linea(ax_ep, t, f_dvst[0], color='r', ls='dotted')
    plotting.agregar_linea(ax_ep, t, f_dvst[1], color='g', ls='dotted')
    plotting.agregar_linea(ax_ep, t, -f_dvst[0], color='r', ls='dotted')
    plotting.agregar_linea(ax_ep, t, -f_dvst[1], color='g', ls='dotted')

    # error velocidad
    ax_ev = plotting.agregar_ax(
        gs[1, 1],
        title='Vel. (error vs. std. dev.)', title_kw={'fontsize': 11},
        xlabel='t [seg]', ylabel=r'velocidad [m/s]', label_kw={'fontsize': 10})
    plotting.agregar_linea(
      ax_ev, t, x[2] - f_x[2], color='r', label='$e_{v_x}$')
    plotting.agregar_linea(
      ax_ev, t, x[3] - f_x[3], color='g', label='$e_{v_y}$')
    plotting.agregar_linea(ax_ev, t, f_dvst[2], color='r', ls='dotted')
    plotting.agregar_linea(ax_ev, t, f_dvst[3], color='g', ls='dotted')
    plotting.agregar_linea(ax_ev, t, -f_dvst[2], color='r', ls='dotted')
    plotting.agregar_linea(ax_ev, t, -f_dvst[3], color='g', ls='dotted')

    # Valores singulares
    ax_sv = plotting.agregar_ax(
        gs[1, 2],
        title=r'Valores singulares: $\sigma(P_x)$', title_kw={'fontsize': 11},
        xlabel='t [seg]', ylabel='', label_kw={'fontsize': 10})
    plotting.agregar_linea(
      ax_sv, t, np.sqrt(f_eig[0]),
      color='m', label=r'$\sqrt{\sigma_{\rm{max}}}$')
    plotting.agregar_linea(
      ax_sv, t, np.sqrt(f_eig[1]), color='0.5', label='')
    plotting.agregar_linea(ax_sv, t, np.sqrt(f_eig[2]), color='0.5', label='')
    plotting.agregar_linea(
      ax_sv, t, np.sqrt(f_eig[3]),
      color='c', label=r'$\sqrt{\sigma_{\rm{max}}}$')

    if arg.save:
        fig.savefig('/tmp/point_opt_ctrl.pdf', format='pdf')
    else:
        plt.show()

    L = [[] for _ in t]
    graph_plotter = plotting.GraphPlotter(
      t, P, L, save=arg.save, landmarks=landmarks)
    if arg.animate:
        graph_plotter.animation2d(
          step=1, plot_kw={'xlim': [-40, 40], 'ylim': [-40, 40]})
