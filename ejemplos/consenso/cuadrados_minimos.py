#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date nov  3 20:39:25 -03 2020
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt  # noqa

import gpsic.plotting.planar as plotting
from uvnpy.modelos import point
from uvnpy.redes import grafo
from uvnpy.filtering import similaridad, kalman

proximidad = grafo.proximidad


def run(tiempo, red, rango_max):
    # logs
    Pos = dict([(v.id, [v.din.p]) for v in red.vehiculos])
    avg = dict([(v.id, [v.promedio[0].x]) for v in red.vehiculos])
    E = [red.enlaces]

    fig, ax = plt.subplots()
    ax.grid(1)
    ax.minorticks_on()
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    plotting.ellipse(ax, xf, Pf, alpha=0.4, color='m')

    for i, t in enumerate(tiempo[1:]):
        red.reconectar(proximidad, rango_max)
        E.append(red.enlaces)
        red.intercambiar()
        color = ['0.3', '0.3', '0.3']
        xs = []
        Ps = []
        for v in red.vehiculos:
            v.consenso_promedio_step(0, t)
            v.consenso_promedio_step(1, t)
            v.box.limpiar_entrada()

            y = N * v.promedio[0].x
            Y = N * v.promedio[1].x
            x, P = similaridad(y, Y)

            xs.append(x)
            Ps.append(P)

            Pos[v.id].append(v.din.p)
            avg[v.id].append(v.promedio[0].x)

        ell = [
            plotting.ellipse(ax, x_i, P_i, alpha=0.4, color=c)
            for c, x_i, P_i in zip(color, xs, Ps)]
        fig.savefig(
            '/tmp/cm/frame_{}.png'.format(str(i).zfill(3)), format='png')
        [e.remove() for e in ell]

    return Pos, E, avg


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
    tiempo = np.arange(arg.ti, arg.tf, arg.h)

    N = arg.agents
    red = grafo.grafo(directed=False)
    red.agregar_vehiculos([point(i) for i in range(N)])

    red.iniciar_dinamica({
        0: [0, 0],
        1: [15, 0],
        2: [-10, 15]
    })

    p = [0, 0]

    P1 = np.array([[20, 2.5],
                   [2.5, 10]])**2
    P2 = np.array([[15, 1],
                   [1, 10]])**2
    P3 = np.array([[15,  5],
                   [5,  25]])**2

    x1 = np.random.multivariate_normal(p, P1)
    x2 = np.random.multivariate_normal(p, P2)
    x3 = np.random.multivariate_normal(p, P3)
    print('x\n', x1, '\n', x2, '\n', x3)

    y1, Y1 = similaridad(x1, P1)
    y2, Y2 = similaridad(x2, P2)
    y3, Y3 = similaridad(x3, P3)

    # print('v\n', y1, '\n', y2, '\n', y3)
    yf, Yf = kalman.fusionar([(y1, Y1), (y2, Y2), (y3, Y3)])
    xf, Pf = similaridad(yf, Yf)

    red.iniciar_consenso_promedio({
        0: y1,
        1: y2,
        2: y3
    })
    red.iniciar_consenso_promedio({
        0: Y1,
        1: Y2,
        2: Y3
    })

    rango_max = np.sqrt(20.**2 + 20.**2)

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    Pos, E, avg = run(tiempo, red, rango_max)
    t = tiempo

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    fig = plt.figure()
    # fig.subplots_adjust(hspace=0.5, wspace=0.25)
    gs = fig.add_gridspec(1, 1)
    ax_avg = plotting.agregar_ax(
        gs[0, 0],
        title='Consenso - Promedio', title_kw={'fontsize': 11},
        xlabel='t [seg]', ylabel='', label_kw={'fontsize': 10})
    plotting.agregar_linea(ax_avg, t, avg[0], color='r', label='$0$')
    plotting.agregar_linea(ax_avg, t, avg[1], color='g', label='$1$')
    plotting.agregar_linea(ax_avg, t, avg[2], color='b', label='$2$')

    if arg.save:
        fig.savefig('/tmp/consenso_promedio.pdf', format='pdf')
    else:
        plt.show()

    graph_plotter = plotting.GraphPlotter(
      t, Pos, E, save=arg.save)
    if arg.animate:
        graph_plotter.animation2d(
            step=1, plot_kw={'xlim': [-40, 40], 'ylim': [-40, 40]})
