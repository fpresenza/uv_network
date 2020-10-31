#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date jue oct 29 17:09:54 -03 2020
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt  # noqa

import gpsic.plotting.planar as plotting
from uvnpy.modelos import point
from uvnpy.redes import grafo, proximidad


def run(tiempo, red, rango_max):
    # logs
    P = dict([(v.id, [v.din.p]) for v in red.vehiculos])
    cmd = dict([(v.id, np.zeros(2)) for v in red.vehiculos])
    avg = dict([(v.id, [v.promedio.x]) for v in red.vehiculos])
    E = [red.enlaces]

    for t in tiempo[1:]:
        red.reconectar(proximidad, rango_max)
        E.append(red.enlaces)
        red.intercambiar()
        for v in red.vehiculos:
            v.din.step(t, cmd[v.id])
            v.consenso_step(t)
            # point.control_step(t)
            P[v.id].append(v.din.p)
            avg[v.id].append(v.promedio.x)

    return P, E, avg


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
    red = grafo(directed=False)
    red.agregar_vehiculos([point(i) for i in range(N)])
    red.iniciar_dinamica({
        0: [0, 0.],
        1: [15., 0],
        2: [-10, 15.]
    })
    red.iniciar_consenso({
        0: [10.],
        1: [30.],
        2: [50.]
    })

    rango_max = np.sqrt(20.**2 + 20.**2)

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    P, E, avg = run(tiempo, red, rango_max)
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
      t, P, E, save=arg.save)
    if arg.animate:
        graph_plotter.animation2d(
            step=1, plot_kw={'xlim': [-40, 40], 'ylim': [-40, 40]})
