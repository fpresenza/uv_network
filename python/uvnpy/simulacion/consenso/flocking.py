#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date vie oct 30 15:12:32 -03 2020
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
    avg = dict([(v.id, [v.promedio.x]) for v in red.vehiculos])
    E = [red.enlaces]

    for t in tiempo[1:]:
        red.reconectar(proximidad, rango_max)
        E.append(red.enlaces)
        red.intercambiar()
        for v in red.vehiculos:
            v.consenso_promedio_step(t)
            v.box.limpiar_entrada()
            cmd = v.promedio.x
            v.din.step(t, cmd)

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
    pi = dict([(v.id, np.random.uniform(-15, 15, 2)) for v in red.vehiculos])
    vi = dict([(v.id, np.random.normal(0, 5, 2)) for v in red.vehiculos])

    red.iniciar_dinamica(
        pi=pi,
        vi=vi)
    red.iniciar_consenso_promedio(vi)

    rango_max = np.sqrt(10.**2 + 10.**2)

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    P, E, avg = run(tiempo, red, rango_max)
    t = tiempo

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.5, wspace=0.25)
    gs = fig.add_gridspec(2, 1)
    fig.suptitle('Consenso - flocking')
    avg_x = plotting.agregar_ax(
        gs[0, 0],
        xlabel='t [seg]', ylabel='$v_x$ [m/s]', label_kw={'fontsize': 10})
    avg_y = plotting.agregar_ax(
        gs[1, 0],
        xlabel='t [seg]', ylabel='$v_y$ [m/s]', label_kw={'fontsize': 10})
    for key, value in avg.items():
        x, y = zip(*value)
        plotting.agregar_linea(avg_x, t, x)
        plotting.agregar_linea(avg_y, t, y)

    if arg.save:
        fig.savefig('/tmp/flocking.pdf', format='pdf')
    else:
        plt.show()

    graph_plotter = plotting.GraphPlotter(
      t, P, E, save=arg.save)
    if arg.animate:
        graph_plotter.animation2d(
            step=1, plot_kw={'xlim': [-40, 40], 'ylim': [-40, 40]})
