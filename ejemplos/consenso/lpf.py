#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date vie oct 30 18:32:31 -03 2020
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt  # noqa

import gpsic.plotting.planar as plotting
from uvnpy.modelos import point
from uvnpy.redes import grafo

proximidad = grafo.proximidad


def run(tiempo, red, rango_max, ui, signal):
    # logs
    P = dict([(v.id, [v.din.p]) for v in red.vehiculos])
    U = dict([(v.id, [ui[v.id]]) for v in red.vehiculos])
    lpf = dict([(v.id, [v.lpf.x]) for v in red.vehiculos])
    E = [red.enlaces]

    for t in tiempo[1:]:
        red.reconectar(proximidad, rango_max)
        E.append(red.enlaces)
        red.intercambiar()
        for v in red.vehiculos:
            u = signal(t, v.id)
            v.consenso_lpf_step(t, u)
            v.box.limpiar_entrada()

            P[v.id].append(v.din.p)
            U[v.id].append(u)
            lpf[v.id].append(v.lpf.x)

    return P, E, U, lpf


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

    sigma = 0.25

    def signal(t, v_id):
        # f = [np.cos(t), np.sin(t)]
        # f = [np.sin(t)]
        f = [1.]
        return np.random.normal(f, sigma)

    N = arg.agents
    red = grafo.grafo(directed=False)
    red.agregar_vehiculos([point(i) for i in range(N)])

    pi = dict([(v.id, np.random.uniform(-5, 5, 2)) for v in red.vehiculos])
    vi = dict([(v.id, [0., 0.]) for v in red.vehiculos])
    ui = dict([(v.id, signal(0, v.id)) for v in red.vehiculos])
    lpfi = dict([
        (v.id, {
            'x': np.zeros_like(ui[v.id]),
            'u': ui[v.id]})
        for v in red.vehiculos])

    red.iniciar_dinamica(
        pi=pi,
        vi=vi)
    red.iniciar_consenso_lpf(lpfi)

    rango_max = np.sqrt(10.**2 + 10.**2)

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    P, E, U, lpf = run(tiempo, red, rango_max, ui, signal)
    t = tiempo
    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.5, wspace=0.25)
    gs = fig.add_gridspec(2, 1)
    fig.suptitle(r'Consenso - lpf ($\sigma$={})'.format(sigma))
    ax_u = plotting.agregar_ax(
        gs[0, 0],
        xlabel='t [seg]', ylabel='$u$', label_kw={'fontsize': 10})
    for key, value in U.items():
        for u_i in zip(*value):
            plotting.agregar_linea(ax_u, t, u_i)

    lpf_x = plotting.agregar_ax(
        gs[1, 0],
        xlabel='t [seg]', ylabel='$x$', label_kw={'fontsize': 10})
    for key, value in lpf.items():
        for x_i in zip(*value):
            plotting.agregar_linea(lpf_x, t, x_i)

    if arg.save:
        fig.savefig('/tmp/lpf.pdf', format='pdf')
    else:
        plt.show()

    graph_plotter = plotting.GraphPlotter(
      t, P, E, save=arg.save)
    if arg.animate:
        graph_plotter.animation2d(
            step=1, plot_kw={'xlim': [-40, 40], 'ylim': [-40, 40]})
