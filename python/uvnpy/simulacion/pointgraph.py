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


desplegar = np.random.uniform


def run(tiempo, red, rango_max):
    # logs
    # x = [points[0].kin.x]
    # u = [points[0].control.u]
    P = dict([(v.id, [v.kin.p]) for v in red.vehiculos])
    V = dict([(v.id, np.random.normal(0, 1, 2)) for v in red.vehiculos])
    E = [red.enlaces]

    for t in tiempo[1:]:
        for v in red.vehiculos:
            v.kin.step(t, V[v.id])
            # point.control_step(t)
            P[v.id].append(v.kin.p)

        red.reconectar(proximidad, rango_max)
        E.append(red.enlaces)
        # x.append(points[0].kin.x)
        # u.append(points[0].control.u)
        # points[0].filtro.guardar()

    return P, E


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
    N = arg.agents
    agentes = [point(i, pi=desplegar(-1, 1, 2)) for i in range(N)]
    red = grafo(directed=False)
    red.agregar_vehiculos(agentes)

    tiempo = np.arange(arg.ti, arg.tf, arg.h)

    rango_max = 10.

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    P, E = run(tiempo, red, rango_max)
    t = tiempo
    # variables
    # x = np.vstack(x)
    # f_x = np.vstack(f.x)
    # f_dvst = np.vstack(f.dvst)
    # f_eigs = np.vstack(f.eigs)
    # u = np.vstack(u)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    graph_plotter = plotting.GraphPlotter(
      t, P, E, save=arg.save)
    if arg.animate:
        graph_plotter.animation2d(
          step=1, plot_kw={'xlim': [-40, 40], 'ylim': [-40, 40]})
