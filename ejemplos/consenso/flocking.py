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
from uvnpy.filtering import consenso
from uvnpy.modelos import vehiculo, doble_integrador
from uvnpy.redes import grafo, mensajeria

normal = np.random.multivariate_normal


def proximidad(i, j, rango_max):
    p_i, p_j = i.p, j.p
    diff = p_i - p_j
    dist = np.sqrt(diff.dot(diff))
    return dist <= rango_max


def mensajes(v_i, v_j):
    msg_i = v_i.box.salida
    msg_j = v_j.box.salida
    v_i.box.recibir(msg_j)
    v_j.box.recibir(msg_i)


class agente(vehiculo):
    def __init__(self, nombre, xi=np.zeros(2)):
        super(agente, self).__init__(nombre, tipo='agente')
        # dinamica del vehiculo
        self.din = doble_integrador(xi=xi)
        # intercambio de mensajes
        self.box = mensajeria.box(out={'id': self.id}, maxlen=30)

    @property
    def p(self):
        return self.din._x[:2]

    @property
    def v(self):
        return self.din._x[2:]

    def step(self, t):
        v_i = self.din._x[2:]
        v_j = self.box.extraer('flocking')
        cmd_i = 0.2 * consenso.promedio.dinamica(v_i, t, v_j)
        self.box.actualizar('flocking', v_i)
        self.box.limpiar()
        self.din.step(t, cmd_i)


def run(tiempo, red, rango_max):
    # logs
    P = dict([(v.id, [v.p]) for v in red.vehiculos])
    V = dict([(v.id, [v.v]) for v in red.vehiculos])
    E = [red.enlaces]

    for t in tiempo[1:]:
        red.reconectar(proximidad, rango_max)
        E.append(red.enlaces)
        red.intercambiar(mensajes)
        for v in red.vehiculos:
            v.step(t)

            P[v.id].append(v.p)
            V[v.id].append(v.v)

    return P, E, V


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
    med, cov = np.zeros(4), np.diag([15., 15., 9., 9.])
    red.agregar_vehiculos(
        [agente(i, xi=normal(med, cov)) for i in range(N)])

    for v in red.vehiculos:
        v.box.actualizar('flocking', v.v)

    rango_max = np.sqrt(10.**2 + 10.**2)

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    P, E, V = run(tiempo, red, rango_max)
    t = tiempo

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.5, wspace=0.25)
    gs = fig.add_gridspec(2, 1)
    fig.suptitle('Consenso - flocking')
    V_x = plotting.agregar_ax(
        gs[0, 0],
        xlabel='t [seg]', ylabel='$v_x$ [m/s]', label_kw={'fontsize': 10})
    V_y = plotting.agregar_ax(
        gs[1, 0],
        xlabel='t [seg]', ylabel='$v_y$ [m/s]', label_kw={'fontsize': 10})
    for key, value in V.items():
        x, y = zip(*value)
        plotting.agregar_linea(V_x, t, x)
        plotting.agregar_linea(V_y, t, y)

    if arg.save:
        fig.savefig('/tmp/flocking.pdf', format='pdf')
    else:
        plt.show()

    graph_plotter = plotting.GraphPlotter(
      t, P, E, save=arg.save)
    if arg.animate:
        graph_plotter.animation2d(
            step=1, plot_kw={'xlim': [-40, 40], 'ylim': [-40, 40]})
