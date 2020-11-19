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
from uvnpy.modelos import vehiculo, integrador
from uvnpy.filtering import consenso
from uvnpy.redes import grafo, mensajeria

normal = np.random.multivariate_normal


def proximidad(i, j, rango_max):
    p_i, p_j = i.din.x, j.din.x
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
        self.din = integrador(xi=xi)
        # intercambio de mensajes
        self.box = mensajeria.box(out={'id': self.id}, maxlen=30)
        self.lpf = consenso.lpf()

    def step(self, t, u):
        x_j = self.box.extraer('x')
        u_j = self.box.extraer('u')
        self.lpf.step(t, ([u, x_j, u_j], ))
        self.box.actualizar('x', self.lpf.x)
        self.box.actualizar('u', u)
        self.box.limpiar()


def run(tiempo, red, rango_max, U, signal):
    # logs
    P = dict([(v.id, [v.din.x]) for v in red.vehiculos])
    lpf = dict([(v.id, [v.lpf.x]) for v in red.vehiculos])
    E = [red.enlaces]

    for t in tiempo[1:]:
        red.reconectar(proximidad, rango_max)
        E.append(red.enlaces)
        red.intercambiar(mensajes)
        for v in red.vehiculos:
            u = signal(t, v.id)
            v.step(t, u)

            P[v.id].append(v.din.x)
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
        f = [np.cos(t), np.sin(t)]
        # f = [np.sin(t)]
        # f = [1.]
        return np.random.normal(f, sigma)

    N = arg.agents
    red = grafo.grafo(directed=False)
    med, cov = np.zeros(2), np.diag([10., 10.])
    red.agregar_vehiculos(
        [agente(i, xi=normal(med, cov)) for i in range(N)])

    U = {}

    for v in red.vehiculos:
        u = signal(0, v.id)
        U[v.id] = [u]
        x = np.zeros_like(u)
        v.lpf.iniciar(x, 0.)
        v.box.actualizar('x', x)
        v.box.actualizar('u', u)

    rango_max = np.sqrt(10.**2 + 10.**2)

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    P, E, U, lpf = run(tiempo, red, rango_max, U, signal)
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
