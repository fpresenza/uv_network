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
        self.comparador = consenso.comparador()

    def step(self):
        norma_j = self.box.extraer('norma')
        p_j = self.box.extraer('p')
        norma_max, p_max, flag = self.comparador.step(norma_j, p_j)
        self.box.actualizar('norma', norma_max)
        self.box.actualizar('p', p_max)
        self.box.limpiar()
        return norma_max, p_max, flag


def run(tiempo, red, rango_max):
    # logs
    P = dict([(v.id, [v.din.x]) for v in red.vehiculos])
    max_x = dict([(v.id, [v.comparador.x]) for v in red.vehiculos])
    max_u = dict([(v.id, [v.comparador.u]) for v in red.vehiculos])
    max_flag = dict([(v.id, [v.comparador.flag]) for v in red.vehiculos])
    E = [red.enlaces]

    for t in tiempo[1:]:
        red.reconectar(proximidad, rango_max)
        E.append(red.enlaces)
        red.intercambiar(mensajes)
        for v in red.vehiculos:
            norma_max, p_max, flag = v.step()

            P[v.id].append(v.din.x)
            max_x[v.id].append(norma_max)
            max_u[v.id].append(p_max)
            max_flag[v.id].append(flag)

    return P, E, max_x, max_u, max_flag


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
    med, cov = np.zeros(2), np.diag([10., 10.])
    red.agregar_vehiculos(
        [agente(i, xi=normal(med, cov)) for i in range(N)])

    for v in red.vehiculos:
        p = v.din.x
        norma = np.linalg.norm(p)
        print(v.id, norma, p)
        v.comparador.iniciar(norma, p, max)
        v.box.actualizar('norma', norma)
        v.box.actualizar('p', p)

    rango_max = np.sqrt(20.**2 + 20.**2)

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    P, E, max_x, max_u, max_flag = run(tiempo, red, rango_max)
    t = tiempo

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.5, wspace=0.25)
    gs = fig.add_gridspec(3, 1)
    fig.suptitle('Consenso - max')
    ax_x = plotting.agregar_ax(
        gs[0, 0],
        xlabel='t [seg]', ylabel='$x$', label_kw={'fontsize': 10})
    for key, value in max_x.items():
        plotting.agregar_linea(ax_x, t, value)

    ax_u = plotting.agregar_ax(
        gs[1, 0],
        xlabel='t [seg]', ylabel='$u$', label_kw={'fontsize': 10})
    for key, value in max_u.items():
        for u_i in zip(*value):
            plotting.agregar_linea(ax_u, t, u_i)

    ax_flag = plotting.agregar_ax(
        gs[2, 0],
        xlabel='t [seg]', ylabel='$flag$', label_kw={'fontsize': 10})
    for i, flag in max_flag.items():
        plotting.agregar_linea(ax_flag, t, flag, label=i)

    if arg.save:
        fig.savefig('/tmp/max.pdf', format='pdf')
    else:
        plt.show()

    graph_plotter = plotting.GraphPlotter(
      t, P, E, save=arg.save)
    if arg.animate:
        graph_plotter.animation2d(
            step=1, plot_kw={'xlim': [-40, 40], 'ylim': [-40, 40]})
