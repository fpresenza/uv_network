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
from uvnpy.modelos import vehiculo, integrador
from uvnpy.filtering import consenso
from uvnpy.redes import grafo, mensajeria
from uvnpy.filtering import similaridad, kalman

normal = np.random.multivariate_normal


def proximidad(v_i, v_j, rango_max):
    p_i, p_j = v_i.din.x, v_j.din.x
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
        self.vector = consenso.promedio()
        self.matriz = consenso.promedio()

    def step(self, t):
        y_j = self.box.extraer('vector')
        Y_j = self.box.extraer('matriz')
        self.vector.step(t, ([y_j], ))
        self.matriz.step(t, ([Y_j], ))
        self.box.actualizar('vector', self.vector.x)
        self.box.actualizar('matriz', self.matriz.x)
        self.box.limpiar()


def run(tiempo, red, rango_max):
    # logs
    Pos = dict([(v.id, [v.din.x]) for v in red.vehiculos])
    avg = dict([(v.id, [v.vector.x]) for v in red.vehiculos])
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
        red.intercambiar(mensajes)
        color = ['0.3', '0.3', '0.3']
        xs = []
        Ps = []
        for v in red.vehiculos:
            v.step(t)

            y = 3 * v.vector.x
            Y = 3 * v.matriz.x
            x, P = similaridad(y, Y)

            xs.append(x)
            Ps.append(P)

            Pos[v.id].append(v.din.x)
            avg[v.id].append(v.vector.x)

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

    arg = parser.parse_args()

    # ------------------------------------------------------------------
    # Configuración
    # ------------------------------------------------------------------
    tiempo = np.arange(arg.ti, arg.tf, arg.h)

    red = grafo.grafo(directed=False)
    red.agregar_vehiculos([
        agente(0, xi=[0, 0.]),
        agente(1, xi=[15, 0.]),
        agente(2, xi=[-10, 15.])
        ])

    p = [0, 0]

    P1 = np.array([[20, 2.5],
                   [2.5, 10]])**2
    P2 = np.array([[15, 1],
                   [1, 10]])**2
    P3 = np.array([[15,  5],
                   [5,  25]])**2

    x1 = normal(p, P1)
    x2 = normal(p, P2)
    x3 = normal(p, P3)
    print('x\n', x1, '\n', x2, '\n', x3)

    y1, Y1 = similaridad(x1, P1)
    y2, Y2 = similaridad(x2, P2)
    y3, Y3 = similaridad(x3, P3)
    yi = [y1, y2, y3]
    Yi = [Y1, Y2, Y3]

    # print('v\n', y1, '\n', y2, '\n', y3)
    yf, Yf = kalman.fusionar(yi, Yi)
    xf, Pf = similaridad(yf, Yf)

    for v in red.vehiculos:
        y = yi[v.id]
        Y = Yi[v.id]
        v.vector.iniciar(y, 0.)
        v.box.actualizar('vector', y)

        v.matriz.iniciar(Y, 0.)
        v.box.actualizar('matriz', Y)

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
