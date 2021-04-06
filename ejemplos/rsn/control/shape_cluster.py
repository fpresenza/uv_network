#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on lun mar 29 15:50:59 -03 2021
@author: fran
"""
import argparse
import collections
import time
import progressbar
import numpy as np
import matplotlib.pyplot as plt

from gpsic.plotting.core import agregar_ax
from gpsic.grafos.plotting import animar_grafo
from uvnpy.modelos.lineal import integrador
import uvnpy.network.graph as gph
import uvnpy.rsn.core as rsn

# ------------------------------------------------------------------
# Definición de variables globales, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple('Logs', 'x u y_ref y')


def measure(A, p):
    dist = np.triu(A * rsn.distances(p))
    y = dist[dist > 0]
    return y


def grid(nv, sep):
    k = np.ceil(np.sqrt(nv)) / 2
    nums = np.arange(-k, k) * sep
    g = np.meshgrid(nums, nums)
    return np.vstack(np.dstack(g))[:nv]


def linspace(nv, lim):
    p = np.empty((nv, 2))
    p[:, 0] = np.linspace(-lim, lim, nv)
    p[:, 1] = 0
    return p


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(steps, logs, t_perf, planta, cuadros):
    # iteración
    bar = progressbar.ProgressBar(max_value=arg.tf).start()
    for k, t in steps[1:]:
        # step planta
        x = planta.x

        # Control
        t_a = time.perf_counter()

        y = measure(A, x)
        u_y = y_ref - y

        H = rsn.distances_jac(A, x)
        Hpinv = np.linalg.pinv(H)
        u = Hpinv.dot(u_y).reshape(nv, dof)

        t_b = time.perf_counter()
        x = planta.step(t, u)
        E = gph.edges_from_adjacency(np.triu(A))
        cuadros[k] = x, E

        logs.x[k] = x
        logs.u[k] = u
        logs.y_ref[k] = y_ref
        logs.y[k] = y

        t_perf.append(t_b - t_a)
        bar.update(np.round(t, 3))

    bar.finish()

    # return
    return logs, t_perf, cuadros


if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Parseo de argumentos
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-s', '--step',
        dest='h', default=20e-3, type=float, help='paso de simulación')
    parser.add_argument(
        '-t', '--ti',
        metavar='T0', default=0.0, type=float, help='tiempo inicial')
    parser.add_argument(
        '-e', '--tf',
        default=1.0, type=float, help='tiempo final')
    parser.add_argument(
        '-r', '--arxiv',
        default='/tmp/config.yaml', type=str, help='arhivo de configuración')
    parser.add_argument(
        '-g', '--save',
        default=False, action='store_true',
        help='flag para guardar archivos')
    parser.add_argument(
        '-a', '--animate',
        default=False, action='store_true',
        help='flag para generar animacion')
    parser.add_argument(
        '-n', '--agents', dest='n',
        default=1, type=int, help='cantidad de agentes')

    arg = parser.parse_args()

    # ------------------------------------------------------------------
    # Configuración
    # ------------------------------------------------------------------
    tiempo = np.arange(arg.ti, arg.tf, arg.h)
    steps = list(enumerate(tiempo))
    t_perf = []

    lim = 20.
    nv = arg.n
    A = 1 - np.eye(nv)
    # A = np.zeros((nv, nv))
    # A[0, 1] = A[1, 0] = 1
    ne = np.count_nonzero(np.triu(A))
    V = range(nv)
    dof = 2
    n = dof * nv
    dmax = 12
    np.random.seed(5)
    x0 = np.random.uniform(-lim, lim, (nv, dof))
    # x0 = np.array([[-5, 0],
    #                [0, -5],
    #                [5, 0],
    #                [0, 5]], dtype=np.float) * 1.5
    # x0 = grid(nv, 5)
    # x0 = linspace(nv, lim*0.75) + np.random.normal(0, 0.5, (nv, dof))
    y_ref = 5.

    planta = integrador(x0, tiempo[0])

    logs = Logs(
        x=np.empty((tiempo.size, nv, dof)),
        u=np.empty((tiempo.size, nv, dof)),
        y_ref=np.empty((tiempo.size, ne)),
        y=np.empty((tiempo.size, ne))
        )
    logs.x[0] = x0
    logs.u[0] = np.zeros((nv, dof))
    logs.y_ref[0] = y_ref
    logs.y[0] = measure(A, x0)

    cuadros = np.empty((tiempo.size, 2), dtype=np.ndarray)
    E0 = gph.complete_undirected_edges(V)
    # E0 = gph.undirected_edges(gph.disk_graph_edges(x0, dmax))
    cuadros[0] = x0, E0

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    logs, t_perf, cuadros = run(steps, logs, t_perf, planta, cuadros)

    x = logs.x
    u = logs.u
    y_ref = logs.y_ref
    y = logs.y
    st = arg.tf - arg.ti
    rt = sum(t_perf)
    prompt = 'RT={:.3f} secs, ST={:.3f} secs  ==>  RTF={:.3f}'
    print(prompt.format(rt, st, st / rt))

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(13, 5))
    fig.subplots_adjust(hspace=0.5, wspace=0.25)
    gs = fig.add_gridspec(dof, 2)

    ax = agregar_ax(
        gs[0, 0],
        title='posición',
        xlabel='t [seg]', ylabel='x [m]', label_kw={'fontsize': 10})
    ax.plot(tiempo, x[..., 0], ds='steps')
    ax = agregar_ax(
        gs[1, 0],
        xlabel='t [seg]', ylabel='y [m]', label_kw={'fontsize': 10})
    ax.plot(tiempo, x[..., 1], ds='steps')
    ax = agregar_ax(
        gs[0, 1],
        title='acción de control',
        xlabel='t [seg]', ylabel='$u_x$ [m/s]', label_kw={'fontsize': 10})
    ax.plot(tiempo, u[..., 0], ds='steps')
    ax = agregar_ax(
        gs[1, 1],
        xlabel='t [seg]', ylabel='$u_y$ [m/s]', label_kw={'fontsize': 10})
    ax.plot(tiempo, u[..., 1], ds='steps')

    if arg.save:
        fig.savefig('/tmp/control.pdf', format='pdf')
    else:
        plt.show()

    fig = plt.figure(figsize=(7, 5))
    gs = fig.add_gridspec(1, 1)

    ax = agregar_ax(
        gs[0, 0],
        title='distancias',
        xlabel='t [seg]', ylabel='d [m]', label_kw={'fontsize': 10})
    ax.plot(tiempo, y_ref, ls='--', ds='steps')
    ax.plot(tiempo, y, ds='steps')

    if arg.save:
        fig.savefig('/tmp/distancias.pdf', format='pdf')
    else:
        plt.show()

    if arg.animate:
        estilos = (
            [V, {'color': 'b', 'marker': 'o', 'markersize': '5'}], )
        fig, ax = plt.subplots()
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.grid(1)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        animar_grafo(
            fig, ax, arg.h, estilos, cuadros,
            edgestyle={'color': '0.2', 'linewidth': 0.7},
            guardar=arg.save,
            archivo='/tmp/animacion.mp4')
