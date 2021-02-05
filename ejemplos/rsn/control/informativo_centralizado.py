#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on jue feb  4 20:05:46 -03 2021
@author: fran
"""
import argparse
import collections
import time
import progressbar
import numpy as np
import matplotlib.pyplot as plt

from gpsic.plotting.planar import agregar_ax
from gpsic.grafos.plotting import animar_grafo
from uvnpy.modelos.lineal import integrador
import uvnpy.redes.core as redes
import uvnpy.rsn.core as rsn
from uvnpy.control import informativo
from uvnpy.filtering import metricas

# ------------------------------------------------------------------
# Definición de variables globales, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple('Logs', 'p u J eig')


def innovacion(x):
    p = x.reshape(-1, 2)
    A = rsn.distances(p)
    A[A != 0] **= -2
    L = rsn.distances_innovation_laplacian(A, p)
    return L


def innovacion_acumulada(x_p):
    return sum([innovacion(x) for x in x_p])


def funcional(Y):
    eigvals = metricas.eigvalsh(Y)
    rsd = metricas.relative_standard_deviation(eigvals)
    return rsd


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(steps, logs, t_perf, planta, cuadros):
    # iteración
    bar = progressbar.ProgressBar(max_value=arg.tf).start()
    for k, t in steps[1:]:
        # u = np.zeros((nv, dof))
        p = planta.x
        a = time.perf_counter()
        u = ctrl.update(p.ravel(), t, ()).reshape(nv, dof)
        b = time.perf_counter()
        p = planta.step(t, u)

        Y = innovacion(p)
        J = funcional(Y)
        eigvals = metricas.eigvalsh(Y)

        E = redes.edges_from_positions(p, dmax)
        cuadros[k] = p, redes.undirected_edges(E)

        logs.p[k] = p
        logs.u[k] = u
        logs.J[k] = J
        logs.eig[k] = eigvals

        t_perf.append(b - a)
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
        dest='h', default=50e-3, type=float, help='paso de simulación')
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
    V = range(nv)
    dof = 2
    n = dof * nv
    dmax = 12.
    # p0 = np.random.uniform(-lim, lim, (nv, dof))
    p0 = np.array([[-5, 0],
                   [0, -5],
                   [5, 0],
                   [0, 5]], dtype=np.float)
    planta = integrador(p0, tiempo[0])

    ctrl = informativo.minimizar(
        metrica=funcional,
        matriz=innovacion_acumulada,
        modelo=integrador,
        Q=(0.4 * np.eye(n), 1 * np.eye(n), 1000),
        dim=n,
        horizonte=np.linspace(0.1, 0.5, 10)
        )

    logs = Logs(
        p=np.empty((tiempo.size, nv, dof)),
        u=np.empty((tiempo.size, nv, dof)),
        J=np.empty((tiempo.size)),
        eig=np.empty((tiempo.size, n))
        )
    logs.p[0] = p0
    logs.u[0] = np.zeros((nv, dof))
    logs.J[0] = None
    logs.eig[0] = None

    cuadros = np.empty((tiempo.size, 2), dtype=np.ndarray)
    E0 = redes.edges_from_positions(p0, dmax)
    cuadros[0] = p0, redes.undirected_edges(E0)

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    logs, t_perf, cuadros = run(steps, logs, t_perf, planta, cuadros)

    p = logs.p
    u = logs.u
    J = logs.J
    eig = logs.eig

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
        xlabel='t [seg]', ylabel='x [m]', label_kw={'fontsize': 10})
    ax.plot(tiempo, p[..., 0])
    ax = agregar_ax(
        gs[1, 0],
        xlabel='t [seg]', ylabel='y [m]', label_kw={'fontsize': 10})
    ax.plot(tiempo, p[..., 1])
    ax = agregar_ax(
        gs[0, 1],
        xlabel='t [seg]', ylabel='$u_x$ [m]', label_kw={'fontsize': 10})
    ax.plot(tiempo, u[..., 0])
    ax = agregar_ax(
        gs[1, 1],
        xlabel='t [seg]', ylabel='$u_y$ [m]', label_kw={'fontsize': 10})
    ax.plot(tiempo, u[..., 1])

    fig = plt.figure(figsize=(13, 5))
    fig.subplots_adjust(hspace=0.5, wspace=0.25)
    gs = fig.add_gridspec(2, 1)

    ax = agregar_ax(
        gs[0, 0],
        xlabel='t [seg]', ylabel='rsd', label_kw={'fontsize': 10})
    ax.plot(tiempo, J)

    ax = agregar_ax(
        gs[1, 0],
        xlabel='t [seg]', ylabel='eigvals', label_kw={'fontsize': 10})
    ax.plot(tiempo, eig)

    if arg.save:
        fig.savefig('/tmp/ensayo.pdf', format='pdf')
    else:
        plt.show()

    if arg.animate:
        estilo = ([V, {'color': 'b', 'marker': 'o', 'markersize': '5'}], )
        fig, ax = plt.subplots()
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.grid(1)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        animar_grafo(fig, ax, arg.h, estilo, cuadros, guardar=arg.save)
