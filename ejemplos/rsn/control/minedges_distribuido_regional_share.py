#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on lun abr  5 16:16:07 -03 2021
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
from gpsic.toolkit import linalg
from uvnpy.modelos.lineal import integrador
import uvnpy.network.core as network
import uvnpy.network.disk_graph as disk_graph
import uvnpy.network.connectivity as connectivity
import uvnpy.rsn.core as rsn
import uvnpy.rsn.distances as distances
import uvnpy.toolkit.calculus as calc

# ------------------------------------------------------------------
# Definición de variables globales, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple('Logs', 'x u J eig avg_dist')

D = calc.derivative_eval
lsd = connectivity.logistic_strength_derivative


def detFi(p):
    Ai = distances.matrix(p)
    Ai[Ai > 0] = connectivity.logistic_strength(Ai[Ai > 0], beta=beta_2, e=e_2)

    M = rsn.pose_and_shape_decomposition(p.mean(0))
    Mf = M[:, 3:]

    Yi = distances.innovation_matrix(Ai, p)
    Fi = np.matmul(Mf.T, np.matmul(Yi, Mf))
    return np.linalg.det(Fi)


def keep_rigid(p):
    u = - a * detFi(p[None])**(-a - 1) * D(detFi, p)
    return -u.reshape(p.shape)


def min_edges(p):
    w = distances.matrix(p)
    w[w > 0] = lsd(w[w > 0], beta=beta_1, e=e_1)
    u = distances.edge_potencial_gradient(w, p)
    return -u.reshape(p.shape)


def repulsion(p):
    w = distances.matrix(p)
    w[w > 0] = connectivity.power_strength_derivative(w[w > 0], a=1)
    u = distances.edge_potencial_gradient(w, p)
    return -u.reshape(p.shape)


def grid(n, sep):
    k = np.ceil(np.sqrt(n)) / 2
    nums = np.arange(-k, k) * sep
    g = np.meshgrid(nums, nums)
    return np.vstack(np.dstack(g))[:n]


def linspace(n, dmax):
    p = np.empty((n, 2))
    p[:, 0] = np.linspace(-dmax, dmax, n)
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
        t_a = np.empty(n)
        t_b = np.empty(n)
        u[:] = 0
        for i in V:
            t_a[i] = time.perf_counter()

            R[i] = disk_graph.neighborhood_histeresis(x, i, R[i], dmin, dmax, inclusive=True)  # noqa
            p = x[R[i]]

            u[R[i]] += keep_rigid(p) + 0.5 * min_edges(p) + (2 / n) * repulsion(p)  # noqa

            t_b[i] = time.perf_counter()

        x = planta.step(t, u)

        # Análisis
        A = R.astype(int) - np.eye(n)
        Y = distances.innovation_matrix(A, x)

        M = rsn.pose_and_shape_decomposition(x)
        Mf = M[:, 3:]
        F = Mf.T.dot(Y).dot(Mf)
        J = np.abs(np.linalg.det(F))**a
        eigvals = np.linalg.eigvalsh(F)

        E = network.edges_from_adjacency(A)
        cuadros[k] = x, E

        logs.x[k] = x
        logs.u[k] = u
        logs.J[k] = (J, len(E))
        logs.eig[k] = eigvals
        logs.avg_dist[k] = linalg.norma2(x[E[:, 0]] - x[E[:, 1]], axis=1).mean()  # noqa

        t_perf.append(np.mean(t_b - t_a))
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

    n = arg.n
    a = 2 / n   # 32 / n**2
    V = range(n)
    dof = 2
    pn = dof * n
    dmax = 10
    dmin = 0.7 * dmax
    beta_1 = 10 / dmax
    beta_2 = 40 / dmax
    e_1 = dmax
    e_2 = 0.7 * dmax

    np.random.seed(1)
    x0 = np.random.uniform(-0.5 * dmax, 0.5 * dmax, (n, dof))
    A0 = disk_graph.adjacency(x0, dmax)
    u = np.zeros((n, dof))
    R = (A0 + np.eye(n)).astype(bool)

    planta = integrador(x0, tiempo[0])

    for i in V:
        p = x0[R[i]]

        Ai = disk_graph.adjacency(p, dmax)
        if distances.rigidity(Ai, p):
            print('Grafo {} rígido.'.format(i))
        else:
            print('Warning!: Grafo {} flexible.'.format(i))

    logs = Logs(
        x=np.empty((tiempo.size, n, dof)),
        u=np.empty((tiempo.size, n, dof)),
        J=np.empty((tiempo.size, 2)),
        eig=np.empty((tiempo.size, pn - 3)),
        avg_dist=np.empty(tiempo.size)
        )
    logs.x[0] = x0
    logs.u[0] = np.zeros((n, dof))
    logs.J[0] = None
    logs.eig[0] = None
    logs.avg_dist[0] = None

    cuadros = np.empty((tiempo.size, 2), dtype=np.ndarray)
    E0 = disk_graph.edges(x0, dmax)
    cuadros[0] = x0, E0

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    logs, t_perf, cuadros = run(steps, logs, t_perf, planta, cuadros)

    x = logs.x
    u = logs.u
    J = logs.J
    eig = logs.eig
    avg_dist = logs.avg_dist

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

    fig = plt.figure(figsize=(13, 5))
    fig.subplots_adjust(hspace=0.5, wspace=0.25)
    gs = fig.add_gridspec(2, 2)

    ax = agregar_ax(
        gs[0, 0],
        xlabel='t [seg]', ylabel=r'$det(F)^a$', label_kw={'fontsize': 10})
    ax.semilogy(tiempo, J[:, 0], ds='steps')
    # ax.set_ylim(bottom=0)
    # ax.legend()

    ax = agregar_ax(
        gs[0, 1],
        xlabel='t [seg]', ylabel=r'$|\mathcal{E}|$', label_kw={'fontsize': 10})
    ax.plot(tiempo, J[:, 1], ds='steps')
    ax.set_ylim(bottom=0)
    # ax.legend()

    ax = agregar_ax(
        gs[1, 0],
        xlabel='t [seg]', ylabel=r'$\lambda(F)$',
        label_kw={'fontsize': 10})
    ax.semilogy(tiempo, eig, ds='steps')
    # ax.set_ylim(bottom=0)

    ax = agregar_ax(
        gs[1, 1],
        xlabel='t [seg]', ylabel=r'$\rm{avg}(d_{ij}) / d_{\rm{max}}$',
        label_kw={'fontsize': 10})
    ax.plot(tiempo, avg_dist / dmax, ds='steps')
    ax.hlines(
        [dmin / dmax, 1],
        xmin=0, xmax=tiempo[-1], ls='--', color='k', alpha=0.7)
    ax.set_ylim(bottom=0)

    if arg.save:
        fig.savefig('/tmp/metricas.pdf', format='pdf')
    else:
        plt.show()

    if arg.animate:
        estilos = (
            [V, {'color': 'b', 'marker': 'o', 'markersize': '5'}], )
        fig, ax = plt.subplots()
        ax.set_xlim(-1.5 * dmax, 1.5 * dmax)
        ax.set_ylim(-1.5 * dmax, 1.5 * dmax)
        ax.set_aspect('equal')
        ax.grid(1)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        animar_grafo(
            fig, ax, arg.h, estilos, cuadros,
            edgestyle={'color': '0.2', 'linewidth': 0.7},
            guardar=arg.save,
            archivo='/tmp/animacion.mp4')
