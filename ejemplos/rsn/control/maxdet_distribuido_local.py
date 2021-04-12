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
from uvnpy.modelos.lineal import integrador
import uvnpy.network.core as net
import uvnpy.network.disk_graph as disk_graph
import uvnpy.network.connectivity as cnt
import uvnpy.rsn.core as rsn
import uvnpy.rsn.distances as distances
import uvnpy.toolkit.calculus as calc

# ------------------------------------------------------------------
# Definición de variables globales, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple('Logs', 'x u J Jp eig eigp')

D = calc.derivative_eval
repulsion = distances.local_edge_potencial_gradient

# metrica = r'$\rm{tr}(Y)$'
metrica = r'$\rm{log}(\rm{det}M(x))$'


def detFi(p, q, w):
    T = distances.local_innovation_matrix(p, q, w)
    return np.linalg.det(T)


def detFi_grad(p, q, w):
    u = a * detFi(p, q, w)**(a - 1) * D(detFi, p[0], q, w)
    return u.reshape(p.shape)


def logdetFi_grad(p, q, w):
    u = detFi(p, q, w)**(-1) * D(detFi, p[0], q, w)
    return u.reshape(p.shape)


def is_rigid(Ar, p):
    Yr = distances.innovation_matrix(Ar, p)
    return np.linalg.matrix_rank(Yr) >= p.size - 3


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

        dist = distances.distances(x)
        A = dist.copy()
        A[A > dmax] = 0
        A[A != 0] = 1
        _, S = rsn.pose_and_shape_basis_2d(x)

        t_a = np.empty(nv)
        t_b = np.empty(nv)
        for i in V:
            t_a[i] = time.perf_counter()
            p = x[i]
            q = np.delete(x, i, axis=0)
            q = disk_graph.local_neighbors(p, q, dmax)
            p = p[None]
            dij = distances.local_distances(p, q)

            wd = cnt.logistic_strength(dij, 1, e=dmax)
            wr = cnt.power_strength_derivative(dij, 4)
            u[i] = 15 * detFi_grad(p, q, wd) - 5 * repulsion(p, q, wr)
            t_b[i] = time.perf_counter()

        x = planta.step(t, u)

        # Análisis
        Aw = dist.copy()
        Aw[Aw > 0] = cnt.logistic_strength(Aw[Aw > 0], w=1, e=dmax)
        Y = distances.innovation_matrix(Aw, x)
        M = S.T.dot(Y).dot(S)
        # J = np.linalg.det(M)**a
        J = np.log(np.linalg.det(M))
        eigvals = np.linalg.eigvalsh(M)

        Yp = distances.innovation_matrix(A, x)
        Mp = S.T.dot(Yp).dot(S)
        # Jp = np.linalg.det(Mp)**a
        Jp = np.log(np.linalg.det(Mp))
        eigvalsp = np.linalg.eigvalsh(Mp)

        E = net.undirected_edges(disk_graph.edges(x, dmax))
        cuadros[k] = x, E

        logs.x[k] = x
        logs.u[k] = u
        logs.J[k] = J
        logs.Jp[k] = Jp
        logs.eig[k] = eigvals
        logs.eigp[k] = eigvalsp

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

    lim = 10.
    nv = arg.n
    a = 2 / nv
    V = range(nv)
    dof = 2
    n = dof * nv
    dmax = 12
    np.random.seed(9)
    x0 = np.random.uniform(-lim, lim, (nv, dof))
    u = np.zeros((nv, dof))
    # x0 = np.array([[-5, 0],
    #                [0, -5],
    #                [5, 0],
    #                [0, 5]], dtype=np.float) * 1.5
    # x0 = grid(nv, 5)
    # x0 = linspace(nv, lim*0.75) + np.random.normal(0, 0.5, (nv, dof))

    planta = integrador(x0, tiempo[0])

    A0 = disk_graph.adjacency(x0, dmax)
    if is_rigid(A0, x0):
        print('---> Grafo rígido <---')
    else:
        print('---> Grafo flexible <---')

    logs = Logs(
        x=np.empty((tiempo.size, nv, dof)),
        u=np.empty((tiempo.size, nv, dof)),
        J=np.empty((tiempo.size)),
        Jp=np.empty((tiempo.size)),
        eig=np.empty((tiempo.size, n - 3)),
        eigp=np.empty((tiempo.size, n - 3))
        )
    logs.x[0] = x0
    logs.u[0] = np.zeros((nv, dof))
    logs.J[0] = None
    logs.Jp[0] = None
    logs.eig[0] = None
    logs.eigp[0] = None

    cuadros = np.empty((tiempo.size, 2), dtype=np.ndarray)
    # E0 = net.complete_undirected_edges(V)
    E0 = net.undirected_edges(disk_graph.edges(x0, dmax))
    cuadros[0] = x0, E0

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    logs, t_perf, cuadros = run(steps, logs, t_perf, planta, cuadros)

    x = logs.x
    u = logs.u
    J = logs.J
    Jp = logs.Jp
    eig = logs.eig
    eigp = logs.eigp

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
    gs = fig.add_gridspec(2, 1)

    ax = agregar_ax(
        gs[0, 0],
        xlabel='t [seg]', ylabel=metrica, label_kw={'fontsize': 10})
    ax.plot(tiempo, J, color='C0', label='(ctrl)', ds='steps')
    ax.plot(tiempo, Jp, color='C1', label='(planta)', ds='steps')
    # ax.plot(
    #     tiempo, eig.prod(1),
    #     color='C2', label=r'$\rm{det}(Y)$ (ctrl)', ds='steps')
    # ax.plot(
    #     tiempo, eigp.prod(1),
    #     color='C3', label=r'$\rm{det}(Y)$ (planta)', ds='steps')
    ax.legend()

    ax = agregar_ax(
        gs[1, 0],
        xlabel='t [seg]', ylabel='eigvals', label_kw={'fontsize': 10})
    ax.plot(tiempo, eig[:, 0], color='C0', label='control', ds='steps')
    ax.plot(tiempo, eig[:, 1:], color='C0', ds='steps')
    ax.plot(tiempo, eigp[:, 0], color='C1', label='planta', ds='steps')
    ax.plot(tiempo, eigp[:, 1:], color='C1', ds='steps')
    ax.legend()

    if arg.save:
        fig.savefig('/tmp/metricas.pdf', format='pdf')
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
