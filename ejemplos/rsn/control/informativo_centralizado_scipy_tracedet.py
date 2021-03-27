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

from gpsic.plotting.core import agregar_ax
from gpsic.grafos.plotting import animar_grafo
from uvnpy.modelos.lineal import integrador
import uvnpy.redes.core as redes
import uvnpy.redes.comunicaciones as com
import uvnpy.rsn.core as rsn
from uvnpy.control import informativo
from uvnpy.control import costos

# ------------------------------------------------------------------
# Definición de variables globales, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple('Logs', 'x u J Jp eig eigp x_p')


def logistic(dist, dmax, w):
    p = dist > 0
    dist[p] = com.logistic_strength(dist[p], w, e=dmax)
    return dist


def on_off(dist, dmax, *args):
    A = dist.copy()
    A[A > dmax] = 0
    A[A != 0] = 1
    return A


atenuacion = logistic          # familia sigmoide


def innovacion(x, A):
    N = len(x)
    p = np.reshape(x, (N, -1, 2))
    dist = rsn.distances(p)
    Aw = atenuacion(dist, dmax, 1)
    # A = atenuacion(dist, 0.5 * dmax, 1)
    # A = np.empty(Aw.shape)
    # A[:] = Aw[0]
    A = np.tile(A, (N, 1, 1))
    # Yw = sum(rsn.distances_innovation_aa(Aw, p))
    Y = sum(rsn.distances_innovation_aa(A, p))
    _, S = rsn.pose_and_shape_basis_2d(p[None, 0])
    S = S[0]
    Ys = S.T.dot(Y.dot(S))
    return Aw, Ys


def funcional(M):
    T = M[0].sum()
    D = np.linalg.det(M[1])
    J1, J2 = T, (D + 1e-6)**(2/nv)
    f = - J1 - 15 * J2
    # print(J1, J2)
    return f


def ca_repulsion(u, x_p, Q):
    N = len(x_p)
    p = np.reshape(x_p, (N, -1, 2))
    return Q * costos.repulsion(p)


def analisis(x, dmax, Vp, atenuacion):
    dist = rsn.distances(x)
    Aw = atenuacion(dist, dmax, 1)
    A = np.ones(Aw.shape) - np.eye(*Aw.shape[-2:])
    Yw = rsn.distances_innovation(Aw, x)
    Y = rsn.distances_innovation(A, x)
    _, S = rsn.pose_and_shape_basis_2d(x)
    Yws = S.T.dot(Yw.dot(S))
    Ys = S.T.dot(Y.dot(S))

    J = funcional((Yws, Ys))
    eigvals = np.linalg.eigvalsh(Yw)
    return J, eigvals


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
        a = time.perf_counter()
        A = redes.adjacency_from_positions(x, dmax)
        u = ctrl.update(x.ravel(), t, ([A], [])).reshape(nv, dof)
        b = time.perf_counter()
        x = planta.step(t, u)

        # nodos de posicion
        # Vp = optimal_position_nodes(x[None, ...])
        Vp = range(nvp)

        # análisis
        J, eigvals = analisis(x, dmax, [], logistic)
        Jp, eigvalsp = analisis(x, dmax, Vp, on_off)

        # E = redes.complete_undirected_edges(V)
        E = redes.undirected_edges(redes.edges_from_positions(x, dmax))
        X = x[list(V) + list(Vp)]
        cuadros[k] = X, E

        logs.x[k] = x
        logs.u[k] = u
        logs.J[k] = J
        logs.Jp[k] = Jp
        logs.eig[k] = eigvals
        logs.eigp[k] = eigvalsp
        logs.x_p[k] = x[Vp]

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
    nvp = 0
    V = range(nv)
    dof = 2
    n = dof * nv
    dmax = 12
    np.random.seed(5)
    x0 = np.random.uniform(-lim/1.7, lim/1.7, (nv, dof))
    A0 = redes.adjacency_from_positions(x0, dmax)
    H0 = rsn.distances_jac(A0, x0)
    if np.all(A0 == redes.complete_adjacency(V)):
        print('---> Grafo completo <---')
    elif np.linalg.matrix_rank(H0) == nv * dof - 3:
        print('---> Grafo rígido <---')
    else:
        print('---> Grafo flexible <---')
    # x0 = np.array([[-4, 0],
    #                [0, -5],
    #                [3, 0],
    #                [0, 5]], dtype=np.float)
    # x0 = grid(nv, 5)
    # x0 = linspace(nv, lim*0.75) + np.random.normal(0, 0.5, (nv, dof))

    planta = integrador(x0, tiempo[0])

    ctrl = informativo.minimizar(
        metrica=funcional,
        matriz=innovacion,
        modelo=integrador,
        Q=(1 * np.eye(n), 3 * np.eye(n), 0.5),
        dim=n,
        horizonte=np.array([0.1, 0.2]),   # np.linspace(0.1, 0.5, 5),
        )
    ctrl.agregar_costo(fun=ca_repulsion, Q=2)

    logs = Logs(
        x=np.empty((tiempo.size, nv, dof)),
        u=np.empty((tiempo.size, nv, dof)),
        J=np.empty((tiempo.size)),
        Jp=np.empty((tiempo.size)),
        eig=np.empty((tiempo.size, n)),
        eigp=np.empty((tiempo.size, n)),
        x_p=np.empty((tiempo.size, nvp, dof))
        )
    logs.x[0] = x0
    logs.u[0] = np.zeros((nv, dof))
    logs.J[0] = None
    logs.Jp[0] = None
    logs.eig[0] = None
    logs.eigp[0] = None
    logs.x_p[0] = None

    cuadros = np.empty((tiempo.size, 2), dtype=np.ndarray)
    # E0 = redes.complete_undirected_edges(V)
    E0 = redes.undirected_edges(redes.edges_from_positions(x0, dmax))
    X0 = x0[list(V) + [0, 1]]
    cuadros[0] = X0, E0

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
    x_p = logs.x_p

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
        xlabel='t [seg]', ylabel='métrica', label_kw={'fontsize': 10})
    ax.plot(tiempo, J, color='C0', label='control', ds='steps')
    ax.plot(tiempo, Jp, color='C1', label='planta', ds='steps')
    # ax.plot(tiempo, eig[:, 3:].prod(1)**(0.2),
    # color='C2', label='det', ds='steps')
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
        if nvp > 0:
            estilos += ([
                [nv, nv + 1], {'color': 'g', 'marker': '*', 'markersize': '8'}
            ],)
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