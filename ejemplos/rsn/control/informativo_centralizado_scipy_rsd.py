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
import scipy.stats
import matplotlib.pyplot as plt

from gpsic.plotting.core import agregar_ax
from gpsic.grafos.plotting import animar_grafo
from uvnpy.modelos.lineal import integrador
import uvnpy.network.graph as gph
import uvnpy.network.connectivity as cnt
import uvnpy.rsn.distances as distances
from uvnpy.control import informativo
from uvnpy.control import costos
from uvnpy.filtering import metricas
from uvnpy.toolkit import core

# ------------------------------------------------------------------
# Definición de variables globales, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple('Logs', 'x u J Jp eig eigp x_p')

rsd = metricas.relative_standard_deviation  # rsd


def di(eigvals):
    return metricas.dispersion_index(eigvals, 0.4)


def inverse_nuclear(eigvals):
    w = 1 / (eigvals + 1)
    nuclear = metricas.weighted_sum(eigvals, w)
    return 1 / nuclear


def maxmin(eigvals):
    return 1 / (min(eigvals) + 1)


mu = rsd
# mu = di
# mu = inverse_nuclear
# mu = maxmin
# mu = np.prod


def q(dist, dmax, *args, **kwargs):
    _q = 1
    A = dist.copy()
    A[A != 0] **= -2 * _q
    return A


def normal_cdf(dist, dmax, scale):
    A = dist.copy()
    A[A > 0] = 1 - scipy.stats.norm.cdf(A[A > 0], loc=dmax, scale=scale)
    return A


def logistic(dist, dmax, w):
    p = dist > 0
    dist[p] = cnt.logistic_strength(dist[p], w, e=dmax)
    return dist


def transparent(dist, dmax, *args):
    return dist


def on_off(dist, dmax, *args):
    A = dist.copy()
    A[A > dmax] = 0
    A[A != 0] = 1
    return A


# atenuacion = q               # potencial
# atenuacion = normal_cdf      # distribucion normal (cdf)
atenuacion = logistic          # familia sigmoide
# atenuacion = transparent     # sin atenuacion


def innovacion(x):
    N = len(x)
    p = np.reshape(x, (N, -1, 2))
    dist = distances.all(p)
    A = atenuacion(dist, dmax, 1)
    # A = atenuacion(dist, 0.5 * dmax, 1)
    # A[:, Vp, Vp] = 1
    Y = distances.innovation_matrix_aa(A, p)
    return sum(Y)


def funcional(Y):
    eigvals = np.linalg.eigvalsh(Y)
    J = mu(eigvals)
    return J.sum()


def ca_repulsion(u, x_p, Q):
    N = len(x_p)
    p = np.reshape(x_p, (N, -1, 2))
    return Q * costos.repulsion(p)


def optimal_position_nodes(x):
    pairs = core.combinations(V, 2)
    m = len(pairs)
    idx = np.arange(m).reshape(-1, 1)

    A = distances.all(x)
    A = on_off(A, dmax)
    x = np.repeat(x, m, axis=0)
    A = np.repeat(A, m, axis=0)
    A[idx, pairs, pairs] = 1

    Y = distances.innovation_matrix_aa(A, x)
    eigvals = np.linalg.eigvalsh(Y)
    opt = eigvals[:, 0].round(2).argmax()   # máx min autovalor
    # opt = np.argmin(rsd(eigvals))         # es invariante :O
    return pairs[opt]


def analisis(x, dmax, Vp, atenuacion):
    dist = distances.all(x)
    A = atenuacion(dist, dmax, 1)
    A[:, Vp, Vp] = 1
    Y = distances.innovation_matrix(A, x)
    eigvals = np.linalg.eigvalsh(Y)
    J = mu(eigvals).sum()
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
        u = ctrl.update(x.ravel(), t, ([], [])).reshape(nv, dof)
        b = time.perf_counter()
        x = planta.step(t, u)

        # nodos de posicion
        # Vp = optimal_position_nodes(x[None, ...])
        Vp = range(nvp)

        # análisis
        J, eigvals = analisis(x, dmax, [], logistic)
        Jp, eigvalsp = analisis(x, dmax, Vp, on_off)
        # print(eigvalsp)

        E = gph.undirected_edges(gph.edges_from_positions(x, dmax))
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
    np.random.seed(2)
    x0 = np.random.uniform(-lim, lim, (nv, dof))
    # x0 = np.array([[-5, 0],
    #                [0, -5],
    #                [5, 0],
    #                [0, 5]], dtype=np.float) * 1.5
    # x0 = grid(nv, 5)
    # x0 = linspace(nv, lim*0.75) + np.random.normal(0, 0.5, (nv, dof))

    planta = integrador(x0, tiempo[0])

    ctrl = informativo.minimizar(
        metrica=funcional,
        matriz=innovacion,
        modelo=integrador,
        Q=(0.2 * np.eye(n), 1 * np.eye(n), 800),
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
    E0 = gph.undirected_edges(gph.edges_from_positions(x0, dmax))
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
