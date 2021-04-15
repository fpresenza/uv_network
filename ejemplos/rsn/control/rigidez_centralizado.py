#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on jue mar 18 13:21:30 -03 2021
@author: fran
"""
import argparse
import collections
import time
import progressbar
import numpy as np
import matplotlib.pyplot as plt

from gpsic.plotting.core import agregar_ax
from gpsic.plotting.planar import animate_matrix
from gpsic.grafos.plotting import animar_grafo
from uvnpy.modelos.lineal import integrador
import uvnpy.network.disk_graph as disk_graph
import uvnpy.network.connectivity as cnt
import uvnpy.rsn.core as rsn
import uvnpy.rsn.distances as distances
import uvnpy.toolkit.calculus as calc

np.set_printoptions(suppress=True, precision=3, linewidth=150)
# ------------------------------------------------------------------
# Definición de variables globales, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple('Logs', 'x u J eig Y')

D = calc.derivative_eval
lsd = cnt.logistic_strength_derivative


def detF(p):
    A = distances.all_aa(p)
    A[A > 0] = cnt.logistic_strength(A[A > 0], w=beta_2, e=e_2)

    _, Mf = rsn.pose_and_shape_basis_2d_aa(p)
    Mf_T = Mf.swapaxes(-2, -1)

    Y = distances.innovation_matrix_aa(A, p)
    F = np.matmul(Mf_T, np.matmul(Y, Mf))
    return np.linalg.det(F)


def detF_grad(p):
    u = a * detF(p[None])**(a - 1) * D(detF, p)
    return u.reshape(p.shape)


def logdetF_grad(p):
    u = detF(p[None])**(-1) * D(detF, p)
    return u.reshape(p.shape)


def keep_rigid(p):
    u = - a * detF(p[None])**(-a - 1) * D(detF, p)
    return -u.reshape(p.shape)


def min_edges(p):
    w = distances.all(p)
    w[w > 0] = lsd(w[w > 0], w=beta_1, e=e_1)
    u = distances.edge_potencial_gradient(w, p)
    return -u


def repulsion(p):
    w = distances.all(p)
    w[w > 0] = cnt.power_strength_derivative(w[w > 0], a=1)
    u = distances.edge_potencial_gradient(w, p)
    return -u


def grid(nv, sep):
    k = np.ceil(np.sqrt(nv)) / 2
    nums = np.arange(-k, k) * sep
    g = np.meshgrid(nums, nums)
    return np.vstack(np.dstack(g))[:nv]


def linspace(nv, dmax):
    p = np.empty((nv, 2))
    p[:, 0] = np.linspace(-dmax, dmax, nv)
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

        # u = keep_rigid(x) + 1.5 * min_edges(x) + (2 / nv) * repulsion(x)
        # u *= 2
        u = 0.2 * detF_grad(x) + 2 * repulsion(x)

        t_b = time.perf_counter()

        x = planta.step(t, u)

        # Análisis
        A = disk_graph.adjacency(x, dmax)
        Y = distances.innovation_matrix(A, x)

        _, Mf = rsn.pose_and_shape_basis_2d(x)
        F = Mf.T.dot(Y).dot(Mf)
        J = np.abs(np.linalg.det(F))**a
        # J = np.log(np.linalg.det(F))
        eigvals = np.linalg.eigvalsh(F)

        E = disk_graph.edges(x, dmax)
        cuadros[k] = x, E

        logs.x[k] = x
        logs.u[k] = u
        logs.J[k] = (J, len(E))
        logs.eig[k] = eigvals
        logs.Y[k] = Y

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

    nv = arg.n
    a = 2 / nv
    V = range(nv)
    dof = 2
    n = dof * nv
    dmax = 10
    # beta_1 = 10 / dmax
    # beta_2 = 40 / dmax
    # e_1 = dmax
    # e_2 = 0.7 * dmax
    beta_1 = 0.5
    beta_2 = 0.5
    e_1 = dmax
    e_2 = dmax

    np.random.seed(1)
    # x0 = np.random.uniform(-0.5 * dmax, 0.5 * dmax, (nv, dof))
    x0 = np.random.uniform(-1.3 * dmax, 1.3 * dmax, (nv, dof))
    # x0 = np.array([
    #    [  8.14 , -14.377],
    #    [  4.009,   7.464],
    #    [ -0.045,  -8.256],
    #    [ -9.058,   7.816],
    #    [ -9.927, -12.35 ],
    #    [  5.561,  13.602],
    #    [-14.882,   0.366],
    #    [  9.379,   3.376],
    #    [  6.653,  -6.244],
    #    [ 12.533,   6.437],
    #    [  1.276, -10.735],
    #    [ -3.8  ,   5.224],
    #    [ -1.745,  -1.98 ],
    #    [ -7.123,   0.45 ],
    #    [ -1.012,  10.45 ],
    #    [  9.157,   0.649]]) * 1.4

    planta = integrador(x0, tiempo[0])

    A0 = disk_graph.adjacency(x0, dmax)
    if distances.rigidity(A0, x0):
        print('---> Grafo rígido <---')
    else:
        print('---> Grafo flexible <---')

    logs = Logs(
        x=np.empty((tiempo.size, nv, dof)),
        u=np.empty((tiempo.size, nv, dof)),
        J=np.empty((tiempo.size, 2)),
        eig=np.empty((tiempo.size, n - 3)),
        Y=np.empty((tiempo.size, n, n))
        )
    logs.x[0] = x0
    logs.u[0] = np.zeros((nv, dof))
    logs.J[0] = None
    logs.eig[0] = None
    logs.Y[0] = distances.innovation_matrix(A0, x0)

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
    Y = logs.Y

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
    # ax.semilogy(tiempo, J[:, 0], ds='steps')
    ax.plot(tiempo, J[:, 0], ds='steps')
    # ax.set_ylim(bottom=0)
    # ax.legend()

    ax = agregar_ax(
        gs[0, 1],
        xlabel='t [seg]', ylabel=r'$|\mathcal{E}|$', label_kw={'fontsize': 10})
    ax.plot(tiempo, J[:, 1], ds='steps')
    ax.set_ylim(bottom=0)
    # ax.legend()

    ax = agregar_ax(
        gs[1, :],
        xlabel='t [seg]', ylabel=r'$\lambda(F)$',
        label_kw={'fontsize': 10})
    # ax.semilogy(tiempo, eig, ds='steps')
    ax.plot(tiempo, eig, ds='steps')
    # ax.set_ylim(bottom=0)

    if arg.save:
        fig.savefig('/tmp/metricas.pdf', format='pdf')
    else:
        plt.show()

    if arg.animate:
        # animar matriz
        fig, ax = plt.subplots()
        animate_matrix(fig, ax, arg.h, Y, save=arg.save)

        # animar grafo
        estilos = (
            [V, {'color': 'b', 'marker': 'o', 'markersize': '5'}], )
        fig, ax = plt.subplots()
        title = r'$n={}, \; d_{{max}}={},$'
        title += '\n'
        title += r'$\beta_1={}, \; \epsilon_1={},$'
        title += '\t'
        title += r'$\beta_2={}, \; \epsilon_2={}$'
        fig.suptitle(title.format(nv, dmax, beta_1, e_1, beta_2, e_2))
        # title += r'$\beta={}, \; \epsilon={},$'
        # fig.suptitle(title.format(nv, dmax, beta_1, e_1))
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
