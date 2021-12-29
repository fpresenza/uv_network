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
import scipy.linalg
import matplotlib.pyplot as plt

from uvnpy.model import linear_models
import uvnpy.network as network
from uvnpy.network import disk_graph
from uvnpy.network import connectivity
from uvnpy.rsn import distances
from uvnpy.toolkit import calculus

np.set_printoptions(suppress=True, precision=3, linewidth=150)
# ------------------------------------------------------------------
# Definición de variables globales, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple('Logs', 'x u J eig L')

D = calculus.derivative_eval
lsd = connectivity.logistic_strength_derivative


def detF(p):
    A = distances.matrix(p)
    A[A > 0] = connectivity.logistic_strength(A[A > 0], beta=beta_2, e=e_2)

    L = network.laplacian_from_adjacency(A)
    F = np.matmul(M.T, np.matmul(L, M))
    return np.linalg.det(F)


def detF_grad(p):
    u = a * detF(p[None])**(a - 1) * D(detF, p)
    return u.reshape(p.shape)


def logdetF_grad(p):
    u = detF(p[None])**(-1) * D(detF, p)
    return u.reshape(p.shape)


def keep_connected(p):
    u = - a * detF(p[None])**(-a - 1) * D(detF, p)
    return -u.reshape(p.shape)


def min_edges(p):
    dw = distances.matrix(p)
    dw[dw > 0] = lsd(dw[dw > 0], beta=beta_1, e=e_1)
    u = distances.edge_potencial_gradient(dw, p)
    return -u


def repulsion(p):
    w = distances.matrix(p)
    w[w > 0] = connectivity.power_strength_derivative(w[w > 0], a=1)
    u = distances.edge_potencial_gradient(w, p)
    return -u


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


def run(steps, logs, t_perf, planta, frames):
    # iteración
    bar = progressbar.ProgressBar(max_value=arg.tf).start()
    for k, t in steps[1:]:
        # step planta
        x = planta.x

        # Control
        t_a = time.perf_counter()

        u = keep_connected(x) + 3.5 * min_edges(x)  # + (2 / n) * repulsion(x)
        # u *= 2
        # u = 0.5 * detF_grad(x) + 0.2 * repulsion(x)

        t_b = time.perf_counter()

        x = planta.step(t, u)

        # Análisis
        A = disk_graph.adjacency(x, dmax)
        L = network.laplacian_from_adjacency(A)

        F = M.T.dot(L).dot(M)
        J = np.abs(np.linalg.det(F))**a
        # J = np.log(np.linalg.det(F))
        eigvals = np.linalg.eigvalsh(F)

        E = disk_graph.edges(x, dmax)
        frames[k] = t, x, E

        logs.x[k] = x
        logs.u[k] = u
        logs.J[k] = (J, len(E))
        logs.eig[k] = eigvals
        logs.L[k] = L

        t_perf.append(t_b - t_a)
        bar.update(np.round(t, 3))

    bar.finish()

    # return
    return logs, t_perf, frames


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
    a = 2 / n
    V = range(n)
    dof = 2
    dmax = 10
    beta_1 = 2 / dmax
    beta_2 = 40 / dmax
    e_1 = dmax
    e_2 = 0.7 * dmax
    # beta_1 = 0.5
    # beta_2 = 0.5
    # e_1 = dmax
    # e_2 = dmax

    M = scipy.linalg.null_space(np.ones((1, n)))

    # np.random.seed(1)
    # x0 = np.random.uniform(-0.5 * dmax, 0.5 * dmax, (n, dof))
    # x0 = np.random.uniform(-dmax, dmax, (n, dof))
    x0 = calculus.circle2d(R=0.1*dmax, N=n)
    x0 += np.random.normal(0, 0.5, (n, dof))
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

    planta = linear_models.integrator(x0, tiempo[0])

    A0 = disk_graph.adjacency(x0, dmax)
    L0 = network.laplacian_from_adjacency(A0)
    if connectivity.algebraic_connectivity(L0):
        print('---> Grafo conectado <---')
    else:
        print('---> Grafo no conectado <---')

    logs = Logs(
        x=np.empty((tiempo.size, n, dof)),
        u=np.empty((tiempo.size, n, dof)),
        J=np.empty((tiempo.size, 2)),
        eig=np.empty((tiempo.size, n - 1)),
        L=np.empty((tiempo.size, n, n))
        )
    logs.x[0] = x0
    logs.u[0] = np.zeros((n, dof))
    logs.J[0] = None
    logs.eig[0] = None
    logs.L[0] = network.laplacian_from_adjacency(A0)

    frames = np.empty((tiempo.size, 3), dtype=np.ndarray)
    E0 = disk_graph.edges(x0, dmax)
    frames[0] = tiempo[0], x0, E0

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    logs, t_perf, frames = run(steps, logs, t_perf, planta, frames)

    x = logs.x
    u = logs.u
    J = logs.J
    eig = logs.eig
    L = logs.L

    st = arg.tf - arg.ti
    rt = sum(t_perf)
    prompt = 'RT={:.3f} secs, ST={:.3f} secs  ==>  RTF={:.3f}'
    print(prompt.format(rt, st, st / rt))

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(2, 1)

    axes[0].set_title('posicion')
    axes[0].set_xlabel('t [seg]')
    axes[0].set_ylabel('(x, y) [m]')
    axes[0].grid(1)
    axes[0].plot(tiempo, x[..., 0], ds='steps')
    plt.gca().set_prop_cycle(None)
    axes[0].plot(tiempo, x[..., 1], ds='steps')

    axes[1].set_title('accion de control')
    axes[1].set_xlabel('t [seg]')
    axes[1].set_ylabel('u [m/s]')
    axes[1].grid(1)
    axes[1].plot(tiempo, u[..., 0], ds='steps')
    plt.gca().set_prop_cycle(None)
    axes[1].plot(tiempo, u[..., 1], ds='steps')
    fig.savefig('/tmp/control.pdf', format='pdf')

    fig, axes = plt.subplots(3, 1)

    axes[0].set_xlabel('t [seg]')
    axes[0].set_ylabel(r'$det(F)^a$')
    axes[0].grid(1)
    # axes[0].semilogy(tiempo, J[:, 0], ds='steps')
    axes[0].plot(tiempo, J[:, 0], ds='steps')
    # axes[0].set_ylim(bottom=0)
    # axes.[0]legend()

    axes[1].set_xlabel('t [seg]')
    axes[1].set_ylabel(r'$|\mathcal{E}|$')
    axes[1].grid(1)
    axes[1].plot(tiempo, J[:, 1], ds='steps')
    axes[1].set_ylim(bottom=0)
    # axes[1].legend()

    axes[2].set_xlabel('t [seg]')
    axes[2].set_ylabel(r'$\lambda(F)$')
    axes[2].grid(1)
    # axes[2].semilogy(tiempo, eig, ds='steps')
    axes[2].plot(tiempo, eig, ds='steps')
    # axes[2].set_ylim(bottom=0)
    fig.savefig('/tmp/metricas.pdf', format='pdf')

    if arg.animate:
        fig, ax = network.plot.figure()
        ax.set_xlim(-1.5*dmax, 1.5*dmax)
        ax.set_ylim(-1.5*dmax, 1.5*dmax)
        anim = network.plot.Animate(fig, ax, arg.h/2, frames, maxlen=50)
        anim.set_teams({'ids': V, 'tail': True})
        anim.ax.legend()
        anim.run()

    plt.show()
