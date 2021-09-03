#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on mié sep  1 20:02:37 -03 2021
@author: fran
"""
import argparse
import collections
import time
import progressbar
import numpy as np
import matplotlib.pyplot as plt

from uvnpy.model import linear_models
import uvnpy.network as network
from uvnpy.network import disk_graph, strength, subsets
from uvnpy.rsn import distances, rigidity
from uvnpy.toolkit.calculus import gradient

# ------------------------------------------------------------------
# Definición de variables globales, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple('Logs', 'x u eig mindist')


def rigidity_eigenvalue(A, x):
    if len(A) == 1:
        return 0.
    L = rigidity.laplacian(A, x)
    return np.linalg.eigvalsh(L)[3]


def objective(x, i):
    d = distances.matrix(x)

    wt = d.copy()
    wt[wt > 0] = strength.logistic_derivative(wt[wt > 0], beta[0], alpha[0])
    ut = distances.edge_potencial_gradient(wt, x)
    # ut = 0

    wc = d.copy()
    wc[wc > 0] = strength.power_derivative(wc[wc > 0], 1)
    uc = distances.edge_potencial_gradient(wc, x)
    # uc = 0
    ur = -a * lambda4(x)**(-a - 1) * gradient(lambda4, x)
    u = 0.1*ut + a/2*uc + 3*ur
    # print(i, lambda4(x))
    # if np.isclose(lambda4(x), 0):
    #     print(i)
    return -u/2


def lambda4(x):
    w = distances.matrix(x)
    w[w > 0] = strength.logistic(w[w > 0], beta[1], alpha[1])
    L = rigidity.laplacian(w, x)
    return np.linalg.eigvalsh(L)[..., 3]

# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(steps, logs, t_perf, A, dinamica, frames):
    # iteración
    bar = progressbar.ProgressBar(max_value=arg.tf).start()
    u = np.zeros(dinamica.x.shape)
    eig = np.empty(n)

    for k, t in steps[1:]:
        # step dinamica
        x = dinamica.x

        # Control
        t_a = np.empty(n)
        t_b = np.empty(n)
        u[:] = 0
        Ah = subsets.multihop_adjacency(A, hops)
        for i in nodes:
            t_a[i] = time.perf_counter()

            Ni = Ah[i]
            Ai = A[Ni][:, Ni]
            xi = x[Ni]
            eig[i] = rigidity_eigenvalue(Ai, xi)
            u[Ni] += objective(xi, i)

            t_b[i] = time.perf_counter()

        # print(np.where(eig < 1e-5))
        x = dinamica.step(t, u)

        # Análisis
        # print(distances.matrix(x))
        A = disk_graph.adjacency(x, dmax)
        E = network.edges_from_adjacency(A)
        frames[k] = t, x, E

        logs.x[k] = x
        logs.u[k] = u
        logs.eig[k, 0] = rigidity_eigenvalue(A, x)
        logs.eig[k, 1:] = eig
        # print(distances.from_adjacency(A, x))
        logs.mindist[k] = distances.from_adjacency(A, x).min()

        t_perf.append(np.mean(t_b - t_a))
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

    arg = parser.parse_args()

    # ------------------------------------------------------------------
    # Configuración
    # ------------------------------------------------------------------
    np.random.seed(0)

    tiempo = np.arange(arg.ti, arg.tf, arg.h)
    steps = list(enumerate(tiempo))
    t_perf = []

    n = 20  # 80
    hops = 2
    L = 25  # 100
    a = 1 / n
    dof = 2
    dmax = 0.4*L
    beta = (10 / dmax, 40 / dmax)
    alpha = (dmax, 0.7 * dmax)

    nodes = np.arange(n)
    # x = np.random.uniform(-0.9*L, 0.9*L, (n, dof))
    x = np.random.uniform(-0.3*L, 0.3*L, (n, dof))
    # print(x, dmax)
    A = disk_graph.adjacency(x, dmax)
    print(subsets.multihop_adjacency(A, hops).sum(1))
    dinamica = linear_models.integrator(x, tiempo[0])

    logs = Logs(
        x=np.empty((tiempo.size, n, dof)),
        u=np.empty((tiempo.size, n, dof)),
        eig=np.empty((tiempo.size, 1 + n)),
        mindist=np.empty(tiempo.size)
        )
    logs.x[0] = x
    logs.u[0] = np.zeros((n, dof))
    logs.eig[0, 0] = rigidity_eigenvalue(A, x)
    subframeworks = [
        subsets.multihop_subframework(A, x, i, hops) for i in range(n)]
    logs.eig[0, 1:] = [rigidity_eigenvalue(Ai, xi) for Ai, xi in subframeworks]
    print(logs.eig[0, 1:].min(), logs.eig[0, 1:].mean(), logs.eig[0, 1:].max())

    logs.mindist[0] = distances.from_adjacency(A, x).min()

    frames = np.empty((tiempo.size, 3), dtype=np.ndarray)
    E = disk_graph.edges(x, dmax)
    frames[0] = tiempo[0], x, E

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    logs, t_perf, frames = run(steps, logs, t_perf, A, dinamica, frames)

    x = logs.x
    u = logs.u
    eig = logs.eig
    mindist = logs.mindist

    st = arg.tf - arg.ti
    rt = sum(t_perf)
    prompt = 'RT={:.3f} secs, ST={:.3f} secs  ==>  RTF={:.3f}'
    print(prompt.format(rt, st, st / rt))

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(2, 2, figsize=(10, 4))

    axes[0, 0].set_xlabel('t [seg]')
    axes[0, 0].set_ylabel('x [m]')
    axes[0, 0].grid(1)
    axes[0, 0].plot(tiempo, x[..., 0])
    plt.gca().set_prop_cycle(None)
    axes[0, 1].set_xlabel('t [seg]')
    axes[0, 1].set_ylabel('y [m]')
    axes[0, 1].grid(1)
    axes[0, 1].plot(tiempo, x[..., 1])

    axes[1, 0].set_xlabel('t [seg]')
    axes[1, 0].set_ylabel('u_x [m/s]')
    axes[1, 0].grid(1)
    axes[1, 0].plot(tiempo, u[..., 0])
    plt.gca().set_prop_cycle(None)
    axes[1, 1].set_xlabel('t [seg]')
    axes[1, 1].set_ylabel('u_y [m/s]')
    axes[1, 1].grid(1)
    axes[1, 1].plot(tiempo, u[..., 1])
    fig.savefig('/tmp/control.pdf', format='pdf')

    fig, axes = plt.subplots(2, 1)

    axes[0].set_xlabel('t [seg]')
    axes[0].set_ylabel(r'$\rho$')
    axes[0].grid(1)
    axes[0].semilogy(tiempo, eig[:, 0], color='purple', ls='--', zorder=10)
    axes[0].semilogy(tiempo, eig[:, 1:], color='0.7')
    # axes[1].set_ylim(bottom=0)
    fig.savefig('/tmp/metricas.pdf', format='pdf')

    axes[1].set_xlabel('t [seg]')
    axes[1].set_ylabel(r'$\rm{min}(d_{ij})$')
    axes[1].grid(1)
    axes[1].plot(tiempo, mindist)
    # axes[2].hlines(
    #     [dmin / dmax, 1],
    #     xmin=0, xmax=tiempo[-1], ls='--', color='k', alpha=0.7)
    axes[1].set_ylim(bottom=0)
    fig.savefig('/tmp/metricas.pdf', format='pdf')

    if arg.animate:
        fig, ax = network.plot.figure()
        ax.set_xlim(-1.5*L, 1.5*L)
        ax.set_ylim(-1.5*L, 1.5*L)
        anim = network.plot.Animate(fig, ax, arg.h/2, frames, maxlen=50)
        anim.set_teams(
            {'ids': np.delete(nodes, (8, 10)), 'tail': True,
                'style': {'color': 'b', 'marker': 'o', 'markersize': 5}},
            {'ids': np.take(nodes, (8, 10)), 'tail': True,
                'style': {'color': 'r', 'marker': 'o', 'markersize': 5}})
        anim.set_edgestyle(color='0.2', alpha=0.6, lw=0.8)
        anim.ax.legend()
        anim.run()

    plt.show()
