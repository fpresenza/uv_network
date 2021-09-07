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
from uvnpy.toolkit.calculus import derivative_eval, gradient  # noqa

# ------------------------------------------------------------------
# Definición de variables globales, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple('Logs', 'x u eig maxdist')


def rigidity_eigenvalue(A, x):
    if len(A) == 1:
        return 0.
    L = rigidity.laplacian(A, x)
    return np.linalg.eigvalsh(L)[3]


def weighted_rigidity_eigenvalue(x):
    w = distances.matrix(x)
    w[w > 0] = strength.logistic(w[w > 0], beta[1], alpha[1])
    L = rigidity.laplacian(w, x)
    return np.linalg.eigvalsh(L)[..., 3]


def eigenvalue_product(x, M):
    w = distances.matrix(x)
    w[w > 0] = strength.logistic(w[w > 0], beta[1], alpha[1])

    L = rigidity.laplacian(w, x)
    S = np.matmul(M.T, np.matmul(L, M))
    return np.linalg.det(S)


def rigidity_maintenance(x):
    # L = rigidity_matrix(x)
    # e, V = np.linalg.eigh(L)
    # dL_dx = derivative_eval(rigidity_matrix, x)
    # v4 = V[:, 3]          # rigidity eigenvector
    # lambda4 = e[3]
    # dlambda4_dx = v4.dot(dL_dx).dot(v4).reshape(x.shape)
    # ur = -a * lambda4**(-a - 1) * dlambda4_dx

    # M = rigidity.nontrivial_motions(x)
    L = rigidity_matrix(x)
    e, V = np.linalg.eigh(L)
    M = V[:, 3:]
    detS = e[3:].prod()

    ur = -a * detS**(-a - 1) * gradient(eigenvalue_product, x, M)

    return -5*ur


def rigidity_matrix(x):
    w = distances.matrix(x)
    w[w > 0] = strength.logistic(w[w > 0], beta[1], alpha[1])
    L = rigidity.laplacian(w, x)
    return L


def disconnect(x):
    w = distances.matrix(x)
    w[w > 0] = strength.logistic_derivative(w[w > 0], beta[0], alpha[0])
    u = distances.edge_potencial_gradient(w, x)
    return -u


def expand(x):
    w = distances.matrix(x)
    w[w > 0] = strength.power_derivative(w[w > 0], 5/n)
    u = distances.edge_potencial_gradient(w, x)
    return -15*u


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(steps, logs, t_perf, A, dinamica, frames):
    # iteración
    bar = progressbar.ProgressBar(max_value=arg.tf).start()
    u = np.zeros(dinamica.x.shape)
    eig = np.empty(n)
    Ah = np.empty((2, n, n), dtype=bool)

    for k, t in steps[1:]:
        # step dinamica
        x = dinamica.x

        # Control
        t_a = np.empty(n)
        t_b = np.empty(n)
        u_t = expand(x) + disconnect(x)
        u[:] = u_t
        R = subsets.reach(A, (0, 1, 2))
        Ah[0] = sum(R[:2]).astype(bool)
        Ah[1] = sum(R[:3]).astype(bool)
        for i in nodes:
            h = hops[i]
            # print('\n', h, i)
            Ni = Ah[h-1, i]
            # print(Ni)
            Ai = A[Ni][:, Ni]
            xi = x[Ni]
            eig[i] = rigidity_eigenvalue(Ai, xi)
            if eig[i] < 1e-6:
                print(t, i)
                raise ValueError('Zero eigenvalue')

            t_a[i] = time.perf_counter()
            u[Ni] += rigidity_maintenance(xi)
            t_b[i] = time.perf_counter()
            # print(i, t_a[i] - t_b[i])

        # print(np.where(eig < 1e-5))
        u *= 2
        x = dinamica.step(t, u)

        # Análisis
        # print(distances.matrix(x))
        # A = disk_graph.adjacency(x, dmin)
        A = disk_graph.adjacency_histeresis(A, x, dmin, dmax)
        E = network.edges_from_adjacency(A)
        frames[k] = t, x, E

        logs.x[k] = x
        logs.u[k, :, 0] = np.linalg.norm(u_t, axis=1)
        logs.u[k, :, 1] = np.linalg.norm(u - u_t, axis=1)
        logs.eig[k, 0] = rigidity_eigenvalue(A, x)
        logs.eig[k, 1:] = eig
        # print(distances.from_adjacency(A, x))
        logs.maxdist[k] = distances.from_adjacency(A, x).max()

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
        dest='h', default=100e-3, type=float, help='paso de simulación')
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

    n = 80
    # hops = 3
    L = 100
    a = 5 / n
    dof = 2
    dmin = 0.4*L
    dmax = 1.4 * dmin
    beta = (5 / dmin, 40 / dmin)
    alpha = (dmax, dmin)

    nodes = np.arange(n)
    x = np.random.uniform(-0.9*L, 0.9*L, (n, dof))
    # x = np.random.uniform(-0.3*L, 0.3*L, (n, dof))
    # print(x, dmin)
    A = disk_graph.adjacency(x, dmin)
    # print(subsets.multihop_adjacency(A, hops).sum(1))
    dinamica = linear_models.integrator(x, tiempo[0])

    logs = Logs(
        x=np.empty((tiempo.size, n, dof)),
        u=np.empty((tiempo.size, n, dof)),
        eig=np.empty((tiempo.size, 1 + n)),
        maxdist=np.empty(tiempo.size)
        )
    logs.x[0] = x
    logs.u[0] = np.zeros((n, dof))
    logs.eig[0, 0] = rigidity_eigenvalue(A, x)

    hops = np.empty(n, dtype=int)
    for i in nodes:
        subset_found = False
        h = 0
        while not subset_found:
            h += 1
            Ai, xi = subsets.multihop_subframework(A, x, i, h)
            re = rigidity_eigenvalue(Ai, xi)
            wre = weighted_rigidity_eigenvalue(xi)
            if re > 1e-3:
                subset_found = True
                logs.eig[0, i+1] = re
                print(
                    'Node {}, hops = {}, RE = {} ~ {}'.format(i, h, re, wre))
        hops[i] = h

    logs.maxdist[0] = distances.from_adjacency(A, x).max()

    frames = np.empty((tiempo.size, 3), dtype=np.ndarray)
    E = disk_graph.edges(x, dmin)
    frames[0] = tiempo[0], x, E

    # xi = [xi for _, xi in subframeworks]
    # l4 = np.array([
    #     weighted_rigidity_eigenvalue(xi) for _, xi in subframeworks])
    # print(l4.min(), l4.mean(), l4.max())
    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    logs, t_perf, frames = run(steps, logs, t_perf, A, dinamica, frames)

    x = logs.x
    u = logs.u
    eig = logs.eig
    maxdist = logs.maxdist

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
    axes[0].semilogy(
        tiempo, eig[:, 0].clip(1e-6), color='purple', ls='--', zorder=10)
    axes[0].semilogy(tiempo, eig[:, 1:].clip(1e-6), color='0.7')
    # axes[1].set_ylim(bottom=0)
    fig.savefig('/tmp/metricas.pdf', format='pdf')

    axes[1].set_xlabel('t [seg]')
    axes[1].set_ylabel(r'$\rm{min}(d_{ij})$')
    axes[1].grid(1)
    axes[1].plot(tiempo, maxdist)
    axes[1].set_ylim(bottom=0)
    fig.savefig('/tmp/metricas.pdf', format='pdf')

    if arg.animate:
        fig, ax = network.plot.figure()
        ax.set_xlim(-2*L, 2*L)
        ax.set_ylim(-2*L, 2*L)
        anim = network.plot.Animate(fig, ax, arg.h/2, frames, maxlen=50)
        anim.set_teams(
            {'ids': np.delete(nodes, (9, )), 'tail': True,
                'style': {'color': 'b', 'marker': 'o', 'markersize': 5}},
            {'ids': np.take(nodes, (9, )), 'tail': True,
                'style': {'color': 'r', 'marker': 'o', 'markersize': 5}})
        anim.set_edgestyle(color='0.2', alpha=0.6, lw=0.8)
        anim.ax.legend()
        anim.run('/tmp/multihop.mp4')

    # plt.show()
