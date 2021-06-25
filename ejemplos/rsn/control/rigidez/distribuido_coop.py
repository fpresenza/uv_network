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

from gpsic.toolkit import linalg
from uvnpy.model import linear_models
import uvnpy.network as network
from uvnpy.network import disk_graph
from uvnpy.network import connectivity
import uvnpy.rsn.core as rsn
from uvnpy.rsn import distances
from uvnpy.toolkit import calculus

# ------------------------------------------------------------------
# Definición de variables globales, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple('Logs', 'x u J eig avg_dist')

D = calculus.derivative_eval
lsd = connectivity.logistic_strength_derivative


def detFi(p):
    Ai = distances.matrix(p)
    Ai[Ai > 0] = connectivity.logistic_strength(Ai[Ai > 0], beta=beta_2, e=e_2)

    M = rsn.shape_basis(p.mean(0))
    Li = distances.laplacian(Ai, p)
    Fi = np.matmul(M.T, np.matmul(Li, M))
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


def run(steps, logs, t_perf, planta, frames):
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

            R[i] = disk_graph.neighborhood_band(x, i, R[i], dmin, dmax, inclusive=True)  # noqa
            p = x[R[i]]

            u[R[i]] += keep_rigid(p) + 0.5 * min_edges(p) + (2 / n) * repulsion(p)  # noqa

            t_b[i] = time.perf_counter()

        x = planta.step(t, u)

        # Análisis
        A = R.astype(int) - np.eye(n)
        L = distances.laplacian(A, x)

        M = rsn.shape_basis(x)
        F = M.T.dot(L).dot(M)
        J = np.abs(np.linalg.det(F))**a
        eigvals = np.linalg.eigvalsh(F)

        E = network.edges_from_adjacency(A)
        frames[k] = t, x, E

        logs.x[k] = x
        logs.u[k] = u
        logs.J[k] = (J, len(E))
        logs.eig[k] = eigvals
        logs.avg_dist[k] = linalg.norma2(x[E[:, 0]] - x[E[:, 1]], axis=1).mean()  # noqa

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
    dn = dof * n
    dmax = 10
    dmin = 0.7 * dmax
    beta_1 = 10 / dmax
    beta_2 = 40 / dmax
    e_1 = dmax
    e_2 = 0.7 * dmax

    # np.random.seed(1)
    x0 = np.random.uniform(-0.5 * dmax, 0.5 * dmax, (n, dof))
    A0 = disk_graph.adjacency(x0, dmax)
    u = np.zeros((n, dof))
    R = (A0 + np.eye(n)).astype(bool)

    planta = linear_models.integrator(x0, tiempo[0])

    for i in V:
        p = x0[R[i]]

        Ai = disk_graph.adjacency(p, dmax)
        Li = distances.laplacian(Ai, p)
        if distances.rigidity(Li, dof):
            print('Grafo {} rígido.'.format(i))
        else:
            print('Warning!: Grafo {} flexible.'.format(i))

    logs = Logs(
        x=np.empty((tiempo.size, n, dof)),
        u=np.empty((tiempo.size, n, dof)),
        J=np.empty((tiempo.size, 2)),
        eig=np.empty((tiempo.size, dn - 3)),
        avg_dist=np.empty(tiempo.size)
        )
    logs.x[0] = x0
    logs.u[0] = np.zeros((n, dof))
    logs.J[0] = None
    logs.eig[0] = None
    logs.avg_dist[0] = None

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
    avg_dist = logs.avg_dist

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

    fig, axes = plt.subplots(4, 1)

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

    axes[3].set_xlabel('t [seg]')
    axes[3].set_ylabel(r'$\rm{avg}(d_{ij}) / d_{\rm{max}}$')
    axes[3].grid(1)
    axes[3].plot(tiempo, avg_dist / dmax, ds='steps')
    axes[3].hlines(
        [dmin / dmax, 1],
        xmin=0, xmax=tiempo[-1], ls='--', color='k', alpha=0.7)
    axes[3].set_ylim(bottom=0)
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
