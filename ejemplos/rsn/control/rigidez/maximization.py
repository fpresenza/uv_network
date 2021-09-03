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

from uvnpy.model import linear_models
import uvnpy.network as network
from uvnpy.rsn import rigidity
from uvnpy.toolkit import calculus
from uvnpy.control.cost_functions import collision

np.set_printoptions(suppress=True, precision=3, linewidth=150)
# ------------------------------------------------------------------
# Definición de variables globales, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple('Logs', 'x u eig')

D = calculus.derivative_eval


def lambda4(x):
    L = rigidity.complete_laplacian(x)
    return np.linalg.eigvalsh(L)[..., 6]


def logdet(x):
    L = rigidity.complete_laplacian(x)
    eig = np.linalg.eigvalsh(L)[..., 6:]
    return np.log(eig.prod(-1))


def gradient(f, x):
    u = D(f, x)
    return u.reshape(x.shape)


def repulsion(x):
    u = D(collision, x, axis=(1, 2))
    return -u.reshape(x.shape)


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

        u = 10 * gradient(logdet, x)  # + repulsion(x)

        t_b = time.perf_counter()

        x = planta.step(t, u)

        # Análisis
        L = rigidity.laplacian(A, x)
        eigvals = np.linalg.eigvalsh(L)[6:]

        E = network.complete_edges(n)
        frames[k] = t, x[:, :2], E

        logs.x[k] = x
        logs.u[k] = u
        logs.eig[k] = eigvals

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
    V = range(n)
    dof = 3
    dn = dof * n
    L = 10
    np.random.seed(1)
    # x0 = np.random.uniform(-0.5 * dmax, 0.5 * dmax, (n, dof))
    x0 = np.random.uniform(-L, L, (n, dof))

    planta = linear_models.integrator(x0, tiempo[0])

    A = 1 - np.eye(n)
    L0 = rigidity.laplacian(A, x0)

    logs = Logs(
        x=np.empty((tiempo.size, n, dof)),
        u=np.empty((tiempo.size, n, dof)),
        eig=np.empty((tiempo.size, dn - 6)),
        )
    logs.x[0] = x0
    logs.u[0] = np.zeros((n, dof))
    logs.eig[0] = None

    frames = np.empty((tiempo.size, 3), dtype=np.ndarray)
    E = network.complete_edges(n)
    frames[0] = tiempo[0], x0[:, :2], E

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    logs, t_perf, frames = run(steps, logs, t_perf, planta, frames)

    x = logs.x
    u = logs.u
    eig = logs.eig

    st = arg.tf - arg.ti
    rt = sum(t_perf)
    prompt = 'RT={:.3f} secs, ST={:.3f} secs  ==>  RTF={:.3f}'
    print(prompt.format(rt, st, st / rt))

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    print(x[-1])

    fig, axes = plt.subplots(2, 1)

    axes[0].set_title('posicion')
    axes[0].set_xlabel('t [seg]')
    axes[0].set_ylabel('(x, y) [m]')
    axes[0].grid(1)
    axes[0].plot(tiempo, x[..., 0], ds='steps')
    plt.gca().set_prop_cycle(None)
    axes[0].plot(tiempo, x[..., 1], ds='steps')
    plt.gca().set_prop_cycle(None)
    axes[0].plot(tiempo, x[..., 2], ds='steps')

    axes[1].set_title('accion de control')
    axes[1].set_xlabel('t [seg]')
    axes[1].set_ylabel('u [m/s]')
    axes[1].grid(1)
    axes[1].plot(tiempo, u[..., 0], ds='steps')
    plt.gca().set_prop_cycle(None)
    axes[1].plot(tiempo, u[..., 1], ds='steps')
    plt.gca().set_prop_cycle(None)
    axes[1].plot(tiempo, u[..., 2], ds='steps')
    fig.savefig('/tmp/control.pdf', format='pdf')

    fig, ax = plt.subplots()

    ax.set_xlabel('t [seg]')
    ax.set_ylabel(r'$\lambda(L)$')
    ax.grid(1)
    # ax.semilogy(tiempo, eig, ds='steps')
    ax.plot(tiempo, eig, ds='steps')
    # ax.set_ylim(bottom=0)
    fig.savefig('/tmp/metricas.pdf', format='pdf')

    fig, ax = network.plot.figure()
    ax.set_xlim(-1.5*L, 1.5*L)
    ax.set_ylim(-1.5*L, 1.5*L)
    anim = network.plot.Animate(fig, ax, arg.h/2, frames, maxlen=50)
    anim.set_teams({'ids': V, 'tail': True})
    anim.ax.legend()
    anim.run()

    plt.show()
