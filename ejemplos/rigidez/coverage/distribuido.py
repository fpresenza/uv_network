#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on lun jun 21 21:42:40 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.linalg
import time
import progressbar
import argparse
import collections
import itertools

from uvnpy.rsn import distances
import uvnpy.network as network
from uvnpy.network import connectivity
from uvnpy.network import disk_graph
from uvnpy.model import linear_models
from uvnpy.toolkit.calculus import circle2d  # noqa
from uvnpy.control import cost_functions


# ------------------------------------------------------------------
# Definición de variables globales, funciones y clases
# ------------------------------------------------------------------

np.set_printoptions(suppress=True, precision=3, linewidth=150)
Logs = collections.namedtuple('Logs', 'x u cost metric E T')


def grid(lim, step):
    g = np.arange(-lim, lim + step, step)
    coords = np.array(list(itertools.product(g, repeat=2)))
    T = np.empty((len(coords), 3), dtype=np.ndarray)
    T[:, :2] = coords
    T[:, 2] = 1
    return T


def make_targets(n, lim):
    coords = np.random.uniform(-lim, lim, (n, 2))
    T = np.empty((len(coords), 3), dtype=np.ndarray)
    T[:, :2] = coords
    T[:, 2] = 1
    return T


def target_allocation(p, t):
    r = p[:, None] - t
    d2 = np.square(r).sum(axis=-1)
    a = p.copy()
    for _ in a:
        if np.any(d2 != np.inf):
            i, j = np.unravel_index(d2.argmin(), d2.shape)
            a[i] = t[j]
            d2[i, :] = d2[:, j] = np.inf
        else:
            return a
    return a


def update_targets(p, T, R):
    r = p[..., None, :] - T[:, :2]
    d2 = np.square(r).sum(axis=-1)
    c2 = (d2 < R**2).any(axis=0)
    T[c2, 2] = 0
    return T.copy()


def untracked_targets(T):
    return np.count_nonzero(T[:, 2])


def mindist(p, E):
    r = p[E[:, 0]] - p[E[:, 1]]
    d2 = np.square(r).sum(axis=-1)
    if len(d2) > 0:
        return np.sqrt(d2.min())


def maxdist(p, E):
    r = p[E[:, 0]] - p[E[:, 1]]
    d2 = np.square(r).sum(axis=-1)
    if len(d2) > 0:
        return np.sqrt(d2.max())


def check_rigidity(p, dmax):
    n, d = p.shape
    A = (disk_graph.adjacency(p, dmax) + np.eye(n)).astype(bool)
    for i in range(n):
        p_i = p[A[i]]
        A_i = disk_graph.adjacency(p_i, dmax)
        L_i = distances.laplacian(A_i, p_i)
        if distances.rigidity(L_i, d):
            print('Grafo {} inicialmente rigido'.format(i))
        else:
            print('Grafo {} inicialmente flexible'.format(i))


class CoverageAnimate(network.plot.Animate):
    def __init__(self, *args, **kwargs):
        super(CoverageAnimate, self).__init__(*args, **kwargs)

    def _update_extra_artists(self, frame):
        P = frame[1]
        T = frame[3]
        for i, p in enumerate(P):
            self._extra_artists[i].center = p
        tracked = T[:, 2] == 0
        Tt = T[tracked]
        Tu = T[~tracked]
        self._extra_artists[-2].set_data(Tt[:, 0], Tt[:, 1])
        self._extra_artists[-1].set_data(Tu[:, 0], Tu[:, 1])


class CoverageControl(object):
    def __init__(self, x, N, h, dmax, a):
        self.horizon = np.linspace(h, N*h, N).reshape(-1, 1, 1)
        self.u = np.zeros(x0.size)
        self.a = a
        self.beta = 40/dmax
        self.e = 0.8*dmax
        self.cost = np.zeros(5)
        self.lambda4 = np.zeros(1)
        self.x = x

    def init(self, x_i, x_j):
        self.x = np.vstack([x_i, x_j])

    def prediction(self, u):
        x = self.x + self.horizon * u.reshape(-1, 2)
        return x

    def distance(self, p, Ta):
        r = p - Ta
        d2 = np.square(r).sum()
        return d2**(1/4)

    def rigidity(self, x):
        A = distances.matrix(x)
        A[A > 0] = connectivity.logistic_strength(A[A > 0], self.beta, self.e)
        L = distances.laplacian(A, x)
        self.lambda4 = np.linalg.eigvalsh(L)[..., 3]
        a = self.a
        return (self.lambda4**a).sum()

    def fun(self, u, u_prev, Ta, c):
        x = self.prediction(u)
        self.cost[0] = c[0] * self.distance(x[:, 0], Ta)
        self.cost[1] = c[1] * self.rigidity(x[-1])
        self.cost[2] = c[2] * cost_functions.collision(x)
        self.cost[3] = c[3] * np.square(u).sum()
        # self.cost[4] = c[4] * np.square(u[:2] - u_prev).sum()
        return self.cost.sum()

    def update(self, x_i, x_j, u_prev, Ta, c):
        self.init(x_i, x_j)
        u0 = np.zeros(self.x.size)
        optimization = scipy.optimize.minimize(
            self.fun,
            u0,
            (u_prev, Ta, c),
            method='SLSQP',
        )
        if not optimization.success:
            print(optimization)
        u = optimization.x
        return u.reshape(-1, 2)

# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(T, R):
    u = np.zeros((n, dof))
    u_prev = u.copy()
    cost = np.zeros(5)

    # iteracion
    bar = progressbar.ProgressBar(max_value=arg.tf).start()
    for k, t in steps[1:]:
        p = formacion.x
        A = disk_graph.adjacency(p, dmax).astype(bool)
        # Control
        t_a = time.perf_counter()
        Tu = T[T[:, 2] == 1]
        Ta = target_allocation(p, Tu[:, :2].astype(float))
        u[:] = 0
        cost[:] = 0

        for i in range(n):
            x_i = p[i]
            N_i = A[i]
            x_j = p[N_i]
            _u = control[i].update(
                x_i, x_j, u_prev[i], Ta[i], c=(5., 2., 3., 1., 1.))
            u[i] += _u[0]
            u[N_i] += _u[1:]
            cost += control[i].cost
            logs.metric[k, i] = control[i].lambda4

        t_b = time.perf_counter()

        # formacion
        u_prev = u.copy()
        x = formacion.step(t, u)
        E = disk_graph.edges(x, dmax)
        T = update_targets(x, T, R)

        logs.x[k] = x
        logs.u[k] = u
        logs.cost[k] = cost
        logs.E[k] = E
        logs.T[k] = T

        perf.append(t_b - t_a)
        bar.update(np.round(t, 3))

    bar.finish()
    st = arg.tf - arg.ti
    rt = sum(perf)
    prompt = 'RT={:.3f} secs, ST={:.3f} secs  ==>  RTF={:.3f}'
    print(prompt.format(rt, st, st / rt))


if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Parseo de argumentos
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-s', '--step',
        dest='h', default=200e-3, type=float, help='paso de simulación')
    parser.add_argument(
        '-t', '--ti',
        metavar='T0', default=0.0, type=float, help='tiempo inicial')
    parser.add_argument(
        '-e', '--tf',
        default=1.0, type=float, help='tiempo final')
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
        default=1, type=int, help='ctntidad de agentes')

    arg = parser.parse_args()

    # ------------------------------------------------------------------
    # Configuración
    # ------------------------------------------------------------------
    tiempo = np.arange(arg.ti, arg.tf, arg.h)
    steps = list(enumerate(tiempo))
    perf = []

    n = arg.n
    if n <= 1:
        raise ValueError('Num. agents must be greater than 1')
    dof = 2
    dmax = 20
    lim = 50.
    R = 3.
    a = -2/n

    N = 5
    h = 1.

    np.random.seed(0)
    # x0 = np.random.uniform(-lim, lim, (n, dof))
    x0 = circle2d(R=dmax/4., N=n)
    # x0 += np.random.normal(0, 1, (n, dof))
    # x0 = np.array(
    #     [[15., 0],
    #     [10, 0],
    #     [5, 0],
    #     [0, 0],
    #     [-5, 0],
    #     [-10, 0]])
    check_rigidity(x0, dmax)
    E0 = disk_graph.edges(x0, dmax)
    # T0 = grid(lim, 1)
    T0 = make_targets(40, lim)

    formacion = linear_models.integrator(x0)
    control = [CoverageControl(x0[i], N, h, dmax, a) for i in range(n)]

    logs = Logs(
        x=np.empty((tiempo.size, n, dof)),
        u=np.empty((tiempo.size, n, dof)),
        cost=np.empty((tiempo.size, 5)),
        metric=np.empty((tiempo.size, n)),
        E=np.empty(tiempo.size, dtype=np.ndarray),
        T=np.empty(tiempo.size, dtype=np.ndarray)
        )

    logs.x[0] = x0
    logs.u[0] = np.zeros((n, dof))
    logs.cost[0] = np.zeros(5)
    logs.metric[0] = None
    logs.E[0] = E0
    logs.T[0] = T0

    run(T0, R)

    fig, ax = plt.subplots(3, 1)
    fig.suptitle('Control')
    ax[0].plot(tiempo, logs.u.reshape(-1, n*dof))
    ax[0].grid(1)
    ax[0].minorticks_on()
    ax[0].set_xlabel(r'$t [sec]$')
    ax[0].set_ylabel(r'$u [m/sec]$')
    ax[1].plot(tiempo, logs.cost[:, 0], label='tracking')
    ax[1].plot(tiempo, logs.cost[:, 1], label='rigidity')
    ax[1].plot(tiempo, logs.cost[:, 2], label='collision')
    ax[1].plot(tiempo, logs.cost[:, 3], label='velocity')
    ax[1].plot(tiempo, logs.cost[:, 4], label='acceleration')
    ax[1].legend()
    ax[1].grid(1)
    ax[1].minorticks_on()
    ax[1].set_xlabel(r'$t [sec]$')
    ax[1].set_ylabel('costs')
    ax[2].plot(tiempo, logs.metric)
    ax[2].grid(1)
    ax[2].minorticks_on()
    ax[2].set_xlabel(r'$t [sec]$')
    ax[2].set_ylabel(r'$\lambda_4$')
    ax[2].set_ylim(0, None)
    fig.savefig('/tmp/control.pdf', format='pdf')

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(tiempo, [untracked_targets(T) for T in logs.T], ds='steps')
    ax[0].grid(1)
    ax[0].minorticks_on()
    ax[0].set_xlabel(r'$t [sec]$')
    ax[0].set_ylabel('Untracked targets')
    mind = [mindist(p, e) for p, e in zip(logs.x, logs.E)]
    maxd = [maxdist(p, e) for p, e in zip(logs.x, logs.E)]
    ax[1].plot(tiempo, mind, ds='steps', label=r'$\rm{min}(d_{ij})$')
    ax[1].plot(tiempo, maxd, ds='steps', label=r'$\rm{max}(d_{ij})$')
    ax[1].hlines(dmax, xmin=0, xmax=tiempo[-1], ls='--', color='k', alpha=0.7)
    ax[1].legend()
    ax[1].grid(1)
    ax[1].minorticks_on()
    ax[1].set_xlabel(r'$t [sec]$')
    ax[1].set_ylabel(r'$dist [m]$')
    fig.savefig('/tmp/targets.pdf', format='pdf')

    fig, ax = network.plot.figure()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_xlim(-1.5*lim, 1.5*lim)
    ax.set_ylim(-1.5*lim, 1.5*lim)

    frames = list(zip(tiempo, logs.x, logs.E, logs.T))
    circles = [plt.Circle(pi, R, alpha=0.4) for pi in x0]
    for circle in circles:
        ax.add_artist(circle)
    tracked = ax.plot([], [], ls='', marker='s', markersize=3, color='y')
    untracked = ax.plot(
        T0[:, 0], T0[:, 1], ls='', marker='s', markersize=3, color='0.5')
    extras = circles + tracked + untracked

    anim = CoverageAnimate(fig, ax, arg.h/4, frames, maxlen=50)
    anim.set_teams({'name': '', 'ids': range(n), 'tail': True})
    anim.set_extra_artists(*extras)
    # anim.ax.legend()
    anim.run(file='/tmp/coverage.mp4')  #
    plt.show()
