#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on jue jul  8 23:21:53 -03 2021
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

import uvnpy.network as network
from uvnpy.network import connectivity
from uvnpy.network import disk_graph
import uvnpy.rsn as rsn
from uvnpy.rsn import distances
from uvnpy.model import linear_models
from uvnpy.filtering import kalman
from uvnpy.toolkit.calculus import circle2d  # noqa
from uvnpy.control import cost_functions


# ------------------------------------------------------------------
# Definición de variables globales, funciones y clases
# ------------------------------------------------------------------

np.set_printoptions(suppress=True, precision=3, linewidth=150)
Logs = collections.namedtuple('Logs', 'x u P cost metric E T')


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


def update_targets(p, T, dcov):
    r = p[..., None, :] - T[:, :2]
    d2 = np.square(r).sum(axis=-1)
    c2 = (d2 < dcov**2).any(axis=0)
    T[c2, 2] = 0
    return T.copy()


def untracked_targets(T):
    return np.count_nonzero(T[:, 2])


def check_rigidity(p, dmax):
    A = disk_graph.adjacency(p, dmax)
    L = distances.laplacian(A, p)
    d = p.shape[-1]
    if distances.rigidity(L, d):
        print('Grafo inicialmente rigido')
    else:
        print('Grafo inicialmente flexible')


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
    def __init__(self, x0, P0, Q, Rij, H, N, tau, dmax):
        n = len(x0)
        self.Q = Q * tau
        self.R = Rij * np.eye(int(n*(n - 1)/2))
        self.H = H
        self.horizon = np.linspace(tau, N*tau, N).reshape(-1, 1, 1)
        self.u = np.zeros(x0.size)
        self.a = -2/n
        self.beta = 10/dmax     # 40/dmax
        self.e = 0.8*dmax
        self.cost = np.zeros(5)
        self.lmax = np.zeros(1)
        self.E = network.complete_edges(n)
        self.init(x0, P0)

    def init(self, x, P):
        self.x = x.copy()
        self.P0 = P.copy()
        self.P = P.copy()

    def prediction(self, u):
        x = self.x + self.horizon * u.reshape(-1, 2)
        d = distances.from_edges(self.E, x)
        w = connectivity.logistic_strength(d, self.beta, self.e)
        [self.covariance_step(xk, wk) for xk, wk in zip(x, w)]
        P = self.P.copy()
        self.P = self.P0.copy()
        return x, P

    def covariance_step(self, xk, wk):
        self.P += self.Q
        H = self.H(self.E, xk, wk**(1/2))
        Pz_inv = np.linalg.inv(H.dot(self.P).dot(H.T) + self.R)
        K = self.P.dot(H.T).dot(Pz_inv)
        self.P -= K.dot(H).dot(self.P)

    def distance(self, p, Ta):
        r = p - Ta
        d2 = np.square(r).sum(axis=-1)
        return (d2**(1/4)).sum()

    def stochastic_rigidity(self, x, P):
        C = rsn.shape_basis(x)
        M = C.T.dot(P).dot(C)
        self.lmax = np.linalg.eigvalsh(M)[-1]
        return np.exp(0.7*self.lmax)

    def fun(self, u, Ta, c):
        x, P = self.prediction(u)
        # self.cost[0] = c[0] * self.distance(x, Ta)
        self.cost[1] = c[1] * self.stochastic_rigidity(x[-1], P)
        self.cost[2] = c[2] * cost_functions.collision(x)
        self.cost[3] = c[3] * np.square(u).sum()
        self.cost[4] = c[4] * np.square(u - self.u).sum()
        return self.cost.sum()

    def update(self, x, P, Ta, c):
        self.init(x, P)
        optimization = scipy.optimize.minimize(
            self.fun,
            self.u,
            (Ta, c, ),
            method='SLSQP',
        )
        if not optimization.success:
            print(optimization)
        self.u = optimization.x
        return self.u.copy().reshape(self.x.shape)

# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(P, T, dcov):
    # iteracion
    bar = progressbar.ProgressBar(max_value=arg.tf).start()
    x = formacion.x
    for k, t in steps[1:]:

        # Control
        t_a = time.perf_counter()
        Tu = T[T[:, 2] == 1]
        Ta = target_allocation(x, Tu[:, :2].astype(float))
        u = control.update(x, P, Ta, c=(5., 3., 5., 1., 5.))
        t_b = time.perf_counter()

        # formacion
        x = formacion.step(t, u)
        E = disk_graph.edges(x, dmax)
        F = np.eye(n*2)
        H = distances.jacobian_from_edges(E, x)
        R = Rij * np.eye(len(E))
        P = kalman.two_step_covariance(P, F, Q*tau, H, R)
        T = update_targets(x, T, dcov)
        C = rsn.shape_basis(x)

        logs.x[k] = x
        logs.u[k] = u
        logs.P[k] = C.T.dot(P).dot(C)
        logs.cost[k] = control.cost
        logs.metric[k] = control.lmax
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
    dcov = 3.

    N = 10
    tau = 0.5

    np.random.seed(0)
    # x0 = np.random.uniform(-lim, lim, (n, dof))
    x0 = circle2d(R=dmax, N=n)
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
    P0 = 2**2 * np.eye(n*dof)
    Q = 0.25**2 * np.eye(n*dof)
    Rij = 60.**2
    H = distances.jacobian_from_edges
    T0 = make_targets(40, lim)
    C0 = rsn.shape_basis(x0)

    formacion = linear_models.integrator(x0)
    control = CoverageControl(x0, P0, Q, Rij, H, N, tau, dmax)

    logs = Logs(
        x=np.empty((tiempo.size, n, dof)),
        u=np.empty((tiempo.size, n, dof)),
        P=np.empty((tiempo.size, dof*n - 3, dof*n - 3)),
        cost=np.empty((tiempo.size, 5)),
        metric=np.empty((tiempo.size)),
        E=np.empty(tiempo.size, dtype=np.ndarray),
        T=np.empty(tiempo.size, dtype=np.ndarray)
        )

    logs.x[0] = x0
    logs.u[0] = np.zeros((n, dof))
    logs.P[0] = C0.T.dot(P0).dot(C0)
    logs.cost[0] = np.zeros(5)
    logs.metric[0] = None
    logs.E[0] = E0
    logs.T[0] = T0

    run(P0, T0, dcov)

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
    ax[2].set_ylabel(r'$\lambda_{\rm{max}}$')
    ax[2].set_ylim(0, None)
    fig.savefig('/tmp/control.pdf', format='pdf')

    fig, ax = plt.subplots()
    ax.plot(tiempo, np.linalg.eigvalsh(logs.P))
    ax.grid(1)
    ax.minorticks_on()
    ax.set_xlabel(r'$t [sec]$')
    ax.set_ylabel('Covariance eigenvalues')
    fig.savefig('/tmp/covariance.pdf', format='pdf')

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
    circles = [plt.Circle(pi, dcov, alpha=0.4) for pi in x0]
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
    anim.run()  # file='/tmp/coverage.mp4'
    plt.show()
