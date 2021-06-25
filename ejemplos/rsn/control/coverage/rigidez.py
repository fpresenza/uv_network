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
Logs = collections.namedtuple('Logs', 'x u E T')


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


def target_allocation(p, q):
    r = p[..., None, :] - q
    d2 = np.square(r).sum(axis=-1)
    if d2.shape[-1] > 0:
        ct = d2.argmin(axis=-1)
        return q[ct]
    else:
        return p


def update_targets(p, T, R):
    r = p[..., None, :] - T[:, :2]
    d2 = np.square(r).sum(axis=-1)
    c2 = (d2 < R**2).any(axis=0)
    T[c2, 2] = 0
    return T.copy()


def check_rigidity(p, dmax):
    A = disk_graph.adjacency(p, dmax)
    L = distances.laplacian(A, p)
    d = p.shape[-1]
    if distances.rigidity(L, d):
        return 'Grafo inicialmente rigido'
    else:
        return 'Grafo inicialmente flexible'


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
    def __init__(self, xi, N, h, dmax):
        n = len(xi)
        self.horizon = np.linspace(h, N*h, N).reshape(-1, 1, 1)
        self.u = np.zeros(xi.size)
        self.a = -2/n
        self.beta = 40/dmax
        self.e = 0.8*dmax
        self.w = np.arange(n)[::-1]
        # self.w = np.ones(n)
        self.init(xi)

    def init(self, x):
        self.x = x.copy()

    def prediction(self, u):
        x = self.x + self.horizon * u.reshape(-1, 2)
        return x

    def distance(self, p, Ta):
        r = p - Ta
        d2 = np.square(r).sum(axis=-1)
        d = np.sqrt(d2)
        return np.sqrt(d.dot(self.w))

    def rigidity(self, x):
        A = distances.matrix(x)
        A[A > 0] = connectivity.logistic_strength(A[A > 0], self.beta, self.e)
        L = distances.laplacian(A, x)
        lambda2 = np.linalg.eigvalsh(L)[:, 3]
        return (lambda2**self.a).sum()

    def fun(self, u, Ta, q):
        x = self.prediction(u)
        ct = self.distance(x[-1], Ta)
        cm = self.rigidity(x)
        cc = cost_functions.collision(x)
        cu = np.square(u).sum()
        cdu = np.square(u - self.u).sum()
        return q[0]*ct + q[1]*cm + q[2]*cc + q[3]*cu + q[4]*cdu

    def update(self, x, Ta, q):
        self.init(x)
        optimization = scipy.optimize.minimize(
            self.fun,
            self.u,
            (Ta, q, ),
            method='SLSQP',
        )
        if not optimization.success:
            print(optimization)
        self.u = optimization.x
        return self.u.copy().reshape(self.x.shape)

# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run(T, R):
    # iteracion
    bar = progressbar.ProgressBar(max_value=arg.tf).start()
    for k, t in steps[1:]:

        # Control
        t_a = time.perf_counter()
        Tu = T[T[:, 2] == 1]
        Ta = target_allocation(formacion.x, Tu[:, :2].astype(float))
        u = control.update(formacion.x, Ta, q=(5., 1., 1., 1., 10.))
        # print(u)
        t_b = time.perf_counter()

        # formacion
        x = formacion.step(t, u)
        E = disk_graph.edges(x, dmax)
        T = update_targets(x, T, R)

        logs.x[k] = x
        logs.u[k] = u
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

    N = 5
    h = 1.

    np.random.seed(0)
    # xi = np.random.uniform(-lim, lim, (n, dof))
    xi = circle2d(R=dmax/4., N=n)
    # xi = np.array(
    #     [[15., 0],
    #     [10, 0],
    #     [5, 0],
    #     [0, 0],
    #     [-5, 0],
    #     [-10, 0]])
    print(check_rigidity(xi, dmax))
    ui = np.zeros((n, dof))
    Ei = disk_graph.edges(xi, dmax)
    # Ti = grid(lim, 1)
    Ti = make_targets(40, lim)

    formacion = linear_models.integrator(xi)
    control = CoverageControl(xi, N, h, dmax)

    logs = Logs(
        x=np.empty((tiempo.size, n, dof)),
        u=np.empty((tiempo.size, n, dof)),
        E=np.empty(tiempo.size, dtype=np.ndarray),
        T=np.empty(tiempo.size, dtype=np.ndarray)
        )

    logs.x[0] = xi
    logs.u[0] = ui
    logs.E[0] = Ei
    logs.T[0] = Ti

    run(Ti, R)

    fig, ax = plt.subplots()
    ax.set_title('Control effort')
    ax.plot(tiempo, logs.u.reshape(-1, n*dof))
    ax.grid(1)
    ax.minorticks_on()
    ax.set_xlabel(r'$t [sec]$')
    ax.set_ylabel(r'$u [m/sec]$')
    fig.savefig('/tmp/control.pdf', format='pdf')

    fig, ax = plt.subplots()
    ax.set_title('Number of untracked targets')
    ax.plot(tiempo, [np.count_nonzero(t[:, 2]) for t in logs.T], ds='steps')
    ax.grid(1)
    ax.minorticks_on()
    ax.set_xlabel(r'$t [sec]$')
    ax.set_ylabel(r'$|T|$')
    fig.savefig('/tmp/targets.pdf', format='pdf')

    fig, ax = network.plot.figure()
    ax.set_xlim(-1.5*lim, 1.5*lim)
    ax.set_ylim(-1.5*lim, 1.5*lim)

    frames = list(zip(tiempo, logs.x, logs.E, logs.T))
    circles = [plt.Circle(pi, R, alpha=0.4) for pi in xi]
    for circle in circles:
        ax.add_artist(circle)
    tracked = ax.plot([], [], ls='', marker='s', markersize=3, color='g')
    untracked = ax.plot(
        Ti[:, 0], Ti[:, 1], ls='', marker='s', markersize=3, color='r')
    extras = circles + tracked + untracked

    anim = CoverageAnimate(fig, ax, arg.h/4, frames, maxlen=50)
    anim.set_teams({'name': 'ground truth', 'ids': range(n), 'tail': True})
    anim.set_extra_artists(*extras)
    # anim.ax.legend()
    anim.run(file='/tmp/coverage.mp4')  # file='/tmp/coverage.mp4'
    plt.show()
