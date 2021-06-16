#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on dom jun 13 20:11:31 -03 2021
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

import uvnpy.rsn.core as rsn
from uvnpy.rsn import distances
import uvnpy.network.core as net
from uvnpy.network import connectivity
from uvnpy.network import disk_graph
from uvnpy.model import linear_models
from uvnpy.filtering import kalman
import uvnpy.network.plot as nplot
from gpsic.grafos.plotting import animar_grafo
from uvnpy.toolkit.calculus import circle2d  # noqa


# ------------------------------------------------------------------
# Definición de variables globales, funciones y clases
# ------------------------------------------------------------------

np.set_printoptions(suppress=True, precision=3, linewidth=150)
Logs = collections.namedtuple('Logs', 'x u E C')


class RangeSensor(object):
    def __init__(self, i, j, var):
        self.__name__ = 'range({}, {})'.format(i, j)
        self.i = i
        self.j = j
        self.var = var
        self.sigma = np.sqrt(var)

    def __call__(self, x):
        p = x.reshape(-1, 2)
        r = p[self.i] - p[self.j]
        hat_z = np.sqrt(np.square(r).sum()).reshape(-1, 1)
        u = r / hat_z
        H = np.zeros(p.shape)
        H[self.i] = u
        H[self.j] = -u
        return hat_z, H.reshape(1, -1), self.var

    def measurement(self, x):
        d = np.sqrt(np.square(x[self.i] - x[self.j]).sum())
        return np.random.normal(d, self.sigma)


class RigidityControl(object):
    def __init__(self, xi, Pi, Q, Rij, H, N, T, dmax, metric):
        self.n = len(xi)
        self.Q = Q * T
        self.R = Rij * np.eye(int(n*(n - 1)/2))
        self.H = H
        self.N = N
        self.horizon = np.linspace(T, N*T, N).reshape(-1, 1, 1)
        self.beta = (10/dmax, 10/dmax)
        self.e = (dmax, dmax)
        self.u = np.zeros(xi.size)
        self.E = np.argwhere(np.triu(1 - np.eye(self.n)))
        self.C = rsn.shape_basis(xi)
        self.metric = metric
        self.init(xi, Pi)

    def init(self, x, P):
        self.xi = x.copy()
        self.x = np.tile(x, self.horizon.shape)
        self.Pi = P.copy()
        self.P = P.copy()

    def restart(self):
        self.x[:] = self.xi.copy()
        self.P = self.Pi.copy()

    def objective_matrix(self):
        self.C = rsn.shape_basis(self.x[-1])
        # C = self.C
        return self.C.T.dot(self.P).dot(self.C)

    def weights(self, i):
        d = distances.from_edges(self.E, self.x)
        w = connectivity.logistic_strength(d, self.beta[i], self.e[i])
        return w

    def covariance_step(self, xk, wk):
        self.P += self.Q
        H = self.H(self.E, xk, wk**(1/2))
        Pz_inv = np.linalg.inv(H.dot(self.P).dot(H.T) + self.R)
        K = self.P.dot(H.T).dot(Pz_inv)
        self.P -= K.dot(H).dot(self.P)

    def prediction(self, u):
        self.x += self.horizon * u.reshape(-1, 2)
        step = zip(self.x, self.weights(1))
        [self.covariance_step(xk, wk) for xk, wk in step]

    def fun(self, u, q):
        self.prediction(u)
        metric = self.metric(self.objective_matrix())
        edges = self.weights(0).sum() / (metric**2)
        effort = u.dot(u)
        self.restart()
        return q[0]*edges + q[1]*(np.exp(metric) - 1) + q[2]*effort

    def update(self, x, P, q):
        self.init(x, P)
        optimization = scipy.optimize.minimize(
            self.fun,
            self.u,
            (q, ),
            method='SLSQP',
        )
        if not optimization.success:
            print(optimization)
        self.u = optimization.x
        return self.u.copy().reshape(self.xi.shape)


def eighmax(M):
    return np.linalg.eigvalsh(M)[-1]
    # return 1/(4.1 - eig)**2


def logdet(M):
    return np.linalg.slogdet(M)[1]


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------


def run():
    # iteracion
    bar = progressbar.ProgressBar(max_value=arg.tf).start()
    # u = np.random.uniform(-2, 2, (n, dof))
    for k, t in steps[1:]:

        # Control
        t_a = time.perf_counter()
        # u = np.zeros((n, dof))
        u = control.update(formacion.x, kf.P, q=(2, 1., 0.05))
        t_b = time.perf_counter()

        # formacion
        x = formacion.step(t, u)
        E = disk_graph.edges(x, dmax)

        # Estimacion
        kf.prediction(kalman.integrator, t, u.reshape(-1, 1), Q * T)
        [kf.correction(s[i, j], s[i, j].measurement(x)) for i, j in E]
        kf.save_data()

        logs.x[k] = x
        logs.u[k] = u
        logs.E[k] = E
        logs.C[k] = rsn.shape_basis(formacion.x)

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
        dest='h', default=100e-3, type=float, help='paso de simulación')
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
        default=1, type=int, help='cantidad de agentes')

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
    dmax = 25.
    lim = 25.

    N = 5
    T = arg.h

    # xi = np.random.uniform(-lim, lim, (n, dof))
    xi = circle2d(R=0.5*lim, N=n)
    ui = np.zeros((n, dof))
    Ei = disk_graph.edges(xi, dmax)
    Pi = 2**2 * np.eye(n*dof)
    Q = 0.1**2 * np.eye(n*dof)
    Rij = 10.**2
    H = distances.jacobian_from_edges
    hat_xi = np.random.multivariate_normal(xi.ravel(), Pi)

    formacion = linear_models.random_walk(xi, Q / T)
    K = net.complete_edges(n)
    s = dict([((e[0], e[1]), RangeSensor(e[0], e[1], Rij)) for e in K])
    kf = kalman.KF(hat_xi, Pi)
    control = RigidityControl(xi, Pi, Q, Rij, H, N, T, dmax, eighmax)

    logs = Logs(
        x=np.empty((tiempo.size, n, dof)),
        u=np.empty((tiempo.size, n, dof)),
        E=np.empty(tiempo.size, dtype=np.ndarray),
        C=np.empty((tiempo.size, n*dof, n*dof - 3))
        )

    logs.x[0] = xi
    logs.u[0] = ui
    logs.E[0] = Ei
    logs.C[0] = rsn.shape_basis(formacion.x)

    run()

    summary = kf.summary()
    kalman.plot(
        summary,
        ground_truth=logs.x.reshape(-1, n*dof, 1),
        basis=logs.C)

    fig, ax = plt.subplots()
    ax.set_title('Control effort')
    ax.plot(tiempo, logs.u.reshape(-1, n*dof))
    ax.grid(1)
    ax.minorticks_on()
    ax.set_xlabel(r'$t [sec]$')
    ax.set_ylabel(r'$u [m/sec]$')

    fig, ax = plt.subplots()
    ax.set_title('Number of edges')
    ax.plot(tiempo, [len(e) for e in logs.E], ds='steps')
    ax.grid(1)
    ax.minorticks_on()
    ax.set_xlabel(r'$t [sec]$')
    ax.set_ylabel(r'$|\mathcal{E}|$')

    # plt.show()
    frame = zip(logs.x, summary.mean.reshape(-1, n, dof), logs.E)
    cuadros = [(np.vstack([x, hat_x]), E) for x, hat_x, E in frame]

    estilos = (
        [range(n), {'color': 'b', 'marker': 'o', 'markersize': '5'}],
        [range(n, 2*n), {'color': 'r', 'marker': '+', 'markersize': '5'}])
    gs = nplot.figure()
    ax, = nplot.xy(gs)
    ax.set_xlim(-1.5*lim, 1.5*lim)
    ax.set_ylim(-1.5*lim, 1.5*lim)

    animar_grafo(
        gs.figure, ax, arg.h, estilos, cuadros,
        edgestyle={'color': '0.2', 'linewidth': 0.7})
