#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date lun jun  7 20:55:18 -03 2021
"""
import numpy as np
import matplotlib.pyplot as plt

from gpsic.grafos.plotting import animar_grafo
from uvnpy.model import linear_models
from uvnpy.filtering import kalman
from uvnpy.rsn import distances
import uvnpy.network.plot as nplot


np.set_printoptions(precision=5, suppress=True)


def LOS(N, repeat, seq):
    in_los = np.empty(N).reshape(repeat, -1)
    b = int(N / repeat / len(seq))
    in_los[:] = np.repeat(seq, b)
    return in_los.ravel()


def range_sensor(x, E, var):
    p = x.reshape(-1, 2)
    hat_z = distances.from_edges(E, p).reshape(-1, 1)
    H = distances.jacobian_from_edges(E, p)
    R = var * np.eye(len(E))
    return hat_z, H, R


if __name__ == '__main__':
    n = 3
    lim = 25

    T = 0.05
    tf = 60
    tiempo = np.arange(0, tf, T)
    N = len(tiempo)

    p, q, r = 5**2, 0.1**2, 3.**2
    Pi = p * np.eye(n*2)
    # Pi = np.kron(np.array([[p, p/2], [p/2, p]]), np.eye(n))
    Q = q * np.eye(n*2)
    # Q = np.kron(np.array([[q, q/2], [q/2, q]]), np.eye(n))
    F = np.eye(n*2)

    # xi = np.random.uniform(-lim, lim, n*2)
    xi = np.array([
        [10.,  0.],
        [-5.,  8.66025404],
        [-5., -8.66025404]]).ravel()
    hat_xi = np.random.multivariate_normal(xi, Pi)

    los = LOS(N, repeat=10, seq=[True])

    planta = linear_models.random_walk(xi, Q / T)
    kf = kalman.KF(hat_xi, Pi)

    x = np.empty((N, n*2))
    x[0] = planta.x
    X = np.hstack([x[0], kf.x.ravel()])

    V = range(n)
    E = np.array([])

    cuadros = np.empty((N, 2), dtype=np.ndarray)
    cuadros[0] = X.reshape(-1, 2), E

    # phi = np.random.uniform(-np.pi, np.pi, (n*2, 1))
    u = np.random.uniform(-1, 1, n*2)
    for k, t in enumerate(tiempo[1:]):
        # u = np.zeros((n*2, 1))
        # u = np.cos(t + phi)
        print(planta.x.shape)
        planta.step(t, u)
        p = planta.x.reshape(-1, 2)

        kf.prediction(kalman.integrator, t, u.reshape(-1, 1), Q * T)

        if los[k+1]:
            E = np.array([
                [0, 1],
                [0, 2],
                [1, 2]])
            d = distances.from_edges(E, p)
            z = d + np.random.normal(0, np.sqrt(r), len(E))
            kf.correction(range_sensor, z.reshape(-1, 1), E, r)
        else:
            E = np.array([])

        x[k+1] = planta.x
        kf.save_data()

        X = np.hstack([x[k+1], kf.x.ravel()])
        cuadros[k+1] = X.reshape(-1, 2), E

    logs = kf.summary()
    # P = logs.cov
    # print(np.linalg.eigvalsh(logs.cov))
    kalman.plot(logs, ground_truth=x)

    plt.show()

    estilos = (
        [V, {'color': 'b', 'marker': 'o', 'markersize': '5'}],
        [range(n, 2*n), {'color': 'r', 'marker': '+', 'markersize': '5'}])
    gs = nplot.figure()
    ax, = nplot.xy(gs)
    ax.set_xlim(-1.5*lim, 1.5*lim)
    ax.set_ylim(-1.5*lim, 1.5*lim)

    animar_grafo(
        gs.figure, ax, T, estilos, cuadros,
        edgestyle={'color': '0.2', 'linewidth': 0.7})
