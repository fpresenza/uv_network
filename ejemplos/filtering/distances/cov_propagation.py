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


if __name__ == '__main__':
    n = 3
    lim = 25

    T = 0.05
    tf = 60
    tiempo = np.arange(0, tf, T)
    N = len(tiempo)

    p, q, r = 5**2, 0.05**2, 3.**2
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

    planta = linear_models.integrator(xi)

    x = np.empty((N, n*2))
    x[0] = planta.x
    P = np.empty((N, n*2, n*2))
    P[0] = Pi

    V = range(n)
    E = np.array([
        [0, 1],
        [0, 2],
        [1, 2]])

    cuadros = np.empty((N, 2), dtype=np.ndarray)
    cuadros[0] = x[0].reshape(-1, 2), E

    dx = np.tile([1., 0], n)
    dt = np.array([-0., 10., -8.66025, -5., 8.66025, -5.])

    u = np.random.uniform(-1, 1, n*2)
    # u = dt * 0.1
    for k, t in enumerate(tiempo[1:]):
        p = planta.x.reshape(-1, 2)
        # u = 2 * np.cos(t + np.random.uniform(-np.pi, np.pi, n*2))
        # u = np.zeros(n*2)

        planta.step(t, u)
        x[k+1] = planta.x

        P[k+1] = kalman.covariance_prediction(P[k], F, Q * T)

        if los[k+1]:
            A = 1 - np.eye(n)
            H = distances.jacobian_from_adjacency(A, x[k+1].reshape(-1, 2))
            R = r * np.eye(3)
            P[k+1] = kalman.covariance_correction(P[k+1], H, R)
            print(np.linalg.eigvalsh(P[k+1]))
            # print(P[k+1] @ dx)

        cuadros[k+1] = x[k+1].reshape(-1, 2), E

    fig, axes = plt.subplots(2, 1)
    axes[0].plot(tiempo, x)
    axes[0].grid(1)
    axes[0].minorticks_on()
    axes[0].set_xlabel(r'$t [sec]$')
    axes[0].set_ylabel(r'$x(t)$')

    eig = np.linalg.eigvalsh(P)
    axes[1].plot(tiempo, np.sqrt(eig))
    axes[1].grid(1)
    axes[1].minorticks_on()
    axes[1].set_xlabel(r'$t [sec]$')
    axes[1].set_ylabel(r'$\sqrt{\lambda(P)(t)}$')
    plt.show()

    estilos = ([V, {'color': 'b', 'marker': 'o', 'markersize': '5'}],)
    gs = nplot.figure()
    ax, = nplot.xy(gs)
    ax.set_xlim(-1.5*lim, 1.5*lim)
    ax.set_ylim(-1.5*lim, 1.5*lim)

    animar_grafo(
        gs.figure, ax, T, estilos, cuadros,
        edgestyle={'color': '0.2', 'linewidth': 0.7})
