#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date lun dic 14 15:27:31 -03 2020
"""
import argparse
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt  # noqa

from uvnpy.modelos.lineal import integrador_ruidoso
from uvnpy.filtering import kalman, metricas
from uvnpy.redes.analisis import incidence_from_edges
from gpsic.grafos.plotting import animar_grafo
from gpsic.plotting.planar import agregar_ax


eigvalsh = metricas.eigvalsh


def norma(v):
    sqr_sum = np.multiply(v, v).sum(1)
    return np.sqrt(sqr_sum)


def medicion(p, E, sigma):
    z = norma([p[i] - p[j] for i, j in E])
    return np.random.normal(z, sigma)


def h(x, E):
    p = x.reshape(-1, 2)
    return norma([p[i] - p[j] for i, j in E])


def H(x, E):
    p = x.reshape(-1, 2)
    V = range(len(p))
    D = incidence_from_edges(V, E)
    diff = D.T.dot(p)
    sqrdiff = diff * diff
    dist = np.sqrt(sqrdiff.sum(1))
    r = diff / dist.reshape(-1, 1)
    M = scipy.linalg.block_diag(*r)
    Dn = np.kron(D, np.eye(p.shape[1]))
    return M.dot(Dn.T)


class ekf(kalman.KF):
    def __init__(self, x, dx, Q, sigma, ti=0.):
        super(ekf, self).__init__(x, dx, ti=0.)
        self.Id = np.identity(len(x))
        self.Q = Q
        self.sigma = sigma

    def prior(self, dt, u):
        self._x = self._x + np.multiply(dt, u)
        self._P = self._P + self.Q * dt

    def observacion(self, z, E):
        x = self._x
        dz = z - h(x, E)
        var = self.sigma[0]**2
        R = np.diag([var for _ in E])
        return dz, H(x, E), R


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
        '-g', '--save',
        default=False, action='store_true',
        help='flag para guardar los videos')
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
    n = arg.n
    dim = 2 * n
    # xi = np.random.uniform(-10, 10, (n, 2))
    xi = np.array([-5, 0., 0, -5, 5, 0, 0, 5])
    dxi = 0.1 * np.ones(dim)
    Q = 0.2 * np.eye(dim)
    sigma_gps = 3.
    sigma_xbee = 3.
    sigma = (sigma_xbee, sigma_gps)
    equipos = ([[0, 1], {'color': 'r', 'marker': '*', 'markersize': '7'}],
               [[2, 3], {'color': 'b', 'marker': 'o', 'markersize': '5'}],
               [[4, 5], {'color': 'r', 'marker': '*', 'markersize': '7',
                         'alpha': 0.4}],
               [[6, 7], {'color': 'b', 'marker': 'o', 'markersize': '5',
                         'alpha': 0.4}])

    t = np.arange(arg.ti, arg.tf, arg.h)
    formacion = integrador_ruidoso(xi, Q)
    filtro = ekf(xi, dxi, Q, sigma)
    V = range(n)
    E = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0]])
    D = incidence_from_edges(V, E)
    # E_est = np.array([
    #     [4, 5],
    #     [5, 6],
    #     [6, 7],
    #     [7, 4]])
    # E_aug = np.vstack([E, E_est])

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    p = np.reshape(formacion.x, (n, 2))
    p_est = np.reshape(filtro.x, (n, 2))
    p_aug = np.vstack([p, p_est])
    X = [p_aug[:, [0]]]
    Y = [p_aug[:, [1]]]
    eigs = [eigvalsh(filtro.P)]
    cuadros = [(p_aug, E)]
    for tk in t[1:]:
        u = np.zeros(dim)

        formacion.step(tk, u)
        p = np.reshape(formacion.x, (n, 2))

        filtro.prediccion(tk, u)
        p_est = np.reshape(filtro.x, (n, 2))

        z = medicion(p, E, sigma[0])
        obsv = filtro.observacion(z, E)
        filtro.actualizacion(*obsv)
        eigs.append(eigvalsh(filtro.P))

        p_aug = np.vstack([p, p_est])
        cuadros.append((p_aug, E))
        X.append(p_aug[:, [0]])
        Y.append(p_aug[:, [1]])

    X = np.hstack(X)
    Y = np.hstack(Y)
    sqrt_eigs = np.sqrt(eigs)
    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    if arg.animate:
        fig, ax = plt.subplots()
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal')
        ax.grid(1)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        animar_grafo(fig, ax, arg.h, equipos, cuadros, guardar=arg.save)

    fig = plt.figure(figsize=(13, 5))
    fig.subplots_adjust(hspace=0.5, wspace=0.25)
    gs = fig.add_gridspec(2, 2)
    # posición
    ax = [None, None, None, None]
    ax[0] = agregar_ax(
        gs[0, 0],
        title='Pos. x (verdadero vs. estimado)', title_kw={'fontsize': 11},
        xlabel='t [seg]', ylabel='x [m]', label_kw={'fontsize': 10})
    ax[1] = agregar_ax(
        gs[0, 1],
        title='Pos. y (verdadero vs. estimado)', title_kw={'fontsize': 11},
        xlabel='t [seg]', ylabel='y [m]', label_kw={'fontsize': 10})
    ax[2] = agregar_ax(
        gs[1, 0],
        title='Pos. x (error)', title_kw={'fontsize': 11},
        xlabel='t [seg]', ylabel='x [m]', label_kw={'fontsize': 10})
    ax[3] = agregar_ax(
        gs[1, 1],
        title='Pos. y (error)', title_kw={'fontsize': 11},
        xlabel='t [seg]', ylabel='y [m]', label_kw={'fontsize': 10})

    for i in V:
        l, = ax[0].plot(t, X[i], label=i)
        ax[0].plot(t, X[i + n], ls='dotted', color=l.get_c())
        l, = ax[1].plot(t, Y[i], label=i)
        ax[1].plot(t, Y[i + n], ls='dotted', color=l.get_c())
        ax[2].plot(t, X[i] - X[i + n], label=i)
        ax[3].plot(t, Y[i] - Y[i + n], label=i)

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()

    fig = plt.figure()
    gs = fig.add_gridspec(1, 1)
    ax = agregar_ax(
        gs[0, 0],
        title='Valores singulares Covarianza $^{1/2}$',
        title_kw={'fontsize': 11},
        xlabel='t [seg]', ylabel='[m]', label_kw={'fontsize': 10})
    ax.plot(t, sqrt_eigs[:, 0], color='c', label=r'$\sqrt{\sigma_{\rm{min}}}$')
    ax.plot(t, sqrt_eigs[:, 1:-1], color='0.5')
    ax.plot(
        t, sqrt_eigs[:, -1], color='m', label=r'$\sqrt{\sigma_{\rm{max}}}$')
    ax.legend()

    if arg.save:
        fig.savefig('/tmp/rango_central_analisis.pdf', format='pdf')
    else:
        plt.show()
