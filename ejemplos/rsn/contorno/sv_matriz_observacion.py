#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar dic 15 10:53:03 -03 2020
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

import uvnpy.redes.core as redes
import uvnpy.rsn.core as rsn
from uvnpy.filtering import metricas
from gpsic.grafos.plotting import animar_grafo
from gpsic.plotting.planar import agregar_ax

H = rsn.distances_jac
incidence_from_edges = redes.incidence_from_edges
undirected_edges = redes.undirected_edges
conectar = redes.edges_from_positions
svdvals = metricas.svdvals


def completar(x, size):
    t = size - len(x)
    return np.pad(x, pad_width=(0, t), mode='constant')


def norma2(singvals):
    return singvals[:, 0]


def nuclear(singvals):
    return singvals.sum(1)


def fro(singvals):
    return np.sqrt((singvals * singvals).sum(1))


def prod(singvals):
    return [sv[sv > 0].prod() for sv in singvals]


def cond(singvals):
    return [sv[0] / sv[sv > 0.1][-1] for sv in singvals]


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
    d = np.arange(0, 10, 0.1)
    p = np.array([[-5, 0],
                  [0, -5],
                  [5., 0],
                  [0,  0]])
    equipos = ([[0, 1, 2], {'color': 'r', 'marker': 's', 'markersize': '5'}],
               [[3], {'color': 'b', 'marker': 'o', 'markersize': '5'}])
    V = range(4)
    E = conectar(p, 8.)

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    cuadros = np.empty((d.size, 2), dtype=np.ndarray)
    cuadros[0, 0] = p.copy()
    cuadros[0, 1] = E.copy()
    # cuadros = [(p, E)]
    D = incidence_from_edges(V, E)
    sv = svdvals(H(D, p))
    sv = completar(sv, 8)
    singvals = [sv.copy()]

    for k, t in enumerate(d[1:]):
        p[3, 1] = t
        E = conectar(p, 8.)
        # cuadros.append((p.copy(), E))
        cuadros[k + 1, 0] = p.copy()
        cuadros[k + 1, 1] = E.copy()
        D = incidence_from_edges(V, E)
        sv = svdvals(H(D, p))
        sv = completar(sv, 8)
        singvals.append(sv)

    singvals = np.vstack(singvals)
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

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle('Topología Dinámica')
    fig.subplots_adjust(hspace=0.5, wspace=0.25)
    gs = fig.add_gridspec(1, 2)
    ax = agregar_ax(
        gs[0, 0],
        title=r'Valores singulares $\sigma(H)$', title_kw={'fontsize': 11},
        xlabel=r'$y_4$ [m]', label_kw={'fontsize': 10})
    ax.plot(
        d, singvals[:, 1],
        color='thistle', label=r'$\sigma_{\rm{2}}, ..., \sigma_{\rm{n}}$',
        ds='steps')
    ax.plot(d, singvals[:, 2:], color='thistle', ds='steps')
    ax.plot(
        d, singvals[:, 0],
        color='m', label=r'$\sigma_{\rm{1}}$', ds='steps')
    ax.vlines([3., 6.2], 0, 2., color='0.5', ls='--')
    ax.legend()

    ax = agregar_ax(
        gs[0, 1],
        title='Funcionales $F(H)$', title_kw={'fontsize': 11},
        xlabel=r'$y_4$ [m]', label_kw={'fontsize': 10})
    ax.plot(
        d, norma2(singvals),
        label=r'$\Vert H \Vert_{2}$', ds='steps')
    ax.plot(
        d, nuclear(singvals),
        label=r'$\Vert H \Vert_{N}$', ds='steps')
    ax.plot(
        d, fro(singvals),
        label=r'$\Vert H \Vert_{F}$', ds='steps')
    ax.plot(
        d, prod(singvals),
        label=r'$\prod_i \; \sigma_i$', ds='steps')
    ax.plot(
        d, cond(singvals),
        label=r'$\kappa$', ds='steps')
    ax.vlines([3., 6.2], 0, 8, color='0.5', ls='--')
    ax.legend()

    if arg.save:
        fig.savefig('/tmp/sv_matriz_observacion.pdf', format='pdf')
    else:
        plt.show()
