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

from uvnpy.redes import analisis
from uvnpy.filtering import metricas
from gpsic.grafos.plotting import animar_grafo
from gpsic.plotting.planar import agregar_ax, agregar_linea

H = analisis.distancia_relativa_jac
matriz_incidencia = analisis.matriz_incidencia
conectar = analisis.disk_graph
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
    return [sv[0] / sv[sv > 0][-1] for sv in singvals]


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
    E = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0]])

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    cuadros = [(p, E)]
    singvals = []
    for t in d:
        p[3, 1] = t
        E = np.array(conectar(p, 8.))
        cuadros.append((p.copy(), E))
        D = matriz_incidencia(V, E)
        sv = svdvals(H(p, D))
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
    agregar_linea(
        ax, d, singvals[:, 1],
        color='thistle', label=r'$\sigma_{\rm{2}}, ..., \sigma_{\rm{n}}$',
        ds='steps')
    agregar_linea(ax, d, singvals[:, 2:], color='thistle', ds='steps')
    agregar_linea(
        ax, d, singvals[:, 0],
        color='m', label=r'$\sigma_{\rm{1}}$', ds='steps')
    ax.vlines([3., 6.2], 0, 2., color='0.5', ls='--')

    ax = agregar_ax(
        gs[0, 1],
        title='Funcionales $F(H)$', title_kw={'fontsize': 11},
        xlabel=r'$y_4$ [m]', label_kw={'fontsize': 10})
    agregar_linea(
        ax, d, norma2(singvals),
        label=r'$\Vert H \Vert_{2}$', ds='steps')
    agregar_linea(
        ax, d, nuclear(singvals),
        label=r'$\Vert H \Vert_{N}$', ds='steps')
    agregar_linea(
        ax, d, fro(singvals),
        label=r'$\Vert H \Vert_{F}$', ds='steps')
    agregar_linea(
        ax, d, prod(singvals),
        label=r'$\prod_i \; \sigma_i$', ds='steps')
    agregar_linea(
        ax, d, cond(singvals),
        label=r'$\kappa$', ds='steps')
    ax.vlines([3., 6.2], 0, 8, color='0.5', ls='--')

    if arg.save:
        fig.savefig('/tmp/sv_matriz_observacion.pdf', format='pdf')
    else:
        plt.show()
