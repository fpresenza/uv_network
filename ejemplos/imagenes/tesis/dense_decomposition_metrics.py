#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import argparse
import collections
import numpy as np
import matplotlib.pyplot as plt
import progressbar

from uvnpy.network.core import geodesics
from uvnpy.rsn.rigidity import extents, minimum_radius
from uvnpy.rsn.distances import matrix as distance_matrix
from uvnpy.rsn.distances import matrix_between as distance_between
from uvnpy.network.disk_graph import adjacency as disk_adjacency

np.set_printoptions(suppress=True, precision=4, linewidth=250)
plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'

# ------------------------------------------------------------------
# Definición de variables, funciones y clases
# ------------------------------------------------------------------
Logs = collections.namedtuple(
    'Logs', 'nodes diam hmax load edges rmin rmax alpha')


def generate_position(n, xlim, ylim, radius):
    """Genera una posicion donde los nodos estan a mas
    de cierto radio de separacion"""
    xl, xh = xlim
    yl, yh = ylim
    p = np.random.uniform((xl, yl), (xh, yh), 2)
    for i in range(1, n):
        found = False
        while not found:
            q = np.random.uniform((xl, yl), (xh, yh), 2)
            dist = distance_between(p, q)
            if np.all(dist > radius):
                found = True
                p = np.vstack([p, q])

    return p


def minimum_alpha(A, p, dist, Rmin, Rmax, threshold=1e-5):
    """Calcula el alfa necesario para que todos los subframeworks
    tengan alcance unitario"""
    A = A.copy()
    h = extents(A, p, threshold)
    if np.all(h == 1):
        return 0.
    else:
        radius = np.unique(dist[dist > Rmin])
        for R in radius:
            A[dist == R] = 1
            h = extents(A, p, threshold)
            if np.all(h == 1):
                alpha = (R - Rmin) / (Rmax - Rmin)
                return alpha


def load_function(degree, geodesics, hops):
    coeff = (hops.reshape(-1, 1) - geodesics).clip(min=0)
    # return coeff.dot(degree).sum() / coeff.sum()
    return coeff.dot(degree).sum() / len(degree)


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------
def run(d, nmin, nmax, logs, threshold, rep):
    bar.start()

    for k, n in enumerate(range(nmin, nmax)):
        bar.update(k)

        diam = [0, 0, 0]
        hmax = [0, 0, 0]
        load = [0, 0, 0]
        edges = [0, 0, 0]
        A = [None, None, None]
        rmin = 0
        rmax = 0
        alpha = 0
        for _ in range(rep):
            # p = generate_position(n, (0, 1), (0, 1), 0.02)
            p = np.random.uniform(0, 1, (n, d))
            A0 = disk_adjacency(p, dmax=2/np.sqrt(n))
            dist = distance_matrix(p)
            Rmax = dist.max()
            A[0], Rmin = minimum_radius(A0, p, threshold, return_radius=True)
            # Alpha = minimum_alpha(A[0], p, dist, Rmin, Rmax, threshold)
            rmin += Rmin
            rmax += Rmax
            # alpha += Alpha

            h = extents(A[0], p, threshold)
            geo = geodesics(A[0])
            deg = A[0].sum(axis=1)
            diam[0] += np.max(geo)
            hmax[0] += np.max(h)
            load[0] += load_function(deg, geo, h)
            edges[0] += np.sum(A[0])

            A[1] = disk_adjacency(p, dmax=Rmin + 0.05 * (Rmax - Rmin))
            h = extents(A[1], p, threshold)
            geo = geodesics(A[1])
            deg = A[1].sum(axis=1)
            diam[1] += np.max(geo)
            hmax[1] += np.max(h)
            load[1] += load_function(deg, geo, h)
            edges[1] += np.sum(A[1])

            A[2] = disk_adjacency(p, dmax=Rmin + 0.1 * (Rmax - Rmin))
            h = extents(A[2], p, threshold)
            geo = geodesics(A[2])
            deg = A[2].sum(axis=1)
            diam[2] += np.max(geo)
            hmax[2] += np.max(h)
            load[2] += load_function(deg, geo, h)
            edges[2] += np.sum(A[2])

        logs.nodes[k] = n
        logs.diam[k] = np.divide(diam, rep)
        logs.hmax[k] = np.divide(hmax, rep)
        logs.load[k] = np.divide(load, rep)
        logs.edges[k] = np.divide(edges, 2*rep)
        logs.rmin[k] = rmin / rep
        logs.rmax[k] = rmax / rep
        logs.alpha[k] = alpha / rep

    bar.finish()
    return logs


if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Parseo de argumentos
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-n', '--nodes',
        default=50, type=int, help='number of nodes')
    parser.add_argument(
        '-r', '--rep',
        default=10, type=int, help='number of repetitions')
    parser.add_argument(
        '-s', '--save',
        default=False, action='store_true', help='flag to store data')

    arg = parser.parse_args()

    # ------------------------------------------------------------------
    # Configuración
    # ------------------------------------------------------------------
    d = 2
    nmin = d + 2
    nmax = arg.nodes
    threshold = 1e-5
    size = nmax - nmin
    logs = Logs(
        nodes=np.empty(size, dtype=int),
        diam=np.empty((size, 3)),
        hmax=np.empty((size, 3)),
        load=np.empty((size, 3)),
        edges=np.empty((size, 3)),
        rmin=np.empty(size),
        rmax=np.empty(size),
        alpha=np.empty(size))

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    bar = progressbar.ProgressBar(maxval=size)

    logs = run(d, nmin, nmax, logs, threshold, arg.rep)
    nodes = logs.nodes
    diam = logs.diam
    hmax = logs.hmax
    load = logs.load
    edges = logs.edges
    rmin = logs.rmin
    rmax = logs.rmax
    alpha = logs.alpha

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    # Diametro y Retardo
    fig, ax = plt.subplots(figsize=(3, 2))
    fig.subplots_adjust(bottom=0.2)
    # ax.set_title(r'$\alpha = 0$')
    ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize='x-small')
    ax.grid(lw=0.4)
    ax.set_xlabel('Número de vértices ($v$)', fontsize=8)
    ax.set_xticks(nodes[::nmax//5])
    ax.set_xticklabels(nodes[::nmax//5])
    ax.plot(nodes, diam[:, 0], label=r'$D$', lw=1)
    ax.plot(nodes, 2*hmax[:, 0], label=r'$\mathcal{D}(h_0)$', lw=1)
    diam_ticks = range(2, int(diam[:, 0].max()) + 2, 2)
    ax.set_yticks(diam_ticks)
    ax.set_yticklabels(diam_ticks)
    ax.legend(
        fontsize='x-small', handlelength=1.5,
        labelspacing=0.5, borderpad=0.2, loc='upper left')
    if arg.save:
        fig.savefig('/tmp/delay_vs_diam_1.png', format='png', dpi=360)

    fig, ax = plt.subplots(figsize=(3, 2))
    fig.subplots_adjust(bottom=0.2)
    # ax.set_title(r'$\alpha = 0.1$')
    ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize='x-small')
    ax.grid(lw=0.4)
    ax.set_xlabel('Número de vértices ($v$)', fontsize=8)
    ax.set_xticks(nodes[::nmax//5])
    ax.set_xticklabels(nodes[::nmax//5])
    ax.plot(nodes, diam[:, 1], label=r'$D$', lw=1)
    ax.plot(nodes, 2*hmax[:, 1], label=r'$\mathcal{D}(h_0)$', lw=1)
    diam_ticks = range(2, int(diam[:, 1].max()) + 2, 2)
    ax.set_yticks(diam_ticks)
    ax.set_yticklabels(diam_ticks)
    ax.legend(
        fontsize='x-small', handlelength=1.5,
        labelspacing=0.5, borderpad=0.2, loc='upper left')
    if arg.save:
        fig.savefig('/tmp/delay_vs_diam_2.png', format='png', dpi=360)

    fig, ax = plt.subplots(figsize=(3, 2))
    fig.subplots_adjust(bottom=0.2)
    # ax.set_title(r'$\alpha = 0.2$')
    ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize='x-small')
    ax.grid(lw=0.4)
    ax.set_xlabel('Número de vértices ($v$)', fontsize=8)
    ax.set_xticks(nodes[::nmax//5])
    ax.set_xticklabels(nodes[::nmax//5])
    ax.plot(nodes, diam[:, 2], label=r'$D$', lw=1)
    ax.plot(nodes, 2*hmax[:, 2], label=r'$\mathcal{D}(h_0)$', lw=1)
    diam_ticks = range(2, int(diam[:, 1].max()) + 1, 1)
    ax.set_yticks(diam_ticks)
    ax.set_yticklabels(diam_ticks)
    ax.legend(
        fontsize='x-small', handlelength=1.5,
        labelspacing=0.5, borderpad=0.2, loc='upper left')
    if arg.save:
        fig.savefig('/tmp/delay_vs_diam_3.png', format='png', dpi=360)

    # Enlaces y Carga
    fig, ax = plt.subplots(figsize=(3, 2))
    fig.subplots_adjust(bottom=0.2)
    # ax.set_title(r'$\alpha = 0$')
    ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize='x-small')
    ax.grid(lw=0.4)
    ax.set_xlabel('Número de vértices ($v$)', fontsize=8)
    ax.set_xticks(nodes[::nmax//5])
    ax.set_xticklabels(nodes[::nmax//5])
    ax.plot(nodes, 2*edges[:, 0] / nodes, label=r'$\bar{\delta}$', lw=1)
    ax.plot(nodes, load[:, 0], label=r'$\mathcal{L}(h_0)$', lw=1)
    # diam_ticks = range(2, int(diam[:, 0].max()) + 2, 2)
    # ax.set_yticks(diam_ticks)
    # ax.set_yticklabels(diam_ticks)
    ax.legend(
        fontsize='x-small', handlelength=1.5,
        labelspacing=0.5, borderpad=0.2, loc='upper left')
    if arg.save:
        fig.savefig('/tmp/load_vs_edges_1.png', format='png', dpi=360)

    fig, ax = plt.subplots(figsize=(3, 2))
    fig.subplots_adjust(bottom=0.2)
    # ax.set_title(r'$\alpha = 0$')
    ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize='x-small')
    ax.grid(lw=0.4)
    ax.set_xlabel('Número de vértices ($v$)', fontsize=8)
    ax.set_xticks(nodes[::nmax//5])
    ax.set_xticklabels(nodes[::nmax//5])
    ax.plot(nodes, 2*edges[:, 1] / nodes, label=r'$\bar{\delta}$', lw=1)
    ax.plot(nodes, load[:, 1], label=r'$\mathcal{L}(h_0)$', lw=1)
    # diam_ticks = range(2, int(diam[:, 0].max()) + 2, 2)
    # ax.set_yticks(diam_ticks)
    # ax.set_yticklabels(diam_ticks)
    ax.legend(
        fontsize='x-small', handlelength=1.5,
        labelspacing=0.5, borderpad=0.2, loc='upper left')
    if arg.save:
        fig.savefig('/tmp/load_vs_edges_2.png', format='png', dpi=360)

    fig, ax = plt.subplots(figsize=(3, 2))
    fig.subplots_adjust(bottom=0.2)
    # ax.set_title(r'$\alpha = 0$')
    ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize='x-small')
    ax.grid(lw=0.4)
    ax.set_xlabel('Número de vértices ($v$)', fontsize=8)
    ax.set_xticks(nodes[::nmax//5])
    ax.set_xticklabels(nodes[::nmax//5])
    ax.plot(nodes, 2*edges[:, 2] / nodes, label=r'$\bar{\delta}$', lw=1)
    ax.plot(nodes, load[:, 2], label=r'$\mathcal{L}(h_0)$', lw=1)
    # diam_ticks = range(2, int(diam[:, 0].max()) + 2, 2)
    # ax.set_yticks(diam_ticks)
    # ax.set_yticklabels(diam_ticks)
    ax.legend(
        fontsize='x-small', handlelength=1.5,
        labelspacing=0.5, borderpad=0.2, loc='upper left')
    if arg.save:
        fig.savefig('/tmp/load_vs_edges_3.png', format='png', dpi=360)

    # fig, ax = plt.subplots(figsize=(3, 2))
    # fig.subplots_adjust(bottom=0.2, left=0.23)
    # ax.tick_params(
    #     axis='both',       # changes apply to the x-axis
    #     which='both',      # both major and minor ticks are affected
    #     pad=1,
    #     labelsize='x-small')
    # ax.grid(lw=0.4)
    # ax.set_xlabel(r'Número de vértices ($v$)', fontsize=8)
    # ax.set_ylabel('Rangos \n normalizados', fontsize=8)
    # ax.set_xticks(nodes[::nmax//5])
    # ax.set_xticklabels(nodes[::nmax//5])
    # ax.plot(nodes, rmin / np.sqrt(2), label=r'$\rho_0 / \sqrt{2}$', lw=1)
    # ax.plot(nodes, rmax / np.sqrt(2), label=r'$\rho_1 / \sqrt{2}$', lw=1)
    # # ax.plot(
    #    nodes, (rmin + rmax) / np.sqrt(2) / 2,
    #    label=r'$\rho_1 / \sqrt{2}$', lw=1)
    # ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.])
    # ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.])
    # ax.legend(
    #     fontsize='x-small', handlelength=1.5,
    #     labelspacing=0.5, borderpad=0.2)
    # if arg.save:
    #     fig.savefig('/tmp/min_max_radius.png', format='png', dpi=360)

    # fig, ax = plt.subplots(figsize=(3, 2))
    # fig.subplots_adjust(bottom=0.2, left=0.23)
    # ax.tick_params(
    #     axis='both',       # changes apply to the x-axis
    #     which='both',      # both major and minor ticks are affected
    #     pad=1,
    #     labelsize='x-small')
    # ax.grid(lw=0.4)
    # ax.set_xlabel(r'Número de vértices ($v$)', fontsize=8)
    # ax.set_ylabel(r'$\alpha^{\star} \in [0, 1]$', fontsize=8)
    # ax.set_xticks(nodes[::nmax//5])
    # ax.set_xticklabels(nodes[::nmax//5])
    # ax.plot(nodes, alpha, label=r'$\alpha$', lw=1)
    #     # ax.legend(
    #     #     fontsize='x-small', handlelength=1.5,
    #     #     labelspacing=0.5, borderpad=0.2, loc='upper left')
    # if arg.save:
    #     fig.savefig('/tmp/minimum_alpha.png', format='png', dpi=360)

    plt.show()
