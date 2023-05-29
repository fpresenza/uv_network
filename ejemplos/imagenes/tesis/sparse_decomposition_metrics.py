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

from uvnpy.network.subsets import geodesics, fast_degree_load_std
from uvnpy.rsn.rigidity import (
    fast_extents,
    minimum_radius,
    sparse_centers_binary_search,
    sparse_centers_two_steps)
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
    'Logs',
    'nodes \
    diam \
    hmax \
    sparse_hmax \
    sparse_hmax_approx \
    load \
    sparse_load \
    sparse_load_approx \
    edges \
    rmin \
    rmax \
    alpha')


def load_function(degree, hops, geodesics):
    return fast_degree_load_std(degree, hops, geodesics) / len(degree)


# ------------------------------------------------------------------
# Función run
# ------------------------------------------------------------------
def run(d, nmin, nmax, cutoff, logs, threshold, rep):
    bar.start()

    for k, n in enumerate(range(nmin, nmax)):
        bar.update(k)

        diam = 0
        hmax = 0
        sparse_hmax = 0
        sparse_hmax_approx = 0
        load = 0
        sparse_load = 0
        sparse_load_approx = 0
        edges = 0
        rmin = 0
        rmax = 0
        alpha = 0
        for _ in range(rep):
            p = np.random.uniform(0, 1, (n, d))
            A0 = disk_adjacency(p, dmax=2/np.sqrt(n))
            A, Rmin = minimum_radius(A0, p, threshold, return_radius=True)

            h = fast_extents(A, p, threshold)
            geo = geodesics(A)
            deg = A.sum(axis=1)
            diam += np.max(geo)
            hmax += np.max(h)
            load += load_function(deg, geo, h)
            edges += np.sum(A)

            if n < nmin + cutoff:
                sparse_h = sparse_centers_binary_search(
                    A, p, h, fast_degree_load_std, threshold,
                    vertices_only=True)
                sparse_hmax += np.max(sparse_h)
                sparse_load += load_function(deg, sparse_h, geo)

            sparse_h_approx = sparse_centers_two_steps(
                A, p, h, fast_degree_load_std, threshold, vertices_only=True)
            sparse_hmax_approx += np.max(sparse_h_approx)
            sparse_load_approx += load_function(deg, sparse_h_approx, geo)

        logs.nodes[k] = n
        logs.diam[k] = np.divide(diam, rep)
        logs.hmax[k] = np.divide(hmax, rep)
        logs.sparse_hmax[k] = np.divide(sparse_hmax, rep)
        logs.sparse_hmax_approx[k] = np.divide(sparse_hmax_approx, rep)
        logs.load[k] = np.divide(load, rep)
        logs.sparse_load[k] = np.divide(sparse_load, rep)
        logs.sparse_load_approx[k] = np.divide(sparse_load_approx, rep)
        logs.edges[k] = np.divide(edges, 2 * rep)
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
        '-f', '--full',
        default=10, type=int, help='full search')
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
    nmax = arg.nodes + 1
    cutoff = arg.full - nmin
    rep = arg.rep
    threshold = 1e-5
    size = nmax - nmin
    logs = Logs(
        nodes=np.empty(size, dtype=int),
        diam=np.empty(size),
        hmax=np.empty(size),
        sparse_hmax=np.empty(size),
        sparse_hmax_approx=np.empty(size),
        load=np.empty(size),
        sparse_load=np.empty(size),
        sparse_load_approx=np.empty(size),
        edges=np.empty(size),
        rmin=np.empty(size),
        rmax=np.empty(size),
        alpha=np.empty(size))

    # ------------------------------------------------------------------
    # Simulación
    # ------------------------------------------------------------------
    bar = progressbar.ProgressBar(maxval=size)

    logs = run(d, nmin, nmax, cutoff, logs, threshold, rep)
    nodes = logs.nodes
    diam = logs.diam
    hmax = logs.hmax
    sparse_hmax = logs.sparse_hmax
    sparse_hmax_approx = logs.sparse_hmax_approx
    load = logs.load
    sparse_load = logs.sparse_load
    sparse_load_approx = logs.sparse_load_approx
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
    ax.plot(nodes, diam, label=r'$D$', lw=1, color='C0')
    ax.plot(
        nodes, 2*sparse_hmax_approx,
        label=r'$\mathcal{D}(\hat{h})$', lw=1, color='C1')
    diam_ticks = range(2, int(diam.max()) + 2, 2)
    ax.set_yticks(diam_ticks)
    ax.set_yticklabels(diam_ticks)
    ax.legend(
        fontsize='x-small', handlelength=1.5,
        labelspacing=0.5, borderpad=0.2, loc='upper left')
    if arg.save:
        fig.savefig('/tmp/sparse_delay_vs_diam.png', format='png', dpi=360)

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
    ax.set_xticks(nodes[::cutoff//5])
    ax.set_xticklabels(nodes[::cutoff//5])
    ax.plot(
        nodes[:cutoff], 2*sparse_hmax_approx[:cutoff],
        label=r'$\mathcal{D}(\hat{h})$', lw=1, color='C1')
    if cutoff > 0:
        ax.plot(
            nodes[:cutoff], 2*sparse_hmax[:cutoff],
            label=r'$\mathcal{D}(h^{\star})$', lw=1, color='C2')
    ax.legend(
        fontsize='x-small', handlelength=1.5,
        labelspacing=0.5, borderpad=0.2, loc='upper left')
    if arg.save:
        fig.savefig(
            '/tmp/sparse_delay_vs_diam_zoom.png', format='png', dpi=360)

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
    ax.plot(nodes, 2*edges / nodes, label=r'$\bar{\delta}$', lw=1, color='C0')
    ax.plot(
        nodes, sparse_load_approx,
        label=r'$\mathcal{L}(\hat{h})$', lw=1, color='C1')
    ax.legend(
        fontsize='x-small', handlelength=1.5,
        labelspacing=0.5, borderpad=0.2, loc='upper left')
    if arg.save:
        fig.savefig('/tmp/sparse_load_vs_edges.png', format='png', dpi=360)

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
    ax.set_xticks(nodes[::cutoff//5])
    ax.set_xticklabels(nodes[::cutoff//5])
    ax.plot(
        nodes[:cutoff], sparse_load_approx[:cutoff],
        label=r'$\mathcal{L}(\hat{h})$', lw=1, color='C1')
    if cutoff > 0:
        ax.plot(
            nodes[:cutoff], sparse_load[:cutoff],
            label=r'$\mathcal{L}(h^{\star})$', lw=1, color='C2')
    ax.legend(
        fontsize='x-small', handlelength=1.5,
        labelspacing=0.5, borderpad=0.2, loc='upper left')
    if arg.save:
        fig.savefig(
            '/tmp/sparse_load_vs_edges_zoom.png', format='png', dpi=360)

    # plt.show()
