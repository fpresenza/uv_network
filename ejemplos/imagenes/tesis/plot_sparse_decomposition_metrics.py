#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt


np.set_printoptions(suppress=True, precision=4, linewidth=250)
plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


# ------------------------------------------------------------------
# Parseo de argumentos
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-f', '--full',
    default=4, type=int, help='full search')

arg = parser.parse_args()


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

nodes = np.loadtxt('/tmp/nodes.csv', delimiter=',')
diam = np.loadtxt('/tmp/diam.csv', delimiter=',')
hmax = np.loadtxt('/tmp/hmax.csv', delimiter=',')
sparse_hmax = np.loadtxt('/tmp/sparse_hmax.csv', delimiter=',')
sparse_hmax_subopt = np.loadtxt('/tmp/sparse_hmax_subopt.csv', delimiter=',')
sparse_load = np.loadtxt('/tmp/sparse_load.csv', delimiter=',')
sparse_load_subopt = np.loadtxt('/tmp/sparse_load_subopt.csv', delimiter=',')
edges = np.loadtxt('/tmp/edges.csv', delimiter=',')

d = 2
nmin = d + 2
nmax = int(nodes.max())
cutoff = arg.full - nmin

# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
# Diametro y Retardo
fig, ax = plt.subplots(figsize=(3, 2))
fig.subplots_adjust(bottom=0.2)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='small')
ax.grid(lw=0.4)
ax.set_xlabel('Número de vértices ($v$)', fontsize=10)
ax.set_xticks(np.arange(20, nmax + 1, 20))
ax.set_xticklabels(np.arange(20, nmax + 1, 20))
ax.plot(
    nodes, 2 * np.median(sparse_hmax_subopt, axis=1),
    label=r'$\mathcal{D}(\hat{h})$', lw=1)
ax.fill_between(
    nodes,
    2 * np.quantile(sparse_hmax_subopt, 0.25, axis=1),
    2 * np.quantile(sparse_hmax_subopt, 0.75, axis=1),
    alpha=0.3)
ax.plot(
    nodes, np.median(diam, axis=1),
    label=r'$D$', lw=1)
ax.fill_between(
    nodes,
    np.quantile(diam, 0.25, axis=1),
    np.quantile(diam, 0.75, axis=1),
    alpha=0.3)
diam_ticks = np.arange(0, 12, 2)
ax.set_yticks(diam_ticks)
ax.set_yticklabels(diam_ticks)
ax.set_ylim(0, 10)
ax.legend(
    fontsize='small', handlelength=1.5,
    labelspacing=0.5, borderpad=0.2, loc='upper left')
fig.savefig('/tmp/sparse_delay_vs_diam.png', format='png', dpi=360)

fig, ax = plt.subplots(figsize=(3, 2))
fig.subplots_adjust(bottom=0.2)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='small')
ax.grid(lw=0.4)
ax.set_xlabel('Número de vértices ($v$)', fontsize=10)
ax.set_xticks(np.arange(4, 16, 4))
ax.set_xticklabels(np.arange(4, 16, 4))
ax.plot(
    nodes[:cutoff], 2 * np.median(sparse_hmax[:cutoff], axis=1),
    label=r'$\mathcal{D}(h^{\star})$', lw=1)
ax.fill_between(
    nodes[:cutoff],
    2 * np.quantile(sparse_hmax[:cutoff], 0.25, axis=1),
    2 * np.quantile(sparse_hmax[:cutoff], 0.75, axis=1),
    alpha=0.3)
ax.plot(
    nodes[:cutoff], 2 * np.median(sparse_hmax_subopt[:cutoff], axis=1),
    label=r'$\mathcal{D}(\hat{h})$', lw=1)
ax.fill_between(
    nodes[:cutoff],
    2 * np.quantile(sparse_hmax_subopt[:cutoff], 0.25, axis=1),
    2 * np.quantile(sparse_hmax_subopt[:cutoff], 0.75, axis=1),
    alpha=0.3)
diam_ticks = np.arange(0, 5, 1)
ax.set_yticks(diam_ticks)
ax.set_yticklabels(diam_ticks)
ax.set_ylim(0, 4)
ax.legend(
    fontsize='small', handlelength=1.5,
    labelspacing=0.5, borderpad=0.2, loc='upper left')
fig.savefig('/tmp/sparse_delay_vs_diam_zoom.png', format='png', dpi=360)

# Enlaces y Carga
fig, ax = plt.subplots(figsize=(3, 2))
fig.subplots_adjust(bottom=0.2)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='small')
ax.grid(lw=0.4)
ax.set_xlabel('Número de vértices ($v$)', fontsize=10)
ax.set_xticks(np.arange(20, nmax + 1, 20))
ax.set_xticklabels(np.arange(20, nmax + 1, 20))
ax.plot(
    nodes, np.median(sparse_load_subopt, axis=1),
    label=r'$\mathcal{L}(\hat{h})$', lw=1)
ax.fill_between(
    nodes,
    np.quantile(sparse_load_subopt, 0.25, axis=1),
    np.quantile(sparse_load_subopt, 0.75, axis=1),
    alpha=0.3)
ax.plot(
    nodes,
    2 * np.median(edges, axis=1) / nodes,
    label=r'$\bar{n}$', lw=1)
ax.fill_between(
    nodes,
    2 * np.quantile(edges, 0.25, axis=1) / nodes,
    2 * np.quantile(edges, 0.75, axis=1) / nodes,
    alpha=0.3)
load_ticks = np.arange(0, 20, 5)
ax.set_yticks(load_ticks)
ax.set_yticklabels(load_ticks)
ax.set_ylim(0, 15)
ax.legend(
    fontsize='small', handlelength=1.5,
    labelspacing=0.5, borderpad=0.2, loc='upper left')
fig.savefig('/tmp/sparse_load_vs_edges.png', format='png', dpi=360)

fig, ax = plt.subplots(figsize=(3, 2))
fig.subplots_adjust(bottom=0.2)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='small')
ax.grid(lw=0.4)
ax.set_xlabel('Número de vértices ($v$)', fontsize=10)
ax.set_xticks(np.arange(4, 16, 4))
ax.set_xticklabels(np.arange(4, 16, 4))
ax.plot(
    nodes[:cutoff], np.median(sparse_load[:cutoff], axis=1),
    label=r'$\mathcal{L}(h^{\star})$', lw=1)
ax.fill_between(
    nodes[:cutoff],
    np.quantile(sparse_load[:cutoff], 0.25, axis=1),
    np.quantile(sparse_load[:cutoff], 0.75, axis=1),
    alpha=0.3)
ax.plot(
    nodes[:cutoff], np.median(sparse_load_subopt[:cutoff], axis=1),
    label=r'$\mathcal{L}(\hat{h})$', lw=1)
ax.fill_between(
    nodes[:cutoff],
    np.quantile(sparse_load_subopt[:cutoff], 0.25, axis=1),
    np.quantile(sparse_load_subopt[:cutoff], 0.75, axis=1),
    alpha=0.3)
load_ticks = np.arange(0, 3, 1)
ax.set_yticks(load_ticks)
ax.set_yticklabels(load_ticks)
ax.set_ylim(0, 3)
ax.legend(
    fontsize='small', handlelength=1.5,
    labelspacing=0.5, borderpad=0.2, loc='upper left')
fig.savefig('/tmp/sparse_load_vs_edges_zoom.png', format='png', dpi=360)
