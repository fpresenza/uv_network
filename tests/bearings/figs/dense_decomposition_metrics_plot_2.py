#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt

from uvnpy.toolkit.data import read_csv

# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------

np.set_printoptions(suppress=True, precision=4, linewidth=250)
plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
nodes = np.loadtxt('/tmp/nodes.csv', delimiter=',')
diam = read_csv('/tmp/diam.csv', rows=(0, np.inf), dtype=float)
diam_count = read_csv('/tmp/diam_count.csv', rows=(0, np.inf), dtype=int)
compl = read_csv('/tmp/compl.csv', rows=(0, np.inf), dtype=float)
compl_count = read_csv('/tmp/compl_count.csv', rows=(0, np.inf), dtype=int)

# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Diameter
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(2.5, 1.5))
fig.subplots_adjust(bottom=0.25, left=0.2)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small')
ax.grid(lw=0.4)
# ax.set_ylim(0.0, 100.0)
ax.set_xlabel(r'$n$', fontsize=9)
ax.set_ylabel(r'ratio', fontsize=9, labelpad=3)
ax.plot(
    nodes, [np.min(np.repeat(u, d)) for u, d in zip(diam, diam_count)],
    label='min', lw=0.8, ds='steps-post'
)
ax.plot(
    nodes, [np.mean(np.repeat(u, d)) for u, d in zip(diam, diam_count)],
    label='mean', lw=0.8, ds='steps-post'
)
ax.legend(
    fontsize='x-small', handlelength=1,
    labelspacing=0.3, borderpad=0.2
)
fig.savefig('/tmp/diameter.png', format='png', dpi=600)
# ------------------------------------------------------------------
# Complexity
# ------------------------------------------------------------------
fig.savefig('/tmp/complexity.png', format='png', dpi=600)
fig, ax = plt.subplots(figsize=(2.5, 1.5))
fig.subplots_adjust(bottom=0.25, left=0.2)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small')
ax.grid(lw=0.4)
# ax.set_ylim(0.0, 100.0)
ax.set_xlabel(r'$n$', fontsize=9)
ax.set_ylabel(r'ratio', fontsize=9, labelpad=3)
ax.plot(
    nodes, [np.min(np.repeat(u, c)) for u, c in zip(compl, compl_count)],
    label='min', lw=0.8, ds='steps-post'
)
# ax.plot(
#     nodes, [np.median(np.repeat(u, c)) for u, c in zip(compl, compl_count)],
#     label='median', lw=0.8, ds='steps'
# )
ax.plot(
    nodes, [np.mean(np.repeat(u, c)) for u, c in zip(compl, compl_count)],
    label='mean', lw=0.8, ds='steps-post'
)
ax.legend(
    fontsize='x-small', handlelength=1,
    labelspacing=0.3, borderpad=0.2
)
fig.savefig('/tmp/complexity.png', format='png', dpi=600)
