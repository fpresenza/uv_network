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
delay = read_csv('/tmp/delay.csv', rows=(0, np.inf), dtype=float)
compl = read_csv('/tmp/compl.csv', rows=(0, np.inf), dtype=float)

# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Delay
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(2.5, 1.5))
fig.tight_layout()
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small'
)
ax.grid(lw=0.4)
ax.set_xlabel(r'$n$', fontsize=8)
ax.set_ylabel(r'Delay', fontsize=8)
ax.plot(
    nodes, np.median(delay, axis=0),
    label=r'$\sigma = 1$', lw=1,
    color='C0',
    marker='o', markersize=3, markevery=10
)
ax.fill_between(
    nodes,
    np.quantile(delay, q=0.25, axis=0),
    np.quantile(delay, q=0.75, axis=0),
    alpha=0.3,
    color='C0',
    # marker='o', markersize=3, markevery=10
)
ax.legend(
    fontsize='x-small', handlelength=1,
    labelspacing=0.3, borderpad=0.2, loc='lower right'
)
fig.savefig('/tmp/bearing_delay.png', format='png', dpi=600)

# ------------------------------------------------------------------
# Complexity
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(2.5, 1.5))
fig.tight_layout()
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small'
)
ax.grid(lw=0.4)
ax.set_xlabel(r'$n$', fontsize=8)
ax.set_ylabel(r'Complexity', fontsize=8)

ax.plot(
    nodes, np.median(compl, axis=0),
    label=r'$\sigma = 1$', lw=1,
    color='C0',
    marker='o', markersize=3, markevery=10
)
ax.fill_between(
    nodes,
    np.quantile(compl, q=0.25, axis=0),
    np.quantile(compl, q=0.75, axis=0),
    alpha=0.3,
    color='C0',
    # marker='o', markersize=3, markevery=10
)
ax.legend(
    fontsize='x-small', handlelength=1,
    labelspacing=0.3, borderpad=0.2, loc='lower right'
)
fig.savefig('/tmp/bearing_complexity.png', format='png', dpi=600)
