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
delay_count = read_csv('/tmp/delay_count.csv', rows=(0, np.inf), dtype=int)
compl = read_csv('/tmp/compl.csv', rows=(0, np.inf), dtype=float)
compl_count = read_csv('/tmp/compl_count.csv', rows=(0, np.inf), dtype=int)

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
ax.set_ylim(0.0, 100.0)
ax.set_xlabel(r'$n$', fontsize=8)
# ax.set_ylabel(r'(%)', fontsize=8)
half = np.array([], dtype=float)
whole = np.array([], dtype=float)

for u, c in zip(delay, delay_count):
    s = c.sum()
    half = np.append(half, c[u <= 0.5].sum() * 100 / s)
    whole = np.append(whole, c[u <= 1.0].sum() * 100 / s)

ax.plot(
    nodes, half,
    label=r'$(h_i \leq 0.5) \, \%$', lw=0.5,
    color='C0',
    marker='o', markersize=3, markevery=10
)
ax.fill_between(
    nodes, 0.0, half,
    color='C0', alpha=0.3
)
ax.plot(
    nodes, whole,
    label=r'$(h_i \leq 1.0) \, \%$', lw=0.5,
    color='C1',
    marker='s', markersize=3, markevery=10
)
ax.fill_between(
    nodes, half, whole,
    color='C1', alpha=0.3
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
ax.set_ylim(0.0, 100.0)
ax.set_xlabel(r'$n$', fontsize=8)
# ax.set_ylabel(r'(%)', fontsize=8)
exp_2 = np.array([], dtype=float)
exp_4 = np.array([], dtype=float)
exp_8 = np.array([], dtype=float)

for u, c in zip(compl, compl_count):
    s = c.sum()
    exp_2 = np.append(exp_2, c[u <= 2].sum() * 100 / s)
    exp_4 = np.append(exp_4, c[u <= 4].sum() * 100 / s)
    exp_8 = np.append(exp_8, c[u <= 8].sum() * 100 / s)

ax.plot(
    nodes, exp_2,
    label=r'$(c_i \leq 2) \, \%$', lw=0.5,
    color='C0',
    marker='o', markersize=3, markevery=10
)
ax.fill_between(
    nodes, 0.0, exp_2,
    color='C0', alpha=0.3
)
ax.plot(
    nodes, exp_4,
    label=r'$(c_i \leq 4) \, \%$', lw=0.5,
    color='C1',
    marker='s', markersize=3, markevery=10
)
ax.fill_between(
    nodes, exp_2, exp_4,
    color='C1', alpha=0.3
)
ax.plot(
    nodes, exp_8,
    label=r'$(c_i \leq 8) \, \%$', lw=0.5,
    color='C2',
    marker='x', markersize=3, markevery=10
)
ax.fill_between(
    nodes, exp_4, exp_8,
    color='C2', alpha=0.3
)
ax.legend(
    fontsize='x-small', handlelength=1,
    labelspacing=0.3, borderpad=0.2, loc='lower right'
)
fig.savefig('/tmp/bearing_complexity.png', format='png', dpi=600)
