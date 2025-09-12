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
hd = read_csv('/tmp/hd.csv', rows=(0, np.inf), dtype=int)
hd_count = read_csv('/tmp/hd_count.csv', rows=(0, np.inf), dtype=int)
hb = read_csv('/tmp/hb.csv', rows=(0, np.inf), dtype=int)
hb_count = read_csv('/tmp/hb_count.csv', rows=(0, np.inf), dtype=int)

# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
# Distances
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
# ax.set_ylabel(r'(%)', fontsize=9, labelpad=3)
one_hop = np.array([], dtype=float)
two_hop = np.array([], dtype=float)
three_hop = np.array([], dtype=float)
for n, u, c in zip(nodes, hd, hd_count):
    s = c.sum()
    if 1 in u:
        one_hop = np.append(one_hop, c[u == 1] * 100 / s)
    else:
        one_hop = np.append(one_hop, 0.0)
    if 2 in u:
        two_hop = np.append(two_hop, c[u == 2] * 100 / s)
    else:
        two_hop = np.append(two_hop, 0.0)
    if 3 in u:
        three_hop = np.append(three_hop, c[u == 3] * 100 / s)
    else:
        three_hop = np.append(three_hop, 0.0)
ax.plot(
    nodes, one_hop,
    label=r'$(r^*_i \leq 1) \ \%$', lw=0.5,
    color='C0',
    marker='o', markersize=3, markevery=5
)
ax.fill_between(
    nodes, 0.0, one_hop,
    color='C0', alpha=0.3
)
ax.plot(
    nodes, one_hop + two_hop,
    label=r'$(r^*_i \leq 2) \ \%$', lw=0.5,
    color='C1',
    marker='s', markersize=3, markevery=5
)
ax.fill_between(
    nodes, one_hop, one_hop + two_hop,
    color='C1', alpha=0.3
)
ax.plot(
    nodes, one_hop + two_hop + three_hop,
    label=r'$(r^*_i \leq 3) \ \%$', lw=0.5,
    color='C2',
    marker='x', markersize=3, markevery=5
)
ax.fill_between(
    nodes, one_hop + two_hop, one_hop + two_hop + three_hop,
    color='C2', alpha=0.3
)
ax.legend(
    fontsize='x-small', handlelength=1,
    labelspacing=0.1, borderpad=0.1, loc='center right', handletextpad=0.3
)
fig.savefig('/tmp/distance_extents.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Bearings
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
# ax.set_ylabel(r'(%)', fontsize=9, labelpad=3)
one_hop = np.array([], dtype=float)
two_hop = np.array([], dtype=float)
three_hop = np.array([], dtype=float)
for n, u, c in zip(nodes, hb, hb_count):
    s = c.sum()
    if 1 in u:
        one_hop = np.append(one_hop, c[u == 1] * 100 / s)
    else:
        one_hop = np.append(one_hop, 0.0)
    if 2 in u:
        two_hop = np.append(two_hop, c[u == 2] * 100 / s)
    else:
        two_hop = np.append(two_hop, 0.0)
    if 3 in u:
        three_hop = np.append(three_hop, c[u == 3] * 100 / s)
    else:
        three_hop = np.append(three_hop, 0.0)
ax.plot(
    nodes, one_hop,
    label=r'$(r^*_i \leq 1) \  \%$', lw=0.5,
    color='C0',
    marker='o', markersize=3, markevery=5
)
ax.fill_between(
    nodes, 0.0, one_hop,
    color='C0', alpha=0.3
)
ax.plot(
    nodes, one_hop + two_hop,
    label=r'$(r^*_i \leq 2) \  \%$', lw=0.5,
    color='C1',
    marker='s', markersize=3, markevery=5
)
ax.fill_between(
    nodes, one_hop, one_hop + two_hop,
    color='C1', alpha=0.3
)
ax.plot(
    nodes, one_hop + two_hop + three_hop,
    label=r'$(r^*_i \leq 3) \ \%$', lw=0.5,
    color='C2',
    marker='x', markersize=3, markevery=5
)
ax.fill_between(
    nodes, one_hop + two_hop, one_hop + two_hop + three_hop,
    color='C2', alpha=0.3
)
ax.legend(
    fontsize='x-small', handlelength=1,
    labelspacing=0.1, borderpad=0.1, loc='center', handletextpad=0.3
)
fig.savefig('/tmp/bearing_extents.pdf', bbox_inches='tight')
