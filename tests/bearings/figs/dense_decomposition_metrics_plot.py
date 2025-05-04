#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

from uvnpy.toolkit.data import read_csv


# ------------------------------------------------------------------
# Parseo de argumentos
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-g', '--degree',
    default=1.0, type=float, help='average vertex degree'
)

arg = parser.parse_args()

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
diam = read_csv('/tmp/diam.csv', rows=(0, np.inf), dtype=int)
diam_count = read_csv('/tmp/diam_count.csv', rows=(0, np.inf), dtype=int)
hd = read_csv('/tmp/hd.csv', rows=(0, np.inf), dtype=int)
hd_count = read_csv('/tmp/hd_count.csv', rows=(0, np.inf), dtype=int)
hb = read_csv('/tmp/hb.csv', rows=(0, np.inf), dtype=int)
hb_count = read_csv('/tmp/hb_count.csv', rows=(0, np.inf), dtype=int)

rep = hd_count[0].item() / (arg.degree + 1)
# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
# Distances
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(2.5, 1.5))
fig.subplots_adjust(bottom=0.25, left=0.2)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small'
)
ax.grid(lw=0.4)
ax.set_ylim(0.0, 100.0)
ax.set_xlabel(r'$n$', fontsize=9)
ax.set_ylabel(r'(%)', fontsize=9, labelpad=3)
# ax.plot(
#     nodes, [np.median(np.repeat(u, c)) for u, c in zip(hd, hd_count)],
#     label=r'$\eta_{\mathrm{dist}}$', lw=0.8
# )
one_hop = []
two_hop = []
three_hop = []
for n, u, c in zip(nodes, hd, hd_count):
    if 1 in u:
        one_hop.append(c[u == 1] * 100 / (rep * n))
    else:
        one_hop.append(0)
    if 2 in u:
        two_hop.append(c[u == 2] * 100 / (rep * n))
    else:
        two_hop.append(0)
    if 3 in u:
        three_hop.append(c[u == 3] * 100 / (rep * n))
    else:
        three_hop.append(0)
ax.plot(
    nodes, one_hop,
    label=r'$n_1$', lw=0.7, marker='o', markersize=3, markevery=5
)
ax.plot(
    nodes, two_hop,
    label=r'$n_2$', lw=0.7, marker='s', markersize=3, markevery=5
)
ax.plot(
    nodes, three_hop,
    label=r'$n_3$', lw=0.7, marker='x', markersize=3, markevery=5
)
ax.legend(
    fontsize='x-small', handlelength=1,
    labelspacing=0.3, borderpad=0.2, loc='center right'
)
fig.savefig('/tmp/distance_extents.png', format='png', dpi=600)

# ------------------------------------------------------------------
# Bearings
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(2.5, 1.5))
fig.subplots_adjust(bottom=0.25, left=0.2)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small')
ax.grid(lw=0.4)
ax.set_ylim(0.0, 100.0)
ax.set_xlabel(r'$n$', fontsize=9)
ax.set_ylabel(r'(%)', fontsize=9, labelpad=3)
# ax.plot(
#     nodes, [np.median(np.repeat(u, c)) for u, c in zip(hb, hb_count)],
#     label=r'$\eta_{\mathrm{bear}}$', lw=0.8
# )
one_hop = []
two_hop = []
three_hop = []
for n, u, c in zip(nodes, hb, hb_count):
    if 1 in u:
        one_hop.append(c[u == 1] * 100 / (rep * n))
    else:
        one_hop.append(0)
    if 2 in u:
        two_hop.append(c[u == 2] * 100 / (rep * n))
    else:
        two_hop.append(0)
    if 3 in u:
        three_hop.append(c[u == 3] * 100 / (rep * n))
    else:
        three_hop.append(0)
ax.plot(
    nodes, one_hop,
    label=r'$n_1$', lw=0.7, marker='o', markersize=3, markevery=5
)
ax.plot(
    nodes, two_hop,
    label=r'$n_2$', lw=0.7, marker='s', markersize=3, markevery=5
)
ax.plot(
    nodes, three_hop,
    label=r'$n_3$', lw=0.7, marker='x', markersize=3, markevery=5
)
ax.legend(
    fontsize='x-small', handlelength=1,
    labelspacing=0.3, borderpad=0.2, loc='center'
)

# ax2 = ax.twinx()
# ax2.tick_params(
#     axis='both',       # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     pad=1,
#     labelsize='x-small')
# ax2.set_yticks([1, 2, 3, 4])
# ax2.plot(
#     nodes, [np.mean(np.repeat(u, c)) for u, c in zip(diam, diam_count)],
#     label=r'$\Delta(\mathcal{G})$', lw=0.8, color='k'
# )
fig.savefig('/tmp/bearing_extents.png', format='png', dpi=600)
