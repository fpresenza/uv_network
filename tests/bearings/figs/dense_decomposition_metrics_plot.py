#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt

from uvnpy.toolkit.data import read_csv


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

# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
# Distances
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(2.5, 2))
fig.subplots_adjust(bottom=0.2, left=0.2)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small'
)
ax.grid(lw=0.4)
ax.set_ylim(0.0, 100.0)
ax.set_xlabel(r'#(nodes)', fontsize=9)
ax.set_ylabel(r'%(subframeworks)', fontsize=9)
# ax.plot(
#     nodes, [np.median(np.repeat(u, c)) for u, c in zip(hd, hd_count)],
#     label=r'$\eta_{\mathrm{dist}}$', lw=0.8
# )
one_hop = []
two_hop = []
three_hop = []
for n, u, c in zip(nodes, hd, hd_count):
    if 1 in u:
        one_hop.append(c[u == 1] * 100 / (50 * n))
    else:
        one_hop.append(0)
    if 2 in u:
        two_hop.append(c[u == 2] * 100 / (50 * n))
    else:
        two_hop.append(0)
    if 3 in u:
        three_hop.append(c[u == 3] * 100 / (50 * n))
    else:
        three_hop.append(0)
ax.plot(
    nodes, one_hop,
    label=r'$1$', lw=0.7, marker='o', markersize=3,
)
ax.plot(
    nodes, two_hop,
    label=r'$2$', lw=0.7, marker='s', markersize=3,
)
ax.plot(
    nodes, three_hop,
    label=r'$3$', lw=0.7, marker='x', markersize=3,
)
ax.legend(
    fontsize='x-small', handlelength=1,
    labelspacing=0.5, borderpad=0.2
)
fig.savefig('/tmp/distance_extents.png', format='png', dpi=400)

# ------------------------------------------------------------------
# Bearings
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(2.5, 2))
fig.subplots_adjust(bottom=0.2, left=0.2)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small')
ax.grid(lw=0.4)
ax.set_ylim(0.0, 100.0)
ax.set_xlabel(r'#(nodes)', fontsize=9)
ax.set_ylabel(r'%(subframeworks)', fontsize=9)
# ax.plot(
#     nodes, [np.median(np.repeat(u, c)) for u, c in zip(hb, hb_count)],
#     label=r'$\eta_{\mathrm{bear}}$', lw=0.8
# )
one_hop = []
two_hop = []
three_hop = []
for n, u, c in zip(nodes, hb, hb_count):
    if 1 in u:
        one_hop.append(c[u == 1] * 100 / (50 * n))
    else:
        one_hop.append(0)
    if 2 in u:
        two_hop.append(c[u == 2] * 100 / (50 * n))
    else:
        two_hop.append(0)
    if 3 in u:
        three_hop.append(c[u == 3] * 100 / (50 * n))
    else:
        three_hop.append(0)
ax.plot(
    nodes, one_hop,
    label=r'$1$', lw=0.7, marker='o', markersize=3,
)
ax.plot(
    nodes, two_hop,
    label=r'$2$', lw=0.7, marker='s', markersize=3,
)
ax.plot(
    nodes, three_hop,
    label=r'$3$', lw=0.7, marker='x', markersize=3,
)
ax.legend(
    fontsize='x-small', handlelength=1,
    labelspacing=0.5, borderpad=0.2
)
fig.savefig('/tmp/bearing_extents.png', format='png', dpi=400)
