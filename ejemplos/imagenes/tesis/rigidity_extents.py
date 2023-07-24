#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

from uvnpy.rsn.rigidity import fast_extents, minimum_radius
from uvnpy.rsn.distances import matrix_between as distance_between
from uvnpy.network import disk_graph, plot
from uvnpy.network.subsets import geodesics

np.set_printoptions(suppress=True, precision=4, linewidth=250)

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


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


parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-s', '--seed',
    default=0, type=int, help='numpy random seed')
arg = parser.parse_args()

n = 20
threshold = 1e-5

i = 0
np.random.seed(arg.seed)
while i < 10:
    # p = np.random.uniform((0, 0), (1, 0.9), (n, 2))
    p = generate_position(n, (0, 1), (0, 0.9), 0.1)
    A0 = disk_graph.adjacency(p, dmax=2/np.sqrt(n))
    A, Rmin = minimum_radius(A0, p, threshold, return_radius=True)

    fig, ax = plt.subplots(figsize=(2.25, 2.25))
    # fig.subplots_adjust(top=0.88, bottom=0.15, wspace=0.28)
    h = fast_extents(A, p, threshold)
    geo = geodesics(A)
    D = np.max(geo)

    if np.max(h) < 4 and np.max(h) != D:
        i += 1
        print('---{}---'.format(i))
        print(p)
    else:
        continue

    one_hop_rigid = h == 1
    two_hop_rigid = h == 2
    three_hop_rigid = h == 3

    ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,
        left=False,
        pad=1,
        labelsize='x-small')
    ax.grid(1, lw=0.4)
    ax.set_aspect('equal')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    # ax.set_xlabel(r'$\mathrm{x}$', fontsize='x-small', labelpad=0.6)
    # ax.set_ylabel(r'$\mathrm{y}$', fontsize='x-small', labelpad=0)
    ax.set_xticks(np.linspace(0, 1, 4, endpoint=True))
    ax.set_yticks(np.linspace(0, 1, 4, endpoint=True))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if np.any(one_hop_rigid):
        plot.nodes(
            ax, p[one_hop_rigid],
            marker='o', color='royalblue', s=11, zorder=10, label=r'$h_r=1$')
    if np.any(two_hop_rigid):
        plot.nodes(
            ax, p[two_hop_rigid],
            marker='D', color='chocolate', s=11, zorder=10, label=r'$h_r=2$')
    if np.any(three_hop_rigid):
        plot.nodes(
            ax, p[three_hop_rigid],
            marker='s', color='mediumseagreen',
            s=11, zorder=10, label=r'$h_r=3$')
    plot.edges(ax, p, A, color='0.0', lw=0.4)
    ax.legend(
        fontsize='xx-small', handlelength=1, labelspacing=0.4,
        borderpad=0.2, handletextpad=0.2, framealpha=1.,
        ncol=3, columnspacing=0.2, loc='upper center')
    fig.savefig(
        '/tmp/rigidity_extents_{}.png'.format(i), format='png', dpi=360)
