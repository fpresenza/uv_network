#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

from uvnpy import network

np.set_printoptions(suppress=True, precision=4, linewidth=250)

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


def path_graph(n, d=1):
    """Creates the (n, d)-path graph on n vertices.

    The (n, d)-path graph is a graph such that i, j are
    connected if and only if 0 < |i - j| <= d.
    For d = 1, it equals the classical path graph.

    Requires: n >= d + 1
    """
    A = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i + 1, min(i + d + 1, n)):
            A[i, j] = A[j, i] = 1
    return A


def cycle_graph(n, d=1):
    """Creates the (n, d)-cycle graph on n vertices.

    The (n, d)-cycle graph is a circulant graph such that i, j are
    connected if and only if 0 < (i - j) mod n <= d or 0 < (j - i) mod n <= d.
    For d = 1, it equals the classical cycle graph.

    Requires: n >= d + 1
    """
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if 0 < min((j - i) % n, (i - j) % n) <= d:
                A[i, j] = A[j, i] = 1

    return A


def path_realization(n):
    # generalized path graph positions
    q = np.zeros((n, 2))
    q[:, 0] = np.arange(n)
    q[1::2, 1] = np.sqrt(3)
    return q


def cycle_realization(n):
    # generalized cycle graph positions
    r = np.empty((2*n, 2))
    r[:n] = path_realization(n) + (0, 1)
    r[n:] = np.flip(r[:n], axis=0) * (1, -1)
    return r


n = 11
d = 3

P = path_graph(n, d)
PP = scipy.linalg.block_diag(P, P)
C = cycle_graph(2*n, d)

p = cycle_realization(n)

# cycle graph symmetry
fig, ax = plt.subplots(figsize=(3.5, 5.5))
# fig.subplots_adjust(top=0.88, bottom=0.15, wspace=0.28)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,
    left=False,
    pad=1,
    labelsize='x-small')
ax.grid(0)
ax.set_aspect('equal')
# ax.set_xlim(-0.05, 1.05)
# ax.set_ylim(-0.05, 1.05)
# ax.set_xlabel(r'$\mathrm{x}$', fontsize='x-small', labelpad=0.6)
# ax.set_ylabel(r'$\mathrm{y}$', fontsize='x-small', labelpad=0)
ax.set_xticks(np.linspace(0, 1, 4, endpoint=True))
ax.set_yticks(np.linspace(0, 1, 4, endpoint=True))
ax.set_xticklabels([])
ax.set_yticklabels([])

network.plot.nodes(
    ax, p[:n],
    marker='o', color='lightblue', s=50, zorder=10)
network.plot.nodes(
    ax, p[n:],
    marker='o', color='mediumseagreen', s=50, zorder=10)
network.plot.edges(
    ax, p[:n], P,
    color='0.0', lw=0.4)
network.plot.edges(
    ax, p[n:], P,
    color='0.0', lw=0.4)
network.plot.edges(
    ax, p, C - PP,
    color='orange', lw=0.6)

ax.hlines(0, -0.5, 10.5, lw=0.4, ls='--', color='k')
ax.vlines(n//2, -3.5, 3.5, lw=0.4, ls='--', color='k')

for i in range(2*n):
    ax.annotate(
        '${}$'.format(i), xy=p[i], color='k',
        fontsize='xx-small', weight='normal',
        horizontalalignment='center',
        verticalalignment='center', zorder=20)


# cycle graph symmetry
fig, ax = plt.subplots(figsize=(3.5, 5.5))
# fig.subplots_adjust(top=0.88, bottom=0.15, wspace=0.28)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,
    left=False,
    pad=1,
    labelsize='x-small')
ax.grid(0)
ax.set_aspect('equal')
# ax.set_xlim(-0.05, 1.05)
# ax.set_ylim(-0.05, 1.05)
# ax.set_xlabel(r'$\mathrm{x}$', fontsize='x-small', labelpad=0.6)
# ax.set_ylabel(r'$\mathrm{y}$', fontsize='x-small', labelpad=0)
ax.set_xticks(np.linspace(0, 1, 4, endpoint=True))
ax.set_yticks(np.linspace(0, 1, 4, endpoint=True))
ax.set_xticklabels([])
ax.set_yticklabels([])

network.plot.nodes(
    ax, p[:n],
    marker='o', color='lightblue', s=70, zorder=10)
network.plot.nodes(
    ax, p[n:],
    marker='o', color='lightblue', s=70, zorder=10)
network.plot.edges(
    ax, p[:n], P,
    color='0.0', lw=0.4)
network.plot.edges(
    ax, p[n:], P,
    color='0.0', lw=0.4)
network.plot.edges(
    ax, p, C - PP,
    color='orange', lw=0.6)

ax.hlines(0, -0.5, 10.5, lw=0.4, ls='--', color='k')
ax.vlines(n//2, -3.5, 3.5, lw=0.4, ls='--', color='k')

for i in range(2*n):
    if i < n:
        if i < n//2:
            j = i
            s = '+'
        elif i > n//2:
            j = n - i - 1
            s = '-'
        else:
            continue
    else:
        if i < n + n//2:
            j = i - n
            s = '-'
        elif i > n + n//2:
            j = 2*n - i - 1
            s = '+'
        else:
            continue
    ax.annotate(
        r'{}${}$'.format(s, j), xy=p[i], color='k',
        fontsize='xx-small', weight='normal',
        horizontalalignment='center',
        verticalalignment='center', zorder=20)

fig.savefig('/tmp/generalized_cycle_symmetry.png', format='png', dpi=360)
