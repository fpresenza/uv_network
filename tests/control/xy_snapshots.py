#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on mié 29 dic 2021 16:41:13 -03
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import progressbar

from uvnpy.network import plot

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


# ------------------------------------------------------------------
# Read simulated data
# ------------------------------------------------------------------
t = np.loadtxt('data/t.csv', delimiter=',')
tc = np.loadtxt('data/tc.csv', delimiter=',')
x = np.loadtxt('data/position.csv', delimiter=',')
hatx = np.loadtxt('data/est_position.csv', delimiter=',')
cov = np.loadtxt('data/covariance.csv', delimiter=',')
u = np.loadtxt('data/action.csv', delimiter=',')
v = np.loadtxt('data/vel_meas_err.csv', delimiter=',')
g = np.loadtxt('data/gps_meas_err.csv', delimiter=',')
r = np.loadtxt('data/range_meas_err.csv', delimiter=',')
fre = np.loadtxt('data/fre.csv', delimiter=',')
re = np.loadtxt('data/re.csv', delimiter=',')
A = np.loadtxt('data/adjacency.csv', delimiter=',')
state_extents = np.loadtxt('data/state_extents.csv', delimiter=',')
targets = np.loadtxt('data/targets.csv', delimiter=',')

n = int(len(x[0])/2)
nodes = np.arange(n)
state_extents = state_extents.astype(int)

# reshapes
x = x.reshape(len(t), n, 2)
hatx = hatx.reshape(len(t), n, 2)
cov = cov.reshape(len(t), n, 2)
u = u.reshape(len(t), n, 2)
v = v.reshape(len(t), n, 2)
g = g.reshape(len(t), n, 2)
re = re.reshape(len(t), -1)
A = A.reshape(len(t), n, n)
targets = targets.reshape(len(t), -1, 3)

# ------------------------------------------------------------------
# Plot snapshots
# ------------------------------------------------------------------
lim = 1000
bar = progressbar.ProgressBar(maxval=tc[-1]).start()

for tk in tc:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize='x-small')
    ax.grid(1, lw=0.4)
    ax.set_aspect('equal')
    # ax.set_xlabel(r'$\mathrm{x}$', fontsize='x-small', labelpad=0.6)
    # ax.set_ylabel(r'$\mathrm{y}$', fontsize='x-small', labelpad=0)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    k = np.argmin(np.abs(t - tk))
    ax.text(
            0.05, 0.01, r't = {:.2f}s'.format(tk),
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes, color='r', fontsize=8)

    for i in nodes:
        plot.nodes(
            ax, x[k, i],
            color='k',
            marker=f'${i}$',
            s=20,
            lw=0.2
        )
        circle = plt.Circle(x[k, i], 30.0, alpha=0.3)
        ax.add_artist(circle)
    plot.edges(ax, x[k], A[k], color=cm.coolwarm(20), lw=0.5, zorder=0)

    untracked = targets[k, :, 2].astype(bool)
    tracked = np.logical_not(untracked)
    ax.scatter(
        targets[k, untracked, 0], targets[k, untracked, 1],
        marker='s', s=4, color='0.6')
    ax.scatter(
        targets[k, tracked, 0], targets[k, tracked, 1],
        marker='s', s=4, color='green')

    fig.savefig('data/snapshots/{}.png'.format(k), format='png', dpi=360)
    plt.close()

bar.finish()
