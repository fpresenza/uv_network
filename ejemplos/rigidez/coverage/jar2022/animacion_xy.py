#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on mié 29 dic 2021 16:41:13 -03
@author: fran
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

from uvnpy import network


plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


class CoverageAnimate(network.plot.Animate2):
    def __init__(self, *args, **kwargs):
        super(CoverageAnimate, self).__init__(*args, **kwargs)

    def _update_extra_artists(self, frame):
        P = frame[1]
        Y = frame[4]
        n = int(len(P)/2)
        for i in range(n):
            self._extra_artists[i].center = P[i]
        untracked = Y[:, 2].astype(bool)
        Yt = Y[~untracked]
        Yu = Y[untracked]
        self._extra_artists[-2].set_data(Yt[:, 0], Yt[:, 1])
        self._extra_artists[-1].set_data(Yu[:, 0], Yu[:, 1])


parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-x', '--vel',
    default=1, type=int, help='velocidad de reproduccion')
arg = parser.parse_args()


# extraigo datos
t = np.loadtxt('/tmp/t.csv', delimiter=',')
x = np.loadtxt('/tmp/position.csv', delimiter=',')
hatx = np.loadtxt('/tmp/est_position.csv', delimiter=',')
u = np.loadtxt('/tmp/action.csv', delimiter=',')
fre = np.loadtxt('/tmp/fre.csv', delimiter=',')
re = np.loadtxt('/tmp/re.csv', delimiter=',')
A = np.loadtxt('/tmp/adjacency.csv', delimiter=',')
extents = np.loadtxt('/tmp/extents.csv', delimiter=',')
targets = np.loadtxt('/tmp/targets.csv', delimiter=',')

n = int(len(x[0])/2)
nodes = np.arange(n)
extents = extents.astype(int)
n_steps = len(t)

# reshapes
x = x.reshape(n_steps, n, 2)
hatx = hatx.reshape(n_steps, n, 2)
# print(x[0], hatx[0])
u = u.reshape(n_steps, n, 2)
A = A.reshape(n_steps, n, n)
teams = np.empty((n_steps, 2*n), dtype=int)
teams[:, :n] = extents
teams[:, n:] = 0
targets = targets.reshape(n_steps, -1, 3)

""" ANIMACION POSICIONES """
lim = 50
timestep = np.diff(t).mean()
frames = np.empty((n_steps, 5), dtype=np.ndarray)
# E = network.edges_from_adjacency(A[0])
transition = np.abs(t - 30.).argmin()
steps = list(enumerate(t))
for k, tk in steps:
    E = network.edges_from_adjacency(A[k])
    X = np.vstack([x[k], hatx[k]])
    T = teams[k]
    Y = targets[k]
    frames[k] = tk, X, E, T, Y
fast_frames = np.vstack([frames[:transition], frames[transition::arg.vel]])


fig, ax = plt.subplots()
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='xx-small')
ax.set_aspect('equal')
ax.grid(1, lw=0.4)
ax.set_xlabel(r'$x$', fontsize='x-small', labelpad=0.6)
ax.set_ylabel(r'$y$', fontsize='x-small', labelpad=0.6)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)

anim = CoverageAnimate(fig, ax, timestep, fast_frames, maxlen=10)

anim.set_teams({
    '$1$-hop': {
        'id': 1,
        'tail': True,
        'style': {
            'color': 'royalblue',
            'marker': 'o',
            'markersize': 5}},
    '$2$-hop': {
        'id': 2,
        'tail': True,
        'style': {
            'color': 'chocolate',
            'marker': 'D',
            'markersize': 5}},
    '$3$-hop': {
        'id': 3,
        'tail': True,
        'style': {
            'color': 'mediumseagreen',
            'marker': 's',
            'markersize': 5}}})
anim.set_edgestyle(color='0.4', alpha=0.6, lw=0.8)

# circles = [plt.Circle(pi, 3., alpha=0.4) for pi in x[0]]
circles = []
for p in x[0]:
    circle = plt.Circle(p, 3., alpha=0.4)
    circles.append(circle)
    ax.add_artist(circle)
tracked = ax.plot([], [], ls='', marker='s', markersize=3, color='orange')
untracked = ax.plot(
    targets[0, :, 0], targets[0, :, 1],
    ls='', marker='s', markersize=3, color='0.6')
extras = circles + tracked + untracked
anim.set_extra_artists(*extras)

anim.ax.legend(
    ncol=3,
    loc='upper center',
    fontsize='small',
    handletextpad=1)
# anim.run()
anim.run('/tmp/rigidity.mp4')
plt.show()
