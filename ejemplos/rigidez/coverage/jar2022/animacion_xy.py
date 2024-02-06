#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on mié 29 dic 2021 16:41:13 -03
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt

from uvnpy import network


plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


def time_index(t, ts):
    return np.argmin(np.abs(t - ts))


def slow_motion(data):
    shape = list(data.shape)
    shape[0] *= 2
    sdata = np.empty(shape, dtype=np.ndarray)
    sdata[::2] = data
    sdata[1::2] = data
    return sdata


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


# extraigo datos
t = np.loadtxt('/tmp/t.csv', delimiter=',')[200:]
t -= t[0]
x = np.loadtxt('/tmp/position.csv', delimiter=',')[200:]
hatx = np.loadtxt('/tmp/est_position.csv', delimiter=',')[200:]
A = np.loadtxt('/tmp/adjacency.csv', delimiter=',')[200:]
extents = np.loadtxt('/tmp/extents.csv', delimiter=',')[200:]
targets = np.loadtxt('/tmp/targets.csv', delimiter=',')[200:]

# slice
print(t.max())
tf = 205
kf = time_index(t, tf)
t = t[:kf]
x = x[:kf]
hatx = hatx[:kf]
A = A[:kf]
extents = extents[:kf]
targets = targets[:kf]

n = int(len(x[0])/2)
nodes = np.arange(n)
extents = extents.astype(int)
n_steps = len(t)

# reshapes
x = x.reshape(n_steps, n, 2)
hatx = hatx.reshape(n_steps, n, 2)
# print(x[0], hatx[0])
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
steps = list(enumerate(t))
for k, tk in steps:
    E = network.edges_from_adjacency(A[k])
    X = np.vstack([x[k], hatx[k]])
    T = teams[k]
    Y = targets[k]
    frames[k] = tk, X, E, T, Y

a, b, c = time_index(t, 15.), time_index(t, 30), time_index(t, 100)
adjusted_frames = np.vstack([
    frames[0:a:1],
    frames[a:b:2],
    frames[b:c:5],
    frames[c::5]])


fig, ax = plt.subplots()
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='xx-small')
ax.set_aspect('equal')
ax.grid(1, lw=0.4)
ax.set_xlabel(r'$x$ [m]', fontsize='small', labelpad=0.6)
ax.set_ylabel(r'$y$ [m]', fontsize='small', labelpad=0.6)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
# ax.set_title(r'Retardo $\tau = 400$ [ms]')
ax.set_title('Sin Retardo')

anim = CoverageAnimate(fig, ax, timestep, adjusted_frames, maxlen=10)

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
anim.run('/tmp/xy.mp4')
# plt.show()
