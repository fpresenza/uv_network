#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on mié 29 dic 2021 16:41:13 -03
@author: fran
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

from uvnpy.network.core import edges_from_adjacency
from uvnpy.network.plot import Animate2


plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


def termination_time_index(t, ts):
    if ts == np.inf:
        return len(t) - 1
    else:
        return np.argmin(np.abs(t - ts))


def slow_motion(data):
    shape = list(data.shape)
    shape[0] *= 2
    sdata = np.empty(shape, dtype=np.ndarray)
    sdata[::2] = data
    sdata[1::2] = data
    return sdata


class CoverageAnimate(Animate2):
    def __init__(self, *args, **kwargs):
        super(CoverageAnimate, self).__init__(*args, **kwargs)

    def set_xlim(self, t):
        return (0.0, 500.0 + t)

    def set_ylim(self, t):
        return (0.0, 500.0 + t)

    def _update_extra_artists(self, frame):
        for i, target in enumerate(frame[4]):
            if (target[2] == 0):
                try:
                    self._extra_artists[i].remove()
                except ValueError:
                    pass


# ------------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-s', '--skip',
    default=1, type=int, help='numbers of frames skipped during animation'
)
arg = parser.parse_args()


# ------------------------------------------------------------------
# Read simulated data
# ------------------------------------------------------------------
t = np.loadtxt('data/t.csv', delimiter=',')
t -= t[0]
x = np.loadtxt('data/position.csv', delimiter=',')
hatx = np.loadtxt('data/est_position.csv', delimiter=',')
A = np.loadtxt('data/adjacency.csv', delimiter=',')
action_extents = np.loadtxt('data/action_extents.csv', delimiter=',')
targets = np.loadtxt('data/targets.csv', delimiter=',')

# preserve only first 'tf' seconds
tf = np.inf
kf = termination_time_index(t, tf)
t = t[:kf]
x = x[:kf]
hatx = hatx[:kf]
A = A[:kf]
action_extents = action_extents[:kf]
targets = targets[:kf]

n = int(len(x[0])/2)
nodes = np.arange(n)
action_extents = action_extents.astype(int)
n_steps = len(t)

# reshapes
x = x.reshape(n_steps, n, 2)
hatx = hatx.reshape(n_steps, n, 2)
A = A.reshape(n_steps, n, n)
teams = np.empty((n_steps, 2*n), dtype=int)
teams[:, :n] = np.minimum(1, action_extents)
teams[:, n:] = -1 - np.arange(n)
targets = targets.reshape(n_steps, -1, 3)

# ------------------------------------------------------------------
# Create Frames
# ------------------------------------------------------------------
lim = 4000.0
timestep = np.diff(t).mean()
frames = np.empty((n_steps, 5), dtype=np.ndarray)
steps = list(enumerate(t))
for k, tk in steps:
    E = edges_from_adjacency(A[k])
    X = np.vstack([x[k], x[k]])
    T = teams[k]
    Y = targets[k]
    frames[k] = tk, X, E, T, Y

# a = termination_time_index(t, 15.0)
# b = termination_time_index(t, 30.0)
# c = termination_time_index(t, 100.0)
# adjusted_frames = np.vstack([
#     frames[0:a:1],
#     frames[a:b:2],
#     frames[b:c:5],
#     frames[c::5]
# ])
adjusted_frames = frames[::arg.skip]

# ------------------------------------------------------------------
# Animation
# ------------------------------------------------------------------
fig, ax = plt.subplots()
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='x-small'
)
ax.set_aspect('equal')
# ax.grid(1, lw=0.4)
ax.set_xlabel(r'$x$ [m]', fontsize='small', labelpad=0.6)
ax.set_ylabel(r'$y$ [m]', fontsize='small', labelpad=0.6)
ax.set_xlim(0.0, 500.0)
ax.set_ylim(0.0, 500.0)
# ax.set_title(r'Retardo $\tau = 400$ [ms]')
# ax.set_title('Sin Retardo')

anim = CoverageAnimate(fig, ax, timestep, adjusted_frames, maxlen=1)
# cm.coolwarm goes from 0 (blue) to 255 (red)
teams_dict = {
    'centers': {
        'id': 1,
        'tail': False,
        'style': {
            'color': 'goldenrod',
            'marker': '*',
            'markersize': 12
        }
    },
    'noncenters': {
        'id': 0,
        'tail': False,
        'style': {
            'color': 'C0',
            'marker': 'o',
            'markersize': 7
        }
    }
}
# for i in range(n):
#     teams_dict[i] = {
#         'id': -1-i,
#         'style': {
#             'color': 'k',
#             'marker': f'${i}$',
#             'markeredgewidth': 0.3
#         }
#     }
anim.set_teams(teams_dict)
anim.set_edgestyle(color='k', lw=0.5, zorder=0)

for p in targets[0]:
    circle = plt.Circle(p, 30.0, color='r', alpha=0.3)
    ax.add_artist(circle)
    anim.add_extra_artists(circle)

# anim.ax.legend(
#     ncol=3,
#     loc='upper center',
#     fontsize='small',
#     handletextpad=1)
# anim.run()
anim.run('data/xy.mp4')
# plt.show()
