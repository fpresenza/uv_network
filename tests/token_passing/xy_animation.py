#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on mié 29 dic 2021 16:41:13 -03
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from uvnpy.network.core import edges_from_adjacency
from uvnpy.network.plot import Animate2


plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


def time_index(t, ts):
    return np.argmin(np.abs(t - ts))


class CoverageAnimate(Animate2):
    def __init__(self, *args, **kwargs):
        super(CoverageAnimate, self).__init__(*args, **kwargs)

    def _update_extra_artists(self, frame):
        P = frame[1]
        n = int(len(P))
        for i in range(2*n):
            self._extra_artists[i].center = P[i % n]


# ------------------------------------------------------------------
# Read simulated data
# ------------------------------------------------------------------
t = np.loadtxt('/tmp/timestamps.csv', delimiter=',')
t -= t[0]
x = np.loadtxt('/tmp/position.csv', delimiter=',')
A = np.loadtxt('/tmp/adjacency.csv', delimiter=',')
extents = np.loadtxt('/tmp/extents.csv', delimiter=',')

# preserve only first 'tf' seconds
tf = 205
kf = time_index(t, tf)
t = t[:kf]
x = x[:kf]
A = A[:kf]
extents = extents[:kf]

n = int(len(x[0])/2)
nodes = np.arange(n)
extents = extents.astype(int)
n_steps = len(t)

# reshapes
x = x.reshape(n_steps, n, 2)
A = A.reshape(n_steps, n, n)
teams = np.empty((n_steps, n), dtype=int)
teams[:, :n] = extents

# ------------------------------------------------------------------
# Create Frames
# ------------------------------------------------------------------
lim = 10
timestep = np.diff(t).mean()
frames = np.empty((n_steps, 4), dtype=np.ndarray)
steps = list(enumerate(t))
for k, tk in steps:
    E = edges_from_adjacency(A[k])
    X = x[k]
    T = teams[k]
    frames[k] = tk, X, E, T

a, b, c = time_index(t, 15.), time_index(t, 30), time_index(t, 100)
adjusted_frames = np.vstack([
    frames[0:a:1],
    frames[a:b:2],
    frames[b:c:5],
    frames[c::5]
])

# ------------------------------------------------------------------
# Animation
# ------------------------------------------------------------------
fig, ax = plt.subplots()
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='medium'
)
ax.set_aspect('equal')
ax.grid(1, lw=0.4)
ax.set_xlabel(r'$x$ [m]', fontsize='medium', labelpad=0.6)
ax.set_ylabel(r'$y$ [m]', fontsize='medium', labelpad=0.6)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
# ax.set_title(r'Retardo $\tau = 400$ [ms]')
# ax.set_title('Sin Retardo')

anim = CoverageAnimate(fig, ax, timestep, adjusted_frames, maxlen=1)
# cm.coolwarm goes from 0 (blue) to 255 (red)
anim.set_teams({
    '$1$-hop': {
        'id': 1,
        'tail': False,
        'style': {
            # 'color': 'royalblue',
            'color': cm.coolwarm(255),
            'marker': 'o',
            'markersize': 10
        }
    }
})
anim.set_edgestyle(color='k', lw=0.75, zorder=0)

circles = []
for p in x[0]:
    circle = plt.Circle(p, 4., alpha=0.2)
    circles.append(circle)
    ax.add_artist(circle)

for p in x[0]:
    circle = plt.Circle(p, 5., alpha=0.2)
    circles.append(circle)
    ax.add_artist(circle)

extras = circles
anim.set_extra_artists(*extras)


# anim.ax.legend(
#     ncol=3,
#     loc='upper center',
#     fontsize='small',
#     handletextpad=1
# )
# anim.run()
anim.run('./figs/xy.mp4')
# plt.show()
