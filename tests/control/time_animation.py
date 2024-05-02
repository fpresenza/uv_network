#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on mié 29 dic 2021 16:41:13 -03
@author: fran
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


class TimeAnimate(object):
    def __init__(self, fig, ax, timestep, time, data, styles=None, file=None):
        """ Animar vs t.

        data.shape = (n_steps, n_variables)
        styles es un dict.
        """
        self.fig = fig
        self.ax = ax
        self.h = timestep
        n_steps, n_var = data.shape
        self.time = time
        self.frames = np.arange(n_steps)
        self.data = data
        self.stamp = self.ax.text(
            0.01, 0.95, r'{:.2f} secs'.format(0.0),
            verticalalignment='top', horizontalalignment='left',
            transform=self.ax.transAxes, color='green', fontsize=10
        )
        for i in range(n_var):
            try:
                style = styles[i]
            except (TypeError, KeyError):
                style = {}
            self.ax.plot([], [], **style)

        self.animator = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=self.frames,
            interval=1000 * self.h,
            blit=True
        )
        if file is not None:
            self.animator.save(
                file,
                fps=1. / self.h,
                dpi=200,
                extra_args=['-vcodec', 'libx264']
            )

    def update(self, k):
        for i, line in enumerate(self.ax.lines):
            line.set_data(self.time[:k], self.data[:k, i])
        self.stamp.set_text('$t = {:.1f} s$'.format(self.time[k]))
        return self.ax.lines + self.ax.texts


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
u = u.reshape(n_steps, n, 2)
A = A.reshape(n_steps, n, n)


# ------------------------------------------------------------------
# Animation: rigidity eigenvalues
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 3))
fig.subplots_adjust(bottom=0.2)
ax.grid(1, lw=0.4)
ax.set_xlim(0, 250)
ax.set_ylim(0, 2.5)
ax.set_xlabel(r'$t [seg]$')
ax.set_ylabel('Autovalores de Rigidez')
timestep = np.diff(t).mean()
data = np.empty((n_steps, 3), dtype=float)
data[:, 0] = re.min(axis=1)
data[:, 1] = re.max(axis=1)
data[:, 2] = fre

transition = np.abs(t - 30.).argmin()
_t = np.hstack([t[:transition], t[transition::arg.vel]])
_data = np.vstack([data[:transition], data[transition::arg.vel]])

anim = TimeAnimate(
    fig, ax,
    timestep, _t, _data,
    styles={
        0: {'label': 'min'},
        1: {'label': 'max'},
        2: {'color': 'k', 'ls': '--', 'label': r'$\it{framework}$'}
    },
    file='/tmp/metrics.mp4'
)
ax.legend(ncol=3)


# ------------------------------------------------------------------
# Animation: localization eror
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 3))
fig.subplots_adjust(left=0.2, bottom=0.2)
ax.grid(1, lw=0.4)
ax.set_xlim(0, 250)
ax.set_ylim(0, 2.5)
ax.set_xlabel(r'$t [seg]$')
ax.set_ylabel(
    r'Error de posición \n $\sqrt{(1/n) \sum_i ||x_i - \hat{x}_i||^2 )}$'
)
timestep = np.diff(t).mean()
data = np.sqrt(np.square(x - hatx).sum(axis=-1).sum(axis=-1)/n)

transition = np.abs(t - 30.).argmin()
_t = np.hstack([t[:transition], t[transition::arg.vel]])
_data = np.hstack([data[:transition], data[transition::arg.vel]])
_data = _data.reshape(-1, 1)

anim = TimeAnimate(
    fig, ax,
    timestep, _t, _data,
    file='/tmp/error.mp4'
)

plt.show()
