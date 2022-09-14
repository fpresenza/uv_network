#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on mié 06 abr 2022 18:04:58 -03
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from uvnpy.network import subsets


plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


class TimeAnimate(object):
    def __init__(
            self, fig, ax, timestep, time, data, type='plot', styles=None):
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
            0.02, 0.1, r'{:.2f} secs'.format(0.0),
            verticalalignment='top', horizontalalignment='left',
            transform=self.ax.transAxes, color='green', fontsize=10)
        for i in range(n_var):
            try:
                style = styles[i]
            except (TypeError, KeyError):
                style = {}
            if type == 'plot':
                self.ax.plot([], [], **style)
            elif type == 'semilogy':
                self.ax.semilogy([], [], **style)

        self.animator = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=self.frames,
            interval=1000 * self.h,
            blit=True
        )

    def update(self, k):
        for i, line in enumerate(self.ax.lines):
            line.set_data(self.time[:k], self.data[:k, i])
        self.stamp.set_text('$t = {:.1f} s$'.format(self.time[k]))
        return self.ax.lines + self.ax.texts

    def save(self, file, dpi=200, extra_args=['-vcodec', 'libx264']):
        self.animator.save(
            file,
            fps=1. / self.h,
            dpi=dpi,
            extra_args=extra_args)


def adjust_speed(data, epochs, multipliers):
    if len(epochs) != len(multipliers):
        raise ValueError('len(epochs) != len(multipliers)')
    adj_data = np.hstack([
        data[b:e:m] for (b, e), m in zip(epochs, multipliers)])
    return adj_data


# extraigo datos
t = np.loadtxt('/tmp/t.csv', delimiter=',')
x = np.loadtxt('/tmp/x.csv', delimiter=',')
hatx = np.loadtxt('/tmp/hatx.csv', delimiter=',')
u = np.loadtxt('/tmp/u.csv', delimiter=',')
fre = np.loadtxt('/tmp/fre.csv', delimiter=',')
re = np.loadtxt('/tmp/re.csv', delimiter=',')
A = np.loadtxt('/tmp/adjacency.csv', delimiter=',')
hops = np.loadtxt('/tmp/hops.csv', delimiter=',')
n = int(len(x[0])/2)
nodes = np.arange(n)
hops = hops.astype(int)

# reshapes
x = x.reshape(len(t), n, 2)
hatx = hatx.reshape(len(t), n, 2)
u = u.reshape(len(t), n, 2)
A = A.reshape(len(t), n, n)

# slice
kf = np.argmin(np.abs(t - 200))
t = t[:kf]
x = x[:kf]
hatx = hatx[:kf]
u = u[:kf]
fre = fre[:kf]
re = re[:kf]
A = A[:kf]

# tiempo muerto al principio
dt = 200
t = np.hstack([t, t[-1] + t[1:dt+1]])
x = np.vstack([np.tile(x[0], (dt, 1, 1)), x])
hatx = np.vstack([np.tile(hatx[0], (dt, 1, 1)), hatx])
fre = np.hstack([np.tile(fre[0], dt), fre])
re = np.vstack([np.tile(re[0], (dt, 1)), re])
A = np.vstack([np.tile(A[0], (dt, 1, 1)), A])


# ajustar velocidad
a, b = (np.abs(t - 20.).argmin(), np.abs(t - 40.).argmin())
epochs = ((0, a), (a, b), (b, len(t)))
multipliers = (1, 2, 5)
adj_t = adjust_speed(t, epochs, multipliers)

adj_eig = np.empty((len(adj_t), 3), dtype=float)
adj_eig[:, 0] = adjust_speed(re.min(axis=1), epochs, multipliers)
adj_eig[:, 1] = adjust_speed(re.max(axis=1), epochs, multipliers)
adj_eig[:, 2] = adjust_speed(fre, epochs, multipliers)

load = np.array([subsets.degree_load_std(adj, hops) for adj in A])
edges = A.sum(-1).sum(-1)/2
load /= 2*edges[0]
adj_load = np.empty((len(adj_t), 1), dtype=float)
adj_load[:, 0] = adjust_speed(load, epochs, multipliers)


""" animacion eigenvalues """
fig, ax = plt.subplots(figsize=(6, 3))
fig.subplots_adjust(bottom=0.2)
ax.grid(1, lw=0.4)
ax.set_xlim(0, 225)
ax.set_ylim(1e-3, 2.5)
ax.set_xlabel(r'$t [seg]$')
ax.set_ylabel('Rigidity eigenvalues')
timestep = np.diff(t).mean()

anim = TimeAnimate(
    fig, ax,
    timestep, adj_t, adj_eig,
    type='semilogy',
    styles={
        0: {'label': r'min $\rho_i$'},
        1: {'label': r'max $\rho_i$'},
        2: {'color': 'k', 'ls': '--', 'label': 'framework'}})
ax.legend(
    fontsize='small',
    ncol=1,
    # columnspacing=0.2,
    loc='lower right')
anim.save('/tmp/eigenvalues.mp4')

""" animacion communication load """
fig, ax = plt.subplots(figsize=(6, 3))
fig.subplots_adjust(bottom=0.2)
ax.grid(1, lw=0.4)
ax.set_xlim(0, 225)
ax.set_ylim(1, 5.3)
ax.set_xlabel(r'$t [seg]$')
ax.set_ylabel('Std. Communication Load')
timestep = np.diff(t).mean()

anim = TimeAnimate(
    fig, ax,
    timestep, adj_t, adj_load,
    type='plot')
anim.save('/tmp/load.mp4')

plt.show()
