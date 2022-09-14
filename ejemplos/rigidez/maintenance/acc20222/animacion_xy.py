#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on mié 06 abr 2022 10:13:39 -03
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

# extraigo datos
t = np.loadtxt('/tmp/t.csv', delimiter=',')
x = np.loadtxt('/tmp/x.csv', delimiter=',')
hatx = np.loadtxt('/tmp/hatx.csv', delimiter=',')
A = np.loadtxt('/tmp/adjacency.csv', delimiter=',')
hops = np.loadtxt('/tmp/hops.csv', delimiter=',')
n = int(len(x[0])/2)
nodes = np.arange(n)
hops = hops.astype(int)

# reshapes
x = x.reshape(len(t), n, 2)
hatx = hatx.reshape(len(t), n, 2)
# print(x[0], hatx[0])
A = A.reshape(len(t), n, n)

lim = np.abs(x).max()

# slice
kf = np.argmin(np.abs(t - 200))
t = t[:kf]
x = x[:kf]
hatx = hatx[:kf]
A = A[:kf]

# tiempo muerto al principio
dt = 200
t = np.hstack([t, t[-1] + t[1:dt+1]])
x = np.vstack([np.tile(x[0], (dt, 1, 1)), x])
hatx = np.vstack([np.tile(hatx[0], (dt, 1, 1)), hatx])
A = np.vstack([np.tile(A[0], (dt, 1, 1)), A])

# animacion
timestep = np.diff(t).mean()
frames = np.empty((t.size, 3), dtype=np.ndarray)
E = network.edges_from_adjacency(A[0])
steps = list(enumerate(t))
for k, tk in steps:
    E = network.edges_from_adjacency(A[k])
    X = np.vstack([x[k], hatx[k]])
    frames[k] = tk, X, E


# ajustar velocidad
transition = (np.abs(t - 20.).argmin(), np.abs(t - 40.).argmin())
adjusted_frames = np.vstack([
    frames[:transition[0]],
    frames[transition[0]:transition[1]:2],
    frames[transition[1]::5]])

fig, ax = plt.subplots()
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='xx-small')
ax.set_aspect('equal')
ax.grid(1, lw=0.4)
ax.set_xlabel(r'$x \; [m]$', fontsize='small', labelpad=0.6)
ax.set_ylabel(r'$y \; [m]$', fontsize='small', labelpad=0.6)
ax.set_xticks([-100, 0, 100])
ax.set_yticks([-100, 0, 100])
ax.set_xticklabels([-100, 0, 100])
ax.set_yticklabels([-100, 0, 100])
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
anim = network.plot.Animate(fig, ax, timestep, adjusted_frames, maxlen=50)
one_hop_rigid = hops == 1
two_hop_rigid = hops == 2
three_hop_rigid = hops == 3
four_hop_rigid = hops == 4
anim.set_teams(
    {
        'name': '$1$-hop',
        'ids': nodes[one_hop_rigid],
        'tail': True,
        'style': {'color': 'royalblue', 'marker': 'o', 'markersize': 5}},
    {
        'name': '$2$-hop',
        'ids': nodes[two_hop_rigid],
        'tail': True,
        'style': {'color': 'chocolate', 'marker': 'D', 'markersize': 5}},
    {
        'name': '$3$-hop',
        'ids': nodes[three_hop_rigid],
        'tail': True,
        'style': {'color': 'mediumseagreen', 'marker': 's', 'markersize': 5}},
    {
        'name': '$4$-hop',
        'ids': nodes[four_hop_rigid],
        'tail': True,
        'style': {'color': 'purple', 'marker': '^', 'markersize': 5}},
    {
        'name': 'est. pos.',
        'ids': nodes + nodes[-1] + 1,
        'tail': False,
        'style': {'color': 'red', 'marker': '+', 'markersize': 5}})
anim.set_edgestyle(color='0.4', alpha=0.6, lw=0.8)
anim.ax.legend(
    fontsize='small',
    handletextpad=0.0,
    borderpad=0.2,
    ncol=5, columnspacing=0.2,
    loc='upper center')
# anim.run()
anim.run('/tmp/xy.mp4')

# plt.show()
