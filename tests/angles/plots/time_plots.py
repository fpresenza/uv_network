#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from uvnpy.toolkit.data import read_csv_numpy
from uvnpy.angles.local_frame.core import angle_function
from uvnpy.graphs.core import edges_from_adjacency

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'


# ------------------------------------------------------------------
# Read simulated data
# ------------------------------------------------------------------
t = read_csv_numpy('data/t.csv')
simu_step_num = len(t)
position = read_csv_numpy('data/position.csv').reshape(simu_step_num, -1, 3)

n = len(position[0])

desired_angles = read_csv_numpy('data/desired_angles.csv')
adjacency = read_csv_numpy('data/adjacency.csv')
edge_set = edges_from_adjacency(adjacency.reshape(n, n))

# ------------------------------------------------------------------
# Plot positions
# ------------------------------------------------------------------
fig, ax = plt.subplots(3, 1, figsize=(6.0, 4.0))
fig.subplots_adjust(
    bottom=0.215,
    top=0.925,
    wspace=0.33,
    right=0.975,
    left=0.18
)

for k, d in enumerate(['x', 'y', 'z']):
    ax[k].tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        pad=1,
        labelsize='8'
    )

    ax[k].set_xlabel(r'$t\ (\mathrm{s})$', fontsize='9')
    ax[k].set_ylabel(fr'$p_{{i, {d}}} \ (\rm m)$', fontsize='9')
    ax[k].set_ylim(-1.5, 1.5)
    ax[k].grid(1)

    ax[k].plot(t, position[:, :, k], lw=0.8, ds='steps-post')

fig.savefig('time_plots/position.pdf', bbox_inches='tight')

# ------------------------------------------------------------------
# Plot angles
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 4.0))
fig.subplots_adjust(
    bottom=0.215,
    top=0.925,
    wspace=0.33,
    right=0.975,
    left=0.18
)

ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='8'
)

ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize='9')
ax.set_ylabel(r'angles cosines', fontsize='9')
ax.set_ylim(-1.0, 1.0)
ax.grid(1)

ax.plot(
    t,
    [angle_function(edge_set, p) for p in position],
    lw=0.8, ds='steps-post'
)
ax.set_prop_cycle(None)    # resets color counter
ax.plot(t, [desired_angles for _ in t], lw=0.8, ds='steps-post', ls='--')

fig.savefig('time_plots/angles.pdf', bbox_inches='tight')
plt.show()
