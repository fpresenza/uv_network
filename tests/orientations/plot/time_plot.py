#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import matplotlib.pyplot as plt

from uvnpy.toolkit import data

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'

# ------------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '-i', '--init',
    default=0.0, type=float, help='init time in milli seconds'
)
parser.add_argument(
    '-e', '--end',
    default=0.0, type=float, help='end time in milli seconds'
)
parser.add_argument(
    '-j', '--jump',
    default=1, type=int, help='numbers of frames jumped'
)
arg = parser.parse_args()

# ------------------------------------------------------------------
# Read simulated data
# ------------------------------------------------------------------
t = np.loadtxt('data/t.csv', delimiter=',')
arg.end = t[-1] if (arg.end == 0) else arg.end

# slices
k_i = int(np.argmin(np.abs(t - arg.init)))
k_e = int(np.argmin(np.abs(t - arg.end))) + 1

t = t[k_i:k_e:arg.jump]

frames = data.read_csv(
    'data/frames.csv',
    rows=(k_i, k_e),
    jump=arg.jump,
    dtype=float,
    shape=(-1, 3, 3),
    asarray=True
)

est_frames = data.read_csv(
    'data/est_frames.csv',
    rows=(k_i, k_e),
    jump=arg.jump,
    dtype=float,
    shape=(-1, 3, 3),
    asarray=True
)

n = len(frames[0])
print(est_frames[-1])

# ------------------------------------------------------------------
# Plot orientation error
# ------------------------------------------------------------------
fig, ax = plt.subplots(n, n, figsize=(8.0, 8.0))
# fig.subplots_adjust(
#     bottom=0.215, top=0.925, wspace=0.33, right=0.975, left=0.18
# )
fig.tight_layout()

for i in range(n):
    for j in range(n):
        Ri = frames[:, i]
        Rj = frames[:, j]
        Ri_hat = est_frames[:, i]
        Rj_hat = est_frames[:, j]

        Rij = np.matmul(Ri.swapaxes(1, 2), Rj)
        Rij_hat = np.matmul(Ri_hat.swapaxes(1, 2), Rj_hat)

        eij = np.sum(np.sum((Rij - Rij_hat)**2, axis=1), axis=1)

        ax[i, j].tick_params(
            axis='both',       # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            pad=1,
            labelsize='small')

        ax[i, j].set_xlabel(r'$t\ (\mathrm{s})$', fontsize=8)
        ax[i, j].set_ylabel(rf'$e_{{{i}{j}}}$', fontsize=8)
        ax[i, j].grid(1)

        ax[i, j].plot(
            t,
            eij,
            lw=0.8,
            ds='steps-post'
        )
fig.savefig('time_plot/errors.pdf', bbox_inches='tight')


plt.show()
