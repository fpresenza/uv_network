#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from uvnpy.toolkit.data import read_csv_numpy

plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'

# ------------------------------------------------------------------
# Read simulated data
# ------------------------------------------------------------------
t = read_csv_numpy('simu_data/t.csv')
log_num_steps = len(t)

orientation = read_csv_numpy(
    'simu_data/orientation.csv'
).reshape(log_num_steps, -1, 3, 3)
n = len(orientation[0])

estimated_orientation = read_csv_numpy(
    'simu_data/estimated_orientation.csv'
).reshape(log_num_steps, n, 3, 3)

a = 0     # leader

# ------------------------------------------------------------------
# Plot orientation error
# ------------------------------------------------------------------
Q = np.matmul(orientation[:, a, np.newaxis].swapaxes(2, 3), orientation)
hatQ = estimated_orientation

fig, ax = plt.subplots(figsize=(9.0, 6.0))
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
    labelsize=9
)

ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=10)
ax.set_ylabel(
    r'$\|\hat{Q}_i - Q_i \|_F$',
    fontsize=10
)
ax.grid(1)

ax.plot(
    t,
    np.linalg.norm(hatQ - Q, axis=(2, 3)),
    lw=1.0,
    ds='steps-post'
)

fig.savefig('time_plots/orientation_error.pdf', bbox_inches='tight')

plt.show()
