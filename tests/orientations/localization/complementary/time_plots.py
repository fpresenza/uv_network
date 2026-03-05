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

R = read_csv_numpy(
    'simu_data/orientation.csv'
).reshape(log_num_steps, 3, 3)

hatR = read_csv_numpy(
    'simu_data/estimated_orientation.csv'
).reshape(log_num_steps, 3, 3)

# ------------------------------------------------------------------
# Plot orientation error
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.0, 8.0))

E = np.matmul(R.swapaxes(1, 2), hatR)
e = 0.5 * (3 - np.trace(E, axis1=1, axis2=2))

ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='small'
)

ax.set_xlabel(r'$t\ (\mathrm{s})$', fontsize=8)
ax.set_ylabel(r'$error$', fontsize=8)
ax.set_ylim(0.0, 2.0)
ax.grid(1)

ax.plot(t, e, lw=0.8, ds='steps-post')
fig.savefig('time_plots/errors.pdf', bbox_inches='tight')


plt.show()
