#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'

fig, ax = plt.subplots(figsize=(3.5, 2.))
fig.subplots_adjust(left=0.2, bottom=0.32)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='small')

d = np.arange(1, 11, 1)
df = 6
a_d = np.zeros(d.size)
a_d[0] = 1
a_d[1:6] = np.random.uniform(0, 1, 5)
ax.bar(d, a_d, zorder=3, width=0.6)
ax.hlines(1, 0, 11, color='k', ls='--', lw=0.8)

ax.set_xlim(0, 11)
ax.set_xlabel(r'$d$', fontsize=10)
ax.set_xticks(d)
ax.set_xticklabels(d)
ax.set_ylabel(r'$a_d(\mathcal{G}) / a_1(\mathcal{G})$', fontsize=10)
ax.set_yticks([0, 0.5, 1])
ax.grid(zorder=0)

fig.savefig('/tmp/d_algebraic_connectivity.png', format='png', dpi=360)

fig, ax = plt.subplots(figsize=(3.5, 2.))
fig.subplots_adjust(left=0.2, bottom=0.32)
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='small')

d = np.arange(1, 11, 1)
df = 6
a_d = np.zeros(d.size)
a_d[0] = 1
for i in range(1, 6):
    a_d[i] = np.random.uniform(0, a_d[i-1])
ax.bar(d, a_d, zorder=3, width=0.6)
ax.hlines(1, 0, 11, color='k', ls='--', lw=0.8)

ax.set_xlim(0, 11)
ax.set_xlabel(r'$d$', fontsize=10)
ax.set_xticks(d)
ax.set_xticklabels(d)
ax.set_ylabel(r'$a_d(\mathcal{G}) / a_1(\mathcal{G})$', fontsize=10)
ax.set_yticks([0, 0.5, 1])
ax.grid(zorder=0)
fig.savefig('/tmp/d_algebraic_connectivity_conj.png', format='png', dpi=360)
