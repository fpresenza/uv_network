#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on jue jul 29 17:38:16 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt

from uvnpy.toolkit.functions import logistic


plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'

fig, ax = plt.subplots(figsize=(3.5, 2.))
fig.subplots_adjust(left=0.14, bottom=0.32)
ax.grid(1)
ax.minorticks_on()
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='small')

d = np.linspace(0, 2, 200)
print(d[::50])
w = logistic(d, steepness=20, midpoint=1)
ax.plot(d, w, label=r'$\beta = 20 / \rho$')
w = logistic(d, steepness=40, midpoint=1)
ax.plot(d, w, label=r'$\beta = 40 / \rho$')

ax.vlines(1, 0, 1, color='k', alpha=0.6, ls='--', lw=0.5)
ax.axvspan(0, 1, color='green', alpha=0.2)
ax.axvspan(1, 2, color='red', alpha=0.2)
ax.set_xlim(0, 2)
ax.set_xlabel(r'$\Vert p_i - p_j \Vert \; / \; \rho$', fontsize=10)
ax.set_ylabel(r'$w(i, j)}$', fontsize=10)
ax.set_xticklabels([0, 0.5, 1, 1.5, 2])
ax.set_xticks([0, 0.5, 1, 1.5, 2])
ax.set_yticklabels([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])
ax.legend(
    fontsize='small', handlelength=1.5,
    labelspacing=0.5, borderpad=0.2)
# plt.show()

fig.savefig('/tmp/edge_weight.png', format='png', dpi=360)
