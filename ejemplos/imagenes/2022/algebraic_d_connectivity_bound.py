#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'

plt.rcParams["legend.borderpad"] = 0.1
plt.rcParams["legend.labelspacing"] = 0.3
plt.rcParams["legend.handlelength"] = 1.0
plt.rcParams["legend.columnspacing"] = 1.0


def new_bound(d, n):
    r = d*(d+1)/2 + 1
    return n*(n-2)/(d*n - r)


def lew_bound(d, n):
    return 2*n/(3*(d-1)) + 1/3


n_agents = np.arange(2, 100)
dim = np.arange(3, 4)

fig, ax = plt.subplots(figsize=(4, 2))
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='xx-small')
ax.set_xlabel('$n$')
ax.grid(1, lw=0.3)

for i, d in enumerate(dim):
    n = n_agents[n_agents > d]
    ax.plot(n, new_bound(d, n), ls='--', color='C{}'.format(i))
    ax.plot(
        n, lew_bound(d, n), color='C{}'.format(i), label='$d={}$'.format(d))

ax.plot(n, n, ls='', marker='o', color='k')


ax.legend(
    fontsize='x-small',
    handletextpad=0.1,
    borderpad=0.2)

plt.show()
