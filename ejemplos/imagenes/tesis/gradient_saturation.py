#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from uvnpy.toolkit.functions import logistic_saturation


plt.rcParams['text.usetex'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'

fig, ax = plt.subplots(figsize=(3.5, 2.))
fig.subplots_adjust(left=0.2, bottom=0.32)
ax.grid(1)
ax.minorticks_on()
ax.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    pad=1,
    labelsize='small')

x = np.linspace(-5, 5, 200)
w = logistic_saturation(x, slope=1)
ax.plot(x, w, label=r'$\gamma = 1$')
w = logistic_saturation(x, slope=2)
ax.plot(x, w, label=r'$\gamma=2$')

ax.set_xlim(-5, 5)
ax.set_xlabel(r'$x$', fontsize=10)
ax.set_ylabel(r'$\mu(x)$', fontsize=10)
ax.set_yticks([-1, 0, 1])
ax.set_yticklabels([r'$-v_{\max}$', 0.5, r'$v_{\max}$'])
ax.legend(
    fontsize='small', handlelength=1.5,
    labelspacing=0.5, borderpad=0.2,
    loc='lower right')
# plt.show()

fig.savefig('/tmp/gradient_saturation.png', format='png', dpi=360)
