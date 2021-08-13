#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on jue jul 29 17:38:16 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt

from uvnpy.network.connectivity import logistic

fig, ax = plt.subplots(figsize=(3.5, 1.25))
fig.subplots_adjust(left=0.14, bottom=0.32)
ax.grid(1)
ax.minorticks_on()

d = np.linspace(0, 2, 200)
w = logistic(d, beta=40, e=0.8)

ax.plot(d, w)
ax.vlines(1, 0, 1, color='k', alpha=0.6, ls='--', lw=0.5)
ax.axvspan(0, 1, color='green', alpha=0.2)
ax.axvspan(1, 2, color='red', alpha=0.2)
ax.set_xlim(0, 2)
ax.set_xlabel(r'$d_{{ij}} / d_{{\max}}$', fontsize=7)
ax.set_ylabel(r'$w_{{ij}}$', fontsize=7)
ax.tick_params(axis='both', which='major', labelsize=6)
plt.show()

fig.savefig('/tmp/edge_weight.pdf', format='pdf')
