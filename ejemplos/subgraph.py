#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on dom abr 11 18:34:34 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt

import uvnpy.network.core as core
import uvnpy.network.disk_graph as disk_graph


N = 16
L = 100
dmax = 65

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.grid(1)
ax.set_xlim(-L, L)
ax.set_ylim(-L, L)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(r'Subgrafo $i$')

np.random.seed(13)
p = np.random.uniform(-0.8 * L, 0.8 * L, (N, 2))
E = disk_graph.edges(p, dmax)
# print(p)
# print(E)

q = disk_graph.local_neighbors(p[8], np.delete(p, 8, axis=0), dmax)
pl = np.vstack([p[8], q])
El = disk_graph.edges(pl, dmax)
# print(pl)
# print(El)

core.plot_graph(ax, p, E, edgestyle={'color': '0.2', 'lw': 0.8})
core.plot_graph(
    ax, pl, El,
    nodestyle={'color': 'g'},
    edgestyle={'color': 'g'})
ax.text(-31, -26, r'nodo-$i$', color='g')
ax.text(-56, -11, r'$\mathcal{G}_i$', color='g', fontsize=15)

# plt.show()
plt.savefig('/tmp/subgrafo.png')
