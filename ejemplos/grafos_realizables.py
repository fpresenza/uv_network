#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Created on lun jul  5 20:19:35 -03 2021
@author: fran
"""
import numpy as np
import matplotlib.pyplot as plt

import uvnpy.network as network
from uvnpy.network import disk_graph
from uvnpy.toolkit.calculus import circle2d

fig, ax = network.plot.figure(1, 2, figsize=(6, 3))

dmax = 1.5
p = circle2d(R=1, N=8)
i = 0
Ni = disk_graph.neighborhood(p, 0, dmax, inclusive=True)
pi = p[Ni]
qi = p[~Ni]

Ep = disk_graph.edges(pi, dmax)
Eq = disk_graph.edges(qi, dmax)
Epq = disk_graph.inter_edges(pi, qi, dmax)
Epq[:, 1] += len(pi)

# grado rigido individualmente rigido
network.plot.nodes(ax[0], qi, color='b', alpha=0.6)
network.plot.edges(ax[0], qi, Eq, color='k', alpha=0.6, lw=0.8)
network.plot.edges(
    ax[0], np.vstack([pi, qi]), Epq, color='k', alpha=0.6, lw=0.8)
network.plot.nodes(ax[0], pi, color='g')
network.plot.edges(ax[0], pi, Ep, color='g')
ax[0].text(p[i, 0] - 0.2, p[i, 1] - 0.05, r'$i$', color='g')
ax[0].text(
    p[i, 0] - 0.15, p[i, 1] + 0.5, r'$\mathcal{G}_i$', color='g', fontsize=15)

# grafo rigido no dindividualmente rigido
Ep = np.delete(Ep, 5, 0)
network.plot.nodes(ax[1], qi, color='b', alpha=0.6)
network.plot.edges(ax[1], qi, Eq, color='k', alpha=0.6, lw=0.8)
network.plot.edges(
    ax[1], np.vstack([pi, qi]), Epq, color='k', alpha=0.6, lw=0.8)
network.plot.nodes(ax[1], pi, color='g')
network.plot.edges(ax[1], pi, Ep, color='g')
ax[1].text(p[i, 0] - 0.2, p[i, 1] - 0.05, r'$i$', color='g')
ax[1].text(
    p[i, 0] - 0.15, p[i, 1] + 0.5, r'$\mathcal{G}_i$', color='g', fontsize=15)

plt.show()
