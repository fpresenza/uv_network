#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar dic 15 10:53:03 -03 2020
"""
import numpy as np
import matplotlib.pyplot as plt

import uvnpy.redes.core as redes
import uvnpy.rsn.core as rsn
from uvnpy.filtering import metricas


def completar(x, size):
    t = size - len(x)
    return np.pad(x, pad_width=(0, t), mode='constant')


jacobiano = rsn.distancia_relativa_jac
incidence_from_edges = redes.incidence_from_edges
conectar = redes.edges_from_positions
svdvals = metricas.svdvals

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('$F(H)$ vs. $(x_4, y_4)$ (Topología Variable)', fontsize=15)
fig.subplots_adjust(hspace=0.5)
cw = plt.cm.get_cmap('coolwarm')

t = np.linspace(-10, 10, 100)
N = range(100)
X, Y = np.meshgrid(t, t)
norma2 = np.empty_like(X)
nuclear = np.empty_like(X)
prod = np.empty_like(X)
cond = np.empty_like(X)
var = np.empty_like(X)

p = np.array([[-5, 0],
              [0, -5],
              [5., 0],
              [0,  0]])
E = np.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0]])
V = [0, 1, 2, 3]
D = incidence_from_edges(V, E)

for i in N:
    for j in N:
        p[3] = X[i, j], Y[i, j]
        E = conectar(p, 8.)
        D = incidence_from_edges(V, E)
        H = jacobiano(p, D)
        sv = svdvals(H)
        norma2[i, j] = sv[0]
        nuclear[i, j] = sv.sum()
        prod[i, j] = sv.prod()
        cond[i, j] = sv[0] / sv[-1]

cbar = axes[0, 0].contourf(X, Y, norma2, levels=20, cmap=cw)
fig.colorbar(cbar, ax=axes[0, 0])
axes[0, 0].set_title('norma-2')

cbar = axes[0, 1].contourf(X, Y, cond, levels=20, cmap=cw)
fig.colorbar(cbar, ax=axes[0, 1])
axes[0, 1].set_title(r'$\kappa(H)$')

cbar = axes[1, 0].contourf(X, Y, nuclear, levels=20, cmap=cw)
fig.colorbar(cbar, ax=axes[1, 0])
axes[1, 0].set_title('norma-nuc')

cbar = axes[1, 1].contourf(X, Y, prod, levels=20, cmap=cw)
fig.colorbar(cbar, ax=axes[1, 1])
axes[1, 1].set_title(r'$\pi(H)$')

for ax in axes.flat:
    ax.scatter([-5, 0, 5], [0, -5, 0], marker='s', s=8, color='k')
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.minorticks_on()

plt.show()
