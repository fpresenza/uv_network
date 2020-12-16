#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mié dic 16 10:13:17 -03 2020
"""
import numpy as np
import matplotlib.pyplot as plt

from uvnpy.redes import analisis
from uvnpy.filtering import metricas


jacobiano = analisis.distancia_relativa_jac
matriz_incidencia = analisis.matriz_incidencia
conectar = analisis.disk_graph
svdvals = metricas.svdvals

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('$F(H)$ vs. $(x_3, y_3)$ (Topología Estática)', fontsize=15)
fig.subplots_adjust(hspace=0.5)
cw = plt.cm.get_cmap('coolwarm')

t = np.linspace(-10, 10, 100)
N = range(100)
X, Y = np.meshgrid(t, t)
norma2 = np.empty_like(X)
nuclear = np.empty_like(X)
prod = np.empty_like(X)
cond = np.empty_like(X)

p = np.array([[-5, 0],
              [0., 0],
              [0,  5]])
E = np.array([
    [0, 1],
    [1, 2],
    [2, 3]])
V = range(3)

for i in N:
    for j in N:
        p[2] = X[i, j], Y[i, j]
        # E = np.array(conectar(p, 8.))
        D = matriz_incidencia(V, E)
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
axes[0, 1].set_title(r'$\kappa = \sigma_1 / \sigma_r$')

cbar = axes[1, 0].contourf(X, Y, nuclear, levels=20, cmap=cw)
fig.colorbar(cbar, ax=axes[1, 0])
axes[1, 0].set_title('norma-nuc')

cbar = axes[1, 1].contourf(X, Y, prod, levels=20, cmap=cw)
fig.colorbar(cbar, ax=axes[1, 1])
axes[1, 1].set_title(r'$\prod_i \sigma_i$')

for ax in axes.flat:
    ax.scatter([-5, 0], [0, 0], marker='s', s=8, color='k')
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.minorticks_on()

plt.show()
fig.savefig('/tmp/contorno_3v.png', format='png')
