#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mié dic 16 12:21:51 -03 2020
"""
import numpy as np
import matplotlib.pyplot as plt

from uvnpy.redes import analisis
from uvnpy.filtering import metricas


jacobiano = analisis.rp_jac
matriz_incidencia = analisis.matriz_incidencia
conectar = analisis.disk_graph
svdvals = metricas.svdvals

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('$F(H)$ vs. $(x_2, y_2)$ (Topología Fija)', fontsize=15)
fig.subplots_adjust(hspace=0.5)
cw = plt.cm.get_cmap('coolwarm')

t = np.linspace(-10, 10, 100)
N = range(100)
X, Y = np.meshgrid(t, t)
norma2 = np.empty_like(X)
nuclear = np.empty_like(X)
prod = np.empty_like(X)
cond = np.empty_like(X)

V = range(4)
p = np.array([[-5, 0],
              [0, -5],
              [5., 0],
              [0,  5]])
Er = conectar(p, 8.)
Ep = [(0, 0), (1, 1)]
Dr = matriz_incidencia(V, Er)
Dp = matriz_incidencia(V, Ep)

for i in N:
    for j in N:
        p[1] = X[i, j], Y[i, j]
        # Er = conectar(p, 8.)
        # Ep = [(0, 0), (1, 1)]
        # Dr = matriz_incidencia(V, Er)
        # Dp = matriz_incidencia(V, Ep)
        H = jacobiano(p, Dr, Dp)
        sv = svdvals(H)
        psv = sv[sv > 1e-3]
        norma2[i, j] = psv[0]
        nuclear[i, j] = psv.sum()
        prod[i, j] = psv.prod()
        cond[i, j] = psv[0] / psv[-1]

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
    ax.scatter([0, 5], [5, 0], marker='s', s=8, color='k')
    ax.scatter([-5], [0], marker='*', s=15, color='g')
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.minorticks_on()

plt.show()
fig.savefig('/tmp/contorno_4v.png', format='png')
