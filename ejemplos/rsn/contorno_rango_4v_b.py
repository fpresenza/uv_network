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


def varianza(eigvals, eigmean):
    d = eigvals - eigmean
    return d.dot(d) / len(eigvals)


jacobiano = rsn.distancia_relativa_jac
incidence_from_edges = redes.incidence_from_edges
conectar = redes.edges_from_positions
eigvalsh = metricas.eigvalsh

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('$F(H)$ vs. $(x_4, y_4)$', fontsize=15)
fig.subplots_adjust(hspace=0.5)
cw = plt.cm.get_cmap('coolwarm')

t = np.linspace(-10, 10, 100)
N = range(100)
X, Y = np.meshgrid(t, t)
var_f = np.empty_like(X)
var_v = np.empty_like(X)
var_norm_f = np.empty_like(X)
var_norm_v = np.empty_like(X)


p = np.array([[-5, 0],
              [0, -5],
              [5., 0],
              [0,  0]])
Ef = np.array([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0]])
V = [0, 1, 2, 3]
Df = incidence_from_edges(V, Ef)


for i in N:
    for j in N:
        p[3] = X[i, j], Y[i, j]

        Hf = jacobiano(p, Df)
        Mf = Hf.T.dot(Hf)
        eigvals_f = eigvalsh(Mf)
        eigmean_f = eigvals_f.mean()
        var_f[i, j] = varianza(eigvals_f, eigmean_f)
        var_norm_f[i, j] = var_f[i, j] / (eigmean_f**2)

        E = np.array(conectar(p, 8.))
        D = incidence_from_edges(V, E)
        H = jacobiano(p, D)
        M = H.T.dot(H)
        eigvals = eigvalsh(M)
        eigmean = eigvals.mean()
        var_v[i, j] = varianza(eigvals, eigmean)
        var_norm_v[i, j] = 1.01 * var_v[i, j] / (eigmean**2)


cbar = axes[0, 0].contourf(X, Y, var_f, levels=20, cmap=cw)
fig.colorbar(cbar, ax=axes[0, 0])
axes[0, 0].set_title(r'Topología Fija $\rm{var}(\lambda)$')

cbar = axes[0, 1].contourf(X, Y, var_v, levels=20, cmap=cw)
fig.colorbar(cbar, ax=axes[0, 1])
axes[0, 1].set_title(r'Topología Variable $\rm{var}(\lambda)$')

cbar = axes[1, 0].contourf(X, Y, var_norm_f, levels=20, cmap=cw)
fig.colorbar(cbar, ax=axes[1, 0])
axes[1, 0].set_title(
    r'Topología Fija $\frac{\rm{var}(\lambda)}{\bar{\lambda}^2}$')

cbar = axes[1, 1].contourf(X, Y, var_norm_v, levels=20, cmap=cw)
fig.colorbar(cbar, ax=axes[1, 1])
axes[1, 1].set_title(
    r'Topología Variable $\frac{\rm{var}(\lambda)}{\bar{\lambda}^2}$')

for ax in axes.flat:
    ax.scatter([-5, 0, 5], [0, -5, 0], marker='s', s=8, color='k')
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.minorticks_on()


fig.savefig('/tmp/contorno_4v_b.png', format='png')

plt.show()
