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


fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('$F(H)$ vs. $(x_4, y_4)$', fontsize=15)
fig.subplots_adjust(hspace=0.5)
cw = plt.cm.get_cmap('coolwarm')

t = np.linspace(-10, 10, 100)
N = range(100)
X, Y = np.meshgrid(t, t)
vmr_f = np.empty_like(X)
vmr_v = np.empty_like(X)
rsd_f = np.empty_like(X)
rsd_v = np.empty_like(X)


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
Af = redes.undirected_adjacency_from_edges(V, Ef)


for i in N:
    for j in N:
        p[3] = X[i, j], Y[i, j]

        Lf = rsn.distances_innovation_laplacian(Af, p)
        eigvals_f = metricas.eigvalsh(Lf)
        vmr_f[i, j] = metricas.variance_to_mean_ratio(eigvals_f)
        rsd_f[i, j] = metricas.relative_standard_deviation(eigvals_f)

        A = redes.adjacency_from_positions(p, dmax=8.)
        A[A != 0] = 1
        L = rsn.distances_innovation_laplacian(A, p)
        eigvals = metricas.eigvalsh(L)
        vmr_v[i, j] = metricas.variance_to_mean_ratio(eigvals)
        rsd_v[i, j] = metricas.relative_standard_deviation(eigvals)


cbar = axes[0, 0].contourf(X, Y, vmr_f, levels=20, cmap=cw)
fig.colorbar(cbar, ax=axes[0, 0])
axes[0, 0].set_title(r'Topología Fija $\rm{vmr}(\lambda_i)$')

cbar = axes[0, 1].contourf(X, Y, vmr_v, levels=20, cmap=cw)
fig.colorbar(cbar, ax=axes[0, 1])
axes[0, 1].set_title(r'Topología Variable $\rm{vmr}(\lambda_i)$')

cbar = axes[1, 0].contourf(X, Y, rsd_f, levels=20, cmap=cw)
fig.colorbar(cbar, ax=axes[1, 0])
axes[1, 0].set_title(
    r'Topología Fija $\rm{rsd}(\lambda_i)}}$')

cbar = axes[1, 1].contourf(X, Y, rsd_v, levels=20, cmap=cw)
fig.colorbar(cbar, ax=axes[1, 1])
axes[1, 1].set_title(
    r'Topología Variable $\rm{rsd}(\lambda_i)}$')

for ax in axes.flat:
    ax.scatter([-5, 0, 5], [0, -5, 0], marker='s', s=8, color='k')
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.minorticks_on()


fig.savefig('/tmp/contorno_4v_b.png', format='png')

plt.show()
