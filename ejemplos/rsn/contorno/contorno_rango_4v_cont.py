#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date vie ene 29 00:23:01 -03 2021
"""
import numpy as np
import matplotlib.pyplot as plt

import uvnpy.rsn.core as rsn
from uvnpy.filtering import metricas


fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('$F(Y)$ vs. $(x_4, y_4)$ (Grafo completo)', fontsize=15)
fig.subplots_adjust(hspace=0.5)
cw = plt.cm.get_cmap('coolwarm')

t = np.linspace(-10, 10, 100)
N = range(100)
X, Y = np.meshgrid(t, t)
Q = [0.125, 0.25, 0.5]
# norma2 = np.empty_like(X)
# nuclear = np.empty_like(X)
# pi = np.empty_like(X)
# kappa = np.empty_like(X)
dims = (len(Q),) + X.shape
vmr = np.empty(dims)
rsd = np.empty(dims)

p = np.array([[-5, 0],
              [0, -5],
              [5., 0],
              [0,  0]])
V = [0, 1, 2, 3]


for i in N:
    for j in N:
        p[3] = X[i, j], Y[i, j]

        A = rsn.distances(p)
        A[A != 0] **= -1
        # sv_H = np.sqrt(eig[eig > 1e-2])
        # norma2[i, j] = max(sv_H)
        # nuclear[i, j] = sv_H.sum()
        # pi[i, j] = sv_H.prod()
        # kappa[i, j] = max(sv_H) / min(sv_H)

        for k, q in enumerate(Q):
            A = rsn.distances(p)
            A[A != 0] **= -2*q
            L = rsn.distances_innovation_laplacian(A, p)
            eig = metricas.eigvalsh(L)
            vmr[k, i, j] = metricas.variance_to_mean_ratio(eig)
            rsd[k, i, j] = metricas.relative_standard_deviation(eig)

cbar = axes[0, 0].contourf(X, Y, vmr[0], levels=20, cmap=cw)
fig.colorbar(cbar, ax=axes[0, 0])
axes[0, 0].set_title(r'vmr$(\lambda_i) \quad (q={})$'.format(Q[0]))

cbar = axes[0, 1].contourf(X, Y, vmr[1], levels=20, cmap=cw)
fig.colorbar(cbar, ax=axes[0, 1])
axes[0, 1].set_title(r'vmr$(\lambda_i) \quad (q={})$'.format(Q[1]))

cbar = axes[0, 2].contourf(X, Y, vmr[2], levels=20, cmap=cw)
fig.colorbar(cbar, ax=axes[0, 2])
axes[0, 2].set_title(r'vmr$(\lambda_i) \quad (q={})$'.format(Q[2]))

cbar = axes[1, 0].contourf(X, Y, rsd[0], levels=20, cmap=cw)
fig.colorbar(cbar, ax=axes[1, 0])
axes[1, 0].set_title(r'rsd$(\lambda_i) \quad (q={})$'.format(Q[0]))

cbar = axes[1, 1].contourf(X, Y, rsd[1], levels=20, cmap=cw)
fig.colorbar(cbar, ax=axes[1, 1])
axes[1, 1].set_title(r'rsd$(\lambda_i) \quad (q={})$'.format(Q[1]))

cbar = axes[1, 2].contourf(X, Y, rsd[2], levels=20, cmap=cw)
fig.colorbar(cbar, ax=axes[1, 2])
axes[1, 2].set_title(r'rsd$(\lambda_i) \quad (q={})$'.format(Q[2]))

for ax in axes.flat:
    ax.scatter([-5, 0, 5], [0, -5, 0], marker='s', s=8, color='k')
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.minorticks_on()

fig.savefig('/tmp/contorno_4v_cont.png', format='png')

plt.show()
