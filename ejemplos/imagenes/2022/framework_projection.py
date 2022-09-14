#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


from uvnpy.network import plot, disk_graph


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'serif'

plt.rcParams["legend.borderpad"] = 0.1
plt.rcParams["legend.labelspacing"] = 0.3
plt.rcParams["legend.handlelength"] = 1.0
plt.rcParams["legend.columnspacing"] = 1.0


def proj(p, basis):
    A = basis.T
    P = A.dot(np.linalg.inv(A.T.dot(A))).dot(A.T)
    return p.dot(P)


def draw_subpace(ax, normal, **kwargs):
    """
    Plotea un plano dado los coeficientes de la expresion
    a*x + b*y + c*z = d
    """
    a, b, c = normal
    xl, xr = np.multiply(ax.get_xlim(), 1.5)
    yl, yr = np.multiply(ax.get_ylim(), 1.5)
    X, Y = np.meshgrid(np.arange(xl, xr), np.arange(xl, xr))
    A = -a/c
    B = -b/c
    Z = A * X + B * Y
    ax.plot_surface(X, Y, Z, **kwargs)


d, n = 3, 10
p = np.random.uniform(-8, 8, (n, d)) + np.array([0, 0, 10])
dmax = 11
A = disk_graph.adjacency(p, dmax)

fig, ax = plot.figure(figsize=(4, 4), subplot_kw={'projection': '3d'})
plot.graph(ax, p, A, lw=0.9, color='C1')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-0, 20)
ax.view_init(elev=15, azim=-45)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])


# 2-dim projection
basis = np.array([[1, 0, 0], [0, 1, 0]])
normal = np.cross(basis[0], basis[1])
q2 = proj(p, basis)
# fig, ax = plot.figure(subplot_kw={'projection': '3d'})
plot.graph(ax, q2, A, lw=0.8, color='C0')
# draw_subpace(ax, normal, alpha=0.5)
# ax.set_xlim(-15, 15)
# ax.set_ylim(-15, 15)
# ax.set_zlim(-15, 15)

plt.show()

fig.savefig('/tmp/framework_projection.png', format='png', dpi=300)
