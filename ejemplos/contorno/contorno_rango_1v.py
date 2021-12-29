import numpy as np
import matplotlib.pyplot as plt

import uvnpy.rsn.distances as distances
import uvnpy.network.connectivity as cnt


fig, ax = plt.subplots()
fig.suptitle('Innovación de información a través de rango', fontsize=15)
fig.subplots_adjust(wspace=0.3)
cw = plt.cm.get_cmap('coolwarm')

t = np.linspace(-10, 10, 100)
N = range(100)
x, y = np.meshgrid(t, t)
z = np.empty(x.shape)
dmax = 15.

q = np.array([
    [5.02869674, 8.68402927],
    [8.16874345, 5.34071037],
    [-0.85135279, -1.83178025],
    [0.29043253, 9.47488418]])


for i in N:
    for j in N:
        p = np.array([x[i, j], y[i, j]])
        w = distances.local_distances(p[None], q)
        w = cnt.logistic_strength(w, 1, e=dmax)
        Y = distances.local_innovation_matrix(p[None], q, w)
        z[i, j] = np.linalg.det(Y)

cbar = ax.contourf(x, y, z, levels=20, cmap=cw)
fig.colorbar(cbar, ax=ax, fraction=0.046, pad=0.04)
ax.scatter(q[:, 0], q[:, 1], marker='*', color='k')
ax.set_title('($m = {}$)'.format(len(q)))

ax.set_aspect('equal')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.minorticks_on()

plt.show()
fig.savefig('/tmp/contorno.png', format='png')
