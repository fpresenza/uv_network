import numpy as np
import matplotlib.pyplot as plt

from uvnpy.sensores import rango

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Innovación de información a través de rango', fontsize=15)
fig.subplots_adjust(wspace=0.3)
cw = plt.cm.get_cmap('coolwarm')

t = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(t, t)
Z = np.empty_like(X)
print(X[1, 2])

landmarks = [(0, -1), (0, 1)]

iter = range(len(t))
for i in iter:
    for j in iter:
        p = [X[i, j], Y[i, j]]
        HtRH = rango.matriz_innovacion(p, landmarks, 1.)
        Z[i, j] = np.linalg.det(HtRH)

cbar = axes[0].contourf(X, Y, Z, levels=20, cmap=cw)
fig.colorbar(cbar, ax=axes[0], fraction=0.046, pad=0.04)
axes[0].scatter(*zip(*landmarks), marker='*', color='k')
axes[0].set_title('($m = 2$)')


landmarks = [(0, -1), (0, 1), (-1, 0)]

for i in iter:
    for j in iter:
        p = [X[i, j], Y[i, j]]
        HtRH = rango.matriz_innovacion(p, landmarks, 1.)
        Z[i, j] = np.linalg.det(HtRH)

cbar = axes[1].contourf(X, Y, Z, levels=20, cmap=cw)
fig.colorbar(cbar, ax=axes[1], fraction=0.046, pad=0.04)
axes[1].scatter(*zip(*landmarks), marker='*', color='k')
axes[1].set_title('($m = 3$)')

for ax in axes:
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.minorticks_on()


plt.show()
fig.savefig('/tmp/contorno.png', format='png')
