#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from uv_network.dyn.holonomic import MotionModelVelocity
from uv_network.lib.graphic_tools import plot_ellipse
cmd_vel = np.array([[1.],[1.], [0.]])
Ts = 0.5
time = np.arange(0, 99, Ts)
N = 100

fig1 = plt.figure(1)
tgraph = fig1.add_subplot(111)
tgraph.grid(True)

fig2 = plt.figure(2)
xygraph = fig2.add_subplot(111)
xygraph.set_aspect('equal')
xygraph.minorticks_on()
xygraph.set_xlabel('$X\,[\mathrm{m}]$')
xygraph.set_ylabel('$Y\,[\mathrm{m}]$')
xygraph.grid(True)

alphas = (0.2,0.2)

for _ in range(N):

    X = np.zeros((9,1))
    X_sample = np.copy(X)
    P = np.zeros((9,9))

    dyn = MotionModelVelocity(3, alphas=alphas)

    ax, ay, ath = np.empty((3, len(time)))
    vx, vy, vth = np.empty((3, len(time)))
    px, py, pth = np.empty((3, len(time)))
 
    for k, t in enumerate(time):
        X_sample, acc = dyn.sample(X_sample, cmd_vel, Ts)
        ax[k], ay[k], ath[k] = acc
        vx[k], vy[k], vth[k] = X_sample[:3]
        px[k], py[k], pth[k] = X_sample[3:6]


    # tgraph.plot(time, ax)
    # tgraph.plot(time, ay)
    # tgraph.plot(time, ath)

    # tgraph.plot(time, vx)
    # tgraph.plot(time, vy)
    # tgraph.plot(time, vth)
    
    # tgraph.plot(time, px)
    # tgraph.plot(time, py)
    # tgraph.plot(time, pth)
    
    points = [[px[0],px[-1]], [py[0],py[-1]]]
    xygraph.scatter(*points, s=1.5)

X = np.zeros((6,1))
X_sample = np.copy(X)
P = np.zeros((6,6))
dyn = MotionModelVelocity(2, alphas=alphas)
for k, t in enumerate(time):
    X, P = dyn.gaussian_propagation(X, P, cmd_vel, Ts)
center = X[2:4]
sigma = P[2:4,2:4]
plot_ellipse(xygraph, center, sigma, color='b')
plt.show()