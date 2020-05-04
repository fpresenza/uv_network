#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from uvnpy.network.vehicles import Rover
import uvnpy.lib.graphix as graphix
from uvnpy.navigation.filters import Ekf
from uvnpy.lib.ros import Vector3
import uvnpy.lib.tools as tools

cmd_vel = np.array([[1.],[1.], [0.]])
Ts = 0.5
time = np.arange(0, 99, Ts)
N = 50 # number of episodes

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

r = Rover(1)

for _ in range(N):
    a = [np.zeros((3,1))]
    v = [np.zeros((3,1))]
    p = [np.zeros((3,1))]

    for t in time[1:]:
        acc = r.motion.step(cmd_vel, t)
        a.append(acc)
        v.append(r.motion.X[:3])
        p.append(r.motion.X[3:])

    r.motion.restart()

    accel = Vector3(*tools.from_arrays(a))
    vel = Vector3(*tools.from_arrays(v)) 
    pos = Vector3(*tools.from_arrays(p)) 

    tgraph.plot(time, accel.x)
    # tgraph.plot(time, ay)
    # tgraph.plot(time, ath)

    tgraph.plot(time, vel.x)
    # tgraph.plot(time, vy)
    # tgraph.plot(time, vth)

    tgraph.plot(time, pos.x)
    # tgraph.plot(time, py)
    # tgraph.plot(time, pth)

    points = [[pos.x[0],pos.x[-1]], [pos.y[0], pos.y[-1]]]
    xygraph.scatter(*points, s=1.5)


X = r.motion.X
dX = 0.5*r.motion.X
ekf = Ekf()
ekf.begin(X, dX)
for t in time[1:]:
    ekf.prediction(cmd_vel, r.motion.f, t)

center = ekf.X[3:5]
sigma = ekf.P[3:5,3:5]
graphix.ellipse(xygraph, center, sigma)
xygraph.set_aspect('equal')
xygraph.grid(1)
xygraph.set_xlim(0,115)
xygraph.set_ylim(0,115)
plt.show()