#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from uvnpy.vehicles.rover import Rover
import uvnpy.graphix.planar as graphix
from uvnpy.navigation.kalman import Ekf
from uvnpy.toolkit.linalg import vector

cmd_vel = (1., 1., 0.5)
Ts = 0.05
time = np.arange(0, 99, Ts)
N = 40 # number of episodes

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
        r.step(cmd_vel, t)
        a.append(r.accel())
        v.append(r.vel())
        p.append(r.pose())

    r.motion.restart()

    accel = vector.vec3(*np.hstack(a))
    vel = vector.vec3(*np.hstack(v)) 
    pos = vector.vec3(*np.hstack(p)) 

    tgraph.plot(time, accel.x)

    tgraph.plot(time, vel.x)

    tgraph.plot(time, pos.x)

    points = [[pos.x[0],pos.x[-1]], [pos.y[0], pos.y[-1]]]
    xygraph.scatter(*points, s=1.5)

ekf = Ekf()
ekf.begin(r.motion.x, 0.5*r.motion.x)
for t in time[1:]:
    ekf.prediction(r.extend(cmd_vel), r.motion, t)

center = ekf.X[6:8]
sigma = ekf.P[6:8,6:8]
graphix.ellipse(xygraph, center, sigma)
xygraph.set_aspect('equal')
xygraph.grid(1)
xygraph.set_xlim(0,115)
xygraph.set_ylim(0,115)
plt.show()