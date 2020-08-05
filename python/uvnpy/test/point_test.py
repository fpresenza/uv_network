#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from uvnpy.vehicles.point import Point
# import uvnpy.graphix.planar as planar
import gpsic.plotting.planar as planar
from uvnpy.navigation.kalman import Ekf
from uvnpy.navigation.metrics import metrics

cmd_vel = (1., 1.)
freq = 100
Ts = 1/freq
time = np.arange(0, 100, Ts)
N = 10 # number of episodes

fig1 = plt.figure(1)
tgraph = fig1.add_subplot(111)
tgraph.grid(True)

fig2 = plt.figure(2)
xygraph = fig2.add_subplot(111)
xygraph.set_aspect('equal')
xygraph.minorticks_on()
xygraph.set_xlabel('$x\,[\mathrm{m}]$')
xygraph.set_ylabel('$y\,[\mathrm{m}]$')
xygraph.grid(True)


for i in range(N):
    r = Point(i, motion_kw={'freq':freq})#, motion_kw={'pi':np.random.normal(0, 1.5, 2)})

    p = [r.p()]
    v = [r.v()]

    for t in time[1:]:
        r.motion.step(cmd_vel, t)
        p.append(r.p())
        v.append(r.v())

    r.motion.restart()

    v = np.vstack(v).T
    p = np.vstack(p).T

    tgraph.plot(time, v[0])
    tgraph.plot(time, p[0])

    points = [[p[0][0],p[0][-1]], [p[1][0], p[1][-1]]]
    xygraph.scatter(*points, s=1.5)

ekf = Ekf(r.motion.x, 0.5*r.motion.x)
for t in time[1:]:
    ekf.prediction(r.motion.f, cmd_vel, t)

center = ekf.x[:2]
sigma = metrics.sqrt_covar(ekf.P[:2,:2])
planar.ellipse(xygraph, center, sigma, alpha=0.4)
xygraph.set_aspect('equal')
xygraph.grid(1)
xygraph.set_xlim(0,115)
xygraph.set_ylim(0,115)
plt.show()