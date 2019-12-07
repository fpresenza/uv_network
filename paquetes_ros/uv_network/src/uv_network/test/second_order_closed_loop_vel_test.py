#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from uv_network.dyn.holonomic import Holonomic
cmd_vel = np.array([[1.],[1.], [0.]])
Ts = 0.1
time = np.arange(0, 99, Ts)
N = 150
# fig1 = plt.figure(1)
# tgraph = fig1.add_subplot(111)
# tgraph.grid(True)

fig2 = plt.figure(2)
xygraph = fig2.add_subplot(111)
# xygraph.axis('scaled')
xygraph.minorticks_on()
xygraph.set_xlabel('$X\,[\mathrm{m}]$')
xygraph.set_ylabel('$Y\,[\mathrm{m}]$')
xygraph.grid(True)

for hyp in range(N):
    alphas = (0.2,0.2)
    dyn = Holonomic(3, 0, np.zeros((6,1)), alphas=alphas)

    ax, ay, ath = np.empty_like(time), np.empty_like(time), np.empty_like(time)
    vx, vy, vth = np.empty_like(time), np.empty_like(time), np.empty_like(time)
    px, py, pth = np.empty_like(time), np.empty_like(time), np.empty_like(time)
 

    for k, t in enumerate(time):
        res = dyn.second_order_closed_loop_vel(cmd_vel, t)
        # ax[k], ay[k], ath[k] = res[:3]
        # vx[k], vy[k], vth[k] = res[3:6]
        px[k], py[k], pth[k] = res[3:]

    points = [[px[0],px[-1]], [py[0],py[-1]]]

        

    # tgraph.plot(time, vx)
    # tgraph.plot(time, px)

    xygraph.scatter(*points, s=1.5)

plt.show()