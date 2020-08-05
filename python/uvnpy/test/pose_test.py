#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import gpsic.toolkit.linalg as linalg

fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

def get_axis(c, att):
    x, y, z = c 
    u, v, w = att
    return x,y,z,u,v,w

C = linalg.rm.ZYX(np.random.normal(0, np.pi, 3))
fixed_quiver = ax.quiver(0,0,5, *C, color='b')
quiver = ax.quiver(0,0,5, *np.eye(3))

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(0, 10)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

def update(yaw):
    global quiver, C
    quiver.remove()
    c = np.random.normal([0, 0, 5], 0.3, 3)
    n, t = np.random.normal(0, 1, 3), np.random.normal(0, 0.15)
    att = linalg.rm.from_vec(n,t) @ C

    quiver = ax.quiver(*c, *att, color='r')

ani = FuncAnimation(fig, update, frames=np.linspace(0,2*np.pi,200), interval=50)
plt.show()