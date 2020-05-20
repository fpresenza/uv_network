#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import control.matlab as cm
import sympy as sym
import recordclass
cm.use_numpy_matrix(False) 

# s = m.tf('s')

# l = 10e5/(s*(s+15)*(s+1000)) # del capitulo 10 del libro de JG y RSP
# tfr = m.minreal(l/(1+l))
# tfp = tfr

# Gr = m.tf2ss(tfr)
# Gp = m.tf2ss(tfp)

# G = m.append(Gr, Gp)

# print('State Space model\nA:\n{}\nB:\n{}\nC:\n{}\nD:\n{}'.format(G.A, G.B, G.C, G.D))

s = lambda x: sym.sin(x)
c = lambda x: sym.cos(x)
t = lambda x: sym.tan(x)
cross = lambda x, y: np.cross(x, y, axis=0)

px, py, pz, vx, vy, vz, er, ep, ey, wx, wy, wz = sym.symbols('px, py, pz, vx, vy, vz, 𝜙, 𝜃, 𝜓, wx, wy, wz', real=True)

p = np.array([[px], [py], [pz]])
v = np.array([[vx], [vy], [vz]])
e = np.array([[er], [ep], [ey]])
w = np.array([[wx], [wy], [wz]])

# Kinematics
Rx = np.array([[1,     0,      0],
               [0, c(er), -s(er)],
               [0, s(er),  c(er)]])

Ry = np.array([[ c(ep), 0, s(ep)],
               [     0, 1,     0],
               [-s(ep), 0, c(ep)]])

Rz = np.array([[c(ey), -s(ey), 0],
               [s(ey),  c(ey), 0],
               [    0,      0, 1]])

# rotation from body to earth
Rzyx = Rz @ Ry @ Rx
# transformation angular velocity to euler angles derivative
T = np.array([[1, s(er)*t(ep), c(er)*t(ep)],
              [0,       c(er),      -s(er)],
              [0, s(er)/c(ep), c(er)/c(ep)]])

dp = Rzyx @ v
de = T @ w

# Dynamics
m, g, Ix, Iy, Iz, Ft, Tx, Ty, Tz = sym.symbols('m, g, Ix, Iy, Iz, Ft, 𝛵x, 𝛵y, 𝛵z', real=True)
I = np.diag([Ix, Iy, Iz])
F = - m * Rzyx.T @ np.array([[0],[0],[g]]) + np.array([[0],[0],[Ft]]) # positive z downwards
dv = - cross(w, v) + F / m

T = np.array([[Tx],[Ty],[Tz]])
dw = np.diag([1/Ix, 1/Iy, 1/Iz]) @ (T - cross(w, I @ w))

x = np.vstack((p, v, e, w))
dx = np.vstack((dp, dv, de, dw))
u = np.array([[Ft],[Tx],[Ty],[Tz]])

f = sym.Matrix(dx)
F_x = f.jacobian(x)
F_u = f.jacobian(u)

# equilibrium point
x_eq = np.vstack((p, np.zeros((9,1))))
u_eq = np.array([[m*g],[0],[0],[0]])
eq_point = list(zip(x.flat, x_eq.flat)) + list(zip(u.flat, u_eq.flat))
F_x_eq = F_x.subs(eq_point)
F_u_eq = F_u.subs(eq_point)

A = np.array(F_x_eq.tolist())
B = np.array(F_u_eq.tolist())

f_eq = np.array(f.subs(eq_point).tolist()) # this should always be equal to zero
if not np.all(f_eq==0):
  raise ValueError('That\'s not an equilibrium point!')
dx_lin = f_eq + A@(x-x_eq) + B@(u-u_eq)

# F_x_eq = F_x_eq.subs(g, 9.81)
# F_u_eq = F_u_eq.subs([(m, 1), (Ix, 10e-3), (Iy, 10e-3), (Iz, 20e-3)])

A = np.array(F_x_eq.tolist())
B = np.array(F_u_eq.tolist())

# symste analysis
# print(np.linalg.eig(A))
# Cab = cm.ctrb(A,B)
# print(np.linalg.matrix_rank(Cab))

### print eqs ###
print('Non-linear model:')
for s, ds in zip(x, dx):
    print('d{}/dt = {}'.format(s[0], ds[0]))
print("------------")
print('Linear model:')
for s, ds in zip(x, dx_lin):
    print('d{}/dt = {}'.format(s[0], ds[0]))
print('A:\n{}'.format(A))
print('B:\n{}'.format(B))

# Non-linear model:
# dpx/dt = vx*cos(𝜃)*cos(𝜓) + vy*(sin(𝜃)*sin(𝜙)*cos(𝜓) - sin(𝜓)*cos(𝜙)) + vz*(sin(𝜃)*cos(𝜓)*cos(𝜙) + sin(𝜓)*sin(𝜙))
# dpy/dt = vx*sin(𝜓)*cos(𝜃) + vy*(sin(𝜃)*sin(𝜓)*sin(𝜙) + cos(𝜓)*cos(𝜙)) + vz*(sin(𝜃)*sin(𝜓)*cos(𝜙) - sin(𝜙)*cos(𝜓))
# dpz/dt = -vx*sin(𝜃) + vy*sin(𝜙)*cos(𝜃) + vz*cos(𝜃)*cos(𝜙)
# dvx/dt = g*sin(𝜃) + vy*wz - vz*wy
# dvy/dt = -g*sin(𝜙)*cos(𝜃) - vx*wz + vz*wx
# dvz/dt = vx*wy - vy*wx + (Ft - g*m*cos(𝜃)*cos(𝜙))/m
# d𝜙/dt = wx + wy*sin(𝜙)*tan(𝜃) + wz*cos(𝜙)*tan(𝜃)
# d𝜃/dt = wy*cos(𝜙) - wz*sin(𝜙)
# d𝜓/dt = wy*sin(𝜙)/cos(𝜃) + wz*cos(𝜙)/cos(𝜃)
# dwx/dt = (Iy*wy*wz - Iz*wy*wz + 𝛵x)/Ix
# dwy/dt = (-Ix*wx*wz + Iz*wx*wz + 𝛵y)/Iy
# dwz/dt = (Ix*wx*wy - Iy*wx*wy + 𝛵z)/Iz
# ------------
# Linear model:
# dpx/dt = vx
# dpy/dt = vy
# dpz/dt = vz
# dvx/dt = g*𝜃
# dvy/dt = -g*𝜙
# dvz/dt = (Ft - g*m)/m
# d𝜙/dt = wx
# d𝜃/dt = wy
# d𝜓/dt = wz
# dwx/dt = 𝛵x/Ix
# dwy/dt = 𝛵y/Iy
# dwz/dt = 𝛵z/Iz
# A:
# [[0 0 0 1 0 0 0 0 0 0 0 0]
#  [0 0 0 0 1 0 0 0 0 0 0 0]
#  [0 0 0 0 0 1 0 0 0 0 0 0]
#  [0 0 0 0 0.0 0 0 g 0 0 0 0.0]
#  [0 0 0 0 0 0.0 -g 0 0 0.0 0 0]
#  [0 0 0 0.0 0 0 0 0 0 0 0.0 0]
#  [0 0 0 0 0 0 0 0.0 0 1 0 0]
#  [0 0 0 0 0 0 0 0 0 0 1 0]
#  [0 0 0 0 0 0 0.0 0 0 0 0 1]
#  [0 0 0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0 0 0]]
# B:
# [[0 0 0 0]
#  [0 0 0 0]
#  [0 0 0 0]
#  [0 0 0 0]
#  [0 0 0 0]
#  [1/m 0 0 0]
#  [0 0 0 0]
#  [0 0 0 0]
#  [0 0 0 0]
#  [0 1/Ix 0 0]
#  [0 0 1/Iy 0]
#  [0 0 0 1/Iz]]