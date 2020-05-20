#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import control.matlab as cm
import sympy as sym
cm.use_numpy_matrix(False) 

s = lambda x: sym.sin(x)
c = lambda x: sym.cos(x)
t = lambda x: sym.tan(x)
cross = lambda x, y: np.cross(x, y, axis=0)

px, py, pz, vx, vy, vz, ar, ap, ay, wx, wy, wz = sym.symbols('px, py, pz, vx, vy, vz, 𝜙, 𝜃, 𝜓, wx, wy, wz', real=True)
# ex, ey, ez, ew = sym.symbols('ex, ey, ez, ew', real=True)
p = np.array([[px], [py], [pz]])
v = np.array([[vx], [vy], [vz]])
a = np.array([[ar], [ap], [ay]])
w = np.array([[wx], [wy], [wz]])
x = np.vstack((v, a, w))

# Kinematics
Rx = np.array([[1,     0,      0],
               [0, c(ar), -s(ar)],
               [0, s(ar),  c(ar)]])

Ry = np.array([[ c(ap), 0, s(ap)],
               [     0, 1,     0],
               [-s(ap), 0, c(ap)]])

Rz = np.array([[c(ay), -s(ay), 0],
               [s(ay),  c(ay), 0],
               [    0,      0, 1]])

# rotation from body to earth
Rzyx = Rz @ Ry @ Rx

# transformation angular velocity to euler angles derivative
T = np.array([[1, s(ar)*t(ap), c(ar)*t(ap)],
              [0,       c(ar),      -s(ar)],
              [0, s(ar)/c(ap), c(ar)/c(ap)]])

dp = v
da = T @ w

# Dynamics
m, g, Ix, Iy, Iz = sym.symbols('m, g, Ix, Iy, Iz', real=True)

Ft, Tx, Ty, Tz, fwx, fwy, fwz = sym.symbols('Ft, 𝛵x, 𝛵y, 𝛵z, fwx, fwy, fwz', real=True)
I = np.diag([Ix, Iy, Iz])
u = np.array([[Ft], [Tx], [Ty], [Tz]])
fw = np.array([[fwx],[fwy],[fwz]])
F = - m * np.array([[0],[0],[g]]) + Rzyx @ np.array([[0],[0],[Ft]]) + fw # positive z upwards
dv = F / m

T = np.array([[Tx], [Ty], [Tz]])
# print(T)
dw= np.diag([1/Ix, 1/Iy, 1/Iz]) @ (T - cross(w, I @ w))

dx = np.vstack((dv, da, dw))

# # equilibrium point
x_op = np.zeros((9,1))
u_op = np.array([[m*g],[0],[0],[0]])
fw_op = np.zeros((3,1))
op_st = list(zip(x.flat, x_op.flat)) + list(zip(u.flat, u_op.flat)) + list(zip(fw.flat, fw_op.flat))

f = sym.Matrix(dx)
F_x = f.jacobian(x)
F_u = f.jacobian(u)

F_x_op = F_x.subs(op_st)
F_u_op = F_u.subs(op_st)

A = np.array(F_x_op.tolist())
B = np.array(F_u_op.tolist())

f_op = np.array(f.subs(op_st).tolist()) # this should always be equal to zero
if not np.all(f_op==0):
  raise ValueError('That\'s not an equilibrium point!')
f_lin = A@(x-x_op) + B@(u-u_op) #+ Br@(r-r_op)

F_x_op = F_x_op.subs(g, 9.81)
F_u_op = F_u_op.subs([(m, 1), (Ix, 10e-3), (Iy, 10e-3), (Iz, 20e-3)])

A = np.array(F_x_op.tolist())
B = np.array(F_u_op.tolist())

# system analysis
# print(np.linalg.eig(A))
Cab = cm.ctrb(A,B)
print('Controllability matrix rank: {}'.format(np.linalg.matrix_rank(Cab)))

### print eqs ###
print('Non-linear model:')
for c, dc in zip(x, dx):
    print('d{}/dt = {}'.format(c[0], dc[0]))
print("------------")
print('Linear model:')
for c, dc in zip(x, f_lin):
    print('d{}/dt = {}'.format(c[0], dc[0]))
print('A:\n{}'.format(A))
print('B:\n{}'.format(B))