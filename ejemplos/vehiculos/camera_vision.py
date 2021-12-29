#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import numpy as np
# import sympy as sym
# import uvnpy.toolkit.linalg as la

# s = lambda x: sym.sin(x)
# c = lambda x: sym.cos(x)
# t = lambda x: sym.tan(x)
# cross = lambda x, y: np.cross(x, y, axis=0)

# # a, b = sym.symbols('α, β', real=True)
# # x = homogeneous.augment([a,b])
# fh, fv = sym.symbols('fh, fv', real=True)
# # S = np.array([[1/fh,    0, 0],
# #               [   0, 1/fv, 0],
# #               [   0,    0, 0],
# #               [   0,    0, 1]])

# # # homogeneous transformation
# # R_a2f = np.array([[ 0, 0, 1],
# #                   [ 0, 1, 0],
# #                   [-1, 0, 0]])
# # # R_a2f = np.eye(3)
# ch, cv, cz = sym.symbols('ch cv cz', real=True)
# # tc_f = [ch/fh, cv/fv, -1]
# # # tf_c = [0, 0, -cz]

# # T_f2a = homogeneous.inverse_matrix(R_a2f, tc_f)

# # M = T_f2a @ S
# # print('M=\n{}\n'.format(M))
# # print('p_c = M*x=\n{}\n'.format(M@x))
# # K = sym.Matrix(M[:3]).inv()
# # print(
#        'K = M^-1=\n{}\n{}\n{}\n'.format(
#           K[0,:].tolist(),
#           K[1,:].tolist(), K[2,:].tolist()))

# # Kc = np.array([[fh,  0, ch, 0],
# #                [ 0, fv, cv, 0],
# #                [ 0,  0,  1, 0],
# #                [ 0,  0,  0, 1]])
# # Kalt = Kc @ homogeneous.matrix(R_a2f, [0,0,0])
# # print('Kalt = Kc * Ta2f\n{}\n'.format(Kalt))

# px, py, pz = sym.symbols('px, py, pz', real=True)
# p = np.array([[px], [py], [pz]])
# # h_x = homogeneous.dot(Kalt, p)
# # print('x = K*p_c=\n{}\n'.format(h_x))

# # print('(α, β) = {}\n'.format(h_x[:2]/h_x[2]))

# # # null space of projection vector
# # nx, ny, nz = sym.symbols('nx, ny, nz', real=True)
# # ns = np.array([[nx], [ny], [nz]])
# # B = np.array([[1, 0, nx],
# #               [0, 1, ny],
# #               [0, 0, nz]])
# # A = np.array([[1, 0, 0],
# #               [0, 1, 0],
# #               [0, 0, 0]])
# # print('Proj_xy_ns =\n{}'.format(B@A@sym.Matrix(B).inv().tolist()))
# # Proj_xy_ns = np.array([[1, 0, -nx/nz],
# #                        [0, 1, -ny/nz],
# #                        [0, 0, 0]])


# tx, ty, tz = sym.symbols('𝜙, 𝜃, 𝜓', real=True)
# # t = sym.sqrt(tx**2+ty**2+tz**2)
# S = linalg.skew([tx, ty, tz])
# # C = np.eye(3) + sym.s(t)*S/(t**2) + (1-sym.cos(t))*np.dot(S,S)/(t**2)


# qx, qy, qz = sym.symbols('qx, qy, qz', real=True)
# q = np.array([[qx], [qy], [qz]])

# Ma = np.array([[1,0,0],
#                [0,1,0]])

# Mb = np.array([[0,0,1]])

# K = np.array([[fh,  0, ch],
#               [ 0, fv, cv],
#               [ 0,  0,  1]])

# print(Mb @ K)

# B = sym.Matrix(Mb@K@(p-q))
# A = Ma@K@(p-q)

# print(B.inv().tolist() * A)


import numpy as np

from gpsic.toolkit import linalg
from uvnpy.modelos import uav

pi = np.array([-10, 2, 3])
dpi = np.multiply([-0.05, 0.1, 0.07], pi)
ei = np.array([0, 1, -0.5])
uav = uav(1, pi=pi, ai=ei)
# print(uav.cam.extrinsic)s

# for t in np.arange(0.1,5,0.1):
#     uav.step((0,0,1,0), t)
#     print(uav.motion.xyzyaw().flatten())

pj = np.array([2, 1, 1.1])
dpj = np.multiply([-0.05, 0.1, 0.07], pj)
# print(pj)
# print(uav.cam.to_pixel(pj))
# print(pj+dpj)
t = 0.1
a = linalg.normalizar([1, 1, 1])
# a = linalg.vector.normalize(pj+dpj-pi-dpi)
phi = a*t
dC = linalg.rodriguez(a, t)
# print(dC)

uav.cam.update_pose(pi, ei)
check_y = uav.cam.to_pixel(pj)
print('y_medida = {}'.format(check_y))
m, H = uav.h_cam_test(pi + dpi, pj + dpj, (dC, ei))
print('y_pred = {}'.format(m.flatten()))
dy = check_y - m.flatten()
print('dy = {}'.format(dy))
dy_est = -np.matmul(H, (dpj - dpi)) - \
    np.matmul(H, np.matmul(linalg.skew(pj - pi), phi))
print('dy_est = {}'.format(dy_est))
