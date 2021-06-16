#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar abr  6 15:11:20 -03 2021
"""
import numpy as np
import scipy.linalg


def traslation_basis(p):
    n = len(p)
    T = np.zeros((p.size, 2))
    T[::2, 0] = 1/np.sqrt(n)                    # dx
    T[1::2, 1] = 1/np.sqrt(n)                   # dy
    return T


def pose_basis(p):
    """Matriz cuyas columnas son una BON del espacio pose.

    args:
        p: array de posiciones (n, dof)

    returns
        M: matriz (n*dof, n*dof)
    """
    n = len(p)
    P = np.zeros((p.size, 3))
    r_cm = p - p.mean(0)

    P[::2, 0] = 1/np.sqrt(n)                    # dx
    P[1::2, 1] = 1/np.sqrt(n)                   # dy
    P[::2, 2] = -r_cm[:, 1]
    P[1::2, 2] = r_cm[:, 0]
    P[:, 2] /= np.sqrt(np.square(r_cm).sum())   # dt
    return P


def shape_basis(p):
    P = pose_basis(p)
    S = scipy.linalg.null_space(P.T)
    return S


def pose_and_shape_decomposition_aa(p):
    """Devuelve dos matrices de proyección.

    P proyecta al subespacio "pose",
    S proyecta al subespacio "shape".

    args:
        p: array de posiciones (..., n, dof)

    returns
        P, S: matrices (..., n * dof, n * dof)
    """
    N, n, d = p.shape
    s = n * d
    A = np.zeros((N, s, 3))     # 3 si 2d, 6 si 3d
    B = np.empty((N, s, s - 3))
    r_cm = p - p.mean(1)[:, None]

    A[:, ::2, 0] = 1/np.sqrt(n)     # dx
    A[:, 1::2, 1] = 1/np.sqrt(n)    # dy
    A[:, ::2, 2] = -r_cm[:, :, 1]
    A[:, 1::2, 2] = r_cm[:, :, 0]
    r_cm = r_cm.reshape(N, s)
    A[:, :, 2] /= np.sqrt(np.square(r_cm).sum(1))[:, None]  # dt

    A_T = A.swapaxes(-2, -1)
    B[:] = [scipy.linalg.null_space(a_T) for a_T in A_T]
    return A, B


def pose_and_shape_projections(p):
    """ Devuelve dos matrices de proyección.

    P proyecta al subespacio "pose",
    S proyecta al subespacio "shape".

    args:
        p: array de posiciones (n, dof)

    returns
        P, S: matrices (n * dof, n * dof)

    """
    n, d = p.shape
    s = n * d
    A = np.zeros((s, 3))     # 3 si 2d, 6 si 3d
    r_cm = p - p.mean(0)

    A[::2, 0] = 1/np.sqrt(n)     # dx
    A[1::2, 1] = 1/np.sqrt(n)    # dy
    A[::2, 2] = -r_cm[:, 1]
    A[1::2, 2] = r_cm[:, 0]
    A[:, 2] /= np.sqrt(np.square(r_cm).sum())  # dt

    P = A.T.dot(A)
    S = np.eye(s) - P
    return P, S


def pose_and_shape_projections_aa(p):
    """ Devuelve dos matrices de proyección.

    P proyecta al subespacio "pose",
    S proyecta al subespacio "shape".

    args:
        p: array de posiciones (..., n, dof)

    returns
        P, S: matrices (..., n * dof, n * dof)

    """
    N, n, d = p.shape
    s = n * d
    A = np.zeros((N, s, 3))     # 3 si 2d, 6 si 3d
    r_cm = p - p.mean(1)[:, None]

    A[:, ::2, 0] = 1/np.sqrt(n)     # dx
    A[:, 1::2, 1] = 1/np.sqrt(n)    # dy
    A[:, ::2, 2] = -r_cm[:, :, 1]
    A[:, 1::2, 2] = r_cm[:, :, 0]
    r_cm = r_cm.reshape(N, s)
    A[:, :, 2] /= np.sqrt(np.square(r_cm).sum(1))[:, None]  # dt

    A_T = A.swapaxes(-2, -1)
    P = np.matmul(A, A_T)
    S = np.eye(s) - P
    return P, S
