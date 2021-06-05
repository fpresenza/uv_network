#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar abr  6 15:11:20 -03 2021
"""
import numpy as np
import scipy.linalg


def pose_and_shape_decomposition(p):
    """Devuelve una matriz ortogonal de cambio de base.

    Se desocompone en subespacio "pose" y subespacio "shape".

    args:
        p: array de posiciones (n, dof)

    returns
        M = [P; S]: matriz (n * dof, n * dof)
    """
    n = len(p)
    s = p.size
    M = np.zeros((s, s))
    d_cm = p - p.mean(0)

    M[::2, 0] = 1/np.sqrt(n)                    # dx
    M[1::2, 1] = 1/np.sqrt(n)                   # dy
    M[::2, 2] = -d_cm[:, 1]
    M[1::2, 2] = d_cm[:, 0]
    M[:, 2] /= np.sqrt(np.square(d_cm).sum())   # dt

    M[:, 3:] = scipy.linalg.null_space(M[:, :3].T)
    return M


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
    d_cm = p - p.mean(1)[:, None]

    A[:, ::2, 0] = 1/np.sqrt(n)     # dx
    A[:, 1::2, 1] = 1/np.sqrt(n)    # dy
    A[:, ::2, 2] = -d_cm[:, :, 1]
    A[:, 1::2, 2] = d_cm[:, :, 0]
    d_cm = d_cm.reshape(N, s)
    A[:, :, 2] /= np.sqrt(np.square(d_cm).sum(1))[:, None]  # dt

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
    d_cm = p - p.mean(0)

    A[::2, 0] = 1/np.sqrt(n)     # dx
    A[1::2, 1] = 1/np.sqrt(n)    # dy
    A[::2, 2] = -d_cm[:, 1]
    A[1::2, 2] = d_cm[:, 0]
    A[:, 2] /= np.sqrt(np.square(d_cm).sum())  # dt

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
    d_cm = p - p.mean(1)[:, None]

    A[:, ::2, 0] = 1/np.sqrt(n)     # dx
    A[:, 1::2, 1] = 1/np.sqrt(n)    # dy
    A[:, ::2, 2] = -d_cm[:, :, 1]
    A[:, 1::2, 2] = d_cm[:, :, 0]
    d_cm = d_cm.reshape(N, s)
    A[:, :, 2] /= np.sqrt(np.square(d_cm).sum(1))[:, None]  # dt

    A_T = A.swapaxes(-2, -1)
    P = np.matmul(A, A_T)
    S = np.eye(s) - P
    return P, S
