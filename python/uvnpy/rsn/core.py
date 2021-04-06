#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
@date mar ene 26 18:33:51 -03 2021
"""
import numpy as np
import scipy.linalg
from transformations import unit_vector


def distances(p):
    """ Devuelve matriz de distancias.

    args:
        p: array de posiciones (n, dof)
    """
    r = p[:, None] - p
    dist = np.sqrt(np.square(r).sum(axis=2))
    return dist


def distances_aa(p):
    """ Devuelve matriz de distancias.

    args:
        p: array de posiciones (-1, n, dof)
    """
    r = p[..., None, :] - p[..., None, :, :]
    dist = np.sqrt(np.square(r).sum(axis=-1))
    return dist


def distances_from_edges(E, p):
    r = p[E[:, 0]] - p[E[:, 1]]
    dist = np.sqrt(np.square(r).sum(axis=-1))
    return dist


def distances_from_incidence(Dr, p):
    r = Dr.T.dot(p)
    dist = np.sqrt(np.square(r).sum(axis=-1))
    return dist


def distances_jac(A, p):
    nv, dof = p.shape
    r = unit_vector(p[:, None] - p, axis=-1)
    ii = np.diag([True] * nv)
    r[ii] = 0
    r *= A[..., None]               # aplicar pesos
    E = np.argwhere(np.triu(A) != 0)
    Ef = np.flip(E, axis=1)
    ne = len(E)
    J = np.zeros((ne, nv, dof))
    i = np.arange(ne).reshape(-1, 1)
    J[i, E] = r[E, Ef]
    J = J.reshape(ne, nv * dof)
    return J


# def distances_jac_aa(A, p):
#     N, nv, dof = p.shape[-3:]
#     r = unit_vector(p[..., None, :] - p[..., None, :, :], axis=-1)
#     ii = np.diag([True] * nv)
#     r[:, ii] = 0
#     r *= A[..., None]               # aplicar pesos
#     E = np.argwhere(np.triu(A) != 0)
#     Ef = np.flip(E, axis=1)
#     ne = len(E)
#     J = np.empty((N, ne, nv,  dof))
#     i = np.arange(ne).reshape(-1, 1)
#     J[:, i, E] = r[:, E, Ef]
#     J = J.reshape(N, ne, nv * dof)
#     return J


def distances_jac_from_incidence(Dr, p):
    nv, dof = p.shape
    ne = Dr.shape[1]
    J = np.zeros((ne, dof * nv))
    eye = np.eye(dof)
    r = unit_vector(Dr.T.dot(p), axis=-1)
    M = scipy.linalg.block_diag(*r)
    Dr = np.kron(Dr, eye)
    J = M.dot(Dr.T)
    return J


def positions_jac(nv, loops, dof):
    Dp = np.zeros((len(loops), nv))
    Dp[range(len(loops)), loops] = 1
    J = np.kron(Dp, np.eye(dof))
    return J


def distances_innovation(A, p):
    """ Laplaciano de innovación.

    Devuelve la matriz de innovación

            Y =  H^T W H

    del modelo de distancias de un grafo de nv agentes
    determinado por la matriz de adyacencia A.

    Si A[i, i] > 0 el nodo i tiene medicion de posición
    Si A[i, j] > 0 los nodos i, j tienen medicion de distancia

    args:
        A: matriz de adyacencia (nv, nv)
        p: array de posiciones (nv, dof)

    returns
        Y: matriz de innovación (nv * dof, nv * dof)
    """
    nv, dof = p.shape
    r = unit_vector(p[:, None] - p, axis=2)
    Y = - r[..., None] * r[..., None, :]  # outer product
    ii = np.diag([True] * nv)
    Y[ii] = np.eye(dof)
    Y *= A[..., None, None]               # aplicar pesos
    Y[ii] -= Y[~ii].reshape(nv, nv - 1, dof, dof).sum(1)
    Y = Y.swapaxes(-3, -2).reshape(nv * dof, nv * dof)
    return Y


def distances_innovation_aa(A, p):
    """ Matriz de innovación.

    Devuelve la matriz de innovación

            Y =  H^T W H

    del modelo de distancias de un grafo de nv agentes
    determinado por la matriz de adyacencia A.

    Si A[i, i] > 0 el nodo i tiene medicion de posición
    Si A[i, j] > 0 los nodos i, j tienen medicion de distancia

    args:
        A: matriz de adyacencia (N, nv, nv)
        p: array de posiciones (N, nv, dof)

    returns
        Y: innovacion (-1, nv * dof, nv * dof)
    """
    nv, dof = p.shape[-2:]
    r = unit_vector(p[..., None, :] - p[..., None, :, :], axis=-1)
    Y = - r[..., None] * r[..., None, :]  # outer product
    ii = np.diag([True] * nv)
    Y[:, ii] = np.eye(dof)
    Y *= A[..., None, None]               # aplicar pesos
    Y[:, ii] -= Y[:, ~ii].reshape(-1, nv, nv - 1, dof, dof).sum(p.ndim - 1)
    Y = Y.swapaxes(-3, -2).reshape(-1, nv * dof, nv * dof)
    return Y


def distances_innovation_diag_aa(A, p):
    """ Diagonal por bloques de la Matriz de innovación.

    args:
        A: matriz de adyacencia (N, nv, nv)
        p: array de posiciones (N, nv, dof)

    returns
        diag: diagonales (-1, nv, dof, dof)
    """
    nv, dof = p.shape[-2:]
    r = unit_vector(p[..., None, :] - p[..., None, :, :], axis=-1)
    Y = r[..., None] * r[..., None, :]  # outer product
    ii = np.diag([True] * nv)
    Y[:, ii] = 0
    Y *= A[..., None, None]
    diag = Y.sum(1)
    return diag


def pose_and_shape_basis_2d(p):
    """ Devuelve dos matrices de proyección.

    P proyecta al subespacio "pose",
    S proyecta al subespacio "shape".

    args:
        p: array de posiciones (nv, dof)

    returns
        P, S: matrices (nv * dof, nv * dof)

    """
    n, d = p.shape
    s = p.size
    A = np.zeros((s, 3))     # 3 si 2d, 6 si 3d
    B = np.empty((s, s - 3))
    d_cm = p - p.mean(0)

    A[::2, 0] = 1/np.sqrt(n)     # dx
    A[1::2, 1] = 1/np.sqrt(n)    # dy
    A[::2, 2] = -d_cm[:, 1]
    A[1::2, 2] = d_cm[:, 0]
    A[:, 2] /= np.sqrt(np.square(d_cm).sum())  # dt

    B = scipy.linalg.null_space(A.T)
    return A, B


def pose_and_shape_basis_2d_aa(p):
    """ Devuelve dos matrices de proyección.

    P proyecta al subespacio "pose",
    S proyecta al subespacio "shape".

    args:
        p: array de posiciones (N, nv, dof)

    returns
        P, S: matrices (N, nv * dof, nv * dof)

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


def pose_and_shape_projections_2d(p):
    """ Devuelve dos matrices de proyección.

    P proyecta al subespacio "pose",
    S proyecta al subespacio "shape".

    args:
        p: array de posiciones (nv, dof)

    returns
        P, S: matrices (nv * dof, nv * dof)

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


def pose_and_shape_projections_2d_aa(p):
    """ Devuelve dos matrices de proyección.

    P proyecta al subespacio "pose",
    S proyecta al subespacio "shape".

    args:
        p: array de posiciones (N, nv, dof)

    returns
        P, S: matrices (N, nv * dof, nv * dof)

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
