#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
"""
import numpy as np

from uvnpy.angles.local_frame.core import angle_rigidity_matrix
from uvnpy.toolkit.functions import cosine_activation, cosine_activation_derivative


def complete_angle_set(out_neighbors):
    i, j = np.triu_indices(out_neighbors.size, k=1)
    return np.column_stack([out_neighbors[i], out_neighbors[j]]).astype(int).tolist()


def sigma_r_r(x, range_lims):
    return cosine_activation(
        np.array([x]), range_lims[0], range_lims[1]
    ).item()


def dsigma_r_r(x, range_lims):
    return cosine_activation_derivative(
        np.array([x]), range_lims[0], range_lims[1]
    ).item()


def sigma_r_f(x, hfov_lims):
    return cosine_activation(
        np.array([x]), hfov_lims[0], hfov_lims[1]
    ).item()


def dsigma_r_f(x, hfov_lims):
    return cosine_activation_derivative(
        np.array([x]), hfov_lims[0], hfov_lims[1]
    ).item()


def weight(indices, p, R, range_lims, hfov_lims):
    i, j, k = indices
    dij = np.sqrt(np.square(p[j] - p[i]).sum())
    bij = (p[j] - p[i]) / dij
    dik = np.sqrt(np.square(p[k] - p[i]).sum())
    bik = (p[k] - p[i]) / dik

    ei = R[i, :, 0]
    nij = ei.dot(bij)
    nik = ei.dot(bik)

    wrij = 1 - sigma_r_r(dij, range_lims)
    wfij = sigma_r_f(nij, hfov_lims)
    wrik = 1 - sigma_r_r(dik, range_lims)
    wfik = sigma_r_f(nik, hfov_lims)

    return dij * dik * wrij * wfij * wrik * wfik


def weights(angle_set, p, R, range_lims, hfov_lims):
    return np.array([
        weight(idx, p, R, range_lims, hfov_lims) for idx in angle_set
    ])


def weighted_angle_rigidity_eigenvalue(angle_set, p, R, range_lims, hfov_lims):
    A = angle_rigidity_matrix(angle_set, p)
    W = weights(angle_set, p, R, range_lims, hfov_lims)
    evals, evecs = np.linalg.eigh(A.T.dot(W[:, np.newaxis] * A))
    return evals[7], evecs[:, 7].reshape(p.shape)


class AngleRigidityMaintenance(object):
    """
    Rigidity Maintenance Control based on the minimization
    of the rigidity matrix (nonzero) eigenvalues' inverses.

    args:
    -----
        N           : number of robots
        range_lims  : range low and high limits
        cos_lims    : cosine low and high limits
    """
    def __init__(self, N, range_lims, hfov_lims):
        self.N = N
        self.range_lims = range_lims
        self.hfov_lims = hfov_lims

    def update(
            self,
            out_neighbors,
            distances, bearings,
            rotations,
            rigidity_val,
            rigidity_vec
            ):
        control_u_r = np.zeros((self.N, 3), dtype=np.float64)
        control_w_r = np.zeros((self.N, 3), dtype=np.float64)
        for i in range(self.N):
            for j, k in complete_angle_set(out_neighbors[i]):
                # --- rigidity control law --- #
                dij = distances[i][j]
                bij = bearings[i][j]
                Pij = np.eye(3) - np.outer(bij, bij)
                Rij = rotations[i][j]

                dik = distances[i][k]
                bik = bearings[i][k]
                Pik = np.eye(3) - np.outer(bik, bik)
                Rik = rotations[i][k]

                # --- weight part --- #
                nij = bij[0]
                nik = bik[0]

                wrij = 1 - sigma_r_r(dij, self.range_lims)
                wfij = sigma_r_f(nij, self.hfov_lims)
                wrik = 1 - sigma_r_r(dik, self.range_lims)
                wfik = sigma_r_f(nik, self.hfov_lims)

                d = dij * dik
                wr = wrij * wrik
                wf = wfij * wfik
                w = wr * wf

                wijk_j = w * dik * bij + d * (
                    - wf * wrik * dsigma_r_r(dij, self.range_lims) * bij
                    + wr * wfik * dsigma_r_f(nij, self.hfov_lims) * Pij[0] / dij
                )

                wijk_k = w * dij * bik + d * (
                    - wf * wrij * dsigma_r_r(dik, self.range_lims) * bik
                    + wr * wfij * dsigma_r_f(nik, self.hfov_lims) * Pik[0] / dik
                )
                wijk_i = - wijk_j - wijk_k

                e1_bij = np.array([0.0, -bij[2], bij[1]])
                e1_bik = np.array([0.0, -bik[2], bik[1]])
                wijk_Ri = d * wr * (
                    wfik * dsigma_r_f(nij, self.hfov_lims) * e1_bij
                    + wfij * dsigma_r_f(nik, self.hfov_lims) * e1_bik
                )

                wijk = d * w

                # --- rigidity matrix part --- #
                qijk = Pij.dot(bik) / dij
                qikj = Pik.dot(bij) / dik

                vec_i = rigidity_vec[i]
                vec_j = Rij.dot(rigidity_vec[j])
                vec_k = Rik.dot(rigidity_vec[k])

                sijk = qijk.dot(vec_j - vec_i) + qikj.dot(vec_k - vec_i)

                Dijk = np.outer(Pij.dot(bik), bij)
                Dijk += np.outer(bij, bik.dot(Pij))
                Dijk += bij.dot(bik) * Pij
                Dijk /= dij**2

                Dikj = np.outer(Pik.dot(bij), bik)
                Dikj += np.outer(bik, bij.dot(Pik))
                Dikj += bik.dot(bij) * Pik
                Dikj /= dik**2

                Eijk = Pij.dot(Pik) / (dij * dik)
                Eikj = Pik.dot(Pij) / (dik * dij)

                sijk_j = - Dijk.dot(vec_j - vec_i) + Eijk.dot(vec_k - vec_i)
                sijk_k = Eikj.dot(vec_j - vec_i) - Dikj.dot(vec_k - vec_i)
                sijk_i = - sijk_j - sijk_k

                control_u_r[i] += sijk * (sijk * wijk_i + 2 * wijk * sijk_i)
                control_u_r[j] += sijk * Rij.T.dot(sijk * wijk_j + 2 * wijk * sijk_j)
                control_u_r[k] += sijk * Rik.T.dot(sijk * wijk_k + 2 * wijk * sijk_k)

                control_w_r[i] += sijk**2 * wijk_Ri

        return control_u_r, control_w_r
