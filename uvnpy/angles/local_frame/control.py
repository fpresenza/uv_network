#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Francisco Presenza
@institute LAR - FIUBA, Universidad de Buenos Aires, Argentina
"""
import numpy as np
from itertools import combinations

from uvnpy.angles.local_frame.core import angle_rigidity_matrix
from uvnpy.toolkit.functions import cosine_activation, cosine_activation_derivative


def complete_angle_set(out_neighbors):
    return [[int(a), int(b)] for a, b in combinations(out_neighbors, 2)]


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
    """Return modes 8 through 3N of the weighted 3D rigidity matrix.

    Eigenvalues are ascending and eigenvectors are columns. The first
    seven modes correspond to the similarity-motion nullspace.
    """
    A = angle_rigidity_matrix(angle_set, p)
    W = weights(angle_set, p, R, range_lims, hfov_lims)
    evals, evecs = np.linalg.eigh(A.T.dot(W[:, np.newaxis] * A))
    return evals[7:], evecs[:, 7:]


class AngleRigidityMaintenance(object):
    """Negative gradient of the log barrier on the nontrivial spectrum.

    The potential is -sum(log(rigidity_vals - epsilon_lambda)).
    epsilon_lambda defaults to zero and must be below the smallest retained
    eigenvalue throughout operation. N is the number of robots and range_lims
    and hfov_lims define the distance and field-of-view weight transitions.
    """

    def __init__(self, N, range_lims, hfov_lims, epsilon_lambda=0.0):
        epsilon_lambda = float(epsilon_lambda)
        if not np.isfinite(epsilon_lambda) or epsilon_lambda < 0:
            raise ValueError('epsilon_lambda must be finite and nonnegative.')
        self.N = N
        self.range_lims = range_lims
        self.hfov_lims = hfov_lims
        self.epsilon_lambda = epsilon_lambda

    def update(
            self,
            out_neighbors,
            distances,
            bearings,
            rotations,
            rigidity_vals,
            rigidity_vecs
            ):
        """Return body-frame commands from the retained weighted modes.

        rigidity_vals has shape (3*N - 7,). rigidity_vecs has shape
        (3*N - 7, N, 3), with each robot's block in its own body frame.
        """
        rigidity_vals = np.asarray(rigidity_vals, dtype=float)
        rigidity_vecs = np.asarray(rigidity_vecs, dtype=float)
        n_modes = 3 * self.N - 7
        if (n_modes <= 0 or rigidity_vals.shape != (n_modes,)
                or rigidity_vecs.shape != (n_modes, self.N, 3)):
            raise ValueError(
                'Expected modes 8 through 3N and their body-frame vectors.'
            )
        gaps = rigidity_vals - self.epsilon_lambda
        if not np.all(np.isfinite(gaps) & (gaps > 0)):
            raise ValueError('Rigidity barrier requires all retained eigenvalues '
                             'to be strictly greater than epsilon_lambda.')

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

                Dijk = np.outer(Pij.dot(bik), bij)
                Dijk += np.outer(bij, bik.dot(Pij))
                Dijk += bij.dot(bik) * Pij
                Dijk /= dij**2

                Dikj = np.outer(Pik.dot(bij), bik)
                Dikj += np.outer(bik, bij.dot(Pik))
                Dikj += bik.dot(bij) * Pik
                Dikj /= dik**2

                Eijk = Pij.dot(Pik) / (dij * dik)
                # Work with all mode rows in the source robot's body frame.
                vec_i = rigidity_vecs[:, i]
                diff_j = rigidity_vecs[:, j] @ Rij.T - vec_i
                diff_k = rigidity_vecs[:, k] @ Rik.T - vec_i
                sijk = diff_j @ qijk + diff_k @ qikj

                # Sum modal contributions before applying the common gradients.
                weighted_s = sijk / gaps
                weight_factor = sijk @ weighted_s
                sum_diff_j = weighted_s @ diff_j
                sum_diff_k = weighted_s @ diff_k
                sum_grad_j = -Dijk @ sum_diff_j + Eijk @ sum_diff_k
                sum_grad_k = Eijk.T @ sum_diff_j - Dikj @ sum_diff_k
                sum_grad_i = -sum_grad_j - sum_grad_k

                control_u_r[i] += weight_factor * wijk_i + 2 * wijk * sum_grad_i
                control_u_r[j] += Rij.T @ (
                    weight_factor * wijk_j + 2 * wijk * sum_grad_j
                )
                control_u_r[k] += Rik.T @ (
                    weight_factor * wijk_k + 2 * wijk * sum_grad_k
                )
                control_w_r[i] += weight_factor * wijk_Ri

        return control_u_r, control_w_r
