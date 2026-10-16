#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Angle-based pose localization in a common frame."""

import numpy as np
from transformations import unit_vector

from uvnpy.dynamics.core import EulerIntegrator
from uvnpy.dynamics.lie_groups import EulerIntegratorOrtogonalGroup


def _projection_matrix(x):
    return np.eye(3) - np.outer(x, x)


def _complete_angle_set(out_neighbors):
    i, j = np.triu_indices(out_neighbors.size, k=1)
    return np.column_stack([out_neighbors[i], out_neighbors[j]])


class AngleBasedPoseObserver(object):
    """First-order pose observer for a network of robots in 3D.

    This extracts the observer from ``tests/angles/anl/simu_coupled_3.py``.
    It keeps the angle-based shape correction, angle-rate scale correction,
    and bearing/angle-rate orientation corrections unchanged.

    Parameters
    ----------
    n : int
        Number of robots, indexed from 0 to n - 1.
    position : array_like, shape (n, 3)
        Initial estimated positions in the common estimation frame.
    orientation : array_like, shape (n, 3, 3)
        Initial estimated rotations from body frames to the estimation frame.
    k_a, k_s, k_o1, k_o2 : float
        Gains for shape, scale, bearing orientation, and angle-rate orientation
        corrections, respectively. Defaults match the simulation script.
    time : float
        Initial time in seconds.

    Examples
    --------
    Initialize once, then supply measurements and directed edges at each step::

        observer = AngleBasedPoseObserver(n, hatp0, hatR0)
        hatp, hatR = observer.update(
            t, u, omega, bearings, bearing_rates, edge_set
        )

    ``position`` and ``orientation`` properties return copies of the stored
    estimates. No true poses or measured distances are used by this class.
    """

    def __init__(
            self, n, position, orientation,
            k_a=2000.0, k_s=10000.0, k_o1=2.0, k_o2=200.0, time=0.0):
        position = np.asarray(position, dtype=float)
        orientation = np.asarray(orientation, dtype=float)
        if n < 1 or int(n) != n:
            raise ValueError('n must be a positive integer.')
        if position.shape != (n, 3):
            raise ValueError('position must have shape (n, 3).')
        if orientation.shape != (n, 3, 3):
            raise ValueError('orientation must have shape (n, 3, 3).')

        self.n = int(n)
        self.t = float(time)
        self.k_a = k_a
        self.k_s = k_s
        self.k_o1 = k_o1
        self.k_o2 = k_o2
        self._p_int = [EulerIntegrator(pi, t=self.t) for pi in position]
        self._R_int = [
            EulerIntegratorOrtogonalGroup(Ri, t=self.t) for Ri in orientation
        ]

    @property
    def position(self):
        """Current estimated positions, shape (n, 3), as a copy."""
        return np.array([integrator.x() for integrator in self._p_int])

    @property
    def orientation(self):
        """Current estimated orientations, shape (n, 3, 3), as a copy."""
        return np.array([integrator.x() for integrator in self._R_int])

    def update(self, time, u, omega, bearings, bearing_rates, edge_set):
        """Compute corrections and advance the stored pose estimates.

        Parameters
        ----------
        time : float
            Current time in seconds, not earlier than the previous update.
        u : array_like, shape (n, 3)
            Body-frame linear velocities of all robots.
        omega : array_like, shape (n, 3)
            Body-frame angular velocities of all robots.
        bearings, bearing_rates : nested mappings or ndarray
            ``bearings[i][j]`` is the measured unit bearing from i to j,
            expressed in i's body frame; ``bearing_rates[i][j]`` is its time
            derivative in that frame. Mapping entries are array-like vectors
            of length 3; arrays have shape (n, n, 3). Only edges in ``edge_set``
            need measurement entries.
        edge_set : array_like, shape (m, 2)
            Current directed sensing edges. Each row (i, j) means that robot i
            observes robot j, with distinct endpoints indexed from 0 to n - 1.
            Edges must be unique. All unordered pairs of outgoing neighbors
            define measured angles centered at i. The edge set may change at
            every update; the order of each robot's outgoing edges is preserved.
            An empty edge set, either [] or an array of shape (0, 2), is allowed.

        Returns
        -------
        position, orientation : ndarray
            Copies of the updated estimates, shapes (n, 3) and (n, 3, 3).

        Notes
        -----
        Matches the simulation's previous-input Euler stepping: corrections
        use a snapshot of the stored estimates, the integrators advance using
        their previously cached inputs, and the new inputs are cached for the
        next step. The initial cached inputs are zero. An update at the initial
        time primes them without advancing the estimates.
        """
        time = float(time)
        if not np.isfinite(time) or time < self.t:
            raise ValueError('time must be finite and nondecreasing.')
        ub = np.asarray(u, dtype=float)
        wb = np.asarray(omega, dtype=float)
        if ub.shape != (self.n, 3) or wb.shape != (self.n, 3):
            raise ValueError('u and omega must have shape (n, 3).')
        E = np.asarray(edge_set, dtype=int)
        if E.ndim == 1 and E.size == 0:
            E = E.reshape(0, 2)
        if E.ndim != 2 or E.shape[1] != 2:
            raise ValueError('edge_set must have shape (m, 2).')

        hatp = self.position
        hatR = self.orientation
        corr_p = np.zeros((self.n, 3), dtype=np.float64)
        corr_R = np.zeros((self.n, 3), dtype=np.float64)
        buffer = np.zeros((self.n, 3), dtype=np.float64)

        for i in range(self.n):
            out_neighbors = E[:, 1][E[:, 0] == i]
            hat_distances = {
                j: np.sqrt(np.square(hatp[j] - hatp[i]).sum())
                for j in out_neighbors
            }
            hat_bearings = {
                j: unit_vector(hatp[j] - hatp[i]) for j in out_neighbors
            }
            bearings_i = {
                j: np.asarray(bearings[i][j], dtype=float)
                for j in out_neighbors
            }
            bearing_rates_i = {
                j: np.asarray(bearing_rates[i][j], dtype=float)
                for j in out_neighbors
            }

            # --- bearing-based orientation correction --- #
            for j in out_neighbors:
                corr_R[i] += self.k_o1 * np.cross(
                    bearings_i[j], hatR[i].T.dot(hatp[j] - hatp[i])
                )

            for j, k in _complete_angle_set(out_neighbors):
                dij = hat_distances[j]
                bij = hat_bearings[j]
                Pij = _projection_matrix(bij)

                dik = hat_distances[k]
                bik = hat_bearings[k]
                Pik = _projection_matrix(bik)

                Xijk = Pij.dot(bik) / dij
                Xikj = Pik.dot(bij) / dik

                # --- angle-based shape correction --- #
                aijk = bearings_i[j].dot(bearings_i[k])
                hat_aijk = bij.dot(bik)
                e_aijk = hat_aijk - aijk

                c_j = self.k_a * e_aijk * Xijk
                c_k = self.k_a * e_aijk * Xikj
                corr_p[i] += c_j + c_k
                corr_p[j] -= c_j
                corr_p[k] -= c_k

                # --- angle-rate-based scale correction --- #
                hat_v_i = hatR[i].dot(ub[i])
                hat_v_j = hatR[j].dot(ub[j])
                hat_v_k = hatR[k].dot(ub[k])

                hat_v_ij = hat_v_j - hat_v_i
                hat_v_ik = hat_v_k - hat_v_i

                dot_aijk = (
                    bearings_i[k].dot(bearing_rates_i[j])
                    + bearings_i[j].dot(bearing_rates_i[k])
                )
                dot_hat_aijk = Xijk.dot(hat_v_ij) + Xikj.dot(hat_v_ik)
                e_dot_aijk = dot_hat_aijk - dot_aijk

                Dijk = (
                    hat_aijk * Pij / dij**2
                    + (np.outer(Xijk, bij) + np.outer(bij, Xijk)) / dij
                )
                Dikj = (
                    hat_aijk * Pik / dik**2
                    + (np.outer(Xikj, bik) + np.outer(bik, Xikj)) / dik
                )
                Eijk = Pij.dot(Pik) / (dij * dik)

                c_j = self.k_s * e_dot_aijk * (
                    Eijk.dot(hat_v_ik) - Dijk.dot(hat_v_ij)
                )
                c_k = self.k_s * e_dot_aijk * (
                    Eijk.T.dot(hat_v_ij) - Dikj.dot(hat_v_ik)
                )
                corr_p[i] += c_j + c_k
                corr_p[j] -= c_j
                corr_p[k] -= c_k

                # --- angle-rate-based orientation correction --- #
                c_j = self.k_o2 * (dot_hat_aijk - dot_aijk) * Xijk
                c_k = self.k_o2 * (dot_hat_aijk - dot_aijk) * Xikj
                buffer[i] += c_j + c_k
                buffer[j] -= c_j
                buffer[k] -= c_k

        for i in range(self.n):
            corr_R[i] += np.cross(ub[i], hatR[i].T.dot(buffer[i]))
            self._p_int[i].step(time, hatR[i].dot(ub[i]) + corr_p[i])
            self._R_int[i].step_left(time, wb[i] + corr_R[i])

        self.t = time
        return self.position, self.orientation
