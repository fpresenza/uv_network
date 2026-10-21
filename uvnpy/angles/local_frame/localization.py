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

    Propagation is corrected by the negative gradients of angle, angle-rate,
    bearing, and optional distance costs.

    Parameters
    ----------
    n : int
        Number of robots, indexed from 0 to n - 1.
    position : array_like, shape (n, 3)
        Initial estimated positions in the common estimation frame.
    orientation : array_like, shape (n, 3, 3)
        Initial estimated rotations from body frames to the estimation frame.
    k_alpha, k_dalpha, k_beta, k_d : float
        Finite, nonnegative weights of the angle, angle-rate, bearing, and
        distance costs. Their weighted sum is the common correction cost
        ``L = k_alpha * L_angle + k_dalpha * L_angle_rate
        + k_beta * L_bearing + k_d * L_distance``. The distance weight ``k_d`` defaults
        to zero, disabling distance correction.
    k_p, k_R : float
        Finite, strictly positive position and orientation gradient gains.
        Corrections are ``-k_p * grad_position(L)`` and
        ``-k_R * grad_orientation(L)``. Equivalently, the product metric has
        position weight ``1 / k_p`` and orientation weight ``1 / k_R`` relative
        to the Euclidean and half-trace metrics, respectively. These gains
        multiply only the corrections, not the kinematic propagation.
    time : float
        Initial time in seconds.
    distance_pair : tuple of int, optional
        Fixed pair of distinct robot indices ``(a, b)`` measuring distance,
        independent of the directed bearing edge set. Required when ``k_d > 0``.

    Examples
    --------
    Initialize once, then supply measurements and directed edges at each step::

        observer = AngleBasedPoseObserver(n, hatp0, hatR0)
        hatp, hatR = observer.update(
            t, u, omega, bearings, bearing_rates, edge_set
        )

    To enable distance correction, set its weight and measuring pair once,
    then supply a current measured distance in metres on every update::

        observer = AngleBasedPoseObserver(
            n, hatp0, hatR0, k_d=0.001, distance_pair=(0, 1)
        )
        hatp, hatR = observer.update(
            t, u, omega, bearings, bearing_rates, edge_set, distance=d01
        )

    ``position`` and ``orientation`` properties return copies of the stored
    estimates. No true poses are used by this class.
    """

    def __init__(
            self, n, position, orientation,
            k_alpha=0.2, k_dalpha=1.0, k_beta=0.01,
            k_p=10000.0, k_R=200.0, time=0.0, k_d=0.0, distance_pair=None):
        position = np.asarray(position, dtype=float)
        orientation = np.asarray(orientation, dtype=float)
        if n < 1 or int(n) != n:
            raise ValueError('n must be a positive integer.')
        if position.shape != (n, 3):
            raise ValueError('position must have shape (n, 3).')
        if orientation.shape != (n, 3, 3):
            raise ValueError('orientation must have shape (n, 3, 3).')
        for name, weight in (
                ('k_alpha', k_alpha), ('k_dalpha', k_dalpha),
                ('k_beta', k_beta), ('k_d', k_d)):
            if not np.isfinite(weight) or weight < 0:
                raise ValueError(f'{name} must be finite and nonnegative.')
        for name, gain in (('k_p', k_p), ('k_R', k_R)):
            if not np.isfinite(gain) or gain <= 0:
                raise ValueError(f'{name} must be finite and strictly positive.')
        if distance_pair is not None:
            pair = np.asarray(distance_pair)
            if (pair.shape != (2,) or not np.issubdtype(pair.dtype, np.integer)
                    or np.any(pair < 0) or np.any(pair >= n) or pair[0] == pair[1]):
                raise ValueError(
                    'distance_pair must contain two distinct robot indices.'
                )
            distance_pair = tuple(int(index) for index in pair)
        elif k_d > 0:
            raise ValueError('distance_pair is required when k_d > 0.')

        self.n = int(n)
        self.t = float(time)
        self.k_alpha = k_alpha
        self.k_dalpha = k_dalpha
        self.k_beta = k_beta
        self.k_d = k_d
        self.distance_pair = distance_pair
        self.k_p = k_p
        self.k_R = k_R
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

    def update(self, time, u, omega, bearings, bearing_rates, edge_set, distance=None):
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
        distance : float, optional
            Current measured distance in metres between the robots in
            ``distance_pair``, not its square. Must be finite and strictly
            positive on every update when ``k_d > 0``; ignored otherwise.

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
        if self.k_d > 0:
            if distance is None:
                raise ValueError('distance is required on every update when k_d > 0.')
            distance = np.asarray(distance, dtype=float)
            if distance.ndim != 0 or not np.isfinite(distance) or distance <= 0:
                raise ValueError('distance must be a finite, strictly positive scalar.')
            distance = float(distance)

        hatp = self.position
        hatR = self.orientation
        corr_p = np.zeros((self.n, 3), dtype=np.float64)
        corr_R = np.zeros((self.n, 3), dtype=np.float64)
        buffer = np.zeros((self.n, 3), dtype=np.float64)

        # --- distance-based scale correction for the fixed measuring pair --- #
        if self.k_d > 0:
            a, b = self.distance_pair
            r = hatp[b] - hatp[a]
            c = self.k_d * (r.dot(r) - distance**2) * r
            corr_p[a] += c
            corr_p[b] -= c

        for i in range(self.n):
            out_neighbors = E[:, 1][E[:, 0] == i]
            hat_distances = {
                j: np.sqrt(np.square(hatp[j] - hatp[i]).sum())
                for j in out_neighbors
            }
            hat_bearings = {
                j: unit_vector(hatp[j] - hatp[i]) for j in out_neighbors
            }

            # --- bearing-based position/orientation corrections --- #
            for j in out_neighbors:
                bij = hat_bearings[j]
                c = self.k_beta * _projection_matrix(bij).dot(
                    hatR[i].dot(bearings[i][j])
                ) / hat_distances[j]
                corr_p[i] -= c
                corr_p[j] += c
                corr_R[i] += self.k_beta * np.cross(
                    bearings[i][j], hatR[i].T.dot(bij)
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
                aijk = np.dot(bearings[i][j], bearings[i][k])
                hat_aijk = bij.dot(bik)
                e_aijk = hat_aijk - aijk

                c_j = self.k_alpha * e_aijk * Xijk
                c_k = self.k_alpha * e_aijk * Xikj
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
                    np.dot(bearings[i][k], bearing_rates[i][j])
                    + np.dot(bearings[i][j], bearing_rates[i][k])
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

                c_j = self.k_dalpha * e_dot_aijk * (
                    Eijk.dot(hat_v_ik) - Dijk.dot(hat_v_ij)
                )
                c_k = self.k_dalpha * e_dot_aijk * (
                    Eijk.T.dot(hat_v_ij) - Dikj.dot(hat_v_ik)
                )
                corr_p[i] += c_j + c_k
                corr_p[j] -= c_j
                corr_p[k] -= c_k

                # --- angle-rate-based orientation correction --- #
                c_j = self.k_dalpha * e_dot_aijk * Xijk
                c_k = self.k_dalpha * e_dot_aijk * Xikj
                buffer[i] += c_j + c_k
                buffer[j] -= c_j
                buffer[k] -= c_k

        for i in range(self.n):
            corr_R[i] += np.cross(ub[i], hatR[i].T.dot(buffer[i]))
            self._p_int[i].step(time, hatR[i].dot(ub[i]) + self.k_p * corr_p[i])
            self._R_int[i].step_left(time, wb[i] + self.k_R * corr_R[i])

        self.t = time
        return self.position, self.orientation
