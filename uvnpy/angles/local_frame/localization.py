#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Angle-based pose localization in a common frame."""

from itertools import combinations

import numpy as np
from transformations import unit_vector

from uvnpy.dynamics.core import EulerIntegrator
from uvnpy.dynamics.lie_groups import EulerIntegratorOrtogonalGroup
from uvnpy.toolkit.geometry import wahba_problem


def _projection_matrix(x):
    return np.eye(3) - np.outer(x, x)


def _complete_angle_set(out_neighbors):
    i, j = np.triu_indices(out_neighbors.size, k=1)
    return np.column_stack([out_neighbors[i], out_neighbors[j]])


class AngleBasedPoseObserverCoupled(object):
    """First-order pose observer for a network of robots in 3D.

    Propagation is corrected by the negative gradients of angle, angle-rate,
    bearing, and optional distance costs. By default, both pose components
    use the full cost gradient. Flags can omit angle-rate and bearing position
    corrections independently, while retaining their orientation corrections.

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
        + k_beta * L_bearing + k_d * L_distance``. The distance weight ``k_d``
        defaults to zero; ``distance_pair`` controls whether the distance
        correction is evaluated.
    k_p, k_R : float
        Finite, strictly positive position and orientation gradient gains.
        With both position corrections enabled, corrections are
        ``-k_p * grad_position(L)`` and ``-k_R * grad_orientation(L)``.
        Equivalently, the product metric has
        position weight ``1 / k_p`` and orientation weight ``1 / k_R`` relative
        to the Euclidean and half-trace metrics, respectively. These gains
        multiply only the corrections, not the kinematic propagation.
    time : float
        Initial time in seconds.
    distance_pair : tuple of int, optional
        Fixed pair of distinct robot indices ``(a, b)`` measuring distance,
        independent of the directed bearing edge set. If None, the distance
        correction is disabled regardless of ``k_d``. Otherwise, a current
        distance measurement is required on every update, even if ``k_d == 0``.
    use_angle_rate_position : bool, optional
        Include the angle-rate position correction, default True. If False,
        skip that position gradient and its associated matrix calculations;
        the angle-rate orientation correction remains active.
    use_bearing_position : bool, optional
        Include the bearing position correction, default True. If False,
        the bearing orientation correction remains active. Neither flag
        affects the angle-based shape or distance-based scale corrections.

    Examples
    --------
    Initialize once, then supply measurements and directed edges at each step::

        observer = AngleBasedPoseObserverCoupled(n, hatp0, hatR0)
        hatp, hatR = observer.update(
            t, u, omega, bearings, bearing_rates, edge_set
        )

    To enable distance correction, set its weight and measuring pair once,
    then supply a current measured distance in metres on every update::

        observer = AngleBasedPoseObserverCoupled(
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
            k_p=10000.0, k_R=200.0, time=0.0, k_d=0.0, distance_pair=None,
            use_angle_rate_position=True, use_bearing_position=True):
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

        self.n = int(n)
        self.t = float(time)
        self.k_alpha = k_alpha
        self.k_dalpha = k_dalpha
        self.k_beta = k_beta
        self.k_d = k_d
        self.distance_pair = distance_pair
        self.k_p = k_p
        self.k_R = k_R
        self.use_angle_rate_position = use_angle_rate_position
        self.use_bearing_position = use_bearing_position
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
            positive on every update when ``distance_pair is not None``;
            ignored otherwise.

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
        if self.distance_pair is not None:
            if distance is None:
                raise ValueError(
                    'distance is required on every update when distance_pair is set.'
                )
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
        if self.distance_pair is not None:
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
                if self.use_bearing_position:
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

                # --- angle-rate-based correction --- #
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

                if self.use_angle_rate_position:
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


class AngleBasedPoseObserverCascade(object):
    """Angle-and-range position observer with bearing/velocity attitude correction.

    Positions and reconstructed velocities are expressed in a common estimation
    frame. Only the downstream attitude correction uses the estimated rotations.
    Robot indices are zero-based. The fixed range pair defaults to (0, 1) and
    need not be a bearing edge. All four innovation gains are strictly positive.

    On each update, robots with at least two outgoing bearings are treated as
    rich. Their Wahba rotations map body bearings to estimated inter-robot
    directions. Weak robots reconstruct velocity from incoming rich robots.
    This requires adequate bearing geometry for both reconstructions.

    As in the coupled observer, update(time, ...) advances using the previously
    cached derivatives and stores the new ones. Call update at the initial
    time to prime the integrators. The returned poses and properties are copies.
    """

    def __init__(
            self, n, position, orientation, k_alpha=500.0, k_d=0.001,
            k_beta=2.0, k_v=2.0, distance_pair=(0, 1), time=0.0):
        position = np.asarray(position, dtype=float)
        orientation = np.asarray(orientation, dtype=float)
        if int(n) != n or n < 2:
            raise ValueError('n must be an integer of at least two.')
        if position.shape != (n, 3) or orientation.shape != (n, 3, 3):
            raise ValueError('Expected position (n, 3) and orientation (n, 3, 3).')
        if not np.isfinite(position).all() or not np.isfinite(orientation).all():
            raise ValueError('Initial poses must be finite.')
        pair = np.asarray(distance_pair)
        if (pair.shape != (2,) or not np.issubdtype(pair.dtype, np.integer)
                or np.any(pair < 0) or np.any(pair >= n) or pair[0] == pair[1]):
            raise ValueError('distance_pair must contain two distinct robot indices.')
        for name, value in (('k_alpha', k_alpha), ('k_d', k_d), ('k_beta', k_beta),
                            ('k_v', k_v)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and strictly positive.')
        self.n = int(n)
        self.t = float(time)
        if not np.isfinite(self.t):
            raise ValueError('Initial time must be finite.')
        self.distance_pair = tuple(int(idx) for idx in pair)
        self.k_alpha, self.k_d = k_alpha, k_d
        self.k_beta, self.k_v = k_beta, k_v
        self._p_int = [EulerIntegrator(pi, t=self.t) for pi in position]
        self._R_int = [
            EulerIntegratorOrtogonalGroup(Ri, t=self.t) for Ri in orientation
        ]

    @property
    def position(self):
        return np.array([integrator.x() for integrator in self._p_int])

    @property
    def orientation(self):
        return np.array([integrator.x() for integrator in self._R_int])

    def update(self, time, u, omega, bearings, bearing_rates, distance, out_neighbors):
        """Advance from one synchronized measurement snapshot.

        u and omega are (n, 3) body-frame velocities. out_neighbors maps each
        source robot ID to a one-dimensional list or array of target robot IDs.
        A missing source has no outgoing edges. bearings[i][j] is a unit
        body-frame vector and bearing_rates[i][j] its body-frame derivative.
        Rates are read only on rich-to-weak edges. Both nested mappings and
        arrays are supported for the measurements. distance is the current
        positive range of distance_pair in metres.
        """
        time = float(time)
        if not np.isfinite(time) or time < self.t:
            raise ValueError('time must be finite and nondecreasing.')
        ub, wb = np.asarray(u, dtype=float), np.asarray(omega, dtype=float)
        if ub.shape != (self.n, 3) or wb.shape != (self.n, 3):
            raise ValueError('u and omega must have shape (n, 3).')
        if not np.isfinite(ub).all() or not np.isfinite(wb).all():
            raise ValueError('u and omega must be finite.')
        distance = np.asarray(distance, dtype=float)
        if distance.ndim != 0 or not np.isfinite(distance) or distance <= 0:
            raise ValueError('distance must be a finite, strictly positive scalar.')
        hatp, hatR = self.position, self.orientation
        neighbors = []
        hatp_ij, hatd, hatb, beta = [], [], [], []
        tilde_T = np.empty_like(hatR)
        tilde_v = np.empty_like(hatp)
        rich = np.zeros(self.n, dtype=bool)

        # Group edge geometry by source and solve Wahba for rich robots.
        for i in range(self.n):
            outgoing = np.asarray(out_neighbors.get(i, ()))
            if (outgoing.ndim != 1
                    or (outgoing.size and not np.issubdtype(outgoing.dtype, np.integer))
                    or np.any(outgoing < 0) or np.any(outgoing >= self.n)
                    or np.any(outgoing == i)):
                raise ValueError(f'Invalid outgoing neighbor IDs for robot {i}.')
            outgoing = outgoing.astype(int, copy=False)
            neighbors.append(outgoing)

            relative = hatp[outgoing] - hatp[i]
            distances = np.linalg.norm(relative, axis=1)
            if np.any(distances == 0):
                raise ValueError('An estimated bearing edge has zero length.')
            directions = relative / distances[:, None]
            body = np.empty((0, 3))
            if outgoing.size:
                body = np.asarray([bearings[i][j] for j in outgoing], dtype=float)
            if body.shape != directions.shape or not np.allclose(
                    np.linalg.norm(body, axis=1), 1):
                raise ValueError('Each active bearing must be a unit 3-vector.')
            hatp_ij.append(relative)
            hatd.append(distances)
            hatb.append(directions)
            beta.append(body)

            if outgoing.size >= 2:
                tilde_T[i] = wahba_problem(body, directions)
                tilde_v[i] = tilde_T[i].dot(ub[i])
                rich[i] = True

        # Accumulate incoming contributions in one pass over rich sources.
        M = np.zeros_like(hatR)
        x = np.zeros_like(hatp)
        for j in np.flatnonzero(rich):
            for edge, i in enumerate(neighbors[j]):
                if rich[i]:
                    continue
                b = beta[j][edge]
                db = np.asarray(bearing_rates[j][i], dtype=float)
                if db.shape != (3,) or not np.isfinite(db).all():
                    raise ValueError(
                        'Each required bearing rate must be a finite 3-vector.'
                    )
                M[i] += _projection_matrix(tilde_T[j].dot(b))
                z = (_projection_matrix(b).dot(ub[j])
                     + hatd[j][edge] * (np.cross(wb[j], b) + db))
                x[i] += tilde_T[j].dot(z)
        for i in np.flatnonzero(~rich):
            tilde_v[i] = np.linalg.solve(M[i], x[i])

        # Position innovations: fixed-pair range and outgoing angle cosines.
        corr_p = np.zeros_like(hatp)

        # distance correction
        a, b = self.distance_pair
        r = hatp[b] - hatp[a]
        c = self.k_d * (r.dot(r) - distance**2) * r
        corr_p[a] += c
        corr_p[b] -= c

        # angle correction
        for i, outgoing in enumerate(neighbors):
            for (ej, j), (ek, k) in combinations(enumerate(outgoing), 2):
                bij, bik = hatb[i][ej], hatb[i][ek]
                residual = bij.dot(bik) - beta[i][ej].dot(beta[i][ek])
                c_j = (self.k_alpha * residual * _projection_matrix(bij).dot(bik)
                       / hatd[i][ej])
                c_k = (self.k_alpha * residual * _projection_matrix(bik).dot(bij)
                       / hatd[i][ek])
                corr_p[i] += c_j + c_k
                corr_p[j] -= c_j
                corr_p[k] -= c_k

        # Both attitude innovations apply to every robot, in its body frame.
        # velocity correction
        corr_R = self.k_v * np.cross(ub, np.einsum('nji,nj->ni', hatR, tilde_v))
        # bearing correction
        for i in range(self.n):
            body_relative = hatp_ij[i].dot(hatR[i])
            corr_R[i] += self.k_beta * np.cross(beta[i], body_relative).sum(axis=0)

        for i in range(self.n):
            self._p_int[i].step(time, tilde_v[i] + corr_p[i])
            self._R_int[i].step_left(time, wb[i] + corr_R[i])
        self.t = time

        return self.position, self.orientation
