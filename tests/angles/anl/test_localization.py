"""Regression checks for the pose observer and its simulation integration."""

import ast
from pathlib import Path
import unittest

import numpy as np
from transformations import unit_vector

from uvnpy.angles.local_frame.localization import AngleBasedPoseObserver
from uvnpy.dynamics.core import EulerIntegrator
from uvnpy.dynamics.lie_groups import EulerIntegratorOrtogonalGroup
from uvnpy.toolkit.geometry import rotation_matrix_from_vector


def simulation_step():
    """Load the simulation functions without running its loop or CLI."""
    source = Path(__file__).with_name('simu_coupled_3.py')
    names = {
        'projection_matrix', 'extract_x', 'simu_step', 'edge_set',
    }
    tree = ast.parse(source.read_text(), filename=str(source))
    tree.body = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    reference = {'np': np, 'unit_vector': unit_vector}
    exec(compile(tree, str(source), 'exec'), reference)
    return reference


class TestAngleBasedPoseObserver(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(2)
        self.n = 5
        self.p = np.array([
            [13.07984706, 0.77778695, 16.48987434],
            [13.05967178, 8.61103406, 9.91004463],
            [6.13945902, 18.57812899, 8.98964021],
            [8.00481825, 12.63401498, 15.87426283],
            [15.0, 5.0, 5.0],
        ])
        self.R = np.array([
            rotation_matrix_from_vector(rng.normal(size=3))
            for _ in range(self.n)
        ])
        self.hatp = self.p + rng.normal(scale=0.2, size=self.p.shape)
        self.hatR = np.array([
            rotation_matrix_from_vector(rng.normal(scale=0.1, size=3)).dot(Ri)
            for Ri in self.R
        ])

    def test_exact_estimates_follow_simulation_through_switching_roles(self):
        reference = simulation_step()
        original_edges = reference['edge_set']
        # Exercise both of the script's graphs in a short regression run.
        reference['edge_set'] = lambda t: original_edges(20 * t)
        reference.update({
            'n': self.n,
            'nodes': np.arange(self.n),
            'p_int': [EulerIntegrator(pi) for pi in self.p],
            'R_int': [EulerIntegratorOrtogonalGroup(Ri) for Ri in self.R],
            'observer': AngleBasedPoseObserver(self.n, self.p, self.R),
            'control_u': {
                i: (lambda t, i=i: np.array([
                    np.cos((i + 1) * t), np.sin(t + i), 0.5
                ])) for i in range(self.n)
            },
            'control_w': {
                i: (lambda t, i=i: np.array([0.1 * i, 0.2, 0.0]))
                for i in range(self.n)
            },
        })
        for step in range(1, 1001):
            t = round(step * 0.001, 3)
            reference['t'] = t
            reference['simu_step']()
            observer = reference['observer']
            self.assertEqual(observer.t, t)
            np.testing.assert_allclose(
                observer.position, reference['extract_x'](reference['p_int']),
                rtol=0, atol=1e-7,
            )
            np.testing.assert_allclose(
                observer.orientation, reference['extract_x'](reference['R_int']),
                rtol=0, atol=1e-7,
            )

    def test_corrections_are_negative_gradients_of_one_cost(self):
        rng = np.random.RandomState(17)
        u = rng.normal(size=(self.n, 3))
        omega = rng.normal(size=(self.n, 3))
        # Include a reciprocal pair, overlapping angles, and a receiving-only node.
        edges = np.array([[0, 1], [0, 2], [0, 3], [1, 0], [1, 2], [2, 4]])
        bearings = np.full((self.n, self.n, 3), np.nan)
        rates = np.full_like(bearings, np.nan)
        v = np.einsum('nij,nj->ni', self.R, u)
        for i, j in edges:
            r = self.p[j] - self.p[i]
            b = r / np.linalg.norm(r)
            bearings[i, j] = self.R[i].T @ b
            b_dot = (v[j] - v[i] - b * b.dot(v[j] - v[i])) / np.linalg.norm(r)
            rates[i, j] = self.R[i].T @ b_dot - np.cross(omega[i], bearings[i, j])

        k_alpha, k_dalpha, k_beta = 1.3, 2.1, 0.8
        k_p, k_R = 2.3, 1.7
        observer = AngleBasedPoseObserver(
            self.n, self.hatp, self.hatR,
            k_alpha=k_alpha, k_dalpha=k_dalpha, k_beta=k_beta,
            k_p=k_p, k_R=k_R,
        )
        observer.update(0.0, u, omega, bearings, rates, edges)

        def cost(p, R):
            """Evaluate residual costs directly, independently of their gradients."""
            velocity = np.einsum('nij,nj->ni', R, u)
            total = 0.0
            for i in range(self.n):
                neighbors = edges[edges[:, 0] == i, 1]
                b, b_dot = {}, {}
                for j in neighbors:
                    r = p[j] - p[i]
                    d = np.linalg.norm(r)
                    b[j] = r / d
                    dv = velocity[j] - velocity[i]
                    b_dot[j] = (dv - b[j] * b[j].dot(dv)) / d
                    total += k_beta * (
                        1.0 - bearings[i, j].dot(R[i].T @ b[j])
                    )
                for index, j in enumerate(neighbors):
                    for k in neighbors[index + 1:]:
                        e_angle = b[j].dot(b[k]) - bearings[i, j].dot(bearings[i, k])
                        e_rate = (
                            b_dot[j].dot(b[k]) + b[j].dot(b_dot[k])
                            - rates[i, j].dot(bearings[i, k])
                            - bearings[i, j].dot(rates[i, k])
                        )
                        total += 0.5 * k_alpha * e_angle**2 + 0.5 * k_dalpha * e_rate**2
            return total

        eps = 1e-6
        grad_p = np.zeros_like(self.hatp)
        grad_R = np.zeros_like(self.hatp)
        for i in range(self.n):
            for axis in range(3):
                perturbation = eps * np.eye(3)[axis]
                plus_p, minus_p = self.hatp.copy(), self.hatp.copy()
                plus_p[i] += perturbation
                minus_p[i] -= perturbation
                grad_p[i, axis] = (
                    cost(plus_p, self.hatR) - cost(minus_p, self.hatR)
                ) / (2 * eps)
                plus_R, minus_R = self.hatR.copy(), self.hatR.copy()
                plus_R[i] = self.hatR[i] @ rotation_matrix_from_vector(perturbation)
                minus_R[i] = self.hatR[i] @ rotation_matrix_from_vector(-perturbation)
                grad_R[i, axis] = (
                    cost(self.hatp, plus_R) - cost(self.hatp, minus_R)
                ) / (2 * eps)

        # An update at t=0 caches derivatives without changing the estimates.
        actual_p = np.array([integrator.u() for integrator in observer._p_int])
        actual_R = np.array([integrator.u() for integrator in observer._R_int])
        np.testing.assert_allclose(
            actual_p - np.einsum('nij,nj->ni', self.hatR, u),
            -k_p * grad_p, rtol=1e-6, atol=1e-9,
        )
        np.testing.assert_allclose(
            actual_R - omega, -k_R * grad_R, rtol=1e-6, atol=1e-9,
        )

    def test_estimates_are_owned_by_observer(self):
        observer = AngleBasedPoseObserver(self.n, self.hatp, self.hatR)
        position, orientation = observer.position, observer.orientation
        self.hatp[:] = 0.0
        self.hatR[:] = 0.0
        np.testing.assert_array_equal(observer.position, position)
        np.testing.assert_array_equal(observer.orientation, orientation)
        position[:] = 0.0
        orientation[:] = 0.0
        self.assertTrue(np.any(observer.position))
        self.assertTrue(np.any(observer.orientation))

    def test_empty_edge_set_preserves_previous_input_stepping(self):
        observer = AngleBasedPoseObserver(
            self.n, self.hatp, self.hatR, time=2.0
        )
        u = np.ones((self.n, 3))
        omega = np.tile([0.1, 0.2, 0.3], (self.n, 1))
        position, orientation = observer.update(
            2.1, u, omega, {}, {}, np.empty((0, 2), dtype=int)
        )
        np.testing.assert_array_equal(position, self.hatp)
        np.testing.assert_array_equal(orientation, self.hatR)
        position, orientation = observer.update(
            2.2, u, omega, {}, {}, []
        )
        dt = 2.2 - 2.1
        expected_position = self.hatp + dt * np.einsum('nij,nj->ni', self.hatR, u)
        expected_orientation = np.array([
            Ri.dot(rotation_matrix_from_vector(dt * wi))
            for Ri, wi in zip(self.hatR, omega)
        ])
        np.testing.assert_allclose(position, expected_position, atol=1e-14)
        np.testing.assert_allclose(orientation, expected_orientation, atol=1e-14)

    def test_list_edges_and_sparse_measurements_match_dense_arrays(self):
        edges = np.array([[0, 2], [0, 1]])
        bearings = np.full((self.n, self.n, 3), np.nan)
        rates = np.full_like(bearings, np.nan)
        for i, j in edges:
            bearings[i, j] = self.R[i].T.dot(unit_vector(self.p[j] - self.p[i]))
            rates[i, j] = np.cross([0.1, 0.2, 0.3], bearings[i, j])
        sparse_bearings = {0: {j: bearings[0, j].tolist() for j in edges[:, 1]}}
        sparse_rates = {0: {j: rates[0, j].tolist() for j in edges[:, 1]}}
        dense = AngleBasedPoseObserver(self.n, self.hatp, self.hatR)
        sparse = AngleBasedPoseObserver(self.n, self.hatp, self.hatR)
        for t in [0.0, 0.001, 0.002]:
            u = np.ones((self.n, 3))
            omega = np.zeros((self.n, 3))
            p_dense, R_dense = dense.update(t, u, omega, bearings, rates, edges)
            p_sparse, R_sparse = sparse.update(
                t, u, omega, sparse_bearings, sparse_rates, edges.tolist()
            )
            self.assertTrue(np.isfinite(p_dense).all())
            self.assertTrue(np.isfinite(R_dense).all())
            np.testing.assert_array_equal(p_dense, p_sparse)
            np.testing.assert_array_equal(R_dense, R_sparse)


if __name__ == '__main__':
    unittest.main()
