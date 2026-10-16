"""Regression checks for extracting the observer from simu_coupled_3.py."""

import ast
from pathlib import Path
import unittest

import numpy as np
from transformations import unit_vector

from uvnpy.angles.local_frame.localization import AngleBasedPoseObserver
from uvnpy.dynamics.core import EulerIntegrator
from uvnpy.dynamics.lie_groups import EulerIntegratorOrtogonalGroup
from uvnpy.toolkit.geometry import rotation_matrix_from_vector


def reference_observer():
    """Load the script's actual observer without running its simulation/CLI."""
    source = Path(__file__).with_name('simu_coupled_3.py')
    names = {
        'projection_matrix', 'extract_x', 'complete_angle_set',
        'simu_step', 'edge_set',
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

    def test_matches_script_through_switching_roles(self):
        reference = reference_observer()
        original_edges = reference['edge_set']
        # Exercise both of the script's graphs in a short regression run.
        reference['edge_set'] = lambda t: original_edges(20 * t)
        reference.update({
            'n': self.n,
            'nodes': np.arange(self.n),
            'p_int': [EulerIntegrator(pi) for pi in self.p],
            'R_int': [EulerIntegratorOrtogonalGroup(Ri) for Ri in self.R],
            'hatp_int': [EulerIntegrator(pi) for pi in self.hatp],
            'hatR_int': [EulerIntegratorOrtogonalGroup(Ri) for Ri in self.hatR],
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
        observer = AngleBasedPoseObserver(self.n, self.hatp, self.hatR)

        for step in range(1, 1001):
            t = round(step * 0.001, 3)
            reference['t'] = t
            p = reference['extract_x'](reference['p_int'])
            R = reference['extract_x'](reference['R_int'])
            u = np.array([reference['control_u'][i](t) for i in range(self.n)])
            omega = np.array([
                reference['control_w'][i](t) for i in range(self.n)
            ])
            v = np.einsum('nij,nj->ni', R, u)
            edges = reference['edge_set'](t)
            bearings, bearing_rates = {}, {}
            for i in range(self.n):
                out_neighbors = edges[:, 1][edges[:, 0] == i]
                bearings[i], bearing_rates[i] = {}, {}
                for j in out_neighbors:
                    beta = R[i].T.dot(unit_vector(p[j] - p[i]))
                    distance = np.sqrt(np.square(p[j] - p[i]).sum())
                    bearings[i][j] = beta
                    bearing_rates[i][j] = (
                        reference['projection_matrix'](beta).dot(
                            R[i].T.dot(v[j] - v[i])
                        ) / distance - np.cross(omega[i], beta)
                    )

            position, orientation = observer.update(
                t, u, omega, bearings, bearing_rates, edges
            )
            reference['simu_step']()
            np.testing.assert_array_equal(
                position, reference['extract_x'](reference['hatp_int'])
            )
            np.testing.assert_array_equal(
                orientation, reference['extract_x'](reference['hatR_int'])
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
