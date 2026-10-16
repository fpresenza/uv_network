"""Checks for pose estimation alongside rigidity maintenance control."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from uvnpy.angles.local_frame.localization import AngleBasedPoseObserver


def simulation_namespace():
    """Initialize the actual simulation without its loop or file writes."""
    source = Path(__file__).with_name('simu.py')
    code = source.read_text().split('# run simulation', 1)[0]
    namespace = {'__name__': 'arm_simulation_test', '__file__': str(source)}
    random_state = np.random.get_state()
    print_options = np.get_printoptions()
    try:
        with redirect_stdout(io.StringIO()), patch('sys.argv', ['simu.py']), \
                patch('uvnpy.toolkit.data.write_json_file'):
            exec(compile(code, str(source), 'exec'), namespace)
    finally:
        np.random.set_state(random_state)
        np.set_printoptions(**print_options)
    return namespace


class TestParallelPoseEstimation(unittest.TestCase):
    def test_observer_tracks_exact_poses_without_affecting_control(self):
        perturbed = simulation_namespace()
        exact = simulation_namespace()
        exact['observer'] = AngleBasedPoseObserver(
            exact['n'], exact['initial_position'], exact['initial_orientation']
        )
        for step in range(1, 201):
            for simulation in [perturbed, exact]:
                simulation['t'] = round(step * 0.001, 3)
                simulation['simu_step']()

            # Different observer states must produce identical plant motion,
            # controls, and sensing graphs in this open-loop test.
            for name in ['p_int', 'R_int']:
                np.testing.assert_array_equal(
                    perturbed['extract_x'](perturbed[name]),
                    exact['extract_x'](exact[name]),
                )
                np.testing.assert_array_equal(
                    perturbed['extract_u'](perturbed[name]),
                    exact['extract_u'](exact[name]),
                )
            np.testing.assert_array_equal(
                perturbed['sensing_graph'].adjacency_matrix(),
                exact['sensing_graph'].adjacency_matrix(),
            )

            # Correct initialization should remain correct: this exercises
            # bearing-rate frames/signs, input timing, and propagation together.
            # High gains can briefly amplify roundoff; allow micrometre error.
            np.testing.assert_allclose(
                exact['observer'].position,
                exact['extract_x'](exact['p_int']), rtol=0, atol=1e-6,
            )
            np.testing.assert_allclose(
                exact['observer'].orientation,
                exact['extract_x'](exact['R_int']), rtol=0, atol=1e-7,
            )
            self.assertTrue(np.isfinite(perturbed['observer'].position).all())
            self.assertTrue(np.isfinite(perturbed['observer'].orientation).all())

        perturbed['log_step']()
        logs = perturbed['logs']
        self.assertEqual(len(logs.estimated_position), len(logs.time))
        self.assertEqual(len(logs.estimated_orientation), len(logs.time))
        np.testing.assert_array_equal(
            logs.estimated_position[-1], perturbed['observer'].position.ravel()
        )
        np.testing.assert_array_equal(
            logs.estimated_orientation[-1],
            perturbed['observer'].orientation.ravel(),
        )


if __name__ == '__main__':
    unittest.main()
