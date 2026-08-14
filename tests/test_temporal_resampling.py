from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from app.temporal_resampling import interpolate_numeric_rows, interpolate_transform_rows


class TemporalResamplingTests(unittest.TestCase):
    def test_numeric_rows_follow_fractional_positions(self) -> None:
        source = np.asarray([[0.0, 10.0], [10.0, 30.0], [20.0, 50.0]], dtype=np.float32)

        output = interpolate_numeric_rows(source, np.asarray([0.0, 0.5, 1.5, 2.0]))

        np.testing.assert_allclose(output, [[0.0, 10.0], [5.0, 20.0], [15.0, 40.0], [20.0, 50.0]])

    def test_transform_rows_use_translation_lerp_and_rotation_slerp(self) -> None:
        source = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
        source[1, :3, :3] = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
        source[1, :3, 3] = (2.0, 4.0, 6.0)

        output = interpolate_transform_rows(source, np.asarray([0.5]))[0]

        np.testing.assert_allclose(output[:3, 3], [1.0, 2.0, 3.0], atol=1e-9)
        angle = Rotation.from_matrix(output[:3, :3]).as_euler("zxy", degrees=True)[0]
        self.assertAlmostEqual(45.0, angle, places=6)
        np.testing.assert_allclose(output[3], [0.0, 0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
