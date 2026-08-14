from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from app.joint_overlay import _h5_points
from app.nexus_mano import (
    NEXUS_MANO21_REVISION,
    convert_nexus20_pose7,
    convert_nexus20_positions,
    detect_nexus20_schema,
    read_nexus20_hdf5_series,
)


def _hand_positions(frame_offset: float = 0.0) -> np.ndarray:
    values = np.zeros((5, 4, 3), dtype=np.float64)
    bases = (
        (-0.055, 0.045, 0.0),  # thumb
        (-0.030, 0.080, 0.0),  # index
        (-0.010, 0.080, 0.0),  # middle
        (0.010, 0.080, 0.0),   # ring
        (0.030, 0.080, 0.0),   # little
    )
    for finger, base in enumerate(bases):
        for joint in range(4):
            values[finger, joint] = np.asarray(base) + (0.0, 0.025 * joint, 0.0)
    values[..., 2] += frame_offset
    return values.reshape(20, 3)


class NexusManoAdapterTests(unittest.TestCase):
    def test_detection_requires_explicit_dexweave_evidence(self) -> None:
        bare = detect_nexus20_schema(shape=(100, 20, 7), field="skeleton")
        dexweave = detect_nexus20_schema(
            shape=(100, 20, 7),
            source_path="mocap/dexweaveg1_left.h5",
            field="skeleton",
        )

        self.assertFalse(bare["detected"])
        self.assertTrue(dexweave["detected"])
        self.assertTrue(dexweave["experimental_node_order"])

    def test_five_chains_map_to_mano_and_root_is_reconstructed(self) -> None:
        positions = _hand_positions()
        converted, valid, diagnostics = convert_nexus20_positions(
            positions,
            wrist_quaternion=np.asarray([0.0, 0.0, 0.0, 1.0]),
        )

        self.assertEqual((21, 3), converted.shape)
        self.assertTrue(valid.all())
        self.assertTrue(np.allclose(converted[1:], positions))
        self.assertTrue(np.allclose(converted[0], [0.0, 0.044, 0.0], atol=1e-8))
        self.assertAlmostEqual(0.06, float(diagnostics["palm_width"]))
        self.assertEqual(1, diagnostics["wrist_axis"])
        self.assertEqual(1, diagnostics["wrist_axis_sign"])
        self.assertEqual(NEXUS_MANO21_REVISION, diagnostics["revision"])

    def test_quaternion_sign_does_not_change_root(self) -> None:
        positions = np.stack([_hand_positions(), _hand_positions(0.01)])
        positive = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0]), (2, 1))
        negative = -positive

        first, _, _ = convert_nexus20_positions(positions, wrist_quaternion=positive)
        second, _, _ = convert_nexus20_positions(positions, wrist_quaternion=negative)

        self.assertTrue(np.allclose(first, second, equal_nan=True))

    def test_pose7_uses_wrist_quaternion_for_mano0(self) -> None:
        pose = np.zeros((20, 7), dtype=np.float64)
        pose[:, :3] = _hand_positions()
        pose[:, 6] = 1.0
        wrist = np.asarray([0.1, 0.2, 0.3, 0.9], dtype=np.float64)

        converted, valid, _ = convert_nexus20_pose7(pose, wrist_quaternion=wrist)

        self.assertEqual((21, 7), converted.shape)
        self.assertTrue(valid.all())
        self.assertTrue(np.allclose(converted[0, 3:7], wrist))
        self.assertTrue(np.allclose(converted[1:, 3:7], pose[:, 3:7]))

    def test_custom_source_finger_order_is_reordered_to_mano(self) -> None:
        canonical = _hand_positions().reshape(5, 4, 3)
        source_order = ("little", "ring", "middle", "index", "thumb")
        source = canonical[[4, 3, 2, 1, 0]].reshape(20, 3)

        converted, valid, diagnostics = convert_nexus20_positions(
            source,
            finger_order=source_order,
        )

        self.assertTrue(valid.all())
        self.assertTrue(np.allclose(converted[1:], canonical.reshape(20, 3)))
        self.assertEqual(source_order, diagnostics["finger_order"])

    def test_hdf5_series_marks_partial_frame_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dexweaveg1_left.h5"
            skeleton = np.zeros((4, 20, 7), dtype=np.float64)
            skeleton[..., :3] = np.stack([_hand_positions(float(index) * 0.01) for index in range(4)])
            skeleton[..., 6] = 1.0
            partial = np.asarray([False, False, True, False])
            with h5py.File(path, "w") as handle:
                handle.create_dataset("skeleton", data=skeleton)
                handle.create_dataset(
                    "wrist_quat",
                    data=np.tile(np.asarray([0.0, 0.0, 0.0, 1.0]), (4, 1)),
                )
                handle.create_dataset("partial", data=partial)
            with h5py.File(path, "r") as handle:
                series = read_nexus20_hdf5_series(handle, source_path=str(path))
            overlay_points, overlay_labels, _, _, coordinate = _h5_points(path, 1)

        self.assertIsNotNone(series)
        assert series is not None
        self.assertEqual((4, 21, 3), series.points.shape)
        self.assertEqual("left", series.side)
        self.assertEqual("leftHand", series.labels[0])
        self.assertTrue(series.valid[[0, 1, 3]].all())
        self.assertFalse(series.valid[2].any())
        self.assertTrue(np.isnan(series.points[2]).all())
        self.assertEqual((21, 3), overlay_points.shape)
        self.assertEqual("leftHand", overlay_labels[0])
        self.assertEqual("tracking", coordinate)

    def test_degenerate_zero_geometry_does_not_invent_a_wrist(self) -> None:
        converted, valid, diagnostics = convert_nexus20_positions(np.zeros((20, 3)))

        self.assertFalse(valid[0])
        self.assertTrue(np.isnan(converted[0]).all())
        self.assertEqual(0.0, float(diagnostics["root_confidence"]))


if __name__ == "__main__":
    unittest.main()
