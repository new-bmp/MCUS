from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from app.dataset_format import inspect_dataset_format
from app.curation_pipeline import _read_hdf5
from app.joint_overlay import _h5_points
from app.openxr_mano import (
    OPENXR_ENUM_NAMES,
    OPENXR_JOINT_NAMES,
    XR_SPACE_POSITION_VALID_BIT,
    convert_openxr_pose7,
    convert_openxr_positions,
    convert_openxr_transforms,
    detect_openxr_schema,
    read_openxr_hdf5_frame,
)
from app.schema_profiler import infer_local_signal_fields


def _positions(offset: float = 0.0) -> np.ndarray:
    values = np.zeros((26, 3), dtype=np.float64)
    values[:, 0] = np.arange(26, dtype=np.float64) + offset
    values[:, 1] = 1.0
    values[:, 2] = 2.0
    return values


class OpenXRManoAdapterTests(unittest.TestCase):
    def test_detection_requires_openxr_semantic_evidence(self) -> None:
        bare = detect_openxr_schema(shape=(120, 26, 3), field="positions")
        named = detect_openxr_schema(shape=(120, 26, 3), field="left/openxr_positions")
        labels = detect_openxr_schema(shape=(120, 26, 3), labels=OPENXR_ENUM_NAMES)

        self.assertFalse(bare["detected"])
        self.assertTrue(named["detected"])
        self.assertTrue(labels["detected"])

    def test_position_mapping_and_wrist_palm_root(self) -> None:
        converted, valid = convert_openxr_positions(_positions())

        self.assertEqual((21, 3), converted.shape)
        self.assertTrue(valid.all())
        self.assertAlmostEqual(0.65, converted[0, 0])
        self.assertEqual(2.0, converted[1, 0])
        self.assertEqual(7.0, converted[5, 0])
        self.assertEqual(25.0, converted[20, 0])

    def test_openxr_location_flags_mask_invalid_points_and_fallback_root(self) -> None:
        flags = np.full(26, XR_SPACE_POSITION_VALID_BIT, dtype=np.uint64)
        flags[1] = 0
        flags[7] = 1  # orientation-valid only; position is not valid

        converted, valid = convert_openxr_positions(_positions(), validity=flags)

        self.assertEqual(0.0, converted[0, 0])
        self.assertTrue(valid[0])
        self.assertFalse(valid[5])
        self.assertTrue(np.isnan(converted[5]).all())

    def test_pose7_and_transform_inputs_preserve_supported_representation(self) -> None:
        pose = np.zeros((26, 7), dtype=np.float64)
        pose[:, :3] = _positions()
        pose[:, 3:] = np.arange(26, dtype=np.float64)[:, None]
        converted_pose, pose_valid = convert_openxr_pose7(pose)

        transforms = np.repeat(np.eye(4, dtype=np.float64)[None], 26, axis=0)
        transforms[:, :3, 3] = _positions()
        transforms[:, 0, 0] = np.arange(26, dtype=np.float64) + 10.0
        converted_transforms, transform_valid = convert_openxr_transforms(transforms)

        self.assertEqual((21, 7), converted_pose.shape)
        self.assertTrue(pose_valid.all())
        self.assertAlmostEqual(0.65, converted_pose[0, 0])
        self.assertTrue(np.all(converted_pose[0, 3:] == 1.0))
        self.assertEqual((21, 4, 4), converted_transforms.shape)
        self.assertTrue(transform_valid.all())
        self.assertAlmostEqual(0.65, converted_transforms[0, 0, 3])
        self.assertEqual(11.0, converted_transforms[0, 0, 0])

    def test_hdf5_reader_combines_explicit_left_and_right_streams(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tracking.h5"
            with h5py.File(path, "w") as handle:
                left = np.stack([_positions(), _positions(100.0)])
                right = np.stack([_positions(200.0), _positions(300.0)])
                handle.create_dataset("left/openxr_positions", data=left)
                handle.create_dataset("right/openxr_positions", data=right)
                handle.create_dataset(
                    "left/location_flags",
                    data=np.full((2, 26), XR_SPACE_POSITION_VALID_BIT, dtype=np.uint64),
                )
                handle.create_dataset(
                    "right/location_flags",
                    data=np.full((2, 26), XR_SPACE_POSITION_VALID_BIT, dtype=np.uint64),
                )
            with h5py.File(path, "r") as handle:
                frame = read_openxr_hdf5_frame(handle, 1, source_path=str(path))
            points, labels, _, _, coordinate = _h5_points(path, 1)

        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual((42, 3), frame.points.shape)
        self.assertEqual(("left", "right"), frame.sides)
        self.assertAlmostEqual(100.65, frame.points[0, 0])
        self.assertAlmostEqual(300.65, frame.points[21, 0])
        self.assertEqual((42, 3), points.shape)
        self.assertEqual("leftHand", labels[0])
        self.assertEqual("rightHand", labels[21])
        self.assertEqual("world", coordinate)

    def test_non_openxr_26_point_hdf5_is_not_reclassified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "generic.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("positions", data=np.zeros((4, 26, 3), dtype=np.float32))
            with h5py.File(path, "r") as handle:
                frame = read_openxr_hdf5_frame(handle, 0, source_path=str(path))

        self.assertIsNone(frame)

    def test_schema_profiler_marks_openxr_conversion_as_mano21(self) -> None:
        fields = [{"key": "left/openxr_joint_locations", "shape": [12, 26, 7], "dtype": "float32"}]

        inferred = infer_local_signal_fields(fields)

        self.assertEqual(1, len(inferred))
        self.assertEqual("openxr_hand_26_to_mano21", inferred[0]["extraction"])
        self.assertEqual("openxr-hand-26", inferred[0]["embodiment_id"])
        self.assertEqual([12, 63], inferred[0]["shape"])
        self.assertEqual([12, 26, 7], inferred[0]["source_shape"])

    def test_curation_reader_uses_canonical_mano21_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "openxr.h5"
            positions = np.stack([_positions(float(frame)) for frame in range(12)])
            flags = np.full((12, 26), XR_SPACE_POSITION_VALID_BIT, dtype=np.uint64)
            flags[4] = 0
            with h5py.File(path, "w") as handle:
                handle.create_dataset("left/openxr_positions", data=positions)
                handle.create_dataset("left/location_flags", data=flags)

            series = _read_hdf5(
                path,
                "left/openxr_positions",
                {"extraction": "openxr_hand_26_to_mano21"},
            )

        self.assertEqual((12, 63), series["values"].shape)
        self.assertAlmostEqual(0.65, series["values"][0, 0])
        self.assertFalse(series["valid_rows"][4])
        self.assertTrue(np.isnan(series["values"][4]).all())

    def test_dataset_preflight_reports_openxr_adapter_without_claiming_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "head_rgb.mp4").touch()
            with h5py.File(root / "hands.h5", "w") as handle:
                handle.create_dataset(
                    "left/openxr_positions",
                    data=np.zeros((12, 26, 3), dtype=np.float32),
                )

            report = inspect_dataset_format(root)

        self.assertEqual("openxr", report["format_family"])
        self.assertEqual("openxr_hand_tracking_v1", report["processing_strategy"]["id"])
        self.assertEqual(["openxr_hand_26_to_mano21"], report["canonical_adapters"])
        self.assertTrue(report["capabilities"]["can_openxr_mano21_adapter"])
        self.assertFalse(report["capabilities"]["can_joint_overlay"])
        self.assertFalse(report["capabilities"]["can_full_export"])
        self.assertIn("openxr_mano21_adapter_ready", {item["code"] for item in report["issues"]})

    def test_standard_joint_name_sequence_matches_spec(self) -> None:
        self.assertEqual("palm", OPENXR_JOINT_NAMES[0])
        self.assertEqual("wrist", OPENXR_JOINT_NAMES[1])
        self.assertEqual("little_tip", OPENXR_JOINT_NAMES[25])


if __name__ == "__main__":
    unittest.main()
