from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from app.behavior_boundary_refiner import load_episode_joint_pose, refine_behavior_boundaries


class BehaviorBoundaryRefinerTests(unittest.TestCase):
    @staticmethod
    def _segments(boundary: int = 50) -> list[dict]:
        return [
            {"start_frame": 0, "end_frame": boundary - 1, "label": "reach", "custom": "left"},
            {"start_frame": boundary, "end_frame": 99, "label": "grasp", "custom": "right"},
        ]

    def test_missing_joint_pose_is_lossless_except_for_provenance(self) -> None:
        source = self._segments()

        result = refine_behavior_boundaries(source, 10.0, 100, None)

        self.assertEqual(["vlm", "vlm"], [item.pop("boundary_source") for item in result])
        self.assertEqual(source, result)
        self.assertNotIn("boundary_source", source[0])

    def test_joint_motion_start_refines_nearby_vlm_boundary(self) -> None:
        pose = np.zeros((100, 2), dtype=np.float64)
        pose[55:, 0] = np.arange(45, dtype=np.float64)

        result = refine_behavior_boundaries(self._segments(50), 10.0, 100, pose)

        self.assertEqual(54, result[0]["end_frame"])
        self.assertEqual(55, result[1]["start_frame"])
        self.assertEqual(["joint_refined", "joint_refined"], [item["boundary_source"] for item in result])
        self.assertEqual(0, result[0]["start_frame"])
        self.assertEqual(99, result[-1]["end_frame"])
        self.assertEqual("left", result[0]["custom"])

    def test_flat_joint_pose_keeps_vlm_boundary(self) -> None:
        result = refine_behavior_boundaries(self._segments(50), 30.0, 100, np.zeros((100, 6)))

        self.assertEqual((49, 50), (result[0]["end_frame"], result[1]["start_frame"]))
        self.assertEqual(["vlm", "vlm"], [item["boundary_source"] for item in result])

    def test_se3_pose_uses_translation_not_rotation_entries(self) -> None:
        transforms = np.repeat(np.eye(4)[None, None, :, :], 100, axis=0)
        transforms[55:, 0, 0, 0] = -1.0  # Rotation-only jump must be ignored.

        ignored = refine_behavior_boundaries(self._segments(50), 10.0, 100, transforms)

        self.assertEqual(["vlm", "vlm"], [item["boundary_source"] for item in ignored])

        transforms[55:, 0, 0, 3] = np.arange(45, dtype=np.float64)
        refined = refine_behavior_boundaries(self._segments(50), 10.0, 100, transforms)
        self.assertEqual(55, refined[1]["start_frame"])

    def test_multiple_boundaries_remain_ordered_and_non_overlapping(self) -> None:
        segments = [
            {"start_frame": 0, "end_frame": 38, "label": "a"},
            {"start_frame": 39, "end_frame": 60, "label": "b"},
            {"start_frame": 61, "end_frame": 99, "label": "c"},
        ]
        pose = np.zeros((100, 1), dtype=np.float64)
        pose[42:58, 0] = np.arange(16, dtype=np.float64)
        pose[58:, 0] = 16.0

        result = refine_behavior_boundaries(segments, 10.0, 100, pose)

        self.assertEqual(0, result[0]["start_frame"])
        self.assertEqual(99, result[-1]["end_frame"])
        for left, right in zip(result, result[1:]):
            self.assertEqual(left["end_frame"] + 1, right["start_frame"])
            self.assertLess(left["end_frame"], right["start_frame"])

    def test_loader_reuses_curation_joint_bundle(self) -> None:
        joint = np.arange(24, dtype=np.float64).reshape(8, 3)
        with patch("app.curation_pipeline._load_signal_bundle", return_value={"joint": joint}) as load, patch(
            "app.sensor_alignment.scan_episode_sensor_alignment", return_value={"streams": []}
        ) as scan:
            result = load_episode_joint_pose(
                {"root_path": "fixture"},
                {"id": "ep", "frame_count": 8},
                frame_count=8,
                reference_media_file_id="wrist-left",
            )

        np.testing.assert_array_equal(joint, result)
        scan.assert_called_once_with(
            {"root_path": "fixture"},
            {"id": "ep", "frame_count": 8},
            force=False,
            reference_media_file_id="wrist-left",
        )
        self.assertEqual(8, load.call_args.kwargs["frame_count"])


if __name__ == "__main__":
    unittest.main()
