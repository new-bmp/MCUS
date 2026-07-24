from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import h5py
import numpy as np

from app.full_export import MANO_44_JOINT_NAMES, classify_behavior, export_episode, filtered_intervals, write_dataset_index


class FullExportTests(unittest.TestCase):
    @staticmethod
    def _write_video(path: Path, frame_count: int) -> None:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
        for index in range(frame_count):
            writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
        writer.release()

    @staticmethod
    def _write_transforms(path: Path, frame_count: int) -> None:
        with h5py.File(path, "w") as output:
            for joint_index, name in enumerate((*MANO_44_JOINT_NAMES, "camera")):
                values = np.repeat(np.eye(4, dtype=np.float32)[None], frame_count, axis=0)
                values[:, 0, 3] = np.arange(frame_count, dtype=np.float32) + joint_index
                output.create_dataset(f"transforms/{name}", data=values)
                if name != "camera":
                    output.create_dataset(f"confidences/{name}", data=np.ones(frame_count, dtype=np.float32))

    def test_only_red_precheck_segments_are_excluded_from_vlm_ranges(self) -> None:
        from app.curation_pipeline import curation_vlm_ranges

        ranges = curation_vlm_ranges({"pre_vlm_segments": [
            {"start_frame": 0, "end_frame": 2, "state": "invalid"},
            {"start_frame": 3, "end_frame": 5, "state": "uncertain"},
            {"start_frame": 6, "end_frame": 9, "state": "valid"},
        ]})

        self.assertEqual([(3, 9)], ranges)

    def test_full_export_writes_mano_camera_wrist_and_video_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_video = root / "smoothed.mp4"
            source_hdf5 = root / "episode.hdf5"
            output_root = root / "full"
            self._write_video(source_video, 20)
            self._write_transforms(source_hdf5, 20)
            manifest = {
                "id": "dataset",
                "name": "dataset",
                "root_path": str(root),
                "files": [{"id": "h5", "relative_path": "episode.hdf5", "episode_id": "ep", "episode_key": "episode"}],
            }
            episode = {"id": "ep", "name": "pick_object", "episode_key": "episode", "frame_count": 20, "fps": 10.0}
            behavior = {"task_label": "插usb", "segments": [
                {"start_frame": 0, "end_frame": 3, "phase_label": "idle"},
                {"start_frame": 4, "end_frame": 13, "phase_label": "grasp", "primary_targets": ["usb plug"]},
                {"start_frame": 14, "end_frame": 19, "phase_label": "reach"},
            ]}
            curation = {"segments": [{"start_frame": 0, "end_frame": 19, "state": "valid"}]}

            with patch.dict(os.environ, {"ALICE_VIDEO_ENCODER": "opencv"}):
                result = export_episode(
                    output_root,
                    manifest,
                    episode,
                    {"path": str(source_video), "frame_count": 20, "fps": 10.0},
                    curation,
                    behavior,
                    lambda _value, _message: None,
                )

            self.assertEqual([(4, 13)], [(item["start_frame"], item["end_frame"]) for item in result["pairs"]])
            pair = result["pairs"][0]
            self.assertEqual("ep1", pair["export_episode"])
            self.assertEqual("插usb", Path(pair["mp4"]).parent.parent.name)
            self.assertEqual("video.mp4", Path(pair["mp4"]).name)
            self.assertEqual("data.hdf5", Path(pair["hdf5"]).name)
            with h5py.File(pair["hdf5"], "r") as output:
                self.assertEqual((10, 44, 4, 4), output["mano/transforms"].shape)
                self.assertEqual((10, 4, 4), output["camera/transform"].shape)
                self.assertEqual((10, 9), output["wrist/left_xyz_rot6d"].shape)
                self.assertEqual((10, 9), output["wrist/right_xyz_rot6d"].shape)
                self.assertEqual([float(value) for value in range(5, 15)], output["wrist/left_xyz_rot6d"][:, 0].tolist())
                self.assertEqual(list(range(4, 14)), output["segment/source_frame_index"][:].tolist())
                self.assertEqual("grasp", output.attrs["phase_label"])
                self.assertEqual("usb plug", output.attrs["primary_target"])
            capture = cv2.VideoCapture(pair["mp4"])
            try:
                self.assertEqual(10, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            finally:
                capture.release()
            index = write_dataset_index(output_root, manifest, result["pairs"], [])
            self.assertTrue(index.is_file())

            second_pair = {**pair, "id": "插usb/ep2", "export_episode": "ep2"}
            write_dataset_index(output_root, manifest, [second_pair], [])
            combined = json.loads(index.read_text(encoding="utf-8"))
            self.assertEqual(2, combined["pair_count"])
            self.assertEqual({"插usb/ep1", "插usb/ep2"}, {item["id"] for item in combined["pairs"]})

    def test_idle_and_reach_are_removed_but_other_vlm_phases_remain(self) -> None:
        intervals, summary = filtered_intervals(
            8,
            {"segments": [{"start_frame": 0, "end_frame": 7, "state": "valid"}]},
            {"segments": [
                {"start_frame": 0, "end_frame": 1, "phase_label": "idle"},
                {"start_frame": 2, "end_frame": 3, "phase_label": "reach"},
                {"start_frame": 4, "end_frame": 7, "phase_label": "unknown"},
            ]},
        )

        self.assertEqual([(4, 7)], intervals)
        self.assertEqual(4, summary["removed_vlm_frame_count"])

    def test_short_quality_gaps_are_joined_and_tiny_islands_are_dropped(self) -> None:
        intervals, summary = filtered_intervals(
            50,
            {"segments": [
                {"start_frame": 0, "end_frame": 19, "state": "valid"},
                {"start_frame": 22, "end_frame": 41, "state": "valid"},
                {"start_frame": 45, "end_frame": 47, "state": "valid"},
            ]},
            {"segments": [{"start_frame": 0, "end_frame": 49, "phase_label": "manipulate"}]},
            fps=10.0,
            max_internal_gap_seconds=0.25,
            min_clip_seconds=0.75,
        )

        self.assertEqual([(0, 41)], intervals)
        self.assertEqual(2, summary["merged_gap_frame_count"])
        self.assertEqual(3, summary["dropped_short_fragment_frame_count"])
        self.assertEqual(2, summary["raw_interval_count"])
        self.assertEqual(1, summary["final_interval_count"])

    def test_vlm_removed_gap_is_never_filled_again(self) -> None:
        intervals, summary = filtered_intervals(
            24,
            {"segments": [{"start_frame": 0, "end_frame": 23, "state": "valid"}]},
            {"segments": [
                {"start_frame": 0, "end_frame": 9, "phase_label": "grasp"},
                {"start_frame": 10, "end_frame": 11, "phase_label": "reach"},
                {"start_frame": 12, "end_frame": 23, "phase_label": "place"},
            ]},
            fps=10.0,
            max_internal_gap_seconds=0.3,
            min_clip_seconds=0.0,
        )

        self.assertEqual([(0, 9), (12, 23)], intervals)
        self.assertEqual(2, summary["removed_vlm_frame_count"])

    def test_classifier_keeps_specific_task_and_enriches_clip_metadata(self) -> None:
        result = classify_behavior({
            "task_label": "insert_usb",
            "direction": "forward",
            "confidence": 0.8,
            "segments": [
                {"start_frame": 0, "end_frame": 9, "phase_label": "align", "confidence": 0.7, "primary_targets": ["usb port"]},
                {"start_frame": 10, "end_frame": 29, "phase_label": "manipulate", "confidence": 0.9, "primary_targets": ["usb plug"], "target_instance": "usb#1"},
            ],
        }, 5, 25)

        self.assertEqual("insert_usb", result["category"])
        self.assertEqual("manipulate", result["phase_label"])
        self.assertEqual("usb plug", result["primary_target"])
        self.assertEqual("vlm_task_label", result["classifier_source"])

    def test_classifier_uses_phase_and_target_only_for_generic_task(self) -> None:
        result = classify_behavior({
            "task_label": "other",
            "segments": [{"start_frame": 0, "end_frame": 20, "phase_label": "grasp", "primary_targets": ["cup"]}],
        }, 0, 20)

        self.assertEqual("grasp_cup", result["category"])
        self.assertEqual("vlm_phase_target_fallback", result["classifier_source"])


if __name__ == "__main__":
    unittest.main()
