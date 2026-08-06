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
import pyarrow.parquet as parquet

from app import storage as storage_module
from app.full_export import MANO_44_JOINT_NAMES, classify_behavior, export_episode, filtered_intervals, write_dataset_index
from app.lerobot_export import HAND_21_JOINT_NAMES
from app.s1_repair import S1_REPAIR_SCHEMA


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
            output.create_dataset("camera/intrinsic", data=np.asarray([
                [40.0, 0.0, 32.0],
                [0.0, 40.0, 24.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32))
            for joint_index, name in enumerate((*MANO_44_JOINT_NAMES, "camera")):
                values = np.repeat(np.eye(4, dtype=np.float32)[None], frame_count, axis=0)
                values[:, 0, 3] = np.arange(frame_count, dtype=np.float32) + joint_index
                output.create_dataset(f"transforms/{name}", data=values)
                if name != "camera":
                    output.create_dataset(f"confidences/{name}", data=np.ones(frame_count, dtype=np.float32))
            head = np.repeat(np.eye(4, dtype=np.float32)[None], frame_count, axis=0)
            head[:, 0, 3] = np.arange(frame_count, dtype=np.float32) + 1000
            output.create_dataset("transforms/head", data=head)
            output.create_dataset("confidences/head", data=np.ones(frame_count, dtype=np.float32))

    def test_format_capability_blocks_unsafe_fixed_hand_export(self) -> None:
        manifest = {
            "id": "nexus",
            "format_map": {
                "capabilities": {"can_full_export": False},
                "issues": [{
                    "severity": "warning",
                    "code": "noncanonical_hand_nodes",
                    "message": "源手部骨架为 20 节点，不会补零冒充 21 点 MANO。",
                }],
            },
        }

        with self.assertRaisesRegex(RuntimeError, "20 节点"):
            export_episode(
                Path("output"),
                manifest,
                {"id": "ep", "frame_count": 10, "fps": 30.0},
                {"frame_count": 10, "fps": 30.0},
                {"segments": []},
                {"segments": []},
                lambda _value, _message: None,
            )

    @staticmethod
    def _write_hand_only_transforms(path: Path, frame_count: int) -> None:
        hand_names = [
            name for name in MANO_44_JOINT_NAMES
            if name not in {"leftForearm", "rightForearm"}
        ]
        with h5py.File(path, "w") as output:
            for joint_index, name in enumerate((*hand_names, "camera")):
                values = np.repeat(np.eye(4, dtype=np.float32)[None], frame_count, axis=0)
                values[:, 0, 3] = np.arange(frame_count, dtype=np.float32) + joint_index
                output.create_dataset(f"transforms/{name}", data=values)

    def test_only_red_precheck_segments_are_excluded_from_vlm_ranges(self) -> None:
        from app.curation_pipeline import curation_vlm_ranges

        ranges = curation_vlm_ranges({"pre_vlm_segments": [
            {"start_frame": 0, "end_frame": 2, "state": "invalid"},
            {"start_frame": 3, "end_frame": 5, "state": "uncertain"},
            {"start_frame": 6, "end_frame": 9, "state": "valid"},
        ]})

        self.assertEqual([(3, 9)], ranges)

    def test_full_export_defaults_to_lerobot_hands_body_and_video(self) -> None:
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
            self.assertEqual("lerobot", pair["output_format"])
            self.assertEqual("episode_000000", pair["id"])
            self.assertEqual(
                Path("data/chunk-000/episode_000000.parquet"),
                Path(pair["data"]).relative_to(output_root),
            )
            self.assertEqual(
                Path("body/chunk-000/episode_000000.parquet"),
                Path(pair["body"]).relative_to(output_root),
            )
            self.assertEqual(
                Path("videos/chunk-000/observation.images.main/episode_000000.mp4"),
                Path(pair["mp4"]).relative_to(output_root),
            )
            data = parquet.read_table(pair["data"])
            self.assertEqual(10, data.num_rows)
            self.assertEqual(21 * 4 * 4, data.schema.field("observation.left_hand.transforms").type.list_size)
            self.assertEqual(21 * 4 * 4, data.schema.field("observation.right_hand.transforms").type.list_size)
            left = np.asarray(data["observation.left_hand.transforms"].combine_chunks().values).reshape(10, 21, 4, 4)
            self.assertEqual([float(value) for value in range(5, 15)], left[:, 0, 0, 3].tolist())
            self.assertEqual(list(range(4, 14)), data["source.frame_index"].to_pylist())
            self.assertEqual(["grasp"] * 10, data["annotation.phase_label"].to_pylist())

            self.assertEqual(["head", "leftForearm", "rightForearm"], pair["body_joint_names"])
            body = parquet.read_table(pair["body"])
            self.assertEqual(10, body.num_rows)
            body_values = np.asarray(body["observation.body.transforms"].combine_chunks().values).reshape(10, 3, 4, 4)
            self.assertEqual([float(value) for value in range(1004, 1014)], body_values[:, 0, 0, 3].tolist())
            self.assertEqual([float(value) for value in range(4, 14)], body_values[:, 1, 0, 3].tolist())
            capture = cv2.VideoCapture(pair["mp4"])
            try:
                self.assertEqual(10, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            finally:
                capture.release()
            index = write_dataset_index(output_root, manifest, result["pairs"], [])
            self.assertTrue(index.is_file())
            self.assertEqual(output_root / "meta" / "info.json", index)
            info = json.loads(index.read_text(encoding="utf-8"))
            self.assertEqual(list(HAND_21_JOINT_NAMES), info["hand_joint_names"])
            self.assertEqual(["head", "leftForearm", "rightForearm"], info["body_joint_names"])
            self.assertEqual([[40.0, 0.0, 32.0], [0.0, 40.0, 24.0], [0.0, 0.0, 1.0]], info["camera_intrinsic"])
            self.assertEqual(info["camera_intrinsic"], pair["camera_intrinsic"])
            self.assertEqual(1, info["total_episodes"])
            self.assertEqual(10, info["total_frames"])
            self.assertIn("annotation.phase_label", info["features"])
            self.assertTrue((output_root / "meta" / "tasks.parquet").is_file())
            self.assertTrue((output_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").is_file())
            combined = json.loads((output_root / "dataset.json").read_text(encoding="utf-8"))
            self.assertEqual("lerobot", combined["default_output_format"])
            self.assertEqual(1, combined["pair_count"])

            discovery = storage_module.discover_dataset_roots(output_root)
            self.assertEqual("single", discovery["mode"])
            with patch.object(storage_module, "save_manifest", return_value=None):
                rescanned = storage_module.scan_dataset(output_root, dataset_id="full-output-rescan")
            self.assertEqual(1, rescanned["episode_count"])
            records = {item["relative_path"]: item for item in rescanned["files"]}
            paired_episode_ids = {
                records["data/chunk-000/episode_000000.parquet"]["episode_id"],
                records["body/chunk-000/episode_000000.parquet"]["episode_id"],
                records["videos/chunk-000/observation.images.main/episode_000000.mp4"]["episode_id"],
            }
            self.assertEqual({rescanned["episodes"][0]["id"]}, paired_episode_ids)

    def test_hdf5_mp4_remains_an_explicit_compatibility_format(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_video = root / "smoothed.mp4"
            source_hdf5 = root / "episode.hdf5"
            output_root = root / "full"
            self._write_video(source_video, 20)
            self._write_transforms(source_hdf5, 20)
            manifest = {
                "id": "dataset",
                "root_path": str(root),
                "files": [{"id": "h5", "relative_path": "episode.hdf5", "episode_id": "ep", "episode_key": "episode"}],
            }
            episode = {"id": "ep", "name": "pick_object", "episode_key": "episode", "frame_count": 20, "fps": 10.0}
            behavior = {"task_label": "insert_usb", "segments": [
                {"start_frame": 0, "end_frame": 3, "phase_label": "idle"},
                {"start_frame": 4, "end_frame": 13, "phase_label": "grasp"},
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
                    output_format="hdf5_mp4",
                )

            pair = result["pairs"][0]
            self.assertEqual("hdf5_mp4", pair["output_format"])
            self.assertEqual("ep1", pair["export_episode"])
            self.assertEqual("video.mp4", Path(pair["mp4"]).name)
            self.assertEqual("data.hdf5", Path(pair["hdf5"]).name)
            with h5py.File(pair["hdf5"], "r") as output:
                self.assertEqual((10, 44, 4, 4), output["mano/transforms"].shape)
                self.assertEqual((10, 4, 4), output["camera/transform"].shape)
                self.assertEqual((10, 9), output["wrist/left_xyz_rot6d"].shape)
                self.assertEqual([float(value) for value in range(5, 15)], output["wrist/left_xyz_rot6d"][:, 0].tolist())

            index = write_dataset_index(output_root, manifest, result["pairs"], [], output_format="hdf5_mp4")
            second_pair = {**pair, "id": "insert_usb/ep2", "export_episode": "ep2"}
            write_dataset_index(output_root, manifest, [second_pair], [], output_format="hdf5_mp4")
            combined = json.loads(index.read_text(encoding="utf-8"))
            self.assertEqual(2, combined["pair_count"])
            self.assertEqual({"insert_usb/ep1", "insert_usb/ep2"}, {item["id"] for item in combined["pairs"]})

    def test_lerobot_export_applies_s1_patch_without_modifying_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_video = root / "smoothed.mp4"
            source_hdf5 = root / "episode.hdf5"
            output_root = root / "output"
            self._write_video(source_video, 10)
            self._write_transforms(source_hdf5, 10)
            manifest = {
                "id": "dataset",
                "name": "dataset",
                "root_path": str(root),
                "files": [{"id": "h5", "relative_path": "episode.hdf5", "episode_id": "ep"}],
            }
            episode = {"id": "ep", "name": "ep", "frame_count": 10, "fps": 10.0}
            patch_payload = {
                "schema": S1_REPAIR_SCHEMA,
                "dataset_id": "dataset",
                "episode_id": "ep",
                "entries": [{
                    "source_path": "episode.hdf5",
                    "dataset_path": "transforms/leftHand",
                    "source_rows": [5],
                    "flat_indices": [3],
                    "values": [999.0],
                }],
            }
            curation = {
                "dataset_id": "dataset",
                "episode_id": "ep",
                "segments": [{"start_frame": 0, "end_frame": 9, "state": "valid"}],
                "s1_repair": {"repaired_frame_count": 1, "patch": patch_payload},
            }
            behavior = {"task_label": "grasp", "segments": [
                {"start_frame": 0, "end_frame": 9, "phase_label": "grasp"},
            ]}

            with patch.dict(os.environ, {"ALICE_VIDEO_ENCODER": "opencv"}):
                result = export_episode(
                    output_root,
                    manifest,
                    episode,
                    {"path": str(source_video), "frame_count": 10, "fps": 10.0},
                    curation,
                    behavior,
                    lambda _value, _message: None,
                )

            data = parquet.read_table(result["pairs"][0]["data"])
            left = np.asarray(data["observation.left_hand.transforms"].combine_chunks().values).reshape(10, 21, 4, 4)
            self.assertEqual(999.0, left[5, 0, 0, 3])
            with h5py.File(source_hdf5, "r") as source:
                self.assertEqual(6.0, source["transforms/leftHand"][5, 0, 3])
            self.assertTrue(result["s1_repair_applied"])

            with patch.dict(os.environ, {"ALICE_VIDEO_ENCODER": "opencv"}):
                compatibility = export_episode(
                    root / "compat-output",
                    manifest,
                    episode,
                    {"path": str(source_video), "frame_count": 10, "fps": 10.0},
                    curation,
                    behavior,
                    lambda _value, _message: None,
                    output_format="hdf5_mp4",
                )
            with h5py.File(compatibility["pairs"][0]["hdf5"], "r") as exported:
                left_hand_index = MANO_44_JOINT_NAMES.index("leftHand")
                self.assertEqual(999.0, exported["mano/transforms"][5, left_hand_index, 0, 3])
                self.assertTrue(exported.attrs["s1_repair_applied"])
            with h5py.File(source_hdf5, "r") as source:
                self.assertEqual(6.0, source["transforms/leftHand"][5, 0, 3])

    def test_lerobot_accepts_hand_only_source_without_body_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_video = root / "smoothed.mp4"
            source_hdf5 = root / "episode.hdf5"
            output_root = root / "full"
            self._write_video(source_video, 10)
            self._write_hand_only_transforms(source_hdf5, 10)
            manifest = {
                "id": "dataset",
                "root_path": str(root),
                "files": [{"id": "h5", "relative_path": "episode.hdf5", "episode_id": "ep"}],
            }
            episode = {"id": "ep", "name": "ep", "frame_count": 10, "fps": 10.0}
            behavior = {"task_label": "grasp", "segments": [
                {"start_frame": 0, "end_frame": 9, "phase_label": "grasp"},
            ]}

            with patch.dict(os.environ, {"ALICE_VIDEO_ENCODER": "opencv"}):
                result = export_episode(
                    output_root,
                    manifest,
                    episode,
                    {"path": str(source_video), "frame_count": 10, "fps": 10.0},
                    {"segments": [{"start_frame": 0, "end_frame": 9, "state": "valid"}]},
                    behavior,
                    lambda _value, _message: None,
                )

            pair = result["pairs"][0]
            self.assertIsNone(pair["body"])
            self.assertEqual([], pair["body_joint_names"])
            self.assertFalse((output_root / "body").exists())

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
