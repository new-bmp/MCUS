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
from app.full_export import MANO_44_JOINT_NAMES, SUBTASK_JSON_SCHEMA, _aligned_positions, classify_behavior, export_episode, filtered_intervals, write_dataset_index
from app.lerobot_export import HAND_21_JOINT_NAMES
from app.s1_repair import S1_REPAIR_SCHEMA


class FullExportTests(unittest.TestCase):
    def test_projection_source_uses_analysis_positions_in_corrected_timeline(self) -> None:
        corrected = Path("corrected.hdf5").resolve()
        positions = [0.0, 1.5, 3.0, 4.0]
        with (
            patch("app.projection_correction.active_projection_source", return_value={"path": corrected}),
            patch("app.full_export._find_transform_source", return_value=(corrected, "source.hdf5", 5)),
            patch("app.full_export.aligned_sensor_positions", side_effect=AssertionError("raw T0 must not remap corrected rows")),
        ):
            result = _aligned_positions(
                {"root_path": str(Path.cwd())},
                {"id": "ep"},
                "source.hdf5",
                5,
                4,
                {"source_frame_positions": positions},
            )

        np.testing.assert_array_equal(np.asarray(positions), result)

    @staticmethod
    def _write_video(path: Path, frame_count: int) -> None:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
        for index in range(frame_count):
            writer.write(np.full((48, 64, 3), (index * 10) % 256, dtype=np.uint8))
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

    @staticmethod
    def _action_report(horizon_frames: int = 3) -> dict:
        fields = [
            *[f"left_{name}" for name in ("dx", "dy", "dz", "drot_x", "drot_y", "drot_z", "grip")],
            *[f"right_{name}" for name in ("dx", "dy", "dz", "drot_x", "drot_y", "drot_z", "grip")],
        ]
        return {
            "profile": {
                "id": "generic_bimanual_delta",
                "robot_family": "generic_bimanual",
                "representation": "delta_pose_axis_angle",
                "control_space": "cartesian_delta",
                "sides": 2,
                "action_dim": 14,
                "fields": fields,
            },
            "config": {
                "profile_id": "generic_bimanual_delta",
                "source_hand": "right",
                "coordinate_frame": "camera",
                "horizon_frames": horizon_frames,
            },
        }

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

            self.assertEqual([(4, 19)], [(item["start_frame"], item["end_frame"]) for item in result["pairs"]])
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
            self.assertEqual(16, data.num_rows)
            self.assertEqual(21 * 4 * 4, data.schema.field("observation.left_hand.transforms").type.list_size)
            self.assertEqual(21 * 4 * 4, data.schema.field("observation.right_hand.transforms").type.list_size)
            left = np.asarray(data["observation.left_hand.transforms"].combine_chunks().values).reshape(16, 21, 4, 4)
            self.assertEqual([float(value) for value in range(5, 21)], left[:, 0, 0, 3].tolist())
            self.assertEqual(list(range(4, 20)), data["source.frame_index"].to_pylist())
            self.assertEqual(["grasp"] * 10 + ["reach"] * 6, data["annotation.phase_label"].to_pylist())
            self.assertEqual(["valid"] * 16, data["quality.state"].to_pylist())

            self.assertEqual(["head", "leftForearm", "rightForearm"], pair["body_joint_names"])
            body = parquet.read_table(pair["body"])
            self.assertEqual(16, body.num_rows)
            body_values = np.asarray(body["observation.body.transforms"].combine_chunks().values).reshape(16, 3, 4, 4)
            self.assertEqual([float(value) for value in range(1004, 1020)], body_values[:, 0, 0, 3].tolist())
            self.assertEqual([float(value) for value in range(4, 20)], body_values[:, 1, 0, 3].tolist())
            capture = cv2.VideoCapture(pair["mp4"])
            try:
                self.assertEqual(16, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
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
            self.assertEqual(16, info["total_frames"])
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
                self.assertEqual((16, 44, 4, 4), output["mano/transforms"].shape)
                self.assertEqual((16, 4, 4), output["camera/transform"].shape)
                self.assertEqual((16, 9), output["wrist/left_xyz_rot6d"].shape)
                self.assertEqual([float(value) for value in range(5, 21)], output["wrist/left_xyz_rot6d"][:, 0].tolist())

            index = write_dataset_index(output_root, manifest, result["pairs"], [], output_format="hdf5_mp4")
            second_pair = {**pair, "id": "insert_usb/ep2", "export_episode": "ep2"}
            write_dataset_index(output_root, manifest, [second_pair], [], output_format="hdf5_mp4")
            combined = json.loads(index.read_text(encoding="utf-8"))
            self.assertEqual(2, combined["pair_count"])
            self.assertEqual({"insert_usb/ep1", "insert_usb/ep2"}, {item["id"] for item in combined["pairs"]})

    def test_lerobot_exports_action_on_the_frozen_analysis_timeline(self) -> None:
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
            episode = {"id": "ep", "name": "pick", "episode_key": "episode", "frame_count": 20, "fps": 10.0}
            behavior = {"task_label": "pick", "segments": [{"start_frame": 0, "end_frame": 19, "phase_label": "grasp"}]}
            curation = {"segments": [{"start_frame": 0, "end_frame": 19, "state": "valid"}]}

            with patch.dict(os.environ, {"ALICE_VIDEO_ENCODER": "opencv"}):
                result = export_episode(
                    output_root,
                    manifest,
                    episode,
                    {"path": str(source_video), "frame_count": 20, "fps": 10.0},
                    curation,
                    behavior,
                    lambda *_: None,
                    action_report=self._action_report(horizon_frames=3),
                )

            pair = result["pairs"][0]
            data = parquet.read_table(pair["data"])
            self.assertEqual(17, data.num_rows)
            self.assertEqual(20, data.schema.field("observation.state").type.list_size)
            self.assertEqual(14, data.schema.field("action").type.list_size)
            self.assertEqual(list(range(3, 20)), data["action.target_source_frame_index"].to_pylist())
            self.assertTrue(all(data["quality.action_valid"].to_pylist()))
            self.assertEqual(3, result["filtering"]["action_tail_removed_frame_count"])
            info = json.loads(write_dataset_index(output_root, manifest, result["pairs"], []).read_text(encoding="utf-8"))
            self.assertEqual([14], info["features"]["action"]["shape"])
            self.assertEqual("generic_bimanual_delta", info["action_policy"]["profile_id"])

    def test_hdf5_mp4_exports_action_on_the_same_filtered_timeline(self) -> None:
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
            episode = {"id": "ep", "name": "pick", "episode_key": "episode", "frame_count": 20, "fps": 10.0}
            behavior = {"task_label": "pick", "segments": [{"start_frame": 0, "end_frame": 19, "phase_label": "grasp"}]}
            curation = {"segments": [{"start_frame": 0, "end_frame": 19, "state": "valid"}]}

            with patch.dict(os.environ, {"ALICE_VIDEO_ENCODER": "opencv"}):
                result = export_episode(
                    output_root,
                    manifest,
                    episode,
                    {"path": str(source_video), "frame_count": 20, "fps": 10.0},
                    curation,
                    behavior,
                    lambda *_: None,
                    output_format="hdf5_mp4",
                    action_report=self._action_report(horizon_frames=3),
                )

            pair = result["pairs"][0]
            self.assertEqual(17, pair["frame_count"])
            self.assertEqual(17, pair["action_valid_frame_count"])
            with h5py.File(pair["hdf5"], "r") as output:
                self.assertEqual((17, 20), output["observation/state"].shape)
                self.assertEqual((17, 14), output["action"].shape)
                self.assertEqual(list(range(3, 20)), output["action_target_source_frame_index"][:].tolist())
                self.assertTrue(output["action_valid"][:].all())
                self.assertEqual("generic_bimanual_delta", output.attrs["action_profile_id"])

    def test_action_never_crosses_bad_frames_or_output_segment_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_video = root / "smoothed.mp4"
            source_hdf5 = root / "episode.hdf5"
            output_root = root / "full"
            self._write_video(source_video, 30)
            self._write_transforms(source_hdf5, 30)
            manifest = {
                "id": "dataset",
                "root_path": str(root),
                "files": [{"id": "h5", "relative_path": "episode.hdf5", "episode_id": "ep", "episode_key": "episode"}],
            }
            episode = {"id": "ep", "name": "pick", "episode_key": "episode", "frame_count": 30, "fps": 10.0}
            behavior = {"task_label": "pick", "segments": [{"start_frame": 0, "end_frame": 29, "phase_label": "grasp"}]}
            curation = {"segments": [
                {"start_frame": 0, "end_frame": 9, "state": "valid"},
                {"start_frame": 10, "end_frame": 10, "state": "invalid"},
                {"start_frame": 11, "end_frame": 29, "state": "valid"},
            ]}

            with patch.dict(os.environ, {"ALICE_VIDEO_ENCODER": "opencv"}):
                result = export_episode(
                    output_root,
                    manifest,
                    episode,
                    {"path": str(source_video), "frame_count": 30, "fps": 10.0},
                    curation,
                    behavior,
                    lambda *_: None,
                    action_report=self._action_report(horizon_frames=3),
                )

            self.assertEqual([(11, 26)], [(pair["start_frame"], pair["end_frame"]) for pair in result["pairs"]])
            data = parquet.read_table(result["pairs"][0]["data"])
            self.assertEqual(list(range(14, 30)), data["action.target_source_frame_index"].to_pylist())
            self.assertTrue(all(data["quality.action_valid"].to_pylist()))
            self.assertEqual(6, result["filtering"]["action_tail_removed_frame_count"])
            self.assertEqual(7, result["filtering"]["action_short_fragment_removed_frame_count"])

    def test_complete_episode_action_is_invalid_when_horizon_crosses_quality_problem(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_video = root / "smoothed.mp4"
            source_hdf5 = root / "episode.hdf5"
            output_root = root / "full"
            self._write_video(source_video, 10)
            self._write_transforms(source_hdf5, 10)
            manifest = {
                "id": "dataset",
                "name": "dataset",
                "root_path": str(root),
                "files": [{"id": "h5", "relative_path": "episode.hdf5", "episode_id": "ep", "episode_key": "episode"}],
            }
            episode = {"id": "ep", "name": "package", "episode_key": "episode", "frame_count": 10, "fps": 10.0}
            behavior = {"task_label": "pick", "segments": [{"start_frame": 0, "end_frame": 9, "phase_label": "grasp"}]}
            curation = {"segments": [
                {"start_frame": 0, "end_frame": 1, "state": "valid"},
                {"start_frame": 2, "end_frame": 2, "state": "invalid"},
                {"start_frame": 3, "end_frame": 6, "state": "valid"},
                {"start_frame": 7, "end_frame": 7, "state": "uncertain"},
                {"start_frame": 8, "end_frame": 9, "state": "valid"},
            ]}

            with patch.dict(os.environ, {"ALICE_VIDEO_ENCODER": "opencv"}):
                result = export_episode(
                    output_root,
                    manifest,
                    episode,
                    {"path": str(source_video), "frame_count": 10, "fps": 10.0},
                    curation,
                    behavior,
                    lambda *_: None,
                    output_format="episode_lerobot_json",
                    action_report=self._action_report(horizon_frames=3),
                )

            data = parquet.read_table(result["pairs"][0]["data"])
            self.assertEqual([False, False, False, True, False, False, False, False, False, False], data["quality.action_valid"].to_pylist())
            self.assertEqual(1, result["pairs"][0]["action_valid_frame_count"])

    def test_episode_lerobot_json_writes_quality_state_per_parquet_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_video = root / "smoothed.mp4"
            source_hdf5 = root / "episode.hdf5"
            output_root = root / "full"
            self._write_video(source_video, 10)
            self._write_transforms(source_hdf5, 10)
            manifest = {
                "id": "dataset",
                "name": "dataset",
                "root_path": str(root),
                "files": [{"id": "h5", "relative_path": "episode.hdf5", "episode_id": "ep", "episode_key": "episode"}],
            }
            episode = {"id": "ep", "name": "package", "episode_key": "episode", "frame_count": 10, "fps": 10.0}
            behavior = {"task_label": "pick", "segments": [{"start_frame": 0, "end_frame": 9, "phase_label": "grasp"}]}
            curation = {"segments": [
                {"start_frame": 0, "end_frame": 2, "state": "valid"},
                {"start_frame": 3, "end_frame": 4, "state": "invalid"},
                {"start_frame": 5, "end_frame": 5, "state": "uncertain"},
                {"start_frame": 6, "end_frame": 9, "state": "valid"},
            ]}

            with patch.dict(os.environ, {"ALICE_VIDEO_ENCODER": "opencv"}):
                result = export_episode(
                    output_root,
                    manifest,
                    episode,
                    {"path": str(source_video), "frame_count": 10, "fps": 10.0},
                    curation,
                    behavior,
                    lambda *_: None,
                    output_format="episode_lerobot_json",
                    run_id="run",
                    timeline_id="timeline",
                )

            pair = result["pairs"][0]
            data = parquet.read_table(pair["data"])
            self.assertEqual(["valid", "valid", "valid", "invalid", "invalid", "review", "valid", "valid", "valid", "valid"], data["quality.state"].to_pylist())
            self.assertEqual([False, False, False, True, True, False, False, False, False, False], data["quality.is_bad"].to_pylist())
            self.assertEqual([False, False, False, False, False, True, False, False, False, False], data["quality.needs_review"].to_pylist())
            self.assertEqual(2, pair["bad_frame_count"])
            self.assertEqual(1, pair["review_frame_count"])
            self.assertEqual("keep_all_frames_mark_quality_in_parquet_and_json", result["filtering"]["policy"])

    def test_lerobot_export_interpolates_fractional_rows_and_writes_eis_camera_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_video = root / "smoothed.mp4"
            source_hdf5 = root / "episode.hdf5"
            output_root = root / "full"
            writer = cv2.VideoWriter(str(source_video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
            for index in range(3):
                writer.write(np.full((48, 64, 3), index * 20, dtype=np.uint8))
            writer.release()
            self._write_transforms(source_hdf5, 4)
            manifest = {
                "id": "dataset",
                "name": "dataset",
                "root_path": str(root),
                "files": [{"id": "h5", "relative_path": "episode.hdf5", "episode_id": "ep", "episode_key": "episode"}],
            }
            episode = {"id": "ep", "name": "pick", "episode_key": "episode", "frame_count": 4, "fps": 60.0}
            positions = np.asarray([0.0, 0.5, 2.0], dtype=np.float64)
            image_transforms = np.repeat(np.eye(3, dtype=np.float64)[None], 3, axis=0)
            image_transforms[1, 0, 2] = 5.0
            media = {
                "path": str(source_video),
                "frame_count": 3,
                "fps": 3.0,
                "source_frame_positions": positions.tolist(),
                "target_stabilization_matrices": image_transforms,
            }
            curation = {"segments": [{"start_frame": 0, "end_frame": 2, "state": "valid"}]}
            behavior = {"task_label": "pick", "segments": [{"start_frame": 0, "end_frame": 2, "phase_label": "grasp"}]}

            with (
                patch.dict(os.environ, {"ALICE_VIDEO_ENCODER": "opencv"}),
                patch("app.full_export._aligned_positions", return_value=positions),
            ):
                result = export_episode(output_root, manifest, episode, media, curation, behavior, lambda *_: None)

            data = parquet.read_table(result["pairs"][0]["data"])
            left = np.asarray(data["observation.left_hand.transforms"].combine_chunks().values).reshape(3, 21, 4, 4)
            intrinsics = np.asarray(data["observation.camera.intrinsic"].combine_chunks().values).reshape(3, 3, 3)
            self.assertEqual([1.0, 1.5, 3.0], left[:, 0, 0, 3].tolist())
            self.assertEqual([0.0, 0.5, 2.0], data["source.hdf5_position"].to_pylist())
            self.assertEqual([0.0, 0.5, 2.0], data["source.video_frame_position"].to_pylist())
            self.assertAlmostEqual(37.0, float(intrinsics[1, 0, 2]), places=5)
            self.assertEqual("per_frame_eis_corrected", result["pairs"][0]["camera_intrinsic_mode"])

    def test_subtask_json_writes_one_document_per_source_episode_with_bad_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "full"
            manifest = {"id": "dataset", "name": "dataset", "root_path": str(root), "files": []}
            episode = {"id": "ep6", "name": "insert_remove_bookshelf", "frame_count": 10, "fps": 5.0}
            behavior = {"task_label": "insert_remove_bookshelf", "medium": [
                {
                    "start_frame": 0,
                    "end_frame": 4,
                    "description": "pick up the book",
                    "confidence": 0.9,
                    "boundary_source": "joint_refined",
                },
                {
                    "start_frame": 5,
                    "end_frame": 9,
                    "description": "place the book",
                    "confidence": 0.8,
                    "boundary_source": "vlm",
                },
            ], "fine": [
                {"start_frame": 0, "end_frame": 1, "phase_label": "reach", "skill": "Reach"},
                {"start_frame": 2, "end_frame": 4, "phase_label": "grasp", "skill": "Grasp"},
                {"start_frame": 5, "end_frame": 9, "phase_label": "place", "skill": "Place"},
            ]}
            curation = {
                "segments": [
                    {"start_frame": 0, "end_frame": 2, "state": "valid"},
                    {"start_frame": 3, "end_frame": 4, "state": "invalid"},
                    {"start_frame": 5, "end_frame": 5, "state": "uncertain"},
                    {"start_frame": 6, "end_frame": 9, "state": "valid"},
                ],
                "findings": [
                    {"start_frame": 3, "end_frame": 4, "stage": "c3", "reason": "hand outside frame"},
                    {"start_frame": 5, "end_frame": 5, "stage": "c3", "reason": "partial hand visibility"},
                ],
                "artifact_path": str(root / "episode.curation.alice"),
            }

            result = export_episode(
                output_root,
                manifest,
                episode,
                {"frame_count": 10, "fps": 5.0},
                curation,
                behavior,
                lambda _value, _message: None,
                output_format="subtask_json",
                run_id="run-1",
                timeline_id="timeline-1",
            )

            path = output_root / "episodes" / "ep6" / "subtasks.json"
            self.assertEqual(path, Path(result["pairs"][0]["subtasks_json"]))
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(SUBTASK_JSON_SCHEMA, payload["schema"])
            self.assertEqual([3, 4], payload["bad_frames"])
            self.assertEqual([5], payload["review_frames"])
            self.assertEqual(2, payload["subtask_count"])
            self.assertEqual((0, 4), (payload["subtasks"][0]["start_frame"], payload["subtasks"][0]["end_frame"]))
            self.assertEqual("medium", payload["subtasks"][0]["level"])
            self.assertEqual(["reach", "grasp"], [item["phase_label"] for item in payload["subtasks"][0]["fine_segments"]])
            self.assertEqual([3, 4], payload["subtasks"][0]["bad_frames"])
            self.assertEqual([5], payload["subtasks"][1]["review_frames"])
            self.assertEqual(["c3:hand outside frame"], payload["bad_frame_ranges"][0]["reasons"])
            index = write_dataset_index(output_root, manifest, result["pairs"], [], output_format="subtask_json")
            combined = json.loads(index.read_text(encoding="utf-8"))
            self.assertEqual({"subtask_json": 1}, combined["output_formats"])

    def test_subtask_json_uses_full_analysis_timeline_after_eis_retiming(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "full"
            manifest = {"id": "dataset", "name": "dataset", "root_path": str(root), "files": []}
            episode = {"id": "ep", "name": "retimed", "frame_count": 120, "fps": 60.0}
            source_positions = [float(index * 2) for index in range(60)]
            curation = {
                "source_video": {
                    "frame_count": 60,
                    "fps": 30.0,
                    "source_frame_positions": source_positions,
                },
                "summary": {"frame_count": 60},
                "segments": [
                    {"start_frame": 0, "end_frame": 29, "state": "valid"},
                    {"start_frame": 30, "end_frame": 30, "state": "invalid"},
                    {"start_frame": 31, "end_frame": 59, "state": "valid"},
                ],
            }

            result = export_episode(
                output_root,
                manifest,
                episode,
                {"frame_count": 60, "fps": 30.0},
                curation,
                {"task_label": "retimed", "segments": []},
                lambda *_: None,
                output_format="subtask_json",
                run_id="run-retimed",
                timeline_id="timeline-retimed",
            )

            payload = json.loads(Path(result["pairs"][0]["subtasks_json"]).read_text(encoding="utf-8"))
            self.assertEqual(60, payload["frame_count"])
            self.assertEqual(30.0, payload["fps"])
            self.assertEqual("full_analysis_video", payload["frame_index_space"])
            self.assertEqual(source_positions, payload["source_frame_positions"])
            self.assertEqual([30], payload["bad_frames"])

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

    def test_idle_is_removed_but_reach_and_other_vlm_phases_remain(self) -> None:
        intervals, summary = filtered_intervals(
            8,
            {"segments": [{"start_frame": 0, "end_frame": 7, "state": "valid"}]},
            {"segments": [
                {"start_frame": 0, "end_frame": 1, "phase_label": "idle"},
                {"start_frame": 2, "end_frame": 3, "phase_label": "reach"},
                {"start_frame": 4, "end_frame": 7, "phase_label": "unknown"},
            ]},
        )

        self.assertEqual([(2, 7)], intervals)
        self.assertEqual(2, summary["removed_vlm_frame_count"])

    def test_short_quality_gaps_are_never_reintroduced_and_tiny_islands_are_dropped(self) -> None:
        intervals, summary = filtered_intervals(
            50,
            {"segments": [
                {"start_frame": 0, "end_frame": 19, "state": "valid"},
                {"start_frame": 20, "end_frame": 21, "state": "invalid"},
                {"start_frame": 22, "end_frame": 41, "state": "valid"},
                {"start_frame": 42, "end_frame": 44, "state": "uncertain"},
                {"start_frame": 45, "end_frame": 47, "state": "valid"},
                {"start_frame": 48, "end_frame": 49, "state": "invalid"},
            ]},
            {"segments": [{"start_frame": 0, "end_frame": 49, "phase_label": "manipulate"}]},
            fps=10.0,
            max_internal_gap_seconds=0.25,
            min_clip_seconds=0.75,
        )

        self.assertEqual([(0, 19), (22, 41)], intervals)
        self.assertEqual(0, summary["merged_gap_frame_count"])
        self.assertEqual(3, summary["dropped_short_fragment_frame_count"])
        self.assertEqual(3, summary["raw_interval_count"])
        self.assertEqual(2, summary["final_interval_count"])

    def test_vlm_removed_gap_is_never_filled_again(self) -> None:
        intervals, summary = filtered_intervals(
            24,
            {"segments": [{"start_frame": 0, "end_frame": 23, "state": "valid"}]},
            {"segments": [
                {"start_frame": 0, "end_frame": 9, "phase_label": "grasp"},
                {"start_frame": 10, "end_frame": 11, "phase_label": "idle"},
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
