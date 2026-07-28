from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from app.curation_pipeline import (
    CURATION_PIPELINE_VERSION,
    CurationJobManager,
    _build_s3_references,
    _episode_records,
    _full_action_config,
    _load_signal_bundle,
    _signal_candidates,
    curation_preflight,
    curation_valid_ranges,
    detect_extreme_values,
    detect_rot6d_jumps,
    detect_sudden_changes,
    estimate_state_action_alignment,
    inspect_video_quality,
    inspect_behavior_state_consistency,
    inspect_instruction_consistency,
    inspect_rot6d_jumps,
    load_curation_report,
    merge_dense_quality_marks,
    repair_isolated_spikes,
    repair_rot6d_spikes,
    repair_s1_bundle,
    resolve_post_vlm_review,
    source_signature,
    source_signatures_match,
)
from app.schemas import CurationJobRequest
from app.schema_profiler import build_inventory


class CurationPipelineTests(unittest.TestCase):
    def test_full_request_locks_one_action_profile_for_every_shard(self) -> None:
        request = CurationJobRequest(
            episode_ids=["ep"],
            full_pipeline=True,
            full_action_profile_id="so100_so101",
            full_action_source_hand="right",
            full_action_coordinate_frame="camera",
            full_action_horizon_frames=3,
        )

        config = _full_action_config(request)

        self.assertEqual("so100_so101", config["profile_id"])
        self.assertEqual("full_pipeline_request", config["source"])

    def test_full_action_is_optional_by_default(self) -> None:
        request = CurationJobRequest(episode_ids=["ep"], full_pipeline=True)

        self.assertIsNone(_full_action_config(request))

    @staticmethod
    def _write_skeletal_hdf5(path: Path, frame_count: int = 12) -> None:
        import h5py

        with h5py.File(path, "w") as handle:
            handle.create_dataset("camera/intrinsic", data=np.eye(3, dtype=np.float32))
            handle.create_dataset("observations/images", data=np.zeros((frame_count, 8, 8, 3), dtype=np.uint8))
            for index, name in enumerate(("camera", "leftHand", "rightHand", "leftForearm", "rightForearm")):
                transforms = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], frame_count, axis=0)
                transforms[:, 0, 3] = np.linspace(0, 1 + index, frame_count)
                handle.create_dataset(f"transforms/{name}", data=transforms)

    def test_local_hdf5_skeleton_stream_runs_s1_s3_without_qwen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "episode.hdf5"
            self._write_skeletal_hdf5(source)
            episode = {
                "id": "ep",
                "name": "ep",
                "type": "video",
                "path": str(root / "episode.mp4"),
                "relative_path": "episode.mp4",
                "episode_key": "ep",
                "fps": 30.0,
                "frame_count": 12,
                "duration": 0.4,
                "width": 8,
                "height": 8,
                "primary_media_file_id": "video",
                "media_streams": [{
                    "file_id": "video",
                    "stream_name": "episode.mp4",
                    "type": "video",
                    "path": str(root / "episode.mp4"),
                    "relative_path": "episode.mp4",
                    "fps": 30.0,
                    "frame_count": 12,
                    "duration": 0.4,
                    "width": 8,
                    "height": 8,
                }],
            }
            inventory = build_inventory(root, [episode], {"episode.hdf5"})
            profiled_streams = [item for item in inventory["candidate_streams"] if item.get("source_path") == "episode.hdf5"]
            manifest = {
                "id": "fixture",
                "root_path": str(root),
                "episodes": [episode],
                "files": [{"id": "h5", "relative_path": "episode.hdf5", "extension": ".hdf5", "episode_id": "ep"}],
                "schema_profile": {"status": "awaiting_vlm", "inventory": inventory, "understanding": None},
            }

            candidates = _signal_candidates(manifest, episode)
            bundle = _load_signal_bundle(manifest, episode, {}, frame_count=12)
            with patch("app.curation_pipeline.get_manifest", return_value=manifest), patch("app.curation_pipeline.load_behavior_annotation", return_value=None):
                preflight = curation_preflight("fixture", "ep")

        self.assertEqual(["transforms/*"], [item["field"] for item in candidates])
        self.assertIn("transforms/*", {item["field"] for item in profiled_streams if item["kind"] == "joint"})
        self.assertNotIn("observations/images", {item["field"] for item in profiled_streams if item["kind"] == "joint"})
        self.assertNotIn("transforms/camera", {item["field"] for item in profiled_streams if item["kind"] == "joint"})
        self.assertEqual("local_schema", candidates[0]["source"])
        self.assertEqual("joint", candidates[0]["kind"])
        self.assertNotIn("transforms/camera", candidates[0]["members"])
        self.assertEqual((12, 12), bundle["joint"].shape)
        self.assertTrue(bundle["semantic_dimensions_known"])
        stages = {item["id"]: item["status"] for item in preflight["stages"]}
        self.assertEqual("ready", stages["s1"])
        self.assertEqual("ready", stages["s3"])

    def test_local_hdf5_explicit_state_action_excludes_image_tensor(self) -> None:
        import h5py

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "episode.h5"
            with h5py.File(source, "w") as handle:
                handle.create_dataset("observations/state", data=np.arange(84, dtype=np.float32).reshape(12, 7))
                handle.create_dataset("action/target_qpos", data=np.arange(84, dtype=np.float32).reshape(12, 7))
                handle.create_dataset("observations/rgb", data=np.zeros((12, 32, 32, 3), dtype=np.uint8))
            episode = {"id": "ep", "frame_count": 12}
            manifest = {
                "root_path": str(root),
                "files": [{"id": "h5", "relative_path": "episode.h5", "extension": ".h5", "episode_id": "ep"}],
                "schema_profile": {"inventory": {"files": []}, "understanding": None},
            }

            candidates = _signal_candidates(manifest, episode)
            bundle = _load_signal_bundle(manifest, episode, {}, frame_count=12)

        self.assertEqual({"joint", "action"}, {item["kind"] for item in candidates})
        self.assertNotIn("observations/rgb", {item["field"] for item in candidates})
        self.assertEqual("absolute", next(item["representation"] for item in candidates if item["kind"] == "action"))
        self.assertEqual((12, 7), bundle["joint"].shape)
        self.assertEqual((12, 7), bundle["action"].shape)

    def test_sudden_change_requires_residual_and_dynamics(self) -> None:
        clean = np.sin(np.linspace(0, 12, 300))[:, None]
        self.assertEqual(0, detect_sudden_changes(clean, 6.0)["event_count"])

        corrupted = clean.copy()
        corrupted[150] += 20
        result = detect_sudden_changes(corrupted, 6.0)
        self.assertTrue(result["mask"][150])
        self.assertGreater(result["event_count"], 0)

    def test_s1_repairs_short_isolated_spike_and_rechecks_detector(self) -> None:
        clean = np.sin(np.linspace(0, 12, 300))[:, None]
        corrupted = clean.copy()
        corrupted[150] += 20
        bundle = {
            "joint": corrupted,
            "action": None,
            "bindings": [{
                "kind": "joint",
                "relative_path": "episode.h5",
                "field": "observations/state",
                "column_start": 0,
                "column_end": 1,
                "dimensions": 1,
                "_source_row_indices": np.arange(300, dtype=np.int64),
            }],
        }

        repaired = repair_s1_bundle(bundle, sigma=6.0, max_gap_frames=5)

        self.assertTrue(repaired["before_mask"].any())
        self.assertFalse(repaired["after_mask"].any())
        self.assertTrue(repaired["repaired_mask"][150])
        self.assertAlmostEqual(clean[150, 0], repaired["values"][150, 0], places=5)
        self.assertTrue(repaired["entries"])

    def test_s1_does_not_repair_long_or_boundary_ranges(self) -> None:
        values = np.linspace(0.0, 1.0, 30)[:, None]
        dimensions = np.ones_like(values, dtype=bool)
        long_mask = np.zeros(30, dtype=bool)
        long_mask[10:16] = True
        boundary_mask = np.zeros(30, dtype=bool)
        boundary_mask[:2] = True

        long = repair_isolated_spikes(values, long_mask, dimensions, max_gap_frames=5)
        boundary = repair_isolated_spikes(values, boundary_mask, dimensions, max_gap_frames=5)

        self.assertFalse(long["cell_mask"].any())
        self.assertFalse(boundary["cell_mask"].any())

    def test_s1_does_not_flatten_a_persistent_state_step(self) -> None:
        values = np.sin(np.linspace(0, 12, 300))[:, None]
        values[150:] += 3.0
        bundle = {
            "joint": values,
            "action": None,
            "bindings": [{
                "kind": "joint",
                "relative_path": "episode.h5",
                "field": "observations/state",
                "column_start": 0,
                "column_end": 1,
                "dimensions": 1,
                "_source_row_indices": np.arange(300, dtype=np.int64),
            }],
        }

        result = repair_s1_bundle(bundle, sigma=6.0, max_gap_frames=5)

        self.assertFalse(result["repaired_mask"].any())
        np.testing.assert_array_equal(values, result["values"])

    def test_rot6d_isolated_spike_uses_rotation_space_interpolation(self) -> None:
        values = self._yaw_rot6d(np.linspace(0.0, 30.0, 120))
        values[60] = 0.0
        detected = detect_rot6d_jumps(values, sigma=6.0)

        repaired = repair_rot6d_spikes(values, detected["mask"], max_gap_frames=5)

        self.assertTrue(repaired["cell_mask"][60].all())
        self.assertEqual(0, detect_rot6d_jumps(repaired["values"], sigma=6.0)["event_count"])

    @staticmethod
    def _yaw_rot6d(angles_degrees: np.ndarray) -> np.ndarray:
        angles = np.radians(np.asarray(angles_degrees, dtype=np.float64))
        rotations = np.zeros((angles.size, 3, 3), dtype=np.float64)
        rotations[:, 0, 0] = np.cos(angles)
        rotations[:, 0, 1] = -np.sin(angles)
        rotations[:, 1, 0] = np.sin(angles)
        rotations[:, 1, 1] = np.cos(angles)
        rotations[:, 2, 2] = 1.0
        return rotations[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)

    def test_rot6d_jump_uses_relative_endpose_rotation(self) -> None:
        angles = np.zeros(120, dtype=np.float64)
        angles[60:] = 120.0

        result = detect_rot6d_jumps(self._yaw_rot6d(angles), sigma=6.0)

        self.assertEqual(1, result["jump_frame_count"])
        self.assertTrue(result["mask"][60])
        self.assertAlmostEqual(120.0, result["max_relative_degrees"], places=5)

    def test_rot6d_jump_accepts_smooth_rotation_and_rejects_degenerate_basis(self) -> None:
        values = self._yaw_rot6d(np.linspace(0.0, 35.0, 120))
        clean = detect_rot6d_jumps(values, sigma=6.0)
        values[40] = 0.0
        invalid = detect_rot6d_jumps(values, sigma=6.0)

        self.assertEqual(0, clean["event_count"])
        self.assertTrue(invalid["mask"][40])
        self.assertEqual(1, invalid["invalid_frame_count"])

    def test_rot6d_bundle_inspection_only_uses_explicit_endpose_semantics(self) -> None:
        values = np.zeros((80, 15), dtype=np.float64)
        values[:, 3:9] = self._yaw_rot6d(np.r_[np.zeros(40), np.full(40, 90.0)])
        bundle = {
            "joint": values,
            "action": None,
            "bindings": [
                {
                    "kind": "joint",
                    "field": "observations/endpose",
                    "modality": "pose",
                    "dimension_names": [f"endpose[{index}]" for index in range(9)],
                    "dimensions": 9,
                    "column_start": 0,
                },
                {
                    "kind": "joint",
                    "field": "observations/embedding",
                    "dimension_names": [f"embedding[{index}]" for index in range(6)],
                    "dimensions": 6,
                    "column_start": 9,
                },
            ],
        }

        result = inspect_rot6d_jumps(bundle, sigma=6.0)

        self.assertEqual(1, result["group_count"])
        self.assertEqual(1, result["event_count"])
        self.assertTrue(result["mask"][40])

    def test_local_endpose_is_profiled_as_absolute_joint_pose(self) -> None:
        import h5py

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "episode.h5"
            with h5py.File(source, "w") as handle:
                handle.create_dataset("observations/endpose", data=np.zeros((12, 9), dtype=np.float32))
            episode = {"id": "ep", "frame_count": 12}
            manifest = {
                "root_path": str(root),
                "files": [{"id": "h5", "relative_path": "episode.h5", "extension": ".h5", "episode_id": "ep"}],
                "schema_profile": {"inventory": {"files": []}, "understanding": None},
            }

            candidates = _signal_candidates(manifest, episode)

        self.assertEqual(1, len(candidates))
        self.assertEqual("joint", candidates[0]["kind"])
        self.assertEqual("pose", candidates[0]["modality"])
        self.assertEqual("absolute", candidates[0]["representation"])

    def test_state_action_alignment_recovers_positive_action_lead(self) -> None:
        rng = np.random.default_rng(4)
        action = np.cumsum(rng.normal(size=(500, 2)), axis=0)
        state = np.vstack([np.repeat(action[:1], 4, axis=0), action[:-4]])

        result = estimate_state_action_alignment(state, action, 50.0, 0.5, 0.65, "absolute")

        self.assertEqual(4, result["lag_frames"])
        self.assertEqual("pass", result["verdict"])
        self.assertGreater(result["directional_agreement"], 0.9)

    def test_state_action_alignment_refuses_unknown_action_semantics(self) -> None:
        values = np.arange(100, dtype=np.float64)[:, None]
        with self.assertRaisesRegex(ValueError, "表示类型未知"):
            estimate_state_action_alignment(values, values, 30.0, 0.5, 0.65, "unknown")

    def test_extreme_value_filter_exempts_known_gripper_dimension(self) -> None:
        values = np.zeros((100, 2), dtype=np.float64)
        values[50, 1] = 100.0

        result = detect_extreme_values(values, 0.1, {1})

        self.assertEqual(0, result["event_count"])
        self.assertFalse(result["mask"][50])

    def test_extreme_value_filter_records_cohort_reference(self) -> None:
        values = np.zeros((20, 1), dtype=np.float64)
        values[-1, 0] = 10.0
        reference = np.zeros((200, 1), dtype=np.float64)

        result = detect_extreme_values(values, 0.1, reference_values=reference, reference_scope="cohort", cohort_id="robot-a")

        self.assertTrue(result["mask"][-1])
        self.assertEqual("cohort", result["reference_scope"])
        self.assertEqual("robot-a", result["cohort_id"])

    def test_qwen_episode_assignment_is_used_for_signal_records(self) -> None:
        manifest = {
            "files": [{"id": "joint-file", "relative_path": "ep2/joints.h5", "episode_id": None}],
            "episode_resolution": {"file_episode_assignments": {"joint-file": "ep-2"}},
        }

        records = _episode_records(manifest, {"id": "ep-2"})

        self.assertEqual(["joint-file"], [item["id"] for item in records])

    def test_qwen_episode_assignment_overrides_local_episode(self) -> None:
        manifest = {
            "files": [{"id": "joint-file", "relative_path": "ep1/joints.h5", "episode_id": "local-ep"}],
            "episode_resolution": {"file_episode_assignments": {"joint-file": "ep-2"}},
        }

        records = _episode_records(manifest, {"id": "ep-2"})

        self.assertEqual(["joint-file"], [item["id"] for item in records])

    def test_qwen_stream_example_expands_to_matching_episode_file(self) -> None:
        source = "collection/ep_0001/mocap/left_joint.h5"
        target = "collection/ep_0002/mocap/left_joint.h5"
        manifest = {
            "files": [
                {"id": "source", "relative_path": source, "episode_id": "ep-1", "episode_key": "collection/ep_0001", "extension": ".h5"},
                {"id": "target", "relative_path": target, "episode_id": "ep-2", "episode_key": "collection/ep_0002", "extension": ".h5"},
            ],
            "schema_profile": {
                "understanding": {
                    "streams": [{
                        "kind": "joint",
                        "source_path": source,
                        "field": "joints",
                        "modality": "position",
                        "side": "left",
                        "confidence": 0.95,
                        "dimension_names": ["j1", "j2"],
                        "embodiment_id": "robot-a",
                    }]
                },
                "inventory": {"files": []},
            },
        }

        candidates = _signal_candidates(manifest, {"id": "ep-2"})

        self.assertEqual(1, len(candidates))
        self.assertEqual(target, candidates[0]["relative_path"])
        self.assertEqual("qwen_schema_template", candidates[0]["source"])
        self.assertEqual(source, candidates[0]["schema_example_path"])
        self.assertEqual("robot-a", candidates[0]["embodiment_id"])

    def test_directory_source_signature_detects_sequence_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence = root / "frames"
            sequence.mkdir()
            (sequence / "000001.jpg").write_bytes(b"frame-one")
            (sequence / "000002.jpg").write_bytes(b"frame-two")
            signature = source_signature(root, "frames")
            self.assertEqual("directory", signature["kind"])
            (sequence / "000002.jpg").write_bytes(b"frame-2x!")
            matches, changed = source_signatures_match(root, [signature], allowed_paths=["frames/000001.jpg", "frames/000002.jpg"])

        self.assertFalse(matches)
        self.assertEqual("frames", changed)

    def test_signal_bundle_uses_selected_media_frame_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "joint.npy"
            source.write_bytes(b"placeholder")
            candidate = {
                "kind": "joint",
                "relative_path": "joint.npy",
                "field": "$",
                "confidence": 1.0,
                "role": "",
                "modality": "",
                "representation": "unknown",
                "dimension_names": ["joint"],
                "gripper_indices": [],
                "embodiment_id": None,
            }
            series = {"values": np.arange(5, dtype=np.float64)[:, None], "row_indices": np.arange(5), "source_count": 5}
            with patch("app.curation_pipeline._signal_candidates", return_value=[candidate]), patch("app.curation_pipeline._read_numeric_series", return_value=series):
                bundle = _load_signal_bundle({"root_path": str(root)}, {"frame_count": 99}, {"streams": []}, frame_count=7)

        self.assertEqual(7, bundle["joint"].shape[0])

    def test_report_loader_rejects_wrong_episode_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            report_root = Path(temporary) / "curation"
            root.mkdir()
            report_root.mkdir()
            source = root / "video.mp4"
            source.write_bytes(b"video")
            manifest = {"id": "fixture", "root_path": str(root), "files": [{"relative_path": "video.mp4"}]}
            report = report_root / "episode-1.curation.alice"
            report.write_text(json.dumps({
                "schema": "alice/paper-curation/v1",
                "dataset_id": "fixture",
                "episode_id": "episode-2",
                "source_signatures": [source_signature(root, "video.mp4")],
            }), encoding="utf-8")
            with patch("app.curation_pipeline.dataset_artifact_dir", return_value=report_root), patch("app.curation_pipeline.get_manifest", return_value=manifest):
                with self.assertRaisesRegex(ValueError, "identity"):
                    load_curation_report("fixture", "episode-1")

    def test_report_loader_marks_non_current_pipeline_versions_for_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            report_root = Path(temporary) / "curation"
            root.mkdir()
            report_root.mkdir()
            source = root / "video.mp4"
            source.write_bytes(b"video")
            manifest = {"id": "fixture", "root_path": str(root), "files": [{"relative_path": "video.mp4"}]}
            report = report_root / "episode-1.curation.alice"
            base_payload = {
                "schema": "alice/paper-curation/v1",
                "dataset_id": "fixture",
                "episode_id": "episode-1",
                "source_signatures": [source_signature(root, "video.mp4")],
            }

            with patch("app.curation_pipeline.dataset_artifact_dir", return_value=report_root), patch("app.curation_pipeline.get_manifest", return_value=manifest):
                for stored_version, expected_version, requires_rerun in ((None, 1, True), (3, 3, True), (CURATION_PIPELINE_VERSION, CURATION_PIPELINE_VERSION, False), (CURATION_PIPELINE_VERSION + 1, CURATION_PIPELINE_VERSION + 1, True)):
                    payload = dict(base_payload)
                    if stored_version is not None:
                        payload["pipeline_version"] = stored_version
                    report.write_text(json.dumps(payload), encoding="utf-8")

                    loaded = load_curation_report("fixture", "episode-1")

                    self.assertEqual(expected_version, loaded["pipeline_version"])
                    self.assertEqual(requires_rerun, loaded["requires_rerun"])

    def test_preflight_does_not_mark_unimplemented_stages_ready(self) -> None:
        episode = {"id": "ep", "name": "ep", "type": "video", "relative_path": "video.mp4", "frame_count": 10, "fps": 10.0}
        manifest = {
            "id": "fixture",
            "episodes": [episode],
            "files": [{"id": "u", "relative_path": "robot.urdf", "extension": ".urdf", "episode_id": "ep"}],
            "schema_profile": {"inventory": {"files": [{"path": "calibration.json"}]}},
        }
        with patch("app.curation_pipeline.get_manifest", return_value=manifest), patch("app.curation_pipeline.load_behavior_annotation", return_value=None):
            payload = curation_preflight("fixture", "ep")

        stages = {item["id"]: item for item in payload["stages"]}
        self.assertEqual("skipped", stages["s4"]["status"])
        self.assertEqual("skipped", stages["s5"]["status"])

    def test_s3_reference_groups_compatible_embodiment_across_episodes(self) -> None:
        episodes = {episode_id: {"id": episode_id, "frame_count": 5} for episode_id in ("ep-1", "ep-2")}
        media = {episode_id: {"frame_count": 5} for episode_id in episodes}

        def bundle(_manifest, episode, _alignment, frame_count=None):
            offset = 0.0 if episode["id"] == "ep-1" else 1.0
            return {
                "joint": np.full((frame_count, 2), offset),
                "action": None,
                "bindings": [{"dimension_names": ["j1", "j2"]}],
                "embodiment_ids": ["robot-a"],
                "semantic_dimensions_known": True,
            }

        with patch("app.curation_pipeline.scan_episode_sensor_alignment", return_value={}), patch("app.curation_pipeline._load_signal_bundle", side_effect=bundle):
            references = _build_s3_references({"root_path": str(Path.cwd())}, episodes, media, list(episodes))

        self.assertEqual("cohort", references["ep-1"]["scope"])
        self.assertEqual(references["ep-1"]["cohort_id"], references["ep-2"]["cohort_id"])
        self.assertEqual(10, references["ep-1"]["matrix"].shape[0])

    def test_source_signature_detects_replaced_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "episode.h5"
            source.write_bytes(b"source-v1" * 4096)
            signature = source_signature(root, source.name)
            self.assertTrue(source_signatures_match(root, [signature])[0])
            source.write_bytes(b"source-v2" * 4096)
            matches, changed = source_signatures_match(root, [signature])

        self.assertFalse(matches)
        self.assertEqual("episode.h5", changed)

    def test_static_video_without_action_is_review_not_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "static.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64))
            checker = ((np.indices((64, 96)).sum(axis=0) % 2) * 255).astype(np.uint8)
            frame = cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)
            for _ in range(30):
                writer.write(frame)
            writer.release()
            request = CurationJobRequest(
                episode_ids=["ep"],
                video_sample_fps=5,
                blur_laplacian_threshold=5,
                static_duration_seconds=1,
            )
            result = inspect_video_quality(
                {"type": "video", "path": str(path), "frame_count": 30, "fps": 10.0},
                None,
                request,
                lambda _value, _message: None,
            )

        self.assertFalse(result["invalid_mask"].any())
        self.assertTrue(result["review_mask"].any())

    def test_dense_quality_marks_merge_below_point_three_seconds_only(self) -> None:
        mask = np.zeros(16, dtype=bool)
        mask[[0, 3, 12]] = True

        merged = merge_dense_quality_marks(mask, fps=30.0, gap_seconds=0.3)

        self.assertTrue(merged[0:4].all())
        self.assertFalse(merged[4:12].any())
        self.assertTrue(merged[12])

    def test_c2_reviews_active_vlm_segment_without_state_motion(self) -> None:
        behavior = {"segments": [{"start_frame": 0, "end_frame": 29, "phase_label": "grasp"}]}
        result = inspect_behavior_state_consistency(
            behavior,
            {"joint": np.zeros((30, 3), dtype=np.float64), "action": None, "action_representation": "unknown"},
            np.ones(30, dtype=bool),
            10.0,
        )

        self.assertEqual("warning", result["status"])
        self.assertTrue(result["review_mask"].all())
        self.assertEqual(1, result["metrics"]["mismatch_segment_count"])

    def test_c1_reviews_explicit_task_mismatch(self) -> None:
        result = inspect_instruction_consistency(
            {"task_label": "remove_lid", "confidence": 0.9},
            {"name": "insert_usb"},
            np.ones(12, dtype=bool),
            10.0,
        )

        self.assertEqual("warning", result["status"])
        self.assertTrue(result["review_mask"].all())

    def test_c1_accepts_matching_task_tokens(self) -> None:
        result = inspect_instruction_consistency(
            {"task_label": "insert_usb", "confidence": 0.9},
            {"name": "insert_usb_ep1"},
            np.ones(12, dtype=bool),
            10.0,
        )

        self.assertEqual("completed", result["status"])
        self.assertFalse(result["review_mask"].any())

    def test_c1_c2_clear_precheck_review_only_after_both_checks_run(self) -> None:
        precheck_review = np.ones(6, dtype=bool)
        clear = np.zeros(6, dtype=bool)
        completed = {"status": "completed", "review_mask": clear}
        skipped = {"status": "skipped", "review_mask": clear}

        self.assertFalse(resolve_post_vlm_review(precheck_review, completed, completed).any())
        self.assertTrue(resolve_post_vlm_review(precheck_review, completed, skipped).all())

    def test_main_curation_job_orders_precheck_vlm_then_c2(self) -> None:
        manager = CurationJobManager()
        manager._jobs["job"] = {"id": "job", "dataset_id": "fixture", "status": "queued"}
        episode = {"id": "ep", "name": "episode", "frame_count": 20, "fps": 10.0}
        manifest = {"id": "fixture", "episodes": [episode]}
        media = {"file_id": "video", "frame_count": 20, "fps": 10.0}
        preliminary = {
            "segments": [{"start_frame": 0, "end_frame": 4, "state": "invalid"}, {"start_frame": 5, "end_frame": 14, "state": "valid"}, {"start_frame": 15, "end_frame": 19, "state": "invalid"}],
            "pre_vlm_segments": [{"start_frame": 0, "end_frame": 4, "state": "invalid"}, {"start_frame": 5, "end_frame": 14, "state": "valid"}, {"start_frame": 15, "end_frame": 19, "state": "invalid"}],
        }
        behavior = {"task_label": "pick", "segments": [], "sampling": {"allowed_ranges": [{"start_frame": 5, "end_frame": 14}]}}
        final = {"artifact_path": "result.alice", "summary": {}, "stages": []}
        order: list[str] = []

        def run_precheck(*args, **kwargs):
            order.append("precheck")
            self.assertFalse(kwargs["behavior_checks"])
            return preliminary

        def run_vlm(*args, **kwargs):
            order.append("vlm")
            self.assertEqual([(5, 14)], kwargs["analysis_frame_ranges"])
            return behavior

        def run_c2(*args, **kwargs):
            order.append("c2")
            self.assertEqual("completed", kwargs["vlm_status"])
            return final

        request = CurationJobRequest(episode_ids=["ep"], media_file_ids={"ep": "video"})
        with (
            patch("app.curation_pipeline.get_manifest", return_value=manifest),
            patch("app.curation_pipeline._build_s3_references", return_value={}),
            patch("app.curation_pipeline.run_episode_curation", side_effect=run_precheck),
            patch("app.curation_pipeline.load_behavior_annotation", return_value=None),
            patch("app.curation_pipeline.annotate_episode_behavior", side_effect=run_vlm),
            patch("app.curation_pipeline.finalize_episode_curation", side_effect=run_c2),
            patch("app.curation_pipeline.registry", SimpleNamespace(has_vlm=True)),
        ):
            manager._run("job", "fixture", ["ep"], {"ep": media}, request)

        self.assertEqual(["precheck", "vlm", "c2"], order)
        completed_job = manager.get("job")
        self.assertEqual("complete", completed_job["status"])
        self.assertEqual(1, completed_job["result"]["vlm_requested_count"])
        self.assertEqual(0, completed_job["result"]["vlm_reused_count"])
        self.assertEqual([(5, 14)], curation_valid_ranges(preliminary, before_c2=True))

    def test_full_job_adds_smoothing_and_pair_export_around_main_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = CurationJobManager()
            manager._jobs["full-job"] = {"id": "full-job", "dataset_id": "fixture", "status": "queued"}
            episode = {"id": "ep", "name": "episode", "frame_count": 20, "fps": 10.0}
            manifest = {"id": "fixture", "name": "fixture", "root_path": str(root), "episodes": [episode]}
            media = {"file_id": "video", "path": str(root / "source.mp4"), "frame_count": 20, "fps": 10.0}
            preliminary = {
                "segments": [{"start_frame": 0, "end_frame": 19, "state": "valid"}],
                "pre_vlm_segments": [{"start_frame": 0, "end_frame": 19, "state": "valid"}],
            }
            behavior = {"task_label": "pick", "segments": [], "sampling": {"allowed_ranges": [{"start_frame": 0, "end_frame": 19}]}}
            final = {"artifact_path": "result.alice", "summary": {}, "stages": []}
            order: list[str] = []
            smoothed = {"output_video": str(root / "smoothed.mp4"), "summary": {"frame_count": 20, "fps": 10.0}}

            request = CurationJobRequest(
                episode_ids=["ep"],
                media_file_ids={"ep": "video"},
                full_pipeline=True,
                full_action_profile_id="generic_bimanual_pose",
            )
            with (
                patch("app.curation_pipeline.get_manifest", return_value=manifest),
                patch("app.curation_pipeline._build_s3_references", return_value={}),
                patch("app.curation_pipeline.smooth_video", side_effect=lambda *args, **kwargs: (order.append("smooth") or smoothed)),
                patch("app.curation_pipeline.generate_episode_action", side_effect=lambda *args, **kwargs: (order.append("action") or {
                    "profile": {"id": "generic_bimanual_pose"},
                    "config": {"profile_id": "generic_bimanual_pose"},
                    "summary": {"action_count": 17},
                    "artifact_path": "action.hdf5",
                    "reused": False,
                })),
                patch("app.curation_pipeline.validate_episode_action_mapping", return_value={
                    "verdict": "pass",
                    "invalid_mask": np.zeros(20, dtype=bool),
                    "profile_id": "generic_bimanual_pose",
                }),
                patch("app.curation_pipeline.run_episode_curation", side_effect=lambda *args, **kwargs: (order.append("precheck") or preliminary)),
                patch("app.curation_pipeline.load_behavior_annotation", return_value=None),
                patch("app.curation_pipeline.annotate_episode_behavior", side_effect=lambda *args, **kwargs: (order.append("vlm") or behavior)),
                patch("app.curation_pipeline.finalize_episode_curation", side_effect=lambda *args, **kwargs: (order.append("c2") or final)),
                patch("app.curation_pipeline.export_episode", side_effect=lambda *args, **kwargs: (order.append("export") or {"pairs": [{"frame_count": 4}], "filtering": {}, "transform_source": "episode.hdf5", "category": "pick"})),
                patch("app.curation_pipeline.write_dataset_index", return_value=root / "dataset.json"),
                patch("app.curation_pipeline.registry", SimpleNamespace(has_vlm=True)),
            ):
                manager._run("full-job", "fixture", ["ep"], {"ep": media}, request)

            self.assertEqual({"smooth", "action"}, set(order[:2]))
            self.assertEqual(["precheck", "vlm", "c2", "export"], order[2:])
            result = manager.get("full-job")["result"]
            self.assertEqual("full_pipeline", result["operation"])
            self.assertEqual(1, result["pair_count"])
            self.assertEqual(1, result["vlm_requested_count"])
            self.assertEqual(str(root / "output"), result["output_root"])
            self.assertEqual("generic_bimanual_pose", result["action_config"]["profile_id"])
            self.assertEqual("pass", result["items"][0]["action_s2"]["validation"]["verdict"])

    def test_full_job_does_not_generate_action_without_an_explicit_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = CurationJobManager()
            manager._jobs["full-no-action"] = {"id": "full-no-action", "dataset_id": "fixture", "status": "queued"}
            episode = {"id": "ep", "name": "episode", "frame_count": 20, "fps": 10.0}
            manifest = {"id": "fixture", "name": "fixture", "root_path": str(root), "episodes": [episode]}
            media = {"file_id": "video", "path": str(root / "source.mp4"), "frame_count": 20, "fps": 10.0}
            preliminary = {
                "segments": [{"start_frame": 0, "end_frame": 19, "state": "invalid"}],
                "pre_vlm_segments": [{"start_frame": 0, "end_frame": 19, "state": "invalid"}],
            }
            final = {"artifact_path": "result.alice", "summary": {}, "stages": []}
            smoothed = {"output_video": str(root / "smoothed.mp4"), "summary": {"frame_count": 20, "fps": 10.0}}
            request = CurationJobRequest(episode_ids=["ep"], media_file_ids={"ep": "video"}, full_pipeline=True)

            with (
                patch("app.curation_pipeline.get_manifest", return_value=manifest),
                patch("app.curation_pipeline._build_s3_references", return_value={}),
                patch("app.curation_pipeline.smooth_video", return_value=smoothed) as smooth,
                patch("app.curation_pipeline.generate_episode_action", side_effect=AssertionError("Action must stay optional")),
                patch("app.curation_pipeline.run_episode_curation", return_value=preliminary),
                patch("app.curation_pipeline.load_behavior_annotation", return_value=None),
                patch("app.curation_pipeline.finalize_episode_curation", return_value=final),
                patch("app.curation_pipeline.write_dataset_index", return_value=root / "dataset.json"),
                patch("app.curation_pipeline.registry", SimpleNamespace(has_vlm=True)),
            ):
                manager._run("full-no-action", "fixture", ["ep"], {"ep": media}, request)

            job = manager.get("full-no-action")
            self.assertEqual("complete", job["status"])
            self.assertIsNone(job["result"]["action_config"])
            self.assertNotIn("action_s2", job["result"]["items"][0])
            smooth.assert_called_once()


if __name__ == "__main__":
    unittest.main()
