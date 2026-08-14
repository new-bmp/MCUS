from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import h5py
import pyarrow as pa
import pyarrow.parquet as parquet

from app.joint_overlay import _candidate_sources, _edges, _h5_points, _semantic_point_indices, _side, joint_overlay_geometry, joint_overlay_status
from app.lerobot_export import HAND_21_JOINT_NAMES
from app.mano21 import MANO21_LAYOUT_VERSION, select_mano21_points, side_hand_joint_names


class JointOverlayTopologyTests(unittest.TestCase):
    def test_full_egodex_skeleton_is_converted_to_two_mano21_hands(self) -> None:
        labels = ["hip", "leftForearm", "leftIndexFingerMetacarpal"]
        labels.extend(side_hand_joint_names("left"))
        labels.extend(["rightForearm", "rightLittleFingerMetacarpal"])
        labels.extend(side_hand_joint_names("right"))
        points = np.arange(len(labels) * 3, dtype=np.float64).reshape(len(labels), 3)

        selected, selected_labels, source_indices, sides = select_mano21_points(points, labels)

        self.assertEqual((42, 3), selected.shape)
        self.assertEqual(("left", "right"), sides)
        self.assertEqual(list(side_hand_joint_names("left")), selected_labels[:21])
        self.assertEqual(list(side_hand_joint_names("right")), selected_labels[21:])
        self.assertNotIn(labels.index("leftIndexFingerMetacarpal"), source_indices.tolist())
        self.assertNotIn(labels.index("rightLittleFingerMetacarpal"), source_indices.tolist())
        self.assertNotIn(labels.index("leftForearm"), source_indices.tolist())

    def test_overlay_reports_mano21_and_restarts_indices_for_each_hand(self) -> None:
        labels = ["hip", *side_hand_joint_names("left"), "leftIndexFingerMetacarpal", *side_hand_joint_names("right")]
        points = np.zeros((len(labels), 3), dtype=np.float64)
        points[:, 0] = np.linspace(10.0, 80.0, len(labels))
        points[:, 1] = 40.0
        points[:, 2] = 1.0
        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "pose.h5"
            source.touch()
            manifest = {"id": "mano-overlay", "root_path": temp_dir}
            episode = {"id": "episode", "fps": 30.0}
            status = {"available": True, "source_path": source.name, "source_kind": "hdf5"}
            candidate = {"path": source.name, "field": None, "source": "hdf5"}
            alignment = {"valid": True, "mode": "identity", "alignment_multiplier": 1.0}
            h5_result = (points, labels, None, None, "camera")
            with (
                patch("app.joint_overlay.joint_overlay_status", return_value=status),
                patch("app.joint_overlay._candidate_sources", return_value=[candidate]),
                patch("app.joint_overlay.map_video_frame_to_sensor", return_value=(0, alignment)),
                patch("app.joint_overlay._h5_points", return_value=h5_result),
                patch("app.joint_overlay._project", side_effect=lambda selected, *_args: selected[:, :2]),
            ):
                geometry = joint_overlay_geometry(manifest, episode, 0, 100, 80, mode="raw")

        self.assertEqual("mano21", geometry["skeleton_schema"])
        self.assertEqual(MANO21_LAYOUT_VERSION, geometry["layout_version"])
        self.assertEqual(42, geometry["joint_count"])
        self.assertEqual(40, len(geometry["edges"]))
        self.assertEqual(list(range(21)) * 2, [point["source_index"] for point in geometry["points"]])

    def test_review_correction_and_raw_hdf5_are_both_kept_as_candidates(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corrected = root / "corrected.hdf5"
            corrected.touch()
            manifest = {
                "id": f"candidate-modes-{root.name}",
                "root_path": str(root),
                "sidecar_path": str(root / ".alicePD"),
                "files": [{"relative_path": "pose.hdf5", "extension": ".hdf5", "episode_id": "episode"}],
            }
            episode = {"id": "episode"}
            applied = {"source_relative_path": "pose.hdf5", "path": str(corrected), "application_id": "apply-1"}

            with patch("app.joint_overlay.review_projection_source", return_value=applied):
                candidates = _candidate_sources(manifest, episode)

        matching = [item for item in candidates if item["path"] == "pose.hdf5"]
        self.assertEqual({"projection_correction", "hdf5"}, {item["source"] for item in matching})

    def test_raw_and_corrected_geometry_select_exact_source_when_paths_match(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "pose.hdf5"
            corrected = root / "corrected.hdf5"
            raw.touch()
            corrected.touch()
            manifest = {"id": "mode-selection", "root_path": str(root)}
            episode = {"id": "episode", "fps": 30.0}
            candidates = [
                {"path": raw.name, "absolute_path": str(corrected), "field": None, "source": "projection_correction"},
                {"path": raw.name, "field": None, "source": "hdf5"},
            ]
            alignment = {
                "valid": True,
                "mode": "identity",
                "alignment_multiplier": 1.0,
                "sensor_hz": 30.0,
                "physical_hz": 60.0,
            }

            def h5_points(path: Path, _: int):
                x = 70.0 if path == corrected else 20.0
                return np.array([[x, 30.0, 1.0]]), ["joint_00"], None, None, "camera"

            def project(points, *_args):
                return points[:, :2]

            with (
                patch("app.joint_overlay._candidate_sources", return_value=candidates),
                patch("app.joint_overlay.change_is_applied", return_value=False),
                patch("app.joint_overlay.map_video_frame_to_sensor", return_value=(0, alignment)),
                patch("app.joint_overlay._h5_points", side_effect=h5_points),
                patch("app.joint_overlay._project", side_effect=project),
            ):
                raw_geometry = joint_overlay_geometry(manifest, episode, 0, 100, 80, mode="raw")
                corrected_geometry = joint_overlay_geometry(manifest, episode, 0, 100, 80, mode="corrected")

        self.assertEqual("hdf5", raw_geometry["source_kind"])
        self.assertEqual("raw", raw_geometry["overlay_mode"])
        self.assertEqual(20.0, raw_geometry["points"][0]["x"])
        self.assertEqual("identity", raw_geometry["alignment_mode"])
        self.assertEqual("projection_correction", corrected_geometry["source_kind"])
        self.assertEqual("corrected", corrected_geometry["overlay_mode"])
        self.assertEqual(70.0, corrected_geometry["points"][0]["x"])
        self.assertEqual("applied_projection_video_aligned", corrected_geometry["alignment_mode"])
        self.assertEqual(60.0, raw_geometry["clock_hz"])
        self.assertEqual(60.0, corrected_geometry["clock_hz"])
        self.assertEqual(30.0, corrected_geometry["sensor_hz"])
        self.assertEqual(60.0, corrected_geometry["physical_hz"])

    def test_corrected_status_requires_a_generated_review_correction(self) -> None:
        with TemporaryDirectory() as temp_dir:
            manifest = {"id": f"no-correction-{Path(temp_dir).name}", "root_path": temp_dir}
            episode = {"id": "episode"}
            with (
                patch("app.joint_overlay._candidate_sources", return_value=[]),
                patch("app.joint_overlay.change_is_applied", return_value=False),
            ):
                status = joint_overlay_status(manifest, episode, mode="corrected")

        self.assertFalse(status["available"])
        self.assertEqual("corrected", status["overlay_mode"])
        self.assertIn("尚未生成", status["reason"])

    def test_pending_correction_is_available_for_comparison_without_application(self) -> None:
        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "pending.hdf5"
            source.touch()
            manifest = {"id": f"pending-correction-{source.parent.name}", "root_path": temp_dir}
            episode = {"id": "episode", "frame_count": 1}
            candidate = {
                "path": "pose.hdf5",
                "absolute_path": str(source),
                "source": "projection_correction",
                "review_status": "pending",
                "applied": False,
                "frame_count": 1,
            }
            with (
                patch("app.joint_overlay.review_projection_source", return_value={"path": source, "review_status": "pending", "applied": False}),
                patch("app.joint_overlay._overlay_candidates", return_value=[candidate]),
                patch("app.joint_overlay.change_is_applied", return_value=False),
                patch("app.joint_overlay._h5_points", return_value=(np.array([[0.0, 0.0, 1.0]]), ["joint_00"], None, None, "camera")),
            ):
                status = joint_overlay_status(manifest, episode, mode="corrected")

        self.assertTrue(status["available"])
        self.assertEqual("pending", status["projection_review_status"])
        self.assertFalse(status["projection_applied"])

    def test_left_and_right_skeleton_edges_stay_separate_and_acyclic(self) -> None:
        labels = [
            "leftArm", "rightArm", "leftForearm", "rightForearm", "leftHand", "rightHand",
        ]

        edges = _edges(labels)
        parents = list(range(len(labels)))

        def find(node: int) -> int:
            while parents[node] != node:
                node = parents[node]
            return node

        for start, end in edges:
            start_side, end_side = _side(labels[start]), _side(labels[end])
            if start_side != "unknown" and end_side != "unknown":
                self.assertEqual(start_side, end_side)
            start_root, end_root = find(start), find(end)
            self.assertNotEqual(start_root, end_root)
            parents[end_root] = start_root

        self.assertEqual(4, len(edges))

    def test_egodex_fingers_follow_named_parent_child_topology(self) -> None:
        # This deliberately follows HDF5's alphabetical traversal order, not
        # anatomical order. Topology must come from field names.
        labels = [
            "leftHand",
            "leftIndexFingerIntermediateBase",
            "leftIndexFingerIntermediateTip",
            "leftIndexFingerKnuckle",
            "leftIndexFingerMetacarpal",
            "leftIndexFingerTip",
            "leftThumbIntermediateBase",
            "leftThumbIntermediateTip",
            "leftThumbKnuckle",
            "leftThumbTip",
        ]
        lookup = {label: index for index, label in enumerate(labels)}

        edges = set(_edges(labels))

        expected = {
            (lookup["leftHand"], lookup["leftIndexFingerMetacarpal"]),
            (lookup["leftIndexFingerMetacarpal"], lookup["leftIndexFingerKnuckle"]),
            (lookup["leftIndexFingerKnuckle"], lookup["leftIndexFingerIntermediateBase"]),
            (lookup["leftIndexFingerIntermediateBase"], lookup["leftIndexFingerIntermediateTip"]),
            (lookup["leftIndexFingerIntermediateTip"], lookup["leftIndexFingerTip"]),
            (lookup["leftHand"], lookup["leftThumbKnuckle"]),
            (lookup["leftThumbKnuckle"], lookup["leftThumbIntermediateBase"]),
            (lookup["leftThumbIntermediateBase"], lookup["leftThumbIntermediateTip"]),
            (lookup["leftThumbIntermediateTip"], lookup["leftThumbTip"]),
        }
        self.assertEqual(expected, edges)

        wrist = lookup["leftHand"]
        wrist_children = {end for start, end in edges if start == wrist}
        self.assertEqual(
            {lookup["leftIndexFingerMetacarpal"], lookup["leftThumbKnuckle"]},
            wrist_children,
        )

    def test_full_egodex_hands_are_separate_acyclic_trees(self) -> None:
        finger_segments = {
            "LittleFinger": ("Metacarpal", "Knuckle", "IntermediateBase", "IntermediateTip", "Tip"),
            "RingFinger": ("Metacarpal", "Knuckle", "IntermediateBase", "IntermediateTip", "Tip"),
            "MiddleFinger": ("Metacarpal", "Knuckle", "IntermediateBase", "IntermediateTip", "Tip"),
            "IndexFinger": ("Metacarpal", "Knuckle", "IntermediateBase", "IntermediateTip", "Tip"),
            "Thumb": ("Knuckle", "IntermediateBase", "IntermediateTip", "Tip"),
        }
        labels: list[str] = []
        for side in ("right", "left"):
            labels.extend((f"{side}Shoulder", f"{side}Arm", f"{side}Forearm", f"{side}Hand"))
            for finger, segments in finger_segments.items():
                labels.extend(f"{side}{finger}{segment}" for segment in segments)
        labels = sorted(labels)

        edges = _edges(labels)
        parents = list(range(len(labels)))

        def find(node: int) -> int:
            while parents[node] != node:
                node = parents[node]
            return node

        for start, end in edges:
            self.assertEqual(_side(labels[start]), _side(labels[end]))
            start_root, end_root = find(start), find(end)
            self.assertNotEqual(start_root, end_root)
            parents[end_root] = start_root

        # 3 arm edges + 5*4 non-thumb edges + 4 thumb edges per side.
        self.assertEqual(54, len(edges))

    def test_unknown_fields_do_not_get_index_based_edges(self) -> None:
        self.assertEqual([], _edges(["joint_00", "joint_01", "joint_02"]))

    def test_named_topology_filters_isolated_body_points(self) -> None:
        labels = [
            "hip", "spine1", "neck1", "leftHand", "leftForearm",
            "leftIndexFingerMetacarpal", "leftIndexFingerKnuckle",
        ]
        edges = _edges(labels)
        visible = _semantic_point_indices(labels, edges)

        self.assertIsNotNone(visible)
        self.assertNotIn(labels.index("hip"), visible)
        self.assertNotIn(labels.index("spine1"), visible)
        self.assertNotIn(labels.index("neck1"), visible)
        self.assertEqual(
            {labels.index("leftHand"), labels.index("leftForearm"),
             labels.index("leftIndexFingerMetacarpal"), labels.index("leftIndexFingerKnuckle")},
            visible,
        )

    def test_unknown_keypoint_array_keeps_all_points(self) -> None:
        labels = ["joint_00", "joint_01", "joint_02"]
        self.assertIsNone(_semantic_point_indices(labels))

    def test_filtered_points_keep_source_data_indices(self) -> None:
        projected = np.array([
            [10.0, 10.0],
            [np.nan, 20.0],
            [30.0, 30.0],
            [200.0, 200.0],
        ])
        labels = ["joint_00", "joint_01", "joint_02", "joint_03"]
        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "pose.h5"
            source.touch()
            manifest = {"id": "dataset", "root_path": temp_dir}
            episode = {"id": "episode", "fps": 30.0}
            status = {"available": True, "source_path": source.name}
            candidate = {"path": source.name, "field": "pose", "modality": "transform"}
            alignment = {"valid": True, "mode": "identity", "alignment_multiplier": 1.0}
            h5_result = (np.zeros((4, 3)), labels, None, None, "camera")
            with (
                patch("app.joint_overlay.joint_overlay_status", return_value=status),
                patch("app.joint_overlay._candidate_sources", return_value=[candidate]),
                patch("app.joint_overlay.map_video_frame_to_sensor", return_value=(0, alignment)),
                patch("app.joint_overlay._h5_points", return_value=h5_result),
                patch("app.joint_overlay._project", return_value=projected),
            ):
                geometry = joint_overlay_geometry(manifest, episode, 0, 100, 100)

        self.assertEqual([0, 2], [point["source_index"] for point in geometry["points"]])

    def test_explicit_hdf5_joint_order_reorders_values_without_renaming_fields(self) -> None:
        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "pose.hdf5"
            with h5py.File(source, "w") as handle:
                transforms = handle.create_group("transforms")
                for index, name in enumerate(("leftHand", "leftIndexFingerKnuckle", "leftIndexFingerMetacarpal")):
                    value = np.eye(4, dtype=np.float32)[None, ...]
                    value[0, 0, 3] = index
                    transforms.create_dataset(name, data=value)
                transforms.attrs["joint_order"] = json.dumps([
                    "leftHand", "leftIndexFingerMetacarpal", "leftIndexFingerKnuckle",
                ])

            points, labels, _, _, _ = _h5_points(source, 0)
            with h5py.File(source, "r") as handle:
                stored_fields = set(handle["transforms"].keys())

        self.assertEqual(
            ["leftHand", "leftIndexFingerMetacarpal", "leftIndexFingerKnuckle"],
            labels,
        )
        self.assertEqual([0.0, 2.0, 1.0], points[:, 0].tolist())
        self.assertEqual({"leftHand", "leftIndexFingerKnuckle", "leftIndexFingerMetacarpal"}, stored_fields)

    def test_hdf5_playback_reuses_cached_points_camera_and_intrinsics(self) -> None:
        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "pose.hdf5"
            with h5py.File(source, "w") as handle:
                transforms = handle.create_group("transforms")
                camera = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 2, axis=0)
                camera[1, 0, 3] = 0.25
                transforms.create_dataset("camera", data=camera)
                for offset, name in enumerate(("leftHand", "leftIndexFingerKnuckle")):
                    values = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 2, axis=0)
                    values[:, 0, 3] = 0.01 * offset
                    transforms.create_dataset(name, data=values)
                intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, ...], 2, axis=0)
                intrinsics[0, 0, 0] = 800.0
                intrinsics[1, 0, 0] = 900.0
                handle.create_group("camera").create_dataset("intrinsic", data=intrinsics)

            _h5_points(source, 0)
            with patch("h5py.File", side_effect=AssertionError("playback reopened HDF5")):
                points, labels, camera_ext, intrinsic, coordinate = _h5_points(source, 1)

        self.assertEqual(2, len(points))
        self.assertEqual(["leftHand", "leftIndexFingerKnuckle"], labels)
        self.assertAlmostEqual(0.25, float(camera_ext[0, 3]))
        self.assertAlmostEqual(900.0, float(intrinsic[0, 0]))
        self.assertEqual("world", coordinate)

    def test_lerobot_parquet_uses_camera_transform_and_egodex_intrinsic_fallback(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "data" / "chunk-000" / "episode_000000.parquet"
            data_path.parent.mkdir(parents=True)
            (root / "meta").mkdir()
            (root / "meta" / "info.json").write_text(json.dumps({
                "robot_type": "egodex_bimanual_hands_body",
                "hand_joint_names": list(HAND_21_JOINT_NAMES),
                "body_joint_names": [],
                "features": {
                    "observation.images.main": {"shape": [80, 100, 3]},
                    "observation.left_hand.transforms": {"names": list(HAND_21_JOINT_NAMES)},
                },
            }), encoding="utf-8")
            left = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 21, axis=0)
            right = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 21, axis=0)
            left[:, 0, 3] = np.arange(21, dtype=np.float32) * 0.002
            right[:, 0, 3] = -np.arange(21, dtype=np.float32) * 0.002
            left[:, 2, 3] = 1.0
            right[:, 2, 3] = 1.0
            camera = np.eye(4, dtype=np.float32)
            table = pa.Table.from_arrays([
                pa.array([left.reshape(-1).tolist()], type=pa.list_(pa.float32(), 21 * 4 * 4)),
                pa.array([right.reshape(-1).tolist()], type=pa.list_(pa.float32(), 21 * 4 * 4)),
                pa.array([camera.reshape(-1).tolist()], type=pa.list_(pa.float32(), 4 * 4)),
            ], names=[
                "observation.left_hand.transforms",
                "observation.right_hand.transforms",
                "observation.camera.transform",
            ])
            parquet.write_table(table, data_path)
            relative = data_path.relative_to(root).as_posix()
            manifest = {
                "id": f"lerobot-{root.name}",
                "root_path": str(root),
                "files": [{"relative_path": relative, "extension": ".parquet", "episode_id": "episode"}],
            }
            episode = {"id": "episode", "fps": 30.0, "frame_count": 1}

            alignment = {
                "valid": True,
                "mode": "paired_frame_index",
                "alignment_multiplier": 1.0,
                "sensor_hz": 30.0,
                "physical_hz": 30.0,
            }
            with (
                patch("app.joint_overlay.change_is_applied", return_value=False),
                patch("app.joint_overlay.map_video_frame_to_sensor", return_value=(0, alignment)),
            ):
                status = joint_overlay_status(manifest, episode)
                geometry = joint_overlay_geometry(manifest, episode, 0, 100, 80)

        self.assertTrue(status["available"])
        self.assertEqual(relative, status["source_path"])
        self.assertEqual(42, status["joint_count"])
        self.assertEqual(42, geometry["joint_count"])
        self.assertEqual(40, len(geometry["edges"]))
        self.assertAlmostEqual(50.0, geometry["points"][0]["x"], places=3)
        self.assertAlmostEqual(40.0, geometry["points"][0]["y"], places=3)

    def test_overlay_returns_missing_geometry_when_t0_has_no_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "pose.h5"
            source.touch()
            manifest = {"id": "unmapped", "root_path": temporary}
            episode = {"id": "episode", "fps": 30.0}
            candidate = {"path": source.name, "field": "skeleton", "source": "hdf5"}
            with (
                patch("app.joint_overlay.joint_overlay_status", return_value={"available": True, "source_path": source.name, "source_kind": "hdf5"}),
                patch("app.joint_overlay._overlay_candidates", return_value=[candidate]),
                patch("app.joint_overlay.map_video_frame_to_sensor", side_effect=KeyError(source.name)),
            ):
                geometry = joint_overlay_geometry(manifest, episode, 0, 100, 80)

        self.assertFalse(geometry["alignment_valid"])
        self.assertEqual("unmapped", geometry["alignment_mode"])
        self.assertIsNone(geometry["sensor_index"])
        self.assertEqual([], geometry["points"])


if __name__ == "__main__":
    unittest.main()
