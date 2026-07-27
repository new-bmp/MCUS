from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import h5py

from app.joint_overlay import _edges, _h5_points, _semantic_point_indices, _side, joint_overlay_geometry


class JointOverlayTopologyTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
