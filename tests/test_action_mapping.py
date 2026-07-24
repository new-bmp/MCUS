from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from app.action_mapping import ACTION_MAPPING_PROFILES, build_action_arrays, generate_episode_action, validate_episode_action_mapping
from app.full_export import MANO_44_JOINT_NAMES
from app.main import app
from app.schemas import ActionMappingRequest


def _transforms(frame_count: int = 8) -> dict[str, np.ndarray]:
    values = {
        name: np.repeat(np.eye(4, dtype=np.float32)[None], frame_count, axis=0)
        for name in (
            "camera",
            "leftHand", "leftThumbTip", "leftIndexFingerTip", "leftMiddleFingerKnuckle",
            "rightHand", "rightThumbTip", "rightIndexFingerTip", "rightMiddleFingerKnuckle",
        )
    }
    time = np.arange(frame_count, dtype=np.float32)
    values["camera"][:, 0, 3] = time * 0.01
    values["leftHand"][:, 0, 3] = time * 0.10
    values["rightHand"][:, 1, 3] = time * 0.20
    for side in ("left", "right"):
        hand = values[f"{side}Hand"][:, :3, 3]
        values[f"{side}ThumbTip"][:, :3, 3] = hand + np.array([0.02, 0.00, 0.00], dtype=np.float32)
        values[f"{side}IndexFingerTip"][:, :3, 3] = hand + np.array([0.04, 0.00, 0.00], dtype=np.float32)
        values[f"{side}MiddleFingerKnuckle"][:, :3, 3] = hand + np.array([0.00, 0.10, 0.00], dtype=np.float32)
    return values


class ActionMappingTests(unittest.TestCase):
    def test_builds_absolute_and_delta_robot_actions(self) -> None:
        transforms = _transforms()
        observation, absolute, grips = build_action_arrays(
            transforms,
            ACTION_MAPPING_PROFILES["generic_bimanual_pose"],
            "right",
            "camera",
            2,
        )

        self.assertEqual((6, 20), observation.shape)
        self.assertEqual((6, 20), absolute.shape)
        self.assertAlmostEqual(0.2, float(absolute[0, 0]), places=5)
        self.assertEqual((8,), grips["left"].shape)
        self.assertTrue(np.isfinite(absolute).all())

        _, bimanual_delta, _ = build_action_arrays(
            transforms,
            ACTION_MAPPING_PROFILES["generic_bimanual_delta"],
            "right",
            "camera",
            2,
        )
        self.assertEqual((6, 14), bimanual_delta.shape)
        self.assertAlmostEqual(0.2, float(bimanual_delta[0, 0]), places=5)

        _, panda, _ = build_action_arrays(
            transforms,
            ACTION_MAPPING_PROFILES["franka_panda"],
            "left",
            "world",
            2,
        )
        self.assertEqual((6, 7), panda.shape)
        self.assertAlmostEqual(0.2, float(panda[0, 0]), places=5)

    def test_generates_atomic_hdf5_report_and_incremental_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "episode.hdf5"
            source_before = None
            transforms = _transforms()
            with h5py.File(source_path, "w") as output:
                for name in (*MANO_44_JOINT_NAMES, "camera"):
                    value = transforms.get(name)
                    if value is None:
                        value = np.repeat(np.eye(4, dtype=np.float32)[None], 8, axis=0)
                    output.create_dataset(f"transforms/{name}", data=value)
            source_before = source_path.read_bytes()
            manifest = {
                "id": "dataset",
                "name": "dataset",
                "root_path": str(root),
                "sidecar_path": str(root / "sidecar"),
                "files": [{"id": "h5", "relative_path": source_path.name, "episode_id": "ep"}],
                "episode_resolution": {"file_episode_assignments": {"h5": "ep"}},
            }
            episode = {"id": "ep", "name": "episode_0", "frame_count": 8, "fps": 30.0}
            request = ActionMappingRequest(
                episode_ids=["ep"],
                profile_id="generic_bimanual_delta",
                coordinate_frame="camera",
                horizon_frames=2,
            )

            with patch("app.action_mapping.dataset_artifact_dir", return_value=root / "sidecar" / "actions"):
                report = generate_episode_action("dataset", manifest, episode, request)
                reused = generate_episode_action("dataset", manifest, episode, request)

            self.assertFalse(report["reused"])
            self.assertTrue(reused["reused"])
            self.assertEqual(source_before, source_path.read_bytes())
            artifact = Path(report["artifact_path"])
            with h5py.File(artifact, "r") as output:
                self.assertEqual((6, 20), output["observation/state"].shape)
                self.assertEqual((6, 14), output["action"].shape)
                self.assertEqual([0, 1, 2, 3, 4, 5], output["frame_index"][:].tolist())
                self.assertEqual([2, 3, 4, 5, 6, 7], output["target_frame_index"][:].tolist())
                self.assertEqual("current_camera_reference", output.attrs["coordinate_semantics"])
            index = json.loads(Path(report["index_path"]).read_text(encoding="utf-8"))
            self.assertEqual(1, index["episode_count"])
            self.assertEqual(6, index["action_row_count"])

            with patch("app.action_mapping.get_manifest", return_value=manifest), patch(
                "app.action_mapping.dataset_artifact_dir", return_value=root / "sidecar" / "actions"
            ):
                validation = validate_episode_action_mapping("dataset", manifest, episode)
            self.assertEqual("pass", validation["verdict"])
            self.assertEqual(0, int(validation["invalid_mask"].sum()))

            with h5py.File(artifact, "r+") as output:
                output["action"][2, 0] += 0.25
            with patch("app.action_mapping.get_manifest", return_value=manifest), patch(
                "app.action_mapping.dataset_artifact_dir", return_value=root / "sidecar" / "actions"
            ):
                corrupted = validate_episode_action_mapping("dataset", manifest, episode)
            self.assertEqual("reject_candidate", corrupted["verdict"])
            self.assertEqual(1, corrupted["mismatch_row_count"])
            self.assertTrue(corrupted["invalid_mask"][2])
            self.assertTrue(corrupted["invalid_mask"][4])

    def test_api_and_frontend_expose_action_mapping(self) -> None:
        routes = {(route.path, method) for route in app.routes for method in getattr(route, "methods", set())}
        self.assertIn(("/api/action-mappings/profiles", "GET"), routes)
        self.assertIn(("/api/datasets/{dataset_id}/action-jobs", "POST"), routes)
        self.assertIn(("/api/datasets/{dataset_id}/episodes/{episode_id}/action-mapping", "GET"), routes)

        root = Path(__file__).resolve().parents[1]
        html = (root / "index.html").read_text(encoding="utf-8")
        javascript = (root / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="actionMappingButton"', html)
        self.assertIn('id="actionRobotProfile"', html)
        self.assertIn('/action-jobs`', javascript)
        self.assertIn('operation === "action_mapping"', javascript)


if __name__ == "__main__":
    unittest.main()
