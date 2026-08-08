from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from app.camera_profiles import NEXUS_OAKD_PRO_W9_PROFILE_ID
from app.dataset_format import describe_source, inspect_dataset_format, is_self_describing_dataset_root, processing_strategy_for_family


class DatasetFormatTests(unittest.TestCase):
    @staticmethod
    def _write_nexus_episode(root: Path, name: str = "ep_0001_20260729_120000") -> Path:
        episode = root / name
        (episode / "meta").mkdir(parents=True)
        for folder in ("camera", "mocap", "tactile", "sensor"):
            (episode / folder).mkdir()
        for relative in (
            "camera/head_rgb.mp4",
            "camera/head_depth.raw",
            "mocap/dexweaveg1_left.h5",
            "mocap/dexweaveg1_left_raw.h5",
            "tactile/left.h5",
            "tactile/left_raw.h5",
            "sensor/head_imu.h5",
            "meta/sync.parquet",
            "meta/video_timestamps.parquet",
            "meta/mocap_dexweaveg1_pairs.json",
        ):
            (episode / relative).touch()
        metadata = {
            "schema_version": "4.0",
            "frame_count": 30,
            "sync": {"tick_hz": 30.0},
            "files": {
                "camera": {"head_rgb": "camera/head_rgb.mp4", "head_depth": "camera/head_depth.raw"},
                "mocap": {
                    "dexweaveg1_left": "mocap/dexweaveg1_left.h5",
                    "dexweaveg1_left_raw": "mocap/dexweaveg1_left_raw.h5",
                },
                "tactile": {"left": "tactile/left.h5", "left_raw": "tactile/left_raw.h5"},
                "meta": {
                    "sync": "meta/sync.parquet",
                    "video_timestamps": "meta/video_timestamps.parquet",
                    "mocap_pairs": "meta/mocap_dexweaveg1_pairs.json",
                },
            },
            "sensors": {
                "camera": {
                    "head": {
                        "resolution": [1280, 800],
                        "storage_fps": 50.0,
                        "path": "camera/head_rgb.mp4",
                        "depth": {
                            "resolution": [1280, 800],
                            "codec": "raw_uint16_le",
                            "storage_fps": 30,
                            "path": "camera/head_depth.raw",
                            "unit": "mm",
                        },
                        "imu": {"path": "sensor/head_imu.h5", "storage_fps": 400.0, "sample_count": 400},
                    }
                },
                "mocap": {
                    "dexweaveg1_left": {
                        "nodes": 20,
                        "rate_hz": 60,
                        "sync_fps": 30,
                        "raw_path": "mocap/dexweaveg1_left_raw.h5",
                        "sync_path": "mocap/dexweaveg1_left.h5",
                    }
                },
                "tactile": {
                    "left": {
                        "channels": 225,
                        "rate_hz": 60,
                        "sync_fps": 30,
                        "raw_path": "tactile/left_raw.h5",
                        "sync_path": "tactile/left.h5",
                    }
                },
            },
        }
        (episode / "meta" / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (episode / "meta" / "camera_extrinsics.json").write_text(json.dumps({"applied": False}), encoding="utf-8")
        return episode

    def test_nexus_root_and_modalities_are_kept_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_nexus_episode(root)

            report = inspect_dataset_format(root)
            self_describing = is_self_describing_dataset_root(root)

        self.assertTrue(self_describing)
        self.assertEqual("dataset", report["root_mode"])
        self.assertEqual("nexus_multimodal", report["format_family"])
        streams = {item["source_path_template"]: item for item in report["declared_streams"]}
        self.assertEqual(50.0, streams["camera/head_rgb.mp4"]["storage_fps"])
        self.assertEqual(30.0, streams["camera/head_rgb.mp4"]["sync_fps"])
        self.assertEqual("synchronized", streams["mocap/dexweaveg1_left.h5"]["variant"])
        self.assertEqual(30.0, streams["mocap/dexweaveg1_left.h5"]["fps"])
        self.assertEqual("raw", streams["mocap/dexweaveg1_left_raw.h5"]["variant"])
        self.assertEqual(60.0, streams["mocap/dexweaveg1_left_raw.h5"]["fps"])
        self.assertEqual("imu", streams["sensor/head_imu.h5"]["modality"])
        self.assertEqual("metadata", streams["meta/mocap_dexweaveg1_pairs.json"]["kind"])
        self.assertEqual("depth", streams["camera/head_depth.raw"]["modality"])
        self.assertTrue(report["capabilities"]["can_s1"])
        self.assertTrue(report["capabilities"]["can_pressure_analysis"])
        self.assertFalse(report["capabilities"]["can_joint_overlay"])
        self.assertFalse(report["capabilities"]["can_full_export"])
        self.assertEqual("nexus_sensor_fusion_v1", report["processing_strategy"]["id"])
        self.assertFalse(report["processing_strategy"]["joint_overlay"])
        self.assertEqual("synchronized", report["processing_strategy"]["pressure"]["preferred_variant"])
        self.assertFalse(report["processing_strategy"]["pressure"]["hard_reject"])
        self.assertTrue(report["camera_calibration"]["requires_profile_selection"])
        self.assertEqual(NEXUS_OAKD_PRO_W9_PROFILE_ID, report["camera_calibration"]["recommended_profile_id"])
        self.assertFalse(report["capabilities"]["can_rgb_depth_registration"])

    def test_nexus_camera_profile_supplies_only_rgb_depth_extrinsics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_nexus_episode(root)

            report = inspect_dataset_format(root, camera_profile_id=NEXUS_OAKD_PRO_W9_PROFILE_ID)

        calibration = report["camera_calibration"]
        self.assertEqual(NEXUS_OAKD_PRO_W9_PROFILE_ID, calibration["selected_profile_id"])
        self.assertFalse(calibration["requires_profile_selection"])
        transforms = calibration["selected_profile"]["transforms"]
        self.assertAlmostEqual(-0.075, transforms["T_right__left"][0][3])
        self.assertAlmostEqual(0.0375, transforms["T_head_rgb__right"][0][3])
        self.assertAlmostEqual(-0.0375, transforms["T_head_rgb__left"][0][3])
        self.assertAlmostEqual(0.0375, transforms["T_head_rgb__head_depth"][0][3])
        self.assertTrue(report["capabilities"]["can_rgb_depth_registration"])
        self.assertTrue(report["capabilities"]["can_depth_arm_localization"])
        self.assertFalse(report["capabilities"]["can_joint_overlay"])
        self.assertFalse(report["capabilities"]["can_hand_visibility"])
        self.assertIn("rgb_depth_camera_profile_selected", {item["code"] for item in report["issues"]})

    def test_egodex_embedded_camera_pose_does_not_require_external_extrinsics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "0.mp4").touch()
            with h5py.File(root / "0.hdf5", "w") as output:
                output.create_dataset("camera/intrinsic", data=np.eye(3, dtype=np.float32))
                output.create_dataset("transforms/camera", data=np.eye(4, dtype=np.float32)[None])
                output.create_dataset("transforms/rightHand", data=np.eye(4, dtype=np.float32)[None])

            report = inspect_dataset_format(root)

        self.assertEqual("egodex", report["format_family"])
        self.assertTrue(report["capabilities"]["can_joint_overlay"])
        self.assertTrue(report["capabilities"]["can_hand_visibility"])
        self.assertFalse(report["camera_calibration"]["hand_projection_requires_extrinsics"])
        self.assertEqual("embedded_camera_pose", report["camera_calibration"]["hand_projection_mode"])
        self.assertNotIn("camera_extrinsics_missing", {item["code"] for item in report["issues"]})

    def test_egodex_and_nexus_use_separate_processing_strategies(self) -> None:
        egodex = processing_strategy_for_family(
            "egodex",
            has_joint=True,
            has_action=False,
            has_contact_sensor=False,
            has_timestamps=True,
        )
        nexus = processing_strategy_for_family(
            "nexus_multimodal",
            has_joint=True,
            has_action=False,
            has_contact_sensor=True,
            has_timestamps=True,
        )

        self.assertEqual("egodex_joint_centric_v1", egodex["id"])
        self.assertTrue(egodex["joint_overlay"])
        self.assertTrue(egodex["pose_recovery"])
        self.assertEqual("nexus_sensor_fusion_v1", nexus["id"])
        self.assertFalse(nexus["joint_overlay"])
        self.assertFalse(nexus["pose_recovery"])
        self.assertEqual("auxiliary_interaction_evidence", nexus["pressure"]["role"])
        self.assertTrue(nexus["pressure"]["empty_value_hard_reject"])
        self.assertFalse(nexus["pressure"]["zero_is_empty"])

    def test_raw_variant_uses_filename_stem(self) -> None:
        descriptor = describe_source("mocap/dexweaveg1_left_raw.h5", field="skeleton", shape=[10, 20, 7])
        self.assertEqual("raw", descriptor["variant"])
        self.assertFalse(descriptor["synchronized"])

    def test_confirmation_token_changes_when_sampled_payload_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = self._write_nexus_episode(root)
            before = inspect_dataset_format(root)["confirmation_token"]
            (episode / "camera" / "head_rgb.mp4").write_bytes(b"changed-media-payload")
            after = inspect_dataset_format(root)["confirmation_token"]

        self.assertNotEqual(before, after)

    def test_headerless_depth_without_geometry_is_not_claimed_importable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = root / "ep_0001"
            (episode / "meta").mkdir(parents=True)
            (episode / "camera").mkdir()
            (episode / "camera" / "head_depth.raw").write_bytes(b"\x00\x01" * 16)
            (episode / "meta" / "metadata.json").write_text(json.dumps({
                "schema_version": "4.0",
                "files": {"camera": {"head_depth": "camera/head_depth.raw"}},
                "sensors": {"camera": {"head": {"depth": {"path": "camera/head_depth.raw"}}}},
            }), encoding="utf-8")

            report = inspect_dataset_format(root)

        self.assertEqual("blocked", report["status"])
        self.assertFalse(report["capabilities"]["can_import"])
        self.assertIn("raw_depth_geometry_unknown", {item["code"] for item in report["issues"]})


if __name__ == "__main__":
    unittest.main()
