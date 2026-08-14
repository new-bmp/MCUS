from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from app import storage as storage_module
from app.camera_profiles import NEXUS_OAKD_PRO_W9_PROFILE_ID
from app.storage import discover_dataset_roots, read_frame, scan_dataset


class DatasetFormatStorageTests(unittest.TestCase):
    def test_corrupt_legacy_display_name_falls_back_to_source_folder(self) -> None:
        manifest = {"name": "????-AlicePose????", "root_path": r"C:\datasets\顺序测试"}

        self.assertTrue(storage_module._repair_manifest_display_name(manifest))
        self.assertEqual("顺序测试", manifest["name"])
        self.assertFalse(storage_module._repair_manifest_display_name(manifest))

    def test_legacy_nexus_manifest_timebases_are_backfilled_without_rescan(self) -> None:
        manifest = {
            "format_family": "nexus_multimodal",
            "format_map": {
                "format_family": "nexus_multimodal",
                "declared_streams": [
                    {"source_path_template": "meta/sync.parquet", "kind": "timestamp", "variant": "synchronized", "fps": 30.0},
                    {"source_path_template": "camera/head_rgb.mp4", "kind": "vision", "variant": "primary", "fps": 50.053},
                ],
            },
            "episodes": [{
                "primary_media_file_id": "head",
                "fps": 50.0,
                "media_streams": [{"file_id": "head", "relative_path": "ep_0002/camera/head_rgb.mp4", "fps": 50.0}],
            }],
        }

        self.assertTrue(storage_module._backfill_manifest_timebases(manifest))
        episode = manifest["episodes"][0]
        stream = episode["media_streams"][0]
        self.assertEqual(50.053, stream["source_fps"])
        self.assertEqual(50.053, stream["storage_fps"])
        self.assertEqual(30.0, stream["sync_fps"])
        self.assertEqual(50.053, episode["source_fps"])
        self.assertEqual(30.0, episode["canonical_sync_fps"])
        self.assertFalse(storage_module._backfill_manifest_timebases(manifest))

    @staticmethod
    def _write_video(path: Path, frame_count: int, width: int, height: int, fps: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("OpenCV MP4 writer is unavailable")
        try:
            for index in range(frame_count):
                writer.write(np.full((height, width, 3), (20 + index * 30) % 256, dtype=np.uint8))
        finally:
            writer.release()

    @staticmethod
    def _nexus_metadata(frame_count: int, width: int, height: int, fps: float) -> dict:
        return {
            "nexus_version": "4.1",
            "schema_version": "4.1",
            "frame_count": frame_count,
            "files": {
                "camera": {
                    "head_rgb": "camera/head_rgb.mp4",
                    "head_depth": "camera/head_depth.raw",
                },
                "pressure": {
                    "left": "pressure/left_pressure.csv",
                },
            },
            "sensors": {
                "camera": {
                    "head": {
                        "resolution": [width, height],
                        "storage_fps": fps,
                        "depth": {
                            "resolution": [width, height],
                            "storage_fps": fps,
                            "codec": "uint16",
                            "unit": "millimeter",
                        },
                    },
                },
            },
        }

    def test_nexus_episode_root_is_not_split_into_child_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "nexus"
            for episode_name in ("ep_000001", "ep_000002"):
                metadata = root / episode_name / "meta" / "metadata.json"
                metadata.parent.mkdir(parents=True)
                metadata.write_text(json.dumps(self._nexus_metadata(2, 8, 6, 10.0)), encoding="utf-8")

            discovery = discover_dataset_roots(root)

        self.assertEqual("single", discovery["mode"])
        self.assertEqual(1, discovery["dataset_count"])
        self.assertEqual(str(root.resolve()), discovery["datasets"][0]["path"])

    def test_scan_persists_canonical_streams_and_keeps_raw_depth_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            root = temporary_root / "nexus"
            episode_root = root / "ep_000001"
            width, height, frame_count, fps = 32, 24, 3, 10.0
            metadata_path = episode_root / "meta" / "metadata.json"
            metadata_path.parent.mkdir(parents=True)
            metadata_path.write_text(
                json.dumps(self._nexus_metadata(frame_count, width, height, fps)),
                encoding="utf-8",
            )
            self._write_video(episode_root / "camera" / "head_rgb.mp4", frame_count, width, height, fps)
            depth_path = episode_root / "camera" / "head_depth.raw"
            depth_values = np.arange(frame_count * width * height, dtype=np.uint16).reshape(frame_count, height, width)
            depth_values.tofile(depth_path)
            pressure_path = episode_root / "pressure" / "left_pressure.csv"
            pressure_path.parent.mkdir(parents=True)
            pressure_path.write_text("pressure\n1.0\n2.0\n3.0\n", encoding="utf-8")

            runtime = temporary_root / "runtime"
            with patch.multiple(
                storage_module,
                RUNTIME=runtime,
                MANIFESTS=runtime / "datasets",
                ANNOTATIONS=runtime / "annotations",
                CACHE=runtime / "cache",
                EXPORTS=runtime / "exports",
            ):
                manifest = scan_dataset(
                    root,
                    dataset_id="nexus-fixture",
                    camera_profile_id=NEXUS_OAKD_PRO_W9_PROFILE_ID,
                )

            format_map_path = Path(manifest["format_map_path"])
            self.assertTrue(format_map_path.is_file())
            self.assertEqual("nexus_multimodal", manifest["format_family"])
            self.assertEqual("nexus_multimodal", json.loads(format_map_path.read_text(encoding="utf-8"))["format_family"])
            self.assertEqual(NEXUS_OAKD_PRO_W9_PROFILE_ID, manifest["camera_calibration"]["selected_profile_id"])
            self.assertAlmostEqual(
                0.0375,
                manifest["camera_calibration"]["selected_profile"]["transforms"]["T_head_rgb__head_depth"][0][3],
            )
            self.assertEqual("completed", manifest["schema_profile"]["status"])

            records = {item["relative_path"]: item for item in manifest["files"]}
            rgb_record = records["ep_000001/camera/head_rgb.mp4"]
            depth_record = records["ep_000001/camera/head_depth.raw"]
            pressure_record = records["ep_000001/pressure/left_pressure.csv"]
            self.assertEqual(("vision", "rgb", "primary"), (rgb_record["canonical_kind"], rgb_record["modality"], rgb_record["variant"]))
            self.assertEqual(("vision", "depth", "primary"), (depth_record["canonical_kind"], depth_record["modality"], depth_record["variant"]))
            self.assertEqual(("sensor", "pressure", "left"), (pressure_record["canonical_kind"], pressure_record["modality"], pressure_record["side"]))

            self.assertEqual(1, manifest["episode_count"])
            episode = manifest["episodes"][0]
            streams = {item["relative_path"]: item for item in episode["media_streams"]}
            rgb_stream = streams["ep_000001/camera/head_rgb.mp4"]
            depth_stream = streams["ep_000001/camera/head_depth.raw"]
            self.assertEqual(rgb_stream["file_id"], episode["primary_media_file_id"])
            self.assertEqual("rgb", rgb_stream["modality"])
            self.assertTrue(rgb_stream["analysis_eligible"])
            self.assertEqual("raw_depth", depth_stream["type"])
            self.assertEqual("depth", depth_stream["modality"])
            self.assertTrue(depth_stream["is_depth_map"])
            self.assertFalse(depth_stream["analysis_eligible"])
            self.assertEqual("uint16", depth_stream["dtype"])
            self.assertEqual(frame_count, depth_stream["frame_count"])

            preview = read_frame(depth_stream, 1)
            self.assertIsNotNone(preview)
            self.assertEqual((height, width, 3), preview.shape)

            local_streams = manifest["schema_profile"]["understanding"]["streams"]
            raw_depth = next(item for item in local_streams if item["source_path"].endswith("head_depth.raw"))
            self.assertEqual("depth", raw_depth["modality"])
            self.assertEqual([frame_count, height, width], raw_depth["shape"])

    def test_nexus_multirate_media_keeps_native_counts_and_uses_sync_only_as_reference(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as parquet

        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            root = temporary_root / "nexus"
            episode_root = root / "ep_000001_20260729_120000"
            width, height = 32, 24
            metadata = {
                "nexus_version": "4.1",
                "schema_version": "4.1",
                "frame_count": 5,
                "sync": {"tick_hz": 30.0},
                "files": {
                    "camera": {
                        "head_rgb": "camera/head_rgb.mp4",
                        "wrist_left": "camera/wrist_left.mp4",
                        "wrist_right": "camera/wrist_right.mp4",
                        "head_depth": "camera/head_depth.raw",
                    },
                    "meta": {
                        "sync": "meta/sync.parquet",
                        "video_timestamps": "meta/video_timestamps.parquet",
                    },
                },
                "sensors": {
                    "camera": {
                        "head": {
                            "resolution": [width, height],
                            "storage_fps": 20.0,
                            "depth": {
                                "resolution": [width, height],
                                "storage_fps": 10.0,
                                "codec": "raw_uint16_le",
                                "unit": "millimeter",
                            },
                        },
                        "wrist_left": {
                            "resolution": [width, height],
                            "storage_fps": 60.0,
                        },
                        "wrist_right": {
                            "resolution": [width, height],
                            "storage_fps": 60.0,
                        },
                    },
                },
            }
            metadata_path = episode_root / "meta" / "metadata.json"
            metadata_path.parent.mkdir(parents=True)
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            sync_path = episode_root / "meta" / "sync.parquet"
            video_timestamps_path = episode_root / "meta" / "video_timestamps.parquet"
            parquet.write_table(pa.table({"frame_idx": list(range(4))}), sync_path)
            timestamp_cameras = (
                ["head"] * 8
                + ["wrist_left"] * 13
                + ["wrist_right"] * 11
                + ["head_depth"] * 5
            )
            timestamp_indices = (
                list(range(8))
                + list(range(13))
                + list(range(11))
                + list(range(5))
            )
            parquet.write_table(
                pa.table({
                    "camera": timestamp_cameras,
                    "frame_idx": timestamp_indices,
                    "pts_us": [index * 20_000 for index in range(len(timestamp_indices))],
                    "ts_wall": [index / 50.0 for index in range(len(timestamp_indices))],
                }),
                video_timestamps_path,
            )

            self._write_video(episode_root / "camera" / "head_rgb.mp4", 9, width, height, 20.0)
            self._write_video(episode_root / "camera" / "wrist_left.mp4", 14, width, height, 60.0)
            self._write_video(episode_root / "camera" / "wrist_right.mp4", 12, width, height, 60.0)
            depth_path = episode_root / "camera" / "head_depth.raw"
            np.arange(5 * width * height, dtype=np.uint16).tofile(depth_path)

            runtime = temporary_root / "runtime"

            with patch.multiple(
                storage_module,
                RUNTIME=runtime,
                MANIFESTS=runtime / "datasets",
                ANNOTATIONS=runtime / "annotations",
                CACHE=runtime / "cache",
                EXPORTS=runtime / "exports",
            ):
                manifest = scan_dataset(root, dataset_id="nexus-multirate-fixture")

            episode = manifest["episodes"][0]
            streams = {Path(item["relative_path"]).name: item for item in episode["media_streams"]}
            self.assertEqual("head_rgb.mp4", Path(episode["relative_path"]).name)
            self.assertEqual(8, episode["frame_count"])
            self.assertEqual(9, episode["source_frame_count"])
            self.assertAlmostEqual(0.4, episode["duration"], places=3)
            self.assertEqual((8, 9, 20.0), (streams["head_rgb.mp4"]["frame_count"], streams["head_rgb.mp4"]["source_frame_count"], streams["head_rgb.mp4"]["fps"]))
            self.assertEqual(30.0, streams["head_rgb.mp4"]["sync_fps"])
            self.assertEqual((13, 14, 60.0), (streams["wrist_left.mp4"]["frame_count"], streams["wrist_left.mp4"]["source_frame_count"], streams["wrist_left.mp4"]["fps"]))
            self.assertEqual((11, 12, 60.0), (streams["wrist_right.mp4"]["frame_count"], streams["wrist_right.mp4"]["source_frame_count"], streams["wrist_right.mp4"]["fps"]))
            self.assertEqual((5, 10.0), (streams["head_depth.raw"]["frame_count"], streams["head_depth.raw"]["fps"]))
            self.assertAlmostEqual(13 / 60.0, streams["wrist_left.mp4"]["duration"], places=3)
            self.assertAlmostEqual(0.5, streams["head_depth.raw"]["duration"], places=3)
            self.assertEqual(4, episode["canonical_sync_frame_count"])
            self.assertEqual(30.0, episode["canonical_sync_fps"])
            self.assertEqual("sensor_alignment", episode["alignment"]["source"])
            self.assertEqual("per_stream_video_timestamp_index", episode["alignment"]["media_frame_count_policy"])
            self.assertEqual({"head": 8, "wrist_left": 13, "wrist_right": 11, "head_depth": 5}, episode["alignment"]["video_timestamp_frame_counts"])
            self.assertNotIn("logical_frame_count", episode["alignment"])
            self.assertEqual(1, streams["head_rgb.mp4"]["trimmed_unindexed_frames"])
            self.assertEqual(1, streams["wrist_left.mp4"]["trimmed_unindexed_frames"])
            self.assertEqual(1, streams["wrist_right.mp4"]["trimmed_unindexed_frames"])

    def test_repeat_import_reuses_dataset_id_sidecar_and_created_at(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            root = temporary_root / "images"
            root.mkdir()
            cv2.imwrite(str(root / "frame.png"), np.full((12, 16, 3), 100, dtype=np.uint8))
            runtime = temporary_root / "runtime"

            with patch.multiple(
                storage_module,
                RUNTIME=runtime,
                MANIFESTS=runtime / "datasets",
                ANNOTATIONS=runtime / "annotations",
                CACHE=runtime / "cache",
                EXPORTS=runtime / "exports",
            ):
                first = scan_dataset(root)
                second = scan_dataset(root)

            self.assertEqual(first["id"], second["id"])
            self.assertEqual(first["sidecar_path"], second["sidecar_path"])
            self.assertEqual(first["created_at"], second["created_at"])
            self.assertIn("updated_at", second)

    def test_episode_camera_images_merge_with_video_and_receive_episode_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            root = temporary_root / "mixed"
            episode_root = root / "ep_000006" / "camera"
            self._write_video(episode_root / "head_rgb.mp4", 2, 16, 12, 10.0)
            cv2.imwrite(str(episode_root / "preview.png"), np.full((12, 16, 3), 160, dtype=np.uint8))
            runtime = temporary_root / "runtime"

            with patch.multiple(
                storage_module,
                RUNTIME=runtime,
                MANIFESTS=runtime / "datasets",
                ANNOTATIONS=runtime / "annotations",
                CACHE=runtime / "cache",
                EXPORTS=runtime / "exports",
            ):
                manifest = scan_dataset(root)

            self.assertEqual(1, manifest["episode_count"])
            episode = manifest["episodes"][0]
            records = {item["relative_path"]: item for item in manifest["files"]}
            self.assertEqual(episode["id"], records["ep_000006/camera/head_rgb.mp4"]["episode_id"])
            self.assertEqual(episode["id"], records["ep_000006/camera/preview.png"]["episode_id"])
            self.assertEqual(2, len(episode["media_streams"]))


if __name__ == "__main__":
    unittest.main()
