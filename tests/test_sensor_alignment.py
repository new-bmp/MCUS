from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from app.sensor_alignment import _episode_records, _nearest_timestamp_lookup, map_video_frame_to_sensor, scan_episode_sensor_alignment


class SensorAlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "dataset"
        self.root.mkdir()
        self.sidecar = Path(self.temporary.name) / ".alicePD" / "fixture"
        self.episode = {
            "id": "episode-1",
            "name": "episode-1",
            "type": "video",
            "relative_path": "video.mp4",
            "fps": 30.0,
            "frame_count": 300,
            "duration": 10.0,
            "primary_media_file_id": "video-1",
            "media_streams": [{
                "file_id": "video-1",
                "type": "video",
                "relative_path": "video.mp4",
                "stream_name": "video.mp4",
                "fps": 30.0,
                "frame_count": 300,
                "duration": 10.0,
            }],
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _manifest(self, name: str) -> dict:
        return {
            "id": "fixture",
            "root_path": str(self.root),
            "sidecar_path": str(self.sidecar),
            "episodes": [self.episode],
            "files": [{
                "id": name,
                "relative_path": name,
                "extension": ".h5",
                "episode_id": self.episode["id"],
            }],
        }

    def _write_h5(self, name: str, count: int, hz: float, *, master_clock: bool = False) -> Path:
        path = self.root / name
        with h5py.File(path, "w") as handle:
            handle.create_dataset("values", data=np.zeros((count, 4), dtype=np.float32))
            handle.create_dataset("timestamps", data=np.arange(count, dtype=np.float64) / hz)
            if master_clock:
                handle.attrs["sample_rate"] = 50
                handle.attrs["master_clock"] = "head"
        return path

    def test_prealigned_master_clock_keeps_identity_mapping(self) -> None:
        path = self._write_h5("aligned.h5", 300, 30.0, master_clock=True)
        source_signature = (path.stat().st_size, path.stat().st_mtime_ns)
        manifest = self._manifest(path.name)

        document = scan_episode_sensor_alignment(manifest, self.episode, force=True)
        stream = document["streams"][0]
        sensor_index, _ = map_video_frame_to_sensor(manifest, self.episode, path.name, 120, alignment=document)

        self.assertEqual("prealigned_master_clock", stream["mode"])
        self.assertEqual(50.0, stream["physical_hz"])
        self.assertEqual(1.0, stream["index_multiplier"])
        self.assertEqual(120, sensor_index)
        self.assertEqual(source_signature, (path.stat().st_size, path.stat().st_mtime_ns))

    def test_qwen_assignment_overrides_local_episode_membership(self) -> None:
        manifest = self._manifest("assigned.h5")
        manifest["files"][0]["episode_id"] = "local-episode"
        manifest["episode_resolution"] = {"file_episode_assignments": {"assigned.h5": self.episode["id"]}}

        records = _episode_records(manifest, self.episode)

        self.assertEqual(["assigned.h5"], [item["relative_path"] for item in records])

    def test_mismatched_rate_uses_timestamp_lookup_and_multiplier(self) -> None:
        path = self._write_h5("fast.h5", 1000, 100.0)
        manifest = self._manifest(path.name)

        document = scan_episode_sensor_alignment(manifest, self.episode, force=True)
        stream = document["streams"][0]
        sensor_index, metadata = map_video_frame_to_sensor(manifest, self.episode, path.name, 30, alignment=document)

        self.assertEqual("timestamp_nearest", stream["mode"])
        self.assertAlmostEqual(100.0 / 30.0, stream["index_multiplier"], places=9)
        self.assertEqual(100, sensor_index)
        self.assertTrue(metadata["valid"])

    def test_same_clock_timestamp_offset_is_preserved(self) -> None:
        reference = 1_700_000_000.0 + np.arange(30, dtype=np.float64) / 30.0
        sensor = 1_700_000_000.5 + np.arange(100, dtype=np.float64) / 100.0

        mapping, quality = _nearest_timestamp_lookup(reference, sensor, 100.0)

        self.assertIsNotNone(mapping)
        self.assertEqual([-1] * 15, mapping[:15])
        self.assertEqual(0, mapping[15])
        self.assertFalse(quality["origins_normalized"])

    def test_equal_count_long_gap_uses_timestamp_lookup(self) -> None:
        path = self.root / "gapped.h5"
        timestamps = np.arange(300, dtype=np.float64) / 30.0
        timestamps[151:] += 1.0
        with h5py.File(path, "w") as handle:
            handle.create_dataset("values", data=np.zeros((300, 4), dtype=np.float32))
            handle.create_dataset("timestamps", data=timestamps)
        manifest = self._manifest(path.name)

        document = scan_episode_sensor_alignment(manifest, self.episode, force=True)
        stream = document["streams"][0]
        sensor_index, metadata = map_video_frame_to_sensor(
            manifest, self.episode, path.name, 160, alignment=document,
        )

        self.assertEqual("timestamp_nearest", stream["mode"])
        self.assertIsNone(sensor_index)
        self.assertFalse(metadata["valid"])
        self.assertGreater(stream["lookup_quality"]["long_gap_rejection_count"], 0)

    def test_multiplier_mapping_marks_out_of_range_as_missing(self) -> None:
        manifest = self._manifest("short.h5")
        document = {
            "reference_video": {"fps": 30.0},
            "streams": [{
                "relative_path": "short.h5",
                "data_count": 100,
                "mode": "rate_multiplier",
                "stored_hz": 100.0,
                "physical_hz": 100.0,
                "index_multiplier": 100.0 / 30.0,
            }],
        }

        sensor_index, metadata = map_video_frame_to_sensor(
            manifest, self.episode, "short.h5", 60, video_fps=30.0, alignment=document,
        )

        self.assertIsNone(sensor_index)
        self.assertFalse(metadata["valid"])


if __name__ == "__main__":
    unittest.main()
