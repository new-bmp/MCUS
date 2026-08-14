from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from app.sensor_alignment import (
    _episode_records,
    ensure_episode_time_sync,
    find_sensor_alignment_stream,
    _nearest_timestamp_lookup,
    _reference_timestamps,
    load_sensor_alignment,
    map_video_frame_to_sensor,
    retime_sensor_alignment,
    scan_episode_sensor_alignment,
    SensorAlignmentJobManager,
    sensor_alignment_path,
)


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

    def _write_parquet(self, relative_path: str, payload: dict) -> dict:
        import pyarrow as pa
        import pyarrow.parquet as parquet

        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_table(pa.table(payload), path)
        return {
            "id": relative_path,
            "relative_path": relative_path,
            "extension": ".parquet",
            "episode_id": self.episode["id"],
        }

    def test_t0_time_sync_gate_accepts_a_valid_reference_and_sensor_mapping(self) -> None:
        sensor = self._write_h5("sensor.h5", 300, 30.0)
        manifest = self._manifest(sensor.name)
        manifest["files"][0]["canonical_kind"] = "sensor"

        document = ensure_episode_time_sync(manifest, self.episode, force=True)

        self.assertEqual("ready", document["gate"]["status"])
        self.assertEqual(30.0, document["gate"]["reference_fps"])
        self.assertEqual(1, len(document["streams"]))

    def test_t0_counts_discovered_repairs_when_cached_file_kind_is_other(self) -> None:
        path = self.root / "legacy_other.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("values", data=np.zeros((300, 4), dtype=np.float32))
        manifest = self._manifest(path.name)
        manifest["files"][0]["canonical_kind"] = "other"

        document = ensure_episode_time_sync(manifest, self.episode, force=True)

        self.assertTrue(document["streams"][0]["repair_applied"])
        self.assertEqual("ready_with_repairs", document["gate"]["quality_status"])
        self.assertEqual(1, document["gate"]["repaired_stream_count"])

    def test_t0_time_sync_gate_blocks_a_missing_critical_sensor(self) -> None:
        manifest = self._manifest("missing.h5")
        manifest["files"][0]["canonical_kind"] = "sensor"

        with self.assertRaisesRegex(RuntimeError, "T0 时间同步失败"):
            ensure_episode_time_sync(manifest, self.episode, force=True)

    def test_t0_time_sync_gate_blocks_a_critical_file_without_a_timeline(self) -> None:
        path = self.root / "empty_sensor.json"
        path.write_text("{}", encoding="utf-8")
        manifest = self._manifest(path.name)
        manifest["files"][0].update({"extension": ".json", "canonical_kind": "sensor"})

        with self.assertRaisesRegex(RuntimeError, "T0 时间同步失败"):
            ensure_episode_time_sync(manifest, self.episode, force=True)

    def test_dataset_t0_job_fails_when_any_episode_cannot_sync(self) -> None:
        manager = SensorAlignmentJobManager(max_workers=1)
        second = {**self.episode, "id": "episode-2", "name": "episode-2"}
        manifest = {
            **self._manifest("sensor.h5"),
            "episodes": [self.episode, second],
        }
        job_id = "t0-job"
        manager._jobs[job_id] = {"id": job_id, "dataset_id": "fixture", "status": "queued"}
        manager._register_cancellation(job_id)
        ready = {
            "streams": [],
            "gate": {"status": "ready"},
            "artifact_path": str(self.sidecar / "sensor-alignment" / "episode-1.json"),
        }
        try:
            with patch(
                "app.sensor_alignment.ensure_episode_time_sync",
                side_effect=[ready, RuntimeError("T0 time sync failed")],
            ) as ensure:
                manager._run(job_id, manifest, [self.episode, second], False)
        finally:
            manager._executor.shutdown(wait=True, cancel_futures=True)

        job = manager.get(job_id)
        self.assertEqual("failed", job["status"])
        self.assertEqual(1, job["result"]["completed_count"])
        self.assertEqual(1, job["result"]["failure_count"])
        self.assertEqual("T0 time sync failed", job["error"])
        self.assertEqual(2, ensure.call_count)

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

    def test_reference_prefers_video_timestamps_and_maps_head_rgb_alias(self) -> None:
        video_record = self._write_parquet("meta/video_timestamps.parquet", {
            "camera": ["wrist_left", "head", "head", "wrist_left", "head", "head"],
            "frame_idx": [0, 0, 1, 1, 2, 3],
            "pts_us": [0, 0, 20_000, 16_667, 40_000, 60_000],
            "ts_wall": [200.0, 100.0, 100.02, 200.016667, 100.04, 100.06],
        })
        sync_record = self._write_parquet("meta/sync.parquet", {
            "frame_idx": [0, 1],
            "master_ts": [900.0, 901.0],
            "head_frame_idx": [0, 2],
            "head_frame_ts": [100.0, 100.04],
            "head_filled": [False, False],
            "partial": [False, False],
        })
        reference = {
            "stream_name": "head_rgb.mp4",
            "relative_path": "camera/head_rgb.mp4",
            "fps": 50.0,
            "frame_count": 4,
        }

        timeline, source = _reference_timestamps(self.root, [sync_record, video_record], reference)

        np.testing.assert_allclose([100.0, 100.02, 100.04, 100.06], timeline)
        self.assertEqual("video_timestamps", source["method"])
        self.assertEqual("head", source["camera"])
        self.assertEqual("meta/video_timestamps.parquet", source["relative_path"])

    def test_sync_fallback_uses_validated_camera_indices_and_preserves_partial(self) -> None:
        sync_record = self._write_parquet("meta/sync.parquet", {
            "frame_idx": [0, 1, 2],
            "master_ts": [900.0, 901.0, 902.0],
            "wrist_left_frame_idx": [0, 2, 4],
            "wrist_left_ts": [100.0, 100.02, 100.04],
            "wrist_left_filled": [False, False, False],
            "partial": [False, True, False],
        })
        reference = {
            "stream_name": "wrist_left.mp4",
            "relative_path": "camera/wrist_left.mp4",
            "fps": 100.0,
            "frame_count": 5,
        }

        timeline, source = _reference_timestamps(self.root, [sync_record], reference)

        self.assertEqual(100.0, timeline[0])
        self.assertTrue(np.isnan(timeline[1]))
        self.assertTrue(np.isnan(timeline[2]))
        self.assertEqual(100.04, timeline[4])
        self.assertEqual("sync_camera_index", source["method"])
        self.assertEqual(1, source["sync_validation"]["partial_row_count"])
        self.assertEqual(2, source["sync_validation"]["legal_row_count"])

    def test_single_missing_trailing_timestamp_is_repaired_without_clamping(self) -> None:
        video_record = self._write_parquet("meta/video_timestamps.parquet", {
            "camera": ["wrist_left"] * 4,
            "frame_idx": [0, 1, 2, 3],
            "pts_us": [0, 16_667, 33_333, 50_000],
            "ts_wall": [100.0 + index / 60.0 for index in range(4)],
        })
        sync_record = self._write_parquet("meta/sync.parquet", {
            "frame_idx": [0, 1, 2],
            "master_ts": [100.0, 100.0 + 1 / 30.0, 100.0 + 2 / 30.0],
            "wrist_left_frame_idx": [0, 2, 999],
            "wrist_left_ts": [100.0, 100.0 + 2 / 60.0, 100.0 + 4 / 60.0],
            "wrist_left_filled": [False, False, False],
            "partial": [False, False, False],
        })
        reference = {
            "stream_name": "wrist_left.mp4",
            "relative_path": "camera/wrist_left.mp4",
            "fps": 60.0,
            "frame_count": 10,
        }

        timeline, source = _reference_timestamps(self.root, [sync_record, video_record], reference)

        self.assertAlmostEqual(100.0 + 4 / 60.0, timeline[4], places=9)
        self.assertTrue(np.isnan(timeline[9]))
        self.assertEqual(1, len(source["repairs"]))
        repair = source["repairs"][0]
        self.assertEqual("single_trailing_timestamp_extrapolation", repair["kind"])
        self.assertEqual(4, repair["frame_idx"])
        self.assertEqual(999, repair["invalid_original_frame_idx"])
        self.assertEqual(1, source["sync_validation"]["rejected_reasons"]["out_of_media_range"])

    def test_hdf5_partial_rows_are_preserved_as_invalid_mapping(self) -> None:
        path = self._write_h5("partial.h5", 300, 30.0, master_clock=True)
        with h5py.File(path, "r+") as handle:
            partial = np.zeros(300, dtype=np.bool_)
            partial[0] = True
            handle.create_dataset("partial", data=partial)
        manifest = self._manifest(path.name)

        document = scan_episode_sensor_alignment(manifest, self.episode, force=True)
        stream = document["streams"][0]
        first_index, first_metadata = map_video_frame_to_sensor(
            manifest, self.episode, path.name, 0, alignment=document,
        )
        second_index, second_metadata = map_video_frame_to_sensor(
            manifest, self.episode, path.name, 1, alignment=document,
        )

        self.assertEqual(-1, stream["frame_to_sensor_index"][0])
        self.assertEqual(1, stream["frame_to_sensor_index"][1])
        self.assertEqual([0], stream["partial_validity"]["partial_rows"])
        self.assertIsNone(first_index)
        self.assertFalse(first_metadata["valid"])
        self.assertEqual("source_partial", first_metadata["invalid_reason"])
        self.assertEqual(1, second_index)
        self.assertTrue(second_metadata["valid"])

    def test_selected_reference_media_uses_own_timeline_and_separate_cache(self) -> None:
        self.episode.update({"frame_count": 4, "fps": 50.0, "duration": 0.08})
        self.episode["media_streams"] = [
            {
                "file_id": "video-1",
                "type": "video",
                "relative_path": "camera/head_rgb.mp4",
                "stream_name": "head_rgb.mp4",
                "fps": 50.0,
                "frame_count": 4,
                "duration": 0.08,
            },
            {
                "file_id": "wrist-left",
                "type": "video",
                "relative_path": "camera/wrist_left.mp4",
                "stream_name": "wrist_left.mp4",
                "fps": 60.0,
                "frame_count": 6,
                "duration": 0.1,
            },
        ]
        sensor = self._write_h5("sensor.h5", 6, 60.0)
        video_record = self._write_parquet("meta/video_timestamps.parquet", {
            "camera": ["head"] * 4 + ["wrist_left"] * 6,
            "frame_idx": list(range(4)) + list(range(6)),
            "pts_us": [index * 20_000 for index in range(4)] + [round(index * 1_000_000 / 60) for index in range(6)],
            "ts_wall": [index / 50.0 for index in range(4)] + [index / 60.0 for index in range(6)],
        })
        manifest = self._manifest(sensor.name)
        manifest["files"].append(video_record)

        primary = scan_episode_sensor_alignment(manifest, self.episode, force=True)
        wrist = scan_episode_sensor_alignment(
            manifest,
            self.episode,
            force=True,
            reference_media_file_id="wrist-left",
        )

        self.assertEqual("video-1", primary["reference_video"]["file_id"])
        self.assertEqual(50.0, primary["reference_video"]["fps"])
        self.assertEqual(4, primary["reference_video"]["frame_count"])
        self.assertEqual("head", primary["reference_timestamp_source"]["camera"])
        self.assertEqual("wrist-left", wrist["reference_video"]["file_id"])
        self.assertEqual(60.0, wrist["reference_video"]["fps"])
        self.assertEqual(6, wrist["reference_video"]["frame_count"])
        self.assertEqual("wrist_left", wrist["reference_timestamp_source"]["camera"])
        self.assertNotEqual(primary["artifact_path"], wrist["artifact_path"])
        self.assertEqual(
            sensor_alignment_path(manifest, self.episode["id"]),
            sensor_alignment_path(manifest, self.episode["id"], "video-1"),
        )
        self.assertEqual("video-1", load_sensor_alignment(manifest, self.episode["id"])["reference_media_file_id"])
        self.assertEqual(
            "wrist-left",
            load_sensor_alignment(manifest, self.episode["id"], "wrist-left")["reference_media_file_id"],
        )

    def test_hdf5_fields_with_different_rates_get_independent_streams(self) -> None:
        path = self.root / "mixed.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("joint/values", data=np.zeros((30, 63), dtype=np.float32))
            handle.create_dataset("joint/timestamps", data=np.arange(30, dtype=np.float64) / 30.0)
            handle.create_dataset("tactile/values", data=np.zeros((100, 8), dtype=np.float32))
            handle.create_dataset("tactile/timestamps", data=np.arange(100, dtype=np.float64) / 100.0)
        manifest = self._manifest(path.name)
        manifest["files"][0]["canonical_kind"] = "sensor"

        document = scan_episode_sensor_alignment(manifest, self.episode, force=True)
        streams = {str(item.get("field")): item for item in document["streams"]}

        self.assertEqual({"joint/values", "tactile/values"}, set(streams))
        self.assertEqual(30, streams["joint/values"]["data_count"])
        self.assertEqual(100, streams["tactile/values"]["data_count"])
        self.assertEqual("timestamp_nearest", streams["joint/values"]["mode"])
        self.assertEqual("timestamp_nearest", streams["tactile/values"]["mode"])

    def test_source_signature_includes_csv_numpy_and_npz(self) -> None:
        csv_path = self.root / "joint.csv"
        csv_path.write_text("timestamp,joint_x\n0,1\n0.033333,2\n0.066667,3\n", encoding="utf-8")
        npy_path = self.root / "joint.npy"
        np.save(npy_path, np.zeros((4, 3), dtype=np.float32))
        npz_path = self.root / "tactile.npz"
        np.savez(npz_path, values=np.zeros((4, 2), dtype=np.float32))
        manifest = {
            **self._manifest(csv_path.name),
            "files": [
                {"id": "csv", "relative_path": csv_path.name, "extension": ".csv", "episode_id": self.episode["id"], "canonical_kind": "joint"},
                {"id": "npy", "relative_path": npy_path.name, "extension": ".npy", "episode_id": self.episode["id"], "canonical_kind": "joint"},
                {"id": "npz", "relative_path": npz_path.name, "extension": ".npz", "episode_id": self.episode["id"], "canonical_kind": "sensor"},
            ],
        }

        document = scan_episode_sensor_alignment(manifest, self.episode, force=True)
        signature_paths = {str(item.get("relative_path")) for item in document["source_signature"]["files"]}

        self.assertEqual({csv_path.name, npy_path.name, npz_path.name}, signature_paths)

    def test_field_stream_lookup_rejects_wrong_source_count(self) -> None:
        document = {
            "streams": [
                {"relative_path": "mixed.h5", "field": "joint/values", "data_count": 30, "kind": "joint"},
                {"relative_path": "mixed.h5", "field": "tactile/values", "data_count": 100, "kind": "sensor"},
            ]
        }
        self.assertEqual(
            "joint/values",
            find_sensor_alignment_stream(document, "mixed.h5", field="joint/values", source_count=30)["field"],
        )
        self.assertIsNone(
            find_sensor_alignment_stream(document, "mixed.h5", field="joint/values", source_count=100),
        )

    def test_retime_sensor_alignment_preserves_fractional_source_positions(self) -> None:
        document = {
            "reference_video": {"file_id": "video-1", "fps": 60.0, "frame_count": 4},
            "streams": [{
                "relative_path": "joint.hdf5",
                "data_count": 7,
                "frame_to_sensor_index": [0, 2, 4, 6],
            }],
        }

        output = retime_sensor_alignment(document, {
            "file_id": "video-1",
            "frame_count": 4,
            "fps": 30.0,
            "source_frame_count": 4,
            "source_frame_positions": [0.0, 0.5, 1.5, 3.0],
            "smoothing_mode": "eis_motion_compensated_30",
        })

        self.assertEqual([0.0, 1.0, 3.0, 6.0], output["streams"][0]["frame_to_sensor_position"])
        self.assertEqual("eis_motion_compensated_30", output["retiming"]["mode"])
        self.assertEqual(2, output["retiming"]["fractional_source_frame_count"])


if __name__ == "__main__":
    unittest.main()
