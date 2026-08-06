from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.qwen_trim import (
    QwenTrimJobManager,
    QwenTrimRequest,
    _atomic_temporary_path,
    _sample_windows,
    _segments,
    _source_fingerprints_match,
    _source_video_fingerprint,
    _smooth_samples,
    _validate_batch_response,
    _write_atomic,
    analyze_qwen_action_trim,
)


class _DeferredExecutor:
    def __init__(self) -> None:
        self.calls = []

    def submit(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return SimpleNamespace()


class QwenTrimTests(unittest.TestCase):
    def test_response_validation_rejects_unknown_and_missing_windows(self) -> None:
        windows = _sample_windows(91, 30.0, 1.0, 1.0, 2)[:3]
        raw = {
            "windows": [
                {"window_id": windows[0]["window_id"], "state": "active", "confidence": 0.9, "reason": "grasp"},
                {"window_id": "invented", "state": "valid", "confidence": 1.0},
                {"window_id": windows[2]["window_id"], "state": "idle", "confidence": 0.8, "reason": "waiting"},
            ],
            "object_nouns": ["cup", "cup"],
        }
        with self.assertRaisesRegex(ValueError, "omitted 1 requested window"):
            _validate_batch_response(raw, windows)

    def test_response_validation_accepts_explicit_complete_uncertain(self) -> None:
        windows = _sample_windows(91, 30.0, 1.0, 1.0, 2)[:3]
        raw = {
            "windows": [
                {"window_id": windows[0]["window_id"], "state": "active", "confidence": 0.9, "reason": "grasp"},
                {"window_id": windows[1]["window_id"], "state": "uncertain", "confidence": 0.4, "reason": "occluded"},
                {"window_id": windows[2]["window_id"], "state": "idle", "confidence": 0.8, "reason": "waiting"},
            ],
            "object_nouns": ["cup", "cup"],
        }
        values, nouns, warnings = _validate_batch_response(raw, windows)
        self.assertEqual([item["raw_state"] for item in values], ["valid", "uncertain", "invalid"])
        self.assertEqual(nouns, ["cup"])
        self.assertEqual(warnings, [])

    def test_batch_exception_aborts_episode_without_artifact_or_change(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.mp4"
            source.write_bytes(b"fake-video-payload" * 128)
            artifact = Path(folder) / "episode.trim.alice"
            media = {
                "path": str(source),
                "type": "video",
                "fps": 30.0,
                "frame_count": 91,
                "file_id": "media-1",
                "stream_name": "head_rgb",
                "relative_path": "source.mp4",
                "modality": "rgb",
                "vlm_eligible": True,
            }
            fake_registry = SimpleNamespace(has_vlm=True)
            with (
                patch("app.qwen_trim.registry", fake_registry),
                patch("app.qwen_trim._request_window_batch", side_effect=RuntimeError("API timeout")),
                patch("app.qwen_trim._artifact_path", return_value=artifact),
                patch("app.qwen_trim.record_change") as record_change,
            ):
                with self.assertRaisesRegex(RuntimeError, "entire Episode was aborted"):
                    analyze_qwen_action_trim(
                        "dataset-1",
                        {},
                        {"id": "episode-1", "name": "Episode 1"},
                        media,
                        QwenTrimRequest(sample_fps=1.0),
                        lambda *_: None,
                    )
            self.assertFalse(artifact.exists())
            record_change.assert_not_called()

    def test_atomic_temporary_names_are_unique_and_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "episode.trim.alice"
            first = _atomic_temporary_path(target)
            second = _atomic_temporary_path(target)
            self.assertNotEqual(first, second)
            self.assertEqual(first.parent, target.parent)
            self.assertTrue(first.name.endswith(".tmp"))
            _write_atomic(target, {"schema": "test/v1"})
            self.assertTrue(target.is_file())
            self.assertEqual(list(target.parent.glob("*.tmp")), [])

    def test_source_video_fingerprint_detects_change(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.mp4"
            source.write_bytes((b"A" * 300_000) + (b"B" * 300_000))
            first = _source_video_fingerprint({"path": str(source)})
            self.assertEqual(first["resolved_path"], str(source.resolve()))
            self.assertEqual(first["size_bytes"], 600_000)
            source.write_bytes((b"C" * 300_000) + (b"D" * 300_000))
            second = _source_video_fingerprint({"path": str(source)})
            self.assertFalse(_source_fingerprints_match(first, second))

    def test_submit_rejects_overlapping_episode_and_list_is_read_only(self) -> None:
        manager = QwenTrimJobManager(max_workers=1)
        manager._executor.shutdown(wait=True, cancel_futures=True)
        manager._executor = _DeferredExecutor()
        manifest = {
            "episodes": [
                {"id": "ep-1", "name": "EP 1", "media_streams": [{"file_id": "m-1", "frame_count": 10, "modality": "rgb", "vlm_eligible": True}]},
                {"id": "ep-2", "name": "EP 2", "media_streams": [{"file_id": "m-2", "frame_count": 10, "modality": "rgb", "vlm_eligible": True}]},
            ]
        }
        fake_registry = SimpleNamespace(has_vlm=True)
        with patch("app.qwen_trim.registry", fake_registry), patch("app.qwen_trim.get_manifest", return_value=manifest):
            all_job = manager.submit(
                "dataset-1",
                QwenTrimRequest(all_episodes=True, media_file_ids={"ep-1": "m-1", "ep-2": "m-2"}),
            )
            with self.assertRaisesRegex(ValueError, "already queued or running"):
                manager.submit(
                    "dataset-1",
                    QwenTrimRequest(episode_ids=["ep-2"], media_file_ids={"ep-2": "m-2"}),
                )
            active = manager.list("dataset-1", active_only=True)
            self.assertEqual([item["id"] for item in active], [all_job["id"]])
            active[0]["message"] = "mutated by caller"
            self.assertNotEqual(manager.get(all_job["id"])["message"], "mutated by caller")

            manager._update(all_job["id"], status="complete")
            manager._release_episodes(all_job["id"])
            next_job = manager.submit(
                "dataset-1",
                QwenTrimRequest(episode_ids=["ep-2"], media_file_ids={"ep-2": "m-2"}),
            )
            self.assertEqual(manager.get(next_job["id"])["status"], "queued")
            manager._release_episodes(next_job["id"])

    def test_run_releases_episode_reservation_after_failure(self) -> None:
        manager = QwenTrimJobManager(max_workers=1)
        manager._executor.shutdown(wait=True, cancel_futures=True)
        job_id = "job-failing"
        manager._jobs[job_id] = {
            "id": job_id,
            "dataset_id": "dataset-1",
            "status": "queued",
            "progress": 0,
            "message": "queued",
        }
        manager._reserve_episodes(job_id, "dataset-1", ["ep-1"])
        manifest = {"episodes": [{"id": "ep-1", "name": "EP 1"}]}
        with (
            patch("app.qwen_trim.get_manifest", return_value=manifest),
            patch("app.qwen_trim.analyze_qwen_action_trim", side_effect=RuntimeError("API failed")),
        ):
            manager._run(
                job_id,
                "dataset-1",
                ["ep-1"],
                {"ep-1": {"file_id": "m-1", "stream_name": "head_rgb"}},
                QwenTrimRequest(episode_ids=["ep-1"], media_file_ids={"ep-1": "m-1"}),
            )
        self.assertEqual(manager.get(job_id)["status"], "failed")
        manager._reserve_episodes("job-retry", "dataset-1", ["ep-1"])
        manager._release_episodes("job-retry")

    def test_temporal_filter_fills_short_internal_miss(self) -> None:
        windows = _sample_windows(150, 30.0, 1.0, 1.0, 2)
        raw_states = ["valid", "valid", "invalid", "valid", "valid", "valid"]
        samples = [{
            **window,
            "raw_state": raw_states[index],
            "confidence": 0.9,
            "reason": raw_states[index],
        } for index, window in enumerate(windows)]
        filtered = _smooth_samples(samples, 150, 30.0, 0.55, 1.1, 0.5)
        self.assertEqual([item["state"] for item in filtered], ["valid"] * len(filtered))
        self.assertEqual(filtered[2]["filter_action"], "filled_short_gap")

    def test_segments_are_binary_contiguous_and_cover_every_frame(self) -> None:
        windows = _sample_windows(101, 25.0, 1.0, 1.0, 2)
        samples = [{
            **window,
            "raw_state": "valid" if index >= 2 else "invalid",
            "confidence": 0.9,
            "reason": "test",
        } for index, window in enumerate(windows)]
        filtered = _smooth_samples(samples, 101, 25.0, 0.55, 0.0, 0.0)
        segments = _segments(filtered, 101, 25.0)
        self.assertEqual(segments[0]["start_frame"], 0)
        self.assertEqual(segments[-1]["end_frame"], 100)
        self.assertEqual({item["state"] for item in segments}, {"valid", "invalid"})
        for previous, current in zip(segments, segments[1:]):
            self.assertEqual(previous["end_frame"] + 1, current["start_frame"])
        self.assertEqual(sum(item["end_frame"] - item["start_frame"] + 1 for item in segments), 101)


if __name__ == "__main__":
    unittest.main()
