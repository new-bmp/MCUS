from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.batch_jobs import BatchAnalysisJobManager
from app.behavior_annotator import (
    BEHAVIOR_ARTIFACT_VERSION,
    BehaviorJobManager,
    behavior_annotation_status,
    load_behavior_annotation,
)
from app.qwen_trim import _source_video_fingerprint
from app.schemas import BatchAnalysisRequest, BehaviorAnnotationRequest


class _DeferredExecutor:
    def __init__(self) -> None:
        self.calls = []

    def submit(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return SimpleNamespace()


class BehaviorAnnotationReuseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.source = root / "episode.mp4"
        self.source.write_bytes(b"behavior-video" * 4096)
        self.sidecar = root / ".alicePD" / "fixture"
        self.episode = {
            "id": "ep-1",
            "name": "EP 1",
            "type": "video",
            "path": str(self.source),
            "relative_path": "episode.mp4",
            "fps": 30.0,
            "frame_count": 10,
            "primary_media_file_id": "media-1",
            "media_streams": [{
                "file_id": "media-1",
                "stream_name": "episode.mp4",
                "type": "video",
                "path": str(self.source),
                "relative_path": "episode.mp4",
                "fps": 30.0,
                "frame_count": 10,
            }],
        }
        self.manifest = {"id": "fixture", "episodes": [self.episode]}

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _artifact_dir(self, _dataset_id: str, category: str) -> Path:
        target = self.sidecar / category
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _write_artifacts(self, fingerprint: bool = True, include_target: bool = True) -> None:
        annotation_path = self._artifact_dir("fixture", "behavior-annotations") / "ep-1.behavior.alice"
        source_video = {
            "file_id": "media-1",
            "stream_name": "episode.mp4",
            "relative_path": "episode.mp4",
            "frame_count": 10,
        }
        if fingerprint:
            source_video["fingerprint"] = _source_video_fingerprint({"path": str(self.source)})
        annotation_path.write_text(json.dumps({
            "schema": "alice/vlm-behavior/v1",
            "artifact_version": BEHAVIOR_ARTIFACT_VERSION,
            "dataset_id": "fixture",
            "episode_id": "ep-1",
            "source_video": source_video if fingerprint else {},
            "sampling": {"media_file_id": "media-1", "stream_name": "episode.mp4", "frames": [0, 9]},
            "task_label": "pick",
            "confidence": 0.9,
            "segments": [{
                "start_frame": 0,
                "end_frame": 9,
                "phase_label": "grasp",
                "label": "grasp",
                "boundary_source": "vlm",
            }],
            "primary_targets": [{"name": "cup"}],
        }), encoding="utf-8")
        if include_target:
            target_path = self._artifact_dir("fixture", "behavior-targets") / "ep-1.targets.alice"
            target_path.write_text(json.dumps({
                "schema": "alice/behavior-targets/v1",
                "dataset_id": "fixture",
                "episode_id": "ep-1",
                "primary_terms": ["cup"],
                "source_annotation": "ep-1.behavior.alice",
            }), encoding="utf-8")

    def test_source_locked_and_legacy_artifacts_are_reusable(self) -> None:
        with patch("app.behavior_annotator.dataset_artifact_dir", side_effect=self._artifact_dir):
            self._write_artifacts(fingerprint=True)
            locked = behavior_annotation_status("fixture", self.manifest, self.episode)
            self.assertTrue(locked["reusable"])
            self.assertEqual("source_fingerprint", locked["validation"])
            self._write_artifacts(fingerprint=False)
            legacy = behavior_annotation_status("fixture", self.manifest, self.episode)
            self.assertTrue(legacy["reusable"])
            self.assertEqual("legacy_frame_descriptor", legacy["validation"])

    def test_changed_source_or_missing_targets_is_not_reused(self) -> None:
        with patch("app.behavior_annotator.dataset_artifact_dir", side_effect=self._artifact_dir):
            self._write_artifacts(fingerprint=True)
            self.source.write_bytes(b"replaced-source" * 4096)
            changed = behavior_annotation_status("fixture", self.manifest, self.episode)
            self.assertFalse(changed["reusable"])
            self.assertEqual("source_media_changed", changed["reason"])
            (self.sidecar / "behavior-targets" / "ep-1.targets.alice").unlink()
            missing = behavior_annotation_status("fixture", self.manifest, self.episode)
            self.assertFalse(missing["reusable"])
            self.assertEqual("missing_behavior_or_target_artifact", missing["reason"])

    def test_previous_phase_protocol_version_requires_rerun(self) -> None:
        with patch("app.behavior_annotator.dataset_artifact_dir", side_effect=self._artifact_dir):
            self._write_artifacts(fingerprint=True)
            path = self.sidecar / "behavior-annotations" / "ep-1.behavior.alice"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["artifact_version"] = BEHAVIOR_ARTIFACT_VERSION - 1
            path.write_text(json.dumps(payload), encoding="utf-8")

            status = behavior_annotation_status("fixture", self.manifest, self.episode)
            loaded = load_behavior_annotation("fixture", "ep-1")

        self.assertFalse(status["reusable"])
        self.assertEqual("invalid_behavior_artifact", status["reason"])
        self.assertIsNone(loaded)

    def test_single_episode_reuses_before_api_configuration_check(self) -> None:
        manager = BehaviorJobManager()
        manager._executor.shutdown(wait=True, cancel_futures=True)
        manager._executor = _DeferredExecutor()
        with (
            patch("app.behavior_annotator.dataset_artifact_dir", side_effect=self._artifact_dir),
            patch("app.behavior_annotator.get_episode", return_value=(self.manifest, self.episode)),
            patch("app.behavior_annotator.registry", SimpleNamespace(has_vlm=False)),
        ):
            self._write_artifacts(fingerprint=True)
            job = manager.submit("fixture", "ep-1", BehaviorAnnotationRequest())
        self.assertEqual("complete", job["status"])
        self.assertTrue(job["reused"])
        self.assertEqual("skipped", job["result"]["reuse"]["status"])
        self.assertEqual([], manager._executor.calls)

    def test_batch_reuses_all_episodes_without_api_or_background_work(self) -> None:
        manager = BatchAnalysisJobManager()
        manager._executor.shutdown(wait=True, cancel_futures=True)
        manager._executor = _DeferredExecutor()
        request = BatchAnalysisRequest(operation="vlm_behavior", episode_ids=["ep-1"])
        with (
            patch("app.behavior_annotator.dataset_artifact_dir", side_effect=self._artifact_dir),
            patch("app.batch_jobs.get_manifest", return_value=self.manifest),
            patch("app.batch_jobs.registry", SimpleNamespace(has_vlm=False)),
        ):
            self._write_artifacts(fingerprint=True)
            job = manager.submit("fixture", request)
        self.assertEqual("complete", job["status"])
        self.assertEqual(1, job["result"]["skipped_count"])
        self.assertEqual("skipped", job["result"]["items"][0]["status"])
        self.assertEqual("existing_valid_annotation", job["result"]["items"][0]["reason"])
        self.assertEqual([], manager._executor.calls)


if __name__ == "__main__":
    unittest.main()
