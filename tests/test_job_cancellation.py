from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.batch_jobs import BatchAnalysisJobManager
from app.curation_pipeline import CurationJobManager
from app.main import app
from app.qwen_trim import QwenTrimJobManager, QwenTrimRequest
from app.schemas import BatchAnalysisRequest, CurationJobRequest


class _DeferredExecutor:
    def __init__(self) -> None:
        self.calls = []

    def submit(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return SimpleNamespace()


def _wait_for_status(manager, job_id: str, expected: str, timeout: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        if (job.get("status") or job.get("state")) == expected:
            return job
        time.sleep(0.01)
    raise AssertionError(f"Job did not reach {expected}: {manager.get(job_id)}")


class JobCancellationTests(unittest.TestCase):
    def test_running_batch_job_stops_without_becoming_failed(self) -> None:
        manager = BatchAnalysisJobManager()
        started = threading.Event()
        release = threading.Event()
        manifest = {"episodes": [{"id": "ep-1", "name": "EP 1"}]}

        def recover(*_args, **_kwargs):
            started.set()
            release.wait(2)
            return {"artifact_path": "pose.json", "recovered_frame_count": 1}

        try:
            with (
                patch("app.batch_jobs.get_manifest", return_value=manifest),
                patch("app.batch_jobs.pose_recovery_status", return_value={"available": True, "artifact_exists": False, "needed": True}),
                patch("app.batch_jobs.recover_episode_pose", side_effect=recover),
            ):
                job = manager.submit("dataset-1", BatchAnalysisRequest(operation="pose_recovery", episode_ids=["ep-1"]))
                self.assertTrue(started.wait(1))
                cancelling = manager.cancel(job["id"])
                self.assertEqual("cancelling", cancelling["status"])
                release.set()
                cancelled = _wait_for_status(manager, job["id"], "cancelled")
        finally:
            release.set()
            manager._executor.shutdown(wait=True, cancel_futures=True)

        self.assertIsNone(cancelled["error"])
        self.assertIn("任务已终止", cancelled["message"])

    def test_queued_curation_cancel_releases_episode_reservation(self) -> None:
        manager = CurationJobManager(max_workers=1)
        manager._executor.shutdown(wait=True, cancel_futures=True)
        manager._executor = _DeferredExecutor()
        episode = {"id": "ep-1", "name": "EP 1", "primary_media_file_id": "video"}
        media = {"file_id": "video", "path": "source.mp4", "frame_count": 10}
        with patch("app.curation_pipeline.get_manifest", return_value={"episodes": [episode]}), patch("app.curation_pipeline.episode_media", return_value=media):
            job = manager.submit("dataset-1", CurationJobRequest(episode_ids=["ep-1"], media_file_ids={"ep-1": "video"}))
        self.assertIn(("dataset-1", "ep-1"), manager._reservations)

        cancelled = manager.cancel(job["id"])

        self.assertEqual("cancelled", cancelled["status"])
        self.assertNotIn(("dataset-1", "ep-1"), manager._reservations)

    def test_queued_qwen_cancel_releases_episode_reservation(self) -> None:
        manager = QwenTrimJobManager(max_workers=1)
        manager._executor.shutdown(wait=True, cancel_futures=True)
        manager._executor = _DeferredExecutor()
        episode = {"id": "ep-1", "name": "EP 1"}
        media = {"file_id": "video", "path": "source.mp4", "frame_count": 10}
        with (
            patch("app.qwen_trim.registry", SimpleNamespace(has_vlm=True)),
            patch("app.qwen_trim.get_manifest", return_value={"episodes": [episode]}),
            patch("app.qwen_trim.episode_media", return_value=media),
        ):
            job = manager.submit("dataset-1", QwenTrimRequest(episode_ids=["ep-1"], media_file_ids={"ep-1": "video"}))
        self.assertIn(("dataset-1", "ep-1"), manager._active_episode_jobs)

        cancelled = manager.cancel(job["id"])

        self.assertEqual("cancelled", cancelled["status"])
        self.assertNotIn(("dataset-1", "ep-1"), manager._active_episode_jobs)

    def test_cancel_route_and_frontend_contract(self) -> None:
        routes = {(route.path, method) for route in app.routes for method in getattr(route, "methods", set())}
        self.assertIn(("/api/jobs/{job_id}/cancel", "POST"), routes)

        root = Path(__file__).resolve().parents[1]
        html = (root / "index.html").read_text(encoding="utf-8")
        javascript = (root / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="cancelAnalysisButton"', html)
        self.assertIn('/api/jobs/${encodeURIComponent(current.id)}/cancel', javascript)
        self.assertIn('if (jobStatus === "cancelled")', javascript)


if __name__ == "__main__":
    unittest.main()
