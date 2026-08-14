from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

from app.import_jobs import DatasetImportJobManager
from app.schemas import PathOpenRequest


def _report(token: str = "valid-token") -> dict:
    return {
        "confirmation_token": token,
        "root_mode": "dataset",
        "status": "ready",
        "capabilities": {"can_import": True},
    }


def _wait_for_terminal(manager: DatasetImportJobManager, job_id: str, timeout: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        if job["status"] in {"complete", "failed", "cancelled"}:
            return job
        time.sleep(0.01)
    raise AssertionError(f"Import job did not finish: {manager.get(job_id)}")


class DatasetImportJobTests(unittest.TestCase):
    def test_completed_job_exposes_scan_progress_and_manifest(self) -> None:
        manager = DatasetImportJobManager()
        manifest = {"id": "dataset", "file_count": 42, "episode_count": 3}

        def scan(*_args, **kwargs):
            kwargs["progress"](50, "正在探测视频", {"file_count": 42, "video_count": 4, "episode_count": 2})
            return manifest

        try:
            with (
                patch("app.import_jobs.inspect_dataset_format", return_value=_report()),
                patch("app.import_jobs.scan_dataset", side_effect=scan),
            ):
                submitted = manager.submit(PathOpenRequest(path="dataset", analyze_schema=False), "valid-token")
                completed = _wait_for_terminal(manager, submitted["id"])
        finally:
            manager._executor.shutdown(wait=True, cancel_futures=True)

        self.assertEqual("complete", completed["status"])
        self.assertEqual(100.0, completed["progress"])
        self.assertEqual(manifest, completed["result"])
        self.assertEqual(42, completed["file_count"])
        self.assertEqual(3, completed["episode_count"])

    def test_stale_token_fails_before_scan(self) -> None:
        manager = DatasetImportJobManager()
        try:
            with (
                patch("app.import_jobs.inspect_dataset_format", return_value=_report("new-token")),
                patch("app.import_jobs.scan_dataset") as scan,
            ):
                submitted = manager.submit(PathOpenRequest(path="dataset", analyze_schema=False), "old-token")
                failed = _wait_for_terminal(manager, submitted["id"])
                scan.assert_not_called()
        finally:
            manager._executor.shutdown(wait=True, cancel_futures=True)

        self.assertEqual("failed", failed["status"])
        self.assertIn("发生变化", failed["error"])

    def test_running_import_can_be_cancelled(self) -> None:
        manager = DatasetImportJobManager()
        started = threading.Event()
        release = threading.Event()

        def scan(*_args, **kwargs):
            started.set()
            release.wait(2)
            kwargs["check_cancelled"]()
            return {"id": "should-not-complete"}

        try:
            with (
                patch("app.import_jobs.inspect_dataset_format", return_value=_report()),
                patch("app.import_jobs.scan_dataset", side_effect=scan),
            ):
                submitted = manager.submit(PathOpenRequest(path="dataset", analyze_schema=False), "valid-token")
                self.assertTrue(started.wait(1))
                cancelling = manager.cancel(submitted["id"])
                self.assertEqual("cancelling", cancelling["status"])
                release.set()
                cancelled = _wait_for_terminal(manager, submitted["id"])
        finally:
            release.set()
            manager._executor.shutdown(wait=True, cancel_futures=True)

        self.assertEqual("cancelled", cancelled["status"])
        self.assertIsNone(cancelled["error"])


if __name__ == "__main__":
    unittest.main()
