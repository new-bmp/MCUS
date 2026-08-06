from __future__ import annotations

import unittest
from unittest.mock import patch

from app.batch_jobs import BatchAnalysisJobManager, _projection_result_matches_request
from app.schemas import BatchAnalysisRequest


class _ImmediateHandPoseRegistry:
    def __init__(self, backend: str = "mediapipe", loaded: bool = True, error: str | None = None) -> None:
        self.hand_pose = {
            "backend": backend,
            "loaded": loaded,
            "loading": False,
            "error": error,
        }
        self.configs = []

    @property
    def has_hand_pose(self) -> bool:
        return bool(self.hand_pose.get("loaded"))

    def status(self) -> dict:
        return {"hand_pose": dict(self.hand_pose)}

    def configure_hand_pose_async(self, config) -> dict:
        self.configs.append(config)
        self.hand_pose = {
            "backend": config.kind,
            "loaded": True,
            "loading": False,
            "error": None,
            "model_path": config.model_path,
        }
        return self.status()


class ProjectionBackendSwitchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = BatchAnalysisJobManager()
        self.manager._jobs["job"] = {"id": "job", "status": "running"}
        self.manager._register_cancellation("job")

    def tearDown(self) -> None:
        self.manager._executor.shutdown(wait=True, cancel_futures=True)

    def test_already_loaded_backend_is_reused(self) -> None:
        fake = _ImmediateHandPoseRegistry("mediapipe", loaded=True)
        request = BatchAnalysisRequest(
            operation="projection_correction",
            episode_ids=["ep"],
            hand_pose_backend="mediapipe",
        )
        with patch("app.batch_jobs.registry", fake):
            status = self.manager._ensure_projection_backend("job", request, timeout_seconds=1)
        self.assertEqual("mediapipe", status["backend"])
        self.assertEqual([], fake.configs)

    def test_requested_alicepose_backend_is_loaded_with_path(self) -> None:
        fake = _ImmediateHandPoseRegistry("mediapipe", loaded=True)
        request = BatchAnalysisRequest(
            operation="projection_correction",
            episode_ids=["ep"],
            hand_pose_backend="alicepose",
            hand_pose_model_path="Alicepose-21k-v1.pt",
        )
        with patch("app.batch_jobs.registry", fake):
            status = self.manager._ensure_projection_backend("job", request, timeout_seconds=1)
        self.assertEqual("alicepose", status["backend"])
        self.assertEqual(1, len(fake.configs))
        self.assertEqual("Alicepose-21k-v1.pt", fake.configs[0].model_path)

    def test_wrist_source_change_does_not_reuse_the_other_result(self) -> None:
        egodex = BatchAnalysisRequest(
            operation="projection_correction",
            episode_ids=["ep"],
            wrist_point_source="egodex",
        )
        model = egodex.model_copy(update={"wrist_point_source": "model"})
        status = {"available": True, "summary": {"wrist_point_source": "egodex"}}

        self.assertTrue(_projection_result_matches_request(status, egodex))
        self.assertFalse(_projection_result_matches_request(status, model))


if __name__ == "__main__":
    unittest.main()
