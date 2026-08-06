from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.analyzer import analyze_episode
from app.batch_jobs import BatchAnalysisJobManager
from app.behavior_annotator import BehaviorJobManager, annotate_episode_behavior
from app.no_action_trim import analyze_no_action_trim
from app.qwen_trim import QwenTrimJobManager, QwenTrimRequest, analyze_qwen_action_trim
from app.schemas import AnalysisRequest, BatchAnalysisRequest, BehaviorAnnotationRequest
from app.storage import require_media_eligibility
from app.video_smoothing import smooth_video


def _media(modality: str = "depth", **updates) -> dict:
    payload = {
        "file_id": "media-1",
        "stream_name": f"head_{modality}",
        "relative_path": f"camera/head_{modality}.raw",
        "path": "missing-media",
        "type": "raw_depth" if modality == "depth" else "video",
        "modality": modality,
        "is_depth_map": modality == "depth",
        "fps": 30.0,
        "frame_count": 10,
        "analysis_eligible": False,
        "vlm_eligible": False,
        "smoothing_eligible": False,
    }
    payload.update(updates)
    return payload


def _episode(media: dict) -> dict:
    return {
        "id": "ep-1",
        "name": "EP 1",
        "primary_media_file_id": media["file_id"],
        "media_streams": [media],
        **media,
    }


class MediaEligibilityGuardTests(unittest.TestCase):
    def test_shared_guard_requires_rgb_and_explicit_operation_flag(self) -> None:
        rgb = _media(
            "rgb",
            type="video",
            is_depth_map=False,
            analysis_eligible=True,
            vlm_eligible=True,
            smoothing_eligible=True,
        )
        for operation in ("analysis", "analysis_vlm", "no_action_trim", "video_smoothing", "vlm_behavior", "qwen_trim"):
            self.assertIs(rgb, require_media_eligibility(rgb, operation))

        requirements = {
            "analysis": "analysis_eligible=true",
            "no_action_trim": "analysis_eligible=true",
            "video_smoothing": "smoothing_eligible=true",
            "vlm_behavior": "vlm_eligible=true",
            "qwen_trim": "vlm_eligible=true",
        }
        for modality in ("depth", "infrared"):
            blocked = _media(
                modality,
                analysis_eligible=True,
                vlm_eligible=True,
                smoothing_eligible=True,
            )
            for operation, requirement in requirements.items():
                with self.subTest(modality=modality, operation=operation):
                    with self.assertRaisesRegex(ValueError, requirement):
                        require_media_eligibility(blocked, operation)
                    try:
                        require_media_eligibility(blocked, operation)
                    except ValueError as exc:
                        self.assertIn("modality=rgb", str(exc))
                        self.assertIn("Depth/IR", str(exc))

    def test_direct_processing_functions_reject_depth_before_models_or_files(self) -> None:
        depth = _media()
        episode = _episode(depth)
        with self.assertRaisesRegex(ValueError, "smoothing_eligible=true"):
            smooth_video("dataset", episode, depth, lambda *_: None)
        with self.assertRaisesRegex(ValueError, "analysis_eligible=true"):
            analyze_no_action_trim("dataset", {}, episode, lambda *_: None, depth)
        with self.assertRaisesRegex(ValueError, "vlm_eligible=true"):
            analyze_qwen_action_trim("dataset", {}, episode, depth, QwenTrimRequest(), lambda *_: None)
        with self.assertRaisesRegex(ValueError, "analysis_eligible=true"):
            analyze_episode("dataset", episode, AnalysisRequest(), lambda *_: None)
        with self.assertRaisesRegex(ValueError, "vlm_eligible=true"):
            annotate_episode_behavior(
                "dataset",
                {"episodes": [episode]},
                episode,
                BehaviorAnnotationRequest(force=True),
                lambda *_: None,
            )

    def test_job_submission_guards_run_before_reuse_or_model_checks(self) -> None:
        depth = _media()
        episode = _episode(depth)
        manifest = {"episodes": [episode], "schema_profile": {"status": "completed"}}

        batch = BatchAnalysisJobManager()
        behavior = BehaviorJobManager()
        qwen = QwenTrimJobManager(max_workers=1)
        try:
            with patch("app.batch_jobs.get_manifest", return_value=manifest):
                with self.assertRaisesRegex(ValueError, "vlm_eligible=true"):
                    batch.submit("dataset", BatchAnalysisRequest(operation="vlm_behavior", episode_ids=["ep-1"]))
            with (
                patch("app.behavior_annotator.get_episode", return_value=(manifest, episode)),
                patch("app.behavior_annotator.registry", SimpleNamespace(has_vlm=False)),
            ):
                with self.assertRaisesRegex(ValueError, "vlm_eligible=true"):
                    behavior.submit("dataset", "ep-1", BehaviorAnnotationRequest())
            with (
                patch("app.qwen_trim.get_manifest", return_value=manifest),
                patch("app.qwen_trim.registry", SimpleNamespace(has_vlm=False)),
            ):
                with self.assertRaisesRegex(ValueError, "vlm_eligible=true"):
                    qwen.submit(
                        "dataset",
                        QwenTrimRequest(episode_ids=["ep-1"], media_file_ids={"ep-1": "media-1"}),
                    )
        finally:
            batch._executor.shutdown(wait=True, cancel_futures=True)
            behavior._executor.shutdown(wait=True, cancel_futures=True)
            qwen._executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    unittest.main()
