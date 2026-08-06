from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.full_run import (
    artifact_record,
    finalize_full_run,
    full_run_manifest_path,
    full_run_review_media,
    full_run_stage_dir,
    latest_full_run_id,
    load_full_run_episode_bundle,
    publish_full_run_episode,
    start_full_run,
    update_full_run_episode,
    write_full_timeline_lock,
    write_stamped_artifact,
)


class FullRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset_id = "fixture"
        self.episode = {"id": "ep", "name": "Episode 1", "frame_count": 20, "fps": 10.0}
        self.source_video = self.root / "source.mp4"
        self.source_video.write_bytes(b"source")
        self.media = {
            "file_id": "head",
            "stream_name": "head_rgb",
            "relative_path": "source.mp4",
            "path": str(self.source_video),
            "frame_count": 20,
            "fps": 10.0,
            "width": 640,
            "height": 480,
            "duration": 2.0,
        }
        self.patcher = patch("app.full_run.dataset_artifact_dir", side_effect=self._artifact_dir)
        self.patcher.start()

    def tearDown(self) -> None:
        self.patcher.stop()
        self.temporary.cleanup()

    def _artifact_dir(self, _dataset_id: str, category: str) -> Path:
        path = self.root / ".alicePD" / category
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _build_run(self, run_id: str, *, status: str = "completed", publish: bool = True) -> tuple[dict, dict]:
        start_full_run(self.dataset_id, run_id, ["ep"], {"full_pipeline": True})
        smoothing_dir = full_run_stage_dir(self.dataset_id, run_id, "ep", "smoothing")
        smoothed_video = smoothing_dir / "smoothed.mp4"
        smoothed_video.write_bytes(b"smoothed")
        smoothing = {
            "artifact_path": str(smoothing_dir / "smooth.alice"),
            "output_video": str(smoothed_video),
            "summary": {"frame_count": 20, "fps": 10.0, "width": 640, "height": 480},
        }
        timeline = write_full_timeline_lock(
            self.dataset_id,
            run_id,
            self.episode,
            self.media,
            {**self.media, "path": str(smoothed_video)},
            smoothing=smoothing,
        )
        timeline_id = str(timeline["timeline_id"])
        smoothing = write_stamped_artifact(
            smoothing["artifact_path"], smoothing, run_id, timeline_id, "smoothing"
        )

        curation_path = full_run_stage_dir(self.dataset_id, run_id, "ep", "curation") / "report.alice"
        behavior_path = full_run_stage_dir(self.dataset_id, run_id, "ep", "vlm") / "behavior.alice"
        write_stamped_artifact(
            curation_path,
            {"summary": {"valid_frame_count": 20}, "segments": [{"start_frame": 0, "end_frame": 19, "state": "valid"}]},
            run_id,
            timeline_id,
            "curation_post_vlm",
        )
        write_stamped_artifact(
            behavior_path,
            {"task_label": "pick up the cup", "segments": [{"start_frame": 0, "end_frame": 19, "description": "The right hand picks up the cup."}]},
            run_id,
            timeline_id,
            "vlm",
        )
        artifacts = {
            "smoothing": artifact_record(
                self.dataset_id,
                run_id,
                smoothing["artifact_path"],
                stage="smoothing",
                video=artifact_record(self.dataset_id, run_id, smoothed_video),
            ),
            "curation": artifact_record(self.dataset_id, run_id, curation_path, stage="curation_post_vlm"),
            "behavior": artifact_record(self.dataset_id, run_id, behavior_path, stage="vlm"),
        }
        entry = update_full_run_episode(
            self.dataset_id,
            run_id,
            "ep",
            status=status,
            media_file_id="head",
            timeline=timeline,
            artifacts=artifacts,
            summary={"pair_count": 1},
        )
        finalize_full_run(self.dataset_id, run_id, status, {"completed_count": 1})
        if publish:
            publish_full_run_episode(self.dataset_id, run_id, "ep", "head")
        return entry, timeline

    def test_bundle_loads_only_one_run_and_one_timeline(self) -> None:
        _, timeline = self._build_run("run-a")

        bundle = load_full_run_episode_bundle(self.dataset_id, "ep", "head", "run-a")

        self.assertIsNotNone(bundle)
        self.assertEqual("run-a", bundle["run_id"])
        self.assertEqual(timeline["timeline_id"], bundle["timeline"]["timeline_id"])
        self.assertEqual("run-a", bundle["smoothing"]["full_run_id"])
        self.assertEqual("run-a", bundle["curation"]["full_run_id"])
        self.assertEqual("run-a", bundle["behavior"]["full_run_id"])
        self.assertTrue(Path(bundle["smoothing_video"]).is_file())

    def test_timeline_mismatch_is_not_loaded(self) -> None:
        entry, _ = self._build_run("run-a")
        behavior_path = self.root / ".alicePD" / "full-runs" / "run-a" / entry["artifacts"]["behavior"]["path"]
        payload = json.loads(behavior_path.read_text(encoding="utf-8"))
        payload["timeline_id"] = "different-timeline"
        behavior_path.write_text(json.dumps(payload), encoding="utf-8")

        bundle = load_full_run_episode_bundle(self.dataset_id, "ep", "head", "run-a")

        self.assertIsNotNone(bundle)
        self.assertIsNone(bundle["behavior"])
        self.assertIsNotNone(bundle["curation"])

    def test_missing_mandatory_curation_rejects_the_entire_bundle(self) -> None:
        entry, _ = self._build_run("run-a")
        curation_path = self.root / ".alicePD" / "full-runs" / "run-a" / entry["artifacts"]["curation"]["path"]
        curation_path.unlink()

        self.assertIsNone(load_full_run_episode_bundle(self.dataset_id, "ep", "head", "run-a"))
        with self.assertRaises(RuntimeError):
            full_run_review_media({"id": self.dataset_id}, self.episode, self.media, "run-a")

    def test_artifact_from_another_run_cannot_be_spliced_into_bundle(self) -> None:
        entry_a, _ = self._build_run("run-a")
        self._build_run("run-b")
        manifest_path = full_run_manifest_path(self.dataset_id, "run-b")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["episodes"]["ep"]["artifacts"]["behavior"] = {
            "path": f"../run-a/{entry_a['artifacts']['behavior']['path']}",
            "stage": "vlm",
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        bundle = load_full_run_episode_bundle(self.dataset_id, "ep", "head", "run-b")

        self.assertIsNotNone(bundle)
        self.assertIsNone(bundle["behavior"])

    def test_latest_pointer_rejects_unfinished_episode(self) -> None:
        start_full_run(self.dataset_id, "running-run", ["ep"], {"full_pipeline": True})
        update_full_run_episode(self.dataset_id, "running-run", "ep", status="running", media_file_id="head")
        with self.assertRaises(RuntimeError):
            publish_full_run_episode(self.dataset_id, "running-run", "ep", "head")

        self._build_run("completed-run")

        self.assertEqual("completed-run", latest_full_run_id(self.dataset_id, "ep", "head"))

    def test_existing_run_id_cannot_be_overwritten(self) -> None:
        start_full_run(self.dataset_id, "run-a", ["ep"], {"full_pipeline": True})

        with self.assertRaises(FileExistsError):
            start_full_run(self.dataset_id, "run-a", ["ep"], {"full_pipeline": True})

    def test_review_media_uses_the_matching_run_smoothing_video(self) -> None:
        _, timeline = self._build_run("run-a")
        manifest = {"id": self.dataset_id}

        review_media, bundle = full_run_review_media(manifest, self.episode, self.media, "run-a")

        self.assertIsNotNone(bundle)
        self.assertEqual("full-run-smoothing", review_media["preview_variant"])
        self.assertEqual("run-a", review_media["full_run_id"])
        self.assertEqual(timeline["timeline_id"], review_media["timeline_id"])
        self.assertTrue(Path(review_media["path"]).is_file())


if __name__ == "__main__":
    unittest.main()
