from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.full_export import SUBTASK_JSON_OUTPUT_FORMAT, export_episode
from app.quality_evidence import QUALITY_EVIDENCE_SCHEMA, build_quality_evidence


class QualityEvidenceTests(unittest.TestCase):
    def test_stage_findings_are_normalized_to_checks(self) -> None:
        document = build_quality_evidence(
            dataset_id="dataset",
            episode_id="episode-1",
            frame_count=20,
            fps=10.0,
            stages=[
                {"id": "c3", "name": "视频质量与整手可见", "status": "warning", "message": "需要复核", "metrics": {"visible_ratio": 0.5}},
                {"id": "s1", "name": "突变与 Jerk", "status": "completed", "metrics": {}},
            ],
            findings=[
                {"stage": "c3", "severity": "review", "start_frame": 2, "end_frame": 4, "reason": "部分点不可视", "confidence": 0.7},
            ],
            segments=[
                {"start_frame": 0, "end_frame": 1, "state": "valid"},
                {"start_frame": 2, "end_frame": 4, "state": "uncertain"},
            ],
            pipeline_version=19,
            pipeline_schema="alice/paper-curation/v1",
            run_id="run-1",
            timeline_id="timeline-1",
            source_video={"file_id": "rgb", "relative_path": "video.mp4"},
            source_signatures=[{"relative_path": "video.mp4", "sample_sha256": "abc"}],
            config={"sudden_change_sigma": 5.0},
        )

        self.assertEqual(QUALITY_EVIDENCE_SCHEMA, document["schema"])
        self.assertEqual("review", document["checks"][0]["verdict"])
        self.assertEqual(1, document["checks"][0]["finding_count"])
        self.assertEqual(3, document["aggregate"]["review_frame_count"])
        self.assertEqual(17, document["aggregate"]["valid_frame_count"])
        self.assertEqual("full_analysis_video", document["frame_index_space"])
        self.assertEqual("run-1", document["pipeline"]["run_id"])
        self.assertEqual(1, document["source"]["source_signature_count"])

    def test_overlapping_findings_do_not_double_count_aggregate_ranges(self) -> None:
        document = build_quality_evidence(
            dataset_id="dataset",
            episode_id="episode-1",
            frame_count=10,
            fps=30.0,
            stages=[{"id": "s1", "status": "completed"}],
            findings=[
                {"stage": "s1", "severity": "reject", "start_frame": 2, "end_frame": 5, "reason": "jump"},
                {"stage": "s1", "severity": "reject", "start_frame": 4, "end_frame": 7, "reason": "rot6d"},
            ],
            segments=[],
        )

        self.assertEqual("fail", document["checks"][0]["verdict"])
        self.assertEqual(6, document["aggregate"]["invalid_frame_count"])
        self.assertEqual(1, document["aggregate"]["invalid_range_count"])

    def test_skipped_stage_is_not_reported_as_pass(self) -> None:
        document = build_quality_evidence(
            dataset_id="dataset",
            episode_id="episode-1",
            frame_count=1,
            fps=1.0,
            stages=[{"id": "s4", "status": "skipped", "message": "未实现"}],
            findings=[],
        )

        self.assertEqual("skipped", document["checks"][0]["verdict"])
        self.assertEqual("skipped", document["aggregate"]["verdict"])

    def test_warning_stage_keeps_aggregate_at_review_without_frame_ranges(self) -> None:
        document = build_quality_evidence(
            dataset_id="dataset",
            episode_id="episode-1",
            frame_count=10,
            fps=30.0,
            stages=[
                {"id": "t0", "status": "completed"},
                {"id": "c3", "status": "warning", "message": "外部标定不可用"},
            ],
            findings=[],
            segments=[{"start_frame": 0, "end_frame": 9, "state": "valid"}],
        )

        self.assertEqual("review", document["checks"][1]["verdict"])
        self.assertEqual("review", document["aggregate"]["verdict"])

    def test_subtask_export_carries_evidence_sidecar(self) -> None:
        with TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "output"
            evidence = build_quality_evidence(
                dataset_id="dataset",
                episode_id="episode-1",
                frame_count=4,
                fps=4.0,
                stages=[{"id": "c3", "status": "completed"}],
                findings=[],
                segments=[{"start_frame": 0, "end_frame": 3, "state": "valid"}],
            )
            result = export_episode(
                output_root,
                {"id": "dataset", "root_path": temporary, "files": []},
                {"id": "episode-1", "name": "demo", "frame_count": 4, "fps": 4.0},
                {"frame_count": 4, "fps": 4.0},
                {
                    "source_video": {"frame_count": 4, "fps": 4.0},
                    "segments": [{"start_frame": 0, "end_frame": 3, "state": "valid"}],
                    "quality_evidence": evidence,
                },
                {"segments": []},
                lambda *_: None,
                output_format=SUBTASK_JSON_OUTPUT_FORMAT,
            )

            pair = result["pairs"][0]
            sidecar = Path(pair["quality_evidence_json"])
            self.assertTrue(sidecar.is_file())
            self.assertEqual(QUALITY_EVIDENCE_SCHEMA, json.loads(sidecar.read_text(encoding="utf-8")).get("schema"))
            self.assertEqual("quality_evidence.json", json.loads(Path(pair["subtasks_json"]).read_text(encoding="utf-8"))["quality_evidence"]["file"])


if __name__ == "__main__":
    unittest.main()
