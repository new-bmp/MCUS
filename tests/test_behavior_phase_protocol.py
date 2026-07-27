from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.behavior_annotator import (
    BEHAVIOR_ARTIFACT_VERSION,
    PHASE_LABELS,
    _apply_dataset_task_fallback,
    _constrain_segments_to_ranges,
    _normalize_phase_label,
    _sample_indices_in_ranges,
    _segments_follow_phase_protocol,
    _validate_result,
    annotate_episode_behavior,
)
from app.behavior_prompt import META_ACTION_TRANSLATIONS, TRI_LEVEL_PROTOCOL_SCHEMA, TRI_LEVEL_PROTOCOL_VERSION
from app.models import ModelRegistry
from app.schemas import BehaviorAnnotationRequest


class BehaviorPhaseProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ontology = {
            "categories": [{"label": "pick", "task": "pick", "verbs": ["pick"], "objects": ["cup"], "descriptions": []}],
        }
        self.episode = {"frame_count": 100, "fps": 10.0}

    def test_valid_range_sampling_never_uses_rejected_gap(self) -> None:
        ranges = [(10, 19), (50, 59)]
        indices = _sample_indices_in_ranges(100, 8, ranges)

        self.assertEqual(8, len(indices))
        self.assertTrue(all(10 <= index <= 19 or 50 <= index <= 59 for index in indices))
        self.assertIn(10, indices)
        self.assertIn(59, indices)

    def test_segments_outside_precheck_ranges_become_unknown(self) -> None:
        constrained = _constrain_segments_to_ranges(
            [{"start_frame": 0, "end_frame": 99, "phase_label": "grasp", "label": "grasp", "confidence": 0.9, "boundary_source": "vlm"}],
            [(10, 19), (50, 59)],
            100,
            10.0,
        )

        self.assertEqual(0, constrained[0]["start_frame"])
        self.assertEqual(99, constrained[-1]["end_frame"])
        self.assertEqual(["unknown", "grasp", "unknown", "grasp", "unknown"], [item["phase_label"] for item in constrained])
        self.assertEqual("curation_precheck", constrained[0]["boundary_source"])
        self.assertTrue(_segments_follow_phase_protocol({"segments": constrained}, 100))

    def test_stage_and_label_inputs_normalize_to_contiguous_phase_labels(self) -> None:
        raw = {
            "task_label": "pick",
            "confidence": 0.9,
            "segments": [
                {"start_frame": 70, "end_frame": 90, "label": "carry", "description": "carry cup"},
                {"start_frame": 45, "end_frame": 60, "stage": "contact", "description": "grasp cup"},
                {"start_frame": 10, "end_frame": 40, "phase_label": "approach", "description": "reach cup"},
            ],
        }

        result = _validate_result(raw, self.ontology, self.episode, [0, 50, 99])
        segments = result["segments"]

        self.assertEqual("pick", result["task_label"])
        self.assertEqual(
            ["unknown", "reach", "unknown", "grasp", "unknown", "transport", "unknown"],
            [item["phase_label"] for item in segments],
        )
        self.assertEqual([item["phase_label"] for item in segments], [item["label"] for item in segments])
        self.assertEqual((0, 99), (segments[0]["start_frame"], segments[-1]["end_frame"]))
        for left, right in zip(segments, segments[1:]):
            self.assertEqual(left["end_frame"] + 1, right["start_frame"])
        self.assertEqual({"vlm"}, {item["boundary_source"] for item in segments})
        self.assertTrue(all("stage" not in item for item in segments))

    def test_adjacent_equivalent_phases_merge_and_unknown_is_explicit(self) -> None:
        raw = {
            "task_label": "pick",
            "segments": [
                {"start_frame": 0, "end_frame": 20, "phase_label": "reach", "primary_targets": ["cup"]},
                {"start_frame": 21, "end_frame": 40, "label": "approach", "primary_targets": ["cup"]},
                {"start_frame": 41, "end_frame": 99, "stage": "invented_phase"},
            ],
        }

        result = _validate_result(raw, self.ontology, self.episode, [0, 50, 99])

        self.assertEqual(["reach", "unknown"], [item["phase_label"] for item in result["segments"]])
        self.assertEqual("unknown", _normalize_phase_label("invented phase"))
        self.assertTrue(set(item["phase_label"] for item in result["segments"]).issubset(set(PHASE_LABELS)))

    def test_other_task_uses_dataset_name_only_when_vlm_content_confirms_it(self) -> None:
        result = {
            "task_label": "other",
            "behavior_description": "Assemble a toy sandwich by stacking bread and vegetables.",
            "object_nouns": ["sandwich", "bread"],
            "segments": [],
            "warnings": [],
        }

        resolved = _apply_dataset_task_fallback(result, {"name": "make_sandwich"})
        unrelated = _apply_dataset_task_fallback(result, {"name": "boil_serve_egg"})

        self.assertEqual("make_sandwich", resolved["task_label"])
        self.assertEqual("dataset_name_confirmed_by_vlm_content", resolved["task_label_source"])
        self.assertEqual("other", unrelated["task_label"])

    def test_same_phase_stays_split_when_object_or_instance_changes(self) -> None:
        raw = {
            "task_label": "pick",
            "primary_targets": [
                {"name": "cup", "visible_evidence_frames": [0], "confidence": 0.9},
                {"name": "lid", "visible_evidence_frames": [50], "confidence": 0.9},
            ],
            "segments": [
                {"start_frame": 0, "end_frame": 30, "phase_label": "manipulate", "primary_targets": ["cup"], "target_instance": "cup#1"},
                {"start_frame": 31, "end_frame": 60, "phase_label": "manipulate", "primary_targets": ["lid"], "target_instance": "lid#1"},
                {"start_frame": 61, "end_frame": 99, "phase_label": "manipulate", "primary_targets": ["cup"], "target_instance": "cup#2"},
            ],
        }

        result = _validate_result(raw, self.ontology, self.episode, [0, 50, 99])

        self.assertEqual(3, len(result["segments"]))
        self.assertEqual(["cup#1", "lid#1", "cup#2"], [item["target_instance"] for item in result["segments"]])
        self.assertEqual([["cup"], ["lid"], ["cup"]], [item["primary_targets"] for item in result["segments"]])

    def test_qwen_prompt_uses_tri_level_v3_meta_action_protocol(self) -> None:
        model = ModelRegistry()
        model._vlm = SimpleNamespace(configured=True, endpoint="https://example.invalid/v1", model="fixture")
        model._vlm_key = "fixture-key"
        frame = np.zeros((16, 16, 3), dtype=np.uint8)

        with patch.object(model, "_request_json", return_value={}) as request:
            model.annotate_behavior(
                [(0, 0.0, frame), (9, 0.9, frame)],
                self.ontology["categories"],
                "fixture",
                video_length=100,
                duration=10.0,
            )

        system_prompt = request.call_args.kwargs["system_prompt"]
        user_prompt = request.call_args.kwargs["content"][0]["text"]
        self.assertIn("三级粒度", system_prompt)
        self.assertIn("原视频总帧数 = 100", system_prompt)
        self.assertIn("每个 Fine 片段原则上至少覆盖 5 秒", system_prompt)
        self.assertIn('"coarse"', system_prompt)
        self.assertIn('"medium"', system_prompt)
        self.assertIn('"fine"', system_prompt)
        self.assertIn("0->0, 1->9", user_prompt)
        for label, translation in META_ACTION_TRANSLATIONS.items():
            self.assertIn(f"- {label}: {translation}", system_prompt)

    def test_tri_level_result_preserves_all_levels_and_maps_fine_skills(self) -> None:
        raw = {
            "coarse": {"summary": "零件装配"},
            "medium": [
                {"start_frame": 0, "end_frame": 49, "description": "右臂抓取插头"},
                {"start_frame": 50, "end_frame": 99, "description": "右臂插入插头"},
            ],
            "fine": [
                {"start_frame": 0, "end_frame": 49, "description": "右臂抓取插头", "skill": "Grasp"},
                {"start_frame": 50, "end_frame": 99, "description": "右臂插入插头", "skill": "Insert"},
            ],
        }

        result = _validate_result(raw, self.ontology, self.episode, [0, 50, 99])

        self.assertEqual({"version": TRI_LEVEL_PROTOCOL_VERSION, "schema": TRI_LEVEL_PROTOCOL_SCHEMA}, result["annotation_protocol"])
        self.assertEqual("零件装配", result["task_label"])
        self.assertEqual("零件装配", result["coarse"]["summary"])
        self.assertEqual(["Grasp", "Insert"], [item["skill"] for item in result["fine"]])
        self.assertEqual(["grasp", "manipulate"], [item["phase_label"] for item in result["segments"]])
        self.assertEqual((0, 99), (result["medium"][0]["start_frame"], result["medium"][-1]["end_frame"]))

    def test_malformed_optional_values_do_not_break_protocol_validation(self) -> None:
        raw = {
            "task_label": "PICK",
            "confidence": "not-a-number",
            "direction": "FORWARD",
            "primary_targets": [None, {"name": "cup", "visible_evidence_frames": ["bad", "50"]}],
            "segments": [
                None,
                {"start_frame": "bad", "end_frame": "20", "phase_label": "伸手", "confidence": None},
                {"start_frame": 30, "end_frame": 99, "phase_label": "抓取", "primary_targets": "cup"},
            ],
            "object_nouns": "cup",
            "warnings": "none",
        }

        result = _validate_result(raw, self.ontology, self.episode, [0, 50, 99])

        self.assertEqual("pick", result["task_label"])
        self.assertEqual("forward", result["direction"])
        self.assertEqual(0.0, result["confidence"])
        self.assertEqual(["reach", "unknown", "grasp"], [item["phase_label"] for item in result["segments"]])
        self.assertEqual(["cup"], result["object_nouns"])
        self.assertEqual([], result["warnings"])

    def test_annotation_integrates_joint_refiner_and_writes_version_three_in_temp_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "episode.mp4"
            source.write_bytes(b"fixture-video")
            media = {
                "file_id": "video",
                "stream_name": "episode.mp4",
                "path": str(source),
                "relative_path": "episode.mp4",
                "fps": 10.0,
                "frame_count": 10,
            }
            episode = {
                "id": "ep",
                "name": "EP",
                "relative_path": "episode.mp4",
                "fps": 10.0,
                "frame_count": 10,
                "primary_media_file_id": "video",
                "media_streams": [media],
            }
            manifest = {"id": "fixture", "root_path": str(root), "files": [], "schema_profile": {}}
            ontology = {
                "source": "builtin",
                "root": "builtin:test",
                "category_count": 1,
                "fingerprint": "fixture",
                "categories": self.ontology["categories"],
            }
            raw = {
                "task_label": "pick",
                "confidence": 0.9,
                "segments": [
                    {"start_frame": 0, "end_frame": 4, "phase_label": "reach"},
                    {"start_frame": 5, "end_frame": 9, "phase_label": "grasp"},
                ],
            }

            def artifact_dir(_dataset_id: str, category: str) -> Path:
                path = root / ".alicePD" / category
                path.mkdir(parents=True, exist_ok=True)
                return path

            def refine(segments, _fps, _frame_count, pose):
                self.assertIsNotNone(pose)
                output = [dict(item) for item in segments]
                output[0]["boundary_source"] = "joint_refined"
                output[1]["boundary_source"] = "joint_refined"
                return output

            registry = SimpleNamespace(
                has_vlm=True,
                annotate_behavior=lambda *_args, **_kwargs: raw,
                status=lambda: {"vlm": {"model": "fixture"}},
            )
            with (
                patch("app.behavior_annotator.registry", registry),
                patch("app.behavior_annotator.load_behavior_ontology", return_value=ontology),
                patch("app.behavior_annotator.episode_media", return_value=media),
                patch("app.behavior_annotator.preferred_smoothed_media", return_value=(media, None)),
                patch("app.behavior_annotator.read_frame", return_value=np.zeros((8, 8, 3), dtype=np.uint8)),
                patch("app.behavior_annotator.load_episode_joint_pose", return_value=np.zeros((10, 2))),
                patch("app.behavior_annotator.refine_behavior_boundaries", side_effect=refine) as refiner,
                patch("app.behavior_annotator.dataset_artifact_dir", side_effect=artifact_dir),
                patch("app.behavior_annotator.record_change", return_value={"id": "change", "status": "pending", "revision": 1}),
            ):
                result = annotate_episode_behavior(
                    "fixture",
                    manifest,
                    episode,
                    BehaviorAnnotationRequest(sample_count=6, force=True),
                    lambda _progress, _message: None,
                )

        self.assertEqual(BEHAVIOR_ARTIFACT_VERSION, result["artifact_version"])
        self.assertEqual(3, result["artifact_version"])
        self.assertEqual(TRI_LEVEL_PROTOCOL_SCHEMA, result["annotation_protocol"]["schema"])
        self.assertEqual("joint_refined", result["boundary_refinement"]["source"])
        self.assertEqual(2, result["boundary_refinement"]["refined_segment_count"])
        self.assertEqual(["reach", "grasp"], [item["phase_label"] for item in result["segments"]])
        refiner.assert_called_once()


if __name__ == "__main__":
    unittest.main()
