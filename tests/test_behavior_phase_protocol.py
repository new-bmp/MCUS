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
    _adaptive_window_indices,
    _constrain_segments_to_ranges,
    _merge_window_segments,
    _normalize_phase_label,
    _plan_behavior_windows,
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

    def test_window_planner_never_crosses_bad_frame_gaps(self) -> None:
        windows = _plan_behavior_windows(1300, 30.0, [(0, 600), (800, 1299)])

        self.assertEqual((0, 600), (windows[0]["start_frame"], windows[0]["end_frame"]))
        self.assertTrue(all(
            0 <= item["start_frame"] <= item["end_frame"] <= 600
            or 800 <= item["start_frame"] <= item["end_frame"] <= 1299
            for item in windows
        ))
        self.assertFalse(any(item["start_frame"] < 800 <= item["end_frame"] for item in windows))

    def test_adaptive_window_sampling_respects_budget_and_keeps_joint_event(self) -> None:
        window = {
            "window_id": "window-1",
            "start_frame": 0,
            "end_frame": 199,
        }
        joint_score = np.zeros(200, dtype=np.float64)
        joint_score[100] = 20.0
        cache = {}

        indices, metrics = _adaptive_window_indices(
            window,
            10.0,
            24,
            cache,
            lambda _index: np.zeros((16, 16, 3), dtype=np.uint8),
            joint_score=joint_score,
        )

        self.assertLessEqual(len(indices), 24)
        self.assertGreaterEqual(len(indices), 12)
        self.assertIn(0, indices)
        self.assertIn(199, indices)
        self.assertIn(100, indices)
        self.assertEqual([100], metrics["joint_event_frames"])

    def test_overlapping_window_results_merge_by_confidence_and_window_center(self) -> None:
        first = {
            "window_id": "w1",
            "start_frame": 0,
            "end_frame": 99,
            "segments": [{
                "start_frame": 0,
                "end_frame": 99,
                "phase_label": "grasp",
                "skill": "Grasp",
                "confidence": 0.8,
                "primary_targets": ["book"],
                "target_instance": "book#1",
            }],
        }
        second = {
            "window_id": "w2",
            "start_frame": 80,
            "end_frame": 179,
            "segments": [{
                "start_frame": 80,
                "end_frame": 179,
                "phase_label": "manipulate",
                "skill": "Insert",
                "confidence": 0.8,
                "primary_targets": ["book"],
                "target_instance": "book#1",
            }],
        }

        merged = _merge_window_segments([first, second], 180, 10.0, [(0, 179)])

        self.assertEqual(0, merged[0]["start_frame"])
        self.assertEqual(179, merged[-1]["end_frame"])
        self.assertEqual("Grasp", merged[0]["skill"])
        self.assertEqual("Insert", merged[-1]["skill"])
        self.assertTrue(all(left["end_frame"] + 1 == right["start_frame"] for left, right in zip(merged, merged[1:])))

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

    def test_qwen_prompt_uses_windowed_multi_image_v4_protocol(self) -> None:
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
        self.assertIn("局部时间窗口", system_prompt)
        self.assertIn("Episode 总帧数 = 100", system_prompt)
        self.assertIn("短暂的 Reach、Touch、Release、Withdraw 必须保留", system_prompt)
        self.assertIn('"coarse"', system_prompt)
        self.assertIn('"medium"', system_prompt)
        self.assertIn('"fine"', system_prompt)
        self.assertIn("0->0, 1->9", user_prompt)
        self.assertIn('"label":"pick"', user_prompt)
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

    def test_annotation_integrates_joint_refiner_and_writes_analysis_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            head_source = root / "head.mp4"
            wrist_source = root / "wrist_left.mp4"
            head_source.write_bytes(b"fixture-head-video")
            wrist_source.write_bytes(b"fixture-wrist-video")
            head_media = {
                "file_id": "head",
                "stream_name": "head.mp4",
                "path": str(head_source),
                "relative_path": "head.mp4",
                "fps": 8.0,
                "frame_count": 8,
                "modality": "rgb",
                "vlm_eligible": True,
            }
            wrist_media = {
                "file_id": "wrist-left",
                "stream_name": "wrist_left.mp4",
                "path": str(wrist_source),
                "relative_path": "wrist_left.mp4",
                "fps": 10.0,
                "frame_count": 10,
                "modality": "rgb",
                "vlm_eligible": True,
            }
            episode = {
                "id": "ep",
                "name": "EP",
                "relative_path": "head.mp4",
                "fps": 8.0,
                "frame_count": 8,
                "primary_media_file_id": "head",
                "media_streams": [head_media, wrist_media],
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
            media_by_id = {"head": head_media, "wrist-left": wrist_media}
            with (
                patch("app.behavior_annotator.registry", registry),
                patch("app.behavior_annotator.load_behavior_ontology", return_value=ontology),
                patch(
                    "app.behavior_annotator.episode_media",
                    side_effect=lambda _episode, file_id: media_by_id[str(file_id)],
                ),
                patch("app.behavior_annotator.read_frame", return_value=np.zeros((8, 8, 3), dtype=np.uint8)),
                patch(
                    "app.behavior_annotator.load_episode_joint_pose",
                    return_value=np.zeros((10, 2)),
                ) as load_joint_pose,
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
                    analysis_media_override=wrist_media,
                    analysis_source_kind="curation_non_rejected_segments",
                )

        self.assertEqual(BEHAVIOR_ARTIFACT_VERSION, result["artifact_version"])
        self.assertEqual(TRI_LEVEL_PROTOCOL_SCHEMA, result["annotation_protocol"]["schema"])
        self.assertEqual("joint_refined", result["boundary_refinement"]["source"])
        self.assertEqual(2, result["boundary_refinement"]["refined_segment_count"])
        self.assertEqual(["reach", "grasp"], [item["phase_label"] for item in result["segments"]])
        self.assertEqual("wrist-left", result["sampling"]["media_file_id"])
        self.assertEqual("wrist_left.mp4", result["sampling"]["stream_name"])
        self.assertEqual("wrist-left", result["source_video"]["file_id"])
        self.assertEqual(10, result["source_video"]["frame_count"])
        self.assertEqual("wrist-left", result["analysis_video"]["file_id"])
        self.assertEqual("analysis_video", result["sampling"]["frame_space"])
        self.assertEqual({"frame_space": "analysis_video", "frame_count": 10, "fps": 10.0}, result["timeline"])
        load_joint_pose.assert_called_once_with(
            manifest,
            episode,
            frame_count=10,
            reference_media_file_id="wrist-left",
        )
        refiner.assert_called_once()

    def test_annotation_runs_windowed_multi_image_requests_and_global_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "episode.mp4"
            source.write_bytes(b"windowed-video-fixture" * 4096)
            media = {
                "file_id": "rgb",
                "stream_name": "episode.mp4",
                "path": str(source),
                "relative_path": "episode.mp4",
                "fps": 10.0,
                "frame_count": 450,
                "modality": "rgb",
                "vlm_eligible": True,
            }
            episode = {
                "id": "ep-windowed",
                "name": "EP windowed",
                "relative_path": "episode.mp4",
                "fps": 10.0,
                "frame_count": 450,
                "primary_media_file_id": "rgb",
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
            calls = []
            summaries = []

            def annotate(_frames, _ontology, _context, **kwargs):
                calls.append(kwargs)
                start = kwargs["window_start"]
                end = kwargs["window_end"]
                return {
                    "window_summary": "move books on a shelf",
                    "fine": [{
                        "start_frame": start,
                        "end_frame": end,
                        "description": "manipulate a book",
                        "skill": "Insert",
                        "confidence": 0.75,
                        "object_nouns": ["book"],
                        "primary_targets": ["book"],
                        "target_instance": "book#1",
                    }],
                    "object_nouns": ["book"],
                }

            def summarize(window_results, _ontology, _context, **_kwargs):
                summaries.append(window_results)
                return {
                    "coarse": {"summary": "organize bookshelf"},
                    "medium": [{"start_frame": 0, "end_frame": 449, "description": "insert books into shelf"}],
                    "confidence": 0.8,
                }

            def artifact_dir(_dataset_id: str, category: str) -> Path:
                path = root / ".alicePD" / category
                path.mkdir(parents=True, exist_ok=True)
                return path

            registry = SimpleNamespace(
                has_vlm=True,
                annotate_behavior=annotate,
                summarize_behavior_windows=summarize,
                status=lambda: {"vlm": {"model": "fixture"}},
            )
            with (
                patch("app.behavior_annotator.registry", registry),
                patch("app.behavior_annotator.load_behavior_ontology", return_value=ontology),
                patch("app.behavior_annotator.episode_media", return_value=media),
                patch("app.behavior_annotator.read_frame", return_value=np.zeros((16, 16, 3), dtype=np.uint8)),
                patch("app.behavior_annotator.load_episode_joint_pose", return_value=None),
                patch("app.behavior_annotator.dataset_artifact_dir", side_effect=artifact_dir),
                patch("app.behavior_annotator.record_change", return_value={"id": "change", "status": "pending", "revision": 1}),
            ):
                result = annotate_episode_behavior(
                    "fixture",
                    manifest,
                    episode,
                    BehaviorAnnotationRequest(sample_count=24, force=True),
                    lambda _progress, _message: None,
                    analysis_media_override=media,
                )

        self.assertEqual(3, len(calls))
        self.assertEqual(1, len(summaries))
        self.assertEqual("windowed_adaptive_multi_image_v1", result["sampling"]["strategy"])
        self.assertEqual(3, len(result["sampling"]["windows"]))
        self.assertLessEqual(result["sampling"]["total_image_count"], 72)
        self.assertIn(0, result["sampling"]["frames"])
        self.assertIn(449, result["sampling"]["frames"])
        self.assertEqual("organize bookshelf", result["coarse"]["summary"])
        self.assertEqual(["book"], result["object_nouns"])


if __name__ == "__main__":
    unittest.main()
