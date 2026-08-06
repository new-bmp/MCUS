from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.models import OPEN_VOCAB_HAND_CLASSES, _open_vocab_proximity_classes, normalize_yoloe_object_terms
from app.no_action_trim import NO_ACTION_TRIM_ARTIFACT_VERSION, analyze_no_action_trim


class YoloeTermNormalizationTests(unittest.TestCase):
    def test_english_modifiers_relations_and_duplicates_reduce_to_detectable_nouns(self) -> None:
        values = [
            "small red plastic block on the left",
            "two blue cups near the gripper",
            "grasped green wooden lid beside the bowl",
            "the second yellow metal bowl at the back",
            "Small RED plastic blocks on the left",
            "orange basket on the right",
            "jenga toy block",
            "2nd shiny red block to the left",
            "a pair of blue cups",
            "red",
            "plastic",
        ]

        self.assertEqual(["block", "cup", "lid", "bowl", "basket", "jenga block"], normalize_yoloe_object_terms(values))
        self.assertEqual(["orange"], normalize_yoloe_object_terms(["orange"]))

    def test_hand_and_robot_arm_aliases_are_canonical(self) -> None:
        self.assertEqual(
            ["human hand", "robot hand", "hand", "robot gripper"],
            normalize_yoloe_object_terms([
                "gloved hands",
                "robotic arm",
                "mechanical arms",
                "left hand",
                "right hand",
                "robotic grippers",
            ]),
        )
        self.assertEqual(
            ["hand", "robot hand", "robot gripper"],
            normalize_yoloe_object_terms(["左手", "机器人手臂", "机械夹爪"]),
        )

    def test_common_chinese_modifier_phrases_reduce_to_object_heads(self) -> None:
        values = [
            "桌子上的两个蓝色杯子",
            "左侧的小红色塑料方块",
            "正在抓取的红色杯子",
            "机器人右手旁边的绿色盖子",
            "木制桌面上的碗",
            "右侧的橙色篮子",
        ]

        self.assertEqual(["杯子", "方块", "盖子", "碗", "篮子"], normalize_yoloe_object_terms(values))

    def test_prompt_classes_never_lose_or_duplicate_required_hand_terms(self) -> None:
        prompts = _open_vocab_proximity_classes([
            "small red plastic block on the left",
            "Hand",
            "robot hand",
            "red",
        ])

        self.assertIn("block", prompts)
        self.assertIn("jenga block", prompts)
        self.assertFalse(any(value in prompts for value in ("red", "plastic", "left")))
        for required in OPEN_VOCAB_HAND_CLASSES:
            self.assertEqual(1, sum(value.casefold() == required.casefold() for value in prompts))

    def test_no_action_trim_uses_derived_terms_without_rewriting_target_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "ep.targets.alice"
            raw_terms = ["white table", "small red plastic block on the left", "robot arm", "monitor"]
            target.write_text(json.dumps({
                "primary_terms": raw_terms,
                "primary_targets": [
                    {"name": "small red plastic block on the left", "role": "manipulated_object"},
                    {"name": "robotic hands", "role": "effector"},
                ],
            }), encoding="utf-8")
            before = target.read_bytes()
            artifact = root / "ep.trim.alice"
            seen_terms: list[list[str]] = []

            def infer(_frame, terms, _threshold):
                seen_terms.append(list(terms))
                return {"detections": [], "hand_count": 0, "object_count": 0, "nearest": None, "close": False}

            fake_registry = SimpleNamespace(
                has_local=True,
                status=lambda: {"local": {"family": "YOLOE"}},
                infer_open_vocab_proximity=infer,
            )
            media = {
                "file_id": "video",
                "stream_name": "video.mp4",
                "relative_path": "video.mp4",
                "frame_count": 1,
                "fps": 1.0,
                "width": 16,
                "height": 16,
                "modality": "rgb",
                "analysis_eligible": True,
            }
            with (
                patch("app.no_action_trim._target_path", return_value=target),
                patch("app.no_action_trim._artifact_path", return_value=artifact),
                patch("app.no_action_trim.registry", fake_registry),
                patch("app.no_action_trim.read_frame", return_value=np.zeros((16, 16, 3), dtype=np.uint8)),
                patch("app.no_action_trim.record_change", return_value={"id": "change", "status": "pending", "revision": 1}),
            ):
                result = analyze_no_action_trim(
                    "fixture",
                    {},
                    {"id": "ep", "name": "EP"},
                    lambda _progress, _message: None,
                    media,
                    sample_fps=1.0,
                )
                after = target.read_bytes()

        self.assertEqual(before, after)
        self.assertEqual([["block"]], seen_terms)
        self.assertEqual(["small red plastic block on the left", "robotic hands"], result["primary_terms"])
        self.assertEqual(["block"], result["detector_terms"])
        self.assertIn("block", result["prompt_classes"])
        self.assertIn("jenga block", result["prompt_classes"])
        for required in OPEN_VOCAB_HAND_CLASSES:
            self.assertEqual(1, sum(value.casefold() == required.casefold() for value in result["prompt_classes"]))
        self.assertEqual(NO_ACTION_TRIM_ARTIFACT_VERSION, result["artifact_version"])
        self.assertEqual("primary_targets_then_primary_terms_fallback", result["prompt_policy"]["source"])


if __name__ == "__main__":
    unittest.main()
