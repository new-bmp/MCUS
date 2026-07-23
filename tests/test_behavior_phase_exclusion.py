from __future__ import annotations

import unittest

from app.annotation_edits import apply_behavior_phase_exclusion


class BehaviorPhaseExclusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.behavior = {
            "created_at": "2026-07-20T00:00:00+00:00",
            "artifact_version": 3,
            "segments": [
                {"start_frame": 0, "end_frame": 1, "phase_label": "reach", "label": "reach"},
                {"start_frame": 2, "end_frame": 5, "phase_label": "transport", "label": "transport"},
                {"start_frame": 6, "end_frame": 7, "phase_label": "reach", "label": "reach"},
                {"start_frame": 8, "end_frame": 9, "phase_label": "place", "label": "place"},
            ],
        }

    def test_selected_phase_is_invalid_and_other_frames_stay_valid(self) -> None:
        result = apply_behavior_phase_exclusion(
            {"segments": [], "samples": [], "summary": {}},
            self.behavior,
            "reach",
            frame_count=10,
            fps=10.0,
        )

        invalid = [(item["start_frame"], item["end_frame"]) for item in result["segments"] if item["state"] == "invalid"]
        valid = [(item["start_frame"], item["end_frame"]) for item in result["segments"] if item["state"] == "valid"]
        self.assertEqual([(0, 1), (6, 7)], invalid)
        self.assertEqual([(2, 5), (8, 9)], valid)
        self.assertTrue(all(item["source"] == "behavior_phase" for item in result["segments"] if item["state"] == "invalid"))
        self.assertEqual(4, result["summary"]["behavior_removed_frame_count"])
        self.assertEqual(1, result["summary"]["behavior_removed_phase_count"])
        self.assertEqual(2, result["behavior_removals"][0]["segment_count"])

    def test_existing_invalid_decisions_are_preserved(self) -> None:
        payload = {
            "segments": [
                {"start_frame": 0, "end_frame": 4, "state": "valid", "reason": "keep", "confidence": 1.0},
                {"start_frame": 5, "end_frame": 5, "state": "invalid", "reason": "existing", "confidence": 0.9},
                {"start_frame": 6, "end_frame": 9, "state": "valid", "reason": "keep", "confidence": 1.0},
            ],
            "samples": [],
            "summary": {},
        }
        result = apply_behavior_phase_exclusion(payload, self.behavior, "place", 10, 10.0)

        self.assertTrue(any(item["start_frame"] == 5 and item["end_frame"] == 5 and item["reason"] == "existing" for item in result["segments"]))
        self.assertTrue(any(item["start_frame"] == 8 and item["end_frame"] == 9 and item["state"] == "invalid" for item in result["segments"]))

    def test_same_behavior_phase_cannot_be_added_twice(self) -> None:
        first = apply_behavior_phase_exclusion({"segments": [], "samples": [], "summary": {}}, self.behavior, "reach", 10, 10.0)
        with self.assertRaisesRegex(ValueError, "已经标记为去除"):
            apply_behavior_phase_exclusion(first, self.behavior, "reach", 10, 10.0)

    def test_unknown_phase_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "没有动作"):
            apply_behavior_phase_exclusion({"segments": []}, self.behavior, "grasp", 10, 10.0)


if __name__ == "__main__":
    unittest.main()
