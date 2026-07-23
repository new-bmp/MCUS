from __future__ import annotations

import re
import unittest
from pathlib import Path


class FrontendBehaviorTimelineContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.html = (root / "index.html").read_text(encoding="utf-8")
        cls.app = (root / "static" / "app.js").read_text(encoding="utf-8")
        cls.css = (root / "static" / "style.css").read_text(encoding="utf-8")

    def test_behavior_phases_are_the_primary_action_track(self) -> None:
        self.assertIn("function renderActionTrack()", self.app)
        self.assertIn("state.behavior?.segments", self.app)
        self.assertIn("if (behaviorSegments.length)", self.app)
        self.assertIn("readableBehaviorPhase(segment)", self.app)
        self.assertIn("behaviorPhaseTone(segment", self.app)
        self.assertIn("const behaviorPhaseTones", self.app)
        self.assertIn("behavior-phase-segment", self.app)

    def test_phase_widths_include_gaps_and_click_to_seek(self) -> None:
        self.assertIn("function timelineIntervalItems", self.app)
        self.assertIn("segment.end_frame - segment.start_frame + 1", self.app)
        self.assertIn("behavior-phase-gap", self.app)
        self.assertIn('data-frame="${segment.start_frame}"', self.app)
        self.assertIn('updateFrame(Number(button.dataset.frame))', self.app)
        self.assertIn('<i class="timeline-cursor"', self.app)

    def test_trim_state_remains_visible_as_a_secondary_track(self) -> None:
        self.assertIn('id="trimTrackGroup"', self.html)
        self.assertIn('id="trimTrack"', self.html)
        self.assertIn("function renderTrimTrack()", self.app)
        self.assertIn("behaviorSegments.length && trimSegments.length", self.app)
        self.assertIn("trim-state-segment", self.app)
        self.assertIn(".trim-track-group{display:contents}", self.css)

    def test_legacy_behavior_payload_still_has_a_readable_label(self) -> None:
        self.assertIn("segment?.phase || segment?.stage_label || segment?.stage || segment?.label", self.app)
        self.assertIn('if (!raw) return "未提供"', self.app)
        self.assertIn("payload.source_video?.file_id", self.app)

    def test_cache_version_was_updated(self) -> None:
        style = re.search(r'/static/style\.css\?v=studio-(\d+)', self.html)
        app = re.search(r'/static/app\.js\?v=studio-(\d+)', self.html)
        self.assertIsNotNone(style)
        self.assertIsNotNone(app)
        self.assertEqual(style.group(1), app.group(1))
        self.assertGreaterEqual(int(app.group(1)), 61)


if __name__ == "__main__":
    unittest.main()
