from __future__ import annotations

import unittest
from pathlib import Path


class FrontendBehaviorInspectorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.html = (root / "index.html").read_text(encoding="utf-8")
        cls.app = (root / "static" / "app.js").read_text(encoding="utf-8")
        cls.css = (root / "static" / "sidecar.css").read_text(encoding="utf-8")
        cls.main = (root / "app" / "main.py").read_text(encoding="utf-8")

    def test_behavior_annotation_lives_in_right_inspector(self) -> None:
        inspector_start = self.html.index('<aside class="inspector"')
        inspector_end = self.html.index("</aside>", inspector_start)
        inspector = self.html[inspector_start:inspector_end]
        self.assertIn('data-inspector="behavior"', inspector)
        self.assertIn('id="behaviorInspector"', inspector)
        self.assertIn('id="behaviorAnnotateButton"', inspector)
        self.assertIn('id="behaviorResult"', inspector)
        self.assertEqual(1, self.html.count('id="behaviorAnnotateButton"'))

    def test_selected_phase_calls_real_sidecar_endpoint(self) -> None:
        self.assertIn('id="behaviorRemoveSelect"', self.html)
        self.assertIn('id="behaviorRemoveButton"', self.html)
        self.assertIn("async function removeBehaviorPhase()", self.app)
        self.assertIn("/behavior-removals", self.app)
        self.assertIn('@app.post("/api/datasets/{dataset_id}/episodes/{episode_id}/behavior-removals")', self.main)

    def test_sidebar_uses_compact_phase_cards(self) -> None:
        self.assertIn("behavior-segment-card", self.app)
        self.assertIn(".behavior-segment-card", self.css)
        self.assertIn(".behavior-remove-card", self.css)

    def test_paper_word_is_absent_from_product_copy(self) -> None:
        self.assertNotIn("论文", self.html)
        self.assertNotIn("论文", self.app)
        self.assertIn("qualityDisplayText", self.app)
        self.assertIn('replaceAll("\\u8bba\\u6587\\u5f0f", "")', self.app)
        self.assertIn("qualityDisplayText(item.reason)", self.app)


if __name__ == "__main__":
    unittest.main()
