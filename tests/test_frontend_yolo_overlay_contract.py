from __future__ import annotations

import unittest
from pathlib import Path


class FrontendYoloOverlayContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.html = (root / "index.html").read_text(encoding="utf-8")
        cls.app = (root / "static" / "app.js").read_text(encoding="utf-8")
        cls.css = (root / "static" / "sidecar.css").read_text(encoding="utf-8")
        cls.trim = (root / "app" / "no_action_trim.py").read_text(encoding="utf-8")

    def test_overlay_has_independent_button_and_canvas(self) -> None:
        self.assertIn('id="yoloOverlayButton"', self.html)
        self.assertIn('id="yoloOverlayCanvas"', self.html)
        self.assertIn('id="jointOverlayCanvas"', self.html)
        self.assertLess(self.html.index('id="yoloOverlayCanvas"'), self.html.index('id="jointOverlayCanvas"'))

    def test_overlay_reuses_saved_trim_evidence(self) -> None:
        self.assertIn('/no-action-trim`', self.app)
        self.assertIn("payload?.samples", self.app)
        self.assertIn("nearestYoloOverlaySample", self.app)
        self.assertNotIn("/yoloe-overlay/frame", self.app)

    def test_async_load_is_guarded_by_selection_and_token(self) -> None:
        self.assertIn("token === state.yoloOverlayLoadToken", self.app)
        self.assertIn("state.dataset?.id === datasetId", self.app)
        self.assertIn("state.episode?.id === episodeId", self.app)
        self.assertIn("(state.media?.file_id || null) === mediaFileId", self.app)

    def test_canvas_uses_contain_mapping_and_marks_sampled_frames(self) -> None:
        self.assertIn("Math.min(width / sourceWidth, height / sourceHeight)", self.app)
        self.assertIn("(width - sourceWidth * scale) / 2", self.app)
        self.assertIn("(height - sourceHeight * scale) / 2", self.app)
        self.assertIn("ctx.setLineDash(exact ? [] : [6, 4])", self.app)
        self.assertIn("pointer-events:none", self.css)

    def test_new_trim_reports_persist_prompt_and_media_geometry(self) -> None:
        self.assertIn('"detector_terms": detector_terms', self.trim)
        self.assertIn('"prompt_classes": _open_vocab_proximity_classes(detector_terms)', self.trim)
        self.assertIn('"width": media.get("width")', self.trim)
        self.assertIn('"height": media.get("height")', self.trim)


if __name__ == "__main__":
    unittest.main()
