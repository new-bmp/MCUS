from __future__ import annotations

import unittest
from pathlib import Path


class FrontendJointIndexOverlayContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.html = (root / "index.html").read_text(encoding="utf-8")
        cls.app = (root / "static" / "app.js").read_text(encoding="utf-8")

    def test_source_index_toggle_is_independent(self) -> None:
        self.assertIn('id="jointIndexButton"', self.html)
        self.assertIn('aria-pressed="false"', self.html)
        self.assertIn('$("#jointIndexButton").addEventListener("click", toggleJointIndices)', self.app)
        self.assertIn("state.jointIndices = !state.jointIndices", self.app)

    def test_canvas_renders_backend_source_index_without_compaction(self) -> None:
        self.assertIn("Number(point.source_index)", self.app)
        self.assertNotIn("String(points.indexOf(point))", self.app)
        self.assertIn("drawJointGeometry(state.jointGeometryCurrent)", self.app)

    def test_joint_centers_use_compact_points(self) -> None:
        self.assertIn("pointRadius = 2.5", self.app)
        self.assertIn("point.y * scale, pointRadius", self.app)

    def test_fallback_frame_request_preserves_index_setting(self) -> None:
        self.assertIn("&joint_indices=${state.jointIndices}", self.app)
        self.assertIn("resetJointIndices();", self.app)


if __name__ == "__main__":
    unittest.main()
