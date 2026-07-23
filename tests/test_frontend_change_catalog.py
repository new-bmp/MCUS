from __future__ import annotations

import unittest
from pathlib import Path


class FrontendChangeCatalogContractTests(unittest.TestCase):
    """Keep the stale-report safety contract visible in the browser bundle."""

    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.app_source = (root / "static" / "app.js").read_text(encoding="utf-8")
        cls.css_source = (root / "static" / "sidecar.css").read_text(encoding="utf-8")

    def test_requires_rerun_reports_are_not_selectable(self) -> None:
        self.assertIn("const requiresRerun = isPending && item.kind === \"paper_curation\"", self.app_source)
        self.assertIn("const selectable = isPending && !requiresRerun", self.app_source)
        self.assertIn("${selectable ? \"checked\" : \"disabled\"}", self.app_source)
        self.assertIn("需重新运行", self.app_source)

    def test_apply_button_uses_runnable_pending_count(self) -> None:
        self.assertIn("runnable_pending_count", self.app_source)
        self.assertIn("$(\"#applyChangesButton\").disabled = runnable === 0", self.app_source)
        self.assertIn(".change-row.requires-rerun", self.css_source)


if __name__ == "__main__":
    unittest.main()
