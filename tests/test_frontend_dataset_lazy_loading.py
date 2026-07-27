from __future__ import annotations

import unittest
from pathlib import Path


class FrontendDatasetLazyLoadingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.app = (root / "static" / "app.js").read_text(encoding="utf-8")
        cls.index = (root / "index.html").read_text(encoding="utf-8")

    def test_dataset_selector_is_persistent_in_explorer_header(self) -> None:
        self.assertIn('id="datasetSelect"', self.index)
        self.assertIn('$("#datasetSelect").addEventListener("change"', self.app)

    def test_collection_discovery_loads_only_selected_dataset(self) -> None:
        self.assertIn('data.mode === "collection"', self.app)
        self.assertIn('body: JSON.stringify({ path: item.path, name: item.name, analyze_schema: false })', self.app)
        self.assertIn('await loadCollectionDataset(data.datasets[0].key)', self.app)
        self.assertIn('不调用 Qwen', self.app)

    def test_loaded_dataset_cache_is_bounded(self) -> None:
        self.assertIn('while (state.datasetCache.size > 3)', self.app)

    def test_frontend_cache_version_is_updated(self) -> None:
        self.assertIn('app.js?v=studio-78', self.index)

    def test_full_completion_refreshes_vlm_panel_and_reports_request_counts(self) -> None:
        self.assertIn('await loadBehaviorAnnotation();', self.app)
        self.assertIn('VLM 请求 ${Number(job.result?.vlm_requested_count || 0)}', self.app)

    def test_full_robot_action_is_optional_and_submitted_with_selected_profile(self) -> None:
        self.assertIn('id="fullGenerateAction"', self.app)
        self.assertIn('id="fullRobotProfile"', self.app)
        self.assertIn('if (state.analysisOperation !== "full_pipeline" || !$("#fullGenerateAction")?.checked) return {};', self.app)
        self.assertIn('full_action_profile_id: profileId', self.app)


if __name__ == "__main__":
    unittest.main()
