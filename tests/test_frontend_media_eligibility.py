from __future__ import annotations

import unittest
from pathlib import Path


class FrontendMediaEligibilityContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.app = (root / "static" / "app.js").read_text(encoding="utf-8")
        cls.index = (root / "index.html").read_text(encoding="utf-8")

    def test_operation_media_filter_keeps_depth_out_of_rgb_tasks(self) -> None:
        self.assertIn("function mediaLooksLikeRgb(media)", self.app)
        self.assertIn('type === "raw_depth"', self.app)
        self.assertIn('media?.is_depth_map === true', self.app)
        self.assertIn('["analysis_eligible", "vlm_eligible", "smoothing_eligible"]', self.app)
        self.assertIn('eligibleMediaStreams(episode)', self.app)
        self.assertIn("const referenceMedia = eligibleMediaStreams(referenceEpisode).find", self.app)
        self.assertNotIn("const referenceMedia = (referenceEpisode?.media_streams || []).find", self.app)
        self.assertIn("深度流仍可在文件预览中查看", self.app)

    def test_full_keeps_curation_available_when_only_export_is_unsupported(self) -> None:
        self.assertIn('id="fullPipelineHint"', self.index)
        self.assertIn("function fullExportBlockReason", self.app)
        self.assertIn("function fullPipelineExportWarning", self.app)
        self.assertIn("capabilities.can_full_export !== false", self.app)
        self.assertNotIn("if (capabilities.can_full_export === false)", self.app)
        self.assertIn("fullButton.disabled = !hasEpisodes || Boolean(reason)", self.app)
        self.assertIn("清洗可运行 · 导出受限", self.app)
        self.assertIn("清洗与 VLM 标注仍可运行", self.app)
        self.assertGreaterEqual(self.app.count('operation === "full_pipeline"'), 2)
        self.assertIn("const blockedReason = analysisOperationBlockReason", self.app)

    def test_top_level_tasks_respect_format_capabilities(self) -> None:
        self.assertIn('["#curationPipelineButton", "paper_curation"]', self.app)
        self.assertIn('["#videoSmoothButton", "video_smoothing"]', self.app)
        self.assertIn('["#noActionTrimButton", "no_action_trim"]', self.app)
        self.assertIn('["#behaviorAnnotateButton", "vlm_behavior"]', self.app)
        self.assertIn('["#qwenTrimButton", "qwen_trim"]', self.app)
        self.assertIn('capabilityBlockReason("can_curation"', self.app)
        self.assertIn('capabilityBlockReason("can_video_smoothing"', self.app)
        self.assertIn('capabilityBlockReason("can_vlm"', self.app)
        self.assertNotIn('$("#behaviorAnnotateButton").disabled = !state.dataset?.episodes?.length', self.app)

    def test_dataset_family_strategy_controls_nexus_joint_overlay(self) -> None:
        self.assertIn('id="datasetStrategyBadge"', self.index)
        self.assertIn("function datasetProcessingStrategy", self.app)
        self.assertIn('can_pressure_analysis: "触觉 / 压力辅助分析"', self.app)
        self.assertIn('strategy.joint_overlay === false', self.app)
        self.assertIn('strategy.id === "nexus_sensor_fusion_v1"', self.app)
        self.assertIn("Nexus 使用触觉/压力与多传感器对齐，不提供 Joint 叠加", self.app)
        self.assertIn('capabilityBlockReason("can_pose_recovery"', self.app)

    def test_nexus_pressure_integrity_has_a_dedicated_timeline_stage(self) -> None:
        self.assertIn('const curationStageOrder = ["t0", "p1", "s1"', self.app)
        self.assertIn('t0: "统一时间轴"', self.app)
        self.assertIn('p1: "Nexus 压力完整性"', self.app)

    def test_time_sync_is_automatic_and_blocks_processing_until_ready(self) -> None:
        self.assertIn("startSensorAlignment(manifest.id)", self.app)
        self.assertIn("function timeSyncBlockReason()", self.app)
        self.assertIn("T0 时间同步尚未完成", self.app)
        self.assertIn("T0 时间同步失败，已阻止后续清洗、VLM、Action 与导出", self.app)

    def test_nexus_media_displays_source_and_sync_fps_separately(self) -> None:
        self.assertIn("function mediaFpsSummary(media, episode = state.episode)", self.app)
        self.assertIn('media?.source_fps || media?.storage_fps || media?.fps', self.app)
        self.assertIn('family === "nexus_multimodal"', self.app)
        self.assertIn("源视频 ${sourceLabel} · 同步处理 ${syncFps.toFixed(2)} FPS", self.app)


if __name__ == "__main__":
    unittest.main()
