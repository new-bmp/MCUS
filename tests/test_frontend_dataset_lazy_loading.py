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
        self.assertIn('body: JSON.stringify({ path: item.path, name: item.name, analyze_schema: false, camera_profile_id: report.selected_camera_profile_id || null })', self.app)
        self.assertIn('await loadCollectionDataset(data.datasets[0].key)', self.app)
        self.assertIn('不调用 Qwen', self.app)

    def test_loaded_dataset_cache_is_bounded(self) -> None:
        self.assertIn('while (state.datasetCache.size > 3)', self.app)

    def test_frontend_cache_version_is_updated(self) -> None:
        self.assertIn('app.js?v=studio-101', self.index)

    def test_full_completion_refreshes_vlm_panel_and_reports_request_counts(self) -> None:
        self.assertIn('await loadBehaviorAnnotation();', self.app)
        self.assertIn('VLM 请求 ${Number(job.result?.vlm_requested_count || 0)}', self.app)

    def test_curation_report_is_loaded_for_the_selected_media_stream(self) -> None:
        self.assertIn('?media_file_id=${encodeURIComponent(mediaFileId)}', self.app)
        self.assertIn('payload.source_video.file_id !== mediaFileId', self.app)
        self.assertIn('requested_media_file_ids: { ...mediaFileIds }', self.app)

    def test_curation_motion_samples_fill_the_motion_track(self) -> None:
        self.assertIn('function motionSeriesSamples()', self.app)
        self.assertIn('return annotationSamples.length ? annotationSamples : (Array.isArray(state.curation?.samples) ? state.curation.samples : []);', self.app)
        self.assertIn('state.curation = payload; state.curationNotice = null; state.curationStageFilter = null; renderMotionSeries();', self.app)

    def test_episode_selection_waits_for_behavior_and_curation_together(self) -> None:
        self.assertIn('const pipelineArtifactsLoad = loadReviewPipelineArtifacts(id, selectedMediaFileId, selectionToken);', self.app)
        self.assertIn('Promise.all([annotationLoad, pipelineArtifactsLoad, loadActionMappingResult(id, selectionToken), jointStatusLoad])', self.app)
        self.assertIn('Promise.all([loadBehaviorAnnotation(), loadCurationReport(episodeId, mediaFileId, selectionToken)])', self.app)
        behavior_render = self.app.split('function renderBehaviorAnnotation(payload) {', 1)[1].split('async function loadBehaviorAnnotation()', 1)[0]
        self.assertIn('refreshTimelineVisibility();', behavior_render)
        self.assertNotIn('motionSeries', behavior_render)

    def test_failed_full_shows_why_no_quality_report_was_created(self) -> None:
        self.assertIn('job.operation === "full_pipeline" ? "Full 失败" : "清洗失败"', self.app)
        self.assertIn('未生成质量报告：${failureMessage}', self.app)
        self.assertIn('state.curationNotice?.message || "当前流未运行质量检查"', self.app)

    def test_partial_full_keeps_curation_visible_and_reports_export_errors(self) -> None:
        self.assertIn('if (currentCurationFailure && !currentCurationItem)', self.app)
        self.assertIn('item?.full_status !== "partial" || !state.curation', self.app)
        self.assertIn('Full 部分完成：${detail}', self.app)
        self.assertIn('item.full_status === "partial"', self.app)

    def test_full_robot_action_is_optional_and_submitted_with_selected_profile(self) -> None:
        self.assertIn('id="fullOutputFormat"', self.app)
        self.assertIn('<option value="lerobot" selected>', self.app)
        self.assertIn('full_output_format: $("#fullOutputFormat")?.value || "lerobot"', self.app)
        self.assertIn('id="fullGenerateAction"', self.app)
        self.assertIn('id="fullRobotProfile"', self.app)
        self.assertIn('if (state.analysisOperation !== "full_pipeline" || !$("#fullGenerateAction")?.checked) return output;', self.app)
        self.assertIn('full_action_profile_id: profileId', self.app)

    def test_s1_isolated_spike_repair_defaults_to_five_frames(self) -> None:
        self.assertIn('id="curationRepairS1" checked', self.app)
        self.assertIn('id="curationS1MaxRepairFrames"', self.app)
        self.assertIn('repair_s1_spikes: $("#curationRepairS1")?.checked !== false', self.app)
        self.assertIn('s1_max_repair_frames: Math.round(trimNumber("curationS1MaxRepairFrames", 5))', self.app)


if __name__ == "__main__":
    unittest.main()
