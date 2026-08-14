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
        self.assertIn("function timelineIntervalStyle(startFrame, endFrame, frameCount)", self.app)
        self.assertIn("const endExclusive = Math.min(total, end + 1)", self.app)
        self.assertIn("(endExclusive - start) / total * 100", self.app)
        self.assertIn("behavior-phase-gap", self.app)
        self.assertIn('data-frame="${segment.start_frame}"', self.app)
        self.assertIn('updateFrame(Number(button.dataset.frame))', self.app)
        self.assertIn('<i class="timeline-cursor"', self.app)

    def test_behavior_unknown_and_quality_invalid_share_absolute_frame_coordinates(self) -> None:
        self.assertIn("function reviewTimelineFrameCount()", self.app)
        self.assertIn("state.behavior?.timeline?.frame_count", self.app)
        self.assertIn("state.behavior?.analysis_video?.frame_count", self.app)
        self.assertIn("function qualityMaskedBehaviorSegments", self.app)
        self.assertIn("state.curation?.pre_vlm_segments", self.app)
        self.assertIn("qualityMaskedBehaviorSegments(state.behavior?.segments, frameCount)", self.app)
        self.assertIn('const track = $("#segmentTrack"), frameCount = reviewTimelineFrameCount();', self.app)
        self.assertIn("timelineIntervalItems(curationTrackSegments(payload, state.curationStageFilter), frameCount)", self.app)
        self.assertIn("timeline-frame-segment behavior-phase-segment", self.app)
        self.assertIn("timeline-frame-segment curation-state-segment", self.app)
        self.assertIn(".timeline-frame-segment{position:absolute;top:0;bottom:0", self.css)
        self.assertIn(".timeline-frame-segment.behavior-phase-segment span{padding:0 5px}", self.css)
        self.assertIn(".timeline-frame-segment.behavior-phase-unknown{background:#626b76}", self.css)
        self.assertNotIn('style="flex:${', self.app)

    def test_trim_state_remains_visible_as_a_secondary_track(self) -> None:
        self.assertIn('id="trimTrackGroup"', self.html)
        self.assertIn('id="trimTrack"', self.html)
        self.assertIn("function renderTrimTrack()", self.app)
        self.assertIn("behaviorSegments.length && trimSegments.length", self.app)
        self.assertIn("trim-state-segment", self.app)
        self.assertIn(".trim-track-group{display:contents}", self.css)

    def test_timeline_is_contained_and_cursor_stays_inside_its_track(self) -> None:
        self.assertIn(".workspace>*,.workarea,.view,.review-scroll,#mediaReview{min-width:0}", self.css)
        self.assertIn(".timeline{width:100%;max-width:100%;min-width:0", self.css)
        self.assertIn(".segments{overflow:hidden}", self.css)
        self.assertIn(".timeline-cursor{top:2px;bottom:2px}", self.css)

    def test_legacy_behavior_payload_still_has_a_readable_label(self) -> None:
        self.assertIn("segment?.phase || segment?.stage_label || segment?.stage || segment?.label", self.app)
        self.assertIn('if (!raw) return "未提供"', self.app)
        self.assertIn("payload.source_video?.file_id", self.app)

    def test_vlm_sampling_defaults_are_high_rate_and_media_bound(self) -> None:
        self.assertIn('sample_count: 56', self.app)
        self.assertIn('vlm_sample_count: Math.round(trimNumber("curationVlmSamples", 56))', self.app)
        self.assertIn('id="curationVlmSamples"', self.app)
        self.assertIn('?media_file_id=${encodeURIComponent(mediaFileId)}', self.app)

    def test_full_review_loads_one_locked_run_bundle(self) -> None:
        self.assertIn("async function loadFullRunBundle", self.app)
        self.assertIn("/full-run?${query}", self.app)
        self.assertIn("payload.smoothing_video", self.app)
        self.assertIn("String(curation.full_run_id || \"\")", self.app)
        self.assertIn("String(curation.timeline_id || \"\")", self.app)
        self.assertIn("if (await loadFullRunBundle(episodeId, mediaFileId, selectionToken, requestedRunId)) return", self.app)
        self.assertIn("full_run_id=${encodeURIComponent(state.fullRunId)}", self.app)
        self.assertIn("拒绝拼接不同运行的工件", self.app)

    def test_cache_version_was_updated(self) -> None:
        style = re.search(r'/static/style\.css\?v=studio-(\d+)', self.html)
        app = re.search(r'/static/app\.js\?v=studio-(\d+)', self.html)
        self.assertIsNotNone(style)
        self.assertIsNotNone(app)
        self.assertEqual(style.group(1), app.group(1))
        self.assertGreaterEqual(int(app.group(1)), 85)


if __name__ == "__main__":
    unittest.main()
