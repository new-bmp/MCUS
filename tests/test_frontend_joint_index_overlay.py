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

    def test_native_overlay_drops_stale_requests_and_has_no_33fps_cap(self) -> None:
        self.assertIn("force && state.jointGeometryAbortController", self.app)
        self.assertNotIn("(force || changed) && state.jointGeometryAbortController", self.app)
        self.assertIn("Math.max(0, 4 - (Date.now() - state.jointGeometryLastAt))", self.app)
        self.assertNotIn("Math.max(0, 30 - (Date.now() - state.jointGeometryLastAt))", self.app)
        self.assertIn("clearJointOverlayCanvas();", self.app)

    def test_native_overlay_prefetches_the_next_sensor_tick(self) -> None:
        self.assertIn("state.jointGeometryCache = new Map()", self.app)
        self.assertIn("function nextJointPrefetchFrame()", self.app)
        self.assertIn("function prefetchJointGeometry(frame)", self.app)
        self.assertIn("queueNextJointGeometryPrefetch()", self.app)
        self.assertIn("state.jointGeometryCache.has(requestedFrame)", self.app)

    def test_native_overlay_follows_sensor_clock(self) -> None:
        self.assertIn("const DEFAULT_JOINT_SYNC_HZ = 30", self.app)
        self.assertIn("function jointClockFrameFromMediaTime(mediaTime)", self.app)
        self.assertIn("Math.round(seconds * hz)", self.app)
        self.assertIn("state.jointPresentedFrame = jointClock.frame", self.app)
        self.assertIn("geometry.clock_hz || geometry.physical_hz || geometry.sensor_hz", self.app)
        self.assertIn("jointClockFrameFromMediaTime(state.nativeMediaTime)", self.app)
        self.assertIn("geometryFrame !== state.jointPresentedFrame", self.app)
        self.assertNotIn("geometryFrame !== state.nativePresentedFrame", self.app)

    def test_manual_joint_delay_control_shifts_only_the_overlay(self) -> None:
        self.assertIn('id="jointFrameOffset"', self.html)
        self.assertIn('id="jointDelayMilliseconds"', self.html)
        self.assertIn("state.jointFrameOffset = 0", self.app)
        self.assertIn("function jointFrameWithOffset(frame)", self.app)
        self.assertIn("function setJointFrameOffset(value)", self.app)
        self.assertIn("&joint_frame_offset=${state.jointFrameOffset}", self.app)
        self.assertIn("视频帧 ${videoFrame} → 请求帧", self.app)


if __name__ == "__main__":
    unittest.main()
