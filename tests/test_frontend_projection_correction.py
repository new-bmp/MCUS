from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FrontendProjectionCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = (ROOT / "index.html").read_text(encoding="utf-8")
        cls.app = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        cls.style = (ROOT / "static" / "style.css").read_text(encoding="utf-8")

    def test_real_correction_entry_and_dedicated_model_slot_exist(self) -> None:
        self.assertIn('id="projectionCorrectionButton"', self.html)
        self.assertIn('value="mediapipe">MediaPipe 21 点全帧手部归正', self.html)
        self.assertIn('value="alicepose">AlicePose 21 点全帧手部归正', self.html)
        self.assertIn('{ slot: "hand_pose", kind: "mediapipe"', self.app)
        self.assertIn('{ slot: "hand_pose", kind: "alicepose"', self.app)
        self.assertIn('openAnalysisScope("projection_correction"', self.app)

    def test_scope_sends_sampling_and_gap_parameters(self) -> None:
        self.assertIn('id="projectionSampleFps"', self.app)
        self.assertIn('id="projectionMaxGapSeconds"', self.app)
        self.assertIn('id="projectionAdjustmentRate"', self.app)
        self.assertIn('id="projectionAdjustmentMode"', self.app)
        self.assertIn('id="projectionWristPointSource"', self.app)
        self.assertIn('id="projectionHandPoseBackend"', self.app)
        self.assertIn('id="projectionAlicePosePath"', self.app)
        self.assertIn('id="projectionDynamicFields"', self.app)
        self.assertIn('max="30"', self.app)
        self.assertIn('value="15"', self.app)
        self.assertIn('sample_fps: trimNumber("projectionSampleFps", 15)', self.app)
        self.assertIn('adjustment_rate: trimNumber("projectionAdjustmentRate", 0.58, 100)', self.app)
        self.assertIn('adjustment_mode: $("#projectionAdjustmentMode")?.value || "uniform"', self.app)
        self.assertIn('wrist_point_source: $("#projectionWristPointSource")?.value || "egodex"', self.app)
        self.assertIn('hand_pose_backend: $("#projectionHandPoseBackend")?.value || "mediapipe"', self.app)
        self.assertIn('dynamic_low_confidence: trimNumber("projectionDynamicLowConfidence", 0.18, 100)', self.app)
        self.assertIn('动态模式按上方锚点用平滑曲线连续缩放', self.app)
        self.assertIn('模型模式仍受前臂长度、球面解和腕部角度限制', self.app)
        self.assertIn('max_gap_seconds: trimNumber("projectionMaxGapSeconds", 0.75)', self.app)
        self.assertIn('原始 HDF5 始终只读', self.app)

    def test_projection_controls_are_compact_and_switchable(self) -> None:
        self.assertIn('value="alicepose">AlicePose</option>', self.app)
        self.assertIn('updateProjectionCorrectionControlVisibility', self.app)
        self.assertIn('#projectionCorrectionSettings input,#projectionCorrectionSettings select', self.style)
        self.assertIn('font-size:9px', self.style)
        self.assertIn('body{font-size:10px}', self.style)
        self.assertIn('.modal-head strong{font-size:12px}', self.style)
        self.assertIn('padding:0 13px;font-size:10px', self.style)

    def test_completed_and_applied_changes_refresh_joint_overlay(self) -> None:
        self.assertIn('job.operation === "projection_correction"', self.app)
        self.assertIn('await updateJointOverlayStatus(); await updateFrame(state.frame);', self.app)
        self.assertIn('status.hand_pose', self.app)

    def test_review_window_exposes_raw_and_corrected_joint_overlays(self) -> None:
        self.assertIn('id="rawJointOverlayButton"', self.html)
        self.assertIn('id="correctedJointOverlayButton"', self.html)
        self.assertIn('toggleJointOverlay("raw")', self.app)
        self.assertIn('toggleJointOverlay("corrected")', self.app)
        self.assertIn('joint_mode=${encodeURIComponent(overlayMode)}', self.app)
        self.assertIn('joint_mode=${encodeURIComponent(jointMode)}', self.app)
        self.assertIn('state.correctedSourceFramePositions', self.app)
        self.assertIn('nearestCorrectedFrame', self.app)
        self.assertIn('对比无需先应用', self.html)
        self.assertIn('待确认，可直接对比归正', self.app)
        self.assertIn('correctedStatus.projection_applied', self.app)

    def test_projection_correction_can_force_a_new_revision(self) -> None:
        self.assertIn('不复用已有归正', self.app)
        self.assertIn('使用本次所选手部模型生成新修订', self.app)
        self.assertIn('["vlm_behavior", "projection_correction"].includes(state.analysisOperation)', self.app)


if __name__ == "__main__":
    unittest.main()
