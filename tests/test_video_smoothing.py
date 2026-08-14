from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from app import video_smoothing
from app.video_smoothing import (
    _analysis_dimensions,
    _configured_gpu_devices,
    _ffmpeg_encoder_args,
    _next_encoder_gpu,
    _next_processing_gpu,
    _target_timeline,
    smooth_video,
)


class VideoSmoothingTests(unittest.TestCase):
    def test_nvenc_devices_rotate_across_configured_gpus(self) -> None:
        with patch.dict(os.environ, {"ALICE_GPU_DEVICES": "0,2,7"}):
            video_smoothing._ENCODER_GPU_CURSOR = 0
            video_smoothing._PROCESSING_GPU_CURSOR = 0
            self.assertEqual(["0", "2", "7"], _configured_gpu_devices())
            self.assertEqual(["0", "2", "7", "0"], [_next_encoder_gpu() for _ in range(4)])
            self.assertEqual(["0", "2", "7", "0"], [_next_processing_gpu() for _ in range(4)])
            self.assertIn("7", _ffmpeg_encoder_args("h264_nvenc", "7"))

    def test_motion_analysis_uses_bounded_proxy_resolution(self) -> None:
        width, height, scale = _analysis_dimensions(1920, 1080, 640)

        self.assertEqual((640, 360), (width, height))
        self.assertAlmostEqual(1 / 3, scale, places=6)

    def test_target_timeline_removes_non_integer_decimation_cadence(self) -> None:
        source_fps = 50.053
        source_times = np.arange(50, dtype=np.float64) / source_fps

        target_times, source_positions = _target_timeline(source_times, source_fps, 30.0)

        self.assertTrue(np.allclose(np.diff(target_times), 1 / 30.0))
        self.assertTrue(np.any(np.abs(source_positions - np.rint(source_positions)) > 1e-6))
        self.assertAlmostEqual(source_fps / 30.0, float(np.mean(np.diff(source_positions))), places=6)

    def test_cpu_fallback_writes_matching_frame_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            output = root / "smoothed.mp4"
            manifest_path = root / "smoothed.alice"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64))
            checker = ((np.indices((64, 96)).sum(axis=0) % 2) * 180).astype(np.uint8)
            for index in range(12):
                frame = cv2.cvtColor(np.roll(checker, index, axis=1), cv2.COLOR_GRAY2BGR)
                writer.write(frame)
            writer.release()

            with (
                patch.dict(os.environ, {"ALICE_VIDEO_ACCELERATOR": "cpu", "ALICE_VIDEO_ENCODER": "opencv"}),
                patch("app.video_smoothing._artifact_paths", return_value=(output, manifest_path)),
                patch("app.video_smoothing.record_change", return_value={"id": "change", "status": "pending", "revision": 1}),
            ):
                result = smooth_video(
                    "dataset",
                    {"id": "episode", "name": "episode"},
                    {"path": str(source), "file_id": "video", "stream_name": "video", "relative_path": "source.mp4", "modality": "rgb", "smoothing_eligible": True},
                    lambda _value, _message: None,
                )

            capture = cv2.VideoCapture(str(output))
            try:
                self.assertTrue(capture.isOpened())
                self.assertEqual(12, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            finally:
                capture.release()
            self.assertEqual("cpu", result["summary"]["processing_backend"])
            self.assertEqual("opencv-mp4v", result["summary"]["encoder"])

    def test_cuda_runtime_failure_falls_back_and_updates_progress(self) -> None:
        class FailingCudaProcessor:
            gpu_device = "3"
            device_name = "A800 test device"

            @staticmethod
            def process(_frame, _matrix):
                raise RuntimeError("simulated CUDA failure")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            output = root / "smoothed.mp4"
            manifest_path = root / "smoothed.alice"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64))
            frame = np.full((64, 96, 3), 80, dtype=np.uint8)
            for _index in range(3):
                writer.write(frame)
            writer.release()
            messages: list[str] = []

            with (
                patch.dict(os.environ, {"ALICE_VIDEO_ENCODER": "opencv"}),
                patch("app.video_smoothing._artifact_paths", return_value=(output, manifest_path)),
                patch("app.video_smoothing._create_cuda_frame_processor", return_value=(FailingCudaProcessor(), None)),
                patch("app.video_smoothing.record_change", return_value={"id": "change", "status": "pending", "revision": 1}),
            ):
                result = smooth_video(
                    "dataset",
                    {"id": "episode", "name": "episode"},
                    {"path": str(source), "file_id": "video", "stream_name": "video", "relative_path": "source.mp4", "modality": "rgb", "smoothing_eligible": True},
                    lambda _value, message: messages.append(message),
                )

            self.assertEqual("cpu-after-cuda-fallback", result["summary"]["processing_backend"])
            self.assertEqual("CPU", result["summary"]["processing_device"])
            self.assertEqual("simulated CUDA failure", result["summary"]["cuda_fallback_error"])
            self.assertFalse(result["summary"]["hardware_accelerated"])
            self.assertTrue(any("CPU fallback" in message for message in messages))

    def test_high_rate_eis_outputs_uniform_30_fps_with_frame_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source-50fps.mp4"
            output = root / "eis-30fps.mp4"
            manifest_path = root / "eis-30fps.alice"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 50.0, (96, 64))
            checker = ((np.indices((64, 96)).sum(axis=0) % 8) * 28).astype(np.uint8)
            for index in range(50):
                frame = cv2.cvtColor(np.roll(checker, index // 2, axis=1), cv2.COLOR_GRAY2BGR)
                writer.write(frame)
            writer.release()

            with (
                patch.dict(os.environ, {"ALICE_VIDEO_ACCELERATOR": "cpu", "ALICE_VIDEO_ENCODER": "opencv"}),
                patch("app.video_smoothing._artifact_paths", return_value=(output, manifest_path)),
                patch("app.video_smoothing.record_change", return_value={"id": "change", "status": "pending", "revision": 1}),
            ):
                result = smooth_video(
                    "dataset",
                    {"id": "episode", "name": "episode"},
                    {"path": str(source), "file_id": "video", "stream_name": "video", "relative_path": source.name, "modality": "rgb", "smoothing_eligible": True},
                    lambda _value, _message: None,
                    mode="eis_30",
                    target_fps=30.0,
                    motion_compensation=True,
                )

            capture = cv2.VideoCapture(str(output))
            try:
                self.assertTrue(capture.isOpened())
                self.assertEqual(30, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
                self.assertAlmostEqual(30.0, float(capture.get(cv2.CAP_PROP_FPS)), places=2)
            finally:
                capture.release()
            self.assertEqual("eis_motion_compensated_30", result["summary"]["smoothing_mode"])
            self.assertEqual(50, result["summary"]["source_frame_count"])
            self.assertEqual(30, result["summary"]["frame_count"])
            self.assertEqual(30, len(result["source_frame_positions"]))
            audit_path = Path(result["frame_audit"]["artifact_path"])
            self.assertTrue(audit_path.is_file())
            with np.load(audit_path) as audit:
                self.assertEqual((50, 2, 3), audit["source_stabilization_matrices"].shape)
                self.assertEqual((30, 2, 3), audit["target_stabilization_matrices"].shape)
                self.assertEqual((30,), audit["source_frame_positions"].shape)
                self.assertTrue(audit["synthetic_mask"].any())

    def test_eis_30_falls_back_to_native_for_30_fps_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source-30fps.mp4"
            output = root / "native.mp4"
            manifest_path = root / "native.alice"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
            for index in range(6):
                writer.write(np.full((48, 64, 3), index * 20, dtype=np.uint8))
            writer.release()

            with (
                patch.dict(os.environ, {"ALICE_VIDEO_ACCELERATOR": "cpu", "ALICE_VIDEO_ENCODER": "opencv"}),
                patch("app.video_smoothing._artifact_paths", return_value=(output, manifest_path)),
                patch("app.video_smoothing.record_change", return_value={"id": "change", "status": "pending", "revision": 1}),
            ):
                result = smooth_video(
                    "dataset",
                    {"id": "episode", "name": "episode"},
                    {"path": str(source), "file_id": "video", "stream_name": "video", "relative_path": source.name, "modality": "rgb", "smoothing_eligible": True},
                    lambda _value, _message: None,
                    mode="eis_30",
                    target_fps=30.0,
                )

            self.assertEqual("native_fps", result["summary"]["smoothing_mode"])
            self.assertEqual("eis_30", result["summary"]["requested_mode"])
            self.assertIn("source_fps_not_above_target", result["summary"]["mode_fallback_reason"])


if __name__ == "__main__":
    unittest.main()
