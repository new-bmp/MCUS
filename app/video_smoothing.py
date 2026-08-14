from __future__ import annotations

import json
import os
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .preview_proxy import _ffmpeg_executable
from .full_run import full_run_stage_dir
from .storage import change_is_applied, dataset_artifact_dir, record_change, require_media_eligibility, slugify


VIDEO_SMOOTHING_SCHEMA = "alice/video-smoothing/v3"
DEFAULT_ANALYSIS_MAX_SIDE = 640
DEFAULT_FLOW_MAX_SIDE = 480
DEFAULT_HIGH_RATE_TARGET_FPS = 30.0
HIGH_RATE_MINIMUM_MARGIN_FPS = 0.25
_ENCODER_PROBE_CACHE: dict[tuple[str, str], bool] = {}
_ENCODER_PROBE_LOCK = threading.Lock()
_ENCODER_GPU_LOCK = threading.Lock()
_ENCODER_GPU_CURSOR = 0
_PROCESSING_GPU_LOCK = threading.Lock()
_PROCESSING_GPU_CURSOR = 0


def _artifact_paths(dataset_id: str, episode_id: str, stream_name: str, run_id: str | None = None) -> tuple[Path, Path]:
    root = (
        full_run_stage_dir(dataset_id, run_id, episode_id, "smoothing")
        if run_id
        else dataset_artifact_dir(dataset_id, "video-smoothing") / slugify(episode_id)
    )
    root.mkdir(parents=True, exist_ok=True)
    stem = slugify(Path(stream_name).stem or "video")
    return root / f"{stem}.smoothed.mp4", root / f"{stem}.smooth.alice"


def _write_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def preferred_smoothed_media(dataset_id: str, episode: dict, media: dict) -> tuple[dict, dict | None]:
    """Use a reviewed smoothing result while keeping the source media descriptor intact."""
    output_path, manifest_path = _artifact_paths(dataset_id, episode["id"], str(media.get("stream_name") or "video"))
    if not manifest_path.is_file() or not output_path.is_file():
        return media, None
    if not change_is_applied(dataset_id, "video_smoothing", episode["id"]):
        return media, None
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return media, None
    if document.get("source_video", {}).get("file_id") != media.get("file_id"):
        return media, None
    summary = document.get("summary") or {}
    source_positions = document.get("source_frame_positions") or []
    frame_audit_path = (document.get("frame_audit") or {}).get("artifact_path")
    smoothing_mode = str(summary.get("smoothing_mode") or "native_fps")
    return {
        **media,
        "path": str(output_path),
        "frame_count": int(summary.get("frame_count") or media.get("frame_count") or 0),
        "fps": float(summary.get("fps") or media.get("fps") or 30.0),
        "duration": float(summary.get("duration_seconds") or media.get("duration") or 0.0),
        "source_frame_count": int(summary.get("source_frame_count") or media.get("frame_count") or 0),
        "source_fps": float(summary.get("source_fps") or media.get("fps") or 0.0),
        "source_frame_positions": source_positions,
        "smoothing_mode": smoothing_mode,
        "video_smoothing_mode": smoothing_mode,
        "smoothing_artifact_path": str(manifest_path),
        "smoothing_frame_audit_path": frame_audit_path,
        "pixel_transform_available": bool((document.get("geometry_contract") or {}).get("pixel_transform_available")),
        "pixel_transform_artifact": frame_audit_path,
    }, document


def _as_array(value: Any) -> np.ndarray:
    return value.get() if isinstance(value, cv2.UMat) else np.asarray(value)


def _estimate_motion_detail(previous_gray: Any, current_gray: Any) -> tuple[float, float, float, bool, float]:
    points = cv2.goodFeaturesToTrack(previous_gray, maxCorners=240, qualityLevel=0.01, minDistance=18, blockSize=3)
    if points is None:
        return 0.0, 0.0, 0.0, False, 0.0
    point_values = _as_array(points)
    if len(point_values) < 8:
        return 0.0, 0.0, 0.0, False, 0.0
    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, points, None)
    if moved is None or status is None:
        return 0.0, 0.0, 0.0, False, 0.0
    valid = _as_array(status).reshape(-1).astype(bool)
    source = point_values.reshape(-1, 2)[valid]
    target = _as_array(moved).reshape(-1, 2)[valid]
    if len(source) < 8:
        return 0.0, 0.0, 0.0, False, 0.0
    matrix, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if matrix is None:
        return 0.0, 0.0, 0.0, False, 0.0
    dx = float(matrix[0, 2])
    dy = float(matrix[1, 2])
    angle = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    inlier_ratio = float(np.asarray(inliers).reshape(-1).mean()) if inliers is not None and len(inliers) else 0.0
    tracked_strength = min(1.0, len(source) / 48.0)
    confidence = max(0.0, min(1.0, inlier_ratio * tracked_strength))
    return dx, dy, angle, True, confidence


def _estimate_motion(previous_gray: Any, current_gray: Any) -> tuple[float, float, float, bool]:
    dx, dy, angle, valid, _confidence = _estimate_motion_detail(previous_gray, current_gray)
    return dx, dy, angle, valid


def _enhance(frame: Any, strength: float) -> Any:
    edge_strength = max(0.0, float(strength)) * 0.22
    kernel = np.array([
        [0.0, -edge_strength, 0.0],
        [-edge_strength, 1.0 + 4.0 * edge_strength, -edge_strength],
        [0.0, -edge_strength, 0.0],
    ], dtype=np.float32)
    return cv2.filter2D(frame, -1, kernel)


def _analysis_dimensions(width: int, height: int, max_side: int) -> tuple[int, int, float]:
    limit = max(160, int(max_side))
    scale = min(1.0, limit / max(width, height))
    if scale >= 0.999:
        return width, height, 1.0
    analysis_width = max(2, int(round(width * scale)))
    analysis_height = max(2, int(round(height * scale)))
    return analysis_width, analysis_height, min(analysis_width / width, analysis_height / height)


def _zero_phase_smooth(values: np.ndarray, smoothing: float) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 2 or source.shape[0] <= 1:
        return source.copy()
    weight = max(0.0, min(0.9999, float(smoothing)))
    forward = source.copy()
    backward = source.copy()
    for index in range(1, len(source)):
        forward[index] = weight * forward[index - 1] + (1.0 - weight) * source[index]
    for index in range(len(source) - 2, -1, -1):
        backward[index] = weight * backward[index + 1] + (1.0 - weight) * source[index]
    return (forward + backward) * 0.5


def _stabilization_matrices(
    motion: np.ndarray,
    width: int,
    height: int,
    smoothing: float,
    border_zoom: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    transforms = np.asarray(motion, dtype=np.float64)
    trajectory = np.cumsum(transforms, axis=0)
    target = _zero_phase_smooth(trajectory, smoothing)
    correction = target - trajectory
    translation_limit_x = max(2.0, width * 0.12)
    translation_limit_y = max(2.0, height * 0.12)
    rotation_limit = np.deg2rad(12.0)
    unclipped = correction.copy()
    correction[:, 0] = np.clip(correction[:, 0], -translation_limit_x, translation_limit_x)
    correction[:, 1] = np.clip(correction[:, 1], -translation_limit_y, translation_limit_y)
    correction[:, 2] = np.clip(correction[:, 2], -rotation_limit, rotation_limit)
    clipped = int(np.count_nonzero(np.any(np.abs(correction - unclipped) > 1e-9, axis=1)))
    zoom = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 0.0, border_zoom)
    zoom_h = np.vstack((zoom, np.asarray([0.0, 0.0, 1.0], dtype=np.float64)))
    matrices = np.empty((len(correction), 2, 3), dtype=np.float32)
    for index, (dx, dy, angle) in enumerate(correction):
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), np.degrees(angle), 1.0)
        matrix[0, 2] += dx
        matrix[1, 2] += dy
        homogeneous = np.vstack((matrix, np.asarray([0.0, 0.0, 1.0], dtype=np.float64)))
        matrices[index] = (zoom_h @ homogeneous)[:2].astype(np.float32)
    return matrices, correction, clipped


def target_stabilization_matrices(media: dict, frame_count: int | None = None) -> np.ndarray | None:
    """Load the per-output-frame source-pixel to stabilized-pixel transforms."""

    explicit = media.get("target_stabilization_matrices")
    if explicit is not None:
        matrices = np.asarray(explicit, dtype=np.float64)
    else:
        audit_value = media.get("smoothing_frame_audit_path") or media.get("frame_audit_path") or media.get("pixel_transform_artifact")
        audit_path = Path(str(audit_value or "")).expanduser()
        if not audit_path.is_file():
            return None
        try:
            with np.load(audit_path, allow_pickle=False) as audit:
                if "target_stabilization_matrices" in audit.files:
                    matrices = np.asarray(audit["target_stabilization_matrices"], dtype=np.float64)
                else:
                    source = np.asarray(audit["source_stabilization_matrices"], dtype=np.float64)
                    left = np.asarray(audit["target_left_frames"], dtype=np.int64).reshape(-1)
                    right = np.asarray(audit["target_right_frames"], dtype=np.int64).reshape(-1)
                    alpha = np.asarray(audit["interpolation_alpha"], dtype=np.float64).reshape(-1)
                    matrices = source[left] * (1.0 - alpha[:, None, None]) + source[right] * alpha[:, None, None]
        except (OSError, KeyError, ValueError):
            return None
    if matrices.ndim != 3 or tuple(matrices.shape[1:]) not in {(2, 3), (3, 3)}:
        return None
    if frame_count is not None and int(frame_count) != len(matrices):
        return None
    if tuple(matrices.shape[1:]) == (2, 3):
        homogeneous = np.repeat(np.eye(3, dtype=np.float64)[None], len(matrices), axis=0)
        homogeneous[:, :2] = matrices
        matrices = homogeneous
    return matrices if np.isfinite(matrices).all() else None


def _normalized_source_timestamps(raw_seconds: list[float], frame_count: int, fps: float) -> tuple[np.ndarray, str]:
    fallback = np.arange(frame_count, dtype=np.float64) / max(0.01, float(fps))
    if len(raw_seconds) != frame_count or frame_count < 2:
        return fallback, "decoded_frame_index/fps"
    values = np.asarray(raw_seconds, dtype=np.float64)
    values -= values[0] if np.isfinite(values[0]) else 0.0
    differences = np.diff(values)
    positive = differences[np.isfinite(differences) & (differences > 1e-6)]
    expected = 1.0 / max(0.01, float(fps))
    if (
        np.isfinite(values).all()
        and positive.size >= max(1, int(round((frame_count - 1) * 0.9)))
        and 0.25 * expected <= float(np.median(positive)) <= 4.0 * expected
    ):
        return values, "opencv_pos_msec"
    return fallback, "decoded_frame_index/fps"


def _target_timeline(source_times: np.ndarray, source_fps: float, target_fps: float) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(source_times, dtype=np.float64).reshape(-1)
    if times.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    period = float(np.median(np.diff(times))) if times.size > 1 else 1.0 / max(0.01, source_fps)
    if not np.isfinite(period) or period <= 0.0:
        period = 1.0 / max(0.01, source_fps)
    duration = max(period, float(times[-1]) + period)
    output_count = max(1, int(np.ceil(duration * target_fps - 1e-9)))
    target_times = np.arange(output_count, dtype=np.float64) / max(0.01, target_fps)
    target_times = target_times[target_times < duration - 1e-9]
    if target_times.size == 0:
        target_times = np.asarray([0.0], dtype=np.float64)
    source_positions = np.interp(target_times, times, np.arange(times.size, dtype=np.float64))
    return target_times, np.clip(source_positions, 0.0, max(0.0, times.size - 1.0))


def _flow_dimensions(width: int, height: int, max_side: int = DEFAULT_FLOW_MAX_SIDE) -> tuple[int, int, float, float]:
    flow_width, flow_height, _ = _analysis_dimensions(width, height, max_side)
    return flow_width, flow_height, width / flow_width, height / flow_height


def _dense_flow(previous: np.ndarray, current: np.ndarray, max_side: int = DEFAULT_FLOW_MAX_SIDE) -> tuple[np.ndarray, np.ndarray, float]:
    height, width = previous.shape[:2]
    flow_width, flow_height, scale_x, scale_y = _flow_dimensions(width, height, max_side)
    previous_small = cv2.resize(previous, (flow_width, flow_height), interpolation=cv2.INTER_AREA)
    current_small = cv2.resize(current, (flow_width, flow_height), interpolation=cv2.INTER_AREA)
    previous_gray = cv2.cvtColor(previous_small, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_small, cv2.COLOR_BGR2GRAY)
    forward = cv2.calcOpticalFlowFarneback(previous_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.1, 0)
    backward = cv2.calcOpticalFlowFarneback(current_gray, previous_gray, None, 0.5, 3, 15, 3, 5, 1.1, 0)
    grid_x, grid_y = np.meshgrid(
        np.arange(flow_width, dtype=np.float32),
        np.arange(flow_height, dtype=np.float32),
    )
    sampled_backward = cv2.remap(
        backward,
        grid_x + forward[..., 0],
        grid_y + forward[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    consistency = np.linalg.norm(forward + sampled_backward, axis=2)
    finite = consistency[np.isfinite(consistency)]
    median_error = float(np.median(finite)) if finite.size else float("inf")
    confidence = float(np.exp(-median_error / 1.5)) if np.isfinite(median_error) else 0.0
    if flow_width != width or flow_height != height:
        forward = cv2.resize(forward, (width, height), interpolation=cv2.INTER_LINEAR)
        backward = cv2.resize(backward, (width, height), interpolation=cv2.INTER_LINEAR)
        forward[..., 0] *= scale_x
        forward[..., 1] *= scale_y
        backward[..., 0] *= scale_x
        backward[..., 1] *= scale_y
    return forward.astype(np.float32), backward.astype(np.float32), max(0.0, min(1.0, confidence))


def _motion_compensated_frame(
    previous: np.ndarray,
    current: np.ndarray,
    alpha: float,
    *,
    minimum_confidence: float = 0.35,
) -> tuple[np.ndarray, float, str | None]:
    blend = max(0.0, min(1.0, float(alpha)))
    if blend <= 1e-6:
        return previous, 1.0, None
    if blend >= 1.0 - 1e-6:
        return current, 1.0, None
    if previous.shape[0] * previous.shape[1] > 3_200_000:
        fallback = previous if blend < 0.5 else current
        return fallback, 0.0, "flow_resolution_limit"
    try:
        forward, backward, confidence = _dense_flow(previous, current)
    except cv2.error as exc:
        fallback = previous if blend < 0.5 else current
        return fallback, 0.0, f"dense_flow_error:{str(exc)[:120]}"
    if confidence < minimum_confidence:
        fallback = previous if blend < 0.5 else current
        return fallback, confidence, "low_flow_confidence"
    height, width = previous.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    previous_warped = cv2.remap(
        previous,
        grid_x - forward[..., 0] * blend,
        grid_y - forward[..., 1] * blend,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    current_warped = cv2.remap(
        current,
        grid_x - backward[..., 0] * (1.0 - blend),
        grid_y - backward[..., 1] * (1.0 - blend),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    output = cv2.addWeighted(previous_warped, 1.0 - blend, current_warped, blend, 0.0)
    return output, confidence, None


def _write_motion_audit(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **arrays)
    temporary.replace(path)


def _opencl_backend() -> tuple[str, str]:
    requested = os.environ.get("ALICE_VIDEO_ACCELERATOR", "auto").strip().casefold()
    if requested in {"cpu", "off", "none"}:
        return "cpu", "CPU"
    try:
        if not cv2.ocl.haveOpenCL():
            return "cpu", "CPU"
        cv2.ocl.setUseOpenCL(True)
        if not cv2.ocl.useOpenCL():
            return "cpu", "CPU"
        device = cv2.ocl.Device_getDefault()
        mode = "opencl-full" if requested in {"opencl", "gpu", "full"} else "opencl-motion"
        return mode, str(device.name() or "OpenCL device")
    except (AttributeError, cv2.error):
        return "cpu", "CPU"


def _configured_gpu_devices() -> list[str]:
    return [
        item.strip()
        for item in os.environ.get("ALICE_GPU_DEVICES", "").split(",")
        if item.strip().isdigit()
    ]


def _next_encoder_gpu() -> str | None:
    global _ENCODER_GPU_CURSOR
    devices = _configured_gpu_devices()
    if not devices:
        return None
    with _ENCODER_GPU_LOCK:
        device = devices[_ENCODER_GPU_CURSOR % len(devices)]
        _ENCODER_GPU_CURSOR += 1
    return device


def _next_processing_gpu() -> str | None:
    global _PROCESSING_GPU_CURSOR
    devices = _configured_gpu_devices()
    if not devices:
        return None
    with _PROCESSING_GPU_LOCK:
        device = devices[_PROCESSING_GPU_CURSOR % len(devices)]
        _PROCESSING_GPU_CURSOR += 1
    return device


class _CudaFrameProcessor:
    def __init__(self, gpu_device: str, width: int, height: int, sharpen_strength: float) -> None:
        import torch
        import torch.nn.functional as functional

        self._torch = torch
        self._functional = functional
        self._device = torch.device(f"cuda:{gpu_device}")
        if not torch.cuda.is_available() or self._device.index is None or self._device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {gpu_device} is unavailable")
        self.gpu_device = gpu_device
        self.device_name = f"{torch.cuda.get_device_name(self._device)} (cuda:{gpu_device})"
        self.width = width
        self.height = height
        y, x = torch.meshgrid(
            torch.arange(height, device=self._device, dtype=torch.float32),
            torch.arange(width, device=self._device, dtype=torch.float32),
            indexing="ij",
        )
        self._destination_pixels = torch.stack((x, y, torch.ones_like(x)), dim=0).reshape(3, -1)
        edge = max(0.0, float(sharpen_strength)) * 0.22
        kernel = torch.tensor(
            [[0.0, -edge, 0.0], [-edge, 1.0 + 4.0 * edge, -edge], [0.0, -edge, 0.0]],
            device=self._device,
            dtype=torch.float32,
        )
        self._kernel = kernel.reshape(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self._sharpen = edge > 0.0

    def process(self, frame: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        torch = self._torch
        functional = self._functional
        with torch.inference_mode():
            homogeneous = np.vstack((np.asarray(matrix, dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32)))
            transform = torch.as_tensor(homogeneous, device=self._device)
            source = torch.linalg.inv(transform)[:2] @ self._destination_pixels
            x = source[0]
            y = source[1]
            x = 2.0 * x / max(1, self.width - 1) - 1.0
            y = 2.0 * y / max(1, self.height - 1) - 1.0
            grid = torch.stack((x, y), dim=-1).reshape(1, self.height, self.width, 2)
            image = torch.from_numpy(np.ascontiguousarray(frame)).to(self._device, dtype=torch.float32)
            image = image.permute(2, 0, 1).unsqueeze(0) / 255.0
            output = functional.grid_sample(
                image,
                grid,
                mode="bilinear",
                padding_mode="reflection",
                align_corners=True,
            )
            if self._sharpen:
                output = functional.conv2d(functional.pad(output, (1, 1, 1, 1), mode="reflect"), self._kernel, groups=3)
            output = output.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
            return output.squeeze(0).permute(1, 2, 0).cpu().numpy()


def _create_cuda_frame_processor(width: int, height: int, sharpen_strength: float) -> tuple[_CudaFrameProcessor | None, str | None]:
    requested = os.environ.get("ALICE_VIDEO_ACCELERATOR", "auto").strip().casefold()
    if requested not in {"auto", "cuda", "gpu", "full"}:
        return None, None
    gpu_device = _next_processing_gpu()
    if gpu_device is None:
        return None, None
    try:
        return _CudaFrameProcessor(gpu_device, width, height, sharpen_strength), None
    except (ImportError, OSError, RuntimeError) as exc:
        return None, str(exc)[:240]


def _ffmpeg_encoder_args(name: str, gpu_device: str | None = None) -> list[str]:
    if name == "h264_nvenc":
        gpu_args = ["-gpu", gpu_device] if gpu_device is not None else []
        return ["-c:v", name, *gpu_args, "-preset", "p3", "-tune", "hq", "-rc", "vbr", "-cq", "22", "-b:v", "0"]
    if name == "h264_qsv":
        return ["-c:v", name, "-preset", "veryfast", "-global_quality", "24"]
    return []


def _encoder_available(ffmpeg: str, name: str) -> bool:
    key = (ffmpeg, name)
    with _ENCODER_PROBE_LOCK:
        cached = _ENCODER_PROBE_CACHE.get(key)
    if cached is not None:
        return cached
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:r=1",
        "-frames:v", "1", *_ffmpeg_encoder_args(name), "-f", "null", "-",
    ]
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False)
        available = result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        available = False
    with _ENCODER_PROBE_LOCK:
        _ENCODER_PROBE_CACHE[key] = available
    return available


class _OpenCVFrameWriter:
    name = "opencv-mp4v"
    hardware_accelerated = False

    def __init__(self, path: Path, fps: float, width: int, height: int) -> None:
        self._writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not self._writer.isOpened():
            self._writer.release()
            raise RuntimeError("无法创建 .alicePD 平滑视频")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def abort(self) -> None:
        self._writer.release()


class _FFmpegFrameWriter:
    hardware_accelerated = True

    def __init__(self, ffmpeg: str, name: str, path: Path, fps: float, width: int, height: int, gpu_device: str | None = None) -> None:
        self.name = name
        self.gpu_device = gpu_device
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{width}x{height}",
            "-framerate", f"{fps:.8f}", "-i", "pipe:0", "-an", *_ffmpeg_encoder_args(name, gpu_device),
            "-pix_fmt", "yuv420p", "-g", str(max(1, round(fps))), "-movflags", "+faststart", str(path),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if self._process.stdin is None:
            self.abort()
            raise RuntimeError(f"无法启动 {name} 视频编码器")

    def write(self, frame: np.ndarray) -> None:
        if self._process.stdin is None or self._process.poll() is not None:
            raise RuntimeError(f"{self.name} 视频编码器意外退出")
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"{self.name} 视频编码失败") from exc

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
            self._process.stdin = None
        detail = self._process.stderr.read().decode("utf-8", errors="replace") if self._process.stderr is not None else ""
        return_code = self._process.wait()
        if return_code != 0:
            message = detail.strip().splitlines()[-1] if detail.strip() else f"exit {return_code}"
            raise RuntimeError(f"{self.name} 视频编码失败: {message}")

    def abort(self) -> None:
        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except OSError:
                pass
            self._process.stdin = None
        if self._process.poll() is None:
            self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()


def _create_frame_writer(path: Path, fps: float, width: int, height: int, gpu_device: str | None = None):
    requested = os.environ.get("ALICE_VIDEO_ENCODER", "auto").strip().casefold()
    if width % 2 == 0 and height % 2 == 0 and requested not in {"opencv", "mp4v", "cpu", "off"}:
        ffmpeg = _ffmpeg_executable()
        if ffmpeg:
            candidates = [requested] if requested not in {"", "auto"} else ["h264_nvenc", "h264_qsv"]
            for name in candidates:
                if name in {"h264_nvenc", "h264_qsv"} and _encoder_available(ffmpeg, name):
                    selected_gpu = (gpu_device or _next_encoder_gpu()) if name == "h264_nvenc" else None
                    return _FFmpegFrameWriter(ffmpeg, name, path, fps, width, height, selected_gpu)
    return _OpenCVFrameWriter(path, fps, width, height)


def _smooth_video_native(
    dataset_id: str,
    episode: dict,
    media: dict,
    progress,
    smoothing: float = 0.9,
    sharpen_strength: float = 0.32,
    border_zoom: float = 1.025,
    run_id: str | None = None,
    requested_mode: str = "native",
    fallback_reason: str | None = None,
) -> dict:
    require_media_eligibility(media, "video_smoothing")
    source_path = Path(str(media.get("path") or ""))
    if not source_path.is_file():
        raise ValueError(f"视频不存在: {media.get('relative_path') or source_path}")

    output_path, manifest_path = _artifact_paths(dataset_id, episode["id"], str(media.get("stream_name") or source_path.name), run_id)
    temporary_video = output_path.with_name(output_path.stem + ".part.mp4")
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法解码视频: {media.get('relative_path') or source_path.name}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or media.get("fps") or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or media.get("width") or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or media.get("height") or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or media.get("frame_count") or 0)
    if width <= 0 or height <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError("视频元数据无效，无法执行平滑")

    try:
        analysis_max_side = int(os.environ.get("ALICE_VIDEO_ANALYSIS_MAX_SIDE", DEFAULT_ANALYSIS_MAX_SIDE))
    except ValueError:
        analysis_max_side = DEFAULT_ANALYSIS_MAX_SIDE
    analysis_width, analysis_height, analysis_scale = _analysis_dimensions(width, height, analysis_max_side)
    cuda_processor, cuda_fallback_error = _create_cuda_frame_processor(width, height, sharpen_strength)
    if cuda_processor is not None:
        accelerator_mode, processing_device = "cuda-frame", cuda_processor.device_name
    else:
        accelerator_mode, processing_device = _opencl_backend()
    use_opencl_motion = accelerator_mode in {"opencl-motion", "opencl-full"}
    use_opencl_full = accelerator_mode == "opencl-full"
    temporary_video.unlink(missing_ok=True)
    try:
        writer = _create_frame_writer(
            temporary_video,
            fps,
            width,
            height,
            cuda_processor.gpu_device if cuda_processor is not None else None,
        )
    except Exception:
        capture.release()
        temporary_video.unlink(missing_ok=True)
        raise
    writer_name = writer.name
    writer_gpu = getattr(writer, "gpu_device", None)
    writer_label = f"{writer_name}:gpu{writer_gpu}" if writer_gpu is not None else writer_name
    writer_hardware_accelerated = writer.hardware_accelerated

    trajectory = np.zeros(3, dtype=np.float64)
    smoothed_trajectory = np.zeros(3, dtype=np.float64)
    raw_motion: list[float] = []
    correction_motion: list[float] = []
    failed_estimates = 0
    processed = 0
    previous_gray: Any | None = None
    zoom_matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 0.0, border_zoom)
    zoom_homogeneous = np.vstack((zoom_matrix, np.array([0.0, 0.0, 1.0], dtype=np.float64)))
    processing_backend = accelerator_mode
    acceleration_label = f"{processing_device} ({accelerator_mode}) + {writer_label}" if cuda_processor is not None or use_opencl_motion or writer_hardware_accelerated else "CPU"
    processing_failed = True
    try:
        progress(2, f"打开视频 {media.get('stream_name') or source_path.name} · {acceleration_label}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            working_frame: Any = cv2.UMat(frame) if use_opencl_full else frame
            if use_opencl_full:
                current_gray = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
                if analysis_width != width or analysis_height != height:
                    current_gray = cv2.resize(current_gray, (analysis_width, analysis_height), interpolation=cv2.INTER_AREA)
            else:
                motion_frame = frame
                if analysis_width != width or analysis_height != height:
                    motion_frame = cv2.resize(frame, (analysis_width, analysis_height), interpolation=cv2.INTER_AREA)
                current_gray = cv2.cvtColor(motion_frame, cv2.COLOR_BGR2GRAY)
                if use_opencl_motion:
                    current_gray = cv2.UMat(current_gray)
            if previous_gray is None:
                adjusted = np.zeros(3, dtype=np.float64)
            else:
                dx, dy, angle, valid = _estimate_motion(previous_gray, current_gray)
                if not valid:
                    failed_estimates += 1
                if analysis_scale < 1.0:
                    dx /= analysis_scale
                    dy /= analysis_scale
                transform = np.array([dx, dy, angle], dtype=np.float64)
                trajectory += transform
                smoothed_trajectory = smoothing * smoothed_trajectory + (1.0 - smoothing) * trajectory
                correction = smoothed_trajectory - trajectory
                adjusted = transform + correction
                raw_motion.append(float(np.linalg.norm(transform[:2])))
                correction_motion.append(float(np.linalg.norm(correction[:2])))
            cosine, sine = float(np.cos(adjusted[2])), float(np.sin(adjusted[2]))
            stabilization = np.array(
                [[cosine, -sine, adjusted[0]], [sine, cosine, adjusted[1]], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            combined_matrix = (zoom_homogeneous @ stabilization)[:2].astype(np.float32)
            if cuda_processor is not None:
                try:
                    output_frame = cuda_processor.process(frame, combined_matrix)
                except RuntimeError as exc:
                    cuda_fallback_error = str(exc)[:240]
                    cuda_processor = None
                    processing_backend = "cpu-after-cuda-fallback"
                    processing_device = "CPU"
                    acceleration_label = f"CPU fallback + {writer_label}" if writer_hardware_accelerated else "CPU fallback"
                    stabilized = cv2.warpAffine(
                        frame,
                        combined_matrix,
                        (width, height),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT,
                    )
                    output_frame = _as_array(_enhance(stabilized, sharpen_strength))
            else:
                stabilized = cv2.warpAffine(
                    working_frame,
                    combined_matrix,
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )
                output_frame = _as_array(_enhance(stabilized, sharpen_strength))
            writer.write(output_frame)
            previous_gray = current_gray
            processed += 1
            if processed == 1 or processed % 10 == 0 or processed == frame_count:
                progress(5 + 88 * processed / max(1, frame_count), f"稳像与清晰度增强 {processed}/{frame_count} · {acceleration_label}")
        writer.close()
        processing_failed = False
    finally:
        capture.release()
        if processing_failed:
            writer.abort()
            temporary_video.unlink(missing_ok=True)

    if processed == 0 or not temporary_video.is_file():
        temporary_video.unlink(missing_ok=True)
        raise RuntimeError("没有成功写出任何视频帧")
    temporary_video.replace(output_path)

    summary = {
        "frame_count": processed,
        "fps": round(fps, 4),
        "source_frame_count": frame_count,
        "source_fps": round(fps, 4),
        "duration_seconds": round(processed / max(0.01, fps), 6),
        "smoothing_mode": "native_fps",
        "requested_mode": requested_mode,
        "mode_fallback_reason": fallback_reason,
        "width": width,
        "height": height,
        "failed_motion_estimates": failed_estimates,
        "mean_camera_motion_px": round(float(np.mean(raw_motion)) if raw_motion else 0.0, 4),
        "mean_stabilization_correction_px": round(float(np.mean(correction_motion)) if correction_motion else 0.0, 4),
        "processing_backend": processing_backend,
        "processing_device": processing_device,
        "encoder": writer_name,
        "encoder_gpu": writer_gpu,
        "cuda_fallback_error": cuda_fallback_error,
        "hardware_accelerated": bool(processing_backend == "cuda-frame" or use_opencl_motion or writer_hardware_accelerated),
        "motion_analysis_width": analysis_width,
        "motion_analysis_height": analysis_height,
    }
    document = {
        "schema": VIDEO_SMOOTHING_SCHEMA,
        "dataset_id": dataset_id,
        "episode_id": episode["id"],
        "full_run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_policy": "Source video remains read-only; the enhanced video is stored only in .alicePD.",
        "source_video": {
            "file_id": media.get("file_id"),
            "stream_name": media.get("stream_name"),
            "relative_path": media.get("relative_path"),
        },
        "output_video": str(output_path),
        "method": "proxy_optical_flow_stabilization_plus_fast_sharpen",
        "limitations": "Stabilization and sharpening can reduce shake and perceived blur, but cannot reconstruct detail lost during exposure.",
        "config": {
            "mode": "native_fps",
            "requested_mode": requested_mode,
            "trajectory_smoothing": smoothing,
            "sharpen_strength": sharpen_strength,
            "border_zoom": border_zoom,
            "motion_analysis_max_side": analysis_max_side,
            "audio_preserved": False,
        },
        "summary": summary,
    }
    _write_atomic(manifest_path, document)
    change = record_change(
        dataset_id,
        "video_smoothing",
        episode["id"],
        f"Video smoothing: {episode['name']} / {media.get('stream_name') or source_path.name}",
        [manifest_path, output_path],
        {**summary, "stream_name": media.get("stream_name")},
        [str(media.get("relative_path") or "")],
    )
    document["artifact_path"] = str(manifest_path)
    document["change"] = {"id": change["id"], "status": change["status"], "revision": change["revision"]}
    progress(99, "平滑视频与更改记录已写入 .alicePD")
    return document


def _smooth_video_eis_30(
    dataset_id: str,
    episode: dict,
    media: dict,
    progress,
    *,
    smoothing: float,
    sharpen_strength: float,
    border_zoom: float,
    target_fps: float,
    motion_compensation: bool,
    run_id: str | None,
) -> dict:
    require_media_eligibility(media, "video_smoothing")
    source_path = Path(str(media.get("path") or ""))
    if not source_path.is_file():
        raise ValueError(f"视频不存在: {media.get('relative_path') or source_path}")
    output_path, manifest_path = _artifact_paths(
        dataset_id,
        episode["id"],
        str(media.get("stream_name") or source_path.name),
        run_id,
    )
    audit_path = manifest_path.with_name(manifest_path.stem + ".frames.npz")
    temporary_video = output_path.with_name(output_path.stem + ".part.mp4")
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法解码视频: {media.get('relative_path') or source_path.name}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or media.get("fps") or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or media.get("width") or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or media.get("height") or 0)
    declared_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or media.get("source_frame_count") or media.get("frame_count") or 0)
    target_fps = max(1.0, min(60.0, float(target_fps)))
    if width <= 0 or height <= 0 or declared_count <= 0 or source_fps <= 0.01:
        capture.release()
        raise RuntimeError("视频元数据无效，无法执行高帧率 EIS")
    if source_fps <= target_fps + HIGH_RATE_MINIMUM_MARGIN_FPS:
        capture.release()
        return _smooth_video_native(
            dataset_id,
            episode,
            media,
            progress,
            smoothing=smoothing,
            sharpen_strength=sharpen_strength,
            border_zoom=border_zoom,
            run_id=run_id,
            requested_mode="eis_30",
            fallback_reason=f"source_fps_not_above_target:{source_fps:.6f}<={target_fps:.6f}",
        )

    try:
        analysis_max_side = int(os.environ.get("ALICE_VIDEO_ANALYSIS_MAX_SIDE", DEFAULT_ANALYSIS_MAX_SIDE))
    except ValueError:
        analysis_max_side = DEFAULT_ANALYSIS_MAX_SIDE
    analysis_width, analysis_height, analysis_scale = _analysis_dimensions(width, height, analysis_max_side)
    motion_rows: list[tuple[float, float, float]] = []
    motion_valid: list[bool] = []
    motion_confidence: list[float] = []
    raw_timestamps: list[float] = []
    previous_gray: np.ndarray | None = None
    decoded_count = 0
    progress(2, f"高帧率 EIS 第一遍：分析 {source_fps:.3f} FPS 相机轨迹")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            raw_timestamps.append(float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0)
            motion_frame = frame
            if analysis_width != width or analysis_height != height:
                motion_frame = cv2.resize(frame, (analysis_width, analysis_height), interpolation=cv2.INTER_AREA)
            current_gray = cv2.cvtColor(motion_frame, cv2.COLOR_BGR2GRAY)
            if previous_gray is None:
                motion_rows.append((0.0, 0.0, 0.0))
                motion_valid.append(True)
                motion_confidence.append(1.0)
            else:
                dx, dy, angle, valid, confidence = _estimate_motion_detail(previous_gray, current_gray)
                if analysis_scale < 1.0:
                    dx /= analysis_scale
                    dy /= analysis_scale
                motion_rows.append((dx if valid else 0.0, dy if valid else 0.0, angle if valid else 0.0))
                motion_valid.append(bool(valid))
                motion_confidence.append(float(confidence if valid else 0.0))
            previous_gray = current_gray
            decoded_count += 1
            if decoded_count == 1 or decoded_count % 20 == 0 or decoded_count == declared_count:
                progress(3 + 32 * decoded_count / max(1, declared_count), f"高帧率轨迹分析 {decoded_count}/{declared_count}")
    finally:
        capture.release()
    if decoded_count <= 0:
        raise RuntimeError("高帧率 EIS 没有解码到任何源帧")

    source_times, timestamp_source = _normalized_source_timestamps(raw_timestamps, decoded_count, source_fps)
    target_times, source_positions = _target_timeline(source_times, source_fps, target_fps)
    if target_times.size <= 0 or source_positions.size != target_times.size:
        raise RuntimeError("无法建立 30 FPS 目标时间轴")
    snapped = np.rint(source_positions)
    source_positions = np.where(np.abs(source_positions - snapped) <= 1e-7, snapped, source_positions)
    target_left = np.floor(source_positions).astype(np.int64)
    target_right = np.ceil(source_positions).astype(np.int64)
    target_right = np.clip(target_right, 0, decoded_count - 1)
    target_left = np.clip(target_left, 0, decoded_count - 1)
    target_alpha = source_positions - target_left
    motion = np.asarray(motion_rows, dtype=np.float64)
    matrices, corrections, clipped_corrections = _stabilization_matrices(
        motion,
        width,
        height,
        smoothing,
        border_zoom,
    )
    target_matrices = (
        matrices[target_left].astype(np.float64) * (1.0 - target_alpha[:, None, None])
        + matrices[target_right].astype(np.float64) * target_alpha[:, None, None]
    ).astype(np.float32)

    cuda_processor, cuda_fallback_error = _create_cuda_frame_processor(width, height, sharpen_strength)
    if cuda_processor is not None:
        processing_backend, processing_device = "cuda-frame", cuda_processor.device_name
    else:
        processing_backend, processing_device = "cpu", "CPU"
    temporary_video.unlink(missing_ok=True)
    writer = _create_frame_writer(
        temporary_video,
        target_fps,
        width,
        height,
        cuda_processor.gpu_device if cuda_processor is not None else None,
    )
    writer_name = writer.name
    writer_gpu = getattr(writer, "gpu_device", None)
    writer_hardware_accelerated = writer.hardware_accelerated
    flow_confidence = np.ones(len(target_times), dtype=np.float32)
    synthetic_mask = np.zeros(len(target_times), dtype=np.bool_)
    fallback_code = np.zeros(len(target_times), dtype=np.int16)
    fallback_reasons = {0: "none", 1: "motion_compensation_disabled", 2: "low_flow_confidence", 3: "dense_flow_error", 4: "missing_bracketing_frame", 5: "flow_resolution_limit"}
    target_index = 0
    source_index = 0
    previous_stabilized: np.ndarray | None = None
    previous_index = -1
    processing_failed = True
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        writer.abort()
        temporary_video.unlink(missing_ok=True)
        raise RuntimeError("高帧率 EIS 第二遍无法重新打开源视频")

    def stabilize(frame: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        nonlocal cuda_processor, cuda_fallback_error, processing_backend, processing_device
        if cuda_processor is not None:
            try:
                return cuda_processor.process(frame, matrix)
            except RuntimeError as exc:
                cuda_fallback_error = str(exc)[:240]
                cuda_processor = None
                processing_backend = "cpu-after-cuda-fallback"
                processing_device = "CPU"
        warped = cv2.warpAffine(
            frame,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return _as_array(_enhance(warped, sharpen_strength))

    progress(36, f"高帧率 EIS 第二遍：生成 {len(target_times)} 帧 @{target_fps:g} FPS")
    try:
        while source_index < decoded_count:
            ok, frame = capture.read()
            if not ok:
                break
            current_stabilized = stabilize(frame, matrices[source_index])
            while target_index < len(target_times) and target_right[target_index] <= source_index:
                left = int(target_left[target_index])
                right = int(target_right[target_index])
                alpha = float(target_alpha[target_index])
                if left == right == source_index:
                    output_frame = current_stabilized
                elif previous_stabilized is not None and previous_index == left and source_index == right:
                    synthetic_mask[target_index] = True
                    if motion_compensation:
                        output_frame, confidence, fallback = _motion_compensated_frame(previous_stabilized, current_stabilized, alpha)
                        flow_confidence[target_index] = confidence
                        if fallback:
                            fallback_code[target_index] = 2 if fallback == "low_flow_confidence" else 5 if fallback == "flow_resolution_limit" else 3
                    else:
                        output_frame = previous_stabilized if alpha < 0.5 else current_stabilized
                        flow_confidence[target_index] = 0.0
                        fallback_code[target_index] = 1
                else:
                    output_frame = current_stabilized
                    flow_confidence[target_index] = 0.0
                    fallback_code[target_index] = 4
                writer.write(output_frame)
                target_index += 1
                if target_index == 1 or target_index % 10 == 0 or target_index == len(target_times):
                    progress(38 + 55 * target_index / max(1, len(target_times)), f"EIS 运动补偿降采样 {target_index}/{len(target_times)}")
            previous_stabilized = current_stabilized
            previous_index = source_index
            source_index += 1
        if previous_stabilized is not None:
            while target_index < len(target_times):
                writer.write(previous_stabilized)
                flow_confidence[target_index] = 0.0
                fallback_code[target_index] = 4
                target_index += 1
        writer.close()
        processing_failed = False
    finally:
        capture.release()
        if processing_failed:
            writer.abort()
            temporary_video.unlink(missing_ok=True)
    if target_index != len(target_times) or not temporary_video.is_file():
        temporary_video.unlink(missing_ok=True)
        raise RuntimeError(f"EIS 30 FPS 输出帧数不完整：{target_index}/{len(target_times)}")
    temporary_video.replace(output_path)

    _write_motion_audit(
        audit_path,
        source_timestamps=source_times.astype(np.float64),
        target_timestamps=target_times.astype(np.float64),
        source_frame_positions=source_positions.astype(np.float64),
        target_left_frames=target_left.astype(np.int64),
        target_right_frames=target_right.astype(np.int64),
        interpolation_alpha=target_alpha.astype(np.float32),
        synthetic_mask=synthetic_mask,
        flow_confidence=flow_confidence,
        fallback_code=fallback_code,
        source_motion=motion.astype(np.float32),
        source_motion_valid=np.asarray(motion_valid, dtype=np.bool_),
        source_motion_confidence=np.asarray(motion_confidence, dtype=np.float32),
        source_stabilization_matrices=matrices.astype(np.float32),
        target_stabilization_matrices=target_matrices,
        source_stabilization_correction=corrections.astype(np.float32),
    )
    fallback_counts = {
        fallback_reasons[int(code)]: int(np.count_nonzero(fallback_code == code))
        for code in np.unique(fallback_code)
        if int(code) != 0
    }
    summary = {
        "frame_count": int(len(target_times)),
        "fps": round(target_fps, 4),
        "source_frame_count": int(decoded_count),
        "source_fps": round(source_fps, 6),
        "duration_seconds": round(len(target_times) / target_fps, 6),
        "smoothing_mode": "eis_motion_compensated_30",
        "requested_mode": "eis_30",
        "width": width,
        "height": height,
        "failed_motion_estimates": int(np.count_nonzero(~np.asarray(motion_valid, dtype=bool))),
        "mean_camera_motion_px": round(float(np.linalg.norm(motion[:, :2], axis=1).mean()), 4),
        "mean_stabilization_correction_px": round(float(np.linalg.norm(corrections[:, :2], axis=1).mean()), 4),
        "clipped_stabilization_frame_count": clipped_corrections,
        "synthetic_frame_count": int(synthetic_mask.sum()),
        "flow_fallback_frame_count": int(np.count_nonzero(fallback_code)),
        "mean_flow_confidence": round(float(flow_confidence[synthetic_mask].mean()) if synthetic_mask.any() else 1.0, 6),
        "timestamp_source": timestamp_source,
        "processing_backend": processing_backend,
        "processing_device": processing_device,
        "encoder": writer_name,
        "encoder_gpu": writer_gpu,
        "cuda_fallback_error": cuda_fallback_error,
        "hardware_accelerated": bool(processing_backend == "cuda-frame" or writer_hardware_accelerated),
        "motion_analysis_width": analysis_width,
        "motion_analysis_height": analysis_height,
    }
    document = {
        "schema": VIDEO_SMOOTHING_SCHEMA,
        "dataset_id": dataset_id,
        "episode_id": episode["id"],
        "full_run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_policy": "Source video remains read-only; EIS output and audit stay in .alicePD.",
        "source_video": {
            "file_id": media.get("file_id"),
            "stream_name": media.get("stream_name"),
            "relative_path": media.get("relative_path"),
            "frame_count": decoded_count,
            "fps": round(source_fps, 6),
        },
        "output_video": str(output_path),
        "method": "two_pass_zero_phase_eis_plus_bidirectional_flow_resample",
        "limitations": "The 30 FPS video is derived. Low-confidence optical flow falls back to a real stabilized source frame.",
        "config": {
            "mode": "eis_30",
            "target_fps": target_fps,
            "motion_compensation": bool(motion_compensation),
            "minimum_flow_confidence": 0.35,
            "trajectory_smoothing": smoothing,
            "sharpen_strength": sharpen_strength,
            "border_zoom": border_zoom,
            "motion_analysis_max_side": analysis_max_side,
            "flow_analysis_max_side": DEFAULT_FLOW_MAX_SIDE,
            "audio_preserved": False,
        },
        "source_frame_positions": [round(float(value), 9) for value in source_positions],
        "retiming": {
            "mode": "source_pts_to_uniform_target_fps",
            "source_frame_count": decoded_count,
            "source_fps": round(source_fps, 6),
            "output_frame_count": int(len(target_times)),
            "target_fps": round(target_fps, 6),
            "timestamp_source": timestamp_source,
            "synthetic_frame_count": int(synthetic_mask.sum()),
            "fallback_counts": fallback_counts,
        },
        "frame_audit": {
            "artifact_path": str(audit_path),
            "format": "npz",
            "arrays": [
                "source_timestamps", "target_timestamps", "source_frame_positions",
                "target_left_frames", "target_right_frames", "interpolation_alpha",
                "synthetic_mask", "flow_confidence", "fallback_code", "source_motion",
                "source_motion_valid", "source_motion_confidence",
                "source_stabilization_matrices", "target_stabilization_matrices",
                "source_stabilization_correction",
            ],
            "fallback_codebook": {str(key): value for key, value in fallback_reasons.items()},
        },
        "geometry_contract": {
            "pixel_transform_available": True,
            "source_stabilization_matrices": "frame_audit.source_stabilization_matrices",
            "target_stabilization_matrices": "frame_audit.target_stabilization_matrices",
            "target_transform_rule": "interpolate left/right source stabilization matrices using interpolation_alpha",
            "default_full_export_enabled": True,
        },
        "summary": summary,
    }
    _write_atomic(manifest_path, document)
    change = record_change(
        dataset_id,
        "video_smoothing",
        episode["id"],
        f"High-rate EIS 30 FPS: {episode['name']} / {media.get('stream_name') or source_path.name}",
        [manifest_path, output_path, audit_path],
        {**summary, "stream_name": media.get("stream_name")},
        [str(media.get("relative_path") or "")],
    )
    document["artifact_path"] = str(manifest_path)
    document["change"] = {"id": change["id"], "status": change["status"], "revision": change["revision"]}
    progress(99, f"高帧率 EIS 已输出 {len(target_times)} 帧 @{target_fps:g} FPS")
    return document


def smooth_video(
    dataset_id: str,
    episode: dict,
    media: dict,
    progress,
    smoothing: float = 0.9,
    sharpen_strength: float = 0.32,
    border_zoom: float = 1.025,
    run_id: str | None = None,
    mode: str = "native",
    target_fps: float = DEFAULT_HIGH_RATE_TARGET_FPS,
    motion_compensation: bool = True,
) -> dict:
    selected_mode = str(mode or "native").strip().casefold()
    if selected_mode in {"eis_30", "high_rate_eis", "flow_resampled_30"}:
        return _smooth_video_eis_30(
            dataset_id,
            episode,
            media,
            progress,
            smoothing=smoothing,
            sharpen_strength=sharpen_strength,
            border_zoom=border_zoom,
            target_fps=target_fps,
            motion_compensation=motion_compensation,
            run_id=run_id,
        )
    return _smooth_video_native(
        dataset_id,
        episode,
        media,
        progress,
        smoothing=smoothing,
        sharpen_strength=sharpen_strength,
        border_zoom=border_zoom,
        run_id=run_id,
        requested_mode=selected_mode,
    )
