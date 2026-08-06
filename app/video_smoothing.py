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


VIDEO_SMOOTHING_SCHEMA = "alice/video-smoothing/v2"
DEFAULT_ANALYSIS_MAX_SIDE = 640
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
    return {
        **media,
        "path": str(output_path),
        "frame_count": int(summary.get("frame_count") or media.get("frame_count") or 0),
        "fps": float(summary.get("fps") or media.get("fps") or 30.0),
    }, document


def _as_array(value: Any) -> np.ndarray:
    return value.get() if isinstance(value, cv2.UMat) else np.asarray(value)


def _estimate_motion(previous_gray: Any, current_gray: Any) -> tuple[float, float, float, bool]:
    points = cv2.goodFeaturesToTrack(previous_gray, maxCorners=240, qualityLevel=0.01, minDistance=18, blockSize=3)
    if points is None:
        return 0.0, 0.0, 0.0, False
    point_values = _as_array(points)
    if len(point_values) < 8:
        return 0.0, 0.0, 0.0, False
    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, points, None)
    if moved is None or status is None:
        return 0.0, 0.0, 0.0, False
    valid = _as_array(status).reshape(-1).astype(bool)
    source = point_values.reshape(-1, 2)[valid]
    target = _as_array(moved).reshape(-1, 2)[valid]
    if len(source) < 8:
        return 0.0, 0.0, 0.0, False
    matrix, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if matrix is None:
        return 0.0, 0.0, 0.0, False
    dx = float(matrix[0, 2])
    dy = float(matrix[1, 2])
    angle = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    return dx, dy, angle, True


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


def smooth_video(
    dataset_id: str,
    episode: dict,
    media: dict,
    progress,
    smoothing: float = 0.9,
    sharpen_strength: float = 0.32,
    border_zoom: float = 1.025,
    run_id: str | None = None,
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
