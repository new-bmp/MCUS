from __future__ import annotations

"""Browser preview preparation.

Source media is never rewritten.  Incompatible codecs are converted into a
small, seekable H.264 MP4 under the dataset's ``.alicePD`` sidecar.  The
manifest keeps the decoded source PTS for frame-accurate overlays.
"""

import json
import math
import os
import shutil
import subprocess
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2

from .storage import dataset_artifact_dir, slugify


PREVIEW_PROXY_SCHEMA = "alice/preview-proxy/v2"
_BROWSER_SOURCE_CACHE: dict[str, dict] = {}
_BROWSER_SOURCE_LOCK = threading.RLock()


def _paths(dataset_id: str, episode_id: str, media: dict) -> tuple[Path, Path]:
    root = dataset_artifact_dir(dataset_id, "preview-proxy") / slugify(episode_id)
    root.mkdir(parents=True, exist_ok=True)
    variant = str(media.get("preview_variant") or "").strip()
    stem = slugify("-".join(filter(None, (str(media.get("stream_name") or Path(str(media.get("path") or "video")).stem), variant))))
    return root / f"{stem}.preview.mp4", root / f"{stem}.preview.alice"


def _source_signature(path: Path) -> dict:
    stat = path.stat()
    return {"size_bytes": stat.st_size, "modified_ns": stat.st_mtime_ns}


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def _fourcc_name(value: float | int) -> str:
    number = int(value or 0)
    if not number:
        return ""
    return "".join(chr((number >> (8 * index)) & 0xFF) for index in range(4)).strip().lower()


def _source_codec_info(path: Path, media: dict | None = None) -> dict:
    """Return cheap codec/container information used before submitting a job."""

    try:
        signature = _source_signature(path)
        cache_key = f"{path.resolve()}:{signature['size_bytes']}:{signature['modified_ns']}"
    except OSError:
        cache_key = str(path)
    with _BROWSER_SOURCE_LOCK:
        cached = _BROWSER_SOURCE_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    suffix = path.suffix.casefold()
    codec = ""
    fps = float((media or {}).get("fps") or 0.0)
    frame_count = int((media or {}).get("frame_count") or 0)
    width = int((media or {}).get("width") or 0)
    height = int((media or {}).get("height") or 0)
    capture = cv2.VideoCapture(str(path))
    if capture.isOpened():
        codec = _fourcc_name(capture.get(cv2.CAP_PROP_FOURCC))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or fps or 30.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or frame_count or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or width or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or height or 0)
    capture.release()

    # Chromium reliably accepts H.264 in MP4 and VP8/VP9/AV1 in WebM.  FMP4
    # (MPEG-4 Part 2), which is common in robot datasets, must be transcoded.
    h264 = codec in {"h264", "avc1", "x264", "avc3"}
    webm_codec = codec in {"vp8", "vp80", "vp9", "vp90", "av1", "av01"}
    if suffix in {".webm", ".weba"} and webm_codec:
        mime_type = "video/webm"
        browser_compatible = True
    elif suffix in {".mp4", ".m4v", ".mov"} and h264:
        mime_type = "video/mp4"
        browser_compatible = True
    else:
        mime_type = "video/mp4" if suffix in {".mp4", ".m4v", ".mov"} else "video/webm"
        browser_compatible = False
    result = {
        "codec": codec or "unknown",
        "mime_type": mime_type,
        "browser_compatible": browser_compatible,
        "fps": fps if fps > 0.01 else 30.0,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "container": suffix.lstrip(".") or "unknown",
    }
    with _BROWSER_SOURCE_LOCK:
        _BROWSER_SOURCE_CACHE[cache_key] = dict(result)
    return result


def _ffmpeg_executable() -> str | None:
    configured = os.environ.get("ALICE_FFMPEG")
    if configured and Path(configured).is_file():
        return configured
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        candidate = imageio_ffmpeg.get_ffmpeg_exe()
        if candidate and Path(candidate).is_file():
            return candidate
    except Exception:
        pass
    return None


def _normalise_pts(values: list[float], fps: float) -> list[float]:
    if not values:
        return []
    first = values[0] if math.isfinite(values[0]) else 0.0
    output: list[float] = []
    previous = 0.0
    step = 1.0 / max(0.01, fps)
    for index, value in enumerate(values):
        point = value - first if math.isfinite(value) else index * step
        if not math.isfinite(point) or point < 0:
            point = index * step
        # A broken timestamp must not make the video seek backwards.  Keep
        # genuine long gaps, but clamp tiny codec rounding regressions.
        if output and point < previous:
            point = previous
        output.append(round(point, 6))
        previous = point
    return output


def _collect_source_pts(source: Path, fps: float, declared_frames: int = 0) -> list[float]:
    """Read packet timestamps without decoding pixels (``grab`` is cheap)."""

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        return []
    values: list[float] = []
    fallback_step = 1000.0 / max(0.01, fps)
    try:
        while capture.grab():
            position = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            if not math.isfinite(position) or position < 0:
                position = len(values) * fallback_step
            values.append(position / 1000.0)
            # Some broken containers report an unbounded frame count.  Do not
            # let a corrupt stream keep a worker alive forever.
            if len(values) > 2_000_000:
                break
    finally:
        capture.release()
    if declared_frames and len(values) > declared_frames * 2 and declared_frames > 10:
        values = values[:declared_frames]
    return _normalise_pts(values, fps)


def _nvenc_available() -> bool:
    """Detect the NVIDIA encode runtime without paying for a failed encode."""

    if shutil.which("nvidia-smi"):
        return True
    if os.name != "nt":
        return False
    try:
        import ctypes

        ctypes.WinDLL("nvEncodeAPI64.dll")
        return True
    except (AttributeError, OSError):
        return False


def _encoder_specs() -> list[tuple[str, list[str], str]]:
    """Ordered encoders: NVENC when present, Windows MF/AMF, then x264."""

    requested = os.environ.get("ALICE_PREVIEW_ENCODER", "").strip()
    if requested:
        names = [requested]
    elif os.name == "nt":
        names = (["h264_nvenc"] if _nvenc_available() else []) + ["h264_mf", "h264_amf", "libx264"]
    else:
        names = (["h264_nvenc"] if _nvenc_available() else []) + ["libx264"]
    specs: list[tuple[str, list[str], str]] = []
    for name in names:
        normalized = name.casefold()
        if normalized == "h264_nvenc":
            specs.append(("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p2", "-tune", "ll", "-rc", "vbr", "-cq", "25", "-b:v", "0"], "nvidia-nvenc"))
        elif normalized == "h264_mf":
            specs.append(("h264_mf", ["-c:v", "h264_mf", "-rate_control", "quality", "-quality", "60"], "mediafoundation"))
        elif normalized == "h264_amf":
            specs.append(("h264_amf", ["-c:v", "h264_amf", "-quality", "speed", "-rc", "qvbr", "-qvbr_quality_level", "28"], "amd-amf"))
        elif normalized in {"libx264", "x264"}:
            specs.append(("libx264", ["-c:v", "libx264", "-preset", "superfast", "-crf", "25"], "software"))
    if not specs:
        specs.append(("libx264", ["-c:v", "libx264", "-preset", "superfast", "-crf", "25"], "software"))
    return specs


class PreviewProxyManager:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="alice-preview-proxy")
        self._jobs: dict[str, dict] = {}
        self._keys: dict[str, str] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(dataset_id: str, episode_id: str, media: dict) -> str:
        return f"{dataset_id}:{episode_id}:{media.get('file_id') or media.get('path')}:{media.get('preview_variant') or 'source'}"

    def _artifact_status(self, dataset_id: str, episode_id: str, media: dict) -> dict | None:
        source = Path(str(media.get("path") or ""))
        output, manifest_path = _paths(dataset_id, episode_id, media)
        if not source.is_file() or not output.is_file() or not manifest_path.is_file():
            return None
        try:
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if document.get("schema") != PREVIEW_PROXY_SCHEMA:
            return None
        if document.get("source_video", {}).get("file_id") != media.get("file_id"):
            return None
        try:
            signature_matches = document.get("source_signature") == _source_signature(source)
        except OSError:
            signature_matches = False
        if not signature_matches:
            return None
        return {
            "status": "ready",
            "progress": 100,
            "message": "浏览器兼容预览已就绪",
            "delivery": "proxy",
            "mime_type": document.get("mime_type", "video/mp4"),
            "codec": document.get("codec", "h264"),
            "encoder": document.get("encoder"),
            "acceleration": document.get("acceleration"),
            "frame_count": document.get("frame_count"),
            "fps": document.get("fps"),
            "width": document.get("width"),
            "height": document.get("height"),
            "mapping_path": str(manifest_path),
        }

    def status(self, dataset_id: str, episode_id: str, media: dict) -> dict:
        key = self._key(dataset_id, episode_id, media)
        with self._lock:
            job_id = self._keys.get(key)
            job = dict(self._jobs[job_id]) if job_id and job_id in self._jobs else None
        if job and job.get("status") in {"queued", "running"}:
            return job
        artifact = self._artifact_status(dataset_id, episode_id, media)
        if artifact:
            return artifact
        if job and job.get("status") == "failed":
            return job
        source = Path(str(media.get("path") or ""))
        if source.is_file():
            info = _source_codec_info(source, media)
            if info["browser_compatible"]:
                return {
                    "status": "ready",
                    "progress": 100,
                    "message": "原始视频可由浏览器直接播放",
                    "delivery": "source",
                    "mime_type": info["mime_type"],
                    "codec": info["codec"],
                    "frame_count": info["frame_count"],
                    "fps": info["fps"],
                    "width": info["width"],
                    "height": info["height"],
                }
            return {
                "status": "missing",
                "progress": 0,
                "message": f"源视频编码 {info['codec']} 需要生成兼容预览",
                "delivery": None,
                "source_codec": info["codec"],
            }
        return {"status": "missing", "progress": 0, "message": "视频文件不存在", "delivery": None}

    def submit(self, dataset_id: str, episode_id: str, media: dict, force_proxy: bool = False) -> dict:
        source = Path(str(media.get("path") or ""))
        if not source.is_file():
            raise ValueError(f"视频不存在: {media.get('relative_path') or source}")
        status = self.status(dataset_id, episode_id, media)
        if status.get("status") == "ready" and (status.get("delivery") == "proxy" or not force_proxy):
            return status
        key = self._key(dataset_id, episode_id, media)
        with self._lock:
            existing_id = self._keys.get(key)
            if existing_id and self._jobs.get(existing_id, {}).get("status") in {"queued", "running"}:
                return dict(self._jobs[existing_id])
            job_id = uuid.uuid4().hex
            job = {
                "id": job_id,
                "kind": "preview_proxy",
                "status": "queued",
                "progress": 0,
                "message": "预览代理已进入后台队列",
                "delivery": "proxy",
                "error": None,
            }
            self._jobs[job_id] = job
            self._keys[key] = job_id
        self._executor.submit(self._run, job_id, dataset_id, episode_id, dict(media))
        return dict(job)

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(changes)

    @staticmethod
    def _run_ffmpeg(ffmpeg: str, source: Path, temporary: Path, width: int, height: int, fps: float, expected: int, spec: tuple[str, list[str], str], update) -> tuple[int, str]:
        encoder, encoder_args, acceleration = spec
        # Media Foundation rejects a setpts filter on some Windows builds.
        # FFmpeg already normalises the usual camera streams to zero and the
        # original PTS are kept separately in the manifest for overlays.
        scale = f"scale=w={width}:h={height}:flags=fast_bilinear,format=yuv420p"
        command = [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            scale,
            "-fps_mode",
            "passthrough",
            *encoder_args,
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(max(1, round(fps))),
            "-movflags",
            "+faststart",
            "-progress",
            "pipe:1",
            str(temporary),
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        progress_frame = 0
        if process.stdout is not None:
            for line in process.stdout:
                line = line.strip()
                if line.startswith("frame="):
                    try:
                        progress_frame = max(progress_frame, int(line.split("=", 1)[1]))
                        if expected:
                            update(progress=round(min(96.0, 5.0 + 90.0 * progress_frame / expected), 1), message=f"生成 H.264 预览 {progress_frame}/{expected}")
                    except ValueError:
                        pass
        stderr = process.stderr.read() if process.stderr is not None else ""
        return_code = process.wait()
        if return_code != 0 or not temporary.is_file() or temporary.stat().st_size <= 0:
            detail = stderr.strip().splitlines()[-1] if stderr.strip() else f"ffmpeg exit {return_code}"
            raise RuntimeError(f"{encoder}: {detail}")
        return progress_frame, acceleration

    @staticmethod
    def _run_opencv(source: Path, temporary: Path, width: int, height: int, fps: float, expected: int, update) -> tuple[int, str]:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("无法打开源视频")
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
        if not writer.isOpened():
            writer.release()
            capture.release()
            raise RuntimeError("FFmpeg 不可用且 OpenCV 无法创建 H.264 预览")
        processed = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                processed += 1
                if processed == 1 or processed % 10 == 0:
                    update(progress=round(min(96.0, 5.0 + 90.0 * processed / max(expected, processed, 1)), 1), message=f"生成 H.264 预览 {processed}/{expected or '?'}")
        finally:
            capture.release()
            writer.release()
        if processed == 0 or not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError("预览代理没有写出有效帧")
        return processed, "opencv-avc1"

    def _run(self, job_id: str, dataset_id: str, episode_id: str, media: dict) -> None:
        source = Path(str(media.get("path") or ""))
        output, manifest_path = _paths(dataset_id, episode_id, media)
        temporary = output.with_name(output.stem + ".part.mp4")
        try:
            self._update(job_id, status="running", progress=1, message="正在读取视频时间戳")
            initial_signature = _source_signature(source)
            info = _source_codec_info(source, media)
            fps = float(info.get("fps") or media.get("fps") or 30.0)
            source_width = int(info.get("width") or media.get("width") or 0)
            source_height = int(info.get("height") or media.get("height") or 0)
            declared_frames = int(info.get("frame_count") or media.get("frame_count") or 0)
            if source_width <= 0 or source_height <= 0:
                raise RuntimeError("视频尺寸无效")
            scale = min(1.0, 1280.0 / source_width, 720.0 / source_height)
            width = max(2, int(round(source_width * scale / 2.0)) * 2)
            height = max(2, int(round(source_height * scale / 2.0)) * 2)
            source_pts = _collect_source_pts(source, fps, declared_frames)
            expected = len(source_pts) or declared_frames
            if not source_pts and expected:
                source_pts = _normalise_pts([index / fps for index in range(expected)], fps)
            temporary.unlink(missing_ok=True)

            ffmpeg = _ffmpeg_executable()
            encoder = "opencv-avc1"
            acceleration = "opencv"
            processed = 0
            if ffmpeg:
                errors: list[str] = []
                for spec in _encoder_specs():
                    temporary.unlink(missing_ok=True)
                    try:
                        processed, acceleration = self._run_ffmpeg(ffmpeg, source, temporary, width, height, fps, expected, spec, lambda **changes: self._update(job_id, **changes))
                        encoder = spec[0]
                        break
                    except Exception as exc:
                        errors.append(str(exc))
                else:
                    self._update(job_id, message="硬件/FFmpeg 编码不可用，切换 OpenCV H.264")
            if not processed:
                temporary.unlink(missing_ok=True)
                processed, acceleration = self._run_opencv(source, temporary, width, height, fps, expected, lambda **changes: self._update(job_id, **changes))
            if processed <= 0 or not temporary.is_file():
                raise RuntimeError("预览代理没有写出有效帧")
            # The source can be edited while a background job is running.  Do
            # not publish an artifact whose mapping belongs to an old source.
            source_signature = _source_signature(source)
            if source_signature != initial_signature:
                raise RuntimeError("源视频在预览生成期间发生变化，已丢弃本次结果")
            temporary.replace(output)
            if len(source_pts) < processed:
                original_count = len(source_pts)
                last_point = source_pts[-1] if source_pts else 0.0
                source_pts.extend(last_point + (index - original_count + 1) / fps for index in range(original_count, processed))
            source_pts = source_pts[:processed]
            document = {
                "schema": PREVIEW_PROXY_SCHEMA,
                "dataset_id": dataset_id,
                "episode_id": episode_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_policy": "Source media remains read-only; this browser proxy is stored only in .alicePD.",
                "source_video": {
                    "file_id": media.get("file_id"),
                    "stream_name": media.get("stream_name"),
                    "relative_path": media.get("relative_path"),
                },
                "source_signature": source_signature,
                "output_video": str(output),
                "codec": "h264",
                "container": "mp4",
                "mime_type": "video/mp4",
                "encoder": encoder,
                "acceleration": acceleration,
                "fps": round(fps, 6),
                "frame_count": processed,
                "width": width,
                "height": height,
                "source_width": source_width,
                "source_height": source_height,
                "mapping": {
                    "rule": "output frame index equals source decoded frame index",
                    "preview_time_rule": "source_pts_seconds[frame]",
                    "source_pts_seconds": source_pts,
                },
            }
            _write_json_atomic(manifest_path, document)
            self._update(job_id, status="ready", progress=100, message="浏览器兼容预览已就绪", codec="h264", encoder=encoder, acceleration=acceleration, frame_count=processed, fps=fps, width=width, height=height, mapping_path=str(manifest_path), mime_type="video/mp4")
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            self._update(job_id, status="failed", progress=100, message=str(exc), error=str(exc), trace=traceback.format_exc(limit=8))

    def media_path(self, dataset_id: str, episode_id: str, media: dict) -> tuple[Path, str]:
        status = self.status(dataset_id, episode_id, media)
        source = Path(str(media.get("path") or ""))
        if status.get("status") != "ready":
            raise RuntimeError(status.get("message") or "预览尚未就绪")
        if status.get("delivery") == "source":
            return source, str(status.get("mime_type") or "video/mp4")
        output, _ = _paths(dataset_id, episode_id, media)
        if not output.is_file():
            raise RuntimeError("预览代理不存在")
        return output, str(status.get("mime_type") or "video/mp4")

    def mapping_path(self, dataset_id: str, episode_id: str, media: dict) -> Path:
        status = self.status(dataset_id, episode_id, media)
        if status.get("status") != "ready" or status.get("delivery") != "proxy":
            raise RuntimeError("当前视频没有代理帧映射")
        _, manifest_path = _paths(dataset_id, episode_id, media)
        if not manifest_path.is_file():
            raise RuntimeError("代理帧映射不存在")
        return manifest_path


preview_proxy_manager = PreviewProxyManager()
