from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .storage import change_is_applied, dataset_artifact_dir, record_change, slugify


VIDEO_SMOOTHING_SCHEMA = "alice/video-smoothing/v1"


def _artifact_paths(dataset_id: str, episode_id: str, stream_name: str) -> tuple[Path, Path]:
    root = dataset_artifact_dir(dataset_id, "video-smoothing") / slugify(episode_id)
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


def _estimate_motion(previous_gray: np.ndarray, current_gray: np.ndarray) -> tuple[float, float, float, bool]:
    points = cv2.goodFeaturesToTrack(previous_gray, maxCorners=240, qualityLevel=0.01, minDistance=18, blockSize=3)
    if points is None or len(points) < 8:
        return 0.0, 0.0, 0.0, False
    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, points, None)
    if moved is None or status is None:
        return 0.0, 0.0, 0.0, False
    valid = status.reshape(-1).astype(bool)
    source = points.reshape(-1, 2)[valid]
    target = moved.reshape(-1, 2)[valid]
    if len(source) < 8:
        return 0.0, 0.0, 0.0, False
    matrix, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if matrix is None:
        return 0.0, 0.0, 0.0, False
    dx = float(matrix[0, 2])
    dy = float(matrix[1, 2])
    angle = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    return dx, dy, angle, True


def _enhance(frame: np.ndarray, strength: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(frame, (0, 0), 1.15)
    return cv2.addWeighted(frame, 1.0 + strength, blurred, -strength, 0)


def smooth_video(
    dataset_id: str,
    episode: dict,
    media: dict,
    progress,
    smoothing: float = 0.9,
    sharpen_strength: float = 0.32,
    border_zoom: float = 1.025,
) -> dict:
    source_path = Path(str(media.get("path") or ""))
    if not source_path.is_file():
        raise ValueError(f"视频不存在: {media.get('relative_path') or source_path}")

    output_path, manifest_path = _artifact_paths(dataset_id, episode["id"], str(media.get("stream_name") or source_path.name))
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

    writer = cv2.VideoWriter(str(temporary_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("无法创建 .alicePD 平滑视频")

    trajectory = np.zeros(3, dtype=np.float64)
    smoothed_trajectory = np.zeros(3, dtype=np.float64)
    raw_motion: list[float] = []
    correction_motion: list[float] = []
    failed_estimates = 0
    processed = 0
    previous_gray: np.ndarray | None = None
    zoom_matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 0.0, border_zoom)
    progress(2, f"打开视频 {media.get('stream_name') or source_path.name}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous_gray is None:
                stabilized = frame
            else:
                dx, dy, angle, valid = _estimate_motion(previous_gray, current_gray)
                if not valid:
                    failed_estimates += 1
                transform = np.array([dx, dy, angle], dtype=np.float64)
                trajectory += transform
                smoothed_trajectory = smoothing * smoothed_trajectory + (1.0 - smoothing) * trajectory
                correction = smoothed_trajectory - trajectory
                adjusted = transform + correction
                cosine, sine = float(np.cos(adjusted[2])), float(np.sin(adjusted[2]))
                matrix = np.array([[cosine, -sine, adjusted[0]], [sine, cosine, adjusted[1]]], dtype=np.float32)
                stabilized = cv2.warpAffine(frame, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                raw_motion.append(float(np.linalg.norm(transform[:2])))
                correction_motion.append(float(np.linalg.norm(correction[:2])))
            stabilized = cv2.warpAffine(stabilized, zoom_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            writer.write(_enhance(stabilized, sharpen_strength))
            previous_gray = current_gray
            processed += 1
            if processed == 1 or processed % 10 == 0 or processed == frame_count:
                progress(5 + 88 * processed / max(1, frame_count), f"稳像与清晰度增强 {processed}/{frame_count}")
    finally:
        capture.release()
        writer.release()

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
    }
    document = {
        "schema": VIDEO_SMOOTHING_SCHEMA,
        "dataset_id": dataset_id,
        "episode_id": episode["id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_policy": "Source video remains read-only; the enhanced video is stored only in .alicePD.",
        "source_video": {
            "file_id": media.get("file_id"),
            "stream_name": media.get("stream_name"),
            "relative_path": media.get("relative_path"),
        },
        "output_video": str(output_path),
        "method": "optical_flow_stabilization_plus_unsharp_mask",
        "limitations": "Stabilization and sharpening can reduce shake and perceived blur, but cannot reconstruct detail lost during exposure.",
        "config": {
            "trajectory_smoothing": smoothing,
            "sharpen_strength": sharpen_strength,
            "border_zoom": border_zoom,
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
