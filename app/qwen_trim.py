from __future__ import annotations

import base64
import hashlib
import json
import math
import threading
import time
import traceback
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from pydantic import BaseModel, Field

from .models import registry
from .storage import dataset_artifact_dir, episode_media, get_manifest, read_frame, record_change, slugify


QWEN_ACTION_TRIM_SCHEMA = "alice/qwen-action-trim/v1"
MAX_IMAGES_PER_REQUEST = 24
SOURCE_EDGE_HASH_BYTES = 256 * 1024


class QwenTrimRequest(BaseModel):
    """Request model for an independent Qwen valid/invalid trimming job."""

    episode_ids: list[str] = Field(default_factory=list)
    all_episodes: bool = False
    media_file_ids: dict[str, str] = Field(default_factory=dict)
    sample_fps: float = Field(default=0.75, ge=0.1, le=4.0)
    window_seconds: float = Field(default=1.5, ge=0.25, le=6.0)
    frames_per_window: int = Field(default=2, ge=2, le=3)
    windows_per_request: int = Field(default=10, ge=1, le=12)
    confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    max_gap_seconds: float = Field(default=1.5, ge=0.0, le=10.0)
    min_valid_seconds: float = Field(default=0.75, ge=0.0, le=10.0)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _artifact_path(dataset_id: str, episode_id: str) -> Path:
    return dataset_artifact_dir(dataset_id, "qwen-action-trim") / f"{slugify(episode_id)}.trim.alice"


def _atomic_temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_temporary_path(path)
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _atomic_temporary_path(path)
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_video_fingerprint(media: dict) -> dict:
    raw_path = str(media.get("path") or "").strip()
    if not raw_path:
        raise ValueError("The selected video has no source path")
    try:
        path = Path(raw_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"The selected video source does not exist: {raw_path}") from exc
    if not path.is_file():
        raise ValueError(f"The selected video source is not a file: {path}")

    before = path.stat()
    size = int(before.st_size)
    digest = hashlib.sha256()
    digest.update(b"alice/source-video-edge/v1\0")
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as source:
        head = source.read(min(SOURCE_EDGE_HASH_BYTES, size))
        digest.update(b"\0head\0")
        digest.update(head)
        tail_offset = max(0, size - SOURCE_EDGE_HASH_BYTES)
        if tail_offset > 0:
            source.seek(tail_offset)
            digest.update(b"\0tail-offset\0")
            digest.update(str(tail_offset).encode("ascii"))
            digest.update(b"\0")
            digest.update(source.read(SOURCE_EDGE_HASH_BYTES))
    after = path.stat()
    identity_before = (int(before.st_dev), int(before.st_ino), int(before.st_size), int(before.st_mtime_ns))
    identity_after = (int(after.st_dev), int(after.st_ino), int(after.st_size), int(after.st_mtime_ns))
    if identity_before != identity_after:
        raise RuntimeError("The source video changed while its fingerprint was being computed")
    return {
        "schema": "alice/source-video-fingerprint/v1",
        "resolved_path": str(path),
        "size_bytes": size,
        "mtime_ns": int(after.st_mtime_ns),
        "edge_sha256": digest.hexdigest(),
        "edge_bytes": SOURCE_EDGE_HASH_BYTES,
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
    }


def _source_fingerprints_match(expected: dict, current: dict) -> bool:
    keys = ("resolved_path", "size_bytes", "mtime_ns", "edge_sha256", "device", "inode")
    return all(expected.get(key) == current.get(key) for key in keys)


def _assert_source_unchanged(media: dict, expected: dict, stage: str) -> dict:
    current = _source_video_fingerprint(media)
    if not _source_fingerprints_match(expected, current):
        raise RuntimeError(f"The source video changed {stage}; Qwen trim was not committed")
    return current


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _bounded_confidence(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, _safe_float(value, default)))


def _normal_state(value: Any) -> str:
    state = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    if state in {"valid", "active", "action", "actionable", "effective", "manipulating", "yes", "true", "有效"}:
        return "valid"
    if state in {"invalid", "idle", "inactive", "no_action", "ineffective", "waiting", "no", "false", "无效"}:
        return "invalid"
    return "uncertain"


def _validated_response_state(value: Any, window_id: str) -> str:
    token = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    state = _normal_state(value)
    if state != "uncertain" or token in {"uncertain", "unknown", "unsure", "ambiguous"}:
        return state
    raise ValueError(f"Qwen response has no explicit valid/invalid/uncertain state for {window_id}")


def _deduplicate_terms(values: list[Any], limit: int = 80) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        term = str(value or "").strip().strip(" ,.;:")[:120]
        key = term.casefold()
        if term and key not in seen:
            seen.add(key)
            output.append(term)
        if len(output) >= limit:
            break
    return output


def _load_behavior_context(dataset_id: str, episode_id: str) -> dict:
    context: dict[str, Any] = {"primary_terms": [], "task_label": None, "behavior_description": None}
    targets_path = dataset_artifact_dir(dataset_id, "behavior-targets") / f"{slugify(episode_id)}.targets.alice"
    behavior_path = dataset_artifact_dir(dataset_id, "behavior-annotations") / f"{slugify(episode_id)}.behavior.alice"
    try:
        targets = json.loads(targets_path.read_text(encoding="utf-8")) if targets_path.is_file() else {}
        context["primary_terms"] = _deduplicate_terms([
            *(targets.get("primary_terms") or []),
            *[item.get("name") for item in targets.get("primary_targets", []) if isinstance(item, dict)],
        ])
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    try:
        behavior = json.loads(behavior_path.read_text(encoding="utf-8")) if behavior_path.is_file() else {}
        context["task_label"] = str(behavior.get("task_label") or "")[:200] or None
        context["behavior_description"] = str(behavior.get("behavior_description") or "")[:1200] or None
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    return context


def _window_frames(anchor: int, frame_count: int, fps: float, seconds: float, count: int) -> list[int]:
    radius = max(1, int(round(max(0.25, seconds) * fps / 2.0)))
    if count <= 2:
        candidates = [anchor - radius, anchor + radius]
    else:
        candidates = [anchor - radius, anchor, anchor + radius]
    return sorted({max(0, min(frame_count - 1, int(value))) for value in candidates})


def _sample_windows(
    frame_count: int,
    fps: float,
    sample_fps: float,
    window_seconds: float,
    frames_per_window: int,
) -> list[dict]:
    if frame_count <= 0:
        return []
    stride = max(1, int(round(fps / max(0.1, sample_fps))))
    anchors = list(range(0, frame_count, stride))
    if anchors[-1] != frame_count - 1:
        anchors.append(frame_count - 1)
    return [{
        "window_id": f"w{position:06d}",
        "anchor_frame": anchor,
        "anchor_time": round(anchor / max(0.01, fps), 4),
        "evidence_frames": _window_frames(anchor, frame_count, fps, window_seconds, frames_per_window),
    } for position, anchor in enumerate(anchors)]


def _encode_frame(frame) -> str | None:
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest > 960:
        scale = 960.0 / longest
        frame = cv2.resize(frame, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 76])
    return base64.b64encode(encoded.tobytes()).decode("ascii") if ok else None


def _qwen_json(content: list[dict], max_tokens: int = 3600) -> dict:
    """Use the registry's retrying/repairing JSON path without exposing the API key."""
    if not registry.has_vlm:
        raise RuntimeError("请先配置 Qwen-VLM API")
    with registry._lock:  # Snapshot configuration; never hold this lock during the network request.
        endpoint = str(registry._vlm.endpoint or "")
        model = str(registry._vlm.model or "")
        api_key = str(registry._vlm_key or "")
    return registry._request_json(endpoint, api_key, model, content, max_tokens)


def _batch_prompt(context: dict, windows: list[dict]) -> str:
    compact_context = {
        "task_label": context.get("task_label"),
        "behavior_description": context.get("behavior_description"),
        "known_object_terms": context.get("primary_terms", []),
    }
    ids = [item["window_id"] for item in windows]
    return (
        "You are reviewing a robotics/VLA manipulation video and classifying short temporal windows. "
        "A window is valid only when a human hand, robot hand, or gripper is visibly performing task-relevant physical manipulation, "
        "contact, grasping, releasing, or purposeful approach/withdrawal that is part of the demonstrated action. "
        "Idle waiting, an empty work area, hand/gripper absent, unrelated motion, camera motion alone, setup after completion, and no meaningful interaction are invalid. "
        "Use uncertain only when occlusion or missing evidence prevents a decision; do not use it merely for low motion during a maintained grasp. "
        "Classify every supplied window_id exactly once and do not invent ids. object_nouns must contain concise physical object noun phrases visibly involved in the action. "
        "Return strict JSON only: {windows:[{window_id:string,state:'valid|invalid|uncertain',confidence:number,reason:string,motion:number,contact:number,object_nouns:[string],evidence_frames:[int]}],object_nouns:[string],warnings:[string]}. "
        f"WINDOW_IDS={json.dumps(ids, ensure_ascii=False, separators=(',', ':'))}; "
        f"EPISODE_CONTEXT={json.dumps(compact_context, ensure_ascii=False, separators=(',', ':'))}"
    )


def _request_window_batch(media: dict, windows: list[dict]) -> dict:
    content: list[dict] = [{"type": "text", "text": _batch_prompt(_load_behavior_context(media["dataset_id"], media["episode_id"]), windows)}]
    readable_counts: dict[str, int] = {}
    for window in windows:
        window_id = window["window_id"]
        content.append({
            "type": "text",
            "text": f"WINDOW {window_id}; anchor_frame={window['anchor_frame']}; anchor_time={window['anchor_time']:.4f}s; evidence_frames={window['evidence_frames']}",
        })
        readable_counts[window_id] = 0
        for frame_index in window["evidence_frames"]:
            frame = read_frame(media, frame_index)
            if frame is None:
                continue
            encoded = _encode_frame(frame)
            if encoded is None:
                continue
            readable_counts[window_id] += 1
            content.append({"type": "text", "text": f"{window_id} FRAME {frame_index}"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
    unreadable_windows = [window_id for window_id, count in readable_counts.items() if count <= 0]
    if unreadable_windows:
        preview = ", ".join(unreadable_windows[:8])
        raise RuntimeError(f"Selected video has no readable evidence for Qwen windows: {preview}")
    raw = _qwen_json(content)
    raw["_readable_counts"] = readable_counts
    return raw


def _response_entries(raw: dict) -> list[dict]:
    value = raw.get("windows")
    if value is None:
        value = raw.get("results", raw.get("verdicts", []))
    if isinstance(value, dict):
        output = []
        for key, item in value.items():
            if isinstance(item, dict):
                output.append({"window_id": key, **item})
        return output
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _validate_batch_response(raw: dict, windows: list[dict]) -> tuple[list[dict], list[str], list[str]]:
    requested = {item["window_id"]: item for item in windows}
    readable = raw.get("_readable_counts") if isinstance(raw.get("_readable_counts"), dict) else {}
    accepted: dict[str, dict] = {}
    nouns: list[Any] = list(raw.get("object_nouns", [])) if isinstance(raw.get("object_nouns"), list) else []
    warnings = [str(value)[:500] for value in raw.get("warnings", [])[:20]] if isinstance(raw.get("warnings"), list) else []
    for item in _response_entries(raw):
        window_id = str(item.get("window_id") or item.get("id") or item.get("sample_id") or "")
        if window_id not in requested:
            continue
        try:
            raw_confidence = float(item["confidence"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Qwen response has no numeric confidence for {window_id}") from exc
        if not math.isfinite(raw_confidence):
            raise ValueError(f"Qwen response has non-finite confidence for {window_id}")
        confidence = _bounded_confidence(raw_confidence, 0.0)
        if window_id in accepted and accepted[window_id]["confidence"] >= confidence:
            continue
        allowed_frames = set(requested[window_id]["evidence_frames"])
        proposed_frames = item.get("evidence_frames", []) if isinstance(item.get("evidence_frames"), list) else []
        evidence_frames = sorted({int(value) for value in proposed_frames if str(value).lstrip("-").isdigit() and int(value) in allowed_frames})
        entry_nouns = item.get("object_nouns", []) if isinstance(item.get("object_nouns"), list) else []
        nouns.extend(entry_nouns)
        accepted[window_id] = {
            **requested[window_id],
            "raw_state": _validated_response_state(item.get("state"), window_id),
            "confidence": confidence,
            "reason": str(item.get("reason") or "Qwen 未提供原因")[:500],
            "motion": _bounded_confidence(item.get("motion"), 0.0),
            "contact": _bounded_confidence(item.get("contact"), 0.0),
            "object_nouns": _deduplicate_terms(entry_nouns, 30),
            "cited_evidence_frames": evidence_frames,
            "readable_frame_count": int(readable.get(window_id, 0) or 0),
        }
    missing = [window["window_id"] for window in windows if window["window_id"] not in accepted]
    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(f"Qwen response omitted {len(missing)} requested window(s): {preview}")
    output = [accepted[window["window_id"]] for window in windows]
    return output, _deduplicate_terms(nouns), warnings


def _coverage_bounds(samples: list[dict], frame_count: int) -> list[tuple[int, int]]:
    if not samples:
        return []
    bounds = []
    for index, sample in enumerate(samples):
        anchor = int(sample["anchor_frame"])
        start = 0 if index == 0 else (int(samples[index - 1]["anchor_frame"]) + anchor) // 2 + 1
        end = frame_count - 1 if index == len(samples) - 1 else (anchor + int(samples[index + 1]["anchor_frame"])) // 2
        bounds.append((max(0, start), min(frame_count - 1, max(start, end))))
    return bounds


def _boolean_runs(values: list[bool]) -> list[tuple[int, int, bool]]:
    output: list[tuple[int, int, bool]] = []
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[end] == values[start]:
            end += 1
        output.append((start, end, values[start]))
        start = end
    return output


def _smooth_samples(
    samples: list[dict],
    frame_count: int,
    fps: float,
    confidence_threshold: float,
    max_gap_seconds: float,
    min_valid_seconds: float,
) -> list[dict]:
    if not samples:
        return []
    output = [dict(item) for item in samples]
    bounds = _coverage_bounds(output, frame_count)
    values = [item["raw_state"] == "valid" and float(item["confidence"]) >= confidence_threshold for item in output]
    initial = list(values)

    if max_gap_seconds > 0:
        for start, end, is_valid in _boolean_runs(values):
            if is_valid or start == 0 or end == len(values) or not values[start - 1] or not values[end]:
                continue
            duration = (bounds[end - 1][1] - bounds[start][0] + 1) / max(0.01, fps)
            if duration <= max_gap_seconds:
                values[start:end] = [True] * (end - start)

    if min_valid_seconds > 0:
        for start, end, is_valid in _boolean_runs(values):
            if not is_valid:
                continue
            duration = (bounds[end - 1][1] - bounds[start][0] + 1) / max(0.01, fps)
            if duration < min_valid_seconds:
                values[start:end] = [False] * (end - start)

    for index, (sample, valid) in enumerate(zip(output, values)):
        sample["state"] = "valid" if valid else "invalid"
        sample["coverage_start_frame"], sample["coverage_end_frame"] = bounds[index]
        sample["filter_action"] = None
        if valid and not initial[index]:
            sample["filter_action"] = "filled_short_gap"
            sample["filtered_reason"] = "时间滤波填补有效动作内部的短暂漏判"
        elif not valid and initial[index]:
            sample["filter_action"] = "dropped_short_valid_run"
            sample["filtered_reason"] = "时间滤波移除过短的有效动作孤立段"
        elif sample["raw_state"] == "uncertain" or float(sample["confidence"]) < confidence_threshold:
            sample["filter_action"] = "uncertain_to_invalid"
            sample["filtered_reason"] = "证据或置信度不足，保守归为无效"
    return output


def _segments(samples: list[dict], frame_count: int, fps: float) -> list[dict]:
    if not samples:
        return []
    output = []
    start = 0
    while start < len(samples):
        end = start + 1
        while end < len(samples) and samples[end]["state"] == samples[start]["state"]:
            end += 1
        group = samples[start:end]
        start_frame = int(group[0]["coverage_start_frame"])
        end_frame = int(group[-1]["coverage_end_frame"])
        reasons = [str(item.get("filtered_reason") or item.get("reason") or "") for item in group]
        reason = Counter(value for value in reasons if value).most_common(1)
        output.append({
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": round(start_frame / max(0.01, fps), 4),
            "end_time": round(end_frame / max(0.01, fps), 4),
            "state": group[0]["state"],
            "confidence": round(sum(float(item.get("confidence", 0.0)) for item in group) / len(group), 4),
            "reason": reason[0][0] if reason else "Qwen 时间窗口判定",
            "sample_count": len(group),
            "source_window_ids": [item["window_id"] for item in group],
        })
        start = end
    if output:
        output[0]["start_frame"] = 0
        output[0]["start_time"] = 0.0
        output[-1]["end_frame"] = max(0, frame_count - 1)
        output[-1]["end_time"] = round(max(0, frame_count - 1) / max(0.01, fps), 4)
    return output


def _call_with_heartbeat(call, progress, progress_value: float, label: str):
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-trim-api") as executor:
        future = executor.submit(call)
        while True:
            try:
                return future.result(timeout=5.0)
            except FutureTimeoutError:
                if future.done():
                    return future.result()
                elapsed = int(time.monotonic() - started)
                progress(progress_value, f"{label} · 已等待 {elapsed} 秒，Qwen API 仍在运行")


def analyze_qwen_action_trim(
    dataset_id: str,
    manifest: dict,
    episode: dict,
    media: dict,
    request: QwenTrimRequest,
    progress,
) -> dict:
    if not registry.has_vlm:
        raise RuntimeError("请先配置 Qwen-VLM API")
    fps = float(media.get("fps", 0.0) or 0.0)
    frame_count = int(media.get("frame_count", 0) or 0)
    if fps <= 0 or frame_count <= 0:
        raise ValueError("所选视频缺少有效帧率或帧数")
    source_fingerprint = _source_video_fingerprint(media)
    media_context = {
        **media,
        "dataset_id": dataset_id,
        "episode_id": episode["id"],
    }
    windows = _sample_windows(
        frame_count,
        fps,
        request.sample_fps,
        request.window_seconds,
        request.frames_per_window,
    )
    if not windows:
        raise RuntimeError("没有可供 Qwen 分析的视频帧")
    batch_size = min(request.windows_per_request, max(1, MAX_IMAGES_PER_REQUEST // request.frames_per_window))
    batches = [windows[index:index + batch_size] for index in range(0, len(windows), batch_size)]
    verdicts: list[dict] = []
    nouns: list[Any] = []
    warnings: list[str] = []
    successful_batches = 0
    progress(4, f"准备 {len(windows)} 个时间窗口，共 {len(batches)} 批 Qwen 请求")
    for position, batch in enumerate(batches):
        base_progress = 8 + 76 * position / max(1, len(batches))
        label = f"Qwen API 正在分析第 {position + 1}/{len(batches)} 批，耗时数分钟属于正常情况"
        progress(base_progress, label)
        try:
            raw = _call_with_heartbeat(
                lambda current=batch: _request_window_batch(media_context, current),
                progress,
                base_progress,
                label,
            )
            batch_verdicts, batch_nouns, batch_warnings = _validate_batch_response(raw, batch)
            successful_batches += 1
            verdicts.extend(batch_verdicts)
            nouns.extend(batch_nouns)
            warnings.extend(batch_warnings)
        except Exception as exc:
            raise RuntimeError(
                f"Qwen batch {position + 1}/{len(batches)} failed; the entire Episode was aborted without writing a trim: {exc}"
            ) from exc
        progress(8 + 76 * (position + 1) / max(1, len(batches)), f"Qwen 时间窗口分析 {position + 1}/{len(batches)} 批完成")
    if successful_batches != len(batches):
        raise RuntimeError("Qwen did not complete every requested batch; the Episode was not committed")

    verdicts.sort(key=lambda item: int(item["anchor_frame"]))
    filtered = _smooth_samples(
        verdicts,
        frame_count,
        fps,
        request.confidence_threshold,
        request.max_gap_seconds,
        request.min_valid_seconds,
    )
    segments = _segments(filtered, frame_count, fps)
    valid_frames = sum(item["end_frame"] - item["start_frame"] + 1 for item in segments if item["state"] == "valid")
    behavior_context = _load_behavior_context(dataset_id, episode["id"])
    object_nouns = _deduplicate_terms([*behavior_context.get("primary_terms", []), *nouns])
    document = {
        "schema": QWEN_ACTION_TRIM_SCHEMA,
        "dataset_id": dataset_id,
        "episode_id": episode["id"],
        "created_at": _now(),
        "source_policy": "Source media remains read-only; this .alice file records a proposed valid/invalid timeline.",
        "method": "qwen_temporal_window_classification",
        "provider": registry.status().get("vlm"),
        "source_video": {
            "file_id": media.get("file_id"),
            "stream_name": media.get("stream_name"),
            "relative_path": media.get("relative_path"),
            "fps": fps,
            "frame_count": frame_count,
            "fingerprint": source_fingerprint,
        },
        "behavior_context": behavior_context,
        "object_nouns": object_nouns,
        "config": {
            "sample_fps": request.sample_fps,
            "window_seconds": request.window_seconds,
            "frames_per_window": request.frames_per_window,
            "windows_per_request": batch_size,
            "confidence_threshold": request.confidence_threshold,
            "max_gap_seconds": request.max_gap_seconds,
            "min_valid_seconds": request.min_valid_seconds,
            "uncertain_policy": "invalid",
        },
        "summary": {
            "window_count": len(filtered),
            "request_batch_count": len(batches),
            "successful_batch_count": successful_batches,
            "segment_count": len(segments),
            "valid_segment_count": sum(item["state"] == "valid" for item in segments),
            "invalid_segment_count": sum(item["state"] == "invalid" for item in segments),
            "valid_frame_count": valid_frames,
            "invalid_frame_count": max(0, frame_count - valid_frames),
            "gap_filled_window_count": sum(item.get("filter_action") == "filled_short_gap" for item in filtered),
            "short_valid_dropped_window_count": sum(item.get("filter_action") == "dropped_short_valid_run" for item in filtered),
            "uncertain_window_count": sum(item.get("raw_state") == "uncertain" for item in filtered),
        },
        "segments": segments,
        "samples": filtered,
        "warnings": warnings[:100],
    }
    progress(91, "写入 .alicePD Qwen 有效/无效片段与待应用记录")
    path = _artifact_path(dataset_id, episode["id"])
    previous_artifact = path.read_bytes() if path.is_file() else None
    _assert_source_unchanged(media_context, source_fingerprint, "before artifact write")
    try:
        _write_atomic(path, document)
        _assert_source_unchanged(media_context, source_fingerprint, "after artifact write")
        change = record_change(
            dataset_id,
            "qwen_action_trim",
            episode["id"],
            f"Qwen action trim: {episode['name']}",
            [path],
            document["summary"],
            [str(media.get("relative_path") or "")],
        )
    except Exception:
        if previous_artifact is None:
            path.unlink(missing_ok=True)
        else:
            _write_bytes_atomic(path, previous_artifact)
        raise
    document["artifact_path"] = str(path)
    document["change"] = {"id": change["id"], "status": change["status"], "revision": change["revision"]}
    progress(99, "Qwen 剪切建议已保存到 .alicePD，源数据集未修改")
    return document


def load_qwen_action_trim(dataset_id: str, episode_id: str) -> dict | None:
    path = _artifact_path(dataset_id, episode_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


class QwenTrimJobManager:
    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="qwen-action-trim")
        self._jobs: dict[str, dict] = {}
        self._active_episode_jobs: dict[tuple[str, str], str] = {}
        self._lock = threading.RLock()

    def _reserve_episodes(self, job_id: str, dataset_id: str, episode_ids: list[str]) -> None:
        with self._lock:
            conflicts = [
                episode_id
                for episode_id in episode_ids
                if (dataset_id, episode_id) in self._active_episode_jobs
            ]
            if conflicts:
                preview = ", ".join(conflicts[:8])
                raise ValueError(f"Qwen trim is already queued or running for Episode(s): {preview}")
            for episode_id in episode_ids:
                self._active_episode_jobs[(dataset_id, episode_id)] = job_id

    def _release_episodes(self, job_id: str) -> None:
        with self._lock:
            owned = [key for key, owner in self._active_episode_jobs.items() if owner == job_id]
            for key in owned:
                self._active_episode_jobs.pop(key, None)

    def submit(self, dataset_id: str, request: QwenTrimRequest) -> dict:
        if not registry.has_vlm:
            raise RuntimeError("请先配置 Qwen-VLM API")
        manifest = get_manifest(dataset_id)
        episodes = {item["id"]: item for item in manifest.get("episodes", [])}
        episode_ids = list(episodes) if request.all_episodes else list(dict.fromkeys(request.episode_ids))
        if not episode_ids:
            raise ValueError("请选择至少一个 Episode，或启用 all_episodes")
        missing = [episode_id for episode_id in episode_ids if episode_id not in episodes]
        if missing:
            raise KeyError(missing[0])
        selected_media: dict[str, dict] = {}
        for episode_id in episode_ids:
            media_file_id = str(request.media_file_ids.get(episode_id) or "").strip()
            if not media_file_id:
                raise ValueError(f"{episodes[episode_id]['name']} 必须明确指定一个视频流")
            try:
                media = episode_media(episodes[episode_id], media_file_id)
            except KeyError as exc:
                raise ValueError(f"{episodes[episode_id]['name']} 的视频流不存在: {media_file_id}") from exc
            if int(media.get("frame_count", 0) or 0) <= 0:
                raise ValueError(f"{episodes[episode_id]['name']} 的所选视频没有可读帧")
            selected_media[episode_id] = media

        job_id = uuid.uuid4().hex
        job = {
            "id": job_id,
            "kind": "qwen_action_trim",
            "status": "queued",
            "progress": 0,
            "message": f"Qwen 剪切任务已进入后台队列 · {len(episode_ids)} Episodes",
            "dataset_id": dataset_id,
            "scope": "all" if request.all_episodes else "selected",
            "episode_count": len(episode_ids),
            "completed_count": 0,
            "created_at": _now(),
            "updated_at": _now(),
            "result": None,
            "error": None,
        }
        self._reserve_episodes(job_id, dataset_id, episode_ids)
        with self._lock:
            self._jobs[job_id] = job
        try:
            self._executor.submit(self._run, job_id, dataset_id, episode_ids, selected_media, request)
        except Exception:
            with self._lock:
                self._jobs.pop(job_id, None)
            self._release_episodes(job_id)
            raise
        return deepcopy(job)

    def get(self, job_id: str) -> dict:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return deepcopy(self._jobs[job_id])

    def list(self, dataset_id: str, active_only: bool = False) -> list[dict]:
        active_statuses = {"queued", "running"}
        with self._lock:
            jobs = [
                deepcopy(job)
                for job in self._jobs.values()
                if job.get("dataset_id") == dataset_id
                and (not active_only or job.get("status") in active_statuses)
            ]
        jobs.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return jobs

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            self._jobs[job_id].update(changes, updated_at=_now())

    def _run(
        self,
        job_id: str,
        dataset_id: str,
        episode_ids: list[str],
        selected_media: dict[str, dict],
        request: QwenTrimRequest,
    ) -> None:
        try:
            manifest = get_manifest(dataset_id)
            episodes = {item["id"]: item for item in manifest.get("episodes", [])}
            results = []
            failures = []
            total = len(episode_ids)
            self._update(job_id, status="running", started_at=_now(), progress=1, message=f"Qwen 后台线程已启动 · 0/{total}")
            for position, episode_id in enumerate(episode_ids):
                episode = episodes[episode_id]
                base = position / total * 100.0
                span = 100.0 / total

                def episode_progress(value: float, message: str) -> None:
                    scaled = min(99.0, base + span * max(0.0, min(100.0, float(value))) / 100.0)
                    self._update(job_id, progress=round(scaled, 1), message=f"{episode['name']} · {message} · {position + 1}/{total}")

                try:
                    payload = analyze_qwen_action_trim(
                        dataset_id,
                        manifest,
                        episode,
                        selected_media[episode_id],
                        request,
                        episode_progress,
                    )
                    results.append({
                        "episode_id": episode_id,
                        "episode_name": episode["name"],
                        "status": "completed",
                        "media_file_id": selected_media[episode_id].get("file_id"),
                        "stream_name": selected_media[episode_id].get("stream_name"),
                        "valid_frame_count": payload["summary"]["valid_frame_count"],
                        "invalid_frame_count": payload["summary"]["invalid_frame_count"],
                        "artifact_path": payload["artifact_path"],
                        "change_id": payload["change"]["id"],
                    })
                except Exception as exc:
                    failures.append({"episode_id": episode_id, "episode_name": episode["name"], "error": str(exc)})
                completed = position + 1
                self._update(job_id, completed_count=completed, progress=round(completed / total * 100.0, 1), message=f"Qwen 剪切后台处理 · {completed}/{total}")
            result = {
                "dataset_id": dataset_id,
                "operation": "qwen_action_trim",
                "scope": "all" if request.all_episodes else "selected",
                "episode_count": total,
                "completed_count": len(results),
                "failure_count": len(failures),
                "items": results,
                "failures": failures,
            }
            if failures and not results:
                self._update(job_id, status="failed", progress=100, finished_at=_now(), message=f"全部 {total} 个 Episode 处理失败", result=result, error=failures[0]["error"])
            else:
                message = f"Qwen 剪切任务完成 · {len(results)}/{total}"
                if failures:
                    message += f" · {len(failures)} 个失败"
                self._update(job_id, status="complete", progress=100, finished_at=_now(), message=message, result=result)
        except Exception as exc:
            self._update(job_id, status="failed", finished_at=_now(), message=str(exc), error=str(exc), trace=traceback.format_exc(limit=8))
        finally:
            self._release_episodes(job_id)


qwen_trim_jobs = QwenTrimJobManager()
