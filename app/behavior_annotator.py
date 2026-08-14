from __future__ import annotations

import hashlib
import json
import os
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .behavior_boundary_refiner import joint_motion_change_score, load_episode_joint_pose, refine_behavior_boundaries
from .behavior_prompt import (
    META_ACTION_TRANSLATIONS,
    TRI_LEVEL_PROTOCOL_SCHEMA,
    TRI_LEVEL_PROTOCOL_VERSION,
    canonical_meta_action,
)
from .full_run import full_run_stage_dir
from .job_control import CancellableJobMixin, JobCancelled
from .models import registry
from .qwen_trim import _source_fingerprints_match, _source_video_fingerprint
from .schemas import BehaviorAnnotationRequest
from .sensor_alignment import ensure_episode_time_sync, retime_sensor_alignment
from .storage import dataset_artifact_dir, episode_media, get_episode, read_frame, record_change, require_media_eligibility, slugify
from .video_smoothing import preferred_smoothed_media


BEHAVIOR_SCHEMA = "alice/vlm-behavior/v1"
TARGET_SCHEMA = "alice/behavior-targets/v1"
BEHAVIOR_ARTIFACT_VERSION = 5

BEHAVIOR_SAMPLING_STRATEGY = "windowed_adaptive_multi_image_v1"
BEHAVIOR_WINDOW_SECONDS = 20.0
BEHAVIOR_WINDOW_OVERLAP_SECONDS = 3.0
BEHAVIOR_BASE_SAMPLE_FPS = 1.5
BEHAVIOR_EVENT_SAMPLE_FPS = 12.0
BEHAVIOR_EVENT_RADIUS_SECONDS = 0.75
BEHAVIOR_MIN_IMAGES_PER_WINDOW = 12
BEHAVIOR_MIN_IMAGES_PER_RANGE = 6

PHASE_LABELS = (
    "idle", "observe", "reach", "grasp", "lift", "transport", "align", "place",
    "release", "withdraw", "manipulate", "inspect", "unknown",
)
_PHASE_LABEL_SET = set(PHASE_LABELS)
_PHASE_ALIASES = {
    "approach": "reach",
    "contact": "grasp",
    "carry": "transport",
    "move": "transport",
    "position": "align",
    "retract": "withdraw",
    "retreat": "withdraw",
    "wait": "idle",
    "other": "unknown",
    "空闲": "idle",
    "静止": "idle",
    "等待": "idle",
    "观察": "observe",
    "接近": "reach",
    "伸手": "reach",
    "抓取": "grasp",
    "接触": "grasp",
    "抬起": "lift",
    "搬运": "transport",
    "移动": "transport",
    "对齐": "align",
    "放置": "place",
    "松手": "release",
    "释放": "release",
    "撤回": "withdraw",
    "收回": "withdraw",
    "操作": "manipulate",
    "检查": "inspect",
    "未知": "unknown",
}

_META_ACTION_PHASES = {
    "Idle": "idle", "Observe": "observe", "Reach": "reach", "Withdraw": "withdraw",
    "Align": "align", "Inspect": "inspect",
    "Grasp": "grasp", "Hold": "grasp", "Pinch": "grasp", "Clip": "grasp",
    "Suction": "grasp", "Catch": "grasp", "TakeOver": "grasp",
    "Lift": "lift", "Raise height": "lift",
    "Transport": "transport", "Carry": "transport", "Move": "transport", "HandOver": "transport",
    "Place": "place", "Drop": "place", "Stack": "place", "Hang": "place",
    "Release": "release",
    "Scan": "inspect",
    "Other": "unknown",
}

_PHASE_META_ACTIONS = {
    "idle": "Idle",
    "observe": "Observe",
    "reach": "Reach",
    "grasp": "Grasp",
    "lift": "Lift",
    "transport": "Transport",
    "align": "Align",
    "place": "Place",
    "release": "Release",
    "withdraw": "Withdraw",
    "manipulate": "Other",
    "inspect": "Inspect",
    "unknown": "Other",
}

BUILTIN_BEHAVIOR_CATEGORIES = [
    {"label": "pick", "task": "pick up an object", "verbs": ["grasp", "lift", "pick"], "objects": [], "descriptions": ["Pick up the target object."]},
    {"label": "place", "task": "place an object", "verbs": ["place", "put", "set"], "objects": [], "descriptions": ["Place the held object at the target location."]},
    {"label": "push", "task": "push an object", "verbs": ["push", "slide"], "objects": [], "descriptions": ["Push the target object across the surface."]},
    {"label": "pull", "task": "pull an object", "verbs": ["drag", "pull"], "objects": [], "descriptions": ["Pull the target object toward the hand or gripper."]},
    {"label": "open", "task": "open an object", "verbs": ["open", "uncover"], "objects": [], "descriptions": ["Open the target container, door, drawer, or lid."]},
    {"label": "close", "task": "close an object", "verbs": ["close", "cover", "shut"], "objects": [], "descriptions": ["Close the target container, door, drawer, or lid."]},
    {"label": "insert", "task": "insert an object", "verbs": ["fit", "insert", "plug"], "objects": [], "descriptions": ["Insert the target object into the destination."]},
    {"label": "remove", "task": "remove an object", "verbs": ["extract", "remove", "take out"], "objects": [], "descriptions": ["Remove the target object from its container or fixture."]},
    {"label": "pour", "task": "pour contents", "verbs": ["decant", "pour", "transfer"], "objects": [], "descriptions": ["Pour the contents from the source container into the destination."]},
    {"label": "press", "task": "press a control", "verbs": ["press", "push", "tap"], "objects": [], "descriptions": ["Press the target button, switch, or control."]},
    {"label": "turn", "task": "turn an object", "verbs": ["rotate", "screw", "turn", "twist"], "objects": [], "descriptions": ["Turn or rotate the target object."]},
    {"label": "wipe", "task": "wipe a surface", "verbs": ["clean", "rub", "wipe"], "objects": [], "descriptions": ["Wipe the target surface with the tool or hand."]},
    {"label": "handover", "task": "hand over an object", "verbs": ["give", "hand over", "receive"], "objects": [], "descriptions": ["Transfer the target object between hands, grippers, or agents."]},
]


def _value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray):
        return [_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _ontology_document(categories: list[dict], *, source: str, root: str, **metadata: str) -> dict:
    serialized = json.dumps(categories, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "source": source,
        "root": root,
        "category_count": len(categories),
        "fingerprint": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "categories": deepcopy(categories),
        **metadata,
    }


def _builtin_ontology(requested_root: str, reason: str) -> dict:
    metadata = {"fallback_reason": reason}
    if requested_root:
        metadata["requested_root"] = requested_root
    return _ontology_document(
        BUILTIN_BEHAVIOR_CATEGORIES,
        source="builtin",
        root="builtin:alice-blue/manipulation-v1",
        **metadata,
    )


@lru_cache(maxsize=4)
def load_behavior_ontology(root_text: str = "") -> dict:
    requested_root = str(root_text or "").strip()
    if not requested_root:
        return _builtin_ontology("", "not_configured")
    root = Path(requested_root).expanduser().resolve()
    if not root.is_dir():
        return _builtin_ontology(str(root), "directory_unavailable")

    import h5py

    categories = []
    for folder in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda item: item.name.casefold()):
        descriptions: list[str] = []
        verbs: set[str] = set()
        objects: set[str] = set()
        task_name = folder.name
        samples = [folder / f"{index}.hdf5" for index in range(6) if (folder / f"{index}.hdf5").is_file()]
        if not samples:
            fallback = next(folder.glob("*.hdf5"), None)
            samples = [fallback] if fallback is not None else []
        for sample in samples:
            try:
                with h5py.File(sample, "r") as handle:
                    task_name = str(_value(handle.attrs.get("task", task_name)))
                    for key in ("llm_description", "llm_description2"):
                        description = str(_value(handle.attrs.get(key, ""))).strip()
                        if description and description.casefold() != "none" and description not in descriptions:
                            descriptions.append(description)
                    verbs.update(str(item).strip().casefold() for item in (_value(handle.attrs.get("llm_verbs", [])) or []) if str(item).strip())
                    objects.update(str(item).strip().casefold() for item in (_value(handle.attrs.get("llm_objects", [])) or []) if str(item).strip())
            except (OSError, ValueError):
                continue
        categories.append({
            "label": folder.name,
            "task": task_name,
            "verbs": sorted(verbs),
            "objects": sorted(objects),
            "descriptions": descriptions[:8],
        })
    readable_categories = [item for item in categories if item["descriptions"] or item["verbs"] or item["objects"]]
    if not readable_categories:
        return _builtin_ontology(str(root), "no_readable_hdf5_annotations")
    return _ontology_document(readable_categories, source="external_hdf5", root=str(root))


# Kept for callers from earlier builds; Part1 is now an optional external ontology.
load_part1_ontology = load_behavior_ontology


def _annotation_path(dataset_id: str, episode_id: str, run_id: str | None = None) -> Path:
    root = full_run_stage_dir(dataset_id, run_id, episode_id, "behavior") if run_id else dataset_artifact_dir(dataset_id, "behavior-annotations")
    return root / f"{slugify(episode_id)}.behavior.alice"


def _target_path(dataset_id: str, episode_id: str, run_id: str | None = None) -> Path:
    root = full_run_stage_dir(dataset_id, run_id, episode_id, "behavior") if run_id else dataset_artifact_dir(dataset_id, "behavior-targets")
    return root / f"{slugify(episode_id)}.targets.alice"


def _media_fingerprint(media: dict) -> dict | None:
    """Return a stable source identity when the selected media is a file.

    Behavior annotation also supports image sequences. Those do not have a
    single video file, so the legacy descriptor checks below remain the
    fallback for that media type.
    """
    path = Path(str(media.get("path") or "")).expanduser()
    if path.is_file():
        return _source_video_fingerprint({"path": str(path)})
    return None


def media_fingerprint_matches(expected: dict | None, media: dict) -> bool:
    if not isinstance(expected, dict):
        return False
    current = _media_fingerprint(media)
    return current is not None and _source_fingerprints_match(expected, current)


def _assert_media_unchanged(media: dict, expected: dict | None, stage: str) -> None:
    if expected is None:
        return
    current = _media_fingerprint(media)
    if current is None or not _source_fingerprints_match(expected, current):
        raise RuntimeError(f"The behavior-annotation source changed {stage}; no artifact was committed")


def _read_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _legacy_source_matches(annotation: dict, media: dict, episode: dict) -> bool:
    """Validate artifacts written before source fingerprints were introduced.

    Old behavior files contain the sampled frame list and, for most datasets,
    the selected media id. Requiring the last sampled frame to be the current
    final frame prevents reusing an annotation after a truncated/replaced
    video while preserving existing .alicePD annotations.
    """
    sampling = annotation.get("sampling") or {}
    expected_id = str(sampling.get("media_file_id") or "")
    actual_id = str(media.get("file_id") or "")
    if expected_id and actual_id and expected_id != actual_id:
        return False
    # Early behavior artifacts did not persist media_file_id. The annotator
    # always selected the Episode primary stream, so use that stable choice
    # instead of invalidating an otherwise complete legacy result merely
    # because the Episode now exposes additional camera streams.
    frames = []
    for value in sampling.get("frames") or []:
        try:
            frames.append(int(value))
        except (TypeError, ValueError):
            continue
    frame_count = int(media.get("frame_count") or episode.get("frame_count") or 0)
    stored_ranges = [
        (_safe_int(item.get("start_frame")), _safe_int(item.get("end_frame")))
        for item in sampling.get("allowed_ranges") or []
        if isinstance(item, dict)
    ]
    expected_last = max((end for _start, end in stored_ranges), default=frame_count - 1)
    if frame_count <= 0 or not frames or max(frames) != expected_last:
        return False
    expected_stream = str(sampling.get("stream_name") or "")
    actual_stream = str(media.get("stream_name") or "")
    return not expected_stream or not actual_stream or expected_stream == actual_stream


def _segments_follow_phase_protocol(annotation: dict, expected_frame_count: int | None = None) -> bool:
    segments = annotation.get("segments")
    if not isinstance(segments, list) or not segments:
        return False
    cursor = 0
    for item in segments:
        if (
            not isinstance(item, dict)
            or item.get("phase_label") not in _PHASE_LABEL_SET
            or item.get("label") != item.get("phase_label")
            or item.get("boundary_source") not in {"vlm", "joint_refined", "curation_precheck"}
        ):
            return False
        start = _safe_int(item.get("start_frame"), -1)
        end = _safe_int(item.get("end_frame"), -1)
        if start != cursor or end < start:
            return False
        cursor = end + 1
    if expected_frame_count is not None and expected_frame_count > 0 and cursor != expected_frame_count:
        return False
    return True


def _payload_is_valid(annotation: dict, target: dict, dataset_id: str, episode: dict, run_id: str | None = None) -> bool:
    if annotation.get("schema") != BEHAVIOR_SCHEMA or target.get("schema") != TARGET_SCHEMA:
        return False
    try:
        artifact_version = int(annotation.get("artifact_version") or 0)
    except (TypeError, ValueError):
        return False
    if artifact_version != BEHAVIOR_ARTIFACT_VERSION:
        return False
    protocol = annotation.get("annotation_protocol") or {}
    if protocol.get("version") != TRI_LEVEL_PROTOCOL_VERSION or protocol.get("schema") != TRI_LEVEL_PROTOCOL_SCHEMA:
        return False
    if str(annotation.get("dataset_id")) != str(dataset_id) or str(target.get("dataset_id")) != str(dataset_id):
        return False
    if str(annotation.get("episode_id")) != str(episode.get("id")) or str(target.get("episode_id")) != str(episode.get("id")):
        return False
    frame_count = _behavior_timeline_frame_count(annotation, episode)
    if not _segments_follow_phase_protocol(annotation, frame_count):
        return False
    if not _tri_level_fields_are_valid(annotation, frame_count):
        return False
    if not isinstance(target.get("primary_terms"), list):
        return False
    if run_id and str(annotation.get("full_run_id") or "") != str(run_id):
        return False
    source_annotation = str(target.get("source_annotation") or "")
    return source_annotation in {"", _annotation_path(dataset_id, str(episode.get("id")), run_id).name}


def behavior_annotation_status(
    dataset_id: str,
    manifest: dict,
    episode: dict,
    *,
    source_media_file_id: str | None = None,
    analysis_media: dict | None = None,
    analysis_frame_ranges: list[tuple[int, int]] | None = None,
) -> dict:
    """Describe whether an existing VLM result can be reused without Qwen.

    This function is intentionally read-only with respect to the source
    dataset. It only reads the two sidecar artifacts and the source media
    identity; callers decide whether to schedule a new task.
    """
    annotation_path = _annotation_path(dataset_id, episode["id"])
    target_path = _target_path(dataset_id, episode["id"])
    base = {
        "reusable": False,
        "reason": "missing_behavior_annotation",
        "validation": "none",
        "artifact_path": str(annotation_path),
        "target_path": str(target_path),
        "payload": None,
    }
    if not annotation_path.is_file() or not target_path.is_file():
        base["reason"] = "missing_behavior_or_target_artifact"
        return base
    annotation = _read_json(annotation_path)
    target = _read_json(target_path)
    if annotation is None or target is None:
        base["reason"] = "unreadable_behavior_artifact"
        return base
    if not _payload_is_valid(annotation, target, dataset_id, episode):
        base["reason"] = "invalid_behavior_artifact"
        return base
    stored_source_id = str((annotation.get("source_video") or {}).get("file_id") or "")
    if source_media_file_id and stored_source_id and stored_source_id != str(source_media_file_id):
        base["reason"] = "requested_media_mismatch"
        return base
    try:
        media = episode_media(episode, (annotation.get("source_video") or {}).get("file_id") or (annotation.get("sampling") or {}).get("media_file_id") or episode.get("primary_media_file_id"))
    except KeyError:
        base["reason"] = "source_media_missing"
        return base
    source = annotation.get("source_video") or {}
    expected_fingerprint = source.get("fingerprint")
    if isinstance(expected_fingerprint, dict):
        try:
            current = _media_fingerprint(media)
        except (OSError, RuntimeError, ValueError):
            current = None
        if current is None or not _source_fingerprints_match(expected_fingerprint, current):
            base["reason"] = "source_media_changed"
            return base
        validation = "source_fingerprint"
    elif not _legacy_source_matches(annotation, media, episode):
        base["reason"] = "legacy_source_descriptor_mismatch"
        return base
    else:
        validation = "legacy_frame_descriptor"
    analysis_source = annotation.get("analysis_video") or {}
    expected_analysis_fingerprint = analysis_source.get("fingerprint")
    if analysis_source.get("kind") == "applied_video_smoothing" and isinstance(expected_analysis_fingerprint, dict):
        current_analysis, smoothing_document = preferred_smoothed_media(dataset_id, episode, media)
        if not smoothing_document:
            base["reason"] = "analysis_video_missing"
            return base
        try:
            current_analysis_fingerprint = _media_fingerprint(current_analysis)
        except (OSError, RuntimeError, ValueError):
            current_analysis_fingerprint = None
        if current_analysis_fingerprint is None or not _source_fingerprints_match(expected_analysis_fingerprint, current_analysis_fingerprint):
            base["reason"] = "analysis_video_changed"
            return base
    if analysis_media is not None:
        if str(analysis_source.get("file_id") or "") != str(analysis_media.get("file_id") or ""):
            base["reason"] = "analysis_media_mismatch"
            return base
        if _safe_int(analysis_source.get("frame_count")) != _safe_int(analysis_media.get("frame_count")):
            base["reason"] = "analysis_frame_count_mismatch"
            return base
        expected = analysis_source.get("fingerprint")
        if isinstance(expected, dict) and not media_fingerprint_matches(expected, analysis_media):
            base["reason"] = "analysis_video_changed"
            return base
    if analysis_frame_ranges is not None:
        expected_ranges = _normalize_frame_ranges(
            _safe_int((analysis_media or analysis_source).get("frame_count")),
            analysis_frame_ranges,
        )
        stored_ranges = (annotation.get("sampling") or {}).get("allowed_ranges")
        normalized_stored = [
            (_safe_int(item.get("start_frame")), _safe_int(item.get("end_frame")))
            for item in stored_ranges or []
            if isinstance(item, dict)
        ]
        if normalized_stored != expected_ranges:
            base["reason"] = "curation_ranges_changed"
            return base
    payload = _apply_dataset_task_fallback(deepcopy(annotation), manifest)
    payload["artifacts"] = {"behavior": str(annotation_path), "targets": str(target_path)}
    payload["reuse"] = {
        "reused": True,
        "status": "skipped",
        "reason": "existing_valid_annotation",
        "validation": validation,
        "artifact_path": str(annotation_path),
        "target_path": str(target_path),
    }
    base.update({"reusable": True, "reason": "existing_valid_annotation", "validation": validation, "payload": payload})
    return base


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _sample_indices(frame_count: int, count: int) -> list[int]:
    if frame_count <= 0:
        return []
    return sorted({int(round(value)) for value in np.linspace(0, frame_count - 1, min(count, frame_count))})


def _normalize_frame_ranges(frame_count: int, ranges: list[tuple[int, int]] | None) -> list[tuple[int, int]]:
    if ranges is None:
        return [(0, frame_count - 1)] if frame_count > 0 else []
    normalized: list[tuple[int, int]] = []
    for raw_start, raw_end in sorted(ranges):
        if frame_count <= 0:
            break
        start = max(0, min(frame_count - 1, int(raw_start)))
        end = max(start, min(frame_count - 1, int(raw_end)))
        if normalized and start <= normalized[-1][1] + 1:
            normalized[-1] = (normalized[-1][0], max(normalized[-1][1], end))
        else:
            normalized.append((start, end))
    return normalized


def _sample_indices_in_ranges(frame_count: int, count: int, ranges: list[tuple[int, int]] | None) -> list[int]:
    normalized = _normalize_frame_ranges(frame_count, ranges)
    total = sum(end - start + 1 for start, end in normalized)
    if total <= 0:
        return []
    offsets = sorted({int(round(value)) for value in np.linspace(0, total - 1, min(count, total))})
    indices: list[int] = []
    range_index = 0
    consumed = 0
    for offset in offsets:
        while range_index < len(normalized):
            start, end = normalized[range_index]
            length = end - start + 1
            if offset < consumed + length:
                indices.append(start + offset - consumed)
                break
            consumed += length
            range_index += 1
    return indices


def _uniform_indices(start: int, end: int, count: int) -> list[int]:
    if start > end or count <= 0:
        return []
    length = end - start + 1
    return sorted({
        start + int(round(value))
        for value in np.linspace(0, length - 1, min(length, count))
    })


def _plan_behavior_windows(
    frame_count: int,
    fps: float,
    ranges: list[tuple[int, int]] | None,
    *,
    window_seconds: float = BEHAVIOR_WINDOW_SECONDS,
    overlap_seconds: float = BEHAVIOR_WINDOW_OVERLAP_SECONDS,
) -> list[dict]:
    normalized = _normalize_frame_ranges(frame_count, ranges)
    window_frames = max(1, int(round(max(1.0, window_seconds) * max(0.01, fps))))
    overlap_frames = max(0, min(window_frames - 1, int(round(max(0.0, overlap_seconds) * max(0.01, fps)))))
    step = max(1, window_frames - overlap_frames)
    windows: list[dict] = []
    for range_index, (range_start, range_end) in enumerate(normalized):
        start = range_start
        while start <= range_end:
            end = min(range_end, start + window_frames - 1)
            if range_end - end <= overlap_frames:
                end = range_end
            windows.append({
                "window_id": f"range-{range_index + 1}-window-{len(windows) + 1}",
                "range_index": range_index,
                "range_start": range_start,
                "range_end": range_end,
                "start_frame": start,
                "end_frame": end,
            })
            if end >= range_end:
                break
            start += step
    for index, window in enumerate(windows):
        window["window_index"] = index
        window["window_count"] = len(windows)
    return windows


def _select_score_events(
    score: np.ndarray | None,
    start: int,
    end: int,
    fps: float,
    limit: int,
) -> list[int]:
    if score is None or limit <= 0 or start > end:
        return []
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    if not values.size:
        return []
    low = max(0, start)
    high = min(end, values.size - 1)
    if low > high:
        return []
    local = values[low:high + 1]
    finite = local[np.isfinite(local)]
    if not finite.size:
        return []
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)) * 1.4826)
    threshold = max(1e-10, median + 0.75 * mad)
    ordered = sorted(
        (
            (float(values[index]), index)
            for index in range(low, high + 1)
            if np.isfinite(values[index]) and float(values[index]) > threshold
        ),
        reverse=True,
    )
    separation = max(1, int(round(max(0.01, fps) * 0.35)))
    selected: list[int] = []
    for _value_at_frame, frame in ordered:
        if all(abs(frame - existing) >= separation for existing in selected):
            selected.append(frame)
            if len(selected) >= limit:
                break
    return sorted(selected)


def _evidence_event_frames(
    evidence: list[dict] | None,
    start: int,
    end: int,
    fps: float,
    limit: int,
) -> list[int]:
    candidates = []
    previous_state = ""
    for item in evidence or []:
        if not isinstance(item, dict):
            continue
        frame = _safe_int(item.get("frame"), -1)
        if frame < start or frame > end:
            continue
        motion = max(0.0, _safe_float(item.get("motion")))
        state = str(item.get("state") or "")
        transition_bonus = 1.0 if previous_state and state and state != previous_state else 0.0
        candidates.append((motion + transition_bonus, frame))
        if state:
            previous_state = state
    separation = max(1, int(round(max(0.01, fps) * 0.35)))
    selected: list[int] = []
    for score, frame in sorted(candidates, reverse=True):
        if score <= 0.0:
            continue
        if all(abs(frame - existing) >= separation for existing in selected):
            selected.append(frame)
            if len(selected) >= max(0, limit):
                break
    return sorted(selected)


def _visual_event_frames(
    indices: list[int],
    frame_cache: dict[int, np.ndarray | None],
    fps: float,
    limit: int,
) -> list[int]:
    differences: list[tuple[float, int]] = []
    previous_frame = None
    previous_index = None
    for index in indices:
        frame = frame_cache.get(index)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else np.asarray(frame)
        gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
        if previous_frame is not None and previous_index is not None:
            difference = float(np.mean(np.abs(gray - previous_frame)) / 255.0)
            differences.append((difference, int(round((previous_index + index) / 2.0))))
        previous_frame = gray
        previous_index = index
    if not differences or limit <= 0:
        return []
    values = np.asarray([item[0] for item in differences], dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)) * 1.4826)
    threshold = max(0.015, median + 0.5 * mad)
    separation = max(1, int(round(max(0.01, fps) * 0.35)))
    selected: list[int] = []
    for difference, frame in sorted(differences, reverse=True):
        if difference < threshold:
            continue
        if all(abs(frame - existing) >= separation for existing in selected):
            selected.append(frame)
            if len(selected) >= limit:
                break
    return sorted(selected)


def _adaptive_window_indices(
    window: dict,
    fps: float,
    maximum_images: int,
    frame_cache: dict[int, np.ndarray | None],
    frame_reader,
    *,
    joint_score: np.ndarray | None = None,
    sampling_evidence: list[dict] | None = None,
) -> tuple[list[int], dict]:
    start = int(window["start_frame"])
    end = int(window["end_frame"])
    length = end - start + 1
    cap = max(1, min(length, int(maximum_images)))
    minimum = min(cap, length, max(BEHAVIOR_MIN_IMAGES_PER_RANGE, BEHAVIOR_MIN_IMAGES_PER_WINDOW))
    duration = length / max(0.01, fps)
    desired_base = max(minimum, int(np.ceil(duration * BEHAVIOR_BASE_SAMPLE_FPS)) + 1)
    event_reserve = min(18, max(0, cap - minimum), cap // 3)
    base_count = min(length, max(minimum, min(desired_base, cap - event_reserve)))
    base_indices = _uniform_indices(start, end, base_count)
    for index in base_indices:
        if index not in frame_cache:
            frame_cache[index] = frame_reader(index)

    event_limit = max(1, min(8, cap // 7))
    joint_events = _select_score_events(joint_score, start, end, fps, event_limit)
    evidence_events = _evidence_event_frames(sampling_evidence, start, end, fps, event_limit)
    visual_events = _visual_event_frames(base_indices, frame_cache, fps, event_limit)
    events = list(dict.fromkeys([*joint_events, *evidence_events, *visual_events]))

    selected = set(base_indices)
    event_step = max(1, int(round(max(0.01, fps) / BEHAVIOR_EVENT_SAMPLE_FPS)))
    radius = max(event_step, int(round(BEHAVIOR_EVENT_RADIUS_SECONDS * max(0.01, fps))))
    offsets = [0]
    for distance in range(event_step, radius + 1, event_step):
        offsets.extend((-distance, distance))
    dense_candidates: list[int] = []
    for offset in offsets:
        for event in events:
            candidate = event + offset
            if start <= candidate <= end and candidate not in selected and candidate not in dense_candidates:
                dense_candidates.append(candidate)
    for candidate in dense_candidates:
        if len(selected) >= cap:
            break
        selected.add(candidate)
    if len(selected) < cap:
        for candidate in _uniform_indices(start, end, cap):
            selected.add(candidate)
            if len(selected) >= cap:
                break
    return sorted(selected), {
        "base_frame_count": len(base_indices),
        "event_frame_count": max(0, len(selected) - len(base_indices)),
        "joint_event_frames": joint_events,
        "quality_motion_event_frames": evidence_events,
        "visual_event_frames": visual_events,
    }


def _constrain_segments_to_ranges(
    segments: list[dict],
    ranges: list[tuple[int, int]],
    frame_count: int,
    fps: float,
) -> list[dict]:
    if frame_count <= 0:
        return []
    pieces: list[dict] = []
    for segment in segments:
        segment_start = int(segment.get("start_frame") or 0)
        segment_end = int(segment.get("end_frame") or segment_start)
        for allowed_start, allowed_end in ranges:
            start = max(segment_start, allowed_start)
            end = min(segment_end, allowed_end)
            if start <= end:
                pieces.append({**segment, "start_frame": start, "end_frame": end})
    pieces.sort(key=lambda item: (int(item["start_frame"]), int(item["end_frame"])))
    output: list[dict] = []
    cursor = 0

    def append_unknown(start: int, end: int) -> None:
        if start > end:
            return
        output.append({
            "start_frame": start,
            "end_frame": end,
            "phase_label": "unknown",
            "label": "unknown",
            "description": "Excluded by the S1-S5/C3 precheck before VLM sampling.",
            "confidence": 1.0,
            "primary_targets": [],
            "target_instance": "",
            "boundary_source": "curation_precheck",
        })

    for item in pieces:
        start = max(cursor, int(item["start_frame"]))
        end = int(item["end_frame"])
        append_unknown(cursor, start - 1)
        if start <= end:
            output.append({**item, "start_frame": start, "end_frame": end})
            cursor = end + 1
    append_unknown(cursor, frame_count - 1)
    for item in output:
        item["start_time"] = round(int(item["start_frame"]) / fps, 3)
        item["end_time"] = round(int(item["end_frame"]) / fps, 3)
    return output


def _list_value(value: Any) -> list:
    return value if isinstance(value, list) else []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _behavior_timeline_frame_count(payload: dict, episode: dict | None = None) -> int:
    for value in (
        (payload.get("timeline") or {}).get("frame_count"),
        (payload.get("analysis_video") or {}).get("frame_count"),
        (payload.get("source_video") or {}).get("frame_count"),
        (episode or {}).get("frame_count"),
    ):
        count = _safe_int(value)
        if count > 0:
            return count
    return 0


def behavior_analysis_context(
    dataset_id: str,
    manifest: dict,
    episode: dict,
    media_file_id: str | None = None,
) -> dict:
    """Bind standalone VLM annotation to the current media and non-red quality ranges."""
    source_media = episode_media(episode, media_file_id or episode.get("primary_media_file_id"))
    try:
        analysis_media, smoothing_document = preferred_smoothed_media(dataset_id, episode, source_media)
    except (KeyError, OSError, RuntimeError, ValueError):
        analysis_media, smoothing_document = source_media, None
    analysis_source_kind = "applied_video_smoothing" if smoothing_document else "source_video"
    try:
        from .projection_correction import preferred_projection_media

        projection_media, projection_document = preferred_projection_media(manifest, episode, source_media)
        if projection_document:
            analysis_media = projection_media
            analysis_source_kind = "applied_projection_correction"
    except (KeyError, OSError, RuntimeError, ValueError):
        pass
    report = None
    frame_ranges = None
    try:
        from .curation_pipeline import curation_vlm_ranges, load_curation_report

        report = load_curation_report(
            dataset_id,
            str(episode["id"]),
            str(source_media.get("file_id") or "") or None,
        )
        report_frame_count = _safe_int((report or {}).get("source_video", {}).get("frame_count"))
        if (
            report
            and report_frame_count == _safe_int(analysis_media.get("frame_count"))
        ):
            frame_ranges = curation_vlm_ranges(report)
            analysis_source_kind = "curation_non_rejected_segments"
    except (KeyError, OSError, RuntimeError, ValueError):
        report = None
        frame_ranges = None
    return {
        "source_media": source_media,
        "analysis_media": analysis_media,
        "analysis_source_kind": analysis_source_kind,
        "analysis_frame_ranges": frame_ranges,
        "curation_report": report,
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _confidence(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, _safe_float(value, default)))


def _apply_dataset_task_fallback(result: dict, manifest: dict) -> dict:
    if str(result.get("task_label") or "").strip().casefold() not in {"", "other", "unknown"}:
        return result
    candidate = str(manifest.get("name") or "").strip()[:80]
    if not candidate or candidate.casefold() in {"data", "dataset", "part1", "spm", "test", "unknown"}:
        return result
    tokens = [
        token
        for token in candidate.casefold().replace("-", " ").replace("_", " ").split()
        if len(token) >= 3 and token not in {"data", "dataset", "episode", "test"}
    ]
    evidence = " ".join([
        str(result.get("behavior_description") or ""),
        *[str(item) for item in result.get("object_nouns") or []],
        *[str(item.get("description") or "") for item in result.get("segments") or [] if isinstance(item, dict)],
    ]).casefold()
    if not tokens or not any(token in evidence for token in tokens):
        return result
    resolved = deepcopy(result)
    resolved["task_label"] = candidate
    resolved["task_label_source"] = "dataset_name_confirmed_by_vlm_content"
    warnings = [str(item) for item in resolved.get("warnings") or []]
    warnings.append(f"VLM returned other; resolved task_label to {candidate} because its content matched the dataset task name.")
    resolved["warnings"] = warnings[:30]
    return resolved


def _normalize_phase_label(value: Any) -> str:
    label = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    label = _PHASE_ALIASES.get(label, label)
    return label if label in _PHASE_LABEL_SET else "unknown"


def _phase_for_meta_action(skill: str) -> str:
    return _META_ACTION_PHASES.get(skill, "manipulate")


def _intervals_are_contiguous(items: Any, frame_count: int, *, require_skill: bool = False) -> bool:
    if not isinstance(items, list) or not items or frame_count <= 0:
        return False
    cursor = 0
    for item in items:
        if not isinstance(item, dict):
            return False
        if require_skill and item.get("skill") not in META_ACTION_TRANSLATIONS:
            return False
        start = _safe_int(item.get("start_frame"), -1)
        end = _safe_int(item.get("end_frame"), -1)
        if start != cursor or end < start:
            return False
        cursor = end + 1
    return cursor == frame_count


def _tri_level_fields_are_valid(annotation: dict, frame_count: int) -> bool:
    coarse = annotation.get("coarse")
    fine = annotation.get("fine")
    segments = annotation.get("segments")
    return bool(
        isinstance(coarse, dict)
        and str(coarse.get("summary") or "").strip()
        and _intervals_are_contiguous(annotation.get("medium"), frame_count)
        and _intervals_are_contiguous(fine, frame_count, require_skill=True)
        and isinstance(segments, list)
        and [(item.get("start_frame"), item.get("end_frame")) for item in fine]
        == [(item.get("start_frame"), item.get("end_frame")) for item in segments]
    )


def _normalize_tri_level_intervals(items: Any, frame_count: int, default_description: str) -> list[dict]:
    if frame_count <= 0:
        return []
    ordered = sorted(
        (dict(item) for item in _list_value(items)[:40] if isinstance(item, dict)),
        key=lambda item: (_safe_int(item.get("start_frame")), _safe_int(item.get("end_frame"))),
    )
    if not ordered:
        ordered = [{"start_frame": 0, "end_frame": frame_count - 1, "description": default_description}]
    splits: list[int] = []
    previous = 0
    for index, (left, right) in enumerate(zip(ordered, ordered[1:])):
        proposed = int(round(((_safe_int(left.get("end_frame")) + 1) + _safe_int(right.get("start_frame"))) / 2.0))
        minimum = previous + 1
        maximum = frame_count - (len(ordered) - index - 1)
        split = max(minimum, min(proposed, maximum))
        splits.append(split)
        previous = split
    output = []
    for index, item in enumerate(ordered):
        output.append({
            "start_frame": 0 if index == 0 else splits[index - 1],
            "end_frame": frame_count - 1 if index == len(ordered) - 1 else splits[index] - 1,
            "description": str(item.get("description") or default_description)[:800],
        })
    return output


def _fine_from_segments(segments: list[dict]) -> list[dict]:
    output = []
    for item in segments:
        skill = canonical_meta_action(item.get("skill"))
        fine = {
            "start_frame": int(item.get("start_frame") or 0),
            "end_frame": int(item.get("end_frame") or item.get("start_frame") or 0),
            "description": str(item.get("description") or "")[:800],
            "skill": skill,
            "skill_zh": META_ACTION_TRANSLATIONS[skill],
            "confidence": _confidence(item.get("confidence")),
            "primary_targets": list(_list_value(item.get("primary_targets")))[:20],
            "target_instance": str(item.get("target_instance") or "")[:120],
            "evidence_frames": sorted({
                _safe_int(value, -1)
                for value in _list_value(item.get("evidence_frames"))
                if _safe_int(value, -1) >= 0
            }),
        }
        if isinstance(item.get("boundary_range"), list):
            fine["boundary_range"] = list(item["boundary_range"])[:2]
        output.append(fine)
    return output


def _segment_object_signature(segment: dict) -> tuple[str, tuple[str, ...]]:
    instance = str(segment.get("target_instance") or "").strip().casefold()
    targets = []
    for value in _list_value(segment.get("primary_targets")):
        name = str(value.get("name") if isinstance(value, dict) else value).strip().casefold()
        if name:
            targets.append(name)
    return instance, tuple(sorted(set(targets)))


def _target_names(value: Any) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in _list_value(value):
        name = str(item.get("name") if isinstance(item, dict) else item).strip()[:120]
        if name and name.casefold() not in seen:
            seen.add(name.casefold())
            output.append(name)
    return output


def _normalize_window_result(
    raw: dict,
    window: dict,
    sampled_frames: list[int],
    fps: float,
) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    start = int(window["start_frame"])
    end = int(window["end_frame"])
    length = end - start + 1
    valid_evidence = set(sampled_frames)
    source_items = raw.get("fine") if isinstance(raw.get("fine"), list) else raw.get("segments")
    segments: list[dict] = []
    warnings = [str(value)[:500] for value in _list_value(raw.get("warnings"))[:30]]
    for item in _list_value(source_items)[:160]:
        if not isinstance(item, dict):
            continue
        item_start = max(start, min(_safe_int(item.get("start_frame"), start), end))
        item_end = max(item_start, min(_safe_int(item.get("end_frame"), item_start), end))
        phase = _normalize_phase_label(item.get("phase_label") or item.get("stage") or item.get("label"))
        supplied_skill = str(item.get("skill") or "").strip()
        skill = canonical_meta_action(supplied_skill) if supplied_skill else _PHASE_META_ACTIONS.get(phase, "Other")
        if supplied_skill and skill == "Other" and supplied_skill.casefold() != "other":
            warnings.append(f"Fine skill '{supplied_skill[:80]}' is outside the meta_action vocabulary and was normalized to Other.")
        if phase == "unknown" and skill != "Other":
            phase = _phase_for_meta_action(skill)
        evidence_frames = sorted({
            frame
            for value in _list_value(item.get("evidence_frames"))
            if (frame := _safe_int(value, -1)) in valid_evidence
        })
        targets = _target_names(item.get("primary_targets"))
        segments.append({
            "start_frame": item_start - start,
            "end_frame": item_end - start,
            "phase_label": phase if supplied_skill == "" else _phase_for_meta_action(skill),
            "label": phase if supplied_skill == "" else _phase_for_meta_action(skill),
            "skill": skill,
            "skill_zh": META_ACTION_TRANSLATIONS[skill],
            "description": str(item.get("description") or "")[:800],
            "confidence": _confidence(item.get("confidence"), 0.5),
            "object_nouns": [str(value).strip()[:120] for value in _list_value(item.get("object_nouns")) if str(value).strip()][:30],
            "primary_targets": targets,
            "target_instance": str(item.get("target_instance") or "").strip()[:120],
            "evidence_frames": evidence_frames,
            "boundary_source": "vlm",
            "source_window_ids": [str(window["window_id"])],
        })
    normalized = _normalize_phase_segments(segments, length, fps)
    for segment in normalized:
        segment["start_frame"] += start
        segment["end_frame"] += start
        segment["start_time"] = round(segment["start_frame"] / max(0.01, fps), 3)
        segment["end_time"] = round(segment["end_frame"] / max(0.01, fps), 3)
        segment["skill"] = canonical_meta_action(segment.get("skill"))
        segment["skill_zh"] = META_ACTION_TRANSLATIONS[segment["skill"]]
        segment["source_window_ids"] = list(segment.get("source_window_ids") or [str(window["window_id"])])

    target_records: list[dict] = []
    for item in _list_value(raw.get("primary_targets"))[:30]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()[:120]
        if not name:
            continue
        evidence_frames = sorted({
            frame
            for value in _list_value(item.get("visible_evidence_frames"))
            if (frame := _safe_int(value, -1)) in valid_evidence
        })
        target_records.append({
            "name": name,
            "role": str(item.get("role") or "behavior_target")[:120],
            "confidence": _confidence(item.get("confidence"), 0.5),
            "visible_evidence_frames": evidence_frames,
            "evidence": str(item.get("evidence") or "")[:500],
        })
    segment_targets = _target_names([
        target
        for segment in normalized
        for target in _list_value(segment.get("primary_targets"))
    ])
    known_targets = {item["name"].casefold() for item in target_records}
    for name in segment_targets:
        if name.casefold() not in known_targets:
            evidence = sorted({
                frame
                for segment in normalized
                if name in _list_value(segment.get("primary_targets"))
                for frame in _list_value(segment.get("evidence_frames"))
            })
            target_records.append({
                "name": name,
                "role": "behavior_target",
                "confidence": max(
                    [_confidence(segment.get("confidence"), 0.5) for segment in normalized if name in _list_value(segment.get("primary_targets"))]
                    or [0.5]
                ),
                "visible_evidence_frames": evidence,
                "evidence": "Derived from window-level Fine annotation.",
            })
            known_targets.add(name.casefold())
    coarse = raw.get("coarse") if isinstance(raw.get("coarse"), dict) else {}
    summary = str(
        raw.get("window_summary")
        or coarse.get("summary")
        or raw.get("behavior_description")
        or raw.get("task_label")
        or ""
    ).strip()[:800]
    object_nouns = []
    seen_nouns: set[str] = set()
    for value in [
        *_list_value(raw.get("object_nouns")),
        *[noun for segment in normalized for noun in _list_value(segment.get("object_nouns"))],
        *[item["name"] for item in target_records],
    ]:
        noun = str(value or "").strip().strip(" ,.;:")[:120]
        if noun and noun.casefold() not in seen_nouns:
            seen_nouns.add(noun.casefold())
            object_nouns.append(noun)
    return {
        "window_id": str(window["window_id"]),
        "start_frame": start,
        "end_frame": end,
        "sampled_frames": list(sampled_frames),
        "summary": summary,
        "coarse": coarse,
        "medium": [dict(item) for item in _list_value(raw.get("medium")) if isinstance(item, dict)][:40],
        "segments": normalized,
        "object_nouns": object_nouns,
        "primary_targets": target_records,
        "confidence": _confidence(raw.get("confidence"), np.mean([segment["confidence"] for segment in normalized]) if normalized else 0.0),
        "warnings": warnings[:30],
    }


def _window_segment_signature(segment: dict) -> tuple:
    return (
        canonical_meta_action(segment.get("skill")),
        _normalize_phase_label(segment.get("phase_label")),
        *_segment_object_signature(segment),
    )


def _merge_window_segments(
    window_results: list[dict],
    frame_count: int,
    fps: float,
    allowed_ranges: list[tuple[int, int]],
) -> list[dict]:
    if frame_count <= 0:
        return []
    candidates: list[dict] = []
    best_score = np.full(frame_count, -np.inf, dtype=np.float64)
    best_candidate = np.full(frame_count, -1, dtype=np.int32)
    for window_result in window_results:
        window_start = int(window_result["start_frame"])
        window_end = int(window_result["end_frame"])
        center = (window_start + window_end) / 2.0
        half = max(1.0, (window_end - window_start + 1) / 2.0)
        for segment in window_result.get("segments") or []:
            item = deepcopy(segment)
            start = max(window_start, min(_safe_int(item.get("start_frame"), window_start), window_end))
            end = max(start, min(_safe_int(item.get("end_frame"), start), window_end))
            item["start_frame"] = start
            item["end_frame"] = end
            item["source_window_ids"] = list(dict.fromkeys([
                *(_list_value(item.get("source_window_ids"))),
                str(window_result["window_id"]),
            ]))
            candidate_index = len(candidates)
            candidates.append(item)
            positions = np.arange(start, end + 1, dtype=np.int64)
            centrality = np.clip(1.0 - np.abs(positions.astype(np.float64) - center) / half, 0.0, 1.0)
            score = 0.75 * _confidence(item.get("confidence"), 0.5) + 0.25 * (0.25 + 0.75 * centrality)
            if _normalize_phase_label(item.get("phase_label")) == "unknown":
                score -= 0.3
            current = best_score[positions]
            update = score > current
            if update.any():
                selected_positions = positions[update]
                best_score[selected_positions] = score[update]
                best_candidate[selected_positions] = candidate_index

    output: list[dict] = []
    for range_start, range_end in allowed_ranges:
        cursor = range_start
        while cursor <= range_end:
            candidate_index = int(best_candidate[cursor])
            signature = _window_segment_signature(candidates[candidate_index]) if candidate_index >= 0 else ("Other", "unknown", "", ())
            end = cursor
            while end + 1 <= range_end:
                next_index = int(best_candidate[end + 1])
                next_signature = _window_segment_signature(candidates[next_index]) if next_index >= 0 else ("Other", "unknown", "", ())
                if next_signature != signature:
                    break
                end += 1
            if candidate_index >= 0:
                item = deepcopy(candidates[candidate_index])
                source_ids = []
                confidences = []
                evidence_frames = []
                for frame in range(cursor, end + 1):
                    index = int(best_candidate[frame])
                    if index < 0 or _window_segment_signature(candidates[index]) != signature:
                        continue
                    source_ids.extend(_list_value(candidates[index].get("source_window_ids")))
                    confidences.append(_confidence(candidates[index].get("confidence"), 0.5))
                    evidence_frames.extend(_list_value(candidates[index].get("evidence_frames")))
                item.update({
                    "start_frame": cursor,
                    "end_frame": end,
                    "start_time": round(cursor / max(0.01, fps), 3),
                    "end_time": round(end / max(0.01, fps), 3),
                    "confidence": float(np.mean(confidences)) if confidences else _confidence(item.get("confidence"), 0.5),
                    "evidence_frames": sorted({
                        _safe_int(value, -1)
                        for value in evidence_frames
                        if cursor <= _safe_int(value, -1) <= end
                    }),
                    "source_window_ids": list(dict.fromkeys(str(value) for value in source_ids if str(value))),
                    "boundary_source": "vlm",
                })
            else:
                item = {
                    "start_frame": cursor,
                    "end_frame": end,
                    "start_time": round(cursor / max(0.01, fps), 3),
                    "end_time": round(end / max(0.01, fps), 3),
                    "phase_label": "unknown",
                    "label": "unknown",
                    "skill": "Other",
                    "skill_zh": META_ACTION_TRANSLATIONS["Other"],
                    "description": "No readable window-level visual evidence covered this interval.",
                    "confidence": 0.0,
                    "primary_targets": [],
                    "target_instance": "",
                    "evidence_frames": [],
                    "source_window_ids": [],
                    "boundary_source": "vlm",
                }
            if output and output[-1]["end_frame"] + 1 == cursor and _window_segment_signature(output[-1]) == signature:
                output[-1]["end_frame"] = end
                output[-1]["end_time"] = item["end_time"]
                output[-1]["confidence"] = max(_confidence(output[-1].get("confidence")), _confidence(item.get("confidence")))
                output[-1]["evidence_frames"] = sorted(set(_list_value(output[-1].get("evidence_frames"))) | set(_list_value(item.get("evidence_frames"))))
                output[-1]["source_window_ids"] = list(dict.fromkeys([
                    *_list_value(output[-1].get("source_window_ids")),
                    *_list_value(item.get("source_window_ids")),
                ]))
            else:
                output.append(item)
            cursor = end + 1
    return output


def _aggregate_window_targets(window_results: list[dict]) -> tuple[list[str], list[dict]]:
    nouns: list[str] = []
    seen_nouns: set[str] = set()
    targets: dict[str, dict] = {}
    for window in window_results:
        for value in window.get("object_nouns") or []:
            noun = str(value or "").strip()[:120]
            if noun and noun.casefold() not in seen_nouns:
                seen_nouns.add(noun.casefold())
                nouns.append(noun)
        for item in window.get("primary_targets") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()[:120]
            if not name:
                continue
            key = name.casefold()
            if key not in targets:
                targets[key] = deepcopy(item)
                targets[key]["visible_evidence_frames"] = list(_list_value(item.get("visible_evidence_frames")))
            else:
                targets[key]["confidence"] = max(_confidence(targets[key].get("confidence")), _confidence(item.get("confidence")))
                targets[key]["visible_evidence_frames"] = sorted(set(
                    _list_value(targets[key].get("visible_evidence_frames"))
                ) | set(_list_value(item.get("visible_evidence_frames"))))
            if key not in seen_nouns:
                seen_nouns.add(key)
                nouns.append(name)
    return nouns, list(targets.values())[:30]


def _fallback_medium_from_segments(segments: list[dict], frame_count: int, summary: str) -> list[dict]:
    if not segments:
        return _normalize_tri_level_intervals([], frame_count, summary)

    def objective(skill: str) -> str:
        if skill in {"Reach", "Grasp", "Hold", "Pinch", "Clip", "Lift"}:
            return "acquire"
        if skill in {"Transport", "Carry", "Move", "Align", "HandOver"}:
            return "transfer"
        if skill in {"Place", "Release", "Drop", "Withdraw"}:
            return "finish"
        return skill

    output: list[dict] = []
    for segment in segments:
        skill = canonical_meta_action(segment.get("skill"))
        key = (objective(skill), _segment_object_signature(segment))
        description = str(segment.get("description") or summary)[:800]
        if output and output[-1]["_key"] == key and output[-1]["end_frame"] + 1 == int(segment["start_frame"]):
            output[-1]["end_frame"] = int(segment["end_frame"])
            if len(description) > len(output[-1]["description"]):
                output[-1]["description"] = description
        else:
            output.append({
                "start_frame": int(segment["start_frame"]),
                "end_frame": int(segment["end_frame"]),
                "description": description,
                "_key": key,
            })
    for item in output:
        item.pop("_key", None)
    return _normalize_tri_level_intervals(output, frame_count, summary)


def _global_result_from_windows(
    window_results: list[dict],
    merged_segments: list[dict],
    summary_raw: dict | None,
    episode: dict,
) -> dict:
    frame_count = max(1, _safe_int(episode.get("frame_count"), 1))
    fps = max(0.01, _safe_float(episode.get("fps"), 30.0))
    normalized_segments = _normalize_phase_segments(merged_segments, frame_count, fps)
    for segment in normalized_segments:
        skill = canonical_meta_action(segment.get("skill"))
        segment["skill"] = skill
        segment["skill_zh"] = META_ACTION_TRANSLATIONS[skill]
    raw = summary_raw if isinstance(summary_raw, dict) else {}
    coarse = raw.get("coarse") if isinstance(raw.get("coarse"), dict) else {}
    summaries = [str(item.get("summary") or "").strip() for item in window_results if str(item.get("summary") or "").strip()]
    summary = str(coarse.get("summary") or "").strip()[:800]
    if not summary and summaries:
        scores: dict[str, tuple[float, str]] = {}
        for item in window_results:
            value = str(item.get("summary") or "").strip()[:800]
            if not value:
                continue
            weight = max(1, int(item["end_frame"]) - int(item["start_frame"]) + 1) * max(0.1, _confidence(item.get("confidence"), 0.5))
            current = scores.get(value.casefold(), (0.0, value))
            scores[value.casefold()] = (current[0] + weight, current[1])
        if scores:
            summary = max(scores.values(), key=lambda item: item[0])[1]
    summary = summary or "other"
    medium_source = raw.get("medium") if isinstance(raw.get("medium"), list) and raw.get("medium") else None
    medium = (
        _normalize_tri_level_intervals(medium_source, frame_count, summary)
        if medium_source
        else _fallback_medium_from_segments(normalized_segments, frame_count, summary)
    )
    object_nouns, primary_targets = _aggregate_window_targets(window_results)
    duration_weight = 0
    confidence_total = 0.0
    for segment in normalized_segments:
        if _normalize_phase_label(segment.get("phase_label")) == "unknown":
            continue
        weight = int(segment["end_frame"]) - int(segment["start_frame"]) + 1
        duration_weight += weight
        confidence_total += weight * _confidence(segment.get("confidence"), 0.5)
    warnings = [
        str(value)[:500]
        for item in window_results
        for value in _list_value(item.get("warnings"))
    ]
    warnings.extend(str(value)[:500] for value in _list_value(raw.get("warnings")))
    return {
        "annotation_protocol": {"version": TRI_LEVEL_PROTOCOL_VERSION, "schema": TRI_LEVEL_PROTOCOL_SCHEMA},
        "task_label": summary,
        "direction": "unknown",
        "behavior_description": summary,
        "confidence": confidence_total / duration_weight if duration_weight else _confidence(raw.get("confidence")),
        "coarse": {"summary": summary},
        "medium": medium,
        "fine": _fine_from_segments(normalized_segments),
        "segments": normalized_segments,
        "object_nouns": object_nouns,
        "primary_targets": primary_targets,
        "warnings": list(dict.fromkeys(warnings))[:30],
    }


def _normalize_phase_segments(segments: list[dict], frame_count: int, fps: float) -> list[dict]:
    if frame_count <= 0:
        return []
    if not segments:
        segments = [{
            "start_frame": 0,
            "end_frame": frame_count - 1,
            "phase_label": "unknown",
            "label": "unknown",
            "description": "No temporal phase was returned by the VLM.",
            "confidence": 0.0,
            "primary_targets": [],
            "target_instance": "",
        }]
    ordered = sorted(
        (item for item in segments if isinstance(item, dict)),
        key=lambda item: (_safe_int(item.get("start_frame")), _safe_int(item.get("end_frame"))),
    )
    merged: list[dict] = []
    for source in ordered:
        item = dict(source)
        item["phase_label"] = _normalize_phase_label(item.get("phase_label") or item.get("stage") or item.get("label"))
        item["label"] = item["phase_label"]
        item["start_frame"] = max(0, min(_safe_int(item.get("start_frame")), frame_count - 1))
        item["end_frame"] = max(
            item["start_frame"],
            min(_safe_int(item.get("end_frame"), item["start_frame"]), frame_count - 1),
        )
        same_object = bool(merged) and _segment_object_signature(item) == _segment_object_signature(merged[-1])
        same_skill = bool(merged) and canonical_meta_action(item.get("skill")) == canonical_meta_action(merged[-1].get("skill"))
        if merged and same_object and same_skill and item["phase_label"] == merged[-1]["phase_label"] and item["start_frame"] <= merged[-1]["end_frame"] + 1:
            merged[-1]["end_frame"] = max(merged[-1]["end_frame"], item["end_frame"])
            merged[-1]["confidence"] = max(
                _confidence(merged[-1].get("confidence")),
                _confidence(item.get("confidence")),
            )
            merged[-1]["primary_targets"] = list(dict.fromkeys([
                *_list_value(merged[-1].get("primary_targets")),
                *_list_value(item.get("primary_targets")),
            ]))
            merged[-1]["evidence_frames"] = sorted(set(
                _list_value(merged[-1].get("evidence_frames"))
            ) | set(_list_value(item.get("evidence_frames"))))
            merged[-1]["source_window_ids"] = list(dict.fromkeys([
                *_list_value(merged[-1].get("source_window_ids")),
                *_list_value(item.get("source_window_ids")),
            ]))
            continue
        merged.append(item)
    if not merged:
        return _normalize_phase_segments([], frame_count, fps)

    # Missing VLM intervals stay explicit. Extending a neighbouring action
    # across an unseen gap would silently invent behavior and can corrupt
    # downstream valid/invalid trimming decisions.
    with_gaps: list[dict] = []
    covered_until = -1
    for item in merged:
        if item["start_frame"] > covered_until + 1:
            gap_start = covered_until + 1
            gap_end = item["start_frame"] - 1
            with_gaps.append({
                "start_frame": gap_start,
                "end_frame": gap_end,
                "phase_label": "unknown",
                "label": "unknown",
                "description": "Interval not explicitly covered by the VLM response.",
                "confidence": 0.0,
                "primary_targets": [],
                "target_instance": "",
                "boundary_source": "vlm",
            })
        with_gaps.append(item)
        covered_until = max(covered_until, item["end_frame"])
    if covered_until < frame_count - 1:
        with_gaps.append({
            "start_frame": covered_until + 1,
            "end_frame": frame_count - 1,
            "phase_label": "unknown",
            "label": "unknown",
            "description": "Interval not explicitly covered by the VLM response.",
            "confidence": 0.0,
            "primary_targets": [],
            "target_instance": "",
            "boundary_source": "vlm",
        })
    merged = with_gaps[:frame_count]

    splits: list[int] = []
    previous = 0
    for index, (left, right) in enumerate(zip(merged, merged[1:])):
        proposed = int(round(((int(left["end_frame"]) + 1) + int(right["start_frame"])) / 2.0))
        minimum = previous + 1
        maximum = frame_count - (len(merged) - index - 1)
        split = max(minimum, min(proposed, maximum))
        splits.append(split)
        previous = split
    for index, item in enumerate(merged):
        start = 0 if index == 0 else splits[index - 1]
        end = frame_count - 1 if index == len(merged) - 1 else splits[index] - 1
        item.update({
            "start_frame": start,
            "end_frame": end,
            "start_time": round(start / fps, 3),
            "end_time": round(end / fps, 3),
            "phase_label": _normalize_phase_label(item.get("phase_label")),
            "label": _normalize_phase_label(item.get("phase_label")),
            "boundary_source": "vlm",
        })
        item.pop("stage", None)
    return merged


def _validate_tri_level_result(raw: dict, episode: dict, sampled_frames: list[int] | None = None) -> dict:
    frame_count = max(1, int(episode.get("frame_count") or 1))
    fps = max(0.01, float(episode.get("fps") or 30.0))
    coarse_source = raw.get("coarse") if isinstance(raw.get("coarse"), dict) else {}
    summary = str(coarse_source.get("summary") or "other").strip()[:800] or "other"
    warnings: list[str] = []
    valid_evidence = set(sampled_frames or [])
    fine_segments = []
    for item in _list_value(raw.get("fine"))[:120]:
        if not isinstance(item, dict):
            continue
        supplied_skill = str(item.get("skill") or "").strip()
        skill = canonical_meta_action(supplied_skill)
        if supplied_skill and skill == "Other" and supplied_skill.casefold() != "other":
            warnings.append(f"Fine skill '{supplied_skill[:80]}' is outside the meta_action vocabulary and was normalized to Other.")
        start = max(0, min(_safe_int(item.get("start_frame")), frame_count - 1))
        end = max(start, min(_safe_int(item.get("end_frame"), start), frame_count - 1))
        evidence_frames = sorted({
            frame
            for value in _list_value(item.get("evidence_frames"))
            if (frame := _safe_int(value, -1)) in valid_evidence
        })
        fine_segments.append({
            "start_frame": start,
            "end_frame": end,
            "phase_label": _phase_for_meta_action(skill),
            "label": _phase_for_meta_action(skill),
            "skill": skill,
            "skill_zh": META_ACTION_TRANSLATIONS[skill],
            "description": str(item.get("description") or "")[:800],
            "confidence": _confidence(item.get("confidence"), 0.5),
            "object_nouns": [str(value).strip()[:120] for value in _list_value(item.get("object_nouns")) if str(value).strip()][:30],
            "primary_targets": _target_names(item.get("primary_targets")),
            "target_instance": str(item.get("target_instance") or "").strip()[:120],
            "evidence_frames": evidence_frames,
            "boundary_source": "vlm",
        })
    fine_segments = _normalize_phase_segments(fine_segments, frame_count, fps)
    for segment in fine_segments:
        skill = canonical_meta_action(segment.get("skill"))
        segment["skill"] = skill
        segment["skill_zh"] = META_ACTION_TRANSLATIONS[skill]
    medium = _normalize_tri_level_intervals(raw.get("medium"), frame_count, summary)
    window_result = {
        "object_nouns": [str(value).strip()[:120] for value in _list_value(raw.get("object_nouns")) if str(value).strip()],
        "primary_targets": [item for item in _list_value(raw.get("primary_targets")) if isinstance(item, dict)],
    }
    object_nouns, primary_targets = _aggregate_window_targets([window_result])
    for segment in fine_segments:
        for noun in _list_value(segment.get("object_nouns")):
            text = str(noun).strip()[:120]
            if text and text.casefold() not in {value.casefold() for value in object_nouns}:
                object_nouns.append(text)
    weights = [max(1, int(item["end_frame"]) - int(item["start_frame"]) + 1) for item in fine_segments]
    overall_confidence = (
        sum(weight * _confidence(item.get("confidence"), 0.5) for weight, item in zip(weights, fine_segments)) / sum(weights)
        if weights else 0.0
    )
    return {
        "annotation_protocol": {"version": TRI_LEVEL_PROTOCOL_VERSION, "schema": TRI_LEVEL_PROTOCOL_SCHEMA},
        "task_label": summary,
        "direction": "unknown",
        "behavior_description": summary,
        "confidence": overall_confidence,
        "coarse": {"summary": summary},
        "medium": medium,
        "fine": _fine_from_segments(fine_segments),
        "segments": fine_segments,
        "object_nouns": object_nouns,
        "primary_targets": primary_targets,
        "warnings": warnings[:30],
    }


def _validate_result(raw: dict, ontology: dict, episode: dict, sampled_frames: list[int]) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    if isinstance(raw.get("coarse"), dict) or isinstance(raw.get("fine"), list):
        return _validate_tri_level_result(raw, episode, sampled_frames)
    categories = {
        str(item.get("label") or "").casefold(): str(item.get("label") or "")
        for item in _list_value(ontology.get("categories"))
        if isinstance(item, dict) and str(item.get("label") or "").strip()
    }
    proposed = str(raw.get("task_label") or "other").strip()
    task_label = categories.get(proposed.casefold(), "other")
    confidence = _confidence(raw.get("confidence"))
    direction = str(raw.get("direction") or "unknown").casefold()
    if direction not in {"forward", "reverse", "unknown"}:
        direction = "unknown"
    frame_count = int(episode.get("frame_count", 0) or 0)
    valid_evidence = set(sampled_frames)

    targets = []
    seen_targets: set[str] = set()
    for item in _list_value(raw.get("primary_targets"))[:20]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name or name.casefold() in seen_targets:
            continue
        evidence_frames = sorted({
            frame
            for value in _list_value(item.get("visible_evidence_frames"))
            if (frame := _safe_int(value, -1)) in valid_evidence
        })
        if not evidence_frames:
            continue
        seen_targets.add(name.casefold())
        targets.append({
            "name": name,
            "role": str(item.get("role") or "behavior_target"),
            "confidence": _confidence(item.get("confidence")),
            "visible_evidence_frames": evidence_frames,
            "evidence": str(item.get("evidence") or "")[:500],
        })

    segments = []
    for item in _list_value(raw.get("segments"))[:80]:
        if not isinstance(item, dict):
            continue
        start = max(0, min(_safe_int(item.get("start_frame")), max(0, frame_count - 1)))
        end = max(start, min(_safe_int(item.get("end_frame"), start), max(0, frame_count - 1)))
        segment_targets = [
            str(name)
            for name in _list_value(item.get("primary_targets"))
            if str(name).casefold() in seen_targets
        ]
        phase_label = _normalize_phase_label(item.get("phase_label") or item.get("stage") or item.get("label"))
        segments.append({
            "start_frame": start,
            "end_frame": end,
            "start_time": round(start / max(0.01, float(episode.get("fps", 30.0))), 3),
            "end_time": round(end / max(0.01, float(episode.get("fps", 30.0))), 3),
            "phase_label": phase_label,
            "label": phase_label,
            "description": str(item.get("description") or "")[:800],
            "confidence": _confidence(item.get("confidence"), confidence),
            "primary_targets": segment_targets,
            "target_instance": str(item.get("target_instance") or "").strip()[:120],
            "boundary_source": "vlm",
        })
    if not segments:
        segments.append({
            "start_frame": 0,
            "end_frame": max(0, frame_count - 1),
            "start_time": 0.0,
            "end_time": round(max(0, frame_count - 1) / max(0.01, float(episode.get("fps", 30.0))), 3),
            "phase_label": "unknown",
            "label": "unknown",
            "description": str(raw.get("behavior_description") or "")[:800],
            "confidence": confidence,
            "primary_targets": [item["name"] for item in targets],
            "target_instance": "",
            "boundary_source": "vlm",
        })
    segments = _normalize_phase_segments(segments, frame_count, max(0.01, float(episode.get("fps", 30.0))))
    object_nouns = []
    seen_nouns: set[str] = set()
    for value in [*_list_value(raw.get("object_nouns"))[:40], *[item["name"] for item in targets]]:
        noun = str(value or "").strip().strip(" ,.;:")[:120]
        if noun and noun.casefold() not in seen_nouns:
            seen_nouns.add(noun.casefold())
            object_nouns.append(noun)
    legacy_result = {
        "annotation_protocol": {"version": TRI_LEVEL_PROTOCOL_VERSION, "schema": TRI_LEVEL_PROTOCOL_SCHEMA, "source_schema": "legacy_compat"},
        "task_label": task_label,
        "direction": direction,
        "behavior_description": str(raw.get("behavior_description") or "")[:1200],
        "confidence": confidence,
        "segments": segments,
        "object_nouns": object_nouns,
        "primary_targets": targets,
        "warnings": [str(item)[:500] for item in _list_value(raw.get("warnings"))[:30]],
    }
    legacy_result["coarse"] = {"summary": legacy_result["behavior_description"] or legacy_result["task_label"]}
    legacy_result["medium"] = _normalize_tri_level_intervals(
        [{"start_frame": 0, "end_frame": max(0, frame_count - 1), "description": legacy_result["behavior_description"]}],
        frame_count,
        legacy_result["task_label"],
    )
    legacy_result["fine"] = _fine_from_segments(segments)
    return legacy_result


def annotate_episode_behavior(
    dataset_id: str,
    manifest: dict,
    episode: dict,
    request: BehaviorAnnotationRequest,
    progress,
    analysis_media_override: dict | None = None,
    analysis_source_kind: str | None = None,
    analysis_frame_ranges: list[tuple[int, int]] | None = None,
    source_media_file_id: str | None = None,
    run_id: str | None = None,
    timeline_id: str | None = None,
    sampling_evidence: list[dict] | None = None,
) -> dict:
    selected_media_file_id = (
        source_media_file_id
        or request.media_file_id
        or (analysis_media_override or {}).get("file_id")
        or episode.get("primary_media_file_id")
    )
    source_media = episode_media(episode, selected_media_file_id)
    require_media_eligibility(source_media, "vlm_behavior")
    progress(2, "T0 正在建立统一时间轴")
    ensure_episode_time_sync(
        manifest,
        episode,
        reference_media_file_id=str(source_media.get("file_id") or "") or None,
    )
    if analysis_media_override is None:
        analysis_media, smoothing_document = preferred_smoothed_media(dataset_id, episode, source_media)
    else:
        analysis_media = {**source_media, **analysis_media_override}
        smoothing_document = {"source": analysis_source_kind or "external_video_smoothing"}
    require_media_eligibility(analysis_media, "vlm_behavior")
    if not request.force:
        existing = behavior_annotation_status(
            dataset_id,
            manifest,
            episode,
            source_media_file_id=str(source_media.get("file_id") or "") or None,
            analysis_media=analysis_media,
            analysis_frame_ranges=analysis_frame_ranges,
        )
        if existing["reusable"]:
            progress(100, "已有有效 VLM 行为标注，已复用且未调用 Qwen")
            return existing["payload"]
    if not registry.has_vlm:
        raise RuntimeError("请先配置 Qwen-VLM API")
    ontology_root = os.getenv("VLA_BEHAVIOR_ONTOLOGY", "").strip()
    if not ontology_root:
        ontology_root = next((candidate for candidate in (r"F:\part1", r"D:\part1") if Path(candidate).is_dir()), "")
    progress(8, "加载行为语言模板")
    ontology = load_behavior_ontology(ontology_root)
    if ontology["source"] == "builtin":
        progress(9, "使用内置通用操作词表")
    source_fingerprint = _media_fingerprint(source_media)
    same_analysis_source = str(analysis_media.get("path") or "") == str(source_media.get("path") or "")
    analysis_fingerprint = source_fingerprint if same_analysis_source else _media_fingerprint(analysis_media)
    if smoothing_document:
        message = "使用指定的视频平滑结果" if analysis_media_override is not None else "使用已应用的视频平滑结果"
        progress(10, f"{message}: {source_media.get('stream_name') or 'primary'}")
    analysis_frame_count = int(analysis_media["frame_count"])
    allowed_ranges = _normalize_frame_ranges(analysis_frame_count, analysis_frame_ranges)
    if not allowed_ranges:
        raise RuntimeError("S1-S5/C3 初筛后没有可供 VLM 标注的有效帧")
    schema_summary = json.dumps((manifest.get("schema_profile") or {}).get("understanding") or {}, ensure_ascii=False)[:3500]
    behavior_frame_count = int(
        analysis_media.get("frame_count")
        or source_media.get("frame_count")
        or episode.get("frame_count")
        or 0
    )
    behavior_fps = max(0.01, float(
        analysis_media.get("fps")
        or source_media.get("fps")
        or episode.get("fps")
        or 30.0
    ))
    timing_episode = {**episode, "frame_count": behavior_frame_count, "fps": behavior_fps}
    source_positions = np.asarray(analysis_media.get("source_frame_positions") or [], dtype=np.float64).reshape(-1)
    joint_alignment_kwargs: dict[str, Any] = {}
    if source_positions.shape == (behavior_frame_count,) and np.isfinite(source_positions).all():
        joint_alignment = ensure_episode_time_sync(
            manifest,
            episode,
            force=False,
            reference_media_file_id=str(analysis_media.get("file_id") or "") or None,
        )
        joint_alignment_kwargs["alignment"] = retime_sensor_alignment(joint_alignment, analysis_media)
    joint_pose = load_episode_joint_pose(
        manifest,
        episode,
        frame_count=behavior_frame_count,
        reference_media_file_id=str(analysis_media.get("file_id") or "") or None,
        **joint_alignment_kwargs,
    )
    joint_score = joint_motion_change_score(joint_pose, behavior_fps, behavior_frame_count)
    windows = _plan_behavior_windows(behavior_frame_count, behavior_fps, allowed_ranges)
    if not windows:
        raise RuntimeError("没有可供 VLM 分窗标注的有效帧区间")

    frame_cache: dict[int, np.ndarray | None] = {}

    def cached_frame(index: int) -> np.ndarray | None:
        if index not in frame_cache:
            frame_cache[index] = read_frame(analysis_media, index)
        return frame_cache[index]

    context = f"Episode {episode['name']}; schema={schema_summary}"
    window_results: list[dict] = []
    window_sampling: list[dict] = []
    for position, window in enumerate(windows):
        progress(12 + 20 * position / max(1, len(windows)), f"自适应抽取多图窗口 {position + 1}/{len(windows)}")
        indices, sampling_metrics = _adaptive_window_indices(
            window,
            behavior_fps,
            request.sample_count,
            frame_cache,
            cached_frame,
            joint_score=joint_score,
            sampling_evidence=sampling_evidence,
        )
        frames = [
            (index, index / behavior_fps, frame)
            for index in indices
            if (frame := cached_frame(index)) is not None
        ]
        if not frames:
            raise RuntimeError(
                f"多图窗口 {position + 1}/{len(windows)} 没有可解码图像，范围="
                f"[{window['start_frame']}, {window['end_frame']}]"
            )
        progress(34 + 36 * position / max(1, len(windows)), f"Qwen-VLM 分析多图窗口 {position + 1}/{len(windows)}")
        raw = registry.annotate_behavior(
            frames,
            ontology["categories"],
            context,
            video_length=behavior_frame_count,
            duration=behavior_frame_count / behavior_fps,
            window_start=int(window["start_frame"]),
            window_end=int(window["end_frame"]),
            window_index=position,
            window_count=len(windows),
        )
        window_result = _normalize_window_result(raw, window, [item[0] for item in frames], behavior_fps)
        window_results.append(window_result)
        window_sampling.append({
            **window,
            "requested_max_images": request.sample_count,
            "frames": [item[0] for item in frames],
            "readable_frame_count": len(frames),
            **sampling_metrics,
        })
        _assert_media_unchanged(source_media, source_fingerprint, f"during Qwen window {position + 1}")
        if not same_analysis_source:
            _assert_media_unchanged(analysis_media, analysis_fingerprint, f"during Qwen window {position + 1}")

    progress(72, "合并重叠窗口的 Fine 动作标注")
    merged_segments = _merge_window_segments(
        window_results,
        behavior_frame_count,
        behavior_fps,
        allowed_ranges,
    )
    summary_raw = None
    summary_warning = ""
    if len(window_results) > 1 and hasattr(registry, "summarize_behavior_windows"):
        progress(74, "Qwen-VLM 正在归纳全局 Coarse 与多个 Medium 子任务")
        try:
            summary_raw = registry.summarize_behavior_windows(
                window_results,
                ontology["categories"],
                context,
                video_length=behavior_frame_count,
                duration=behavior_frame_count / behavior_fps,
                allowed_ranges=allowed_ranges,
            )
        except Exception as exc:
            summary_warning = f"Global window summarization failed; deterministic fallback was used: {str(exc)[:300]}"
    result = _apply_dataset_task_fallback(
        _global_result_from_windows(window_results, merged_segments, summary_raw, timing_episode),
        manifest,
    )
    if summary_warning:
        result["warnings"] = [*result.get("warnings", []), summary_warning][:30]

    progress(78, "使用已对齐 Joint Pose 微调窗口内动作边界")
    refined_pieces: list[dict] = []
    for range_start, range_end in allowed_ranges:
        pieces = []
        for segment in result["segments"]:
            start = max(range_start, int(segment.get("start_frame") or 0))
            end = min(range_end, int(segment.get("end_frame") or start))
            if start <= end:
                pieces.append({**segment, "start_frame": start, "end_frame": end})
        if pieces:
            refined_pieces.extend(refine_behavior_boundaries(
                pieces,
                behavior_fps,
                behavior_frame_count,
                joint_pose,
            ))
    result["segments"] = _constrain_segments_to_ranges(
        refined_pieces,
        allowed_ranges,
        behavior_frame_count,
        behavior_fps,
    )
    result["fine"] = _fine_from_segments(result["segments"])
    joint_refined_count = sum(item.get("boundary_source") == "joint_refined" for item in result["segments"])
    boundary_refinement = {
        "source": "joint_refined" if joint_refined_count else "vlm",
        "joint_pose_available": joint_pose is not None,
        "search_seconds": 0.5,
        "refined_segment_count": joint_refined_count,
    }
    _assert_media_unchanged(source_media, source_fingerprint, "during Joint boundary refinement")
    if not same_analysis_source:
        _assert_media_unchanged(analysis_media, analysis_fingerprint, "during Joint boundary refinement")
    all_sampled_frames = sorted({
        frame
        for window in window_sampling
        for frame in window.get("frames", [])
    })
    created_at = datetime.now(timezone.utc).isoformat()
    document = {
        "schema": BEHAVIOR_SCHEMA,
        "artifact_version": BEHAVIOR_ARTIFACT_VERSION,
        "dataset_id": dataset_id,
        "episode_id": episode["id"],
        "full_run_id": run_id,
        "timeline_id": timeline_id,
        "full_run_stage": "vlm" if run_id else None,
        "created_at": created_at,
        "provider": registry.status()["vlm"],
        "language_source": {
            key: ontology[key]
            for key in ("source", "root", "category_count", "fingerprint", "fallback_reason", "requested_root")
            if key in ontology
        },
        "sampling": {
            "frame_space": "analysis_video",
            "strategy": BEHAVIOR_SAMPLING_STRATEGY,
            "requested": request.sample_count,
            "requested_max_images_per_window": request.sample_count,
            "frames": all_sampled_frames,
            "total_image_count": sum(int(item.get("readable_frame_count") or 0) for item in window_sampling),
            "unique_image_count": len(all_sampled_frames),
            "window_seconds": BEHAVIOR_WINDOW_SECONDS,
            "window_overlap_seconds": BEHAVIOR_WINDOW_OVERLAP_SECONDS,
            "base_sample_fps": BEHAVIOR_BASE_SAMPLE_FPS,
            "event_sample_fps": BEHAVIOR_EVENT_SAMPLE_FPS,
            "event_radius_seconds": BEHAVIOR_EVENT_RADIUS_SECONDS,
            "windows": window_sampling,
            "allowed_ranges": [
                {"start_frame": start, "end_frame": end}
                for start, end in allowed_ranges
            ] if analysis_frame_ranges is not None else None,
            "allowed_frame_count": sum(end - start + 1 for start, end in allowed_ranges),
            "media_file_id": analysis_media.get("file_id"),
            "stream_name": analysis_media.get("stream_name"),
            "used_applied_video_smoothing": bool(smoothing_document),
        },
        "window_annotations": window_results,
        "source_video": {
            "file_id": source_media.get("file_id"),
            "stream_name": source_media.get("stream_name"),
            "relative_path": source_media.get("relative_path"),
            "fps": float(source_media.get("fps") or 0.0),
            "frame_count": int(source_media.get("frame_count") or 0),
            "fingerprint": source_fingerprint,
        },
        "analysis_video": {
            "kind": analysis_source_kind or ("applied_video_smoothing" if smoothing_document else "source_video"),
            "file_id": analysis_media.get("file_id"),
            "stream_name": analysis_media.get("stream_name"),
            "relative_path": analysis_media.get("relative_path"),
            "fps": float(analysis_media.get("fps") or 0.0),
            "frame_count": int(analysis_media.get("frame_count") or 0),
            "fingerprint": analysis_fingerprint,
        },
        "timeline": {
            "frame_space": "analysis_video",
            "frame_count": behavior_frame_count,
            "fps": behavior_fps,
        },
        "boundary_refinement": boundary_refinement,
        **result,
    }
    target_document = {
        "schema": TARGET_SCHEMA,
        "dataset_id": dataset_id,
        "episode_id": episode["id"],
        "full_run_id": run_id,
        "timeline_id": timeline_id,
        "created_at": created_at,
        "task_label": result["task_label"],
        "behavior_description": result["behavior_description"],
        "primary_terms": result["object_nouns"],
        "primary_targets": result["primary_targets"],
        "source_annotation": _annotation_path(dataset_id, episode["id"], run_id).name,
    }
    progress(90, "写入 .alicePD 行为标注与目标索引")
    annotation_path = _annotation_path(dataset_id, episode["id"], run_id)
    target_path = _target_path(dataset_id, episode["id"], run_id)
    _write_atomic(annotation_path, document)
    _write_atomic(target_path, target_document)
    change = record_change(
        dataset_id,
        "vlm_behavior",
        episode["id"],
        f"VLM behavior: {episode['name']}",
        [annotation_path, target_path],
        {
            "task_label": result["task_label"],
            "confidence": result["confidence"],
            "target_count": len(result["primary_targets"]),
            "joint_refined_segment_count": joint_refined_count,
        },
        [str(episode.get("relative_path") or "")],
    )
    document["artifacts"] = {
        "behavior": str(annotation_path),
        "targets": str(target_path),
    }
    document["change"] = {"id": change["id"], "status": change["status"], "revision": change["revision"]}
    return document


def snapshot_behavior_annotation_for_run(
    dataset_id: str,
    episode: dict,
    payload: dict,
    run_id: str,
    timeline_id: str,
) -> dict:
    """Copy a validated reusable VLM result into one immutable Full run."""
    document = deepcopy(payload)
    for key in ("artifact_path", "artifacts", "change", "reuse"):
        document.pop(key, None)
    document.update({
        "full_run_id": run_id,
        "timeline_id": timeline_id,
        "full_run_stage": "vlm_reused",
        "reused_from_created_at": payload.get("created_at"),
        "snapshotted_at": datetime.now(timezone.utc).isoformat(),
    })
    annotation_path = _annotation_path(dataset_id, str(episode["id"]), run_id)
    target_path = _target_path(dataset_id, str(episode["id"]), run_id)
    target_document = {
        "schema": TARGET_SCHEMA,
        "dataset_id": dataset_id,
        "episode_id": episode["id"],
        "full_run_id": run_id,
        "timeline_id": timeline_id,
        "created_at": document.get("created_at"),
        "task_label": document.get("task_label"),
        "behavior_description": document.get("behavior_description"),
        "primary_terms": list(document.get("object_nouns") or []),
        "primary_targets": list(document.get("primary_targets") or []),
        "source_annotation": annotation_path.name,
    }
    _write_atomic(annotation_path, document)
    _write_atomic(target_path, target_document)
    document["artifacts"] = {"behavior": str(annotation_path), "targets": str(target_path)}
    document["reuse"] = {
        "reused": True,
        "status": "snapshotted",
        "reason": "matching_artifact_copied_into_full_run",
    }
    return document


def load_behavior_annotation(dataset_id: str, episode_id: str, run_id: str | None = None) -> dict | None:
    path = _annotation_path(dataset_id, episode_id, run_id)
    payload = _read_json(path) if path.is_file() else None
    if payload is None or payload.get("schema") != BEHAVIOR_SCHEMA:
        return None
    try:
        version = int(payload.get("artifact_version") or 0)
    except (TypeError, ValueError):
        return None
    if version != BEHAVIOR_ARTIFACT_VERSION:
        return None
    if run_id and str(payload.get("full_run_id") or "") != str(run_id):
        return None
    frame_count = _behavior_timeline_frame_count(payload)
    if not _segments_follow_phase_protocol(payload, frame_count or None):
        return None
    protocol = payload.get("annotation_protocol") or {}
    if (
        protocol.get("version") != TRI_LEVEL_PROTOCOL_VERSION
        or protocol.get("schema") != TRI_LEVEL_PROTOCOL_SCHEMA
        or not _tri_level_fields_are_valid(payload, frame_count)
    ):
        return None
    try:
        manifest, _episode = get_episode(dataset_id, episode_id)
    except KeyError:
        manifest = {}
    return _apply_dataset_task_fallback(payload, manifest)


class BehaviorJobManager(CancellableJobMixin):
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm-behavior")
        self._jobs: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._init_cancellation()

    def submit(self, dataset_id: str, episode_id: str, request: BehaviorAnnotationRequest) -> dict:
        manifest, episode = get_episode(dataset_id, episode_id)
        context = behavior_analysis_context(dataset_id, manifest, episode, request.media_file_id)
        media = context["source_media"]
        require_media_eligibility(media, "vlm_behavior")
        if not request.force:
            existing = behavior_annotation_status(
                dataset_id,
                manifest,
                episode,
                source_media_file_id=str(media.get("file_id") or "") or None,
                analysis_media=context["analysis_media"],
                analysis_frame_ranges=context["analysis_frame_ranges"],
            )
            if existing["reusable"]:
                job_id = uuid.uuid4().hex
                job = {
                    "id": job_id,
                    "kind": "vlm_behavior",
                    "status": "complete",
                    "progress": 100,
                    "message": "已有有效 VLM 行为标注，已跳过 Qwen 请求并复用现有结果",
                    "result": existing["payload"],
                    "reused": True,
                    "skip_reason": "existing_valid_annotation",
                    "error": None,
                }
                with self._lock:
                    self._jobs[job_id] = job
                return dict(job)
        if not registry.has_vlm:
            raise RuntimeError("请先配置 Qwen-VLM API")
        job_id = uuid.uuid4().hex
        job = {"id": job_id, "kind": "vlm_behavior", "status": "queued", "progress": 0, "message": "等待 VLM 行为标注", "result": None, "reused": False, "error": None}
        with self._lock:
            self._jobs[job_id] = job
            self._register_cancellation(job_id)
        self._executor.submit(self._run, job_id, dataset_id, episode_id, request)
        return dict(job)

    def get(self, job_id: str) -> dict:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return dict(self._jobs[job_id])

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            self._jobs[job_id].update(changes)

    def _run(self, job_id: str, dataset_id: str, episode_id: str, request: BehaviorAnnotationRequest) -> None:
        try:
            self._start_unless_cancelled(job_id, status="running", progress=3, message="准备行为标注")
            manifest, episode = get_episode(dataset_id, episode_id)
            context = behavior_analysis_context(dataset_id, manifest, episode, request.media_file_id)
            def progress(value: float, message: str) -> None:
                self._raise_if_cancelled(job_id)
                self._update(job_id, progress=value, message=message)

            result = annotate_episode_behavior(
                dataset_id,
                manifest,
                episode,
                request,
                progress,
                analysis_media_override=context["analysis_media"],
                analysis_source_kind=context["analysis_source_kind"],
                analysis_frame_ranges=context["analysis_frame_ranges"],
                source_media_file_id=str(context["source_media"].get("file_id") or "") or None,
                sampling_evidence=(context.get("curation_report") or {}).get("samples"),
            )
            self._raise_if_cancelled(job_id)
            self._update(job_id, status="complete", progress=100, message="VLM 行为标注完成", result=result)
        except JobCancelled:
            self._mark_cancelled(job_id)
        except Exception as exc:
            self._update(job_id, status="failed", message=str(exc), error=str(exc), trace=traceback.format_exc(limit=8))
        finally:
            self._forget_cancellation(job_id)


behavior_jobs = BehaviorJobManager()
