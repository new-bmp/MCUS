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

import numpy as np

from .behavior_boundary_refiner import load_episode_joint_pose, refine_behavior_boundaries
from .job_control import CancellableJobMixin, JobCancelled
from .models import registry
from .qwen_trim import _source_fingerprints_match, _source_video_fingerprint
from .schemas import BehaviorAnnotationRequest
from .storage import dataset_artifact_dir, episode_media, get_episode, read_frame, record_change, slugify
from .video_smoothing import preferred_smoothed_media


BEHAVIOR_SCHEMA = "alice/vlm-behavior/v1"
TARGET_SCHEMA = "alice/behavior-targets/v1"
BEHAVIOR_ARTIFACT_VERSION = 3

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


def _annotation_path(dataset_id: str, episode_id: str) -> Path:
    root = dataset_artifact_dir(dataset_id, "behavior-annotations")
    return root / f"{slugify(episode_id)}.behavior.alice"


def _target_path(dataset_id: str, episode_id: str) -> Path:
    root = dataset_artifact_dir(dataset_id, "behavior-targets")
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
    if frame_count <= 0 or not frames or max(frames) != frame_count - 1:
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


def _payload_is_valid(annotation: dict, target: dict, dataset_id: str, episode: dict) -> bool:
    if annotation.get("schema") != BEHAVIOR_SCHEMA or target.get("schema") != TARGET_SCHEMA:
        return False
    try:
        artifact_version = int(annotation.get("artifact_version") or 0)
    except (TypeError, ValueError):
        return False
    if artifact_version != BEHAVIOR_ARTIFACT_VERSION:
        return False
    if str(annotation.get("dataset_id")) != str(dataset_id) or str(target.get("dataset_id")) != str(dataset_id):
        return False
    if str(annotation.get("episode_id")) != str(episode.get("id")) or str(target.get("episode_id")) != str(episode.get("id")):
        return False
    frame_count = _safe_int(
        (annotation.get("source_video") or {}).get("frame_count"),
        _safe_int(episode.get("frame_count")),
    )
    if not _segments_follow_phase_protocol(annotation, frame_count):
        return False
    if not isinstance(target.get("primary_terms"), list):
        return False
    source_annotation = str(target.get("source_annotation") or "")
    return source_annotation in {"", _annotation_path(dataset_id, str(episode.get("id"))).name}


def behavior_annotation_status(
    dataset_id: str,
    manifest: dict,
    episode: dict,
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


def _segment_object_signature(segment: dict) -> tuple[str, tuple[str, ...]]:
    instance = str(segment.get("target_instance") or "").strip().casefold()
    targets = []
    for value in _list_value(segment.get("primary_targets")):
        name = str(value.get("name") if isinstance(value, dict) else value).strip().casefold()
        if name:
            targets.append(name)
    return instance, tuple(sorted(set(targets)))


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
        if merged and same_object and item["phase_label"] == merged[-1]["phase_label"] and item["start_frame"] <= merged[-1]["end_frame"] + 1:
            merged[-1]["end_frame"] = max(merged[-1]["end_frame"], item["end_frame"])
            merged[-1]["confidence"] = max(
                _confidence(merged[-1].get("confidence")),
                _confidence(item.get("confidence")),
            )
            merged[-1]["primary_targets"] = list(dict.fromkeys([
                *_list_value(merged[-1].get("primary_targets")),
                *_list_value(item.get("primary_targets")),
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


def _validate_result(raw: dict, ontology: dict, episode: dict, sampled_frames: list[int]) -> dict:
    raw = raw if isinstance(raw, dict) else {}
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
    return {
        "task_label": task_label,
        "direction": direction,
        "behavior_description": str(raw.get("behavior_description") or "")[:1200],
        "confidence": confidence,
        "segments": segments,
        "object_nouns": object_nouns,
        "primary_targets": targets,
        "warnings": [str(item)[:500] for item in _list_value(raw.get("warnings"))[:30]],
    }


def annotate_episode_behavior(
    dataset_id: str,
    manifest: dict,
    episode: dict,
    request: BehaviorAnnotationRequest,
    progress,
    analysis_media_override: dict | None = None,
    analysis_source_kind: str | None = None,
    analysis_frame_ranges: list[tuple[int, int]] | None = None,
) -> dict:
    if not request.force:
        existing = behavior_annotation_status(dataset_id, manifest, episode)
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
    primary_media = episode_media(episode, episode.get("primary_media_file_id"))
    if analysis_media_override is None:
        analysis_media, smoothing_document = preferred_smoothed_media(dataset_id, episode, primary_media)
    else:
        analysis_media = {**primary_media, **analysis_media_override}
        smoothing_document = {"source": analysis_source_kind or "external_video_smoothing"}
    source_fingerprint = _media_fingerprint(primary_media)
    same_analysis_source = str(analysis_media.get("path") or "") == str(primary_media.get("path") or "")
    analysis_fingerprint = source_fingerprint if same_analysis_source else _media_fingerprint(analysis_media)
    if smoothing_document:
        message = "使用 minRE 视频平滑结果" if analysis_media_override is not None else "使用已应用的视频平滑结果"
        progress(10, f"{message}: {primary_media.get('stream_name') or 'primary'}")
    analysis_frame_count = int(analysis_media["frame_count"])
    allowed_ranges = _normalize_frame_ranges(analysis_frame_count, analysis_frame_ranges)
    indices = _sample_indices_in_ranges(analysis_frame_count, request.sample_count, analysis_frame_ranges)
    if not indices:
        raise RuntimeError("S1-S5/C3 初筛后没有可供 VLM 标注的有效帧")
    frames: list[tuple[int, float, np.ndarray]] = []
    for position, index in enumerate(indices):
        frame = read_frame(analysis_media, index)
        if frame is not None:
            frames.append((index, index / max(0.01, float(analysis_media["fps"])), frame))
        progress(12 + 24 * (position + 1) / max(1, len(indices)), f"抽取视频关键帧 {position + 1}/{len(indices)}")
    if len(frames) < 4:
        raise RuntimeError("可读取的视频关键帧不足，无法进行 VLM 行为标注")
    schema_summary = json.dumps((manifest.get("schema_profile") or {}).get("understanding") or {}, ensure_ascii=False)[:3500]
    progress(42, "Qwen-VLM 正在标注行为与主要目标")
    raw = registry.annotate_behavior(frames, ontology["categories"], f"Episode {episode['name']}; schema={schema_summary}")
    _assert_media_unchanged(primary_media, source_fingerprint, "during the Qwen request")
    if not same_analysis_source:
        _assert_media_unchanged(analysis_media, analysis_fingerprint, "during the Qwen request")
    behavior_frame_count = int(
        analysis_media.get("frame_count")
        or primary_media.get("frame_count")
        or episode.get("frame_count")
        or 0
    )
    behavior_fps = max(0.01, float(
        analysis_media.get("fps")
        or primary_media.get("fps")
        or episode.get("fps")
        or 30.0
    ))
    timing_episode = {**episode, "frame_count": behavior_frame_count, "fps": behavior_fps}
    result = _apply_dataset_task_fallback(
        _validate_result(raw, ontology, timing_episode, [item[0] for item in frames]),
        manifest,
    )
    progress(78, "使用已对齐 Joint Pose 微调 VLM 阶段边界")
    joint_pose = load_episode_joint_pose(manifest, episode, frame_count=behavior_frame_count)
    result["segments"] = refine_behavior_boundaries(
        result["segments"],
        behavior_fps,
        behavior_frame_count,
        joint_pose,
    )
    if analysis_frame_ranges is not None:
        result["segments"] = _constrain_segments_to_ranges(
            result["segments"],
            allowed_ranges,
            behavior_frame_count,
            behavior_fps,
        )
    joint_refined_count = sum(item.get("boundary_source") == "joint_refined" for item in result["segments"])
    boundary_refinement = {
        "source": "joint_refined" if joint_refined_count else "vlm",
        "joint_pose_available": joint_pose is not None,
        "search_seconds": 0.5,
        "refined_segment_count": joint_refined_count,
    }
    _assert_media_unchanged(primary_media, source_fingerprint, "during Joint boundary refinement")
    if not same_analysis_source:
        _assert_media_unchanged(analysis_media, analysis_fingerprint, "during Joint boundary refinement")
    created_at = datetime.now(timezone.utc).isoformat()
    document = {
        "schema": BEHAVIOR_SCHEMA,
        "artifact_version": BEHAVIOR_ARTIFACT_VERSION,
        "dataset_id": dataset_id,
        "episode_id": episode["id"],
        "created_at": created_at,
        "provider": registry.status()["vlm"],
        "language_source": {
            key: ontology[key]
            for key in ("source", "root", "category_count", "fingerprint", "fallback_reason", "requested_root")
            if key in ontology
        },
        "sampling": {
            "requested": request.sample_count,
            "frames": [item[0] for item in frames],
            "allowed_ranges": [
                {"start_frame": start, "end_frame": end}
                for start, end in allowed_ranges
            ] if analysis_frame_ranges is not None else None,
            "allowed_frame_count": sum(end - start + 1 for start, end in allowed_ranges),
            "media_file_id": primary_media.get("file_id"),
            "stream_name": primary_media.get("stream_name"),
            "used_applied_video_smoothing": bool(smoothing_document),
        },
        "source_video": {
            "file_id": primary_media.get("file_id"),
            "stream_name": primary_media.get("stream_name"),
            "relative_path": primary_media.get("relative_path"),
            "fps": float(primary_media.get("fps") or 0.0),
            "frame_count": int(primary_media.get("frame_count") or 0),
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
        "boundary_refinement": boundary_refinement,
        **result,
    }
    target_document = {
        "schema": TARGET_SCHEMA,
        "dataset_id": dataset_id,
        "episode_id": episode["id"],
        "created_at": created_at,
        "task_label": result["task_label"],
        "behavior_description": result["behavior_description"],
        "primary_terms": result["object_nouns"],
        "primary_targets": result["primary_targets"],
        "source_annotation": _annotation_path(dataset_id, episode["id"]).name,
    }
    progress(90, "写入 .alicePD 行为标注与目标索引")
    annotation_path = _annotation_path(dataset_id, episode["id"])
    target_path = _target_path(dataset_id, episode["id"])
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


def load_behavior_annotation(dataset_id: str, episode_id: str) -> dict | None:
    path = _annotation_path(dataset_id, episode_id)
    payload = _read_json(path) if path.is_file() else None
    if payload is None or payload.get("schema") != BEHAVIOR_SCHEMA:
        return None
    try:
        version = int(payload.get("artifact_version") or 0)
    except (TypeError, ValueError):
        return None
    if version != BEHAVIOR_ARTIFACT_VERSION:
        return None
    frame_count = _safe_int((payload.get("source_video") or {}).get("frame_count"))
    if not _segments_follow_phase_protocol(payload, frame_count or None):
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
        if not request.force:
            existing = behavior_annotation_status(dataset_id, manifest, episode)
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
            def progress(value: float, message: str) -> None:
                self._raise_if_cancelled(job_id)
                self._update(job_id, progress=value, message=message)

            result = annotate_episode_behavior(dataset_id, manifest, episode, request, progress)
            self._raise_if_cancelled(job_id)
            self._update(job_id, status="complete", progress=100, message="VLM 行为标注完成", result=result)
        except JobCancelled:
            self._mark_cancelled(job_id)
        except Exception as exc:
            self._update(job_id, status="failed", message=str(exc), error=str(exc), trace=traceback.format_exc(limit=8))
        finally:
            self._forget_cancellation(job_id)


behavior_jobs = BehaviorJobManager()
