"""Evidence-first quality results shared by curation, export and catalog adapters.

The curation report remains the authoritative Alice artifact.  This module adds
one small, stable view of that report so downstream systems can consume checks
without knowing every stage's private metrics layout.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any


QUALITY_EVIDENCE_SCHEMA = "alice/quality-evidence/v1"
QUALITY_EVIDENCE_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """Convert numpy-like scalar values without importing numpy in this layer."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _json_safe(value.tolist())
        except (TypeError, ValueError):
            pass
    return value


def _fingerprint(value: Any) -> str:
    serialized = json.dumps(
        _json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _frame_range(item: dict[str, Any]) -> tuple[int, int] | None:
    try:
        start = int(item.get("start_frame"))
        end = int(item.get("end_frame"))
    except (TypeError, ValueError):
        return None
    if end < start:
        start, end = end, start
    return max(0, start), max(0, end)


def _merge_ranges(ranges: list[tuple[int, int]], frame_count: int | None = None) -> list[tuple[int, int]]:
    bounded: list[tuple[int, int]] = []
    limit = int(frame_count or 0)
    for start, end in ranges:
        if limit > 0:
            if start >= limit:
                continue
            end = min(end, limit - 1)
        bounded.append((max(0, start), max(0, end)))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(bounded):
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _range_count(ranges: list[tuple[int, int]]) -> int:
    return sum(end - start + 1 for start, end in ranges)


def _subtract_ranges(ranges: list[tuple[int, int]], blocked: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Remove blocked frames from ranges while keeping inclusive boundaries."""
    remaining = list(ranges)
    for block_start, block_end in blocked:
        next_ranges: list[tuple[int, int]] = []
        for start, end in remaining:
            if block_end < start or block_start > end:
                next_ranges.append((start, end))
                continue
            if start < block_start:
                next_ranges.append((start, block_start - 1))
            if end > block_end:
                next_ranges.append((block_end + 1, end))
        remaining = next_ranges
    return remaining


def _verdict(status: str, findings: list[dict[str, Any]]) -> str:
    normalized = str(status or "").casefold()
    if normalized in {"failed", "error"}:
        return "fail"
    if normalized == "skipped":
        return "skipped"
    if any(str(item.get("severity") or "").casefold() in {"reject", "error", "fail"} for item in findings):
        return "fail"
    if normalized == "warning" or any(
        str(item.get("severity") or "").casefold() in {"review", "warning", "uncertain"}
        for item in findings
    ):
        return "review"
    return "pass" if normalized in {"completed", "reused", "ready", "pass"} else "unknown"


def _finding_interval(item: dict[str, Any], fps: float) -> dict[str, Any] | None:
    frame_range = _frame_range(item)
    if frame_range is None:
        return None
    start, end = frame_range
    severity = str(item.get("severity") or "review").casefold()
    state = "invalid" if severity in {"reject", "error", "fail"} else "review"
    result = {
        "start_frame": start,
        "end_frame": end,
        "state": state,
        "reason": str(item.get("reason") or ""),
        "severity": severity,
    }
    if fps > 0:
        result["start_time"] = round(start / fps, 6)
        result["end_time"] = round(end / fps, 6)
    if item.get("confidence") is not None:
        try:
            result["confidence"] = max(0.0, min(1.0, float(item["confidence"])))
        except (TypeError, ValueError):
            pass
    return result


def _segment_intervals(segments: list[dict[str, Any]], state: str, fps: float) -> list[dict[str, Any]]:
    intervals = []
    for segment in segments:
        if str(segment.get("state") or "") != state:
            continue
        frame_range = _frame_range(segment)
        if frame_range is None:
            continue
        start, end = frame_range
        item: dict[str, Any] = {
            "start_frame": start,
            "end_frame": end,
            "state": state,
            "reason": str(segment.get("reason") or ""),
        }
        if fps > 0:
            item["start_time"] = round(start / fps, 6)
            item["end_time"] = round(end / fps, 6)
        intervals.append(item)
    return intervals


def build_quality_evidence(
    *,
    dataset_id: str,
    episode_id: str,
    frame_count: int,
    fps: float,
    stages: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    segments: list[dict[str, Any]] | None = None,
    pipeline_version: int | str | None = None,
    pipeline_schema: str | None = None,
    run_id: str | None = None,
    timeline_id: str | None = None,
    source_video: dict[str, Any] | None = None,
    source_signatures: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
    artifact_paths: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a normalized evidence document while preserving stage-specific metrics."""
    normalized_fps = max(0.0, float(fps or 0.0))
    stage_documents: list[dict[str, Any]] = []
    normalized_findings = [item for item in findings if isinstance(item, dict)]
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        stage_id = str(stage.get("id") or "unknown")
        stage_findings = [item for item in normalized_findings if str(item.get("stage") or "") == stage_id]
        intervals = [
            interval
            for item in stage_findings
            if (interval := _finding_interval(item, normalized_fps)) is not None
        ]
        invalid_ranges = _merge_ranges([
            (item["start_frame"], item["end_frame"])
            for item in intervals
            if item["state"] == "invalid"
        ], frame_count)
        review_ranges = _merge_ranges([
            (item["start_frame"], item["end_frame"])
            for item in intervals
            if item["state"] == "review"
        ], frame_count)
        metrics = _json_safe(deepcopy(stage.get("metrics") or {}))
        if not isinstance(metrics, dict):
            metrics = {"value": metrics}
        metrics.update({
            "invalid_frame_count": _range_count(invalid_ranges),
            "review_frame_count": _range_count(review_ranges),
        })
        stage_documents.append({
            "check_id": f"{stage_id}.v1",
            "stage": stage_id,
            "name": str(stage.get("name") or stage_id),
            "status": str(stage.get("status") or "unknown"),
            "verdict": _verdict(str(stage.get("status") or ""), stage_findings),
            "message": str(stage.get("message") or ""),
            "measurements": metrics,
            "intervals": intervals,
            "tags": ["quality", stage_id],
            "finding_count": len(stage_findings),
        })

    all_segments = [item for item in (segments or []) if isinstance(item, dict)]
    invalid_intervals = _segment_intervals(all_segments, "invalid", normalized_fps)
    review_intervals = _segment_intervals(all_segments, "uncertain", normalized_fps)
    if all_segments:
        invalid_ranges = _merge_ranges([
            (item["start_frame"], item["end_frame"]) for item in invalid_intervals
        ], frame_count)
        review_ranges = _merge_ranges([
            (item["start_frame"], item["end_frame"]) for item in review_intervals
        ], frame_count)
    else:
        invalid_ranges = _merge_ranges([
            (item["start_frame"], item["end_frame"])
            for stage in stage_documents
            for item in stage["intervals"]
            if item["state"] == "invalid"
        ], frame_count)
        review_ranges = _merge_ranges([
            (item["start_frame"], item["end_frame"])
            for stage in stage_documents
            for item in stage["intervals"]
            if item["state"] == "review"
        ], frame_count)

    review_ranges = _subtract_ranges(review_ranges, invalid_ranges)
    invalid_count = _range_count(invalid_ranges)
    review_count = _range_count(review_ranges)
    bounded_frame_count = max(0, int(frame_count or 0))
    valid_count = max(0, bounded_frame_count - invalid_count - review_count)
    check_verdicts = {str(item.get("verdict") or "unknown") for item in stage_documents}
    aggregate_verdict = (
        "fail" if invalid_count
        else "review" if review_count
        else "fail" if "fail" in check_verdicts
        else "review" if "review" in check_verdicts
        else "pass" if "pass" in check_verdicts
        else "skipped" if check_verdicts and check_verdicts <= {"skipped"}
        else "unknown"
    )
    source = _json_safe(deepcopy(source_video or {}))
    signatures = _json_safe(deepcopy(source_signatures or []))
    safe_config = _json_safe(deepcopy(config or {}))
    return {
        "schema": QUALITY_EVIDENCE_SCHEMA,
        "version": QUALITY_EVIDENCE_VERSION,
        "created_at": _utc_now(),
        "dataset_id": str(dataset_id),
        "episode_id": str(episode_id),
        "frame_index_space": "full_analysis_video" if timeline_id else "source_video",
        "frame_count": bounded_frame_count,
        "fps": normalized_fps,
        "pipeline": {
            "schema": str(pipeline_schema or "alice/paper-curation/v1"),
            "version": pipeline_version,
            "run_id": run_id,
            "timeline_id": timeline_id,
        },
        "source": {
            "video": source,
            "source_signature_count": len(signatures),
            "source_fingerprint": _fingerprint(signatures),
        },
        "checks": stage_documents,
        "intervals": {
            "invalid": invalid_intervals,
            "review": review_intervals,
        },
        "aggregate": {
            "verdict": aggregate_verdict,
            "valid_frame_count": valid_count,
            "review_frame_count": review_count,
            "invalid_frame_count": invalid_count,
            "invalid_range_count": len(invalid_ranges),
            "review_range_count": len(review_ranges),
        },
        "provenance": {
            "config_fingerprint": _fingerprint(safe_config),
            "source_signatures": signatures,
            "artifact_paths": _json_safe(deepcopy(artifact_paths or {})),
        },
    }
