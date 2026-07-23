from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any


def _with_times(segment: dict, fps: float) -> dict:
    start = int(segment["start_frame"])
    end = int(segment["end_frame"])
    return {
        **segment,
        "start_frame": start,
        "end_frame": end,
        "start_time": round(start / fps, 3),
        "end_time": round(end / fps, 3),
    }


def _merge_adjacent(segments: list[dict], fps: float) -> list[dict]:
    merged: list[dict] = []
    for segment in sorted(segments, key=lambda item: (int(item["start_frame"]), int(item["end_frame"]))):
        normalized = _with_times(segment, fps)
        if merged:
            previous = merged[-1]
            compatible = (
                previous.get("state") == normalized.get("state")
                and previous.get("reason") == normalized.get("reason")
                and float(previous.get("confidence", 0.0)) == float(normalized.get("confidence", 0.0))
                and int(previous["end_frame"]) + 1 >= int(normalized["start_frame"])
            )
            if compatible:
                previous["end_frame"] = max(int(previous["end_frame"]), int(normalized["end_frame"]))
                previous["end_time"] = round(int(previous["end_frame"]) / fps, 3)
                continue
        merged.append(normalized)
    return merged


def apply_segment_override(payload: dict, edit: dict, frame_count: int, fps: float) -> dict:
    if frame_count <= 0 or fps <= 0:
        raise ValueError("视频帧数或帧率无效")
    start = int(edit.get("start_frame", 0))
    end = int(edit.get("end_frame", start))
    if end < start:
        raise ValueError("结束帧不能早于开始帧")
    if start < 0 or end >= frame_count:
        raise ValueError(f"区间必须位于 0 到 {frame_count - 1} 帧之间")
    state = str(edit.get("state") or "")
    if state not in {"valid", "invalid", "uncertain"}:
        raise ValueError("区间状态无效")

    edit_source = str(edit.get("source") or "manual_range")[:64]
    edit_metadata = {
        key: edit[key]
        for key in (
            "behavior_phase",
            "behavior_label",
            "behavior_removal_id",
            "behavior_annotation_created_at",
        )
        if edit.get(key) is not None
    }
    replacement = _with_times({
        "start_frame": start,
        "end_frame": end,
        "state": state,
        "reason": str(edit.get("reason") or "人工指定区间")[:500],
        "confidence": max(0.0, min(1.0, float(edit.get("confidence", 1.0)))),
        "source": edit_source,
        **edit_metadata,
    }, fps)
    output: list[dict] = []
    for current in payload.get("segments", []):
        current_start = max(0, int(current.get("start_frame", 0)))
        current_end = min(frame_count - 1, int(current.get("end_frame", current_start)))
        if current_start > current_end:
            continue
        normalized = {**current, "start_frame": current_start, "end_frame": current_end}
        if current_end < start or current_start > end:
            output.append(normalized)
            continue
        if current_start < start:
            output.append({**normalized, "end_frame": start - 1})
        if current_end > end:
            output.append({**normalized, "start_frame": end + 1})
    output.append(replacement)
    segments = _merge_adjacent(output, fps)

    samples = []
    for sample in payload.get("samples", []):
        frame = int(sample.get("frame", -1))
        if start <= frame <= end:
            samples.append({**sample, "state": state, "reason": replacement["reason"], "confidence": replacement["confidence"], "manual_override": True})
        else:
            samples.append(sample)

    edits = list(payload.get("manual_edits") or [])
    edits.append({
        "id": uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **replacement,
    })
    valid_frames = sum(int(item["end_frame"]) - int(item["start_frame"]) + 1 for item in segments if item.get("state") == "valid")
    invalid_frames = sum(int(item["end_frame"]) - int(item["start_frame"]) + 1 for item in segments if item.get("state") == "invalid")
    summary = {
        **(payload.get("summary") or {}),
        "segment_count": len(segments),
        "valid_count": sum(item.get("state") == "valid" for item in segments),
        "invalid_count": sum(item.get("state") == "invalid" for item in segments),
        "uncertain_count": sum(item.get("state") == "uncertain" for item in segments),
        "valid_frame_count": valid_frames,
        "invalid_frame_count": invalid_frames,
        "manual_edit_count": len(edits),
    }
    return {
        **payload,
        "segments": segments,
        "samples": samples,
        "manual_edits": edits,
        "summary": summary,
    }


def _readable_label(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, dict):
        return str(value).strip()
    for key in ("value", "label", "name", "en", "english", "zh", "zh_cn", "chinese"):
        text = str(value.get(key) or "").strip()
        if text:
            return text
    return ""


def behavior_phase_key(value: Any) -> str:
    text = _readable_label(value).casefold().replace("-", "_").replace(" ", "_")
    return re.sub(r"_+", "_", text).strip("_")


def _segment_phase(segment: dict) -> tuple[str, str]:
    raw = (
        segment.get("phase_label")
        or segment.get("phase")
        or segment.get("stage_label")
        or segment.get("stage")
        or segment.get("label")
    )
    return behavior_phase_key(raw), _readable_label(raw)


def _fill_uncovered_segments(payload: dict, frame_count: int, fps: float) -> dict:
    """Keep unlabelled frames usable before applying a behavior exclusion."""
    source_segments = sorted(
        (item for item in payload.get("segments", []) if isinstance(item, dict)),
        key=lambda item: (int(item.get("start_frame", 0)), int(item.get("end_frame", item.get("start_frame", 0)))),
    )
    output: list[dict] = []
    cursor = 0
    for source in source_segments:
        start = max(cursor, max(0, min(frame_count - 1, int(source.get("start_frame", 0)))))
        end = max(start, min(frame_count - 1, int(source.get("end_frame", start))))
        if start > cursor:
            output.append(_with_times({
                "start_frame": cursor,
                "end_frame": start - 1,
                "state": "valid",
                "reason": "未被按动作去除",
                "confidence": 1.0,
                "source": "behavior_phase_keep",
            }, fps))
        output.append(_with_times({**source, "start_frame": start, "end_frame": end}, fps))
        cursor = end + 1
        if cursor >= frame_count:
            break
    if cursor < frame_count:
        output.append(_with_times({
            "start_frame": cursor,
            "end_frame": frame_count - 1,
            "state": "valid",
            "reason": "未被按动作去除",
            "confidence": 1.0,
            "source": "behavior_phase_keep",
        }, fps))
    return {**payload, "segments": output}


def _covered_frame_count(intervals: list[tuple[int, int]]) -> int:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start + 1 for start, end in merged)


def apply_behavior_phase_exclusion(
    payload: dict,
    behavior: dict,
    phase_label: str,
    frame_count: int,
    fps: float,
    reason: str | None = None,
) -> dict:
    if frame_count <= 0 or fps <= 0:
        raise ValueError("视频帧数或帧率无效")
    requested = behavior_phase_key(phase_label)
    if not requested:
        raise ValueError("请选择要去除的 VLM 动作")
    matches: list[dict] = []
    display_label = phase_label.strip()
    for segment in behavior.get("segments", []):
        if not isinstance(segment, dict):
            continue
        key, label = _segment_phase(segment)
        if key != requested:
            continue
        start = max(0, min(frame_count - 1, int(segment.get("start_frame", 0))))
        end = max(start, min(frame_count - 1, int(segment.get("end_frame", start))))
        matches.append({**segment, "start_frame": start, "end_frame": end})
        display_label = label or display_label
    if not matches:
        raise ValueError(f"当前 VLM 行为标注中没有动作：{phase_label}")

    annotation_created_at = str(behavior.get("created_at") or "")
    existing_removals = list(payload.get("behavior_removals") or [])
    if any(
        behavior_phase_key(item.get("phase_label")) == requested
        and str(item.get("behavior_annotation_created_at") or "") == annotation_created_at
        for item in existing_removals
        if isinstance(item, dict)
    ):
        raise ValueError(f"动作“{display_label}”已经标记为去除")

    result = _fill_uncovered_segments(payload, frame_count, fps)
    removal_id = uuid.uuid4().hex
    removal_reason = (reason or f"按 VLM 动作去除：{display_label}").strip()[:500]
    for segment in matches:
        result = apply_segment_override(result, {
            "start_frame": segment["start_frame"],
            "end_frame": segment["end_frame"],
            "state": "invalid",
            "reason": removal_reason,
            "confidence": 1.0,
            "source": "behavior_phase",
            "behavior_phase": requested,
            "behavior_label": display_label,
            "behavior_removal_id": removal_id,
            "behavior_annotation_created_at": annotation_created_at,
        }, frame_count, fps)

    intervals = [(int(item["start_frame"]), int(item["end_frame"])) for item in matches]
    removal = {
        "id": removal_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase_label": requested,
        "display_label": display_label,
        "reason": removal_reason,
        "segment_count": len(matches),
        "frame_count": _covered_frame_count(intervals),
        "intervals": [[start, end] for start, end in intervals],
        "behavior_annotation_created_at": annotation_created_at,
        "behavior_artifact_version": behavior.get("artifact_version"),
    }
    removals = [*existing_removals, removal]
    all_intervals = [
        (int(interval[0]), int(interval[1]))
        for item in removals
        if isinstance(item, dict)
        for interval in item.get("intervals", [])
        if isinstance(interval, (list, tuple)) and len(interval) == 2
    ]
    summary = {
        **(result.get("summary") or {}),
        "behavior_removal_count": len(removals),
        "behavior_removed_phase_count": len({behavior_phase_key(item.get("phase_label")) for item in removals if isinstance(item, dict)}),
        "behavior_removed_frame_count": _covered_frame_count(all_intervals),
    }
    return {**result, "behavior_removals": removals, "summary": summary}
