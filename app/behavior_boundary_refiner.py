from __future__ import annotations

"""Refine coarse VLM behavior boundaries with aligned joint motion.

The VLM remains responsible for behavior semantics.  This module only moves
the split between two adjacent semantic segments to a nearby, mechanically
observed change in joint-motion intensity.  It deliberately has no artifact
or source-dataset write path so it can be used before a behavior annotation is
committed to ``.alicePD``.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np


def load_episode_joint_pose(
    manifest: dict,
    episode: dict,
    *,
    frame_count: int | None = None,
    alignment: dict | None = None,
    reference_media_file_id: str | None = None,
) -> np.ndarray | None:
    """Load the same real, video-aligned Joint Pose used by paper curation.

    Missing/unsupported joint data is an optional refinement input, not a
    behavior-annotation failure, so this adapter returns ``None`` on failure.
    ``alignment`` can be supplied by a caller that has already scanned the
    Episode; otherwise the cached alignment for ``reference_media_file_id``
    is used. Omitting the media id preserves the primary-video behavior.
    """

    try:
        # Keep these imports local: curation does not depend on behavior
        # annotation, and importing its private loader here avoids a cycle.
        from .curation_pipeline import _load_signal_bundle
        from .sensor_alignment import get_valid_sensor_alignment

        target_count = int(frame_count if frame_count is not None else episode.get("frame_count") or 0)
        if target_count <= 0:
            return None
        resolved_alignment = alignment
        if resolved_alignment is None:
            resolved_alignment = get_valid_sensor_alignment(
                manifest,
                episode,
                reference_media_file_id=reference_media_file_id,
            )
        bundle = _load_signal_bundle(manifest, episode, resolved_alignment or {}, frame_count=target_count)
        values = bundle.get("joint")
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float64)
        return array if array.ndim >= 1 and array.shape[0] >= 2 else None
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None


def _fallback_segments(segments: Sequence[Mapping[str, Any]]) -> list[dict]:
    output = [deepcopy(dict(segment)) for segment in segments]
    for segment in output:
        segment["boundary_source"] = "vlm"
    return output


def _mapping_pose(value: Mapping[str, Any]) -> Any:
    for key in ("joint_pose", "joint", "values"):
        if key in value and value[key] is not None:
            return value[key]
    return None


def _coerce_pose(joint_pose: Any, frame_count: int) -> np.ndarray | None:
    if isinstance(joint_pose, Mapping):
        joint_pose = _mapping_pose(joint_pose)
    if joint_pose is None:
        return None
    try:
        values = np.asarray(joint_pose, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if values.ndim == 0 or not values.size or values.shape[0] < 3:
        return None

    # Accept both one SE(3) trajectory [T,4,4] and a skeleton trajectory
    # [T,J,4,4].  Rotation-matrix entries must not become 16 pseudo-joints;
    # behavior timing only needs the world-space translation trajectories.
    if values.ndim >= 3 and values.shape[-2:] == (4, 4):
        values = values[..., :3, 3]
    elif values.ndim > 3:
        return None
    if values.ndim == 1:
        values = values[:, None]
    else:
        values = values.reshape(values.shape[0], -1)
    if not values.shape[1]:
        return None

    finite_columns = np.isfinite(values).sum(axis=0) >= 2
    if not finite_columns.any():
        return None
    values = values[:, finite_columns]
    source_x = np.arange(values.shape[0], dtype=np.float64)
    for column in range(values.shape[1]):
        finite = np.isfinite(values[:, column])
        values[:, column] = np.interp(source_x, source_x[finite], values[finite, column])

    # Curation normally supplies an already aligned matrix.  The normalized
    # interpolation is a safe fallback for callers passing a raw pose array.
    if values.shape[0] != frame_count:
        targets = np.linspace(0.0, values.shape[0] - 1.0, frame_count)
        values = np.column_stack(
            [np.interp(targets, source_x, values[:, column]) for column in range(values.shape[1])]
        )
    return values


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, min(int(window), len(values)))
    if window <= 1:
        return values
    if window % 2 == 0:
        window = max(1, window - 1)
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, np.ones(window, dtype=np.float64) / window, mode="valid")


def _motion_change_score(pose: np.ndarray, fps: float) -> np.ndarray | None:
    delta = np.diff(pose, axis=0, prepend=pose[:1])
    median = np.median(delta, axis=0)
    mad = np.median(np.abs(delta - median), axis=0) * 1.4826
    q90 = np.percentile(np.abs(delta), 90, axis=0)
    scale = np.maximum(mad, q90 * 0.1)
    useful = scale > 1e-10
    if not useful.any():
        return None
    normalized = delta[:, useful] / scale[useful]
    intensity = np.sqrt(np.mean(np.square(normalized), axis=1))
    intensity = _moving_average(intensity, max(1, round(fps * 0.1)))

    # A split at frame b separates [..., b-1] from [b, ...].  Compare short
    # sustained windows rather than a single derivative so tracking jitter is
    # less likely to pull the VLM boundary away from a true stop/start event.
    span = max(1, round(fps * 0.15))
    prefix = np.r_[0.0, np.cumsum(intensity)]
    score = np.zeros(len(intensity), dtype=np.float64)
    for split in range(1, len(intensity)):
        left = max(0, split - span)
        right = min(len(intensity), split + span)
        before = (prefix[split] - prefix[left]) / max(1, split - left)
        after = (prefix[right] - prefix[split]) / max(1, right - split)
        score[split] = abs(after - before)
    return score


def joint_motion_change_score(joint_pose: Any | None, fps: float, frame_count: int) -> np.ndarray | None:
    """Return a video-aligned motion-change score for adaptive VLM sampling."""

    try:
        total = int(frame_count)
        rate = float(fps)
    except (TypeError, ValueError):
        return None
    if total <= 0 or not np.isfinite(rate) or rate <= 0:
        return None
    pose = _coerce_pose(joint_pose, total)
    if pose is None:
        return None
    score = _motion_change_score(pose, rate)
    return score if score is not None and score.shape == (total,) else None


def _original_splits(segments: list[dict], first: int, last: int) -> list[int]:
    count = len(segments)
    splits: list[int] = []
    previous = first
    for index, (left, right) in enumerate(zip(segments, segments[1:])):
        left_end = int(left.get("end_frame", previous) or 0) + 1
        right_start = int(right.get("start_frame", left_end) or 0)
        proposed = int(round((left_end + right_start) / 2.0))
        minimum = previous + 1
        maximum = last - (count - index - 2)
        proposed = max(minimum, min(proposed, maximum))
        splits.append(proposed)
        previous = proposed
    return splits


def refine_behavior_boundaries(
    segments: Sequence[Mapping[str, Any]],
    fps: float,
    frame_count: int,
    joint_pose: Any | None,
    *,
    search_seconds: float = 0.5,
) -> list[dict]:
    """Return VLM segments whose internal splits follow nearby Joint motion.

    Frame intervals are inclusive.  When refinement is available, adjacent
    output segments tile the original outer interval: the left segment ends at
    ``split - 1`` and the right segment starts at ``split``.  The first start,
    final end, input order and at least one frame per segment are preserved.
    With no usable Joint Pose, the input is returned unchanged except for the
    required ``boundary_source='vlm'`` provenance field.
    """

    fallback = _fallback_segments(segments)
    count = len(fallback)
    try:
        total = int(frame_count)
        rate = float(fps)
    except (TypeError, ValueError):
        return fallback
    if count < 2 or total <= 0 or not np.isfinite(rate) or rate <= 0:
        return fallback

    first = max(0, min(int(fallback[0].get("start_frame", 0) or 0), total - 1))
    last = max(first, min(int(fallback[-1].get("end_frame", total - 1) or 0), total - 1))
    if last - first + 1 < count:
        return fallback
    score = joint_motion_change_score(joint_pose, rate, total)
    if score is None:
        return fallback

    original = _original_splits(fallback, first, last)
    radius = max(1, int(round(max(0.0, float(search_seconds)) * rate)))
    finite_scores = score[np.isfinite(score)]
    baseline = float(np.median(finite_scores)) if finite_scores.size else 0.0
    dispersion = float(np.median(np.abs(finite_scores - baseline)) * 1.4826) if finite_scores.size else 0.0
    evidence_floor = max(1e-8, baseline + dispersion)

    selected: list[int] = []
    accepted: list[bool] = []
    for index, proposed in enumerate(original):
        minimum = (selected[-1] + 1) if selected else first + 1
        maximum = last - (len(original) - index - 1)
        low = max(minimum, proposed - radius)
        high = min(maximum, proposed + radius)
        if low > high:
            selected.append(max(minimum, min(proposed, maximum)))
            accepted.append(False)
            continue
        candidates = np.arange(low, high + 1, dtype=np.int64)
        local_scores = score[candidates]
        best_score = float(np.nanmax(local_scores))
        if not np.isfinite(best_score) or best_score <= evidence_floor:
            selected.append(max(minimum, min(proposed, maximum)))
            accepted.append(False)
            continue
        near_best = candidates[np.isclose(local_scores, best_score, rtol=1e-9, atol=1e-12)]
        best = int(near_best[np.argmin(np.abs(near_best - proposed))])
        selected.append(best)
        accepted.append(True)

    output = [deepcopy(segment) for segment in fallback]
    for index, segment in enumerate(output):
        start = first if index == 0 else selected[index - 1]
        end = last if index == count - 1 else selected[index] - 1
        segment["start_frame"] = start
        segment["end_frame"] = end
        segment["start_time"] = round(start / rate, 3)
        segment["end_time"] = round(end / rate, 3)
        segment["boundary_source"] = "vlm"
    for index, was_refined in enumerate(accepted):
        if was_refined:
            output[index]["boundary_source"] = "joint_refined"
            output[index + 1]["boundary_source"] = "joint_refined"
    return output
