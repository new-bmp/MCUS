from __future__ import annotations

import json
import math
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import h5py
import numpy as np

from .egodex_mano import (
    EGODEX_MANO_REVISION,
    egodex_mano_source_names,
    fit_egodex_mano_template,
    has_egodex_mano_source,
    retarget_egodex_mano_frame,
    source_is_retargeted,
)
from .dataset_modes import dataset_mode
from .lerobot_export import (
    HAND_21_JOINT_NAMES,
    _sample_repaired_transforms,
    side_hand_joint_names,
    write_lerobot_metadata,
    write_lerobot_pair,
)
from .sensor_alignment import aligned_sensor_positions, aligned_sensor_rows
from .s1_repair import load_s1_repair, s1_repair_cell_count
from .temporal_resampling import sample_hdf5_numeric
from .video_smoothing import _create_frame_writer, target_stabilization_matrices


FULL_DATASET_SCHEMA = "alice/full-dataset/v2"
LEGACY_FULL_DATASET_SCHEMA = "alice/full-mano-dataset/v1"
FULL_PAIR_SCHEMA = "alice/full-mano-pair/v1"
SUBTASK_JSON_SCHEMA = "alice/episode-subtasks/v1"
SUBTASK_JSON_OUTPUT_FORMAT = "subtask_json"
EPISODE_LEROBOT_JSON_OUTPUT_FORMAT = "episode_lerobot_json"
FULL_EXPORT_PIPELINE_VERSION = 8
DEFAULT_FULL_OUTPUT_FORMAT = "lerobot"
SUPPORTED_FULL_OUTPUT_FORMATS = {"lerobot", "hdf5_mp4", SUBTASK_JSON_OUTPUT_FORMAT, EPISODE_LEROBOT_JSON_OUTPUT_FORMAT}
REMOVED_VLM_PHASES = {"idle"}
DEFAULT_MAX_INTERNAL_GAP_SECONDS = 0.25
DEFAULT_MIN_CLIP_SECONDS = 0.75
_INDEX_LOCK = threading.RLock()


def _side_joint_names(side: str) -> list[str]:
    return [f"{side}Forearm", *side_hand_joint_names(side)]


MANO_44_JOINT_NAMES = tuple([*_side_joint_names("left"), *_side_joint_names("right")])


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _episode_records(manifest: dict, episode: dict) -> list[dict]:
    episode_id = str(episode.get("id") or "")
    episode_key = str(episode.get("episode_key") or "")
    assignments = (manifest.get("episode_resolution") or {}).get("file_episode_assignments") or {}
    return [
        item
        for item in manifest.get("files") or []
        if str(assignments.get(str(item.get("id") or "")) or item.get("episode_id") or "") == episode_id
        or (not assignments.get(str(item.get("id") or "")) and not item.get("episode_id") and str(item.get("episode_key") or "") == episode_key)
    ]


def _find_transform_source(
    manifest: dict,
    episode: dict,
    output_format: str = "hdf5_mp4",
    *,
    prefer_applied: bool = True,
) -> tuple[Path, str, int]:
    root = Path(str(manifest["root_path"])).expanduser().resolve()
    required_names = MANO_44_JOINT_NAMES if output_format == "hdf5_mp4" else (
        *side_hand_joint_names("left"),
        *side_hand_joint_names("right"),
    )
    if prefer_applied:
        from .projection_correction import active_projection_source

        applied = active_projection_source(manifest, episode)
    else:
        applied = None
    if applied is not None:
        try:
            with h5py.File(applied["path"], "r") as source:
                transforms = source.get("transforms")
                if isinstance(transforms, h5py.Group):
                    names = (*required_names, "camera")
                    counts = {int(transforms[name].shape[0]) for name in names if name in transforms}
                    if len(counts) == 1 and all(name in transforms and transforms[name].shape[1:] == (4, 4) for name in names):
                        return Path(applied["path"]), str(applied.get("source_relative_path") or "projection-correction.hdf5"), counts.pop()
        except (OSError, KeyError, ValueError):
            pass
    candidates = [
        item for item in _episode_records(manifest, episode)
        if Path(str(item.get("relative_path") or "")).suffix.casefold() in {".h5", ".hdf5", ".h5df"}
    ]
    missing: list[str] = []
    for record in candidates:
        relative = str(record.get("relative_path") or "").replace("\\", "/")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
            with h5py.File(path, "r") as source:
                transforms = source.get("transforms")
                if not isinstance(transforms, h5py.Group):
                    continue
                absent = [name for name in (*required_names, "camera") if name not in transforms]
                if absent:
                    missing.extend(absent)
                    continue
                counts = {int(transforms[name].shape[0]) for name in (*required_names, "camera")}
                if len(counts) != 1 or any(transforms[name].shape[1:] != (4, 4) for name in (*required_names, "camera")):
                    continue
                return path, relative, counts.pop()
        except (OSError, KeyError, ValueError):
            continue
    detail = f"; missing: {', '.join(sorted(set(missing))[:8])}" if missing else ""
    required_label = "MANO 44x4x4" if output_format == "hdf5_mp4" else "left/right hand 21x4x4"
    raise RuntimeError(f"Episode has no HDF5 source with {required_label} + camera{detail}")


def _aligned_rows(
    manifest: dict,
    episode: dict,
    relative: str,
    source_count: int,
    video_count: int,
    reference_media_file_id: str | None = None,
) -> np.ndarray:
    return aligned_sensor_rows(
        manifest,
        episode,
        relative,
        source_count,
        video_count,
        reference_media_file_id=reference_media_file_id,
        require_complete=True,
    )


def _aligned_positions(
    manifest: dict,
    episode: dict,
    relative: str,
    source_count: int,
    video_count: int,
    media: dict,
) -> np.ndarray:
    from .projection_correction import active_projection_source

    projection = active_projection_source(manifest, episode)
    if projection is not None:
        try:
            projection_path = Path(projection["path"]).expanduser().resolve()
            selected_path, _, _ = _find_transform_source(manifest, episode)
        except (KeyError, OSError, RuntimeError, ValueError):
            projection_path = None
            selected_path = None
        if projection_path is not None and selected_path is not None and projection_path == selected_path.resolve():
            positions = _source_video_positions(media, video_count)
            if positions.shape == (video_count,) and np.isfinite(positions).all() and np.all(positions >= 0.0) and np.all(positions <= source_count - 1):
                return positions
            raise RuntimeError("Projection correction timeline does not align with the Full analysis video")
    try:
        return aligned_sensor_positions(
            manifest,
            episode,
            relative,
            source_count,
            video_count,
            source_frame_positions=media.get("source_frame_positions") or None,
            reference_media_file_id=str(media.get("file_id") or "") or None,
            require_complete=True,
        )
    except RuntimeError:
        # Applied projection snapshots are stored one row per source-video
        # frame but deliberately retain the immutable source relative path.
        # They may therefore be absent from the persisted raw T0 stream list.
        positions = _source_video_positions(media, video_count)
        if positions.shape == (video_count,) and np.isfinite(positions).all() and np.all(positions >= 0.0) and np.all(positions <= source_count - 1):
            return positions
        raise


def _source_video_positions(media: dict, frame_count: int) -> np.ndarray:
    positions = np.asarray(media.get("source_frame_positions") or [], dtype=np.float64).reshape(-1)
    return positions if positions.shape == (frame_count,) else np.arange(frame_count, dtype=np.float64)


def _camera_image_transforms(media: dict, frame_count: int) -> np.ndarray:
    matrices = target_stabilization_matrices(media, frame_count)
    return matrices if matrices is not None else np.repeat(np.eye(3, dtype=np.float64)[None], frame_count, axis=0)


def _quality_states(frame_count: int, curation: dict) -> np.ndarray:
    invalid, review = _curation_state_masks(frame_count, curation)
    states = np.full(frame_count, "valid", dtype=object)
    states[review] = "review"
    states[invalid] = "invalid"
    return states


def _build_export_action_payload(
    source_path: Path,
    source_relative: str,
    source_rows: np.ndarray,
    repair: dict | None,
    action_report: dict | None,
) -> dict | None:
    if not action_report:
        return None
    from .action_mapping import OBSERVATION_FIELDS, build_action_arrays

    profile = dict(action_report.get("profile") or {})
    config = dict(action_report.get("config") or {})
    profile_id = str(config.get("profile_id") or profile.get("id") or "")
    horizon = int(config.get("horizon_frames") or 0)
    frame_count = int(np.asarray(source_rows).size)
    if not profile_id or horizon <= 0 or frame_count <= horizon:
        raise RuntimeError("Full Action 配置缺少有效 profile 或 horizon")
    required = [
        "camera",
        "leftHand", "leftThumbTip", "leftIndexFingerTip", "leftMiddleFingerKnuckle",
        "rightHand", "rightThumbTip", "rightIndexFingerTip", "rightMiddleFingerKnuckle",
    ]
    with h5py.File(source_path, "r") as source:
        transforms = source.get("transforms")
        if not isinstance(transforms, h5py.Group):
            raise RuntimeError("Full Action 源 HDF5 缺少 transforms")
        missing = [name for name in required if name not in transforms]
        if missing:
            raise RuntimeError(f"Full Action 缺少变换字段: {', '.join(missing)}")
        sampled = {
            name: _sample_repaired_transforms(
                transforms[name], source_rows, repair, source_relative, f"transforms/{name}",
            ).astype(np.float32)
            for name in required
        }
    observation, action, _ = build_action_arrays(
        sampled,
        profile,
        str(config.get("source_hand") or "right"),
        str(config.get("coordinate_frame") or "camera"),
        horizon,
    )
    action_dim = int(profile.get("action_dim") or action.shape[1])
    full_observation = np.full((frame_count, 20), np.nan, dtype=np.float32)
    full_action = np.full((frame_count, action_dim), np.nan, dtype=np.float32)
    target_frames = np.full(frame_count, -1, dtype=np.int64)
    valid = np.zeros(frame_count, dtype=bool)
    count = int(action.shape[0])
    full_observation[:count] = observation
    full_action[:count] = action
    target_frames[:count] = np.arange(horizon, horizon + count, dtype=np.int64)
    valid[:count] = np.isfinite(observation).all(axis=1) & np.isfinite(action).all(axis=1)
    return {
        "observation_state": full_observation,
        "action": full_action,
        "target_frame_index": target_frames,
        "valid": valid,
        "action_profile_id": profile_id,
        "action_dim": action_dim,
        "action_fields": list(profile.get("fields") or []),
        "action_representation": str(profile.get("representation") or "unknown"),
        "action_coordinate_frame": str(config.get("coordinate_frame") or "camera"),
        "action_horizon_frames": horizon,
        "observation_state_fields": list(OBSERVATION_FIELDS),
        "source_hand": str(config.get("source_hand") or "right"),
        "robot_family": str(profile.get("robot_family") or "unknown"),
        "control_space": str(profile.get("control_space") or "unknown"),
    }


def _slice_action_payload(payload: dict | None, frames: np.ndarray) -> dict | None:
    if payload is None:
        return None
    indices = np.asarray(frames, dtype=np.int64).reshape(-1)
    return {
        **{
            key: value
            for key, value in payload.items()
            if key not in {"observation_state", "action", "target_frame_index", "valid"}
        },
        "observation_state": np.asarray(payload["observation_state"])[indices],
        "action": np.asarray(payload["action"])[indices],
        "target_frame_index": np.asarray(payload["target_frame_index"])[indices],
        "valid": np.asarray(payload["valid"], dtype=bool)[indices],
    }


def _action_safe_intervals(
    intervals: list[tuple[int, int]],
    action_payload: dict | None,
    fps: float,
) -> tuple[list[tuple[int, int]], dict]:
    if action_payload is None:
        return intervals, {
            "action_requested": False,
            "action_tail_removed_frame_count": 0,
            "action_short_fragment_removed_frame_count": 0,
        }
    valid = np.asarray(action_payload["valid"], dtype=bool)
    targets = np.asarray(action_payload["target_frame_index"], dtype=np.int64)
    minimum_frames = max(1, int(math.ceil(DEFAULT_MIN_CLIP_SECONDS * max(0.01, fps))))
    output: list[tuple[int, int]] = []
    tail_removed = 0
    short_removed = 0
    for start, end in intervals:
        segment_valid = valid[start:end + 1].copy()
        segment_targets = targets[start:end + 1]
        segment_valid &= (segment_targets >= start) & (segment_targets <= end)
        safe_ranges = _mask_intervals(segment_valid)
        covered = 0
        for local_start, local_end in safe_ranges:
            resolved_start = start + local_start
            resolved_end = start + local_end
            count = resolved_end - resolved_start + 1
            covered += count
            if count < minimum_frames:
                short_removed += count
                continue
            output.append((resolved_start, resolved_end))
        tail_removed += max(0, end - start + 1 - covered)
    return output, {
        "action_requested": True,
        "action_profile_id": action_payload.get("action_profile_id"),
        "action_horizon_frames": int(action_payload.get("action_horizon_frames") or 0),
        "action_tail_removed_frame_count": tail_removed,
        "action_short_fragment_removed_frame_count": short_removed,
    }


def _take_rows(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(rows, return_inverse=True)
    return np.asarray(dataset[unique.tolist()])[inverse]


def _rot6d(transform: np.ndarray) -> np.ndarray:
    rotation = np.asarray(transform[:, :3, :3], dtype=np.float32)
    return rotation[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)


def _phase_key(value: object) -> str:
    return str(value or "unknown").strip().casefold().replace("-", "_").replace(" ", "_")


def _category_name(value: object) -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(value or "other").strip())
    return text.strip(" .")[:80] or "other"


def _mask_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _curation_state_masks(frame_count: int, curation: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return red/bad and yellow/review masks using the final curation ranges."""

    invalid = np.zeros(max(0, int(frame_count)), dtype=bool)
    review = np.zeros_like(invalid)
    if invalid.size == 0:
        return invalid, review
    for segment in curation.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        state = str(segment.get("state") or "").strip().casefold()
        if state not in {"invalid", "uncertain", "review"}:
            continue
        start = max(0, min(invalid.size - 1, int(segment.get("start_frame") or 0)))
        end = max(start, min(invalid.size - 1, int(segment.get("end_frame") or start)))
        if state == "invalid":
            invalid[start:end + 1] = True
        else:
            review[start:end + 1] = True
    review &= ~invalid
    return invalid, review


def _mask_range_records(mask: np.ndarray, findings: list[dict] | None = None) -> list[dict]:
    """Serialize contiguous frame marks, retaining relevant quality reasons."""

    values = np.asarray(mask, dtype=bool)
    records: list[dict] = []
    findings = [item for item in (findings or []) if isinstance(item, dict)]
    for start, end in _mask_intervals(values):
        reasons = sorted({
            f"{str(item.get('stage') or 'quality')}:{str(item.get('reason') or 'marked')}"
            for item in findings
            if int(item.get("start_frame") or item.get("frame") or 0) <= end
            and int(item.get("end_frame") or item.get("frame") or 0) >= start
        })
        records.append({
            "start_frame": int(start),
            "end_frame": int(end),
            "frame_count": int(end - start + 1),
            "reasons": reasons,
        })
    return records


def _subtask_json_path(output_root: Path, episode: dict) -> Path:
    episode_key = _category_name(str(episode.get("id") or episode.get("name") or "episode"))
    return output_root / "episodes" / episode_key / "subtasks.json"


def _export_subtask_json(
    output_root: Path,
    manifest: dict,
    episode: dict,
    curation: dict,
    behavior: dict | None,
    *,
    run_id: str | None,
    timeline_id: str | None,
    output_format: str = SUBTASK_JSON_OUTPUT_FORMAT,
    episode_root: Path | None = None,
) -> dict:
    """Write one frame-indexed subtask document for a source Episode."""

    curation_video = curation.get("source_video") or {}
    frame_count = max(0, int(
        curation_video.get("frame_count")
        or (curation.get("summary") or {}).get("frame_count")
        or episode.get("frame_count")
        or 0
    ))
    fps = max(0.01, float(
        curation_video.get("fps")
        or episode.get("fps")
        or 30.0
    ))
    source_frame_positions = list(curation_video.get("source_frame_positions") or [])
    if len(source_frame_positions) != frame_count:
        source_frame_positions = []
    invalid, review = _curation_state_masks(frame_count, curation)
    findings = curation.get("findings") or []
    invalid_ranges = _mask_range_records(invalid, findings)
    review_ranges = _mask_range_records(review, findings)
    medium_segments = [item for item in ((behavior or {}).get("medium") or []) if isinstance(item, dict)]
    fine_segments = [item for item in ((behavior or {}).get("fine") or (behavior or {}).get("segments") or []) if isinstance(item, dict)]
    source_subtasks = medium_segments
    if not source_subtasks:
        source_subtasks = [{
            "start_frame": 0,
            "end_frame": max(0, frame_count - 1),
            "description": "No medium-level subtask annotation was returned.",
            "confidence": 0.0,
            "boundary_source": "fallback",
        }]
    subtasks: list[dict] = []
    for index, source in enumerate(source_subtasks, start=1):
        if frame_count:
            start = max(0, min(frame_count - 1, int(source.get("start_frame") or 0)))
            end = max(start, min(frame_count - 1, int(source.get("end_frame") or start)))
        else:
            start = max(0, int(source.get("start_frame") or 0))
            end = max(start, int(source.get("end_frame") or start))
        bad_frames = np.flatnonzero(invalid[start:end + 1]).astype(np.int64) + start if frame_count else np.asarray([], dtype=np.int64)
        review_frames = np.flatnonzero(review[start:end + 1]).astype(np.int64) + start if frame_count else np.asarray([], dtype=np.int64)
        state = "bad" if bad_frames.size == end - start + 1 else "review" if review_frames.size else "valid"
        if bad_frames.size and bad_frames.size < end - start + 1:
            state = "mixed"
        nested_fine = []
        for fine in fine_segments:
            fine_start = max(start, int(fine.get("start_frame") or 0))
            fine_end = min(end, int(fine.get("end_frame") or fine_start))
            if fine_end < fine_start:
                continue
            nested_fine.append({
                "start_frame": fine_start,
                "end_frame": fine_end,
                "phase_label": str(fine.get("phase_label") or fine.get("label") or "unknown"),
                "skill": fine.get("skill"),
                "description": str(fine.get("description") or "")[:800],
            })
        description = str(source.get("description") or "").strip()[:800]
        subtasks.append({
            "id": f"subtask_{index:03d}",
            "level": "medium",
            "name": description or f"subtask_{index:03d}",
            "description": description,
            "start_frame": int(start),
            "end_frame": int(end),
            "start_time": round(start / fps, 3),
            "end_time": round(end / fps, 3),
            "frame_count": int(end - start + 1),
            "state": state,
            "bad_frames": [int(value) for value in bad_frames.tolist()],
            "review_frames": [int(value) for value in review_frames.tolist()],
            "bad_frame_count": int(bad_frames.size),
            "review_frame_count": int(review_frames.size),
            "confidence": max(0.0, min(1.0, float(source.get("confidence") or 0.0))),
            "boundary_source": str(source.get("boundary_source") or "unknown"),
            "primary_targets": list(source.get("primary_targets") or []),
            "target_instance": str(source.get("target_instance") or ""),
            "fine_segments": nested_fine,
        })
    path = episode_root / "subtasks.json" if episode_root is not None else _subtask_json_path(output_root, episode)
    quality_evidence = curation.get("quality_evidence")
    quality_evidence_path = path.with_name("quality_evidence.json") if isinstance(quality_evidence, dict) else None
    payload = {
        "schema": SUBTASK_JSON_SCHEMA,
        "output_format": output_format,
        "dataset_id": manifest.get("id"),
        "episode_id": episode.get("id"),
        "episode_name": episode.get("name") or episode.get("id"),
        "full_run_id": run_id,
        "timeline_id": timeline_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frame_index_space": "full_analysis_video" if timeline_id else "source_video",
        "frame_end_inclusive": True,
        "source_frame_positions": source_frame_positions,
        "subtask_level": "medium",
        "fine_level": "fine_segments",
        "frame_count": frame_count,
        "fps": fps,
        "subtask_count": len(subtasks),
        "subtasks": subtasks,
        "bad_frames": [int(value) for value in np.flatnonzero(invalid).tolist()],
        "review_frames": [int(value) for value in np.flatnonzero(review).tolist()],
        "bad_frame_ranges": invalid_ranges,
        "review_frame_ranges": review_ranges,
        "quality_evidence": {
            "schema": quality_evidence.get("schema"),
            "file": quality_evidence_path.name,
        } if quality_evidence_path is not None else None,
        "summary": {
            "bad_frame_count": int(invalid.sum()),
            "review_frame_count": int(review.sum()),
            "valid_frame_count": int(frame_count - invalid.sum() - review.sum()),
            "curation_artifact_path": curation.get("artifact_path"),
            "behavior_artifact_path": (behavior or {}).get("artifacts", {}).get("behavior"),
        },
    }
    _write_json_atomic(path, payload)
    if quality_evidence_path is not None:
        _write_json_atomic(quality_evidence_path, quality_evidence)
    classification = classify_behavior(behavior or {}, 0, max(0, frame_count - 1))
    pair = {
        "id": str(episode.get("id") or path.parent.name),
        "output_format": output_format,
        "episode_id": episode.get("id"),
        "episode_name": episode.get("name") or episode.get("id"),
        "full_run_id": run_id,
        "timeline_id": timeline_id,
        **classification,
        "start_frame": 0,
        "end_frame": max(0, frame_count - 1),
        "frame_count": frame_count,
        "fps": fps,
        "subtask_count": len(subtasks),
        "bad_frame_count": int(invalid.sum()),
        "review_frame_count": int(review.sum()),
        "subtasks_json": str(path),
        "json": str(path),
        "quality_evidence_json": str(quality_evidence_path) if quality_evidence_path is not None else None,
    }
    return {
        "pairs": [pair],
        "filtering": {
            "source_frame_count": frame_count,
            "retained_frame_count": int(frame_count - invalid.sum() - review.sum()),
            "bad_frame_count": int(invalid.sum()),
            "review_frame_count": int(review.sum()),
        },
        "transform_source": None,
        "category": classification["category"],
        "categories": [classification["category"]],
        "output_format": output_format,
        "full_run_id": run_id,
        "timeline_id": timeline_id,
    }


def _export_episode_lerobot_json(
    output_root: Path,
    manifest: dict,
    episode: dict,
    smoothed_media: dict,
    curation: dict,
    behavior: dict | None,
    progress,
    *,
    run_id: str | None,
    timeline_id: str | None,
    action_report: dict | None,
) -> dict:
    """Export one complete source Episode as an independent LeRobot dataset.

    The normal Full LeRobot mode exports only retained clips into one shared
    dataset.  This mode keeps every source-video frame and puts the quality
    decisions in ``subtasks.json`` beside that Episode's LeRobot metadata.
    """

    format_map = manifest.get("format_map") or {}
    capabilities = format_map.get("capabilities") or {}
    if capabilities and capabilities.get("can_full_export") is False:
        blocking_issues = [
            str(item.get("message") or "")
            for item in format_map.get("issues") or []
            if item.get("severity") in {"warning", "error"}
        ]
        detail = "; ".join(item for item in blocking_issues if item) or "源数据没有可验证的左右手 21 点变换与相机变换"
        raise RuntimeError(f"当前格式不能安全输出 Episode LeRobot：{detail}")

    frame_count = int(smoothed_media.get("frame_count") or episode.get("frame_count") or 0)
    if frame_count <= 0:
        raise RuntimeError("Episode 没有可导出的完整视频帧")
    episode_key = _category_name(str(episode.get("id") or episode.get("name") or "episode"))
    episode_root = output_root / "episodes" / episode_key
    episode_root.mkdir(parents=True, exist_ok=True)

    json_episode = {**episode, "frame_count": frame_count, "fps": smoothed_media.get("fps") or episode.get("fps")}
    subtask_result = _export_subtask_json(
        output_root,
        manifest,
        json_episode,
        curation,
        behavior,
        run_id=run_id,
        timeline_id=timeline_id,
        output_format=EPISODE_LEROBOT_JSON_OUTPUT_FORMAT,
        episode_root=episode_root,
    )
    source_path, source_relative, source_count = _find_transform_source(manifest, episode, "lerobot")
    repair = load_s1_repair(curation)
    repair_cell_count = s1_repair_cell_count(repair, source_relative, "transforms/")
    row_map = _aligned_positions(manifest, episode, source_relative, source_count, frame_count, smoothed_media)
    source_video_positions = _source_video_positions(smoothed_media, frame_count)
    camera_image_transforms = _camera_image_transforms(smoothed_media, frame_count)
    quality_states = _quality_states(frame_count, curation)
    action_payload = _build_export_action_payload(
        source_path,
        source_relative,
        row_map,
        repair,
        action_report,
    )
    if action_payload is not None:
        target_frames = np.asarray(action_payload["target_frame_index"], dtype=np.int64)
        quality_valid = quality_states == "valid"
        action_path_valid = np.zeros(frame_count, dtype=bool)
        candidates = (target_frames >= 0) & (target_frames < frame_count)
        if np.any(candidates):
            current_frames = np.arange(frame_count, dtype=np.int64)[candidates]
            candidate_targets = target_frames[candidates]
            path_starts = np.minimum(current_frames, candidate_targets)
            path_ends = np.maximum(current_frames, candidate_targets)
            blocked_prefix = np.concatenate((
                np.zeros(1, dtype=np.int64),
                np.cumsum(~quality_valid, dtype=np.int64),
            ))
            action_path_valid[candidates] = (
                blocked_prefix[path_ends + 1] - blocked_prefix[path_starts]
            ) == 0
        action_payload["valid"] = (
            np.asarray(action_payload["valid"], dtype=bool)
            & action_path_valid
        )
    frames = np.arange(frame_count, dtype=np.int64)
    classification = classify_behavior(behavior or {}, 0, frame_count - 1)
    staging_root = output_root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    video_temporary = staging_root / f".episode-video.{uuid.uuid4().hex}.part.mp4"
    try:
        progress(5.0, "准备 Episode 全帧 LeRobot 数据")
        video_info = _write_video_clip(Path(str(smoothed_media["path"])), video_temporary, 0, frame_count - 1)
        if int(video_info.get("frame_count") or 0) != frame_count or _video_frame_count(video_temporary) != frame_count:
            raise RuntimeError("Episode LeRobot 视频帧数校验失败")
        lerobot_pair = write_lerobot_pair(
            episode_root,
            video_temporary,
            video_info,
            source_path,
            source_relative,
            source_count,
            row_map[frames],
            frames,
            _phase_labels(behavior or {}, frames),
            manifest,
            episode,
            classification,
            repair,
            source_video_positions=source_video_positions[frames],
            camera_image_transforms=camera_image_transforms[frames],
            quality_states=quality_states,
            action_payload=action_payload,
        )
        info_path = write_lerobot_metadata(episode_root, manifest, [lerobot_pair])
        progress(100.0, "已写入 Episode LeRobot + JSON")
    finally:
        video_temporary.unlink(missing_ok=True)

    json_pair = subtask_result["pairs"][0]
    pair = {
        **json_pair,
        "id": str(episode.get("id") or episode_key),
        "output_format": EPISODE_LEROBOT_JSON_OUTPUT_FORMAT,
        "episode_index": lerobot_pair.get("episode_index"),
        "task_index": lerobot_pair.get("task_index"),
        "start_frame": 0,
        "end_frame": frame_count - 1,
        "frame_count": frame_count,
        "fps": video_info.get("fps"),
        "width": video_info.get("width"),
        "height": video_info.get("height"),
        "data": lerobot_pair.get("data"),
        "body": lerobot_pair.get("body"),
        "mp4": lerobot_pair.get("mp4"),
        "meta_info": str(info_path),
        "lerobot_root": str(episode_root),
        "source_hdf5": source_relative,
        "body_joint_names": lerobot_pair.get("body_joint_names") or [],
        "camera_intrinsic": lerobot_pair.get("camera_intrinsic"),
        "s1_repair_applied": repair_cell_count > 0,
        "s1_repair_cell_count": repair_cell_count,
        "action_available": action_payload is not None,
        "action_valid_frame_count": int(np.asarray(action_payload["valid"], dtype=bool).sum()) if action_payload is not None else 0,
        "bad_frame_count": int(json_pair.get("bad_frame_count") or 0),
        "review_frame_count": int(json_pair.get("review_frame_count") or 0),
        "subtasks_json": str(episode_root / "subtasks.json"),
        "json": str(episode_root / "subtasks.json"),
    }
    return {
        "pairs": [pair],
        "filtering": {
            "source_frame_count": frame_count,
            "source_valid_frame_count": max(0, frame_count - int(json_pair.get("bad_frame_count") or 0) - int(json_pair.get("review_frame_count") or 0)),
            "retained_frame_count": frame_count,
            "removed_frame_count": 0,
            "removed_vlm_frame_count": 0,
            "bad_frame_count": int(json_pair.get("bad_frame_count") or 0),
            "review_frame_count": int(json_pair.get("review_frame_count") or 0),
            "policy": "keep_all_frames_mark_quality_in_parquet_and_json",
        },
        "transform_source": source_relative,
        "category": classification["category"],
        "categories": [classification["category"]],
        "output_format": EPISODE_LEROBOT_JSON_OUTPUT_FORMAT,
        "full_run_id": run_id,
        "timeline_id": timeline_id,
        "episode_root": str(episode_root),
        "subtasks_json": str(episode_root / "subtasks.json"),
        "lerobot_info": str(info_path),
    }


def _nexus_source_episode_root(manifest: dict, episode: dict) -> Path:
    root = Path(str(manifest["root_path"])).expanduser().resolve()
    candidates: list[Path] = []
    if (root / "meta" / "sync.parquet").is_file():
        candidates.append(root)
    for record in _episode_records(manifest, episode):
        relative = str(record.get("relative_path") or "")
        if not relative:
            continue
        current = (root / relative).resolve().parent
        while current == root or root in current.parents:
            if (current / "meta" / "sync.parquet").is_file():
                candidates.append(current)
                break
            if current == root:
                break
            current = current.parent
    unique = sorted({path.resolve() for path in candidates}, key=lambda path: (len(path.parts), str(path).casefold()))
    if len(unique) != 1:
        raise RuntimeError("无法为当前 Nexus Episode 唯一定位源目录")
    return unique[0]


def _export_nexus_episode_lerobot_json(
    output_root: Path,
    manifest: dict,
    episode: dict,
    smoothed_media: dict,
    curation: dict,
    behavior: dict | None,
    progress,
    *,
    run_id: str | None,
    timeline_id: str | None,
) -> dict:
    from .nexus_lerobot_export import CAMERA_STREAMS, convert_nexus_to_lerobot

    source_episode_root = _nexus_source_episode_root(manifest, episode)
    cameras = tuple(
        name
        for name, config in CAMERA_STREAMS.items()
        if (source_episode_root / str(config["relative_path"])).is_file()
    )
    if not cameras:
        raise RuntimeError("Nexus Episode 没有可导出的 RGB 相机流")
    episode_key = _category_name(str(episode.get("id") or episode.get("name") or source_episode_root.name))
    episode_root = output_root / "episodes" / episode_key
    behavior_path = str(((behavior or {}).get("artifacts") or {}).get("behavior") or "") or None
    curation_path = str(curation.get("artifact_path") or "") or None
    progress(5.0, "准备 Nexus 完整 Episode 多传感器 LeRobot")
    converted = convert_nexus_to_lerobot(
        source_episode_root,
        episode_root,
        cameras=cameras,
        curation=curation_path,
        annotations=behavior_path,
        keep_all_frames=True,
        progress=lambda value, message: progress(5.0 + max(0.0, min(100.0, value)) * 0.85, message),
    )
    frame_count = int(converted.get("frame_count") or 0)
    expected_count = int((curation.get("source_video") or {}).get("frame_count") or smoothed_media.get("frame_count") or 0)
    if expected_count > 0 and frame_count != expected_count:
        raise RuntimeError(f"Nexus Full 时间轴与清洗报告不一致：export={frame_count}, curation={expected_count}")
    json_episode = {**episode, "frame_count": frame_count, "fps": converted.get("fps") or 30.0}
    subtask_result = _export_subtask_json(
        output_root,
        manifest,
        json_episode,
        curation,
        behavior,
        run_id=run_id,
        timeline_id=timeline_id,
        output_format=EPISODE_LEROBOT_JSON_OUTPUT_FORMAT,
        episode_root=episode_root,
    )
    if len(converted.get("episodes") or []) != 1:
        raise RuntimeError("Nexus 完整 Episode 导出应只生成一个 LeRobot Episode")
    converted_episode = converted["episodes"][0]
    classification = classify_behavior(behavior or {}, 0, max(0, frame_count - 1))
    json_pair = subtask_result["pairs"][0]
    videos = dict(converted_episode.get("videos") or {})
    pair = {
        **json_pair,
        "id": str(episode.get("id") or episode_key),
        "output_format": EPISODE_LEROBOT_JSON_OUTPUT_FORMAT,
        **classification,
        "episode_index": int(converted_episode.get("episode_index") or 0),
        "task_index": int(converted_episode.get("task_index") or 0),
        "start_frame": 0,
        "end_frame": max(0, frame_count - 1),
        "frame_count": frame_count,
        "fps": float(converted.get("fps") or 30.0),
        "data": converted_episode.get("data"),
        "body": None,
        "mp4": videos.get("observation.images.head") or next(iter(videos.values()), None),
        "videos": videos,
        "meta_info": str(episode_root / "meta" / "info.json"),
        "lerobot_root": str(episode_root),
        "source_episode_root": str(source_episode_root),
        "body_joint_names": [],
        "camera_intrinsic": None,
        "action_available": False,
        "action_valid_frame_count": 0,
        "subtasks_json": str(episode_root / "subtasks.json"),
        "json": str(episode_root / "subtasks.json"),
    }
    progress(100.0, "已写入 Nexus Episode LeRobot + JSON")
    return {
        "pairs": [pair],
        "filtering": {
            "source_frame_count": frame_count,
            "source_valid_frame_count": max(0, frame_count - int(json_pair.get("bad_frame_count") or 0) - int(json_pair.get("review_frame_count") or 0)),
            "retained_frame_count": frame_count,
            "removed_frame_count": 0,
            "bad_frame_count": int(json_pair.get("bad_frame_count") or 0),
            "review_frame_count": int(json_pair.get("review_frame_count") or 0),
            "policy": "keep_all_frames_mark_quality_in_parquet_and_json",
        },
        "transform_source": "nexus_v4_multimodal",
        "category": classification["category"],
        "categories": [classification["category"]],
        "output_format": EPISODE_LEROBOT_JSON_OUTPUT_FORMAT,
        "full_run_id": run_id,
        "timeline_id": timeline_id,
        "episode_root": str(episode_root),
        "subtasks_json": str(episode_root / "subtasks.json"),
        "lerobot_info": str(episode_root / "meta" / "info.json"),
        "nexus_multimodal": True,
    }


def _fill_short_internal_gaps(
    mask: np.ndarray,
    maximum_gap_frames: int,
    blocked_mask: np.ndarray | None = None,
) -> np.ndarray:
    filled = np.asarray(mask, dtype=bool).copy()
    if maximum_gap_frames <= 0 or filled.size < 3:
        return filled
    blocked = (
        np.zeros(filled.shape, dtype=bool)
        if blocked_mask is None
        else np.asarray(blocked_mask, dtype=bool)
    )
    if blocked.shape != filled.shape:
        raise ValueError("blocked gap mask must match the retained-frame mask")
    for start, end in _mask_intervals(~filled):
        if start == 0 or end == filled.size - 1:
            continue
        if blocked[start:end + 1].any():
            continue
        if end - start + 1 <= maximum_gap_frames:
            filled[start:end + 1] = True
    return filled


def filtered_intervals(
    frame_count: int,
    curation: dict,
    behavior: dict,
    *,
    fps: float | None = None,
    max_internal_gap_seconds: float = 0.0,
    min_clip_seconds: float = 0.0,
) -> tuple[list[tuple[int, int]], dict]:
    if frame_count <= 0:
        return [], {
            "source_frame_count": 0,
            "source_valid_frame_count": 0,
            "retained_frame_count": 0,
            "removed_vlm_frame_count": 0,
            "removed_vlm_phases": {},
            "raw_interval_count": 0,
            "final_interval_count": 0,
            "merged_gap_frame_count": 0,
            "dropped_short_fragment_frame_count": 0,
        }
    keep = np.zeros(frame_count, dtype=bool)
    quality_blocked = np.ones(frame_count, dtype=bool)
    for segment in curation.get("segments") or []:
        start = max(0, min(frame_count - 1, int(segment.get("start_frame") or 0)))
        end = max(start, min(frame_count - 1, int(segment.get("end_frame") or start)))
        if str(segment.get("state") or "") == "valid":
            keep[start:end + 1] = True
            quality_blocked[start:end + 1] = False
        else:
            quality_blocked[start:end + 1] = True
    source_valid_frame_count = int(keep.sum())
    effective_fps = max(0.01, float(fps or 0.0)) if fps is not None else None
    maximum_gap_frames = int(round(max(0.0, max_internal_gap_seconds) * effective_fps)) if effective_fps else 0
    before_gap_fill = keep.copy()
    # Anti-fragmentation may never recover a curation bad/review frame.
    # Unknown or uncovered ranges are also blocked conservatively.
    keep = _fill_short_internal_gaps(keep, maximum_gap_frames, quality_blocked)
    gap_fill_mask = keep & ~before_gap_fill

    removed: dict[str, int] = {}
    removed_mask = np.zeros(frame_count, dtype=bool)
    for segment in behavior.get("segments") or []:
        phase = _phase_key(segment.get("phase_label") or segment.get("label"))
        if phase not in REMOVED_VLM_PHASES:
            continue
        start = max(0, min(frame_count - 1, int(segment.get("start_frame") or 0)))
        end = max(start, min(frame_count - 1, int(segment.get("end_frame") or start)))
        count = int(keep[start:end + 1].sum())
        removed[phase] = removed.get(phase, 0) + count
        removed_mask[start:end + 1] = True
        keep[start:end + 1] = False

    raw_intervals = _mask_intervals(keep)
    minimum_clip_frames = int(math.ceil(max(0.0, min_clip_seconds) * effective_fps)) if effective_fps else 0
    dropped_short = np.zeros(frame_count, dtype=bool)
    if minimum_clip_frames > 1:
        for start, end in raw_intervals:
            if end - start + 1 < minimum_clip_frames:
                dropped_short[start:end + 1] = True
        keep[dropped_short] = False
    intervals = _mask_intervals(keep)
    return intervals, {
        "source_frame_count": frame_count,
        "source_valid_frame_count": source_valid_frame_count,
        "retained_frame_count": int(keep.sum()),
        "removed_vlm_frame_count": sum(removed.values()),
        "removed_vlm_phases": removed,
        "raw_interval_count": len(raw_intervals),
        "final_interval_count": len(intervals),
        "merged_gap_frame_count": int((gap_fill_mask & ~removed_mask).sum()),
        "dropped_short_fragment_frame_count": int(dropped_short.sum()),
        "max_internal_gap_seconds": max(0.0, float(max_internal_gap_seconds)),
        "max_internal_gap_frames": maximum_gap_frames,
        "min_clip_seconds": max(0.0, float(min_clip_seconds)),
        "min_clip_frames": minimum_clip_frames,
    }


def _object_name(value: object) -> str:
    if isinstance(value, dict):
        value = value.get("name")
    name = str(value or "").strip().strip(" ,.;:")
    name = re.sub(r"#\d+$", "", name).strip()
    if name.casefold() in {"hand", "hands", "human hand", "robot hand", "gripper", "arm", "unknown"}:
        return ""
    return name[:80]


def classify_behavior(behavior: dict, start_frame: int, end_frame: int) -> dict:
    phase_scores: dict[str, int] = {}
    phase_confidence: dict[str, float] = {}
    target_scores: dict[str, int] = {}
    target_names: dict[str, str] = {}
    instance_scores: dict[str, int] = {}
    for segment in behavior.get("segments") or []:
        segment_start = int(segment.get("start_frame") or 0)
        segment_end = int(segment.get("end_frame") or segment_start)
        overlap = max(0, min(end_frame, segment_end) - max(start_frame, segment_start) + 1)
        if overlap <= 0:
            continue
        phase = _phase_key(segment.get("phase_label") or segment.get("label"))
        if phase not in REMOVED_VLM_PHASES:
            phase_scores[phase] = phase_scores.get(phase, 0) + overlap
            confidence = max(0.0, min(1.0, float(segment.get("confidence") or 0.0)))
            phase_confidence[phase] = phase_confidence.get(phase, 0.0) + confidence * overlap
        targets = [_object_name(value) for value in segment.get("primary_targets") or []]
        instance = str(segment.get("target_instance") or "").strip()[:120]
        if not any(targets) and instance:
            targets = [_object_name(instance)]
        for target in (value for value in targets if value):
            key = target.casefold()
            target_scores[key] = target_scores.get(key, 0) + overlap
            target_names.setdefault(key, target)
        if instance:
            instance_scores[instance] = instance_scores.get(instance, 0) + overlap

    phase_label = max(phase_scores, key=lambda key: (phase_scores[key], key)) if phase_scores else "unknown"
    target_key = max(target_scores, key=lambda key: (target_scores[key], key)) if target_scores else ""
    primary_target = target_names.get(target_key, "")
    if not primary_target:
        top_level_targets = [_object_name(value) for value in behavior.get("primary_targets") or []]
        fallback_targets = [value for value in top_level_targets if value]
        if not fallback_targets:
            fallback_targets = [value for value in (_object_name(item) for item in behavior.get("object_nouns") or []) if value]
        primary_target = fallback_targets[0] if fallback_targets else ""
    target_instance = max(instance_scores, key=lambda key: (instance_scores[key], key)) if instance_scores else ""

    raw_task_label = str(behavior.get("task_label") or "other").strip() or "other"
    task_is_generic = raw_task_label.casefold() in {"other", "unknown", "unclassified"}
    if task_is_generic:
        fallback_parts = [value for value in (phase_label if phase_label != "unknown" else "", primary_target) if value]
        category = _category_name("_".join(fallback_parts) or "other")
        classifier_source = "vlm_phase_target_fallback" if fallback_parts else "vlm_unclassified"
    else:
        category = _category_name(raw_task_label)
        classifier_source = "vlm_task_label"

    behavior_confidence = max(0.0, min(1.0, float(behavior.get("confidence") or 0.0)))
    if phase_label in phase_scores and phase_scores[phase_label] > 0:
        dominant_phase_confidence = phase_confidence.get(phase_label, 0.0) / phase_scores[phase_label]
        classifier_confidence = (behavior_confidence + dominant_phase_confidence) / 2.0
    else:
        classifier_confidence = behavior_confidence
    return {
        "category": category,
        "task_label": raw_task_label,
        "phase_label": phase_label,
        "primary_target": primary_target,
        "target_instance": target_instance,
        "direction": str(behavior.get("direction") or "unknown").casefold(),
        "classifier_source": classifier_source,
        "classifier_confidence": round(classifier_confidence, 4),
    }


def _phase_labels(behavior: dict, frames: np.ndarray) -> np.ndarray:
    labels = np.full(frames.shape, "unknown", dtype=object)
    for segment in behavior.get("segments") or []:
        start = int(segment.get("start_frame") or 0)
        end = int(segment.get("end_frame") or start)
        labels[(frames >= start) & (frames <= end)] = _phase_key(segment.get("phase_label") or segment.get("label"))
    return labels


def _write_video_clip(source: Path, target: Path, start: int, end: int) -> dict:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot decode smoothed video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid smoothed-video geometry: {source}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    writer = _create_frame_writer(target, fps, width, height)
    failed = True
    written = 0
    try:
        for _ in range(start, end + 1):
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
        writer.close()
        failed = False
    finally:
        capture.release()
        if failed:
            writer.abort()
            target.unlink(missing_ok=True)
    expected = end - start + 1
    if written != expected:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Video clip stopped at {written}/{expected} frames")
    return {
        "frame_count": written,
        "fps": fps,
        "width": width,
        "height": height,
        "encoder": writer.name,
        "encoder_gpu": getattr(writer, "gpu_device", None),
    }


def _write_mano_hdf5(
    target: Path,
    source_path: Path,
    source_relative: str,
    source_rows: np.ndarray,
    source_frames: np.ndarray,
    fps: float,
    manifest: dict,
    episode: dict,
    behavior: dict,
    classification: dict,
    repair: dict | None = None,
    source_video_positions: np.ndarray | None = None,
    camera_image_transforms: np.ndarray | None = None,
    action_payload: dict | None = None,
) -> None:
    count = int(source_frames.size)
    resolved_video_positions = (
        np.asarray(source_video_positions, dtype=np.float64).reshape(-1)
        if source_video_positions is not None
        else source_frames.astype(np.float64)
    )
    resolved_image_transforms = (
        np.asarray(camera_image_transforms, dtype=np.float64)
        if camera_image_transforms is not None
        else np.repeat(np.eye(3, dtype=np.float64)[None], count, axis=0)
    )
    if resolved_video_positions.shape != (count,) or resolved_image_transforms.shape != (count, 3, 3):
        raise ValueError("HDF5 retiming/geometry arrays do not match the exported video frames")
    if action_payload is not None:
        action_dim = int(action_payload.get("action_dim") or 0)
        if (
            np.asarray(action_payload.get("observation_state")).shape != (count, 20)
            or np.asarray(action_payload.get("action")).shape != (count, action_dim)
            or np.asarray(action_payload.get("target_frame_index")).shape != (count,)
            or np.asarray(action_payload.get("valid")).shape != (count,)
        ):
            raise ValueError("HDF5 Action arrays do not match exported frames")
    chunk = max(1, min(128, count))
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(source_path, "r") as source, h5py.File(target, "w") as output:
        output.attrs.update({
            "schema": FULL_PAIR_SCHEMA,
            "dataset_id": str(manifest["id"]),
            "episode_id": str(episode["id"]),
            "episode_name": str(episode.get("name") or episode["id"]),
            "task_label": str(behavior.get("task_label") or "other"),
            "category": str(classification.get("category") or "other"),
            "phase_label": str(classification.get("phase_label") or "unknown"),
            "primary_target": str(classification.get("primary_target") or ""),
            "target_instance": str(classification.get("target_instance") or ""),
            "direction": str(classification.get("direction") or "unknown"),
            "classifier_source": str(classification.get("classifier_source") or "vlm_unclassified"),
            "classifier_confidence": float(classification.get("classifier_confidence") or 0.0),
            "source_hdf5": source_relative,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mano_joint_count": 44,
            "transform_count_with_camera": 45,
            "s1_repair_applied": s1_repair_cell_count(repair, source_relative, "transforms/") > 0,
            "s1_repair_cell_count": s1_repair_cell_count(repair, source_relative, "transforms/"),
            "hand_geometry_schema": "mano21_kinematic_retarget",
            "hand_geometry_revision": EGODEX_MANO_REVISION,
            "action_available": action_payload is not None,
        })
        if action_payload is not None:
            output.attrs.update({
                "action_profile_id": str(action_payload.get("action_profile_id") or ""),
                "action_representation": str(action_payload.get("action_representation") or "unknown"),
                "action_coordinate_frame": str(action_payload.get("action_coordinate_frame") or "unknown"),
                "action_horizon_frames": int(action_payload.get("action_horizon_frames") or 0),
                "action_fields": json.dumps(list(action_payload.get("action_fields") or []), ensure_ascii=False),
                "observation_state_fields": json.dumps(list(action_payload.get("observation_state_fields") or []), ensure_ascii=False),
            })
            observation_group = output.require_group("observation")
            observation_group.create_dataset("state", data=np.asarray(action_payload["observation_state"], dtype=np.float32), compression="lzf", shuffle=True)
            output.create_dataset("action", data=np.asarray(action_payload["action"], dtype=np.float32), compression="lzf", shuffle=True)
            output.create_dataset("action_target_source_frame_index", data=np.asarray(action_payload["target_frame_index"], dtype=np.int64), compression="lzf")
            output.create_dataset("action_valid", data=np.asarray(action_payload["valid"], dtype=bool), compression="lzf")
        mano = output.require_group("mano")
        mano.attrs["joint_names"] = json.dumps(MANO_44_JOINT_NAMES, ensure_ascii=False)
        transforms_out = mano.create_dataset("transforms", shape=(count, 44, 4, 4), dtype=np.float32, chunks=(chunk, 44, 4, 4), compression="lzf", shuffle=True)
        confidence_out = mano.create_dataset("confidence", shape=(count, 44), dtype=np.float32, chunks=(chunk, 44), compression="lzf", shuffle=True)
        camera_out = output.create_dataset("camera/transform", shape=(count, 4, 4), dtype=np.float32, chunks=(chunk, 4, 4), compression="lzf", shuffle=True)
        intrinsic_out = output.create_dataset("camera/intrinsic", shape=(count, 3, 3), dtype=np.float32, chunks=(chunk, 3, 3), compression="lzf", shuffle=True)
        image_transform_out = output.create_dataset("camera/image_transform", shape=(count, 3, 3), dtype=np.float32, chunks=(chunk, 3, 3), compression="lzf", shuffle=True)
        left_wrist = output.create_dataset("wrist/left_xyz_rot6d", shape=(count, 9), dtype=np.float32, chunks=(chunk, 9), compression="lzf", shuffle=True)
        right_wrist = output.create_dataset("wrist/right_xyz_rot6d", shape=(count, 9), dtype=np.float32, chunks=(chunk, 9), compression="lzf", shuffle=True)
        output["wrist"].attrs["rot6d_convention"] = "first_two_rotation_columns"
        transforms = source["transforms"]
        confidences = source.get("confidences")
        base_intrinsic = None
        for key in ("camera/intrinsic", "camera/intrinsics", "intrinsic", "intrinsics"):
            candidate = source.get(key)
            if isinstance(candidate, h5py.Dataset):
                value = np.asarray(candidate[()], dtype=np.float64)
                base_intrinsic = value[0] if value.ndim == 3 else value
                if base_intrinsic.shape == (3, 3) and np.isfinite(base_intrinsic).all():
                    break
                base_intrinsic = None
        if base_intrinsic is None:
            from .lerobot_export import scaled_egodex_camera_intrinsic

            base_intrinsic = scaled_egodex_camera_intrinsic(
                int(episode.get("width") or 1920),
                int(episode.get("height") or 1080),
            ).astype(np.float64)
        already_retargeted = source_is_retargeted(source)
        templates = {
            side: (
                None
                if already_retargeted or not has_egodex_mano_source(transforms, side)
                else fit_egodex_mano_template(transforms, side)
            )
            for side in ("left", "right")
        }
        left_index = MANO_44_JOINT_NAMES.index("leftHand")
        right_index = MANO_44_JOINT_NAMES.index("rightHand")
        for offset in range(0, count, chunk):
            right = min(count, offset + chunk)
            rows = source_rows[offset:right]
            required_names = tuple(dict.fromkeys((
                *MANO_44_JOINT_NAMES,
                *egodex_mano_source_names(transforms, "left"),
                *egodex_mano_source_names(transforms, "right"),
            )))
            available_names = tuple(name for name in required_names if name in transforms)
            blocks = {
                name: _sample_repaired_transforms(
                    transforms[name], rows, repair, source_relative, f"transforms/{name}",
                )
                for name in available_names
            }
            values = np.stack([blocks[name] for name in MANO_44_JOINT_NAMES], axis=1).astype(np.float32)
            for side, destination_index in (("left", left_index), ("right", right_index)):
                template = templates[side]
                if template is None:
                    hand_values = np.stack([blocks[name] for name in side_hand_joint_names(side)], axis=1)
                else:
                    required = egodex_mano_source_names(transforms, side)
                    hand_values = np.stack([
                        retarget_egodex_mano_frame(
                            {name: blocks[name][local_index] for name in required},
                            template,
                        )
                        for local_index in range(len(rows))
                    ])
                values[:, destination_index:destination_index + 21] = hand_values.astype(np.float32)
            transforms_out[offset:right] = values
            camera_out[offset:right] = _sample_repaired_transforms(
                transforms["camera"], rows, repair, source_relative, "transforms/camera",
            ).astype(np.float32)
            image_transforms = resolved_image_transforms[offset:right]
            image_transform_out[offset:right] = image_transforms.astype(np.float32)
            intrinsic_out[offset:right] = np.einsum("fij,jk->fik", image_transforms, base_intrinsic).astype(np.float32)
            confidence_values = np.full((right - offset, 44), np.nan, dtype=np.float32)
            if isinstance(confidences, h5py.Group):
                for index, name in enumerate(MANO_44_JOINT_NAMES):
                    if name in confidences:
                        confidence_values[:, index] = sample_hdf5_numeric(confidences[name], rows).reshape(-1).astype(np.float32)
            confidence_out[offset:right] = confidence_values
            for destination, joint_index in ((left_wrist, left_index), (right_wrist, right_index)):
                wrist_transform = values[:, joint_index]
                destination[offset:right] = np.concatenate((wrist_transform[:, :3, 3], _rot6d(wrist_transform)), axis=1)
        metadata = output.require_group("segment")
        metadata.create_dataset("source_frame_index", data=source_frames, compression="lzf")
        metadata.create_dataset("source_hdf5_row", data=np.rint(source_rows).astype(np.int64), compression="lzf")
        metadata.create_dataset("source_hdf5_position", data=source_rows.astype(np.float64), compression="lzf")
        metadata.create_dataset("source_video_frame_position", data=resolved_video_positions, compression="lzf")
        metadata.create_dataset("timestamp", data=source_frames.astype(np.float64) / max(0.01, fps), compression="lzf")
        metadata.create_dataset("phase_label", data=_phase_labels(behavior, source_frames), dtype=string_dtype)


def _video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if capture.isOpened() else 0
    capture.release()
    return count


def export_episode(
    output_root: Path,
    manifest: dict,
    episode: dict,
    smoothed_media: dict,
    curation: dict,
    behavior: dict,
    progress,
    output_format: str = DEFAULT_FULL_OUTPUT_FORMAT,
    run_id: str | None = None,
    timeline_id: str | None = None,
    action_report: dict | None = None,
) -> dict:
    if output_format not in SUPPORTED_FULL_OUTPUT_FORMATS:
        raise ValueError(f"Unsupported Full output format: {output_format}")
    if output_format == SUBTASK_JSON_OUTPUT_FORMAT:
        progress(100.0, "写入 Episode subtask JSON")
        return _export_subtask_json(
            output_root,
            manifest,
            episode,
            curation,
            behavior,
            run_id=run_id,
            timeline_id=timeline_id,
        )
    if output_format == EPISODE_LEROBOT_JSON_OUTPUT_FORMAT:
        if dataset_mode(manifest)["family"] == "nexus_multimodal":
            return _export_nexus_episode_lerobot_json(
                output_root,
                manifest,
                episode,
                smoothed_media,
                curation,
                behavior,
                progress,
                run_id=run_id,
                timeline_id=timeline_id,
            )
        return _export_episode_lerobot_json(
            output_root,
            manifest,
            episode,
            smoothed_media,
            curation,
            behavior,
            progress,
            run_id=run_id,
            timeline_id=timeline_id,
            action_report=action_report,
        )
    format_map = manifest.get("format_map") or {}
    capabilities = format_map.get("capabilities") or {}
    if capabilities and capabilities.get("can_full_export") is False:
        blocking_issues = [
            str(item.get("message") or "")
            for item in format_map.get("issues") or []
            if item.get("severity") in {"warning", "error"}
        ]
        detail = "；".join(item for item in blocking_issues if item) or "源数据没有可验证的左右手 21 点变换与相机变换"
        raise RuntimeError(f"当前格式可执行清洗与标注，但不能安全执行固定 MANO/LeRobot Full 导出：{detail}")
    repair = load_s1_repair(curation)
    frame_count = int(smoothed_media.get("frame_count") or episode.get("frame_count") or 0)
    fps = max(0.01, float(smoothed_media.get("fps") or episode.get("fps") or 30.0))
    intervals, filtering = filtered_intervals(
        frame_count,
        curation,
        behavior,
        fps=fps,
        max_internal_gap_seconds=DEFAULT_MAX_INTERNAL_GAP_SECONDS,
        min_clip_seconds=DEFAULT_MIN_CLIP_SECONDS,
    )
    source_path, source_relative, source_count = _find_transform_source(manifest, episode, output_format)
    repair_cell_count = s1_repair_cell_count(repair, source_relative, "transforms/")
    row_map = _aligned_positions(manifest, episode, source_relative, source_count, frame_count, smoothed_media)
    source_video_positions = _source_video_positions(smoothed_media, frame_count)
    camera_image_transforms = _camera_image_transforms(smoothed_media, frame_count)
    action_payload = _build_export_action_payload(
        source_path,
        source_relative,
        row_map,
        repair,
        action_report,
    )
    intervals, action_filtering = _action_safe_intervals(intervals, action_payload, fps)
    filtering.update(action_filtering)
    filtering["retained_frame_count"] = sum(end - start + 1 for start, end in intervals)
    filtering["final_interval_count"] = len(intervals)
    next_episode_numbers: dict[str, int] = {}
    pairs: list[dict] = []
    for position, (start, end) in enumerate(intervals, start=1):
        classification = classify_behavior(behavior, start, end)
        category = classification["category"]
        progress(100 * (position - 1) / max(1, len(intervals)), f"准备导出片段 {position}/{len(intervals)}")
        frames = np.arange(start, end + 1, dtype=np.int64)
        expected = int(frames.size)
        if output_format == "lerobot":
            staging_root = output_root / ".staging"
            staging_root.mkdir(parents=True, exist_ok=True)
            video_temporary = staging_root / f".video.{uuid.uuid4().hex}.part.mp4"
            try:
                video_info = _write_video_clip(Path(str(smoothed_media["path"])), video_temporary, start, end)
                if int(video_info.get("frame_count") or 0) != expected or _video_frame_count(video_temporary) != expected:
                    raise RuntimeError("LeRobot staged video frame count mismatch")
                pair = write_lerobot_pair(
                    output_root,
                    video_temporary,
                    video_info,
                    source_path,
                    source_relative,
                    source_count,
                    row_map[frames],
                    frames,
                    _phase_labels(behavior, frames),
                    manifest,
                    episode,
                    classification,
                    repair,
                    source_video_positions=source_video_positions[frames],
                    camera_image_transforms=camera_image_transforms[frames],
                    quality_states=np.full(expected, "valid", dtype=object),
                    action_payload=_slice_action_payload(action_payload, frames),
                )
                pair.update({"start_frame": start, "end_frame": end, "full_run_id": run_id, "timeline_id": timeline_id})
                pairs.append(pair)
                progress(100 * position / max(1, len(intervals)), f"导出 LeRobot Episode {position}/{len(intervals)}")
            finally:
                video_temporary.unlink(missing_ok=True)
        else:
            category_root = output_root / category
            category_root.mkdir(parents=True, exist_ok=True)
            if category not in next_episode_numbers:
                existing_numbers = [
                    int(match.group(1))
                    for path in category_root.iterdir()
                    if path.is_dir() and (match := re.fullmatch(r"ep(\d+)", path.name, flags=re.IGNORECASE))
                ]
                next_episode_numbers[category] = max(existing_numbers, default=0) + 1
            while True:
                export_episode_id = f"ep{next_episode_numbers[category]}"
                next_episode_numbers[category] += 1
                episode_root = category_root / export_episode_id
                try:
                    episode_root.mkdir(parents=True, exist_ok=False)
                    break
                except FileExistsError:
                    continue
            video_path = episode_root / "video.mp4"
            hdf5_path = episode_root / "data.hdf5"
            video_temporary = episode_root / f".video.{uuid.uuid4().hex}.part.mp4"
            hdf5_temporary = episode_root / f".data.{uuid.uuid4().hex}.part.hdf5"
            try:
                video_info = _write_video_clip(Path(str(smoothed_media["path"])), video_temporary, start, end)
                _write_mano_hdf5(
                    hdf5_temporary,
                    source_path,
                    source_relative,
                    row_map[frames],
                    frames,
                    video_info["fps"],
                    manifest,
                    episode,
                    behavior,
                    classification,
                    repair,
                    source_video_positions[frames],
                    camera_image_transforms[frames],
                    _slice_action_payload(action_payload, frames),
                )
                with h5py.File(hdf5_temporary, "r") as output:
                    hdf5_frames = int(output["mano/transforms"].shape[0])
                    shape = tuple(output["mano/transforms"].shape[1:])
                    camera_frames = int(output["camera/transform"].shape[0])
                video_frames = _video_frame_count(video_temporary)
                if video_frames != expected or hdf5_frames != expected or camera_frames != expected or shape != (44, 4, 4):
                    raise RuntimeError(f"Full pair verification failed: video={video_frames}, hdf5={hdf5_frames}, camera={camera_frames}, expected={expected}")
                video_temporary.replace(video_path)
                hdf5_temporary.replace(hdf5_path)
                pairs.append({
                    "id": f"{category}/{export_episode_id}",
                    "output_format": "hdf5_mp4",
                    "export_episode": export_episode_id,
                    "episode_id": episode["id"],
                    "full_run_id": run_id,
                    "timeline_id": timeline_id,
                    **classification,
                    "start_frame": start,
                    "end_frame": end,
                    "frame_count": expected,
                    "fps": video_info["fps"],
                    "width": video_info["width"],
                    "height": video_info["height"],
                    "video_encoder": video_info["encoder"],
                    "video_encoder_gpu": video_info["encoder_gpu"],
                    "s1_repair_applied": repair_cell_count > 0,
                    "s1_repair_cell_count": repair_cell_count,
                    "action_available": action_payload is not None,
                    "action_valid_frame_count": expected if action_payload is not None else 0,
                    "mp4": str(video_path),
                    "hdf5": str(hdf5_path),
                })
                progress(100 * position / max(1, len(intervals)), f"导出 HDF5/视频 {position}/{len(intervals)}")
            finally:
                video_temporary.unlink(missing_ok=True)
                hdf5_temporary.unlink(missing_ok=True)
    categories = sorted({str(pair["category"]) for pair in pairs})
    return {
        "pairs": pairs,
        "filtering": filtering,
        "transform_source": source_relative,
        "category": categories[0] if len(categories) == 1 else "mixed" if categories else None,
        "categories": categories,
        "output_format": output_format,
        "full_run_id": run_id,
        "timeline_id": timeline_id,
        "s1_repair_applied": repair_cell_count > 0,
        "s1_repair_cell_count": repair_cell_count,
    }


def write_dataset_index(
    output_root: Path,
    manifest: dict,
    pairs: list[dict],
    failures: list[dict],
    output_format: str = DEFAULT_FULL_OUTPUT_FORMAT,
    run_id: str | None = None,
    timeline_ids: dict[str, str] | None = None,
) -> Path:
    if output_format not in SUPPORTED_FULL_OUTPUT_FORMATS:
        raise ValueError(f"Unsupported Full output format: {output_format}")
    path = output_root / "dataset.json"
    with _INDEX_LOCK:
        existing: dict = {}
        try:
            candidate = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            if candidate.get("schema") in {FULL_DATASET_SCHEMA, LEGACY_FULL_DATASET_SCHEMA} and str(candidate.get("source_dataset_id") or "") == str(manifest.get("id") or ""):
                existing = candidate
        except (OSError, json.JSONDecodeError):
            existing = {}
        combined_by_id = {
            str(pair.get("id")): {
                **pair,
                "output_format": pair.get("output_format") or ("hdf5_mp4" if pair.get("hdf5") else "lerobot"),
            }
            for pair in existing.get("pairs") or []
            if isinstance(pair, dict) and pair.get("id")
        }
        combined_by_id.update({
            str(pair.get("id")): {**pair, "output_format": pair.get("output_format") or output_format}
            for pair in pairs if pair.get("id")
        })
        combined_pairs = list(combined_by_id.values())
        categories: dict[str, int] = {}
        output_formats: dict[str, int] = {}
        for pair in combined_pairs:
            label = str(pair.get("category") or pair.get("task_label") or "other")
            categories[label] = categories.get(label, 0) + 1
            pair_format = str(pair.get("output_format") or "unknown")
            output_formats[pair_format] = output_formats.get(pair_format, 0) + 1
        combined_failures: list[dict] = []
        seen_failures: set[tuple[str, str]] = set()
        for failure in [*(existing.get("failures") or []), *failures]:
            if not isinstance(failure, dict):
                continue
            key = (str(failure.get("episode_id") or ""), str(failure.get("error") or ""))
            if key not in seen_failures:
                seen_failures.add(key)
                combined_failures.append(failure)
        now = datetime.now(timezone.utc).isoformat()
        _write_json_atomic(path, {
            "schema": FULL_DATASET_SCHEMA,
            "created_at": existing.get("created_at") or now,
            "updated_at": now,
            "source_dataset_id": manifest.get("id"),
            "full_run_id": run_id,
            "timeline_ids": dict(timeline_ids or {}),
            "source_root": manifest.get("root_path"),
            "pipeline_version": FULL_EXPORT_PIPELINE_VERSION,
            "pipeline": ["video_smoothing", "nexus_pressure_empty_p1_when_applicable", "optional_action_s2", "s1_s5_c3", "vlm_non_red_segments", "c1_c2", "drop_idle", "action_safe_filter", "anti_fragment", "classify_and_export"],
            "default_output_format": DEFAULT_FULL_OUTPUT_FORMAT,
            "output_formats": output_formats,
            "subtask_json_schema": SUBTASK_JSON_SCHEMA,
            "subtask_frame_index_convention": {
                "space": "source_video",
                "base": 0,
                "end_inclusive": True,
                "bad_frames": "curation invalid/red",
                "review_frames": "curation uncertain/yellow",
            },
            "hand_joint_names": list(HAND_21_JOINT_NAMES),
            "hand_joint_count_per_side": 21,
            "hand_transform_shape": ["T", 21, 4, 4],
            "body_joint_policy": "every named transform except the two 21-joint hands and camera is written to the body Parquet",
            "mano_joint_names": list(MANO_44_JOINT_NAMES),
            "mano_shape": ["T", 44, 4, 4],
            "camera_shape": ["T", 4, 4],
            "camera_intrinsic": next((pair.get("camera_intrinsic") for pair in combined_pairs if pair.get("camera_intrinsic")), None),
            "wrist_pose_shape": ["T", 9],
            "wrist_pose_fields": ["x", "y", "z", "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5"],
            "removed_vlm_phases": sorted(REMOVED_VLM_PHASES),
            "reach_policy": "keep",
            "action_profiles": sorted({
                str(pair.get("action_profile_id"))
                for pair in combined_pairs
                if pair.get("action_profile_id")
            }),
            "action_available_pair_count": sum(bool(pair.get("action_available")) for pair in combined_pairs),
            "anti_fragment": {
                "max_internal_gap_seconds": DEFAULT_MAX_INTERNAL_GAP_SECONDS,
                "min_clip_seconds": DEFAULT_MIN_CLIP_SECONDS,
                "vlm_removed_phases_are_never_gap_filled": True,
            },
            "pair_count": len(combined_pairs),
            "retained_frame_count": sum(int(pair.get("frame_count") or 0) for pair in combined_pairs),
            "categories": categories,
            "pairs": combined_pairs,
            "failures": combined_failures,
        })
        if output_format == "lerobot":
            return write_lerobot_metadata(output_root, manifest, combined_pairs)
    return path
