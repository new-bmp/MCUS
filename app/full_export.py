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

from .lerobot_export import (
    HAND_21_JOINT_NAMES,
    side_hand_joint_names,
    write_lerobot_metadata,
    write_lerobot_pair,
)
from .sensor_alignment import load_sensor_alignment, scan_episode_sensor_alignment
from .video_smoothing import _create_frame_writer


FULL_DATASET_SCHEMA = "alice/full-dataset/v2"
LEGACY_FULL_DATASET_SCHEMA = "alice/full-mano-dataset/v1"
FULL_PAIR_SCHEMA = "alice/full-mano-pair/v1"
FULL_EXPORT_PIPELINE_VERSION = 3
DEFAULT_FULL_OUTPUT_FORMAT = "lerobot"
SUPPORTED_FULL_OUTPUT_FORMATS = {"lerobot", "hdf5_mp4"}
REMOVED_VLM_PHASES = {"idle", "reach"}
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


def _find_transform_source(manifest: dict, episode: dict, output_format: str = "hdf5_mp4") -> tuple[Path, str, int]:
    root = Path(str(manifest["root_path"])).expanduser().resolve()
    required_names = MANO_44_JOINT_NAMES if output_format == "hdf5_mp4" else (
        *side_hand_joint_names("left"),
        *side_hand_joint_names("right"),
    )
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


def _aligned_rows(manifest: dict, episode: dict, relative: str, source_count: int, video_count: int) -> np.ndarray:
    if source_count == video_count:
        return np.arange(video_count, dtype=np.int64)
    alignment = load_sensor_alignment(str(manifest["id"]), str(episode["id"])) or scan_episode_sensor_alignment(manifest, episode)
    stream = next((
        item for item in alignment.get("streams") or []
        if str(item.get("relative_path") or "").replace("\\", "/").casefold() == relative.casefold()
    ), None)
    if stream is None:
        raise RuntimeError(f"MANO source cannot be aligned to video frames: {relative}")
    lookup = stream.get("frame_to_sensor_index")
    if isinstance(lookup, list) and len(lookup) >= video_count:
        rows = np.asarray(lookup[:video_count], dtype=np.int64)
    elif stream.get("mode") in {"prealigned_master_clock", "paired_frame_index"}:
        rows = np.arange(video_count, dtype=np.int64)
    elif stream.get("index_multiplier") is not None:
        rows = np.rint(np.arange(video_count) * float(stream["index_multiplier"])).astype(np.int64)
    else:
        raise RuntimeError(f"MANO alignment has no frame mapping: {relative}")
    if (rows < 0).any() or (rows >= source_count).any():
        raise RuntimeError(f"MANO alignment contains out-of-range rows: {relative}")
    return rows


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


def _fill_short_internal_gaps(mask: np.ndarray, maximum_gap_frames: int) -> np.ndarray:
    filled = np.asarray(mask, dtype=bool).copy()
    if maximum_gap_frames <= 0 or filled.size < 3:
        return filled
    for start, end in _mask_intervals(~filled):
        if start == 0 or end == filled.size - 1:
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
    for segment in curation.get("segments") or []:
        if str(segment.get("state") or "") != "valid":
            continue
        start = max(0, min(frame_count - 1, int(segment.get("start_frame") or 0)))
        end = max(start, min(frame_count - 1, int(segment.get("end_frame") or start)))
        keep[start:end + 1] = True
    source_valid_frame_count = int(keep.sum())
    effective_fps = max(0.01, float(fps or 0.0)) if fps is not None else None
    maximum_gap_frames = int(round(max(0.0, max_internal_gap_seconds) * effective_fps)) if effective_fps else 0
    before_gap_fill = keep.copy()
    keep = _fill_short_internal_gaps(keep, maximum_gap_frames)
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
) -> None:
    count = int(source_frames.size)
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
        })
        mano = output.require_group("mano")
        mano.attrs["joint_names"] = json.dumps(MANO_44_JOINT_NAMES, ensure_ascii=False)
        transforms_out = mano.create_dataset("transforms", shape=(count, 44, 4, 4), dtype=np.float32, chunks=(chunk, 44, 4, 4), compression="lzf", shuffle=True)
        confidence_out = mano.create_dataset("confidence", shape=(count, 44), dtype=np.float32, chunks=(chunk, 44), compression="lzf", shuffle=True)
        camera_out = output.create_dataset("camera/transform", shape=(count, 4, 4), dtype=np.float32, chunks=(chunk, 4, 4), compression="lzf", shuffle=True)
        left_wrist = output.create_dataset("wrist/left_xyz_rot6d", shape=(count, 9), dtype=np.float32, chunks=(chunk, 9), compression="lzf", shuffle=True)
        right_wrist = output.create_dataset("wrist/right_xyz_rot6d", shape=(count, 9), dtype=np.float32, chunks=(chunk, 9), compression="lzf", shuffle=True)
        output["wrist"].attrs["rot6d_convention"] = "first_two_rotation_columns"
        transforms = source["transforms"]
        confidences = source.get("confidences")
        left_index = MANO_44_JOINT_NAMES.index("leftHand")
        right_index = MANO_44_JOINT_NAMES.index("rightHand")
        for offset in range(0, count, chunk):
            right = min(count, offset + chunk)
            rows = source_rows[offset:right]
            values = np.stack([_take_rows(transforms[name], rows) for name in MANO_44_JOINT_NAMES], axis=1).astype(np.float32)
            transforms_out[offset:right] = values
            camera_out[offset:right] = _take_rows(transforms["camera"], rows).astype(np.float32)
            confidence_values = np.full((right - offset, 44), np.nan, dtype=np.float32)
            if isinstance(confidences, h5py.Group):
                for index, name in enumerate(MANO_44_JOINT_NAMES):
                    if name in confidences:
                        confidence_values[:, index] = _take_rows(confidences[name], rows).reshape(-1).astype(np.float32)
            confidence_out[offset:right] = confidence_values
            for destination, joint_index in ((left_wrist, left_index), (right_wrist, right_index)):
                wrist_transform = values[:, joint_index]
                destination[offset:right] = np.concatenate((wrist_transform[:, :3, 3], _rot6d(wrist_transform)), axis=1)
        metadata = output.require_group("segment")
        metadata.create_dataset("source_frame_index", data=source_frames, compression="lzf")
        metadata.create_dataset("source_hdf5_row", data=source_rows, compression="lzf")
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
) -> dict:
    if output_format not in SUPPORTED_FULL_OUTPUT_FORMATS:
        raise ValueError(f"Unsupported Full output format: {output_format}")
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
    row_map = _aligned_rows(manifest, episode, source_relative, source_count, frame_count)
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
                )
                pair.update({"start_frame": start, "end_frame": end})
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
                    **classification,
                    "start_frame": start,
                    "end_frame": end,
                    "frame_count": expected,
                    "fps": video_info["fps"],
                    "width": video_info["width"],
                    "height": video_info["height"],
                    "video_encoder": video_info["encoder"],
                    "video_encoder_gpu": video_info["encoder_gpu"],
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
    }


def write_dataset_index(
    output_root: Path,
    manifest: dict,
    pairs: list[dict],
    failures: list[dict],
    output_format: str = DEFAULT_FULL_OUTPUT_FORMAT,
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
            "source_root": manifest.get("root_path"),
            "pipeline_version": FULL_EXPORT_PIPELINE_VERSION,
            "pipeline": ["video_smoothing", "optional_action_s2", "s1_s5_c3", "vlm_non_red_segments", "c1_c2", "drop_idle_reach", "anti_fragment", "classify_and_export"],
            "default_output_format": DEFAULT_FULL_OUTPUT_FORMAT,
            "output_formats": output_formats,
            "hand_joint_names": list(HAND_21_JOINT_NAMES),
            "hand_joint_count_per_side": 21,
            "hand_transform_shape": ["T", 21, 4, 4],
            "body_joint_policy": "every named transform except the two 21-joint hands and camera is written to the body Parquet",
            "mano_joint_names": list(MANO_44_JOINT_NAMES),
            "mano_shape": ["T", 44, 4, 4],
            "camera_shape": ["T", 4, 4],
            "wrist_pose_shape": ["T", 9],
            "wrist_pose_fields": ["x", "y", "z", "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5"],
            "removed_vlm_phases": sorted(REMOVED_VLM_PHASES),
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
