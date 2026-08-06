from __future__ import annotations

import fnmatch
import json
import math
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet

from .video_smoothing import _create_frame_writer


NEXUS_LEROBOT_SCHEMA = "alice/nexus-lerobot-dataset/v1"
NEXUS_LEROBOT_CODEBASE_VERSION = "v2.1"
NEXUS_MASTER_FPS = 30.0
NEXUS_CHUNK_SIZE = 1_000
NEXUS_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
NEXUS_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
NEXUS_RAW_PATH = "raw_sensors/chunk-{episode_chunk:03d}/episode_{episode_index:06d}"

NEXUS_SKELETON_NODE_NAMES = tuple(f"source_node_{index:02d}" for index in range(20))
NEXUS_SKELETON_FIELDS = ("x", "y", "z", "quat_x", "quat_y", "quat_z", "quat_w")
NEXUS_JOINT_CHANNEL_NAMES = tuple(f"device_channel_{index}" for index in range(6))
NEXUS_TACTILE_CHANNEL_NAMES = tuple(f"taxel_{row:02d}_{column:02d}" for row in range(15) for column in range(15))
NEXUS_TACTILE_FEATURE_NAMES = (
    "active_taxel_count",
    "active_taxel_ratio",
    "pressure_sum",
    "active_pressure_mean",
    "pressure_max",
    "pressure_centroid_x",
    "pressure_centroid_y",
)

CAMERA_STREAMS = {
    "head": {
        "relative_path": "camera/head_rgb.mp4",
        "sync_column": "head_frame_idx",
        "video_key": "observation.images.head",
    },
    "wrist_left": {
        "relative_path": "camera/wrist_left.mp4",
        "sync_column": "wrist_left_frame_idx",
        "video_key": "observation.images.wrist_left",
    },
    "wrist_right": {
        "relative_path": "camera/wrist_right.mp4",
        "sync_column": "wrist_right_frame_idx",
        "video_key": "observation.images.wrist_right",
    },
}
DEFAULT_CAMERAS = tuple(CAMERA_STREAMS)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_parquet_atomic(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        parquet.write_table(table, temporary, compression="zstd")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _fixed_list(values: np.ndarray, width: int, value_type: pa.DataType) -> pa.FixedSizeListArray:
    contiguous = np.ascontiguousarray(values).reshape(-1, width)
    return pa.FixedSizeListArray.from_arrays(pa.array(contiguous.reshape(-1), type=value_type), width)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read Nexus JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Nexus JSON root must be an object: {path}")
    return value


def _is_nexus_episode(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "meta" / "metadata.json").is_file()
        and (path / "meta" / "sync.parquet").is_file()
        and (path / "mocap" / "dexweaveg1_left.h5").is_file()
        and (path / "mocap" / "dexweaveg1_right.h5").is_file()
    )


def discover_nexus_episodes(root: Path, patterns: Iterable[str] = ()) -> list[Path]:
    source = root.expanduser().resolve()
    candidates = [source] if _is_nexus_episode(source) else [path for path in source.iterdir() if _is_nexus_episode(path)]
    requested = [value.strip() for value in patterns if value.strip()]
    if requested:
        candidates = [path for path in candidates if any(fnmatch.fnmatchcase(path.name, pattern) for pattern in requested)]
        unmatched = [pattern for pattern in requested if not any(fnmatch.fnmatchcase(path.name, pattern) for path in candidates)]
        if unmatched:
            raise RuntimeError(f"Nexus Episode not found: {', '.join(unmatched)}")
    episodes = sorted(candidates, key=lambda path: path.name.casefold())
    if not episodes:
        raise RuntimeError(f"No Nexus v4 Episodes found under: {source}")
    return episodes


def _validate_output_root(source_root: Path, output_root: Path) -> None:
    source = source_root.expanduser().resolve()
    output = output_root.expanduser().resolve()
    if output == source:
        raise RuntimeError("Nexus source and LeRobot output paths must be different")
    if output.exists():
        contents = list(output.iterdir())
        if len(contents) == 1 and contents[0].name == "conversion_failed.json":
            contents[0].unlink(missing_ok=True)
        elif contents:
            raise RuntimeError(f"LeRobot output directory is not empty: {output}")


def _h5_dataset(file: h5py.File, name: str, shape_tail: tuple[int, ...], count: int) -> np.ndarray:
    value = file.get(name)
    if not isinstance(value, h5py.Dataset) or value.shape != (count, *shape_tail):
        actual = tuple(value.shape) if isinstance(value, h5py.Dataset) else None
        raise RuntimeError(f"Nexus dataset {file.filename}:{name} has shape {actual}, expected {(count, *shape_tail)}")
    return np.asarray(value[()])


def _h5_vector(file: h5py.File, name: str, count: int, default: object, dtype: np.dtype) -> np.ndarray:
    value = file.get(name)
    if isinstance(value, h5py.Dataset) and value.shape == (count,):
        return np.asarray(value[()], dtype=dtype)
    return np.full(count, default, dtype=dtype)


def _tactile_features(adc: np.ndarray) -> np.ndarray:
    values = np.asarray(adc, dtype=np.float32).reshape(-1, 15, 15)
    active = values > 0
    counts = active.sum(axis=(1, 2)).astype(np.float32)
    totals = values.sum(axis=(1, 2), dtype=np.float64).astype(np.float32)
    maximum = values.max(axis=(1, 2)).astype(np.float32)
    means = np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)
    x_grid = np.broadcast_to(np.arange(15, dtype=np.float32)[None, None, :], values.shape)
    y_grid = np.broadcast_to(np.arange(15, dtype=np.float32)[None, :, None], values.shape)
    centroid_x = np.divide((values * x_grid).sum(axis=(1, 2)), totals, out=np.full_like(totals, np.nan), where=totals > 0)
    centroid_y = np.divide((values * y_grid).sum(axis=(1, 2)), totals, out=np.full_like(totals, np.nan), where=totals > 0)
    return np.column_stack((counts, counts / 225.0, totals, means, maximum, centroid_x, centroid_y)).astype(np.float32)


def _aggregate_imu(episode_root: Path, master_timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = episode_root / "sensor" / "head_imu.h5"
    count = master_timestamps.size
    acceleration = np.full((count, 3), np.nan, dtype=np.float32)
    angular_velocity = np.full((count, 3), np.nan, dtype=np.float32)
    sample_counts = np.zeros(count, dtype=np.int32)
    valid = np.zeros(count, dtype=bool)
    if not path.is_file():
        return acceleration, angular_velocity, sample_counts, valid
    with h5py.File(path, "r") as source:
        accel = source.get("imu/accel")
        gyro = source.get("imu/gyro")
        timestamps = source.get("imu/host_arrival_ts_ns")
        if not all(isinstance(value, h5py.Dataset) for value in (accel, gyro, timestamps)):
            return acceleration, angular_velocity, sample_counts, valid
        accel_values = np.asarray(accel[()], dtype=np.float32)
        gyro_values = np.asarray(gyro[()], dtype=np.float32)
        timestamp_values = np.asarray(timestamps[()], dtype=np.float64) / 1_000_000_000.0
    if accel_values.shape != gyro_values.shape or accel_values.ndim != 2 or accel_values.shape[1] != 3 or timestamp_values.size != accel_values.shape[0]:
        return acceleration, angular_velocity, sample_counts, valid
    if count == 1:
        half_period = 0.5 / NEXUS_MASTER_FPS
        edges = np.asarray([master_timestamps[0] - half_period, master_timestamps[0] + half_period], dtype=np.float64)
    else:
        midpoints = (master_timestamps[:-1] + master_timestamps[1:]) * 0.5
        edges = np.concatenate((
            [master_timestamps[0] - (midpoints[0] - master_timestamps[0])],
            midpoints,
            [master_timestamps[-1] + (master_timestamps[-1] - midpoints[-1])],
        ))
    starts = np.searchsorted(timestamp_values, edges[:-1], side="left")
    ends = np.searchsorted(timestamp_values, edges[1:], side="left")
    finite = np.isfinite(accel_values).all(axis=1) & np.isfinite(gyro_values).all(axis=1)
    for row, (start, end) in enumerate(zip(starts, ends)):
        mask = finite[start:end]
        if mask.any():
            acceleration[row] = accel_values[start:end][mask].mean(axis=0)
            angular_velocity[row] = gyro_values[start:end][mask].mean(axis=0)
            sample_counts[row] = int(mask.sum())
            valid[row] = True
    return acceleration, angular_velocity, sample_counts, valid


def _load_episode_arrays(episode_root: Path) -> dict[str, np.ndarray | list[str]]:
    sync = parquet.read_table(episode_root / "meta" / "sync.parquet")
    required_sync = {
        "frame_idx", "master_ts", "partial", "partial_reason",
        *(str(item["sync_column"]) for item in CAMERA_STREAMS.values()),
    }
    missing = sorted(required_sync.difference(sync.column_names))
    if missing:
        raise RuntimeError(f"{episode_root.name}: sync.parquet is missing {', '.join(missing)}")
    count = sync.num_rows
    if count <= 0:
        raise RuntimeError(f"{episode_root.name}: sync.parquet has no rows")
    frame_idx = np.asarray(sync["frame_idx"].to_numpy(), dtype=np.int64)
    if not np.array_equal(frame_idx, np.arange(count, dtype=np.int64)):
        raise RuntimeError(f"{episode_root.name}: sync frame_idx must be contiguous from zero")
    master_ts = np.asarray(sync["master_ts"].to_numpy(), dtype=np.float64)
    if not np.isfinite(master_ts).all() or np.any(np.diff(master_ts) <= 0):
        raise RuntimeError(f"{episode_root.name}: master timestamps are invalid")
    result: dict[str, np.ndarray | list[str]] = {
        "source_frame_index": frame_idx,
        "master_timestamp": master_ts,
        "sync_partial": np.asarray(sync["partial"].to_numpy(), dtype=bool),
        "partial_reason": [str(value or "") for value in sync["partial_reason"].to_pylist()],
    }
    for camera, config in CAMERA_STREAMS.items():
        result[f"source_{camera}_frame_index"] = np.asarray(sync[str(config["sync_column"])].to_numpy(), dtype=np.int64)
    for key in (
        "depth_frame_idx", "tactile_left_source_seq", "tactile_right_source_seq", "mocap_dexweaveg1_source_seq",
    ):
        result[f"source_{key}"] = np.asarray(sync[key].to_numpy(), dtype=np.int64) if key in sync.column_names else np.full(count, -1, dtype=np.int64)

    for side in ("left", "right"):
        mocap_path = episode_root / "mocap" / f"dexweaveg1_{side}.h5"
        tactile_path = episode_root / "tactile" / f"{side}.h5"
        with h5py.File(mocap_path, "r") as source:
            result[f"{side}_skeleton"] = _h5_dataset(source, "skeleton", (20, 7), count).astype(np.float32)
            result[f"{side}_joints"] = _h5_dataset(source, "joints", (6,), count).astype(np.uint8)
            result[f"{side}_wrist_quaternion"] = _h5_dataset(source, "wrist_quat", (4,), count).astype(np.float32)
            result[f"{side}_mocap_partial"] = _h5_vector(source, "partial", count, False, np.dtype(bool))
        with h5py.File(tactile_path, "r") as source:
            tactile = _h5_dataset(source, "adc", (225,), count).astype(np.uint16)
            result[f"{side}_tactile"] = tactile
            result[f"{side}_tactile_features"] = _tactile_features(tactile)
            result[f"{side}_tactile_partial"] = _h5_vector(source, "partial", count, False, np.dtype(bool))

    acceleration, angular_velocity, imu_counts, imu_valid = _aggregate_imu(episode_root, master_ts)
    result["head_imu_acceleration"] = acceleration
    result["head_imu_angular_velocity"] = angular_velocity
    result["head_imu_sample_count"] = imu_counts
    result["head_imu_valid"] = imu_valid
    return result


def _document_for_episode(value: str | Path | None, episode_name: str) -> dict:
    if value is None:
        return {}
    path = Path(value).expanduser().resolve()
    if path.is_file():
        return _read_json(path)
    if not path.is_dir():
        raise RuntimeError(f"Sidecar path does not exist: {path}")
    names = (
        f"{episode_name}.json", f"{episode_name}.alice", f"{episode_name}.curation.alice",
        f"{episode_name}.behavior.alice", f"{episode_name}.annotation.alice",
    )
    for name in names:
        candidate = path / name
        if candidate.is_file():
            return _read_json(candidate)
    matches = [candidate for candidate in path.rglob("*") if candidate.is_file() and episode_name in candidate.name]
    if len(matches) == 1:
        return _read_json(matches[0])
    if len(matches) > 1:
        raise RuntimeError(f"More than one sidecar matches {episode_name} under {path}")
    content_matches: list[dict] = []
    for candidate in path.rglob("*"):
        if not candidate.is_file() or candidate.suffix.casefold() not in {".json", ".alice"}:
            continue
        try:
            payload = _read_json(candidate)
        except RuntimeError:
            continue
        identity_values = [
            payload.get("episode_name"),
            payload.get("source_episode"),
            (payload.get("source_video") or {}).get("relative_path") if isinstance(payload.get("source_video"), dict) else None,
            (payload.get("analysis_video") or {}).get("relative_path") if isinstance(payload.get("analysis_video"), dict) else None,
        ]
        if any(episode_name == str(item or "") or episode_name in Path(str(item or "")).parts for item in identity_values):
            content_matches.append(payload)
    if len(content_matches) == 1:
        return content_matches[0]
    if len(content_matches) > 1:
        raise RuntimeError(f"More than one sidecar document describes {episode_name} under {path}")
    return {}


def _merge_intervals(intervals: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    normalized = sorted((max(0, int(start)), min(count - 1, int(end))) for start, end in intervals if int(end) >= 0 and int(start) < count)
    merged: list[tuple[int, int]] = []
    for start, end in normalized:
        if end < start:
            continue
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _selected_intervals(curation: dict, count: int, minimum_frames: int) -> list[tuple[int, int]]:
    segments = curation.get("segments") or curation.get("pre_vlm_segments") or []
    valid = [
        (int(item.get("start_frame", 0)), int(item.get("end_frame", -1)))
        for item in segments
        if isinstance(item, dict) and str(item.get("state") or "").casefold() == "valid"
    ]
    intervals = _merge_intervals(valid, count) if valid else [(0, count - 1)]
    return [(start, end) for start, end in intervals if end - start + 1 >= max(1, minimum_frames)]


def _annotation_arrays(annotation: dict, count: int) -> tuple[np.ndarray, np.ndarray, str]:
    labels = np.full(count, "unknown", dtype=object)
    descriptions = np.full(count, "", dtype=object)
    segments = annotation.get("fine") or annotation.get("segments") or []
    for item in segments:
        if not isinstance(item, dict):
            continue
        try:
            start = max(0, int(item.get("start_frame", 0)))
            end = min(count - 1, int(item.get("end_frame", -1)))
        except (TypeError, ValueError):
            continue
        if end < start:
            continue
        label = str(item.get("skill") or item.get("phase_label") or item.get("label") or "unknown").strip() or "unknown"
        description = str(item.get("description") or item.get("sentence") or "").strip()
        labels[start:end + 1] = label
        descriptions[start:end + 1] = description
    coarse = annotation.get("coarse") or {}
    task = str((coarse.get("summary") or "") if isinstance(coarse, dict) else "").strip()
    task = task or str(annotation.get("task_label") or annotation.get("summary") or "").strip()
    return labels, descriptions, task


def _video_geometry(path: Path) -> tuple[int, int, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot decode Nexus video: {path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    if width <= 0 or height <= 0 or frame_count <= 0:
        raise RuntimeError(f"Nexus video metadata is invalid: {path}")
    return width, height, frame_count


def _write_resampled_video(source_path: Path, target_path: Path, source_indices: np.ndarray, fps: float) -> dict:
    indices = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0 or (indices < 0).any() or np.any(np.diff(indices) < 0):
        raise RuntimeError(f"Nexus video mapping is invalid: {source_path}")
    width, height, source_count = _video_geometry(source_path)
    if int(indices[-1]) >= source_count:
        raise RuntimeError(f"Nexus video mapping exceeds {source_path.name}: index={indices[-1]}, frames={source_count}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source_path))
    writer = _create_frame_writer(target_path, fps, width, height)
    writer_failed = True
    pointer = 0
    source_index = 0
    written = 0
    try:
        while pointer < indices.size:
            ok, frame = capture.read()
            if not ok:
                break
            if source_index == indices[pointer]:
                while pointer < indices.size and indices[pointer] == source_index:
                    writer.write(frame)
                    pointer += 1
                    written += 1
            source_index += 1
        writer.close()
        writer_failed = False
    finally:
        capture.release()
        if writer_failed:
            writer.abort()
    if written != indices.size:
        target_path.unlink(missing_ok=True)
        raise RuntimeError(f"Nexus video resampling failed: wrote={written}, expected={indices.size}, source={source_path}")
    if _video_geometry(target_path)[2] != indices.size:
        target_path.unlink(missing_ok=True)
        raise RuntimeError(f"LeRobot video frame count mismatch: {target_path}")
    return {
        "width": width,
        "height": height,
        "frame_count": written,
        "encoder": writer.name,
        "encoder_gpu": getattr(writer, "gpu_device", None),
    }


def _main_schema() -> pa.Schema:
    fields = [
        pa.field("observation.left_hand.skeleton", pa.list_(pa.float32(), 20 * 7)),
        pa.field("observation.right_hand.skeleton", pa.list_(pa.float32(), 20 * 7)),
        pa.field("observation.left_hand.joints", pa.list_(pa.uint8(), 6)),
        pa.field("observation.right_hand.joints", pa.list_(pa.uint8(), 6)),
        pa.field("observation.left_hand.wrist_quaternion", pa.list_(pa.float32(), 4)),
        pa.field("observation.right_hand.wrist_quaternion", pa.list_(pa.float32(), 4)),
        pa.field("observation.left_hand.tactile", pa.list_(pa.uint16(), 225)),
        pa.field("observation.right_hand.tactile", pa.list_(pa.uint16(), 225)),
        pa.field("observation.left_hand.tactile_features", pa.list_(pa.float32(), len(NEXUS_TACTILE_FEATURE_NAMES))),
        pa.field("observation.right_hand.tactile_features", pa.list_(pa.float32(), len(NEXUS_TACTILE_FEATURE_NAMES))),
        pa.field("observation.head_imu.acceleration_mean", pa.list_(pa.float32(), 3)),
        pa.field("observation.head_imu.angular_velocity_mean", pa.list_(pa.float32(), 3)),
        pa.field("observation.head_imu.sample_count", pa.int32()),
        pa.field("observation.head_imu.valid", pa.bool_()),
        pa.field("annotation.phase_label", pa.string()),
        pa.field("annotation.action_description", pa.string()),
        pa.field("quality.partial", pa.bool_()),
        pa.field("quality.partial_reason", pa.string()),
        pa.field("quality.left_mocap_partial", pa.bool_()),
        pa.field("quality.right_mocap_partial", pa.bool_()),
        pa.field("quality.left_tactile_partial", pa.bool_()),
        pa.field("quality.right_tactile_partial", pa.bool_()),
        pa.field("source.master_frame_index", pa.int64()),
        pa.field("source.master_timestamp", pa.float64()),
        pa.field("source.head_frame_index", pa.int64()),
        pa.field("source.wrist_left_frame_index", pa.int64()),
        pa.field("source.wrist_right_frame_index", pa.int64()),
        pa.field("source.depth_frame_index", pa.int64()),
        pa.field("source.tactile_left_sequence", pa.int64()),
        pa.field("source.tactile_right_sequence", pa.int64()),
        pa.field("source.mocap_sequence", pa.int64()),
        pa.field("timestamp", pa.float32()),
        pa.field("frame_index", pa.int64()),
        pa.field("episode_index", pa.int64()),
        pa.field("index", pa.int64()),
        pa.field("task_index", pa.int64()),
    ]
    return pa.schema(fields)


def _episode_table(
    arrays: dict[str, np.ndarray | list[str]],
    rows: np.ndarray,
    labels: np.ndarray,
    descriptions: np.ndarray,
    episode_index: int,
    global_offset: int,
    task_index: int,
) -> pa.Table:
    count = rows.size
    local_frames = np.arange(count, dtype=np.int64)
    global_indices = global_offset + local_frames
    left_mocap_partial = np.asarray(arrays["left_mocap_partial"], dtype=bool)[rows]
    right_mocap_partial = np.asarray(arrays["right_mocap_partial"], dtype=bool)[rows]
    left_tactile_partial = np.asarray(arrays["left_tactile_partial"], dtype=bool)[rows]
    right_tactile_partial = np.asarray(arrays["right_tactile_partial"], dtype=bool)[rows]
    partial = (
        np.asarray(arrays["sync_partial"], dtype=bool)[rows]
        | left_mocap_partial | right_mocap_partial | left_tactile_partial | right_tactile_partial
    )
    columns = [
        _fixed_list(np.asarray(arrays["left_skeleton"])[rows], 20 * 7, pa.float32()),
        _fixed_list(np.asarray(arrays["right_skeleton"])[rows], 20 * 7, pa.float32()),
        _fixed_list(np.asarray(arrays["left_joints"])[rows], 6, pa.uint8()),
        _fixed_list(np.asarray(arrays["right_joints"])[rows], 6, pa.uint8()),
        _fixed_list(np.asarray(arrays["left_wrist_quaternion"])[rows], 4, pa.float32()),
        _fixed_list(np.asarray(arrays["right_wrist_quaternion"])[rows], 4, pa.float32()),
        _fixed_list(np.asarray(arrays["left_tactile"])[rows], 225, pa.uint16()),
        _fixed_list(np.asarray(arrays["right_tactile"])[rows], 225, pa.uint16()),
        _fixed_list(np.asarray(arrays["left_tactile_features"])[rows], len(NEXUS_TACTILE_FEATURE_NAMES), pa.float32()),
        _fixed_list(np.asarray(arrays["right_tactile_features"])[rows], len(NEXUS_TACTILE_FEATURE_NAMES), pa.float32()),
        _fixed_list(np.asarray(arrays["head_imu_acceleration"])[rows], 3, pa.float32()),
        _fixed_list(np.asarray(arrays["head_imu_angular_velocity"])[rows], 3, pa.float32()),
        pa.array(np.asarray(arrays["head_imu_sample_count"])[rows], type=pa.int32()),
        pa.array(np.asarray(arrays["head_imu_valid"])[rows], type=pa.bool_()),
        pa.array([str(value) for value in labels[rows]], type=pa.string()),
        pa.array([str(value) for value in descriptions[rows]], type=pa.string()),
        pa.array(partial, type=pa.bool_()),
        pa.array([str(arrays["partial_reason"][int(row)]) for row in rows], type=pa.string()),
        pa.array(left_mocap_partial, type=pa.bool_()),
        pa.array(right_mocap_partial, type=pa.bool_()),
        pa.array(left_tactile_partial, type=pa.bool_()),
        pa.array(right_tactile_partial, type=pa.bool_()),
        pa.array(np.asarray(arrays["source_frame_index"])[rows], type=pa.int64()),
        pa.array(np.asarray(arrays["master_timestamp"])[rows], type=pa.float64()),
        pa.array(np.asarray(arrays["source_head_frame_index"])[rows], type=pa.int64()),
        pa.array(np.asarray(arrays["source_wrist_left_frame_index"])[rows], type=pa.int64()),
        pa.array(np.asarray(arrays["source_wrist_right_frame_index"])[rows], type=pa.int64()),
        pa.array(np.asarray(arrays["source_depth_frame_idx"])[rows], type=pa.int64()),
        pa.array(np.asarray(arrays["source_tactile_left_source_seq"])[rows], type=pa.int64()),
        pa.array(np.asarray(arrays["source_tactile_right_source_seq"])[rows], type=pa.int64()),
        pa.array(np.asarray(arrays["source_mocap_dexweaveg1_source_seq"])[rows], type=pa.int64()),
        pa.array(local_frames.astype(np.float32) / NEXUS_MASTER_FPS, type=pa.float32()),
        pa.array(local_frames, type=pa.int64()),
        pa.array(np.full(count, episode_index, dtype=np.int64), type=pa.int64()),
        pa.array(global_indices, type=pa.int64()),
        pa.array(np.full(count, task_index, dtype=np.int64), type=pa.int64()),
    ]
    return pa.Table.from_arrays(columns, schema=_main_schema())


def _video_feature(width: int, height: int) -> dict:
    return {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": height,
            "video.width": width,
            "video.channels": 3,
            "video.fps": NEXUS_MASTER_FPS,
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }


def _features(camera_geometry: dict[str, tuple[int, int]]) -> dict:
    features = {
        "observation.left_hand.skeleton": {"dtype": "float32", "shape": [20, 7], "names": list(NEXUS_SKELETON_NODE_NAMES), "coordinate_names": list(NEXUS_SKELETON_FIELDS)},
        "observation.right_hand.skeleton": {"dtype": "float32", "shape": [20, 7], "names": list(NEXUS_SKELETON_NODE_NAMES), "coordinate_names": list(NEXUS_SKELETON_FIELDS)},
        "observation.left_hand.joints": {"dtype": "uint8", "shape": [6], "names": list(NEXUS_JOINT_CHANNEL_NAMES)},
        "observation.right_hand.joints": {"dtype": "uint8", "shape": [6], "names": list(NEXUS_JOINT_CHANNEL_NAMES)},
        "observation.left_hand.wrist_quaternion": {"dtype": "float32", "shape": [4], "names": ["x", "y", "z", "w"]},
        "observation.right_hand.wrist_quaternion": {"dtype": "float32", "shape": [4], "names": ["x", "y", "z", "w"]},
        "observation.left_hand.tactile": {"dtype": "uint16", "shape": [225], "names": list(NEXUS_TACTILE_CHANNEL_NAMES)},
        "observation.right_hand.tactile": {"dtype": "uint16", "shape": [225], "names": list(NEXUS_TACTILE_CHANNEL_NAMES)},
        "observation.left_hand.tactile_features": {"dtype": "float32", "shape": [7], "names": list(NEXUS_TACTILE_FEATURE_NAMES)},
        "observation.right_hand.tactile_features": {"dtype": "float32", "shape": [7], "names": list(NEXUS_TACTILE_FEATURE_NAMES)},
        "observation.head_imu.acceleration_mean": {"dtype": "float32", "shape": [3], "names": ["x", "y", "z"]},
        "observation.head_imu.angular_velocity_mean": {"dtype": "float32", "shape": [3], "names": ["x", "y", "z"]},
        "observation.head_imu.sample_count": {"dtype": "int32", "shape": [1], "names": None},
        "observation.head_imu.valid": {"dtype": "bool", "shape": [1], "names": None},
        "annotation.phase_label": {"dtype": "string", "shape": [1], "names": None},
        "annotation.action_description": {"dtype": "string", "shape": [1], "names": None},
        "quality.partial": {"dtype": "bool", "shape": [1], "names": None},
        "quality.partial_reason": {"dtype": "string", "shape": [1], "names": None},
        "quality.left_mocap_partial": {"dtype": "bool", "shape": [1], "names": None},
        "quality.right_mocap_partial": {"dtype": "bool", "shape": [1], "names": None},
        "quality.left_tactile_partial": {"dtype": "bool", "shape": [1], "names": None},
        "quality.right_tactile_partial": {"dtype": "bool", "shape": [1], "names": None},
        "source.master_frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "source.master_timestamp": {"dtype": "float64", "shape": [1], "names": None},
        "source.head_frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "source.wrist_left_frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "source.wrist_right_frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "source.depth_frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "source.tactile_left_sequence": {"dtype": "int64", "shape": [1], "names": None},
        "source.tactile_right_sequence": {"dtype": "int64", "shape": [1], "names": None},
        "source.mocap_sequence": {"dtype": "int64", "shape": [1], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for camera, (width, height) in camera_geometry.items():
        features[str(CAMERA_STREAMS[camera]["video_key"])] = _video_feature(width, height)
    return features


def _copy_raw_sidecars(episode_root: Path, output_root: Path, episode_index: int) -> str:
    chunk = episode_index // NEXUS_CHUNK_SIZE
    target = output_root / NEXUS_RAW_PATH.format(episode_chunk=chunk, episode_index=episode_index)
    selected = [
        episode_root / "camera" / "head_depth.raw",
        episode_root / "sensor" / "head_imu.h5",
        *(episode_root / "mocap" / name for name in ("dexweaveg1_left_raw.h5", "dexweaveg1_right_raw.h5")),
        *(episode_root / "tactile" / name for name in ("left_raw.h5", "right_raw.h5")),
    ]
    for source in selected:
        if not source.is_file():
            continue
        relative = source.relative_to(episode_root)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return str(target)


def convert_nexus_to_lerobot(
    input_root: str | Path,
    output_root: str | Path,
    *,
    episode_patterns: Iterable[str] = (),
    cameras: Iterable[str] = DEFAULT_CAMERAS,
    curation: str | Path | None = None,
    annotations: str | Path | None = None,
    alice_sidecar: str | Path | None = None,
    preserve_raw_sensors: bool = False,
    minimum_segment_seconds: float = 0.0,
    task: str | None = None,
    progress=None,
) -> dict:
    source_root = Path(input_root).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    _validate_output_root(source_root, destination)
    episodes = discover_nexus_episodes(source_root, episode_patterns)
    selected_cameras = tuple(dict.fromkeys(str(value).strip() for value in cameras if str(value).strip()))
    unknown_cameras = sorted(set(selected_cameras).difference(CAMERA_STREAMS))
    if unknown_cameras:
        raise RuntimeError(f"Unknown Nexus cameras: {', '.join(unknown_cameras)}")
    if not selected_cameras:
        raise RuntimeError("Select at least one Nexus RGB camera")
    if alice_sidecar is not None:
        sidecar = Path(alice_sidecar).expanduser().resolve()
        if not sidecar.is_dir():
            raise RuntimeError(f"Alice sidecar directory does not exist: {sidecar}")
        if curation is None:
            curation = sidecar / "curation"
        if annotations is None:
            annotations = sidecar / "behavior-annotations"
    destination.mkdir(parents=True, exist_ok=True)
    notify = progress or (lambda _value, _message: None)
    minimum_frames = max(1, int(math.ceil(max(0.0, minimum_segment_seconds) * NEXUS_MASTER_FPS)))

    output_episodes: list[dict] = []
    tasks: dict[str, int] = {}
    camera_geometry: dict[str, tuple[int, int]] = {}
    raw_sidecars: dict[str, str] = {}
    global_offset = 0
    episode_index = 0
    try:
        for source_position, episode_root in enumerate(episodes, start=1):
            notify(100.0 * (source_position - 1) / max(1, len(episodes)), f"Reading Nexus Episode {episode_root.name}")
            metadata = _read_json(episode_root / "meta" / "metadata.json")
            if not str(metadata.get("schema_version") or metadata.get("nexus_version") or "").startswith("4"):
                raise RuntimeError(f"{episode_root.name}: only Nexus v4 is supported")
            arrays = _load_episode_arrays(episode_root)
            source_count = len(np.asarray(arrays["source_frame_index"]))
            curation_document = _document_for_episode(curation, episode_root.name)
            annotation_document = _document_for_episode(annotations, episode_root.name)
            labels, descriptions, annotation_task = _annotation_arrays(annotation_document, source_count)
            intervals = _selected_intervals(curation_document, source_count, minimum_frames)
            if not intervals:
                continue
            task_name = str(task or annotation_task or episode_root.name).strip() or episode_root.name
            if task_name not in tasks:
                tasks[task_name] = len(tasks)
            task_index = tasks[task_name]

            for segment_number, (start, end) in enumerate(intervals, start=1):
                rows = np.arange(start, end + 1, dtype=np.int64)
                count = rows.size
                chunk = episode_index // NEXUS_CHUNK_SIZE
                data_path = destination / NEXUS_DATA_PATH.format(episode_chunk=chunk, episode_index=episode_index)
                video_paths: dict[str, str] = {}
                staged: list[tuple[Path, Path]] = []
                raw_sidecar = None
                try:
                    table = _episode_table(arrays, rows, labels, descriptions, episode_index, global_offset, task_index)
                    _write_parquet_atomic(data_path, table)
                    if parquet.ParquetFile(data_path).metadata.num_rows != count:
                        raise RuntimeError(f"LeRobot Parquet row count mismatch: {data_path}")
                    for camera in selected_cameras:
                        config = CAMERA_STREAMS[camera]
                        source_video = episode_root / str(config["relative_path"])
                        video_key = str(config["video_key"])
                        final_path = destination / NEXUS_VIDEO_PATH.format(
                            episode_chunk=chunk,
                            episode_index=episode_index,
                            video_key=video_key,
                        )
                        final_path.parent.mkdir(parents=True, exist_ok=True)
                        temporary = final_path.with_name(f".{final_path.stem}.{uuid.uuid4().hex}.part.mp4")
                        info = _write_resampled_video(
                            source_video,
                            temporary,
                            np.asarray(arrays[f"source_{camera}_frame_index"])[rows],
                            NEXUS_MASTER_FPS,
                        )
                        geometry = (int(info["width"]), int(info["height"]))
                        if camera in camera_geometry and camera_geometry[camera] != geometry:
                            raise RuntimeError(f"Nexus camera geometry changed for {camera}: {camera_geometry[camera]} -> {geometry}")
                        camera_geometry[camera] = geometry
                        staged.append((temporary, final_path))
                        video_paths[video_key] = str(final_path)
                    for temporary, final_path in staged:
                        temporary.replace(final_path)
                    if preserve_raw_sensors:
                        raw_sidecar = raw_sidecars.get(episode_root.name)
                        if raw_sidecar is None:
                            raw_sidecar = _copy_raw_sidecars(episode_root, destination, episode_index)
                            raw_sidecars[episode_root.name] = raw_sidecar
                except Exception:
                    data_path.unlink(missing_ok=True)
                    for temporary, final_path in staged:
                        temporary.unlink(missing_ok=True)
                        final_path.unlink(missing_ok=True)
                    raise
                item = {
                    "episode_index": episode_index,
                    "source_episode": episode_root.name,
                    "source_start_frame": start,
                    "source_end_frame": end,
                    "segment_number": segment_number,
                    "task": task_name,
                    "task_index": task_index,
                    "length": int(count),
                    "data": str(data_path),
                    "videos": video_paths,
                    "raw_sensor_sidecar": raw_sidecar,
                }
                output_episodes.append(item)
                global_offset += count
                episode_index += 1
            notify(100.0 * source_position / max(1, len(episodes)), f"Converted Nexus Episode {episode_root.name}")

        if not output_episodes:
            raise RuntimeError("No Nexus frames remained after curation filtering")
        task_items = sorted(tasks.items(), key=lambda item: item[1])
        _write_parquet_atomic(destination / "meta" / "tasks.parquet", pa.table({
            "task_index": pa.array([index for _, index in task_items], type=pa.int64()),
            "task": pa.array([name for name, _ in task_items], type=pa.string()),
        }))
        by_chunk: dict[int, list[dict]] = {}
        for item in output_episodes:
            by_chunk.setdefault(int(item["episode_index"]) // NEXUS_CHUNK_SIZE, []).append(item)
        for chunk, items in by_chunk.items():
            _write_parquet_atomic(destination / "meta" / "episodes" / f"chunk-{chunk:03d}" / "file-000.parquet", pa.table({
                "episode_index": pa.array([int(item["episode_index"]) for item in items], type=pa.int64()),
                "tasks": pa.array([[str(item["task"])] for item in items], type=pa.list_(pa.string())),
                "length": pa.array([int(item["length"]) for item in items], type=pa.int64()),
            }))
        features = _features(camera_geometry)
        info = {
            "codebase_version": NEXUS_LEROBOT_CODEBASE_VERSION,
            "robot_type": "nexus_dexweaveg1_bimanual_multimodal",
            "fps": NEXUS_MASTER_FPS,
            "total_episodes": len(output_episodes),
            "total_frames": global_offset,
            "total_tasks": len(tasks),
            "total_videos": len(output_episodes) * len(selected_cameras),
            "total_chunks": max(1, math.ceil(len(output_episodes) / NEXUS_CHUNK_SIZE)),
            "chunks_size": NEXUS_CHUNK_SIZE,
            "splits": {"train": f"0:{len(output_episodes)}"},
            "data_path": NEXUS_DATA_PATH,
            "video_path": NEXUS_VIDEO_PATH,
            "features": features,
            "source_format": "nexus_v4",
            "master_timeline": "meta/sync.parquet at 30Hz",
            "skeleton_policy": "Preserve the source 20-node order; never pad or reinterpret it as MANO 21.",
            "tactile_policy": "Preserve 225 zero-corrected taxels and add auxiliary contact statistics; tactile is not a hard deletion signal.",
            "imu_policy": "Mean acceleration and angular velocity inside each 30Hz master interval.",
            "action_policy": "No action feature is emitted unless a real robot action exists.",
            "raw_sensor_sidecars": NEXUS_RAW_PATH if preserve_raw_sensors else None,
        }
        _write_json_atomic(destination / "meta" / "info.json", info)
        _write_json_atomic(destination / "meta" / "stats.json", {})
        report = {
            "schema": NEXUS_LEROBOT_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source_root),
            "output_root": str(destination),
            "source_episode_count": len(episodes),
            "output_episode_count": len(output_episodes),
            "frame_count": global_offset,
            "fps": NEXUS_MASTER_FPS,
            "cameras": list(selected_cameras),
            "preserve_raw_sensors": preserve_raw_sensors,
            "curation_source": str(curation) if curation is not None else None,
            "annotation_source": str(annotations) if annotations is not None else None,
            "alice_sidecar": str(alice_sidecar) if alice_sidecar is not None else None,
            "episodes": output_episodes,
        }
        _write_json_atomic(destination / "dataset.json", report)
        return report
    except Exception:
        _write_json_atomic(destination / "conversion_failed.json", {
            "schema": NEXUS_LEROBOT_SCHEMA,
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source_root),
            "output_root": str(destination),
            "completed_episodes": output_episodes,
        })
        raise


def command_nexus_lerobot(args) -> int:
    cameras = [item.strip() for item in str(args.cameras).split(",") if item.strip()]

    def progress(value: float, message: str) -> None:
        if not args.json:
            print(f"[{value:6.2f}%] {message}")

    result = convert_nexus_to_lerobot(
        args.input,
        args.output,
        episode_patterns=args.episode or (),
        cameras=cameras,
        curation=args.curation,
        annotations=args.annotations,
        alice_sidecar=args.alice_sidecar,
        preserve_raw_sensors=bool(args.preserve_raw_sensors),
        minimum_segment_seconds=float(args.minimum_segment_seconds),
        task=args.task,
        progress=progress,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(
            f"Nexus -> LeRobot complete: {result['output_episode_count']} Episodes, "
            f"{result['frame_count']} frames, output={result['output_root']}"
        )
    return 0


def add_parser(subparsers):
    parser = subparsers.add_parser("nexus-lerobot", help="Convert processed Nexus v4 data to a 30Hz LeRobot dataset")
    parser.add_argument("input", help="Nexus Episode directory or a directory containing Nexus Episodes")
    parser.add_argument("output", help="New empty LeRobot output directory")
    parser.add_argument("--episode", action="append", help="Episode name/glob; repeat this option to select more")
    parser.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS), help="Comma-separated: head,wrist_left,wrist_right")
    parser.add_argument("--curation", help="Curation JSON/.alice file or directory; valid ranges become separate LeRobot Episodes")
    parser.add_argument("--annotations", help="VLM annotation JSON/.alice file or directory for labels and English descriptions")
    parser.add_argument("--alice-sidecar", help="Alice .alicePD dataset directory; automatically reads curation and behavior-annotations")
    parser.add_argument("--task", help="Override the LeRobot task text")
    parser.add_argument("--minimum-segment-seconds", type=float, default=0.0)
    parser.add_argument("--preserve-raw-sensors", action="store_true", help="Copy native-rate tactile, mocap, IMU and raw depth sidecars")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(handler=command_nexus_lerobot)
    return parser
