from __future__ import annotations

import json
import math
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet

from .egodex_mano import (
    EGODEX_MANO_REVISION,
    egodex_mano_source_names,
    fit_egodex_mano_template,
    has_egodex_mano_source,
    retarget_egodex_mano_frame,
    source_is_retargeted,
)
from .mano21 import HAND_21_JOINT_NAMES, mano21_transforms_from_named, side_hand_joint_names
from .s1_repair import apply_s1_repair, s1_repair_cell_count
from .temporal_resampling import interpolate_transform_rows, sample_hdf5_numeric


LEROBOT_CODEBASE_VERSION = "v2.1"
LEROBOT_CHUNK_SIZE = 1_000
LEROBOT_VIDEO_KEY = "observation.images.main"
LEROBOT_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
LEROBOT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/observation.images.main/episode_{episode_index:06d}.mp4"
LEROBOT_BODY_PATH = "body/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
EGODEX_CAMERA_INTRINSIC_1920X1080 = np.asarray([
    [736.6339, 0.0, 960.0],
    [0.0, 736.6339, 540.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

_LOCK = threading.RLock()


LEFT_HAND_SOURCE_NAMES = side_hand_joint_names("left")
RIGHT_HAND_SOURCE_NAMES = side_hand_joint_names("right")
HAND_SOURCE_NAMES = (*LEFT_HAND_SOURCE_NAMES, *RIGHT_HAND_SOURCE_NAMES)


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


def _take_rows(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(rows, return_inverse=True)
    return np.asarray(dataset[unique.tolist()])[inverse]


def _sample_repaired_transforms(
    dataset: h5py.Dataset,
    positions: np.ndarray,
    repair: dict | None,
    source_relative: str,
    field: str,
) -> np.ndarray:
    resolved = np.asarray(positions, dtype=np.float64).reshape(-1)
    left = np.floor(resolved).astype(np.int64)
    right = np.ceil(resolved).astype(np.int64)
    rows = np.unique(np.concatenate((left, right)))
    values = apply_s1_repair(
        np.asarray(dataset[rows.tolist()]),
        repair,
        source_relative,
        field,
        rows,
    )
    left_local = np.searchsorted(rows, left)
    right_local = np.searchsorted(rows, right)
    alpha = resolved - left
    local_positions = left_local.astype(np.float64) * (1.0 - alpha) + right_local.astype(np.float64) * alpha
    return interpolate_transform_rows(values, local_positions)


def _rot6d(transform: np.ndarray) -> np.ndarray:
    rotation = np.asarray(transform[:, :3, :3], dtype=np.float32)
    return rotation[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)


def _fixed_float_list(values: np.ndarray, width: int) -> pa.FixedSizeListArray:
    contiguous = np.ascontiguousarray(values, dtype=np.float32).reshape(-1, width)
    return pa.FixedSizeListArray.from_arrays(pa.array(contiguous.reshape(-1), type=pa.float32()), width)


def scaled_egodex_camera_intrinsic(width: int, height: int) -> np.ndarray:
    intrinsic = EGODEX_CAMERA_INTRINSIC_1920X1080.copy()
    intrinsic[0] *= max(1, int(width)) / 1920.0
    intrinsic[1] *= max(1, int(height)) / 1080.0
    intrinsic[2] = (0.0, 0.0, 1.0)
    return intrinsic


def _source_camera_intrinsic(source_path: Path, width: int, height: int) -> np.ndarray:
    with h5py.File(source_path, "r") as source:
        value = source.get("camera/intrinsic")
        if isinstance(value, h5py.Dataset):
            intrinsic = np.asarray(value[()], dtype=np.float32)
            if intrinsic.ndim == 3:
                intrinsic = intrinsic[0]
            if intrinsic.shape == (3, 3) and np.isfinite(intrinsic).all():
                return intrinsic
    return scaled_egodex_camera_intrinsic(width, height)


def discover_body_joint_names(source_path: Path, source_count: int) -> tuple[str, ...]:
    excluded = {name.casefold() for name in (*HAND_SOURCE_NAMES, "camera")}
    with h5py.File(source_path, "r") as source:
        transforms = source.get("transforms")
        if not isinstance(transforms, h5py.Group):
            return ()
        return tuple(sorted(
            name
            for name, value in transforms.items()
            if name.casefold() not in excluded
            and isinstance(value, h5py.Dataset)
            and value.shape == (source_count, 4, 4)
        ))


def _read_state(output_root: Path) -> dict:
    path = output_root / "meta" / "alice_state.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") == "alice/lerobot-export-state/v1":
            return payload
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return {
        "schema": "alice/lerobot-export-state/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "next_episode_index": 0,
        "next_frame_index": 0,
        "tasks": {},
        "fps": None,
        "width": None,
        "height": None,
        "body_joint_names": None,
        "camera_intrinsic": None,
        "action_profile_id": None,
        "action_dim": None,
        "action_fields": None,
        "action_representation": None,
        "action_coordinate_frame": None,
        "action_horizon_frames": None,
        "observation_state_fields": None,
    }


def _validate_dataset_contract(
    state: dict,
    fps: float,
    width: int,
    height: int,
    body_names: tuple[str, ...],
    camera_intrinsic: np.ndarray,
    action_metadata: dict | None,
) -> None:
    if state.get("fps") is not None and not math.isclose(float(state["fps"]), fps, rel_tol=0.0, abs_tol=1e-3):
        raise RuntimeError(f"LeRobot output requires one FPS: existing={state['fps']}, current={fps}")
    if state.get("width") is not None and (int(state["width"]), int(state["height"])) != (width, height):
        raise RuntimeError(
            "LeRobot output requires one video geometry: "
            f"existing={state['width']}x{state['height']}, current={width}x{height}"
        )
    existing_body = state.get("body_joint_names")
    if existing_body is not None and tuple(existing_body) != body_names:
        raise RuntimeError("LeRobot body joint schema differs from previous exported Episodes")
    existing_intrinsic = state.get("camera_intrinsic")
    if existing_intrinsic is not None:
        resolved = np.asarray(existing_intrinsic, dtype=np.float32)
        if resolved.shape != (3, 3) or not np.allclose(resolved, camera_intrinsic, rtol=0.0, atol=1e-3):
            raise RuntimeError("LeRobot camera intrinsic differs from previous exported Episodes")
    current_action_dim = int(action_metadata.get("action_dim") or 0) if action_metadata else None
    existing_action_dim = state.get("action_dim")
    has_existing_episodes = int(state.get("next_episode_index") or 0) > 0
    if has_existing_episodes and (existing_action_dim is not None or current_action_dim is not None):
        if int(existing_action_dim or 0) != int(current_action_dim or 0):
            raise RuntimeError("LeRobot Action schema differs from previous exported Episodes")
        for key in (
            "action_profile_id", "action_fields", "action_representation",
            "action_coordinate_frame", "action_horizon_frames", "observation_state_fields",
        ):
            existing_value = state.get(key)
            current_value = (action_metadata or {}).get(key)
            if existing_value is not None and existing_value != current_value:
                raise RuntimeError(f"LeRobot {key} differs from previous exported Episodes")


def _episode_paths(output_root: Path, episode_index: int) -> tuple[Path, Path, Path]:
    chunk = episode_index // LEROBOT_CHUNK_SIZE
    data = output_root / LEROBOT_DATA_PATH.format(episode_chunk=chunk, episode_index=episode_index)
    video = output_root / LEROBOT_VIDEO_PATH.format(episode_chunk=chunk, episode_index=episode_index)
    body = output_root / LEROBOT_BODY_PATH.format(episode_chunk=chunk, episode_index=episode_index)
    return data, video, body


def _main_schema(action_dim: int | None = None) -> pa.Schema:
    fields = [
        pa.field("observation.left_hand.transforms", pa.list_(pa.float32(), 21 * 4 * 4)),
        pa.field("observation.left_hand.confidence", pa.list_(pa.float32(), 21)),
        pa.field("observation.right_hand.transforms", pa.list_(pa.float32(), 21 * 4 * 4)),
        pa.field("observation.right_hand.confidence", pa.list_(pa.float32(), 21)),
        pa.field("observation.camera.transform", pa.list_(pa.float32(), 4 * 4)),
        pa.field("observation.camera.intrinsic", pa.list_(pa.float32(), 3 * 3)),
        pa.field("observation.camera.image_transform", pa.list_(pa.float32(), 3 * 3)),
        pa.field("observation.left_wrist.xyz_rot6d", pa.list_(pa.float32(), 9)),
        pa.field("observation.right_wrist.xyz_rot6d", pa.list_(pa.float32(), 9)),
        pa.field("annotation.phase_label", pa.string()),
        pa.field("quality.state", pa.string()),
        pa.field("quality.is_bad", pa.bool_()),
        pa.field("quality.needs_review", pa.bool_()),
    ]
    if action_dim is not None:
        fields.extend([
            pa.field("observation.state", pa.list_(pa.float32(), 20)),
            pa.field("action", pa.list_(pa.float32(), int(action_dim))),
            pa.field("action.target_source_frame_index", pa.int64()),
            pa.field("quality.action_valid", pa.bool_()),
        ])
    fields.extend([
        pa.field("source.frame_index", pa.int64()),
        pa.field("source.hdf5_row", pa.int64()),
        pa.field("source.hdf5_position", pa.float64()),
        pa.field("source.video_frame_position", pa.float64()),
        pa.field("source.timestamp", pa.float64()),
        pa.field("timestamp", pa.float32()),
        pa.field("frame_index", pa.int64()),
        pa.field("episode_index", pa.int64()),
        pa.field("index", pa.int64()),
        pa.field("task_index", pa.int64()),
    ])
    return pa.schema(fields)


def _body_schema(body_count: int) -> pa.Schema:
    return pa.schema([
        pa.field("observation.body.transforms", pa.list_(pa.float32(), body_count * 4 * 4)),
        pa.field("observation.body.confidence", pa.list_(pa.float32(), body_count)),
        pa.field("timestamp", pa.float32()),
        pa.field("frame_index", pa.int64()),
        pa.field("episode_index", pa.int64()),
        pa.field("index", pa.int64()),
    ])


def _confidence_values(
    confidences: h5py.Group | None,
    names: tuple[str, ...],
    positions: np.ndarray,
) -> np.ndarray:
    values = np.full((positions.size, len(names)), np.nan, dtype=np.float32)
    if isinstance(confidences, h5py.Group):
        for index, name in enumerate(names):
            if name in confidences:
                values[:, index] = sample_hdf5_numeric(confidences[name], positions).reshape(-1).astype(np.float32)
    return values


def _write_episode_parquet(
    data_path: Path,
    body_path: Path,
    source_path: Path,
    source_rows: np.ndarray,
    source_frames: np.ndarray,
    source_video_positions: np.ndarray,
    phase_labels: np.ndarray,
    body_names: tuple[str, ...],
    fps: float,
    episode_index: int,
    frame_offset: int,
    task_index: int,
    source_relative: str,
    repair: dict | None,
    camera_intrinsic: np.ndarray,
    camera_image_transforms: np.ndarray,
    quality_states: np.ndarray,
    action_payload: dict | None,
) -> None:
    count = int(source_frames.size)
    resolved_quality = np.asarray(quality_states, dtype=object).reshape(-1)
    if resolved_quality.shape != (count,):
        raise ValueError("LeRobot quality state array does not match exported frames")
    action_dim = int(action_payload.get("action_dim") or 0) if action_payload else None
    if action_payload is not None:
        if (
            np.asarray(action_payload.get("observation_state")).shape != (count, 20)
            or np.asarray(action_payload.get("action")).shape != (count, action_dim)
            or np.asarray(action_payload.get("target_frame_index")).shape != (count,)
            or np.asarray(action_payload.get("valid")).shape != (count,)
        ):
            raise ValueError("LeRobot Action arrays do not match exported frames")
    main_schema = _main_schema(action_dim)
    chunk_size = max(1, min(256, count))
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if body_names:
        body_path.parent.mkdir(parents=True, exist_ok=True)
    data_temporary = data_path.with_name(f".{data_path.name}.{uuid.uuid4().hex}.tmp")
    body_temporary = body_path.with_name(f".{body_path.name}.{uuid.uuid4().hex}.tmp")
    data_writer = parquet.ParquetWriter(data_temporary, main_schema, compression="zstd")
    body_writer = parquet.ParquetWriter(body_temporary, _body_schema(len(body_names)), compression="zstd") if body_names else None
    failed = True
    try:
        with h5py.File(source_path, "r") as source:
            transforms = source["transforms"]
            confidences = source.get("confidences")
            already_retargeted = source_is_retargeted(source)
            templates = {
                side: (
                    None
                    if already_retargeted or not has_egodex_mano_source(transforms, side)
                    else fit_egodex_mano_template(transforms, side)
                )
                for side in ("left", "right")
            }

            def read_hand(side: str, positions: np.ndarray) -> np.ndarray:
                template = templates[side]
                required = side_hand_joint_names(side) if template is None else egodex_mano_source_names(transforms, side)
                blocks = {
                    name: _sample_repaired_transforms(
                        transforms[name], positions, repair, source_relative, f"transforms/{name}",
                    )
                    for name in required
                }
                if template is None:
                    return mano21_transforms_from_named(blocks, side).astype(np.float32)
                return np.stack([
                    retarget_egodex_mano_frame(
                        {name: values[local_index] for name, values in blocks.items()},
                        template,
                    )
                    for local_index in range(len(positions))
                ]).astype(np.float32)

            for offset in range(0, count, chunk_size):
                right = min(count, offset + chunk_size)
                rows = source_rows[offset:right]
                left = read_hand("left", rows)
                right_hand = read_hand("right", rows)
                camera = _sample_repaired_transforms(
                    transforms["camera"], rows, repair, source_relative, "transforms/camera",
                ).astype(np.float32)
                image_transforms = camera_image_transforms[offset:right].astype(np.float32)
                corrected_intrinsics = np.einsum("fij,jk->fik", image_transforms, camera_intrinsic).astype(np.float32)
                left_wrist = np.concatenate((left[:, 0, :3, 3], _rot6d(left[:, 0])), axis=1)
                right_wrist = np.concatenate((right_hand[:, 0, :3, 3], _rot6d(right_hand[:, 0])), axis=1)
                local_frames = np.arange(offset, right, dtype=np.int64)
                global_indices = frame_offset + local_frames
                timestamps = local_frames.astype(np.float32) / max(0.01, fps)
                chunk_quality = np.asarray(resolved_quality[offset:right], dtype=object)
                columns = [
                    _fixed_float_list(left, 21 * 4 * 4),
                    _fixed_float_list(_confidence_values(confidences, LEFT_HAND_SOURCE_NAMES, rows), 21),
                    _fixed_float_list(right_hand, 21 * 4 * 4),
                    _fixed_float_list(_confidence_values(confidences, RIGHT_HAND_SOURCE_NAMES, rows), 21),
                    _fixed_float_list(camera, 4 * 4),
                    _fixed_float_list(corrected_intrinsics, 3 * 3),
                    _fixed_float_list(image_transforms, 3 * 3),
                    _fixed_float_list(left_wrist, 9),
                    _fixed_float_list(right_wrist, 9),
                    pa.array([str(value) for value in phase_labels[offset:right]], type=pa.string()),
                    pa.array([str(value) for value in chunk_quality], type=pa.string()),
                    pa.array(chunk_quality == "invalid", type=pa.bool_()),
                    pa.array(chunk_quality == "review", type=pa.bool_()),
                ]
                if action_payload is not None:
                    columns.extend([
                        _fixed_float_list(np.asarray(action_payload["observation_state"])[offset:right], 20),
                        _fixed_float_list(np.asarray(action_payload["action"])[offset:right], action_dim),
                        pa.array(np.asarray(action_payload["target_frame_index"])[offset:right], type=pa.int64()),
                        pa.array(np.asarray(action_payload["valid"])[offset:right], type=pa.bool_()),
                    ])
                columns.extend([
                    pa.array(source_frames[offset:right], type=pa.int64()),
                    pa.array(np.rint(rows).astype(np.int64), type=pa.int64()),
                    pa.array(rows, type=pa.float64()),
                    pa.array(source_video_positions[offset:right], type=pa.float64()),
                    pa.array(source_frames[offset:right].astype(np.float64) / max(0.01, fps), type=pa.float64()),
                    pa.array(timestamps, type=pa.float32()),
                    pa.array(local_frames, type=pa.int64()),
                    pa.array(np.full(right - offset, episode_index, dtype=np.int64), type=pa.int64()),
                    pa.array(global_indices, type=pa.int64()),
                    pa.array(np.full(right - offset, task_index, dtype=np.int64), type=pa.int64()),
                ])
                data_writer.write_table(pa.Table.from_arrays(columns, schema=main_schema))
                if body_writer is not None:
                    body = np.stack([
                        _sample_repaired_transforms(
                            transforms[name], rows, repair, source_relative, f"transforms/{name}",
                        )
                        for name in body_names
                    ], axis=1).astype(np.float32)
                    body_writer.write_table(pa.Table.from_arrays([
                        _fixed_float_list(body, len(body_names) * 4 * 4),
                        _fixed_float_list(_confidence_values(confidences, body_names, rows), len(body_names)),
                        pa.array(timestamps, type=pa.float32()),
                        pa.array(local_frames, type=pa.int64()),
                        pa.array(np.full(right - offset, episode_index, dtype=np.int64), type=pa.int64()),
                        pa.array(global_indices, type=pa.int64()),
                    ], schema=_body_schema(len(body_names))))
        data_writer.close()
        data_writer = None
        if body_writer is not None:
            body_writer.close()
            body_writer = None
        data_temporary.replace(data_path)
        if body_names:
            body_temporary.replace(body_path)
        failed = False
    finally:
        if data_writer is not None:
            data_writer.close()
        if body_writer is not None:
            body_writer.close()
        if failed:
            data_path.unlink(missing_ok=True)
            body_path.unlink(missing_ok=True)
        data_temporary.unlink(missing_ok=True)
        body_temporary.unlink(missing_ok=True)


def write_lerobot_pair(
    output_root: Path,
    staged_video: Path,
    video_info: dict,
    source_path: Path,
    source_relative: str,
    source_count: int,
    source_rows: np.ndarray,
    source_frames: np.ndarray,
    phase_labels: np.ndarray,
    manifest: dict,
    episode: dict,
    classification: dict,
    repair: dict | None = None,
    source_video_positions: np.ndarray | None = None,
    camera_image_transforms: np.ndarray | None = None,
    quality_states: np.ndarray | None = None,
    action_payload: dict | None = None,
) -> dict:
    fps = max(0.01, float(video_info["fps"]))
    width, height = int(video_info["width"]), int(video_info["height"])
    body_names = discover_body_joint_names(source_path, source_count)
    camera_intrinsic = _source_camera_intrinsic(source_path, width, height)
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
        raise ValueError("LeRobot retiming/geometry arrays do not match the exported video frames")
    resolved_quality = (
        np.asarray(quality_states, dtype=object).reshape(-1)
        if quality_states is not None
        else np.full(count, "valid", dtype=object)
    )
    if resolved_quality.shape != (count,) or not set(str(value) for value in resolved_quality).issubset({"valid", "review", "invalid"}):
        raise ValueError("LeRobot quality states must be valid/review/invalid and match the exported frames")
    action_metadata = {
        key: action_payload.get(key)
        for key in (
            "action_profile_id", "action_dim", "action_fields", "action_representation",
            "action_coordinate_frame", "action_horizon_frames", "observation_state_fields",
        )
    } if action_payload is not None else None
    category = str(classification.get("category") or "other")
    with _LOCK:
        state = _read_state(output_root)
        _validate_dataset_contract(state, fps, width, height, body_names, camera_intrinsic, action_metadata)
        tasks = {str(key): int(value) for key, value in (state.get("tasks") or {}).items()}
        if category not in tasks:
            tasks[category] = max(tasks.values(), default=-1) + 1
        task_index = tasks[category]
        episode_index = int(state.get("next_episode_index") or 0)
        frame_offset = int(state.get("next_frame_index") or 0)
        data_path, video_path, body_path = _episode_paths(output_root, episode_index)
        if any(path.exists() for path in (data_path, video_path, body_path)):
            raise RuntimeError(f"LeRobot Episode {episode_index} already exists")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _write_episode_parquet(
                data_path,
                body_path,
                source_path,
                source_rows,
                source_frames,
                resolved_video_positions,
                phase_labels,
                body_names,
                fps,
                episode_index,
                frame_offset,
                task_index,
                source_relative,
                repair,
                camera_intrinsic,
                resolved_image_transforms,
                resolved_quality,
                action_payload,
            )
            staged_video.replace(video_path)
            if parquet.ParquetFile(data_path).metadata.num_rows != count:
                raise RuntimeError("LeRobot data Parquet frame count mismatch")
            if body_names and parquet.ParquetFile(body_path).metadata.num_rows != count:
                raise RuntimeError("LeRobot body Parquet frame count mismatch")
            state.update({
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "next_episode_index": episode_index + 1,
                "next_frame_index": frame_offset + count,
                "tasks": tasks,
                "fps": fps,
                "width": width,
                "height": height,
                "body_joint_names": list(body_names),
                "camera_intrinsic": camera_intrinsic.tolist(),
                **(action_metadata or {}),
            })
            _write_json_atomic(output_root / "meta" / "alice_state.json", state)
        except Exception:
            data_path.unlink(missing_ok=True)
            video_path.unlink(missing_ok=True)
            body_path.unlink(missing_ok=True)
            raise
    return {
        "id": f"episode_{episode_index:06d}",
        "output_format": "lerobot",
        "episode_index": episode_index,
        "task_index": task_index,
        "episode_id": str(episode["id"]),
        **classification,
        "frame_count": count,
        "fps": fps,
        "width": width,
        "height": height,
        "video_encoder": video_info.get("encoder"),
        "video_encoder_gpu": video_info.get("encoder_gpu"),
        "s1_repair_applied": s1_repair_cell_count(repair, source_relative, "transforms/") > 0,
        "s1_repair_cell_count": s1_repair_cell_count(repair, source_relative, "transforms/"),
        "data": str(data_path),
        "body": str(body_path) if body_names else None,
        "mp4": str(video_path),
        "body_joint_names": list(body_names),
        "camera_intrinsic": camera_intrinsic.tolist(),
        "camera_intrinsic_mode": "per_frame_eis_corrected" if not np.allclose(resolved_image_transforms, np.eye(3), atol=1e-9) else "static",
        "source_hdf5": source_relative,
        "quality_frame_count": count,
        "bad_frame_count": int(np.count_nonzero(resolved_quality == "invalid")),
        "review_frame_count": int(np.count_nonzero(resolved_quality == "review")),
        "action_available": action_payload is not None,
        "action_valid_frame_count": int(np.asarray(action_payload.get("valid"), dtype=bool).sum()) if action_payload is not None else 0,
        **(action_metadata or {}),
    }


def _feature_info(
    width: int,
    height: int,
    fps: float,
    body_names: list[str],
    action_metadata: dict | None = None,
) -> tuple[dict, dict]:
    features = {
        "observation.left_hand.transforms": {
            "dtype": "float32", "shape": [21, 4, 4], "names": list(HAND_21_JOINT_NAMES),
        },
        "observation.left_hand.confidence": {
            "dtype": "float32", "shape": [21], "names": list(HAND_21_JOINT_NAMES),
        },
        "observation.right_hand.transforms": {
            "dtype": "float32", "shape": [21, 4, 4], "names": list(HAND_21_JOINT_NAMES),
        },
        "observation.right_hand.confidence": {
            "dtype": "float32", "shape": [21], "names": list(HAND_21_JOINT_NAMES),
        },
        "observation.camera.transform": {"dtype": "float32", "shape": [4, 4], "names": None},
        "observation.camera.intrinsic": {"dtype": "float32", "shape": [3, 3], "names": None},
        "observation.camera.image_transform": {"dtype": "float32", "shape": [3, 3], "names": None},
        "observation.left_wrist.xyz_rot6d": {"dtype": "float32", "shape": [9], "names": ["x", "y", "z", *[f"rot6d_{index}" for index in range(6)]]},
        "observation.right_wrist.xyz_rot6d": {"dtype": "float32", "shape": [9], "names": ["x", "y", "z", *[f"rot6d_{index}" for index in range(6)]]},
        "annotation.phase_label": {"dtype": "string", "shape": [1], "names": None},
        "quality.state": {"dtype": "string", "shape": [1], "names": None},
        "quality.is_bad": {"dtype": "bool", "shape": [1], "names": None},
        "quality.needs_review": {"dtype": "bool", "shape": [1], "names": None},
        "source.frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "source.hdf5_row": {"dtype": "int64", "shape": [1], "names": None},
        "source.hdf5_position": {"dtype": "float64", "shape": [1], "names": None},
        "source.video_frame_position": {"dtype": "float64", "shape": [1], "names": None},
        "source.timestamp": {"dtype": "float64", "shape": [1], "names": None},
        LEROBOT_VIDEO_KEY: {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": height,
                "video.width": width,
                "video.channels": 3,
                "video.fps": fps,
                "video.is_depth_map": False,
                "has_audio": False,
            },
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    if action_metadata is not None:
        features.update({
            "observation.state": {
                "dtype": "float32",
                "shape": [20],
                "names": list(action_metadata.get("observation_state_fields") or []),
            },
            "action": {
                "dtype": "float32",
                "shape": [int(action_metadata.get("action_dim") or 0)],
                "names": list(action_metadata.get("action_fields") or []),
            },
            "action.target_source_frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "quality.action_valid": {"dtype": "bool", "shape": [1], "names": None},
        })
    body_features = {
        "observation.body.transforms": {
            "dtype": "float32", "shape": [len(body_names), 4, 4], "names": body_names,
        },
        "observation.body.confidence": {
            "dtype": "float32", "shape": [len(body_names)], "names": body_names,
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
    } if body_names else {}
    return features, body_features


def write_lerobot_metadata(output_root: Path, manifest: dict, pairs: list[dict]) -> Path:
    with _LOCK:
        episodes = sorted(
            (pair for pair in pairs if pair.get("output_format") == "lerobot"),
            key=lambda item: int(item.get("episode_index") or 0),
        )
        state = _read_state(output_root)
        tasks = sorted(
            ((str(task), int(index)) for task, index in (state.get("tasks") or {}).items()),
            key=lambda item: item[1],
        )
        task_table = pa.table({
            "task_index": pa.array([index for _, index in tasks], type=pa.int64()),
            "task": pa.array([task for task, _ in tasks], type=pa.string()),
        })
        _write_parquet_atomic(output_root / "meta" / "tasks.parquet", task_table)
        by_chunk: dict[int, list[dict]] = {}
        for item in episodes:
            by_chunk.setdefault(int(item.get("episode_index") or 0) // LEROBOT_CHUNK_SIZE, []).append(item)
        for chunk, items in by_chunk.items():
            _write_parquet_atomic(
                output_root / "meta" / "episodes" / f"chunk-{chunk:03d}" / "file-000.parquet",
                pa.table({
                    "episode_index": pa.array([int(item["episode_index"]) for item in items], type=pa.int64()),
                    "tasks": pa.array([[str(item.get("category") or "other")] for item in items], type=pa.list_(pa.string())),
                    "length": pa.array([int(item.get("frame_count") or 0) for item in items], type=pa.int64()),
                }),
            )
        fps = float(state.get("fps") or (episodes[0].get("fps") if episodes else 30.0) or 30.0)
        width = int(state.get("width") or (episodes[0].get("width") if episodes else 0) or 0)
        height = int(state.get("height") or (episodes[0].get("height") if episodes else 0) or 0)
        body_names = list(state.get("body_joint_names") or [])
        action_metadata = {
            key: state.get(key)
            for key in (
                "action_profile_id", "action_dim", "action_fields", "action_representation",
                "action_coordinate_frame", "action_horizon_frames", "observation_state_fields",
            )
        } if state.get("action_dim") is not None else None
        features, body_features = _feature_info(width, height, fps, body_names, action_metadata)
        info_path = output_root / "meta" / "info.json"
        _write_json_atomic(info_path, {
            "codebase_version": LEROBOT_CODEBASE_VERSION,
            "robot_type": "egodex_bimanual_hands_body",
            "fps": fps,
            "total_episodes": len(episodes),
            "total_frames": sum(int(item.get("frame_count") or 0) for item in episodes),
            "total_tasks": len(tasks),
            "total_videos": len(episodes),
            "total_chunks": max(1, math.ceil(len(episodes) / LEROBOT_CHUNK_SIZE)),
            "chunks_size": LEROBOT_CHUNK_SIZE,
            "splits": {"train": f"0:{len(episodes)}"},
            "data_path": LEROBOT_DATA_PATH,
            "video_path": LEROBOT_VIDEO_PATH,
            "features": features,
            "body_path": LEROBOT_BODY_PATH if body_names else None,
            "body_features": body_features,
            "body_joint_names": body_names,
            "hand_joint_names": list(HAND_21_JOINT_NAMES),
            "hand_joint_count_per_side": 21,
            "hand_geometry_schema": "mano21_kinematic_retarget",
            "hand_geometry_revision": EGODEX_MANO_REVISION,
            "hand_geometry_policy": "EgoDex full palm rigid fit and relative joint-rotation FK; applied MediaPipe/AlicePose snapshots are read as already-retargeted MANO21",
            "camera_intrinsic": state.get("camera_intrinsic") or scaled_egodex_camera_intrinsic(width, height).tolist(),
            "camera_intrinsic_policy": "Base source intrinsic is metadata; observation.camera.intrinsic stores the per-frame EIS-corrected matrix.",
            "quality_policy": "quality.state/is_bad/needs_review are stored per frame; quality.action_valid marks rows whose future target remains usable.",
            "action_policy": (
                {
                    "profile_id": action_metadata.get("action_profile_id"),
                    "representation": action_metadata.get("action_representation"),
                    "coordinate_frame": action_metadata.get("action_coordinate_frame"),
                    "horizon_frames": action_metadata.get("action_horizon_frames"),
                    "fields": action_metadata.get("action_fields"),
                    "observation_state_fields": action_metadata.get("observation_state_fields"),
                }
                if action_metadata is not None else None
            ),
            "source_dataset_id": manifest.get("id"),
        })
        _write_json_atomic(output_root / "meta" / "stats.json", {})
        return info_path
