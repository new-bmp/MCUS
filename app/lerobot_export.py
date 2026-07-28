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


LEROBOT_CODEBASE_VERSION = "v2.1"
LEROBOT_CHUNK_SIZE = 1_000
LEROBOT_VIDEO_KEY = "observation.images.main"
LEROBOT_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
LEROBOT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/observation.images.main/episode_{episode_index:06d}.mp4"
LEROBOT_BODY_PATH = "body/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"

HAND_21_JOINT_NAMES = (
    "Hand",
    "ThumbKnuckle",
    "ThumbIntermediateBase",
    "ThumbIntermediateTip",
    "ThumbTip",
    "IndexFingerKnuckle",
    "IndexFingerIntermediateBase",
    "IndexFingerIntermediateTip",
    "IndexFingerTip",
    "MiddleFingerKnuckle",
    "MiddleFingerIntermediateBase",
    "MiddleFingerIntermediateTip",
    "MiddleFingerTip",
    "RingFingerKnuckle",
    "RingFingerIntermediateBase",
    "RingFingerIntermediateTip",
    "RingFingerTip",
    "LittleFingerKnuckle",
    "LittleFingerIntermediateBase",
    "LittleFingerIntermediateTip",
    "LittleFingerTip",
)

_LOCK = threading.RLock()


def side_hand_joint_names(side: str) -> tuple[str, ...]:
    return tuple(f"{side}{name}" for name in HAND_21_JOINT_NAMES)


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


def _rot6d(transform: np.ndarray) -> np.ndarray:
    rotation = np.asarray(transform[:, :3, :3], dtype=np.float32)
    return rotation[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)


def _fixed_float_list(values: np.ndarray, width: int) -> pa.FixedSizeListArray:
    contiguous = np.ascontiguousarray(values, dtype=np.float32).reshape(-1, width)
    return pa.FixedSizeListArray.from_arrays(pa.array(contiguous.reshape(-1), type=pa.float32()), width)


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
    }


def _validate_dataset_contract(state: dict, fps: float, width: int, height: int, body_names: tuple[str, ...]) -> None:
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


def _episode_paths(output_root: Path, episode_index: int) -> tuple[Path, Path, Path]:
    chunk = episode_index // LEROBOT_CHUNK_SIZE
    data = output_root / LEROBOT_DATA_PATH.format(episode_chunk=chunk, episode_index=episode_index)
    video = output_root / LEROBOT_VIDEO_PATH.format(episode_chunk=chunk, episode_index=episode_index)
    body = output_root / LEROBOT_BODY_PATH.format(episode_chunk=chunk, episode_index=episode_index)
    return data, video, body


def _main_schema() -> pa.Schema:
    return pa.schema([
        pa.field("observation.left_hand.transforms", pa.list_(pa.float32(), 21 * 4 * 4)),
        pa.field("observation.left_hand.confidence", pa.list_(pa.float32(), 21)),
        pa.field("observation.right_hand.transforms", pa.list_(pa.float32(), 21 * 4 * 4)),
        pa.field("observation.right_hand.confidence", pa.list_(pa.float32(), 21)),
        pa.field("observation.camera.transform", pa.list_(pa.float32(), 4 * 4)),
        pa.field("observation.left_wrist.xyz_rot6d", pa.list_(pa.float32(), 9)),
        pa.field("observation.right_wrist.xyz_rot6d", pa.list_(pa.float32(), 9)),
        pa.field("annotation.phase_label", pa.string()),
        pa.field("source.frame_index", pa.int64()),
        pa.field("source.hdf5_row", pa.int64()),
        pa.field("source.timestamp", pa.float64()),
        pa.field("timestamp", pa.float32()),
        pa.field("frame_index", pa.int64()),
        pa.field("episode_index", pa.int64()),
        pa.field("index", pa.int64()),
        pa.field("task_index", pa.int64()),
    ])


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
    rows: np.ndarray,
) -> np.ndarray:
    values = np.full((rows.size, len(names)), np.nan, dtype=np.float32)
    if isinstance(confidences, h5py.Group):
        for index, name in enumerate(names):
            if name in confidences:
                values[:, index] = _take_rows(confidences[name], rows).reshape(-1).astype(np.float32)
    return values


def _write_episode_parquet(
    data_path: Path,
    body_path: Path,
    source_path: Path,
    source_rows: np.ndarray,
    source_frames: np.ndarray,
    phase_labels: np.ndarray,
    body_names: tuple[str, ...],
    fps: float,
    episode_index: int,
    frame_offset: int,
    task_index: int,
) -> None:
    count = int(source_frames.size)
    chunk_size = max(1, min(256, count))
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if body_names:
        body_path.parent.mkdir(parents=True, exist_ok=True)
    data_temporary = data_path.with_name(f".{data_path.name}.{uuid.uuid4().hex}.tmp")
    body_temporary = body_path.with_name(f".{body_path.name}.{uuid.uuid4().hex}.tmp")
    data_writer = parquet.ParquetWriter(data_temporary, _main_schema(), compression="zstd")
    body_writer = parquet.ParquetWriter(body_temporary, _body_schema(len(body_names)), compression="zstd") if body_names else None
    failed = True
    try:
        with h5py.File(source_path, "r") as source:
            transforms = source["transforms"]
            confidences = source.get("confidences")
            for offset in range(0, count, chunk_size):
                right = min(count, offset + chunk_size)
                rows = source_rows[offset:right]
                left = np.stack([_take_rows(transforms[name], rows) for name in LEFT_HAND_SOURCE_NAMES], axis=1).astype(np.float32)
                right_hand = np.stack([_take_rows(transforms[name], rows) for name in RIGHT_HAND_SOURCE_NAMES], axis=1).astype(np.float32)
                camera = _take_rows(transforms["camera"], rows).astype(np.float32)
                left_wrist = np.concatenate((left[:, 0, :3, 3], _rot6d(left[:, 0])), axis=1)
                right_wrist = np.concatenate((right_hand[:, 0, :3, 3], _rot6d(right_hand[:, 0])), axis=1)
                local_frames = np.arange(offset, right, dtype=np.int64)
                global_indices = frame_offset + local_frames
                timestamps = local_frames.astype(np.float32) / max(0.01, fps)
                data_writer.write_table(pa.Table.from_arrays([
                    _fixed_float_list(left, 21 * 4 * 4),
                    _fixed_float_list(_confidence_values(confidences, LEFT_HAND_SOURCE_NAMES, rows), 21),
                    _fixed_float_list(right_hand, 21 * 4 * 4),
                    _fixed_float_list(_confidence_values(confidences, RIGHT_HAND_SOURCE_NAMES, rows), 21),
                    _fixed_float_list(camera, 4 * 4),
                    _fixed_float_list(left_wrist, 9),
                    _fixed_float_list(right_wrist, 9),
                    pa.array([str(value) for value in phase_labels[offset:right]], type=pa.string()),
                    pa.array(source_frames[offset:right], type=pa.int64()),
                    pa.array(rows, type=pa.int64()),
                    pa.array(source_frames[offset:right].astype(np.float64) / max(0.01, fps), type=pa.float64()),
                    pa.array(timestamps, type=pa.float32()),
                    pa.array(local_frames, type=pa.int64()),
                    pa.array(np.full(right - offset, episode_index, dtype=np.int64), type=pa.int64()),
                    pa.array(global_indices, type=pa.int64()),
                    pa.array(np.full(right - offset, task_index, dtype=np.int64), type=pa.int64()),
                ], schema=_main_schema()))
                if body_writer is not None:
                    body = np.stack([_take_rows(transforms[name], rows) for name in body_names], axis=1).astype(np.float32)
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
) -> dict:
    fps = max(0.01, float(video_info["fps"]))
    width, height = int(video_info["width"]), int(video_info["height"])
    body_names = discover_body_joint_names(source_path, source_count)
    count = int(source_frames.size)
    category = str(classification.get("category") or "other")
    with _LOCK:
        state = _read_state(output_root)
        _validate_dataset_contract(state, fps, width, height, body_names)
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
                phase_labels,
                body_names,
                fps,
                episode_index,
                frame_offset,
                task_index,
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
        "data": str(data_path),
        "body": str(body_path) if body_names else None,
        "mp4": str(video_path),
        "body_joint_names": list(body_names),
        "source_hdf5": source_relative,
    }


def _feature_info(width: int, height: int, fps: float, body_names: list[str]) -> tuple[dict, dict]:
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
        "observation.left_wrist.xyz_rot6d": {"dtype": "float32", "shape": [9], "names": ["x", "y", "z", *[f"rot6d_{index}" for index in range(6)]]},
        "observation.right_wrist.xyz_rot6d": {"dtype": "float32", "shape": [9], "names": ["x", "y", "z", *[f"rot6d_{index}" for index in range(6)]]},
        "annotation.phase_label": {"dtype": "string", "shape": [1], "names": None},
        "source.frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "source.hdf5_row": {"dtype": "int64", "shape": [1], "names": None},
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
        features, body_features = _feature_info(width, height, fps, body_names)
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
            "source_dataset_id": manifest.get("id"),
        })
        _write_json_atomic(output_root / "meta" / "stats.json", {})
        return info_path
