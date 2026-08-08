from __future__ import annotations

"""Project dataset joint streams onto the currently reviewed video frame."""

from pathlib import Path
from functools import lru_cache
import json
import threading
from typing import Any

import cv2
import numpy as np

from .egodex_mano import (
    direct_mano21_transforms,
    egodex_mano_source_names,
    fit_egodex_mano_template,
    has_egodex_mano_source,
    required_egodex_mano_names,
    retarget_egodex_mano_frame,
    source_is_retargeted,
)
from .file_preview import preview_file_frame
from .lerobot_export import HAND_21_JOINT_NAMES, scaled_egodex_camera_intrinsic
from .mano21 import MANO21_LAYOUT_VERSION, mano21_local_index, select_mano21_points, side_hand_joint_names
from .pose_recovery import load_recovered_points
from .projection_correction import review_projection_source
from .sensor_alignment import map_video_frame_to_sensor
from .storage import change_is_applied


_JOINT_HINTS = (
    "joint", "transform", "skeleton", "hand", "finger", "wrist", "forearm",
    "shoulder", "elbow", "arm", "pose", "keypoint",
)
_COLORS = {
    "left": (235, 128, 64),
    "right": (72, 180, 235),
    "shared": (93, 205, 116),
    "unknown": (224, 224, 224),
}
_JOINT_POINT_RADIUS = 3
_CACHE_LOCK = threading.RLock()
_CANDIDATE_CACHE: dict[tuple, list[dict]] = {}
_STATUS_CACHE: dict[tuple, dict] = {}


def _generic_labels(count: int, source_path: str = "") -> list[str]:
    side = "left" if "left" in source_path.casefold() else "right" if "right" in source_path.casefold() else ""
    if count == 20 and side:
        labels = [f"{side}Hand"]
        labels.extend(f"{side}Thumb{i}" for i in range(1, 4))
        for finger in ("Index", "Middle", "Ring", "Little"):
            labels.extend(f"{side}{finger}Finger{i}" for i in range(1, 5))
        return labels
    return [f"joint_{i:02d}" for i in range(count)]


def _stream_source_id(source_id: str) -> tuple[str | None, str | None]:
    if not source_id.startswith("field::"):
        return None, None
    _, relative, field = source_id.split("::", 2)
    return relative, field


def _episode_records(manifest: dict, episode: dict) -> list[dict]:
    return [
        item for item in manifest.get("files", [])
        if item.get("episode_id") == episode.get("id")
    ]


def _is_lerobot_data_path(root: Path, relative_path: str) -> bool:
    relative = Path(relative_path)
    return (
        relative.suffix.casefold() == ".parquet"
        and bool(relative.parts)
        and relative.parts[0].casefold() == "data"
        and (root / "meta" / "info.json").is_file()
    )


def _candidate_sources(manifest: dict, episode: dict) -> list[dict]:
    understanding = (manifest.get("schema_profile") or {}).get("understanding") or {}
    current_path = Path(str(manifest.get("sidecar_path") or "")) / "changes" / "current.alice"
    applied_revision = current_path.stat().st_mtime_ns if current_path.is_file() else 0
    review = review_projection_source(manifest, episode)
    review_metadata_path = Path(str((review or {}).get("metadata_path") or ""))
    review_revision = review_metadata_path.stat().st_mtime_ns if review_metadata_path.is_file() else 0
    cache_key = (
        manifest.get("id"), episode.get("id"), len(manifest.get("files", [])),
        len(understanding.get("streams", [])), applied_revision, review_revision,
    )
    with _CACHE_LOCK:
        cached = _CANDIDATE_CACHE.get(cache_key)
    if cached is not None:
        return [dict(item) for item in cached]
    root = Path(manifest["root_path"]).resolve()
    records = {item.get("relative_path"): item for item in _episode_records(manifest, episode)}
    candidates: list[dict] = []
    if review is not None and review.get("source_relative_path"):
        candidates.append({
            "path": str(review["source_relative_path"]),
            "absolute_path": str(review["path"]),
            "field": None,
            "side": "unknown",
            "modality": "transform",
            "role": "corrected_hand_pose",
            "source": "projection_correction",
            "application_id": review.get("application_id"),
            "review_status": review.get("review_status") or "pending",
            "applied": bool(review.get("applied")),
            "frame_count": int(review.get("frame_count") or 0),
            "source_frame_positions": list((((review.get("metadata") or {}).get("retiming") or {}).get("source_frame_positions") or [])),
        })
    for stream in understanding.get("streams", []):
        if stream.get("kind") != "joint":
            continue
        relative = stream.get("source_path")
        field = stream.get("field")
        if not relative:
            relative, field = _stream_source_id(str(stream.get("source_id") or ""))
        record = records.get(relative)
        if record and relative:
            candidates.append({
                "path": relative,
                "field": field,
                "side": stream.get("side") or "unknown",
                "modality": stream.get("modality") or "",
                "role": stream.get("role") or "",
                "source": "qwen",
            })

    # EgoDex/ARKit-style recordings keep one transform dataset per joint.
    for record in records.values():
        suffix = str(record.get("extension") or "").lower()
        relative = str(record.get("relative_path") or "")
        lowered = relative.casefold()
        if suffix in {".h5", ".hdf5", ".h5df"}:
            candidates.append({"path": relative, "field": None, "side": "unknown", "modality": "", "role": "", "source": "hdf5"})
        elif _is_lerobot_data_path(root, relative):
            candidates.append({"path": relative, "field": None, "side": "unknown", "modality": "transform", "role": "", "source": "lerobot"})
        elif suffix in {".parquet", ".json", ".jsonl", ".npy", ".npz"} and any(token in lowered for token in _JOINT_HINTS):
            candidates.append({"path": relative, "field": None, "side": "unknown", "modality": "", "role": "", "source": "structured"})
    # The applied correction deliberately keeps the original relative path for
    # auditability while reading from an immutable sidecar snapshot.  Preserve
    # that candidate alongside the raw source even though path/field match.
    seen: set[tuple[str, str, str | None]] = set()
    unique: list[dict] = []
    for item in candidates:
        source_group = "projection_correction" if item.get("source") == "projection_correction" else "raw"
        key = (source_group, str(item["path"]), item.get("field"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    # A transform file contains the camera calibration and full skeleton topology;
    # prefer it over a flattened joint vector when both are present in one episode.
    source_priority = {"projection_correction": -1, "hdf5": 0, "lerobot": 0, "qwen": 1, "structured": 2}
    result = sorted(unique, key=lambda item: source_priority.get(str(item.get("source") or ""), 3))
    with _CACHE_LOCK:
        _CANDIDATE_CACHE[cache_key] = [dict(item) for item in result]
    return result


def _normalize_overlay_mode(mode: str) -> str:
    normalized = str(mode or "auto").strip().casefold()
    if normalized not in {"auto", "raw", "corrected"}:
        raise ValueError(f"Unsupported joint overlay mode: {mode}")
    return normalized


def _overlay_candidates(manifest: dict, episode: dict, mode: str) -> list[dict]:
    normalized = _normalize_overlay_mode(mode)
    candidates = _candidate_sources(manifest, episode)
    if normalized == "raw":
        return [item for item in candidates if item.get("source") != "projection_correction"]
    if normalized == "corrected":
        return [item for item in candidates if item.get("source") == "projection_correction"]
    return candidates


def _candidate_path(root: Path, candidate: dict) -> Path:
    absolute = str(candidate.get("absolute_path") or "").strip()
    return Path(absolute).expanduser().resolve() if absolute else (root / candidate["path"]).resolve()


@lru_cache(maxsize=256)
def _cached_h5_dataset_names(path_string: str, modified_ns: int, size_bytes: int) -> tuple[str, ...]:
    import h5py

    names: list[str] = []
    with h5py.File(path_string, "r") as handle:
        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset) and obj.ndim and obj.shape[0] > 0:
                names.append(name)
        handle.visititems(visitor)
    return tuple(names)


def _h5_dataset_names(path: Path) -> list[str]:
    stat = path.stat()
    return list(_cached_h5_dataset_names(str(path), stat.st_mtime_ns, stat.st_size))


@lru_cache(maxsize=32)
def _cached_h5_mano_templates(
    path_string: str,
    modified_ns: int,
    size_bytes: int,
) -> tuple[Any, Any]:
    del modified_ns, size_bytes
    import h5py

    with h5py.File(path_string, "r") as handle:
        transforms = handle.get("transforms")
        if not isinstance(transforms, h5py.Group):
            return None, None
        if source_is_retargeted(handle):
            return None, None
        return tuple(
            fit_egodex_mano_template(transforms, side) if has_egodex_mano_source(transforms, side) else None
            for side in ("left", "right")
        )


@lru_cache(maxsize=1024)
def _cached_h5_mano_frame(
    path_string: str,
    modified_ns: int,
    size_bytes: int,
    index: int,
) -> tuple[tuple[str, ...], np.ndarray | None]:
    import h5py

    templates = _cached_h5_mano_templates(path_string, modified_ns, size_bytes)
    labels: list[str] = []
    hands: list[np.ndarray] = []
    with h5py.File(path_string, "r") as handle:
        transforms = handle.get("transforms")
        if not isinstance(transforms, h5py.Group):
            return tuple(), None
        already_retargeted = source_is_retargeted(handle)
        for side_index, side in enumerate(("left", "right")):
            names = side_hand_joint_names(side) if already_retargeted else egodex_mano_source_names(transforms, side)
            if not all(name in transforms for name in names):
                continue
            frame_count = min(int(transforms[name].shape[0]) for name in names)
            if frame_count <= 0:
                continue
            row = min(max(0, int(index)), frame_count - 1)
            named = {name: np.asarray(transforms[name][row], dtype=np.float64) for name in names}
            matrices = (
                direct_mano21_transforms(named, side)
                if already_retargeted
                else retarget_egodex_mano_frame(named, templates[side_index])
            )
            labels.extend(side_hand_joint_names(side))
            hands.append(matrices[:, :3, 3])
    return (tuple(labels), np.concatenate(hands, axis=0)) if hands else (tuple(), None)


def _explicit_h5_joint_order(handle: Any, labels: list[str]) -> list[str] | None:
    transforms = handle.get("/transforms")
    if transforms is None or "joint_order" not in transforms.attrs:
        return None
    raw = transforms.attrs["joint_order"]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)):
        return None
    order = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in raw]
    if len(order) != len(labels) or len(set(order)) != len(order) or set(order) != set(labels):
        return None
    return order


@lru_cache(maxsize=64)
def _cached_h5_transform_points(path_string: str, modified_ns: int, size_bytes: int) -> tuple[tuple[str, ...], np.ndarray | None]:
    import h5py

    path = Path(path_string)
    names = _h5_dataset_names(path)
    records: list[tuple[str, np.ndarray]] = []
    with h5py.File(path_string, "r") as handle:
        for name in names:
            lowered = name.casefold()
            if "camera" in lowered or not any(token in lowered for token in _JOINT_HINTS):
                continue
            dataset = handle[name]
            if dataset.ndim < 3 or dataset.shape[-2:] != (4, 4):
                continue
            value = np.asarray(dataset[..., :3, 3], dtype=np.float64)
            if value.ndim == 2:
                value = value[:, None, :]
                item_labels = [name.rsplit("/", 1)[-1]]
            elif value.ndim == 3:
                stem = name.rsplit("/", 1)[-1]
                item_labels = [f"{stem}_{index:02d}" for index in range(value.shape[1])]
            else:
                continue
            records.extend(zip(item_labels, np.moveaxis(value, 1, 0)))
        labels = [label for label, _ in records]
        explicit_order = _explicit_h5_joint_order(handle, labels)
        if explicit_order:
            by_label = {label: value for label, value in records}
            records = [(label, by_label[label]) for label in explicit_order]
    labels = [label for label, _ in records]
    trajectories = [value[:, None, :] for _, value in records]
    if len(labels) < 2 or not trajectories:
        return tuple(), None
    frame_count = min(item.shape[0] for item in trajectories)
    return tuple(labels), np.concatenate([item[:frame_count] for item in trajectories], axis=1)


@lru_cache(maxsize=64)
def _cached_h5_projection_calibration(
    path_string: str,
    modified_ns: int,
    size_bytes: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load camera transforms and intrinsics once for low-latency playback."""
    del modified_ns, size_bytes
    import h5py

    cameras = None
    intrinsics = None
    with h5py.File(path_string, "r") as handle:
        if "/transforms/camera" in handle:
            value = np.asarray(handle["/transforms/camera"][()], dtype=np.float64)
            if value.ndim >= 3 and value.shape[-2:] == (4, 4):
                cameras = value
        for key in ("/camera/intrinsic", "/camera/intrinsics", "/intrinsic", "/intrinsics"):
            if key not in handle:
                continue
            value = np.asarray(handle[key][()], dtype=np.float64)
            if value.shape == (3, 3) or (value.ndim == 3 and value.shape[-2:] == (3, 3)):
                intrinsics = value
                break
    return cameras, intrinsics


def _h5_points(path: Path, index: int) -> tuple[np.ndarray, list[str], np.ndarray | None, np.ndarray | None, str]:
    import h5py

    stat = path.stat()
    cached_cameras, cached_intrinsics = _cached_h5_projection_calibration(str(path), stat.st_mtime_ns, stat.st_size)
    camera_ext = None
    intrinsic = None
    if cached_cameras is not None and len(cached_cameras):
        camera_ext = cached_cameras[min(index, len(cached_cameras) - 1)]
    if cached_intrinsics is not None:
        intrinsic = cached_intrinsics[min(index, len(cached_intrinsics) - 1)] if cached_intrinsics.ndim == 3 else cached_intrinsics
    dataset_leaf_names = {name.rsplit("/", 1)[-1] for name in _h5_dataset_names(path)}
    has_complete_egodex_hand = any(
        all(name in dataset_leaf_names for name in required_egodex_mano_names(side))
        for side in ("left", "right")
    )
    if has_complete_egodex_hand:
        mano_labels, mano_points = _cached_h5_mano_frame(
            str(path), stat.st_mtime_ns, stat.st_size, max(0, int(index)),
        )
        if mano_points is not None and len(mano_points):
            return mano_points, list(mano_labels), camera_ext, intrinsic, "world"
    cached_labels, cached_points = _cached_h5_transform_points(str(path), stat.st_mtime_ns, stat.st_size)
    if cached_points is not None and len(cached_points):
        frame_index = min(index, len(cached_points) - 1)
        return cached_points[frame_index], list(cached_labels), camera_ext, intrinsic, "world"

    with h5py.File(path, "r") as handle:
        transform_items: list[tuple[str, np.ndarray]] = []
        generic_names = sorted(
            _h5_dataset_names(path),
            key=lambda item: (0 if "skeleton" in item.casefold() else 1 if "keypoint" in item.casefold() else 2, item),
        )
        for name in generic_names:
            lowered = name.casefold()
            if "camera" in lowered or not any(token in lowered for token in _JOINT_HINTS):
                continue
            dataset = handle[name]
            if dataset.shape[0] <= index:
                continue
            value = np.asarray(dataset[index], dtype=np.float64)
            if value.shape == (4, 4):
                transform_items.append((name.rsplit("/", 1)[-1], value[:3, 3]))
            elif value.ndim >= 2 and value.shape[-2:] == (4, 4):
                transform_items.extend((f"{name.rsplit('/', 1)[-1]}_{i:02d}", item[:3, 3]) for i, item in enumerate(value))

        if len(transform_items) >= 2:
            labels = [item[0] for item in transform_items]
            return np.asarray([item[1] for item in transform_items]), labels, camera_ext, intrinsic, "world"

        # Generic keypoint/skeleton arrays, including mocap (N, J, 3/7) files.
        generic_names = sorted(
            _h5_dataset_names(path),
            key=lambda item: (0 if "skeleton" in item.casefold() else 1 if "keypoint" in item.casefold() else 2, item),
        )
        for name in generic_names:
            lowered = name.casefold()
            if not any(token in lowered for token in ("skeleton", "keypoint", "joint")):
                continue
            dataset = handle[name]
            if dataset.shape[0] <= index:
                continue
            value = np.asarray(dataset[index], dtype=np.float64)
            points = _coerce_points(value)
            if points is not None and len(points) and np.isfinite(points).all() and np.linalg.norm(points, axis=1).max(initial=0.0) > 1e-6:
                labels = _generic_labels(len(points), str(path))
                return points, labels, camera_ext, intrinsic, "world" if value.shape[-1] >= 3 else "pixel"
    return np.empty((0, 3)), [], camera_ext, intrinsic, "unknown"


@lru_cache(maxsize=64)
def _cached_lerobot_metadata(
    info_path_string: str,
    modified_ns: int,
    size_bytes: int,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[float, ...]]:
    del modified_ns, size_bytes
    payload = json.loads(Path(info_path_string).read_text(encoding="utf-8"))
    features = payload.get("features") if isinstance(payload.get("features"), dict) else {}
    hand_feature = features.get("observation.left_hand.transforms") if isinstance(features, dict) else {}
    hand_names = tuple(
        str(name)
        for name in (
            (hand_feature or {}).get("names")
            or payload.get("hand_joint_names")
            or HAND_21_JOINT_NAMES
        )
    )
    body_names = tuple(str(name) for name in (payload.get("body_joint_names") or []))
    intrinsic = None
    try:
        candidate = np.asarray(payload.get("camera_intrinsic"), dtype=np.float64)
        if candidate.shape == (3, 3) and np.isfinite(candidate).all():
            intrinsic = candidate
    except (TypeError, ValueError):
        intrinsic = None
    if intrinsic is None and "egodex" in str(payload.get("robot_type") or "").casefold():
        video = features.get("observation.images.main") if isinstance(features, dict) else {}
        shape = (video or {}).get("shape") or []
        height = int(shape[0]) if len(shape) >= 2 else 1080
        width = int(shape[1]) if len(shape) >= 2 else 1920
        intrinsic = scaled_egodex_camera_intrinsic(width, height).astype(np.float64)
    return hand_names, body_names, tuple(intrinsic.reshape(-1).tolist()) if intrinsic is not None else ()


def _lerobot_metadata(root: Path) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray | None]:
    info_path = root / "meta" / "info.json"
    stat = info_path.stat()
    hand_names, body_names, intrinsic_values = _cached_lerobot_metadata(
        str(info_path), stat.st_mtime_ns, stat.st_size,
    )
    intrinsic = np.asarray(intrinsic_values, dtype=np.float64).reshape(3, 3) if intrinsic_values else None
    return hand_names, body_names, intrinsic


def _fixed_list_values(table: Any, field: str) -> np.ndarray:
    column = table[field].combine_chunks()
    values = np.asarray(column.values, dtype=np.float64)
    return values.reshape(len(column), -1)


@lru_cache(maxsize=48)
def _cached_lerobot_episode(
    data_path_string: str,
    data_modified_ns: int,
    data_size_bytes: int,
    body_path_string: str,
    body_modified_ns: int,
    body_size_bytes: int,
    hand_names: tuple[str, ...],
    body_names: tuple[str, ...],
    fallback_intrinsic: tuple[float, ...],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray | None, np.ndarray | None]:
    del data_modified_ns, data_size_bytes, body_modified_ns, body_size_bytes
    import pyarrow.parquet as parquet

    data_path = Path(data_path_string)
    schema_names = set(parquet.ParquetFile(data_path).schema_arrow.names)
    required = {
        "observation.left_hand.transforms",
        "observation.right_hand.transforms",
        "observation.camera.transform",
    }
    if not required.issubset(schema_names):
        return tuple(), np.empty((0, 3)), None, None
    columns = sorted(required | ({"observation.camera.intrinsic"} & schema_names))
    table = parquet.read_table(data_path, columns=columns)
    left_values = _fixed_list_values(table, "observation.left_hand.transforms")
    right_values = _fixed_list_values(table, "observation.right_hand.transforms")
    if left_values.shape[1] % 16 or right_values.shape[1] % 16:
        return tuple(), np.empty((0, 3)), None, None
    left = left_values.reshape(len(left_values), -1, 4, 4)
    right = right_values.reshape(len(right_values), -1, 4, 4)
    frame_count = min(len(left), len(right))
    if frame_count <= 0:
        return tuple(), np.empty((0, 3)), None, None
    left_names = hand_names[:left.shape[1]] or tuple(f"joint_{index:02d}" for index in range(left.shape[1]))
    right_names = hand_names[:right.shape[1]] or tuple(f"joint_{index:02d}" for index in range(right.shape[1]))
    if len(left_names) != left.shape[1]:
        left_names = tuple(f"joint_{index:02d}" for index in range(left.shape[1]))
    if len(right_names) != right.shape[1]:
        right_names = tuple(f"joint_{index:02d}" for index in range(right.shape[1]))
    labels = tuple(f"left{name}" for name in left_names) + tuple(f"right{name}" for name in right_names)
    points = np.concatenate((left[:frame_count, :, :3, 3], right[:frame_count, :, :3, 3]), axis=1)

    body_path = Path(body_path_string) if body_path_string else None
    if body_path is not None and body_path.is_file() and body_names:
        body_schema = set(parquet.ParquetFile(body_path).schema_arrow.names)
        if "observation.body.transforms" in body_schema:
            body_table = parquet.read_table(body_path, columns=["observation.body.transforms"])
            body_values = _fixed_list_values(body_table, "observation.body.transforms")
            if body_values.shape[1] % 16 == 0:
                body = body_values.reshape(len(body_values), -1, 4, 4)
                if len(body) == frame_count and body.shape[1] == len(body_names):
                    points = np.concatenate((points, body[:, :, :3, 3]), axis=1)
                    labels += body_names

    camera_values = _fixed_list_values(table, "observation.camera.transform")
    camera = camera_values.reshape(len(camera_values), 4, 4)[:frame_count] if camera_values.shape[1] == 16 else None
    intrinsic = None
    if "observation.camera.intrinsic" in schema_names:
        intrinsic_values = _fixed_list_values(table, "observation.camera.intrinsic")
        if intrinsic_values.shape[1] == 9:
            intrinsic = intrinsic_values.reshape(len(intrinsic_values), 3, 3)[:frame_count]
    elif fallback_intrinsic:
        single = np.asarray(fallback_intrinsic, dtype=np.float64).reshape(3, 3)
        intrinsic = np.repeat(single[None, ...], frame_count, axis=0)
    return labels, points, camera, intrinsic


def _lerobot_points(
    root: Path,
    data_path: Path,
    index: int,
) -> tuple[np.ndarray, list[str], np.ndarray | None, np.ndarray | None, str]:
    hand_names, body_names, fallback_intrinsic = _lerobot_metadata(root)
    relative = data_path.relative_to(root)
    body_relative = Path("body", *relative.parts[1:]) if relative.parts and relative.parts[0].casefold() == "data" else Path()
    body_path = root / body_relative if body_relative.parts else None
    data_stat = data_path.stat()
    body_stat = body_path.stat() if body_path is not None and body_path.is_file() else None
    labels, trajectories, cameras, intrinsics = _cached_lerobot_episode(
        str(data_path),
        data_stat.st_mtime_ns,
        data_stat.st_size,
        str(body_path) if body_stat is not None else "",
        body_stat.st_mtime_ns if body_stat is not None else 0,
        body_stat.st_size if body_stat is not None else 0,
        hand_names,
        body_names,
        tuple(fallback_intrinsic.reshape(-1).tolist()) if fallback_intrinsic is not None else (),
    )
    if not len(trajectories):
        return np.empty((0, 3)), [], None, fallback_intrinsic, "unknown"
    frame_index = min(max(0, int(index)), len(trajectories) - 1)
    camera = cameras[frame_index] if cameras is not None and len(cameras) else None
    intrinsic = intrinsics[frame_index] if intrinsics is not None and len(intrinsics) else fallback_intrinsic
    return trajectories[frame_index], list(labels), camera, intrinsic, "world"


def _coerce_points(value: Any) -> np.ndarray | None:
    if isinstance(value, dict):
        for key in ("points", "keypoints", "joints", "skeleton", "positions", "xyz"):
            if key in value:
                return _coerce_points(value[key])
        return None
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.size == 0 or array.ndim == 0:
        return None
    if array.ndim >= 2 and array.shape[-2:] == (4, 4):
        return array.reshape(-1, 4, 4)[:, :3, 3]
    if array.ndim == 1:
        if array.size % 3 == 0:
            return array.reshape(-1, 3)
        if array.size % 2 == 0:
            return array.reshape(-1, 2)
        return None
    last = array.shape[-1]
    if last in (2, 3):
        return array.reshape(-1, last)
    if last >= 7:
        return array[..., :3].reshape(-1, 3)
    return None


def _structured_points(path: Path, field: str | None, index: int) -> tuple[np.ndarray, list[str], np.ndarray | None, np.ndarray | None, str]:
    payload = preview_file_frame(path, path.name, index=index, field=field)
    if payload.get("mode") == "error":
        return np.empty((0, 3)), [], None, None, "unknown"
    value = payload.get("value")
    points = _coerce_points(value)
    if points is None:
        return np.empty((0, 3)), [], None, None, "unknown"
    labels = [f"joint_{i:02d}" for i in range(len(points))]
    return points, labels, None, None, "pixel" if points.shape[1] == 2 else "world"


def _project(points: np.ndarray, coordinate: str, camera_ext: np.ndarray | None, intrinsic: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    height, width = shape[:2]
    if points.shape[1] == 2:
        result = points[:, :2].astype(np.float64)
        if np.nanmax(np.abs(result), initial=0.0) <= 1.5:
            result[:, 0] *= width
            result[:, 1] *= height
        return result
    xyz = points[:, :3].astype(np.float64)
    if intrinsic is not None:
        if camera_ext is not None and camera_ext.shape == (4, 4):
            hom = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1)
            xyz = (np.linalg.inv(camera_ext) @ hom.T).T[:, :3]
        projected, _ = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), intrinsic, np.zeros(5))
        result = projected.reshape(-1, 2)
        result[~np.isfinite(xyz).all(axis=1) | (xyz[:, 2] <= 1e-6)] = np.nan
        return result
    if coordinate == "world" and np.nanmax(np.abs(xyz), initial=0.0) <= 2.0:
        return np.column_stack([(xyz[:, 0] + 1.0) * width / 2.0, (1.0 - xyz[:, 1]) * height / 2.0])
    return xyz[:, :2]


def _side(label: str) -> str:
    lowered = label.casefold()
    if "left" in lowered or lowered.startswith("l_"):
        return "left"
    if "right" in lowered or lowered.startswith("r_"):
        return "right"
    return "unknown"


def _acyclic_edges(node_count: int, edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Keep only valid edges that do not create a closed joint topology."""
    parents = list(range(max(0, node_count)))
    ranks = [0] * len(parents)

    def find(node: int) -> int:
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    output: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for start, end in edges:
        if start == end or start < 0 or end < 0 or start >= node_count or end >= node_count:
            continue
        canonical = (min(start, end), max(start, end))
        if canonical in seen:
            continue
        seen.add(canonical)
        start_root, end_root = find(start), find(end)
        if start_root == end_root:
            continue
        if ranks[start_root] < ranks[end_root]:
            start_root, end_root = end_root, start_root
        parents[end_root] = start_root
        if ranks[start_root] == ranks[end_root]:
            ranks[start_root] += 1
        output.append((start, end))
    return output


_EGODEX_FINGER_SEGMENTS = {
    "little": (
        "LittleFingerMetacarpal",
        "LittleFingerKnuckle",
        "LittleFingerIntermediateBase",
        "LittleFingerIntermediateTip",
        "LittleFingerTip",
    ),
    "ring": (
        "RingFingerMetacarpal",
        "RingFingerKnuckle",
        "RingFingerIntermediateBase",
        "RingFingerIntermediateTip",
        "RingFingerTip",
    ),
    "middle": (
        "MiddleFingerMetacarpal",
        "MiddleFingerKnuckle",
        "MiddleFingerIntermediateBase",
        "MiddleFingerIntermediateTip",
        "MiddleFingerTip",
    ),
    "index": (
        "IndexFingerMetacarpal",
        "IndexFingerKnuckle",
        "IndexFingerIntermediateBase",
        "IndexFingerIntermediateTip",
        "IndexFingerTip",
    ),
    "thumb": (
        "ThumbKnuckle",
        "ThumbIntermediateBase",
        "ThumbIntermediateTip",
        "ThumbTip",
    ),
}


def _joint_label_key(label: str) -> str:
    """Normalize a field name without discarding its joint semantics."""
    leaf = str(label).replace("\\", "/").rsplit("/", 1)[-1]
    return "".join(character for character in leaf.casefold() if character.isalnum())


def _edges(labels: list[str]) -> list[tuple[int, int]]:
    # Only connect unambiguous, semantically named joints. Dataset order is not
    # topology: HDF5 traversal is alphabetical and would connect (for example)
    # IntermediateTip back to Knuckle, producing the fan/crossing pattern.
    buckets: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        buckets.setdefault(_joint_label_key(label), []).append(index)
    lookup = {key: indexes[0] for key, indexes in buckets.items() if len(indexes) == 1}
    edges: list[tuple[int, int]] = []

    def add_chain(names: tuple[str, ...] | list[str]) -> None:
        keys = [_joint_label_key(name) for name in names]
        for parent, child in zip(keys, keys[1:]):
            if parent in lookup and child in lookup:
                edges.append((lookup[parent], lookup[child]))

    for side in ("left", "right"):
        add_chain((f"{side}Shoulder", f"{side}Arm", f"{side}Forearm", f"{side}Hand"))
        for segments in _EGODEX_FINGER_SEGMENTS.values():
            add_chain((f"{side}Hand", *(f"{side}{segment}" for segment in segments)))
            if segments[0].endswith("Metacarpal"):
                # Alice LeRobot hands intentionally omit metacarpals. This
                # alternate chain attaches Knuckle directly to Hand there;
                # the acyclic filter discards it when a metacarpal exists.
                add_chain((f"{side}Hand", *(f"{side}{segment}" for segment in segments[1:])))

        # Generic 20-point hand arrays receive semantic labels in
        # _generic_labels. Their numeric suffixes describe the chain order.
        add_chain((f"{side}Hand", *(f"{side}Thumb{i}" for i in range(1, 4))))
        for finger in ("Index", "Middle", "Ring", "Little"):
            add_chain((f"{side}Hand", *(f"{side}{finger}Finger{i}" for i in range(1, 5))))

    return _acyclic_edges(len(labels), edges)


def _semantic_point_indices(labels: list[str], edges: list[tuple[int, int]] | None = None) -> set[int] | None:
    """Return points belonging to a named topology, or ``None`` for unknown data.

    A non-empty semantic topology is authoritative: points such as EgoDex's
    hip/spine/neck transforms are not part of the 2D hand overlay and should
    not appear as isolated dots. ``None`` deliberately keeps generic arrays
    intact when no safe field-name topology can be inferred.
    """
    resolved_edges = _edges(labels) if edges is None else edges
    if not resolved_edges:
        return None
    return {point for edge in resolved_edges for point in edge}


def joint_overlay_status(manifest: dict, episode: dict, mode: str = "auto") -> dict:
    overlay_mode = _normalize_overlay_mode(mode)
    root = Path(manifest["root_path"]).resolve()
    has_joint_state = False
    missing_initial_position = False
    records_dir = Path(str(manifest.get("sidecar_path") or "")) / "changes" / "records"
    change_revision = records_dir.stat().st_mtime_ns if records_dir.is_dir() else 0
    current_path = Path(str(manifest.get("sidecar_path") or "")) / "changes" / "current.alice"
    applied_revision = current_path.stat().st_mtime_ns if current_path.is_file() else 0
    review = review_projection_source(manifest, episode) if overlay_mode in {"auto", "corrected"} else None
    review_metadata_path = Path(str((review or {}).get("metadata_path") or ""))
    review_revision = review_metadata_path.stat().st_mtime_ns if review_metadata_path.is_file() else 0
    cache_key = (manifest.get("id"), episode.get("id"), overlay_mode, change_revision, applied_revision, review_revision)
    with _CACHE_LOCK:
        cached = _STATUS_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    recovery_applied = change_is_applied(manifest["id"], "pose_recovery", episode["id"])
    candidates = _overlay_candidates(manifest, episode, overlay_mode)
    for candidate in candidates:
        path = _candidate_path(root, candidate)
        if not path.is_file():
            continue
        try:
            if candidate.get("modality") == "state" or candidate.get("role") in {"proprioception", "joint_state"} or "observation.state" in str(candidate.get("field") or "").lower():
                has_joint_state = True
                continue
            if candidate.get("source") == "lerobot":
                points, labels, _, _, coordinate = _lerobot_points(root, path, 0)
                if not len(points):
                    missing_initial_position = True
            elif path.suffix.lower() in {".h5", ".hdf5", ".h5df"}:
                points, labels, _, _, coordinate = _h5_points(path, 0)
                if not len(points) and recovery_applied and candidate.get("source") != "projection_correction":
                    recovered = load_recovered_points(manifest["id"], episode["id"], candidate["path"], 0)
                    if recovered is not None:
                        points, labels = recovered
                        coordinate = "world"
                if not len(points):
                    missing_initial_position = True
            else:
                points, labels, _, _, coordinate = _structured_points(path, candidate.get("field"), 0)
                if not len(points):
                    missing_initial_position = True
            if len(points):
                points, labels, _, mano_sides = select_mano21_points(points, labels)
                result = {
                    "available": True,
                    "initial_position_available": True,
                    "source_path": candidate["path"],
                    "source_kind": candidate.get("source") or "unknown",
                    "field": candidate.get("field"),
                    "joint_count": len(points),
                    "joint_count_per_hand": 21 if mano_sides else None,
                    "hand_sides": list(mano_sides),
                    "skeleton_schema": "mano21" if mano_sides else "source",
                    "layout_version": MANO21_LAYOUT_VERSION if mano_sides else None,
                    "coordinate_system": coordinate,
                    "overlay_mode": overlay_mode,
                    "frame_count": int(candidate.get("frame_count") or episode.get("frame_count") or 0),
                    "source_frame_positions": list(candidate.get("source_frame_positions") or []),
                    "projection_review_status": candidate.get("review_status"),
                    "projection_applied": bool(candidate.get("applied")),
                }
                with _CACHE_LOCK:
                    _STATUS_CACHE[cache_key] = dict(result)
                return result
        except (OSError, ValueError, RuntimeError):
            continue
    if missing_initial_position:
        result = {
            "available": False,
            "initial_position_available": False,
            "joint_count": 0,
            "joint_state_available": has_joint_state,
            "overlay_mode": overlay_mode,
            "reason": "归正结果缺少有效起始位置" if overlay_mode == "corrected" else "第 0 帧缺少有效起始位置，Joint 叠加已拒绝启动",
        }
        with _CACHE_LOCK:
            _STATUS_CACHE[cache_key] = dict(result)
        return result
    if overlay_mode == "corrected":
        reason = "尚未生成手部归正结果" if not candidates else "手部归正结果不可读取或不可投影"
    else:
        reason = "已检测到关节状态，但缺少机器人运动学模型和相机标定，无法投影到视频" if has_joint_state else "未找到可投影的原始 joint/transform 数据"
    result = {
        "available": False,
        "initial_position_available": False,
        "joint_count": 0,
        "joint_state_available": has_joint_state,
        "overlay_mode": overlay_mode,
        "reason": reason,
    }
    with _CACHE_LOCK:
        _STATUS_CACHE[cache_key] = dict(result)
    return result


def joint_overlay_geometry(
    manifest: dict,
    episode: dict,
    index: int,
    width: int,
    height: int,
    media: dict | None = None,
    mode: str = "auto",
) -> dict:
    overlay_mode = _normalize_overlay_mode(mode)
    status = joint_overlay_status(manifest, episode, overlay_mode)
    if not status.get("available"):
        raise ValueError(status.get("reason") or "Joint overlay requires a valid initial position")
    root = Path(manifest["root_path"]).resolve()
    recovery_applied: bool | None = None
    for candidate in _overlay_candidates(manifest, episode, overlay_mode):
        if candidate.get("path") != status.get("source_path"):
            continue
        if status.get("source_kind") and candidate.get("source") != status.get("source_kind"):
            continue
        path = _candidate_path(root, candidate)
        if not path.is_file():
            continue
        try:
            media_fps = float((media or {}).get("fps") or episode.get("fps") or 30.0)
            try:
                if candidate.get("source") == "projection_correction":
                    # Corrected snapshots are deliberately stored one row per
                    # video frame, but their presentation cadence must still
                    # follow the original sensor clock.
                    try:
                        _, source_alignment = map_video_frame_to_sensor(
                            manifest,
                            episode,
                            candidate["path"],
                            index,
                            media_fps,
                            reference_media_file_id=str((media or {}).get("file_id") or "") or None,
                        )
                    except KeyError:
                        source_alignment = {}
                    sensor_index, alignment = index, {
                        "video_frame": index,
                        "sensor_index": index,
                        "valid": True,
                        "mode": "applied_projection_video_aligned",
                        "alignment_multiplier": 1.0,
                        "sensor_hz": source_alignment.get("sensor_hz"),
                        "physical_hz": source_alignment.get("physical_hz"),
                    }
                else:
                    sensor_index, alignment = map_video_frame_to_sensor(
                        manifest,
                        episode,
                        candidate["path"],
                        index,
                        media_fps,
                        reference_media_file_id=str((media or {}).get("file_id") or "") or None,
                    )
            except KeyError:
                sensor_index, alignment = index, {
                    "video_frame": index,
                    "sensor_index": index,
                    "valid": True,
                    "mode": "unindexed_identity",
                    "alignment_multiplier": 1.0,
                    "sensor_hz": None,
                    "physical_hz": None,
                }
            clock_hz = float(alignment.get("physical_hz") or alignment.get("sensor_hz") or media_fps)
            if sensor_index is None:
                return {
                    "source_path": candidate["path"],
                    "source_kind": candidate.get("source") or "unknown",
                    "overlay_mode": overlay_mode,
                    "field": candidate.get("field"),
                    "joint_count": 0,
                    "coordinate_system": "unknown",
                    "frame_index": index,
                    "sensor_index": None,
                    "alignment_valid": False,
                    "alignment_mode": alignment.get("mode"),
                    "alignment_multiplier": alignment.get("alignment_multiplier"),
                    "sensor_hz": alignment.get("sensor_hz"),
                    "physical_hz": alignment.get("physical_hz"),
                    "clock_hz": clock_hz,
                    "width": width,
                    "height": height,
                    "points": [],
                    "edges": [],
                }
            if candidate.get("modality") == "state" or candidate.get("role") in {"proprioception", "joint_state"} or "observation.state" in str(candidate.get("field") or "").lower():
                continue
            if candidate.get("source") == "lerobot":
                points, labels, camera_ext, intrinsic, coordinate = _lerobot_points(root, path, sensor_index)
            elif path.suffix.lower() in {".h5", ".hdf5", ".h5df"}:
                points, labels, camera_ext, intrinsic, coordinate = _h5_points(path, sensor_index)
                if not len(points) and recovery_applied is None:
                    recovery_applied = change_is_applied(manifest["id"], "pose_recovery", episode["id"])
                if not len(points) and recovery_applied and candidate.get("source") != "projection_correction":
                    recovered = load_recovered_points(manifest["id"], episode["id"], candidate["path"], index)
                    if recovered is not None:
                        points, labels = recovered
                        coordinate = "world"
            else:
                points, labels, camera_ext, intrinsic, coordinate = _structured_points(path, candidate.get("field"), sensor_index)
            if not len(points):
                continue
            points, labels, source_indices, mano_sides = select_mano21_points(points, labels)
            projected = _project(points, coordinate, camera_ext, intrinsic, (height, width, 3))
            valid = np.isfinite(projected).all(axis=1)
            semantic_edges = _edges(labels)
            semantic_point_indices = _semantic_point_indices(labels, semantic_edges)
            point_records: list[dict] = []
            index_map: dict[int, int] = {}
            for point_index, point in enumerate(projected):
                if semantic_point_indices is not None and point_index not in semantic_point_indices:
                    continue
                if not valid[point_index]:
                    continue
                x, y = point.astype(float)
                if x < -20 or y < -20 or x >= width + 20 or y >= height + 20:
                    continue
                index_map[point_index] = len(point_records)
                local_index = mano21_local_index(labels[point_index]) if mano_sides else None
                point_records.append({
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "side": _side(labels[point_index]) if labels else "unknown",
                    "source_index": local_index if local_index is not None else int(source_indices[point_index]),
                    "joint_name": labels[point_index] if labels else None,
                })
            edge_records: list[list[int]] = []
            for start, end in semantic_edges:
                if start not in index_map or end not in index_map:
                    continue
                edge_records.append([index_map[start], index_map[end]])
            return {
                "source_path": candidate["path"],
                "source_kind": candidate.get("source") or "unknown",
                "overlay_mode": overlay_mode,
                "field": candidate.get("field"),
                "joint_count": len(point_records),
                "joint_count_per_hand": 21 if mano_sides else None,
                "hand_sides": list(mano_sides),
                "skeleton_schema": "mano21" if mano_sides else "source",
                "layout_version": MANO21_LAYOUT_VERSION if mano_sides else None,
                "coordinate_system": coordinate,
                "frame_index": index,
                "sensor_index": sensor_index,
                "alignment_valid": bool(alignment.get("valid", True)),
                "alignment_mode": alignment.get("mode"),
                "alignment_multiplier": alignment.get("alignment_multiplier"),
                "sensor_hz": alignment.get("sensor_hz"),
                "physical_hz": alignment.get("physical_hz"),
                "clock_hz": clock_hz,
                "width": width,
                "height": height,
                "points": point_records,
                "edges": edge_records,
            }
        except (OSError, ValueError, RuntimeError, IndexError, np.linalg.LinAlgError):
            continue
    raise ValueError("当前 Episode 没有可读取或可投影的 joint/transform 数据")


def draw_joint_overlay(
    frame: np.ndarray,
    manifest: dict,
    episode: dict,
    index: int,
    media: dict | None = None,
    show_indices: bool = False,
    mode: str = "auto",
) -> tuple[np.ndarray, dict]:
    geometry = joint_overlay_geometry(manifest, episode, index, frame.shape[1], frame.shape[0], media, mode)
    output = frame.copy()
    points = geometry["points"]
    for start, end in geometry["edges"]:
        a, b = points[start], points[end]
        color = _COLORS.get(a.get("side", "unknown"), _COLORS["unknown"])
        cv2.line(output, (round(a["x"]), round(a["y"])), (round(b["x"]), round(b["y"])), color, 2, cv2.LINE_AA)
    for point in points:
        color = _COLORS.get(point.get("side", "unknown"), _COLORS["unknown"])
        cv2.circle(output, (round(point["x"]), round(point["y"])), _JOINT_POINT_RADIUS, color, -1, cv2.LINE_AA)
    if show_indices:
        occupied_badges: list[tuple[int, int, int, int]] = []
        directions = ((0.7, -0.7), (-0.7, -0.7), (1.0, 0.0), (-1.0, 0.0), (0.0, -1.0), (0.7, 0.7), (-0.7, 0.7), (0.0, 1.0))
        for point in points:
            label = str(point.get("source_index", ""))
            if not label:
                continue
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
            badge_width, badge_height = max(13, text_width + 6), text_height + baseline + 4
            point_x, point_y = round(point["x"]), round(point["y"])
            x, top, radius = point_x + 6, point_y - badge_height - 3, 11
            for candidate_radius in (11, 20, 29, 38, 47, 56, 65):
                found = False
                for direction_x, direction_y in directions:
                    candidate_x = round(point_x + direction_x * candidate_radius - badge_width / 2)
                    candidate_top = round(point_y + direction_y * candidate_radius - badge_height / 2)
                    candidate_x = min(max(1, candidate_x), max(1, output.shape[1] - badge_width - 1))
                    candidate_top = min(max(1, candidate_top), max(1, output.shape[0] - badge_height - 1))
                    if any(candidate_x < right + 2 and candidate_x + badge_width + 2 > left and candidate_top < bottom + 2 and candidate_top + badge_height + 2 > badge_top for left, badge_top, right, bottom in occupied_badges):
                        continue
                    x, top, radius, found = candidate_x, candidate_top, candidate_radius, True
                    break
                if found:
                    break
            y = top + badge_height
            occupied_badges.append((x, top, x + badge_width, y))
            if radius > 16:
                cv2.line(output, (point_x, point_y), (x + badge_width // 2, top + badge_height // 2), (218, 224, 229), 1, cv2.LINE_AA)
            cv2.rectangle(output, (x, y - badge_height), (x + badge_width, y), (20, 24, 28), -1)
            cv2.putText(output, label, (x + 3, y - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(output, (0, 0), (output.shape[1], 30), (18, 21, 25), -1)
    cv2.putText(output, f"JOINTS {geometry['joint_count']}", (9, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return output, {key: geometry.get(key) for key in ("source_path", "field", "joint_count", "coordinate_system")}
