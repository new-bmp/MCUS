from __future__ import annotations

"""Project dataset joint streams onto the currently reviewed video frame."""

from pathlib import Path
from functools import lru_cache
import json
import threading
from typing import Any

import cv2
import numpy as np

from .file_preview import preview_file_frame
from .pose_recovery import load_recovered_points
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


def _candidate_sources(manifest: dict, episode: dict) -> list[dict]:
    understanding = (manifest.get("schema_profile") or {}).get("understanding") or {}
    cache_key = (
        manifest.get("id"), episode.get("id"), len(manifest.get("files", [])),
        len(understanding.get("streams", [])),
    )
    with _CACHE_LOCK:
        cached = _CANDIDATE_CACHE.get(cache_key)
    if cached is not None:
        return [dict(item) for item in cached]
    records = {item.get("relative_path"): item for item in _episode_records(manifest, episode)}
    candidates: list[dict] = []
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
        elif suffix in {".parquet", ".json", ".jsonl", ".npy", ".npz"} and any(token in lowered for token in _JOINT_HINTS):
            candidates.append({"path": relative, "field": None, "side": "unknown", "modality": "", "role": "", "source": "structured"})
    seen: set[tuple[str, str | None]] = set()
    unique = [item for item in candidates if not ((item["path"], item.get("field")) in seen or seen.add((item["path"], item.get("field"))))]
    # A transform file contains the camera calibration and full skeleton topology;
    # prefer it over a flattened joint vector when both are present in one episode.
    result = sorted(unique, key=lambda item: 0 if item.get("source") == "hdf5" else 1)
    with _CACHE_LOCK:
        _CANDIDATE_CACHE[cache_key] = [dict(item) for item in result]
    return result


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


def _h5_points(path: Path, index: int) -> tuple[np.ndarray, list[str], np.ndarray | None, np.ndarray | None, str]:
    import h5py

    with h5py.File(path, "r") as handle:
        camera_ext = None
        intrinsic = None
        if "/transforms/camera" in handle:
            camera = handle["/transforms/camera"]
            if camera.ndim >= 3:
                camera_ext = np.asarray(camera[min(index, camera.shape[0] - 1)], dtype=np.float64)
        for key in ("/camera/intrinsic", "/camera/intrinsics", "/intrinsic", "/intrinsics"):
            if key in handle:
                value = np.asarray(handle[key][()], dtype=np.float64)
                if value.ndim == 3:
                    value = value[min(index, value.shape[0] - 1)]
                if value.shape == (3, 3):
                    intrinsic = value
                    break

        stat = path.stat()
        cached_labels, cached_points = _cached_h5_transform_points(str(path), stat.st_mtime_ns, stat.st_size)
        if cached_points is not None and len(cached_points):
            frame_index = min(index, len(cached_points) - 1)
            return cached_points[frame_index], list(cached_labels), camera_ext, intrinsic, "world"

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
        return projected.reshape(-1, 2)
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


def joint_overlay_status(manifest: dict, episode: dict) -> dict:
    root = Path(manifest["root_path"]).resolve()
    has_joint_state = False
    missing_initial_position = False
    records_dir = Path(str(manifest.get("sidecar_path") or "")) / "changes" / "records"
    change_revision = records_dir.stat().st_mtime_ns if records_dir.is_dir() else 0
    cache_key = (manifest.get("id"), episode.get("id"), change_revision)
    with _CACHE_LOCK:
        cached = _STATUS_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    recovery_applied = change_is_applied(manifest["id"], "pose_recovery", episode["id"])
    for candidate in _candidate_sources(manifest, episode):
        path = (root / candidate["path"]).resolve()
        if not path.is_file():
            continue
        try:
            if candidate.get("modality") == "state" or candidate.get("role") in {"proprioception", "joint_state"} or "observation.state" in str(candidate.get("field") or "").lower():
                has_joint_state = True
                continue
            if path.suffix.lower() in {".h5", ".hdf5", ".h5df"}:
                points, labels, _, _, coordinate = _h5_points(path, 0)
                if not len(points) and recovery_applied:
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
                result = {"available": True, "initial_position_available": True, "source_path": candidate["path"], "field": candidate.get("field"), "joint_count": len(points), "coordinate_system": coordinate}
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
            "reason": "第 0 帧缺少有效起始位置，Joint 叠加已拒绝启动",
        }
        with _CACHE_LOCK:
            _STATUS_CACHE[cache_key] = dict(result)
        return result
    reason = "已检测到关节状态，但缺少机器人运动学模型和相机标定，无法投影到视频" if has_joint_state else "未找到可投影的 joint/transform 数据"
    result = {"available": False, "initial_position_available": False, "joint_count": 0, "joint_state_available": has_joint_state, "reason": reason}
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
) -> dict:
    status = joint_overlay_status(manifest, episode)
    if not status.get("available"):
        raise ValueError(status.get("reason") or "Joint overlay requires a valid initial position")
    root = Path(manifest["root_path"]).resolve()
    recovery_applied: bool | None = None
    for candidate in _candidate_sources(manifest, episode):
        if candidate.get("path") != status.get("source_path"):
            continue
        path = (root / candidate["path"]).resolve()
        if not path.is_file():
            continue
        try:
            try:
                sensor_index, alignment = map_video_frame_to_sensor(
                    manifest,
                    episode,
                    candidate["path"],
                    index,
                    float((media or {}).get("fps") or episode.get("fps") or 30.0),
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
            if sensor_index is None:
                return {
                    "source_path": candidate["path"],
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
                    "width": width,
                    "height": height,
                    "points": [],
                    "edges": [],
                }
            if candidate.get("modality") == "state" or candidate.get("role") in {"proprioception", "joint_state"} or "observation.state" in str(candidate.get("field") or "").lower():
                continue
            if path.suffix.lower() in {".h5", ".hdf5", ".h5df"}:
                points, labels, camera_ext, intrinsic, coordinate = _h5_points(path, sensor_index)
                if not len(points) and recovery_applied is None:
                    recovery_applied = change_is_applied(manifest["id"], "pose_recovery", episode["id"])
                if not len(points) and recovery_applied:
                    recovered = load_recovered_points(manifest["id"], episode["id"], candidate["path"], index)
                    if recovered is not None:
                        points, labels = recovered
                        coordinate = "world"
            else:
                points, labels, camera_ext, intrinsic, coordinate = _structured_points(path, candidate.get("field"), sensor_index)
            if not len(points):
                continue
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
                point_records.append({
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "side": _side(labels[point_index]) if labels else "unknown",
                    "source_index": point_index,
                })
            edge_records: list[list[int]] = []
            for start, end in semantic_edges:
                if start not in index_map or end not in index_map:
                    continue
                edge_records.append([index_map[start], index_map[end]])
            return {
                "source_path": candidate["path"],
                "field": candidate.get("field"),
                "joint_count": len(point_records),
                "coordinate_system": coordinate,
                "frame_index": index,
                "sensor_index": sensor_index,
                "alignment_valid": bool(alignment.get("valid", True)),
                "alignment_mode": alignment.get("mode"),
                "alignment_multiplier": alignment.get("alignment_multiplier"),
                "sensor_hz": alignment.get("sensor_hz"),
                "physical_hz": alignment.get("physical_hz"),
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
) -> tuple[np.ndarray, dict]:
    geometry = joint_overlay_geometry(manifest, episode, index, frame.shape[1], frame.shape[0], media)
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
