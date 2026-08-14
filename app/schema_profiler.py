from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .openxr_mano import detect_openxr_schema


STRUCTURED_EXTENSIONS = {".json", ".jsonl", ".h5", ".hdf5", ".h5df", ".parquet", ".npz", ".npy", ".csv", ".tsv"}
MAX_FILES = 300
MAX_FIELDS = 600
MAX_FOLDERS = 96
MAX_FILES_PER_FOLDER = 4
MAX_EPISODE_SAMPLES = 12

_SKELETON_GROUP_NAMES = {"transforms", "joint_transforms", "skeleton_transforms"}
_NON_JOINT_TRANSFORM_NAMES = {
    "base", "camera", "camera_link", "map", "object", "origin", "root", "scene", "sensor", "world",
}
_NON_JOINT_TRANSFORM_COMPACT_NAMES = {item.replace("_", "") for item in _NON_JOINT_TRANSFORM_NAMES}
_ANATOMICAL_JOINT_TOKENS = (
    "ankle", "arm", "chest", "elbow", "finger", "foot", "forearm", "hand", "head", "hip",
    "knee", "neck", "pelvis", "shoulder", "spine", "thumb", "torso", "wrist",
)


def _shape(value: Any) -> list[int] | None:
    if isinstance(value, (list, tuple)):
        dimensions = []
        current = value
        while isinstance(current, (list, tuple)) and len(dimensions) < 5:
            dimensions.append(len(current))
            current = current[0] if current else None
        return dimensions
    return None


def _flatten_json(value: Any, prefix: str = "", depth: int = 0, limit: int = 200) -> list[dict]:
    if depth > 6 or limit <= 0:
        return []
    fields: list[dict] = []
    if isinstance(value, dict):
        for key, child in list(value.items())[:80]:
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, (dict, list)):
                shape = _shape(child)
                fields.append({"key": path, "dtype": type(child).__name__, "shape": shape})
                fields.extend(_flatten_json(child[0] if isinstance(child, list) and child else child, path, depth + 1, limit - len(fields)))
            else:
                fields.append({"key": path, "dtype": type(child).__name__, "shape": None})
            if len(fields) >= limit:
                break
    elif isinstance(value, list) and value:
        fields.extend(_flatten_json(value[0], prefix or "[]", depth + 1, limit))
    return fields[:limit]


def _probe_json(path: Path) -> list[dict]:
    try:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8", errors="replace") as source:
                for line in source:
                    if line.strip():
                        return _flatten_json(json.loads(line))
            return []
        if path.stat().st_size > 12 * 1024 * 1024:
            return [{"key": "$", "dtype": "json", "shape": None, "note": "file_too_large_for_safe_preview"}]
        return _flatten_json(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        return [{"key": "$", "dtype": "unreadable_json", "shape": None, "error": str(exc)[:160]}]


def _probe_hdf5(path: Path) -> list[dict]:
    try:
        import h5py

        fields: list[dict] = []
        with h5py.File(path, "r") as handle:
            def visitor(name, obj):
                if len(fields) >= 250:
                    return
                if isinstance(obj, h5py.Dataset):
                    fields.append({"key": name, "dtype": str(obj.dtype), "shape": list(obj.shape), "chunks": list(obj.chunks) if obj.chunks else None})
            handle.visititems(visitor)
        return fields
    except Exception as exc:
        return [{"key": "$", "dtype": "unreadable_hdf5", "shape": None, "error": str(exc)[:160]}]


def _probe_parquet(path: Path) -> list[dict]:
    try:
        import pyarrow.parquet as parquet

        file = parquet.ParquetFile(path)
        return [{"key": field.name, "dtype": str(field.type), "shape": [file.metadata.num_rows]} for field in file.schema_arrow]
    except Exception as exc:
        return [{"key": "$", "dtype": "unreadable_parquet", "shape": None, "error": str(exc)[:160]}]


def _probe_numpy(path: Path) -> list[dict]:
    try:
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                return [{"key": key, "dtype": str(archive[key].dtype), "shape": list(archive[key].shape)} for key in archive.files[:200]]
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return [{"key": "$", "dtype": str(array.dtype), "shape": list(array.shape)}]
    except Exception as exc:
        return [{"key": "$", "dtype": "unreadable_numpy", "shape": None, "error": str(exc)[:160]}]


def _probe_table(path: Path) -> list[dict]:
    try:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with path.open("r", encoding="utf-8", errors="replace", newline="") as source:
            reader = csv.reader(source, delimiter=delimiter)
            header = next(reader, [])
        return [{"key": str(column), "dtype": "column", "shape": None} for column in header[:250]]
    except Exception as exc:
        return [{"key": "$", "dtype": "unreadable_table", "shape": None, "error": str(exc)[:160]}]


def _probe_file(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return _probe_json(path)
    if suffix in {".h5", ".hdf5", ".h5df"}:
        return _probe_hdf5(path)
    if suffix == ".parquet":
        return _probe_parquet(path)
    if suffix in {".npz", ".npy"}:
        return _probe_numpy(path)
    if suffix in {".csv", ".tsv"}:
        return _probe_table(path)
    return []


def _side(text: str) -> str:
    lowered = text.lower().replace("-", "_")
    if any(token in lowered for token in ("left", "l_hand", "l_arm", "camera_l", "cam_l")):
        return "left"
    if any(token in lowered for token in ("right", "r_hand", "r_arm", "camera_r", "cam_r")):
        return "right"
    if any(token in lowered for token in ("bimanual", "both", "dual_arm", "stereo")):
        return "shared"
    return "unknown"


def _kind(text: str) -> tuple[str, str]:
    lowered = text.lower().replace("\\", "/")
    leaf = lowered.rsplit("/", 1)[-1]
    # Field semantics take precedence over a parent folder name such as
    # `camera/`.  Otherwise `camera/head_imu.h5::imu/gyro` is incorrectly
    # classified as RGB merely because the recorder groups it with cameras.
    if any(token in leaf for token in ("timestamp", "timestamps", "master_ts", "sensor_ts", "time_ns", "time_us", "time_ms", "frame_time")):
        return "timestamp", "time"
    if any(token in lowered for token in ("action", "command", "control", "target_qpos", "target_joint")):
        return "action", "command"
    if any(token in lowered for token in ("pressure", "tactile", "force", "torque", "wrench", "load_cell", "ft_sensor")):
        modality = "pressure" if "pressure" in lowered else "tactile" if "tactile" in lowered else "force_torque"
        return "sensor", modality
    if any(token in lowered for token in ("/imu", "imu/", "_imu", "accelerometer", "accel", "gyroscope", "gyro")):
        return "sensor", "imu"
    if leaf == "joints" and any(token in lowered for token in ("mocap", "dexweave", "glove", "hand_device")):
        # Nexus/DexWeave stores six uint8 device channels under `joints`; they
        # are human-hand sensor channels, not a robot State or Action vector.
        return "sensor", "hand_device"
    if any(token in lowered for token in ("skeleton", "keypoint", "mocap", "wrist_quat")):
        return "joint", "pose"
    if any(token in lowered for token in ("qpos", "joint_pos", "joint_position", "joint_angle", "joints")):
        return "joint", "position"
    if any(token in lowered for token in ("endpose", "end_pose", "end_effector_pose", "eef_pose", "tcp_pose")):
        return "joint", "pose"
    if "observation.state" in lowered or lowered.endswith(".state") or lowered.endswith("/state"):
        return "joint", "state"
    if any(token in lowered for token in ("qvel", "joint_vel", "joint_velocity")):
        return "joint", "velocity"
    if any(token in lowered for token in ("gripper", "proprio", "robot_state", "arm_state")):
        return "joint", "state"
    if any(token in lowered for token in ("rgb", "image", "camera", "vision", "video", "depth", "color")):
        modality = "depth" if "depth" in lowered else "rgb"
        return "vision", modality
    return "other", "unknown"


def _numeric_dtype(dtype: Any) -> bool:
    try:
        return np.dtype(str(dtype)).kind in "biufc"
    except (TypeError, ValueError):
        return False


def _signal_shape(shape: Any) -> list[int] | None:
    if not isinstance(shape, (list, tuple)) or not shape:
        return None
    try:
        values = [int(value) for value in shape]
    except (TypeError, ValueError):
        return None
    if values[0] < 4 or any(value <= 0 for value in values):
        return None
    return values


def _anatomical_joint_name(field: str) -> bool:
    leaf = field.replace("\\", "/").rsplit("/", 1)[-1].casefold().replace("-", "_")
    compact = leaf.replace("_", "")
    if leaf in _NON_JOINT_TRANSFORM_NAMES or compact in _NON_JOINT_TRANSFORM_COMPACT_NAMES:
        return False
    return any(token in compact for token in _ANATOMICAL_JOINT_TOKENS)


def is_skeletal_transform_field(field: str, shape: Any, dtype: Any) -> bool:
    """Return true only for time-varying anatomical 4x4 transforms.

    The group and leaf-name checks deliberately exclude camera calibration,
    object poses and image tensors.  This covers datasets such as EgoDex where
    every body/hand joint is stored as ``transforms/<jointName>``.
    """
    dimensions = _signal_shape(shape)
    normalized = str(field).replace("\\", "/").strip("/")
    parts = normalized.split("/")
    return bool(
        dimensions
        and len(dimensions) == 3
        and dimensions[-2:] == [4, 4]
        and len(parts) >= 2
        and parts[-2].casefold() in _SKELETON_GROUP_NAMES
        and _numeric_dtype(dtype)
        and _anatomical_joint_name(normalized)
    )


def _joint_priority(field: str) -> tuple[int, str]:
    leaf = field.replace("\\", "/").rsplit("/", 1)[-1].casefold().replace("_", "")
    if leaf.endswith("hand"):
        priority = 0
    elif leaf.endswith("forearm"):
        priority = 1
    elif leaf.endswith("arm"):
        priority = 2
    elif leaf.endswith("shoulder"):
        priority = 3
    elif leaf.endswith("fingertip") or leaf.endswith("thumbtip"):
        priority = 4
    elif leaf.endswith("intermediatetip"):
        priority = 5
    elif leaf.endswith("knuckle"):
        priority = 6
    elif leaf.endswith("metacarpal"):
        priority = 7
    elif leaf.endswith("intermediatebase"):
        priority = 8
    else:
        priority = 9
    return priority, leaf


def infer_local_signal_fields(fields: list[dict], max_dimensions: int | None = None) -> list[dict]:
    """Infer explicit numeric Joint/Action streams without an API call.

    Only one- or two-dimensional named numeric series are accepted directly.
    Anatomical transform matrices are handled as a group and converted to XYZ
    joint positions by the curation reader.  Image-like tensors are therefore
    never accepted merely because their containing file has a suggestive name.
    """
    descriptors: list[dict] = []
    skeleton_groups: dict[str, dict[int, list[dict]]] = {}
    skeleton_fields: set[str] = set()
    native_skeleton_fields: set[str] = set()
    openxr_fields: set[str] = set()

    # OpenXR is a semantic joint set, not merely a tensor with 26 rows.  Only
    # promote tensors carrying an explicit OpenXR/XR-hand field hint here;
    # arbitrary 26-point arrays remain generic until a recorder schema or
    # standard joint-name list proves their meaning.
    for field in fields:
        field_name = str(field.get("key") or "")
        shape = _signal_shape(field.get("shape"))
        if not shape or not _numeric_dtype(field.get("dtype")):
            continue
        detection = detect_openxr_schema(field=field_name, shape=shape)
        if not detection.get("detected"):
            continue
        source_shape = list(shape)
        side_hint = _side(field_name)
        output_shape = [int(shape[0]), 21 * 3]
        dimension_names = [
            f"mano21_{node:02d}.{axis}"
            for node in range(21)
            for axis in ("x", "y", "z")
        ]
        representation = (
            "transform" if tuple(shape[-3:]) == (26, 4, 4)
            else "pose7_xyzw" if shape[-1] == 7
            else "absolute"
        )
        descriptors.append({
            "field": field_name,
            "kind": "joint",
            "modality": "pose",
            "shape": output_shape,
            "source_shape": source_shape,
            "dtype": str(field.get("dtype") or "float"),
            "side_hint": side_hint,
            "role": "hand_skeleton",
            "representation": representation,
            "dimension_names": dimension_names,
            "members": [],
            "gripper_indices": [],
            "embodiment_id": "openxr-hand-26",
            "node_count": 26,
            "confidence": float(detection.get("confidence") or 0.96),
            "evidence": f"openxr_hand_26_{detection.get('evidence', 'schema')}",
            "extraction": "openxr_hand_26_to_mano21",
        })
        openxr_fields.add(field_name)

    for field in fields:
        field_name = str(field.get("key") or "")
        if field_name in openxr_fields:
            continue
        shape = _signal_shape(field.get("shape"))
        if is_skeletal_transform_field(field_name, shape, field.get("dtype")):
            group = field_name.replace("\\", "/").rsplit("/", 1)[0]
            skeleton_groups.setdefault(group, {}).setdefault(int(shape[0]), []).append(field)
            skeleton_fields.add(field_name)

    for group, counts in skeleton_groups.items():
        frame_count, members = max(counts.items(), key=lambda item: (len(item[1]), item[0]))
        if len(members) < 4:
            continue
        ordered = sorted((str(item["key"]) for item in members), key=_joint_priority)
        if max_dimensions is not None:
            ordered = ordered[: max(1, int(max_dimensions) // 3)]
        dimension_names = [
            f"{member.rsplit('/', 1)[-1]}.{axis}"
            for member in ordered
            for axis in ("x", "y", "z")
        ]
        rig_signature = hashlib.sha1("\n".join(sorted(str(item["key"]) for item in members)).encode("utf-8")).hexdigest()[:12]
        descriptors.append({
            "field": f"{group}/*",
            "kind": "joint",
            "modality": "position",
            "shape": [frame_count, len(dimension_names)],
            "dtype": str(members[0].get("dtype") or "float"),
            "side_hint": "shared",
            "role": "skeletal_joint_positions",
            "representation": "absolute",
            "dimension_names": dimension_names,
            "members": ordered,
            "gripper_indices": [],
            "embodiment_id": f"skeleton-{rig_signature}",
            "confidence": 0.96,
            "evidence": "local_hdf5_skeletal_transform_group",
            "extraction": "matrix_translation_xyz",
        })

    for field in fields:
        field_name = str(field.get("key") or "")
        shape = _signal_shape(field.get("shape"))
        lowered = field_name.replace("\\", "/").casefold()
        if (
            not shape
            or len(shape) != 3
            or shape[-1] < 3
            or not _numeric_dtype(field.get("dtype"))
            or not any(token in lowered for token in ("skeleton", "keypoint"))
        ):
            continue
        node_count = int(shape[1])
        retained_nodes = node_count
        if max_dimensions is not None:
            retained_nodes = min(retained_nodes, max(1, int(max_dimensions) // 3))
        dimension_names = [
            f"node_{node:02d}.{axis}"
            for node in range(retained_nodes)
            for axis in ("x", "y", "z")
        ]
        descriptors.append({
            "field": field_name,
            "kind": "joint",
            "modality": "pose",
            "shape": [int(shape[0]), len(dimension_names)],
            "source_shape": shape,
            "dtype": str(field.get("dtype") or "float"),
            "side_hint": "unknown",
            "role": "hand_skeleton",
            "representation": "absolute",
            "dimension_names": dimension_names,
            "members": [],
            "gripper_indices": [],
            "embodiment_id": f"native-hand-{node_count}",
            "node_count": node_count,
            "confidence": 0.98,
            "evidence": "local_native_skeleton_tensor",
            "extraction": "skeleton_xyz",
        })
        native_skeleton_fields.add(field_name)

    for field in fields:
        field_name = str(field.get("key") or "")
        if field_name in openxr_fields:
            continue
        if field_name in skeleton_fields:
            continue
        if field_name in native_skeleton_fields:
            continue
        if native_skeleton_fields and field_name.replace("\\", "/").rsplit("/", 1)[-1].casefold() == "wrist_quat":
            # The same orientation is already embedded per node in the native
            # skeleton tensor.  Keeping a standalone quaternion in S1 would
            # make harmless q/-q sign changes look like numeric jumps.
            continue
        shape = _signal_shape(field.get("shape"))
        if not shape or len(shape) > 2 or not _numeric_dtype(field.get("dtype")):
            continue
        leaf = field_name.replace("\\", "/").rsplit("/", 1)[-1].casefold()
        try:
            byte_device_channels = leaf == "joints" and np.dtype(str(field.get("dtype"))).kind in "iu" and np.dtype(str(field.get("dtype"))).itemsize <= 1
        except (TypeError, ValueError):
            byte_device_channels = False
        if byte_device_channels:
            # A bare byte-valued `joints` vector is ambiguous and commonly
            # represents glove/device channels.  The inventory layer can use
            # the full source path to classify it safely; do not promote it to
            # robot State here.
            continue
        kind, modality = _kind(field_name)
        if kind not in {"joint", "action"}:
            continue
        width = 1 if len(shape) == 1 else shape[1]
        if width > 4096:
            continue
        leaf = field_name.replace("\\", "/").rsplit("/", 1)[-1]
        representation = "unknown"
        lowered = field_name.casefold()
        if kind == "action":
            if any(token in lowered for token in ("delta", "increment")):
                representation = "delta"
            elif any(token in lowered for token in ("velocity", "qvel", "speed")):
                representation = "velocity"
            elif any(token in lowered for token in ("target_qpos", "target_joint", "position_command", "endpose", "end_pose", "eef_pose", "tcp_pose")):
                representation = "absolute"
        elif modality == "pose" or any(token in lowered for token in ("endpose", "end_pose", "eef_pose", "tcp_pose")):
            representation = "absolute"
        descriptors.append({
            "field": field_name,
            "kind": kind,
            "modality": modality,
            "shape": shape,
            "dtype": str(field.get("dtype") or ""),
            "side_hint": _side(field_name),
            "role": "",
            "representation": representation,
            "dimension_names": [leaf] if width == 1 else [f"{leaf}[{index}]" for index in range(width)],
            "members": [],
            "gripper_indices": [],
            "embodiment_id": None,
            "confidence": 0.82,
            "evidence": "local_numeric_field_name_and_shape",
        })
    return descriptors


def probe_local_signal_fields(path: Path, max_dimensions: int | None = None) -> list[dict]:
    return infer_local_signal_fields(_probe_file(path), max_dimensions=max_dimensions)


def _even_sample(values: list[Any], limit: int) -> list[Any]:
    if len(values) <= limit:
        return values
    if limit <= 1:
        return values[:1]
    indices = {round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)}
    return [values[index] for index in sorted(indices)]


def sample_profile_paths(root: Path, files: list[Path]) -> tuple[list[Path], dict]:
    """Select representative files across folders instead of taking a prefix."""
    folders: dict[str, list[Path]] = {}
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix().casefold()):
        relative = path.relative_to(root)
        folders.setdefault(relative.parent.as_posix(), []).append(path)

    folder_names = _even_sample(sorted(folders, key=str.casefold), MAX_FOLDERS)
    ranked: dict[str, list[Path]] = {}
    for folder in folder_names:
        paths = folders[folder]

        def priority(path: Path) -> tuple[int, str]:
            suffix = path.suffix.casefold()
            stem = path.stem.casefold()
            schema_hint = any(token in stem for token in ("schema", "metadata", "info", "episode", "manifest"))
            structured = suffix in STRUCTURED_EXTENSIONS
            return (0 if structured and schema_hint else 1 if structured else 2, path.name.casefold())

        by_extension: dict[str, list[Path]] = {}
        for path in sorted(paths, key=priority):
            by_extension.setdefault(path.suffix.casefold() or "<none>", []).append(path)
        representatives = [items[0] for _, items in sorted(by_extension.items())]
        remaining = [path for path in sorted(paths, key=priority) if path not in representatives]
        ranked[folder] = (representatives + remaining)[:MAX_FILES_PER_FOLDER]

    selected: list[Path] = []
    for offset in range(MAX_FILES_PER_FOLDER):
        for folder in folder_names:
            candidates = ranked[folder]
            if offset < len(candidates):
                selected.append(candidates[offset])
                if len(selected) >= MAX_FILES:
                    break
        if len(selected) >= MAX_FILES:
            break
    return selected, {
        "strategy": "folder_extension_stratified_v1",
        "folder_count": len(folders),
        "folders_sampled": len(folder_names),
        "files_sampled": len(selected),
        "max_files_per_folder": MAX_FILES_PER_FOLDER,
    }


def build_inventory(root: Path, episodes: list[dict], included_paths: set[str] | None = None) -> dict:
    if included_paths is None:
        files = [path for path in root.rglob("*") if path.is_file()]
    else:
        # scan_dataset already verified these paths. Avoid a second full stat pass.
        files = [root / Path(relative) for relative in sorted(included_paths, key=str.casefold)]
    extension_counts: dict[str, int] = {}
    for path in files:
        extension_counts[path.suffix.lower() or "<none>"] = extension_counts.get(path.suffix.lower() or "<none>", 0) + 1

    profiled_files, sampling = sample_profile_paths(root, files)

    entries: list[dict] = []
    streams: list[dict] = []
    stream_ids: set[str] = set()

    sampled_episodes = _even_sample(
        sorted(episodes, key=lambda item: str(item.get("relative_path") or item.get("name") or "").casefold()),
        MAX_EPISODE_SAMPLES,
    )
    for episode in sampled_episodes:
        media_streams = episode.get("media_streams") or [episode]
        for media in media_streams:
            stream_id = f"media::{media['relative_path']}"
            modality = str(media.get("modality") or ("image_sequence" if media["type"] == "images" else "video"))
            is_depth = modality == "depth" or bool(media.get("is_depth_map"))
            channels = 1 if is_depth else 3
            stream = {
                "id": stream_id,
                "source_path": media["relative_path"],
                "field": None,
                "kind": "vision",
                "modality": modality,
                "side_hint": str(media.get("side") or _side(media["relative_path"])),
                "shape": [media["frame_count"], media["height"], media["width"], channels],
                "dtype": str(media.get("dtype") or ("uint16" if is_depth else "uint8")),
                "role": str(media.get("role") or ("depth_map" if is_depth else "camera_stream")),
                "variant": str(media.get("variant") or "primary"),
                "evidence": "decoded_raw_depth" if media.get("type") == "raw_depth" else "decoded_media",
            }
            streams.append(stream)
            stream_ids.add(stream_id)

    field_count = 0
    for path in profiled_files:
        relative = path.relative_to(root).as_posix()
        try:
            size_bytes = path.stat().st_size
        except OSError:
            continue
        entry = {"path": relative, "extension": path.suffix.lower(), "size_bytes": size_bytes}
        if path.suffix.lower() in STRUCTURED_EXTENSIONS and field_count < MAX_FIELDS:
            fields = _probe_file(path)[: max(0, MAX_FIELDS - field_count)]
            entry["fields"] = fields
            field_count += len(fields)
            local_signals = infer_local_signal_fields(fields)
            local_by_field = {str(item["field"]): item for item in local_signals}
            actual_fields = {str(field.get("key", "$")) for field in fields}
            for field in fields:
                field_key = str(field.get("key", "$"))
                stream_id = f"field::{relative}::{field_key}"
                if stream_id in stream_ids:
                    continue
                local = local_by_field.get(field_key)
                kind, modality = (local["kind"], local["modality"]) if local else _kind(f"{relative}/{field_key}")
                local_side = str((local or {}).get("side_hint") or "unknown")
                streams.append({
                    "id": stream_id,
                    "source_path": relative,
                    "field": field_key,
                    "kind": kind,
                    "modality": modality,
                    "side_hint": local_side if local_side != "unknown" else _side(f"{relative}/{field_key}"),
                    "shape": field.get("shape"),
                    "dtype": field.get("dtype"),
                    "evidence": (local or {}).get("evidence") or "field_name_heuristic",
                    **({
                        key: local[key]
                        for key in (
                            "role", "representation", "dimension_names", "gripper_indices",
                            "embodiment_id", "extraction", "node_count", "source_shape", "confidence",
                        )
                        if key in local
                    } if local else {}),
                })
                stream_ids.add(stream_id)
            for local in local_signals:
                field_key = str(local["field"])
                if field_key in actual_fields:
                    continue
                stream_id = f"field::{relative}::{field_key}"
                if stream_id in stream_ids:
                    continue
                streams.append({
                    "id": stream_id,
                    "source_path": relative,
                    **{key: value for key, value in local.items() if key != "members"},
                })
                stream_ids.add(stream_id)
        entries.append(entry)

    return {
        "root_name": root.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "files_profiled": len(entries),
        "field_count": field_count,
        "extension_counts": extension_counts,
        "episodes": [{key: episode[key] for key in ("id", "name", "type", "relative_path", "fps", "frame_count", "duration", "width", "height")} for episode in sampled_episodes],
        "files": entries,
        "candidate_streams": streams,
        "sampling": sampling,
    }


def pending_profile(inventory: dict) -> dict:
    return {"status": "awaiting_vlm", "inventory": inventory, "understanding": None, "warnings": ["Qwen-VLM has not analyzed this dataset schema."], "updated_at": datetime.now(timezone.utc).isoformat()}


_SUPPORTED_UNDERSTANDING_KINDS = {"vision", "joint", "sensor", "action", "timestamp"}
_CANONICAL_LOCK_CONFIDENCE = 0.95


def _bounded_confidence(value: Any, fallback: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return max(0.0, min(1.0, float(fallback)))


def _authoritative_canonical_stream(stream: dict) -> bool:
    """Return whether recorder/local canonical classification outranks Qwen.

    Filename/shape heuristics intentionally remain editable by Qwen.  Recorder
    declarations and future high-confidence canonical adapters are treated as
    data contracts so a tactile/IMU/depth stream cannot become an Action or a
    Joint merely because a language model proposed the same-shaped category.
    """
    canonical_kind = str(stream.get("canonical_kind") or stream.get("kind") or "")
    if canonical_kind not in _SUPPORTED_UNDERSTANDING_KINDS:
        return False
    evidence = str(stream.get("canonical_evidence") or "").strip().casefold()
    confidence = _bounded_confidence(stream.get("canonical_confidence"), 0.0)
    return evidence == "recorder_metadata" or evidence.startswith("recorder_metadata:") or confidence >= _CANONICAL_LOCK_CONFIDENCE


def _canonical_value(stream: dict, key: str, fallback_key: str | None = None, default: Any = "") -> Any:
    value = stream.get(key)
    if value is None and fallback_key:
        value = stream.get(fallback_key)
    return default if value is None else value


def _validated_stream(actual: dict, proposed: dict, *, canonical_locked: bool) -> dict | None:
    canonical_kind = str(_canonical_value(actual, "canonical_kind", "kind", "other"))
    canonical_confidence = _bounded_confidence(actual.get("canonical_confidence"), 0.0)
    actual_kind = str(actual.get("kind") or "other")
    proposed_kind = proposed.get("kind")
    if canonical_locked:
        final_kind = canonical_kind
    else:
        final_kind = proposed_kind if proposed_kind in _SUPPORTED_UNDERSTANDING_KINDS else actual_kind
    if final_kind not in _SUPPORTED_UNDERSTANDING_KINDS:
        return None

    shape = actual.get("shape") if isinstance(actual.get("shape"), list) else []
    dimension_count = int(shape[-1]) if len(shape) > 1 and isinstance(shape[-1], int) else None
    proposed_dimensions = proposed.get("dimension_names") if isinstance(proposed.get("dimension_names"), list) else []
    actual_dimensions = actual.get("dimension_names") if isinstance(actual.get("dimension_names"), list) else []
    dimension_names = [
        str(item)[:120]
        for item in (proposed_dimensions or actual_dimensions)
        if str(item).strip()
    ][:MAX_FIELDS]

    proposed_grippers = proposed.get("gripper_indices") if isinstance(proposed.get("gripper_indices"), list) else None
    actual_grippers = actual.get("gripper_indices") if isinstance(actual.get("gripper_indices"), list) else []
    gripper_indices = []
    for value in proposed_grippers if proposed_grippers is not None else actual_grippers:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if index >= 0 and (dimension_count is None or index < dimension_count):
            gripper_indices.append(index)

    representation_source = (
        (actual.get("representation") or "unknown")
        if canonical_locked
        else (proposed.get("representation") or actual.get("representation") or "unknown")
    )
    representation = str(representation_source)
    if representation not in {"absolute", "delta", "velocity", "unknown"}:
        representation = "unknown"

    if canonical_locked:
        modality = str(actual.get("modality") or "unknown")
        side = str(_canonical_value(actual, "side", "side_hint", "unknown"))
        role = str(actual.get("role") or "")
        evidence = str(actual.get("canonical_evidence") or actual.get("evidence") or "canonical_mapping")
        confidence = canonical_confidence or _bounded_confidence(actual.get("confidence"), 0.95)
    else:
        modality = str(proposed.get("modality") or actual.get("modality") or "unknown")
        proposed_side = proposed.get("side")
        side = str(proposed_side if proposed_side in {"left", "right", "shared", "unknown"} else _canonical_value(actual, "side_hint", "side", "unknown"))
        role = str(proposed.get("role") or actual.get("role") or "")
        evidence = str(proposed.get("evidence") or actual.get("evidence") or "")
        confidence = _bounded_confidence(proposed.get("confidence"), 0.5)

    proposed_embodiment = str(proposed.get("embodiment_id") or "").strip()
    actual_embodiment = str(actual.get("embodiment_id") or "").strip()
    return {
        "source_id": str(actual.get("id") or proposed.get("source_id") or ""),
        "kind": final_kind,
        "modality": modality,
        "side": side if side in {"left", "right", "shared", "unknown"} else "unknown",
        "role": role,
        "representation": representation,
        "dimension_names": dimension_names,
        "gripper_indices": sorted(set(gripper_indices)),
        "embodiment_id": proposed_embodiment or actual_embodiment or None,
        "confidence": confidence,
        "evidence": evidence,
        "source_path": actual.get("source_path"),
        "field": actual.get("field"),
        "shape": actual.get("shape"),
        "dtype": actual.get("dtype"),
        "variant": str(actual.get("variant") or "primary"),
        "extraction": str(actual.get("extraction") or ""),
        "node_count": actual.get("node_count"),
        "canonical_locked": canonical_locked,
        "canonical_kind": canonical_kind,
        "canonical_confidence": canonical_confidence,
        "canonical_evidence": str(actual.get("canonical_evidence") or ""),
    }


def validate_understanding(inventory: dict, raw: dict) -> tuple[dict, list[str]]:
    candidates = {stream["id"]: stream for stream in inventory.get("candidate_streams", [])}
    warnings: list[str] = []
    streams = []
    accepted_ids: set[str] = set()
    for proposed in raw.get("streams") or []:
        if not isinstance(proposed, dict):
            warnings.append("Qwen returned a malformed stream entry and it was discarded.")
            continue
        source_id = str(proposed.get("source_id", ""))
        if source_id not in candidates:
            warnings.append(f"Qwen returned an unknown stream and it was discarded: {source_id}")
            continue
        if source_id in accepted_ids:
            warnings.append(f"Qwen returned a duplicate stream and the duplicate was discarded: {source_id}")
            continue
        actual = candidates[source_id]
        canonical_locked = _authoritative_canonical_stream(actual)
        stream = _validated_stream(actual, proposed, canonical_locked=canonical_locked)
        if stream is None:
            warnings.append(f"Stream discarded because Qwen did not assign a supported kind: {source_id}")
            continue
        if canonical_locked:
            conflicts = []
            canonical_fields = {
                "kind": stream["kind"],
                "modality": stream["modality"],
                "side": stream["side"],
                "role": stream["role"],
                "variant": stream["variant"],
                "extraction": stream["extraction"],
            }
            for key, canonical_value in canonical_fields.items():
                proposed_value = proposed.get(key)
                if proposed_value is not None and proposed_value != "" and str(proposed_value) != str(canonical_value):
                    conflicts.append(key)
            if conflicts:
                warnings.append(
                    f"Qwen classification was ignored for authoritative canonical stream {source_id}: "
                    + ", ".join(conflicts)
                )
        streams.append(stream)
        accepted_ids.add(source_id)

    restored_ids = []
    for source_id, actual in candidates.items():
        if source_id in accepted_ids or not _authoritative_canonical_stream(actual):
            continue
        stream = _validated_stream(actual, {}, canonical_locked=True)
        if stream is None:
            continue
        streams.append(stream)
        accepted_ids.add(source_id)
        restored_ids.append(source_id)
    if restored_ids:
        preview = ", ".join(restored_ids[:8])
        suffix = "" if len(restored_ids) <= 8 else f" (+{len(restored_ids) - 8} more)"
        warnings.append(
            f"Restored {len(restored_ids)} authoritative canonical stream(s) omitted by Qwen: {preview}{suffix}"
        )

    valid_streams = {stream["source_id"]: stream for stream in streams}
    valid_ids = set(valid_streams)
    associations = []
    for relation in raw.get("associations") or []:
        if not isinstance(relation, dict):
            warnings.append("Qwen returned a malformed association and it was discarded.")
            continue
        vision_id = str(relation.get("vision_id", ""))
        joint_ids = [str(item) for item in relation.get("joint_ids") or [] if str(item) in valid_ids and valid_streams[str(item)]["kind"] == "joint"]
        sensor_ids = [str(item) for item in relation.get("sensor_ids") or [] if str(item) in valid_ids and valid_streams[str(item)]["kind"] == "sensor"]
        if vision_id not in valid_ids or valid_streams[vision_id]["kind"] != "vision":
            warnings.append(f"Association discarded because vision stream is unknown: {vision_id}")
            continue
        associations.append({
            "vision_id": vision_id,
            "joint_ids": joint_ids,
            "sensor_ids": sensor_ids,
            "side": relation.get("side") if relation.get("side") in {"left", "right", "shared", "unknown"} else "unknown",
            "time_alignment": str(relation.get("time_alignment") or "unknown"),
            "timestamp_id": str(relation.get("timestamp_id")) if relation.get("timestamp_id") in valid_ids and valid_streams[str(relation.get("timestamp_id"))]["kind"] == "timestamp" else None,
            "confidence": _bounded_confidence(relation.get("confidence"), 0.5),
            "reason": str(relation.get("reason") or ""),
        })

    warnings.extend(str(item) for item in (raw.get("warnings") or [])[:30])
    understanding = {
        "format_family": str(raw.get("format_family") or "unknown"),
        "format_confidence": _bounded_confidence(raw.get("format_confidence"), 0.0),
        "summary": str(raw.get("summary") or ""),
        "episode_organization": str(raw.get("episode_organization") or "unknown"),
        "streams": streams,
        "associations": associations,
    }
    if not any(stream["kind"] == "vision" for stream in streams):
        warnings.append("No verified vision stream was returned by Qwen.")
    if not any(stream["kind"] == "joint" for stream in streams):
        warnings.append("No verified joint stream was returned by Qwen.")
    return understanding, warnings
