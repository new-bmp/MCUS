from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


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
    lowered = text.lower()
    if any(token in lowered for token in ("rgb", "image", "camera", "vision", "video", "depth", "color")):
        modality = "depth" if "depth" in lowered else "rgb"
        return "vision", modality
    if any(token in lowered for token in ("pressure", "tactile", "force", "torque", "wrench", "load_cell", "ft_sensor")):
        modality = "pressure" if "pressure" in lowered else "tactile" if "tactile" in lowered else "force_torque"
        return "sensor", modality
    if any(token in lowered for token in ("action", "command", "control", "target_qpos", "target_joint")):
        return "action", "command"
    if any(token in lowered for token in ("qpos", "joint_pos", "joint_position", "joint_angle", "joints")):
        return "joint", "position"
    if "observation.state" in lowered or lowered.endswith(".state") or lowered.endswith("/state"):
        return "joint", "state"
    if any(token in lowered for token in ("qvel", "joint_vel", "joint_velocity")):
        return "joint", "velocity"
    if any(token in lowered for token in ("gripper", "proprio", "robot_state", "arm_state")):
        return "joint", "state"
    if any(token in lowered for token in ("timestamp", "time_ns", "time_us", "time_ms", "frame_time")):
        return "timestamp", "time"
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
    for field in fields:
        field_name = str(field.get("key") or "")
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
        if field_name in skeleton_fields:
            continue
        shape = _signal_shape(field.get("shape"))
        if not shape or len(shape) > 2 or not _numeric_dtype(field.get("dtype")):
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
            elif any(token in lowered for token in ("target_qpos", "target_joint", "position_command")):
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
            modality = "image_sequence" if media["type"] == "images" else "video"
            stream = {"id": stream_id, "source_path": media["relative_path"], "field": None, "kind": "vision", "modality": modality, "side_hint": _side(media["relative_path"]), "shape": [media["frame_count"], media["height"], media["width"], 3], "dtype": "uint8", "evidence": "decoded_media"}
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
                streams.append({"id": stream_id, "source_path": relative, "field": field_key, "kind": kind, "modality": modality, "side_hint": (local or {}).get("side_hint") or _side(f"{relative}/{field_key}"), "shape": field.get("shape"), "dtype": field.get("dtype"), "evidence": (local or {}).get("evidence") or "field_name_heuristic"})
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


def validate_understanding(inventory: dict, raw: dict) -> tuple[dict, list[str]]:
    candidates = {stream["id"]: stream for stream in inventory.get("candidate_streams", [])}
    warnings: list[str] = []
    streams = []
    for proposed in raw.get("streams", []):
        source_id = str(proposed.get("source_id", ""))
        if source_id not in candidates:
            warnings.append(f"Qwen returned an unknown stream and it was discarded: {source_id}")
            continue
        actual = candidates[source_id]
        proposed_kind = proposed.get("kind")
        final_kind = proposed_kind if proposed_kind in {"vision", "joint", "sensor", "action", "timestamp"} else actual["kind"]
        if final_kind == "other":
            warnings.append(f"Stream discarded because Qwen did not assign a supported kind: {source_id}")
            continue
        shape = actual.get("shape") if isinstance(actual.get("shape"), list) else []
        dimension_count = int(shape[-1]) if len(shape) > 1 and isinstance(shape[-1], int) else None
        dimension_names = [str(item)[:120] for item in proposed.get("dimension_names", []) if str(item).strip()][:MAX_FIELDS]
        gripper_indices = []
        for value in proposed.get("gripper_indices", []):
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if index >= 0 and (dimension_count is None or index < dimension_count):
                gripper_indices.append(index)
        representation = str(proposed.get("representation") or "unknown")
        if representation not in {"absolute", "delta", "velocity", "unknown"}:
            representation = "unknown"
        streams.append({
            "source_id": source_id,
            "kind": final_kind,
            "modality": str(proposed.get("modality") or actual["modality"]),
            "side": proposed.get("side") if proposed.get("side") in {"left", "right", "shared", "unknown"} else actual["side_hint"],
            "role": str(proposed.get("role") or ""),
            "representation": representation,
            "dimension_names": dimension_names,
            "gripper_indices": sorted(set(gripper_indices)),
            "embodiment_id": str(proposed.get("embodiment_id") or "").strip() or None,
            "confidence": max(0.0, min(1.0, float(proposed.get("confidence", 0.5)))),
            "evidence": str(proposed.get("evidence") or actual["evidence"]),
            "source_path": actual["source_path"],
            "field": actual["field"],
            "shape": actual["shape"],
            "dtype": actual["dtype"],
        })

    valid_streams = {stream["source_id"]: stream for stream in streams}
    valid_ids = set(valid_streams)
    associations = []
    for relation in raw.get("associations", []):
        vision_id = str(relation.get("vision_id", ""))
        joint_ids = [str(item) for item in relation.get("joint_ids", []) if str(item) in valid_ids and valid_streams[str(item)]["kind"] == "joint"]
        sensor_ids = [str(item) for item in relation.get("sensor_ids", []) if str(item) in valid_ids and valid_streams[str(item)]["kind"] == "sensor"]
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
            "confidence": max(0.0, min(1.0, float(relation.get("confidence", 0.5)))),
            "reason": str(relation.get("reason") or ""),
        })

    warnings.extend(str(item) for item in raw.get("warnings", [])[:30])
    understanding = {
        "format_family": str(raw.get("format_family") or "unknown"),
        "format_confidence": max(0.0, min(1.0, float(raw.get("format_confidence", 0.0)))),
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
