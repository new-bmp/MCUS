from __future__ import annotations

"""Read-only dataset format detection and canonical stream mapping.

The adapter never rewrites source data.  It produces a bounded preflight
report before import and a canonical stream map that the rest of Alice can
consume in the same way regardless of the recorder that produced the files.
"""

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .camera_profiles import NEXUS_OAKD_PRO_W9_PROFILE_ID, nexus_camera_profile, nexus_camera_profiles


FORMAT_MAP_SCHEMA = "alice/dataset-format-map/v1"
_EPISODE_DIRECTORY = re.compile(r"^(?:ep|episode)[_-]?\d+(?:[_-].*)?$", re.IGNORECASE)
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_STRUCTURED_EXTENSIONS = {".h5", ".hdf5", ".h5df", ".parquet", ".npz", ".npy", ".csv", ".tsv", ".json", ".jsonl"}
_IGNORED_DIRECTORIES = {
    ".alicepd", ".git", ".hg", ".svn", ".venv", "venv", ".vla_lens",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules", "output",
}
_MODALITY_DIRECTORIES = {
    "action", "actions", "body", "calibration", "camera", "cameras", "color", "data", "depth",
    "image", "images", "imu", "joint", "joints", "meta", "metadata", "mocap", "pressure", "rgb",
    "sensor", "sensors", "state", "tactile", "video", "videos", "wrench",
}
_SUPPORTED_STREAM_KINDS = {"vision", "joint", "sensor", "action", "timestamp"}
_MAX_JSON_BYTES = 8 * 1024 * 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normal(value: Any) -> str:
    return str(value or "").replace("\\", "/").strip("/").casefold().replace("-", "_")


def _safe_json(path: Path) -> dict:
    try:
        if not path.is_file() or path.stat().st_size > _MAX_JSON_BYTES:
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}


def _has_transform(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value.get("matrix") or value.get("rotation") or value.get("translation"))
    if not isinstance(value, list) or len(value) not in {3, 4}:
        return False
    return all(isinstance(row, list) and len(row) in {3, 4} for row in value)


def _visible_directories(root: Path) -> list[Path]:
    result: list[Path] = []
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if entry.name.casefold() in _IGNORED_DIRECTORIES or entry.name.startswith("."):
                    continue
                try:
                    if entry.is_dir(follow_symlinks=False):
                        result.append(Path(entry.path))
                except OSError:
                    continue
    except OSError:
        return []
    return sorted(result, key=lambda item: item.name.casefold())


def _looks_like_episode_directory(path: Path) -> bool:
    if (path / "meta" / "metadata.json").is_file() or (path / "metadata.json").is_file():
        return True
    if not _EPISODE_DIRECTORY.fullmatch(path.name):
        return False
    for relative in ("camera", "video", "videos", "rgb", "images"):
        folder = path / relative
        if folder.is_dir():
            try:
                if any(item.is_file() and item.suffix.casefold() in (_VIDEO_EXTENSIONS | _IMAGE_EXTENSIONS) for item in folder.iterdir()):
                    return True
            except OSError:
                pass
    return False


def _is_alice_full_root(root: Path) -> bool:
    index = root / "dataset.json"
    payload = _safe_json(index)
    return str(payload.get("schema") or "") in {"alice/full-dataset/v2", "alice/full-mano-dataset/v1"}


def _is_lerobot_root(root: Path) -> bool:
    return (root / "meta" / "info.json").is_file() and (root / "data").is_dir() and (root / "videos").is_dir()


def is_self_describing_dataset_root(path: str | Path) -> bool:
    """Return whether immediate child folders are modalities/Episodes, not datasets."""
    root = Path(path).expanduser().resolve()
    if _is_alice_full_root(root) or _is_lerobot_root(root):
        return True
    if (root / "meta" / "metadata.json").is_file() or (root / "metadata.json").is_file():
        return True
    children = _visible_directories(root)
    if not children:
        return False
    episode_children = [child for child in children if _looks_like_episode_directory(child)]
    if episode_children and len(episode_children) >= max(1, round(len(children) * 0.7)):
        return True
    modality_count = sum(child.name.casefold() in _MODALITY_DIRECTORIES for child in children)
    return modality_count >= max(2, round(len(children) * 0.7))


def _even_sample(values: list[Path], limit: int) -> list[Path]:
    if len(values) <= limit:
        return values
    indices = {round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)}
    return [values[index] for index in sorted(indices)]


def _bounded_files(root: Path, episode_directories: list[Path], limit: int = 500) -> list[Path]:
    seeds = _even_sample(episode_directories, 8) if episode_directories else [root]
    result: list[Path] = []
    for seed in seeds:
        stack: list[tuple[Path, int]] = [(seed, 0)]
        while stack and len(result) < limit:
            folder, depth = stack.pop()
            try:
                entries = sorted(folder.iterdir(), key=lambda item: item.name.casefold(), reverse=True)
            except OSError:
                continue
            for item in entries:
                if item.name.casefold() in _IGNORED_DIRECTORIES or item.name.startswith("."):
                    continue
                if item.is_file():
                    result.append(item)
                    if len(result) >= limit:
                        break
                elif item.is_dir() and depth < 3:
                    stack.append((item, depth + 1))
    return result


def _nested_get(value: Any, keys: Iterable[str]) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _file_leaves(value: Any, trail: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], str]]:
    leaves: list[tuple[tuple[str, ...], str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            leaves.extend(_file_leaves(child, trail + (str(key),)))
    elif isinstance(value, str):
        leaves.append((trail, value.replace("\\", "/")))
    return leaves


def _side(text: str) -> str:
    lowered = _normal(text)
    if any(token in lowered for token in ("left", "l_hand", "l_arm", "camera_l", "cam_l")):
        return "left"
    if any(token in lowered for token in ("right", "r_hand", "r_arm", "camera_r", "cam_r")):
        return "right"
    if any(token in lowered for token in ("bimanual", "both", "dual_arm", "stereo")):
        return "shared"
    return "unknown"


def _declared_sensor_config(metadata: dict, trail: tuple[str, ...]) -> dict:
    sensors = metadata.get("sensors") if isinstance(metadata.get("sensors"), dict) else {}
    if not trail:
        return {}
    category = trail[0]
    name = trail[1] if len(trail) > 1 else ""
    if category == "camera":
        camera_name = "head" if name.startswith("head_") else name
        camera = _nested_get(sensors, ("camera", camera_name))
        if name.endswith("depth") and isinstance(camera, dict) and isinstance(camera.get("depth"), dict):
            return camera["depth"]
        return camera if isinstance(camera, dict) else {}
    if category in {"tactile", "mocap"}:
        candidate = _nested_get(sensors, (category, name.removesuffix("_raw")))
        return candidate if isinstance(candidate, dict) else {}
    if category == "sensor" and name == "head_imu":
        candidate = _nested_get(sensors, ("camera", "head", "imu"))
        return candidate if isinstance(candidate, dict) else {}
    return {}


def _stream_descriptor_from_declared(metadata: dict, trail: tuple[str, ...], source_path: str) -> dict:
    config = _declared_sensor_config(metadata, trail)
    descriptor = describe_source(source_path)
    resolution = config.get("resolution") if isinstance(config.get("resolution"), (list, tuple)) else []
    width = int(resolution[0]) if len(resolution) >= 2 and str(resolution[0]).isdigit() else None
    height = int(resolution[1]) if len(resolution) >= 2 and str(resolution[1]).isdigit() else None
    normalized_source = _normal(source_path)
    normalized_raw = _normal(config.get("raw_path"))
    normalized_sync = _normal(config.get("sync_path"))
    if normalized_raw and normalized_source == normalized_raw:
        descriptor["variant"] = "raw"
        descriptor["synchronized"] = False
        fps_keys = ("source_fps", "rate_hz", "storage_fps", "fps")
    elif normalized_sync and normalized_source == normalized_sync:
        descriptor["variant"] = "synchronized"
        descriptor["synchronized"] = True
        fps_keys = ("sync_fps", "storage_fps", "rate_hz", "fps")
    else:
        fps_keys = ("storage_fps", "source_fps", "rate_hz", "fps", "sync_fps")
    fps = next((config.get(key) for key in fps_keys if config.get(key) is not None), None)
    source_fps = next((config.get(key) for key in ("source_fps", "actual_fps", "rate_hz", "fps", "storage_fps") if config.get(key) is not None), None)
    storage_fps = next((config.get(key) for key in ("storage_fps", "fps", "actual_fps", "source_fps", "rate_hz") if config.get(key) is not None), None)
    sync_fps = config.get("sync_fps")
    master_rate = (metadata.get("sync") or {}).get("tick_hz") if isinstance(metadata.get("sync"), dict) else None
    if sync_fps is None and isinstance(master_rate, (int, float)):
        sync_fps = master_rate
    if descriptor["variant"] == "synchronized" and config.get("sync_fps") is None:
        if isinstance(master_rate, (int, float)):
            fps = master_rate
    frame_count = metadata.get("frame_count") if descriptor["modality"] == "depth" else config.get("sample_count")
    codec = str(config.get("codec") or "")
    dtype = "uint16" if "uint16" in codec.casefold() else None
    return {
        "source_path_template": source_path,
        "kind": descriptor["kind"],
        "modality": descriptor["modality"],
        "side": descriptor["side"],
        "role": descriptor["role"],
        "variant": descriptor["variant"],
        "width": width,
        "height": height,
        "fps": float(fps) if isinstance(fps, (int, float)) else None,
        "source_fps": float(source_fps) if isinstance(source_fps, (int, float)) else None,
        "storage_fps": float(storage_fps) if isinstance(storage_fps, (int, float)) else None,
        "sync_fps": float(sync_fps) if isinstance(sync_fps, (int, float)) else None,
        "frame_count": int(frame_count) if isinstance(frame_count, int) else None,
        "dtype": dtype,
        "unit": config.get("unit"),
        "channels": config.get("channels") or config.get("nodes"),
        "node_count": config.get("nodes"),
        "declared_by": "meta/metadata.json",
    }


def _declared_streams(metadata: dict) -> list[dict]:
    files = metadata.get("files") if isinstance(metadata.get("files"), dict) else {}
    streams = [_stream_descriptor_from_declared(metadata, trail, path) for trail, path in _file_leaves(files)]
    # Some recorders declare a stream only in the sensor configuration.  The
    # Nexus head IMU is one such example: it is not duplicated under `files`.
    sensors = metadata.get("sensors") if isinstance(metadata.get("sensors"), dict) else {}
    for category, devices in sensors.items():
        if not isinstance(devices, dict):
            continue
        for device_name, config in devices.items():
            if not isinstance(config, dict):
                continue
            nested_configs = [(str(device_name), config)]
            nested_configs.extend(
                (f"{device_name}_{nested_name}", nested)
                for nested_name, nested in config.items()
                if isinstance(nested, dict)
            )
            for declared_name, declared_config in nested_configs:
                for key in ("path", "sync_path", "raw_path"):
                    source_path = declared_config.get(key)
                    if not isinstance(source_path, str) or not source_path.strip():
                        continue
                    descriptor = describe_source(source_path)
                    if key == "raw_path":
                        descriptor["variant"] = "raw"
                        descriptor["synchronized"] = False
                    elif key == "sync_path":
                        descriptor["variant"] = "synchronized"
                        descriptor["synchronized"] = True
                    if key == "raw_path":
                        fps_keys = ("source_fps", "rate_hz", "storage_fps", "fps")
                    elif key == "sync_path":
                        fps_keys = ("sync_fps", "storage_fps", "rate_hz", "fps", "source_fps")
                    else:
                        fps_keys = ("storage_fps", "source_fps", "rate_hz", "fps", "sync_fps")
                    fps = next((declared_config.get(name) for name in fps_keys if declared_config.get(name) is not None), None)
                    source_fps = next((declared_config.get(name) for name in ("source_fps", "actual_fps", "rate_hz", "fps", "storage_fps") if declared_config.get(name) is not None), None)
                    storage_fps = next((declared_config.get(name) for name in ("storage_fps", "fps", "actual_fps", "source_fps", "rate_hz") if declared_config.get(name) is not None), None)
                    sync_fps = declared_config.get("sync_fps")
                    master_rate = (metadata.get("sync") or {}).get("tick_hz") if isinstance(metadata.get("sync"), dict) else None
                    if sync_fps is None and isinstance(master_rate, (int, float)):
                        sync_fps = master_rate
                    streams.append({
                        "source_path_template": source_path,
                        **{name: descriptor[name] for name in ("kind", "modality", "side", "role", "variant")},
                        "fps": float(fps) if isinstance(fps, (int, float)) else None,
                        "source_fps": float(source_fps) if isinstance(source_fps, (int, float)) else None,
                        "storage_fps": float(storage_fps) if isinstance(storage_fps, (int, float)) else None,
                        "sync_fps": float(sync_fps) if isinstance(sync_fps, (int, float)) else None,
                        "frame_count": declared_config.get("sample_count"),
                        "dtype": None,
                        "unit": declared_config.get("unit"),
                        "channels": declared_config.get("channels"),
                        "node_count": declared_config.get("nodes"),
                        "declared_by": f"sensors.{category}.{declared_name}",
                    })
    unique: dict[str, dict] = {}
    for stream in streams:
        unique.setdefault(str(stream["source_path_template"]).casefold(), stream)
    return list(unique.values())


def _has_egodex_transform_file(files: list[Path]) -> bool:
    candidates = [path for path in files if path.suffix.casefold() in {".h5", ".hdf5", ".h5df"}]
    for path in candidates[:8]:
        try:
            import h5py

            with h5py.File(path, "r") as source:
                transforms = source.get("transforms")
                if isinstance(transforms, h5py.Group) and "camera" in transforms and any("Hand" in key for key in transforms.keys()):
                    return True
        except Exception:
            continue
    return False


def _has_embedded_camera_intrinsics(files: list[Path]) -> bool:
    """Detect camera intrinsics stored inside an EgoDex HDF5 episode."""
    candidates = [path for path in files if path.suffix.casefold() in {".h5", ".hdf5", ".h5df"}]
    for path in candidates[:8]:
        try:
            import h5py

            with h5py.File(path, "r") as source:
                for key in ("camera/intrinsic", "camera/intrinsics", "intrinsic", "intrinsics"):
                    if key in source:
                        return True
        except Exception:
            continue
    return False


def _metadata_paths(root: Path, episode_directories: list[Path]) -> list[Path]:
    direct = root / "meta" / "metadata.json"
    if direct.is_file():
        return [direct]
    direct = root / "metadata.json"
    if direct.is_file():
        return [direct]
    paths = [episode / "meta" / "metadata.json" for episode in episode_directories]
    return [path for path in _even_sample(paths, 8) if path.is_file()]


def _issue(severity: str, code: str, message: str) -> dict:
    return {"severity": severity, "code": code, "message": message}


def processing_strategy_for_family(
    family: str,
    *,
    has_joint: bool,
    has_action: bool,
    has_contact_sensor: bool,
    has_timestamps: bool,
) -> dict:
    """Return the explicit processing policy for a recognized dataset family."""
    if family == "egodex":
        return {
            "id": "egodex_joint_centric_v1",
            "label": "EgoDex 关节轨迹策略",
            "description": "以双手 Joint/腕部位姿为主，允许关节投影、手部可见性与轨迹质量检查。",
            "motion_sources": ["joint", "action", "video"],
            "joint_overlay": True,
            "pose_recovery": bool(has_joint),
            "sensor_alignment": "when_available" if has_timestamps else "frame_index_fallback",
            "pressure": {
                "enabled": bool(has_contact_sensor),
                "role": "auxiliary_interaction_evidence",
                "preferred_variant": "synchronized",
                "raw_variant_role": "sensor_quality_only",
                "baseline": "per_episode_adaptive",
                "hard_reject": False,
            },
        }
    if family == "nexus_multimodal":
        return {
            "id": "nexus_sensor_fusion_v1",
            "label": "Nexus 多传感器融合策略",
            "description": "以同步视频、Mocap、IMU 与触觉/压力融合为主，不使用 Joint 画面叠加。",
            "motion_sources": ["mocap", "imu", "video"],
            "joint_overlay": False,
            "pose_recovery": False,
            "sensor_alignment": "required",
            "pressure": {
                "enabled": bool(has_contact_sensor),
                "role": "auxiliary_interaction_evidence",
                "preferred_variant": "synchronized",
                "raw_variant_role": "sensor_quality_only",
                "baseline": "per_episode_adaptive",
                "hard_reject": False,
                "empty_value_hard_reject": True,
                "zero_is_empty": False,
                "features": [
                    "pressure_level",
                    "active_taxel_count",
                    "pressure_change",
                    "contact_onset",
                    "contact_release",
                ],
            },
        }
    return {
        "id": "generic_capability_driven_v1",
        "label": "通用能力驱动策略",
        "description": "仅启用由格式检查明确确认的能力，不推断缺失的传感器或标定。",
        "motion_sources": [source for source, available in (("joint", has_joint), ("action", has_action), ("video", True)) if available],
        "joint_overlay": True,
        "pose_recovery": family == "egodex" and has_joint,
        "sensor_alignment": "when_available" if has_timestamps else "frame_index_fallback",
        "pressure": {
            "enabled": bool(has_contact_sensor),
            "role": "auxiliary_interaction_evidence",
            "preferred_variant": "synchronized",
            "raw_variant_role": "sensor_quality_only",
            "baseline": "per_episode_adaptive",
            "hard_reject": False,
        },
    }


def inspect_dataset_format(path: str | Path, camera_profile_id: str | None = None) -> dict:
    root = Path(path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    children = _visible_directories(root)
    episode_directories = [child for child in children if _looks_like_episode_directory(child)]
    metadata_paths = _metadata_paths(root, episode_directories)
    metadata_samples = [_safe_json(item) for item in metadata_paths]
    metadata_samples = [item for item in metadata_samples if item]
    representative = metadata_samples[0] if metadata_samples else {}
    sampled_files = _bounded_files(root, episode_directories)

    if _is_lerobot_root(root):
        family, confidence = "lerobot", 0.99
    elif _is_alice_full_root(root):
        family, confidence = "alice_full", 0.99
    elif representative.get("nexus_version") or (
        str(representative.get("schema_version") or "").startswith("4")
        and isinstance(representative.get("sensors"), dict)
        and isinstance(representative.get("files"), dict)
    ):
        family, confidence = "nexus_multimodal", 0.99
    elif _has_egodex_transform_file(sampled_files):
        family, confidence = "egodex", 0.98
    elif any(item.suffix.casefold() in _VIDEO_EXTENSIONS for item in sampled_files) and any(
        item.suffix.casefold() in _STRUCTURED_EXTENSIONS for item in sampled_files
    ):
        family, confidence = "generic_multimodal", 0.78
    elif any(item.suffix.casefold() in (_VIDEO_EXTENSIONS | _IMAGE_EXTENSIONS) for item in sampled_files):
        family, confidence = "vision_only", 0.72
    else:
        family, confidence = "unknown", 0.2

    declared_streams = []
    for metadata in metadata_samples or ([representative] if representative else []):
        declared_streams.extend(_declared_streams(metadata))
    if declared_streams:
        declared_streams = list({
            str(item.get("source_path_template") or "").casefold(): item
            for item in reversed(declared_streams)
            if item.get("source_path_template")
        }.values())
    if not declared_streams:
        declared_streams = []
        for item in sampled_files:
            relative = item.relative_to(root).as_posix()
            descriptor = describe_source(relative)
            if descriptor["kind"] in _SUPPORTED_STREAM_KINDS:
                declared_streams.append({
                    "source_path_template": relative,
                    **{key: descriptor[key] for key in ("kind", "modality", "side", "role", "variant")},
                    "declared_by": "bounded_filename_probe",
                })

    modality_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    for stream in declared_streams:
        modality = str(stream.get("modality") or "unknown")
        kind = str(stream.get("kind") or "other")
        modality_counts[modality] = modality_counts.get(modality, 0) + 1
        kind_counts[kind] = kind_counts.get(kind, 0) + 1

    has_rgb = modality_counts.get("rgb", 0) > 0
    has_depth = modality_counts.get("depth", 0) > 0
    has_infrared = modality_counts.get("infrared", 0) > 0
    decodable_depth = any(
        item.get("modality") == "depth"
        and (
            not str(item.get("source_path_template") or "").casefold().endswith(".raw")
            or bool(item.get("width") and item.get("height") and item.get("dtype"))
        )
        for item in declared_streams
    )
    has_reviewable_visual = has_rgb or has_infrared or decodable_depth
    has_joint = kind_counts.get("joint", 0) > 0 or family == "egodex"
    has_sensor = kind_counts.get("sensor", 0) > 0
    has_action = kind_counts.get("action", 0) > 0
    has_timestamps = kind_counts.get("timestamp", 0) > 0 or any("sync" in _normal(item.name) or "timestamp" in _normal(item.name) for item in sampled_files)
    has_contact_sensor = any(
        item.get("modality") in {"tactile", "pressure", "force_torque"} or item.get("role") == "contact_sensor"
        for item in declared_streams
    )
    processing_strategy = processing_strategy_for_family(
        family,
        has_joint=has_joint,
        has_action=has_action,
        has_contact_sensor=has_contact_sensor,
        has_timestamps=has_timestamps,
    )
    has_intrinsics = any("camera_" in item.name.casefold() and item.suffix.casefold() == ".json" for item in sampled_files)
    if family == "egodex" and not has_intrinsics:
        has_intrinsics = _has_embedded_camera_intrinsics(sampled_files)
    extrinsics_payloads = [_safe_json(path.parent / "camera_extrinsics.json") for path in metadata_paths]
    source_extrinsics_applied = any(bool(item.get("applied")) for item in extrinsics_payloads if item)
    source_rgb_depth_extrinsics = any(
        _has_transform(item.get("T_head_rgb__head_depth"))
        for item in extrinsics_payloads
        if item
    )
    selected_camera_profile = nexus_camera_profile(camera_profile_id) if camera_profile_id else None
    if selected_camera_profile and family != "nexus_multimodal":
        raise ValueError("Camera fallback profiles are only available for Nexus datasets")
    effective_rgb_depth_extrinsics = bool(source_rgb_depth_extrinsics or selected_camera_profile)
    hand_projection_requires_extrinsics = family == "nexus_multimodal"
    hand_projection_ready = bool(
        has_intrinsics
        and (not hand_projection_requires_extrinsics or source_extrinsics_applied)
    )
    available_camera_profiles = (
        nexus_camera_profiles()
        if family == "nexus_multimodal" and has_rgb and has_depth and not source_rgb_depth_extrinsics
        else []
    )
    camera_calibration = {
        "source_extrinsics_applied": source_extrinsics_applied,
        "source_rgb_depth_extrinsics": source_rgb_depth_extrinsics,
        "effective_rgb_depth_extrinsics": effective_rgb_depth_extrinsics,
        "requires_profile_selection": bool(available_camera_profiles and not selected_camera_profile),
        "recommended_profile_id": NEXUS_OAKD_PRO_W9_PROFILE_ID if available_camera_profiles else None,
        "selected_profile_id": selected_camera_profile.get("id") if selected_camera_profile else None,
        "selected_profile": selected_camera_profile,
        "profiles": available_camera_profiles,
        "source_files_modified": False,
        "hand_projection_mode": "embedded_camera_pose" if family == "egodex" else "calibrated_extrinsics",
        "hand_projection_requires_extrinsics": hand_projection_requires_extrinsics,
        "hand_projection_ready": hand_projection_ready,
        "intrinsics_available": has_intrinsics,
    }
    node_counts: set[int] = set()
    for metadata in metadata_samples or ([representative] if representative else []):
        mocap_configs = (metadata.get("sensors") or {}).get("mocap") if isinstance(metadata.get("sensors"), dict) else {}
        node_counts.update(
            int(config.get("nodes"))
            for config in (mocap_configs or {}).values()
            if isinstance(config, dict) and isinstance(config.get("nodes"), int)
        )

    issues: list[dict] = []
    if not has_reviewable_visual:
        issues.append(_issue("error", "no_visual_stream", "未识别到可用的 RGB、Depth 或图像流，无法建立可审阅 Episode。"))
    if has_depth:
        raw_depth = any(str(item.get("source_path_template") or "").casefold().endswith(".raw") for item in declared_streams)
        if raw_depth:
            known_geometry = any(item.get("width") and item.get("height") and item.get("dtype") for item in declared_streams if item.get("modality") == "depth")
            if known_geometry:
                issues.append(_issue("info", "raw_depth_declared", "已从采集元数据确认原始 uint16 深度图的分辨率、单位与帧率；深度不会被当成 RGB。"))
            else:
                severity = "warning" if has_rgb else "error"
                issues.append(_issue(severity, "raw_depth_geometry_unknown", "发现原始深度文件，但缺少可靠的分辨率或 dtype，深度只会索引而不会猜测解码。"))
    if has_joint and has_rgb and hand_projection_requires_extrinsics and not source_extrinsics_applied:
        issues.append(_issue("warning", "camera_extrinsics_missing", "手部/关节轨迹存在，但相机外参未应用；可做时序质量检查，不能宣称 2D 投影或整手可见性准确。"))
    if has_joint and not has_action:
        issues.append(_issue("info", "action_missing", "未发现原生 Action；S2 需要先生成 Action，其他可用检查不受影响。"))
    if node_counts and node_counts != {21}:
        issues.append(_issue("warning", "noncanonical_hand_nodes", f"源手部骨架节点数为 {sorted(node_counts)}，不会补零冒充 21 点 MANO；固定 21 点 Full 导出将被能力检查阻止。"))
    if family == "nexus_multimodal":
        issues.append(_issue("info", "nexus_sensor_fusion_strategy", "Nexus 使用独立的多传感器融合策略；Mocap 保留用于时序与运动分析，但不会启用 Joint 画面叠加。"))
        if has_rgb and has_depth:
            if selected_camera_profile:
                issues.append(_issue(
                    "info",
                    "rgb_depth_camera_profile_selected",
                    f"已选择 {selected_camera_profile['label']} 作为 RGB–Depth 外参回退；配置写入 Alice sidecar，不改写 Nexus 源文件。",
                ))
            elif not source_rgb_depth_extrinsics:
                issues.append(_issue(
                    "warning",
                    "rgb_depth_extrinsics_missing",
                    "RGB–Depth 外参为空；导入前请选择摄像头预设，或保持未标定并停用深度到 RGB 的空间配准。",
                ))
        if has_contact_sensor:
            issues.append(_issue("info", "pressure_auxiliary_only", "触觉/压力数值仅作为交互与动作边界的辅助证据，数值为 0 不会删帧；同步流缺行、partial 或空值会作为 Nexus 传感器错误直接标红。"))

    can_full_export = family in {"egodex", "lerobot", "alice_full"} and (not node_counts or node_counts == {21})
    capabilities = {
        "can_import": has_reviewable_visual,
        "can_vlm": has_rgb,
        "can_video_smoothing": has_rgb,
        "can_s1": has_joint or has_action,
        "can_s2": has_action,
        "can_curation": has_rgb and (has_joint or has_action),
        "can_sensor_alignment": has_timestamps or has_sensor,
        "can_depth_preview": any(
            item.get("modality") == "depth" and item.get("width") and item.get("height") and item.get("dtype") == "uint16"
            for item in declared_streams
        ),
        "can_joint_overlay": bool(processing_strategy["joint_overlay"] and has_joint and hand_projection_ready),
        "can_hand_visibility": bool(has_joint and hand_projection_ready),
        "can_rgb_depth_registration": bool(has_rgb and has_depth and has_intrinsics and effective_rgb_depth_extrinsics),
        "can_depth_arm_localization": bool(has_depth and has_intrinsics and effective_rgb_depth_extrinsics),
        "can_pose_recovery": bool(processing_strategy["pose_recovery"]),
        "can_pressure_analysis": has_contact_sensor,
        "can_full_export": can_full_export,
    }
    severities = {item["severity"] for item in issues}
    status = "blocked" if "error" in severities else "warning" if "warning" in severities else "ready"
    root_mode = "dataset" if is_self_describing_dataset_root(root) or not children else "collection"
    sampled_signatures: list[str] = []
    for item in [*metadata_paths, *sampled_files]:
        try:
            stat = item.stat()
            relative = item.relative_to(root).as_posix()
        except (OSError, ValueError):
            continue
        sampled_signatures.append(f"{relative}:{stat.st_size}:{stat.st_mtime_ns}")
    signature_parts = [
        str(root).casefold(), family, str(len(episode_directories)),
        *sorted(set(sampled_signatures)),
        *[child.name for child in children[:64]],
    ]
    token = hashlib.sha256("\n".join(signature_parts).encode("utf-8")).hexdigest()[:24]
    extension_counts: dict[str, int] = {}
    for item in sampled_files:
        suffix = item.suffix.casefold() or "<none>"
        extension_counts[suffix] = extension_counts.get(suffix, 0) + 1

    return {
        "schema": FORMAT_MAP_SCHEMA,
        "generated_at": _utc_now(),
        "root_path": str(root),
        "root_mode": root_mode,
        "format_family": family,
        "format_version": str(representative.get("schema_version") or "") or None,
        "format_confidence": confidence,
        "status": status,
        "confirmation_token": token,
        "episode_layout": "episode_directories" if episode_directories else "flat_or_modality_tree",
        "episode_count_hint": len(episode_directories) if episode_directories else None,
        "episode_samples": [path.name for path in _even_sample(episode_directories, 8)],
        "metadata_samples": [path.relative_to(root).as_posix() for path in metadata_paths],
        "sampled_file_count": len(sampled_files),
        "sampled_extension_counts": extension_counts,
        "declared_streams": declared_streams,
        "modality_counts": modality_counts,
        "kind_counts": kind_counts,
        "processing_strategy": processing_strategy,
        "camera_calibration": camera_calibration,
        "capabilities": capabilities,
        "issues": issues,
        "mapping_rules": [
            "RGB、Depth、Joint、Action、Tactile/Pressure、IMU 与 Timestamp 保持独立流，不按数组形状互相猜测。",
            "优先使用 recorder 声明的 sync 文件和已同步 HDF5；存在 *_raw 对时不覆盖同步流。",
            "未知字段保留在源文件索引并报告，不补零、不伪造关节、不自动当作 Action。",
        ],
        "source_policy": "Preflight is read-only. Source files are never rewritten; canonical mappings are stored only in .alicePD.",
    }


def _declared_match(report: dict | None, relative_path: str) -> dict:
    wanted = _normal(relative_path)
    for stream in (report or {}).get("declared_streams", []):
        template = _normal(stream.get("source_path_template"))
        if template and (wanted == template or wanted.endswith(f"/{template}")):
            return dict(stream)
    return {}


def describe_source(
    relative_path: str,
    field: str | None = None,
    shape: list[int] | None = None,
    dtype: str | None = None,
    report: dict | None = None,
) -> dict:
    """Classify one file/field without conflating similarly shaped modalities."""
    declared = _declared_match(report, relative_path)
    path_text = _normal(relative_path)
    field_text = _normal(field)
    text = f"{path_text}/{field_text}".strip("/")
    suffix = Path(relative_path).suffix.casefold()
    side = str(declared.get("side") or _side(text))
    raw_variant = any(
        Path(part).stem.endswith("_raw") or Path(part).stem == "raw"
        for part in text.split("/")
    )
    variant = str(declared.get("variant") or ("raw" if raw_variant else "synchronized" if "sync" in text else "primary"))

    kind = str(declared.get("kind") or "")
    modality = str(declared.get("modality") or "")
    role = str(declared.get("role") or "")
    if field_text:
        leaf = field_text.rsplit("/", 1)[-1]
        if (
            any(token in leaf for token in ("timestamp", "timestamps", "master_ts", "sensor_ts", "time_ns", "time_us", "time_ms", "ts_wall", "frame_time", "pts_us", "arrival_ts"))
            or leaf.endswith("_ts")
            or leaf.endswith("_mono_ns")
            or leaf in {"source_seq", "frame_idx", "sequence"}
        ):
            kind, modality, role = "timestamp", "time", "clock_or_index"
        elif leaf in {"partial", "partial_reason", "filled", "valid", "validity", "device_timestamp_valid"} or leaf.endswith("_filled"):
            kind, modality, role = "other", "validity", "validity_mask"
        elif any(token in field_text for token in ("/imu", "imu/", "accel", "accelerometer", "gyro", "gyroscope")):
            kind, modality, role = "sensor", "imu", "inertial_sensor"
        elif leaf == "adc" or any(token in field_text for token in ("pressure", "tactile", "force", "torque", "wrench")):
            kind = "sensor"
            modality = str(declared.get("modality") or ("tactile" if leaf == "adc" else "pressure"))
            role = "contact_sensor"
        elif any(token in leaf for token in ("skeleton", "keypoint")):
            kind, modality, role = "joint", "pose", "hand_skeleton"
        elif leaf == "joints" and str(declared.get("role") or "") == "hand_skeleton":
            kind, modality, role = "sensor", "hand_device", "human_hand_device_channels"
        elif leaf == "wrist_quat":
            kind, modality, role = "joint", "pose", "hand_orientation_auxiliary"
    if not kind:
        if field_text and any(token in field_text for token in ("timestamp", "timestamps", "master_ts", "sensor_ts", "time_ns", "time_us", "time_ms", "ts_wall", "frame_time", "pts_us")):
            kind, modality, role = "timestamp", "time", "clock"
        elif suffix in {".json", ".yaml", ".yml", ".toml"} and (
            path_text.startswith("meta/") or "/meta/" in f"/{path_text}/"
        ):
            kind, modality, role = "metadata", "metadata", "dataset_metadata"
        elif suffix in _VIDEO_EXTENSIONS or suffix in _IMAGE_EXTENSIONS:
            kind = "vision"
            modality = "depth" if any(token in text for token in ("depth", "disparity")) else "infrared" if any(token in text for token in ("infrared", "_ir", "/ir")) else "rgb"
            role = "camera_stream"
        elif suffix == ".raw" and any(token in text for token in ("depth", "disparity")):
            kind, modality, role = "vision", "depth", "raw_depth_frames"
        elif any(token in text for token in ("tactile", "pressure", "force", "torque", "wrench", "load_cell", "ft_sensor", "/adc")):
            kind = "sensor"
            modality = "tactile" if "tactile" in text or "/adc" in text else "pressure" if "pressure" in text else "force_torque"
            role = "contact_sensor"
        elif any(token in text for token in ("/imu", "imu/", "_imu", "imu.", "accel", "accelerometer", "gyro", "gyroscope")):
            kind, modality, role = "sensor", "imu", "inertial_sensor"
        elif any(token in text for token in ("action", "command", "control", "target_qpos", "target_joint")):
            kind, modality, role = "action", "command", "robot_action"
        elif any(token in text for token in ("skeleton", "keypoint", "mocap", "joint_pos", "joint_position", "joint_angle", "qpos", "wrist_quat", "endpose", "end_pose", "eef_pose", "tcp_pose")):
            kind = "joint"
            modality = "pose" if any(token in text for token in ("skeleton", "keypoint", "mocap", "quat", "pose")) else "position"
            role = "hand_skeleton" if any(token in text for token in ("skeleton", "hand", "mocap")) else "joint_state"
        elif field_text and shape and len(shape) >= 3 and any(token in text for token in ("depth", "image", "rgb")):
            kind, modality, role = "vision", "depth" if "depth" in text else "rgb", "embedded_image_tensor"
        elif any(token in text for token in ("sync.parquet", "video_timestamps", "arrival_timestamps")):
            kind, modality, role = "timestamp", "time", "alignment_table"
        elif suffix in {".json", ".yaml", ".yml", ".toml"} or any(token in text for token in ("metadata", "calibration", "manifest")):
            kind, modality, role = "metadata", "metadata", "dataset_metadata"
        else:
            kind, modality, role = "other", "unknown", ""

    result = {
        "kind": kind,
        "modality": modality or "unknown",
        "side": side if side in {"left", "right", "shared", "unknown"} else "unknown",
        "role": role,
        "variant": variant,
        "synchronized": variant != "raw",
        "confidence": 0.98 if declared else 0.86 if kind in _SUPPORTED_STREAM_KINDS else 0.4,
        "evidence": "recorder_metadata" if declared else "path_field_shape_heuristic",
    }
    for key in ("width", "height", "fps", "source_fps", "storage_fps", "sync_fps", "frame_count", "dtype", "unit", "channels", "node_count", "declared_by"):
        if declared.get(key) is not None:
            result[key] = declared[key]
    if dtype and "dtype" not in result:
        result["dtype"] = dtype
    return result


def _dimension_names(descriptor: dict, field: str | None, shape: list[int] | None) -> list[str]:
    if not shape or len(shape) < 2 or not isinstance(shape[-1], int):
        return []
    leaf = str(field or descriptor.get("role") or "value").replace("/", ".")
    if descriptor.get("role") == "hand_skeleton" and len(shape) == 3 and shape[-1] >= 3:
        return [f"node_{node:02d}.{axis}" for node in range(int(shape[1])) for axis in ("x", "y", "z")]
    width = min(512, int(shape[-1]))
    return [f"{leaf}[{index}]" for index in range(width)]


def build_local_schema_profile(inventory: dict, report: dict) -> dict:
    streams: list[dict] = []
    for candidate in inventory.get("candidate_streams", []):
        source_path = str(candidate.get("source_path") or "")
        field = candidate.get("field")
        shape = candidate.get("shape") if isinstance(candidate.get("shape"), list) else []
        descriptor = describe_source(source_path, str(field) if field is not None else None, shape, str(candidate.get("dtype") or ""), report)
        if descriptor["kind"] in _SUPPORTED_STREAM_KINDS:
            kind = descriptor["kind"]
        elif descriptor.get("evidence") == "recorder_metadata" and descriptor["kind"] in {"metadata", "other"}:
            # A recorder-declared metadata/validity container is authoritative.
            # Do not let a word such as `state` or `mocap` inside JSON promote
            # it to a robot joint stream.
            continue
        else:
            kind = str(candidate.get("kind") or "other")
        if kind not in _SUPPORTED_STREAM_KINDS:
            continue
        modality = descriptor["modality"] if descriptor["modality"] != "unknown" else str(candidate.get("modality") or "unknown")
        extraction = "skeleton_xyz" if descriptor.get("role") == "hand_skeleton" and len(shape) == 3 and shape[-1] >= 3 else ""
        representation = "absolute" if kind == "joint" and modality == "pose" else "unknown"
        streams.append({
            "source_id": str(candidate.get("id") or ""),
            "kind": kind,
            "modality": modality,
            "side": descriptor["side"] if descriptor["side"] != "unknown" else str(candidate.get("side_hint") or "unknown"),
            "role": descriptor["role"],
            "representation": representation,
            "dimension_names": _dimension_names(descriptor, str(field) if field is not None else None, shape),
            "gripper_indices": [],
            "embodiment_id": "dexweaveg1" if report.get("format_family") == "nexus_multimodal" and kind == "joint" else None,
            "confidence": descriptor["confidence"],
            "evidence": descriptor["evidence"],
            "source_path": source_path,
            "field": field,
            "shape": shape,
            "dtype": candidate.get("dtype"),
            "variant": descriptor["variant"],
            "synchronized": descriptor["synchronized"],
            "extraction": extraction,
        })

    by_id = {item["source_id"]: item for item in streams if item.get("source_id")}
    vision = [item for item in streams if item["kind"] == "vision"]
    joints = [item for item in streams if item["kind"] == "joint" and item.get("variant") != "raw"]
    sensors = [item for item in streams if item["kind"] == "sensor" and item.get("variant") != "raw"]
    timestamps = [item for item in streams if item["kind"] == "timestamp"]
    preferred_timestamp = next((item for item in timestamps if "sync" in _normal(item.get("source_path"))), timestamps[0] if timestamps else None)
    associations = []
    for visual in vision:
        side = str(visual.get("side") or "unknown")
        related_joints = [item["source_id"] for item in joints if side in {"unknown", "shared"} or item.get("side") in {side, "shared", "unknown"}]
        related_sensors = [item["source_id"] for item in sensors if side in {"unknown", "shared"} or item.get("side") in {side, "shared", "unknown"}]
        associations.append({
            "vision_id": visual["source_id"],
            "joint_ids": [item for item in related_joints if item in by_id],
            "sensor_ids": [item for item in related_sensors if item in by_id],
            "side": side,
            "time_alignment": "recorder_sync_table" if report.get("format_family") == "nexus_multimodal" else "frame_index_or_timestamp",
            "timestamp_id": preferred_timestamp.get("source_id") if preferred_timestamp else None,
            "confidence": 0.97 if report.get("format_family") == "nexus_multimodal" else 0.82,
            "reason": "Local format adapter kept modality and side boundaries explicit.",
        })

    warnings = [str(item.get("message") or "") for item in report.get("issues", []) if item.get("severity") in {"warning", "error"}]
    understanding = {
        "format_family": report.get("format_family") or "unknown",
        "format_confidence": float(report.get("format_confidence") or 0.0),
        "summary": f"Local read-only adapter recognized {len(vision)} vision, {len(joints)} joint, {len(sensors)} sensor, and {len(timestamps)} timestamp streams.",
        "episode_organization": str(report.get("episode_layout") or "unknown"),
        "streams": streams,
        "associations": associations,
        "processing_strategy": dict(report.get("processing_strategy") or {}),
        "capabilities": dict(report.get("capabilities") or {}),
    }
    completed = bool((report.get("capabilities") or {}).get("can_import") and streams)
    return {
        "status": "completed" if completed else "awaiting_vlm",
        "inventory": inventory,
        "understanding": understanding if completed else None,
        "warnings": warnings or (["Local format adapter could not establish a safe canonical map."] if not completed else []),
        "provider": {"kind": "local_format_adapter", "schema": FORMAT_MAP_SCHEMA, "requires_api": False},
        "error": None,
        "updated_at": _utc_now(),
    }


def write_format_report(path: str | Path, report: dict) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return target


def _print_human(report: dict) -> None:
    print(f"format: {report['format_family']} ({float(report['format_confidence']) * 100:.0f}%)")
    print(f"status: {report['status']}")
    print(f"root_mode: {report['root_mode']}")
    print(f"episodes: {report.get('episode_count_hint') or 'unknown'}")
    print("modalities: " + ", ".join(f"{key}={value}" for key, value in sorted((report.get("modality_counts") or {}).items())))
    for issue in report.get("issues", []):
        print(f"[{str(issue.get('severity') or 'info').upper()}] {issue.get('message')}")


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect and canonicalize a dataset format without rewriting source files")
    parser.add_argument("path")
    parser.add_argument("--output", help="Optional JSON report path")
    parser.add_argument("--json", action="store_true", help="Print the full JSON report")
    parser.add_argument("--require-ready", action="store_true", help="Return non-zero for warning/blocked reports")
    args = parser.parse_args(argv)
    report = inspect_dataset_format(args.path)
    if args.output:
        write_format_report(args.output, report)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_human(report)
    if report["status"] == "blocked" or (args.require_ready and report["status"] != "ready"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
