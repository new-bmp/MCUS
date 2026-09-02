from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

from .action_mapping import generate_episode_action, load_episode_action_mapping, validate_episode_action_mapping
from .behavior_annotator import annotate_episode_behavior, load_behavior_annotation, media_fingerprint_matches, snapshot_behavior_annotation_for_run
from .dataset_modes import dataset_mode
from .full_export import EPISODE_LEROBOT_JSON_OUTPUT_FORMAT, SUBTASK_JSON_OUTPUT_FORMAT, _find_transform_source, export_episode, write_dataset_index
from .full_run import (
    artifact_record,
    finalize_full_run,
    full_run_stage_dir,
    publish_full_run_episode,
    start_full_run,
    update_full_run_episode,
    write_full_timeline_lock,
    write_stamped_artifact,
)
from .hand_visibility import external_hand_projection_calibration, inspect_full_hand_visibility
from .job_control import CancellableJobMixin, JobCancelled
from .models import registry
from .nexus_mano import mano21_points_from_nexus_value
from .openxr_mano import mano21_points_from_openxr_value
from .projection_correction import (
    FULL_PROJECTION_SOURCE_OVERRIDES,
    PROJECTION_RUNTIME_LOCK,
    preferred_projection_media,
    projection_source_from_document,
    run_projection_correction,
)
from .quality_evidence import build_quality_evidence
from .schema_profiler import infer_local_signal_fields, probe_local_signal_fields
from .schemas import ActionMappingRequest, BehaviorAnnotationRequest, CurationJobRequest, HandPoseModelConfig
from .s1_repair import S1_REPAIR_SCHEMA, load_s1_repair
from .sensor_alignment import (
    find_sensor_alignment_stream,
    retime_sensor_alignment,
    scan_episode_sensor_alignment,
    validate_episode_time_sync,
)
from .storage import dataset_artifact_dir, episode_media, get_manifest, record_change, slugify, storage_slug
from .video_smoothing import smooth_video


CURATION_SCHEMA = "alice/paper-curation/v1"
CURATION_PIPELINE_VERSION = 19
FULL_EGODEX_PROJECTION_ADJUSTMENT_RATE = 0.65
FULL_EGODEX_PROJECTION_SAMPLE_FPS = 15.0
QUALITY_MARK_GAP_SECONDS = 0.3
ROT6D_ABSOLUTE_JUMP_DEGREES = 45.0
NEXUS_TACTILE_SPIKE_MAX_FRAMES = 2
NEXUS_TACTILE_MIN_RELATIVE_EXCURSION = 0.6
S2_LOCAL_WINDOW_SECONDS = 0.5
S2_LOCAL_MINIMUM_SPAN_SECONDS = 0.15
MAX_SIGNAL_ROWS = 120_000
# Two native 20-node hands require 120 XYZ dimensions.  Keep bounded headroom
# for wrists/actions without silently dropping the second hand.
MAX_SIGNAL_DIMS = 192
MAX_REPORT_FINDINGS = 2_000
NUMERIC_EXTENSIONS = {".h5", ".hdf5", ".h5df", ".parquet", ".npy", ".npz", ".json", ".jsonl", ".csv", ".tsv"}

STAGE_DEFINITIONS = [
    ("t0", "统一时间轴"),
    ("p1", "Nexus 压力完整性"),
    ("s1", "突变与 Jerk"),
    ("s2", "State-Action 对齐与导出一致性"),
    ("s3", "分位极值"),
    ("s4", "FK 一致性"),
    ("s5", "基座与方向统一"),
    ("c3", "视频质量与整手可见"),
    ("vlm", "非红片段行为标注"),
    ("c1", "指令一致性"),
    ("c2", "视频-State 一致性"),
]

NEXUS_PRESSURE_EXPECTED_SIDES = ("left", "right")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def curation_report_path(dataset_id: str, episode_id: str, media_file_id: str | None = None, run_id: str | None = None) -> Path:
    stem = storage_slug(episode_id)
    if media_file_id:
        stem = f"{stem}--media-{storage_slug(str(media_file_id))}"
    root = full_run_stage_dir(dataset_id, run_id, episode_id, "curation") if run_id else dataset_artifact_dir(dataset_id, "curation")
    return root / f"{stem}.curation.alice"


def s1_repair_path(dataset_id: str, episode_id: str, media_file_id: str | None = None, run_id: str | None = None) -> Path:
    stem = storage_slug(episode_id)
    if media_file_id:
        stem = f"{stem}--media-{storage_slug(str(media_file_id))}"
    root = full_run_stage_dir(dataset_id, run_id, episode_id, "curation") if run_id else dataset_artifact_dir(dataset_id, "curation-repairs")
    return root / f"{stem}.s1-repair.alice"


def _write_curation_report(dataset_id: str, episode_id: str, media_file_id: str | None, payload: dict, run_id: str | None = None) -> Path:
    """Persist one report per media stream and retain the episode alias for old clients."""
    exact = curation_report_path(dataset_id, episode_id, media_file_id, run_id)
    _write_json_atomic(exact, payload)
    if run_id:
        return exact
    alias = curation_report_path(dataset_id, episode_id)
    if alias != exact:
        _write_json_atomic(alias, payload)
    return exact


def load_curation_report(dataset_id: str, episode_id: str, media_file_id: str | None = None, run_id: str | None = None) -> dict | None:
    preferred = curation_report_path(dataset_id, episode_id, media_file_id, run_id)
    if run_id:
        candidates = [preferred]
    else:
        alias = curation_report_path(dataset_id, episode_id)
        legacy = alias.with_name(f"{slugify(episode_id)}.curation.alice")
        candidates = list(dict.fromkeys((preferred, alias, legacy)))
    alias = curation_report_path(dataset_id, episode_id)
    if media_file_id is None and not run_id:
        prefix = f"{storage_slug(episode_id)}--media-"
        archived = sorted(
            alias.parent.glob(f"{prefix}*.curation.alice"),
            key=lambda item: item.stat().st_mtime_ns,
            reverse=True,
        )
        candidates.extend(path for path in archived if path not in candidates)
    payload = None
    path = None
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("schema") != CURATION_SCHEMA:
            continue
        if run_id and str(value.get("full_run_id") or "") != str(run_id):
            continue
        if str(value.get("dataset_id") or "") != str(dataset_id) or str(value.get("episode_id") or "") != str(episode_id):
            if candidate == preferred:
                raise ValueError("Paper curation report identity does not match the requested Dataset/Episode")
            continue
        stored_media_id = str((value.get("source_video") or {}).get("file_id") or "")
        if media_file_id and stored_media_id and stored_media_id != str(media_file_id):
            continue
        payload = value
        path = candidate
        break
    if payload is None or path is None:
        return None
    signatures = payload.get("source_signatures") or []
    if not signatures:
        raise ValueError("Paper curation report has no source-version lock; rerun curation")
    manifest = get_manifest(dataset_id)
    root = Path(manifest["root_path"]).expanduser().resolve()
    allowed_paths = [str(item.get("relative_path") or "") for item in manifest.get("files", [])]
    matches, changed = source_signatures_match(root, signatures, allowed_paths=allowed_paths)
    if not matches:
        raise ValueError(f"Paper curation report is stale: {changed or 'unknown source'}")
    try:
        pipeline_version = int(payload.get("pipeline_version") or 1)
    except (TypeError, ValueError):
        pipeline_version = 0
    payload["pipeline_version"] = pipeline_version
    payload["requires_rerun"] = pipeline_version != CURATION_PIPELINE_VERSION
    payload["artifact_path"] = str(path)
    return payload


def _episode_records(manifest: dict, episode: dict) -> list[dict]:
    episode_id = str(episode.get("id"))
    assignments = (manifest.get("episode_resolution") or {}).get("file_episode_assignments") or {}
    return [
        record
        for record in manifest.get("files", [])
        if str(assignments.get(str(record.get("id"))) or record.get("episode_id") or "") == episode_id
    ]


def _sample_digest(path: Path) -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    with path.open("rb") as source:
        digest.update(source.read(1024 * 1024))
        if size > 1024 * 1024:
            source.seek(max(0, size - 1024 * 1024))
            digest.update(source.read(1024 * 1024))
    return digest.hexdigest()


def _directory_signature(path: Path) -> dict:
    root = path.resolve()
    files = []
    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        files.append(resolved)
    files.sort(key=lambda item: item.relative_to(root).as_posix().casefold())
    digest = hashlib.sha256()
    total_size = 0
    latest_mtime = int(root.stat().st_mtime_ns)
    for item in files:
        stat = item.stat()
        relative = item.relative_to(root).as_posix()
        total_size += int(stat.st_size)
        latest_mtime = max(latest_mtime, int(stat.st_mtime_ns))
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    if files:
        sample_count = min(16, len(files))
        sample_indices = sorted({round(index * (len(files) - 1) / max(1, sample_count - 1)) for index in range(sample_count)})
        for index in sample_indices:
            item = files[index]
            digest.update(item.relative_to(root).as_posix().encode("utf-8"))
            digest.update(_sample_digest(item).encode("ascii"))
    return {
        "kind": "directory",
        "file_count": len(files),
        "size_bytes": total_size,
        "mtime_ns": latest_mtime,
        "sample_sha256": digest.hexdigest(),
    }


def source_signature(root: Path, relative_path: str) -> dict:
    normalized = str(relative_path).replace("\\", "/")
    root = root.expanduser().resolve()
    path = (root / normalized).resolve()
    path.relative_to(root)
    if path.is_dir():
        return {"relative_path": normalized, **_directory_signature(path)}
    stat = path.stat()
    return {
        "relative_path": normalized,
        "kind": "file",
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sample_sha256": _sample_digest(path),
    }


def source_signatures_match(root: Path, signatures: list[dict], allowed_paths: list[str] | None = None) -> tuple[bool, str | None]:
    allowed = {str(value).replace("\\", "/").strip("/") for value in (allowed_paths or []) if str(value).strip("/\\")}
    for expected in signatures:
        relative = str(expected.get("relative_path") or "").replace("\\", "/").strip("/")
        if allowed:
            expected_kind = str(expected.get("kind") or "file")
            present = relative in allowed if expected_kind != "directory" else any(value == relative or value.startswith(f"{relative}/") for value in allowed)
            if not present:
                return False, relative
        try:
            current = source_signature(root, relative)
        except (OSError, ValueError):
            return False, relative
        keys = ["size_bytes", "mtime_ns", "sample_sha256"]
        if expected.get("kind") is not None:
            keys.append("kind")
        if expected.get("kind") == "directory":
            keys.append("file_count")
        for key in keys:
            if current.get(key) != expected.get(key):
                return False, relative
    return True, None


@lru_cache(maxsize=512)
def _cached_local_signal_fields(path_value: str, size_bytes: int, mtime_ns: int) -> tuple[dict, ...]:
    del size_bytes, mtime_ns
    return tuple(probe_local_signal_fields(Path(path_value), max_dimensions=MAX_SIGNAL_DIMS))


def _local_signal_fields(path: Path, inventory_fields: list[dict]) -> list[dict]:
    inferred = infer_local_signal_fields(inventory_fields, max_dimensions=MAX_SIGNAL_DIMS)
    if path.suffix.casefold() in {".h5", ".hdf5", ".h5df"} and path.is_file():
        stat = path.stat()
        inferred.extend(
            dict(item)
            for item in _cached_local_signal_fields(str(path), int(stat.st_size), int(stat.st_mtime_ns))
        )
    deduplicated: dict[tuple[str, str], dict] = {}
    for item in inferred:
        key = (str(item.get("kind") or ""), str(item.get("field") or "").casefold())
        if key not in deduplicated or float(item.get("confidence") or 0.0) > float(deduplicated[key].get("confidence") or 0.0):
            deduplicated[key] = item
    return list(deduplicated.values())


def _stream_path_for_episode(manifest: dict, record_paths: dict[str, dict], source_path: str) -> str | None:
    """Resolve a Qwen stream example to the equivalent file in another Episode."""
    normalized = str(source_path or "").replace("\\", "/").strip("/")
    if not normalized:
        return None
    exact = next((path for path in record_paths if path.casefold() == normalized.casefold()), None)
    if exact:
        return exact

    manifest_records = {
        str(item.get("relative_path") or "").replace("\\", "/").strip("/").casefold(): item
        for item in manifest.get("files", [])
    }
    source_record = manifest_records.get(normalized.casefold()) or {}
    suffixes: list[str] = []
    episode_key = str(source_record.get("episode_key") or "").replace("\\", "/").strip("/")
    if episode_key and normalized.casefold().startswith(f"{episode_key.casefold()}/"):
        suffixes.append(normalized[len(episode_key) + 1:])
    parts = [part for part in normalized.split("/") if part]
    for count in (3, 2, 1):
        if len(parts) >= count:
            suffixes.append("/".join(parts[-count:]))

    for suffix in dict.fromkeys(item for item in suffixes if item):
        folded = suffix.casefold()
        matches = [
            path
            for path in record_paths
            if path.casefold() == folded or path.casefold().endswith(f"/{folded}")
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def _signal_candidates(manifest: dict, episode: dict) -> list[dict]:
    records = _episode_records(manifest, episode)
    record_paths = {str(item.get("relative_path") or "").replace("\\", "/"): item for item in records}
    candidates: list[dict] = []
    understanding = (manifest.get("schema_profile") or {}).get("understanding") or {}
    for stream in understanding.get("streams", []):
        kind = str(stream.get("kind") or "")
        source_relative = str(stream.get("source_path") or "").replace("\\", "/")
        relative = _stream_path_for_episode(manifest, record_paths, source_relative)
        field = stream.get("field")
        if kind not in {"joint", "action"} or not relative or not field:
            continue
        candidates.append({
            "kind": kind,
            "relative_path": relative,
            "field": str(field),
            "confidence": float(stream.get("confidence") or 0.5),
            "role": str(stream.get("role") or ""),
            "modality": str(stream.get("modality") or ""),
            "representation": str(stream.get("representation") or "unknown"),
            "dimension_names": [str(item) for item in stream.get("dimension_names", [])],
            "gripper_indices": [int(item) for item in stream.get("gripper_indices", []) if isinstance(item, int) or str(item).isdigit()],
            "embodiment_id": stream.get("embodiment_id"),
            "side": str(stream.get("side") or (record_paths.get(relative) or {}).get("side") or "unknown"),
            "variant": str(
                stream.get("variant")
                or (record_paths.get(relative) or {}).get("variant")
                or ("raw" if re.search(r"_raw(?=\.[^./]+$)", relative, flags=re.IGNORECASE) else "primary")
            ),
            "extraction": str(stream.get("extraction") or ""),
            "node_count": stream.get("node_count"),
            "source": "qwen_schema" if relative.casefold() == source_relative.casefold() else "qwen_schema_template",
            "schema_example_path": source_relative,
        })
    inventory_files = {
        str(item.get("path") or "").replace("\\", "/").casefold(): item
        for item in ((manifest.get("schema_profile") or {}).get("inventory") or {}).get("files", [])
    }
    root_value = str(manifest.get("root_path") or "").strip()
    root = Path(root_value).expanduser().resolve() if root_value else None
    for relative, record in record_paths.items():
        suffix = str(record.get("extension") or Path(relative).suffix).casefold()
        if suffix not in NUMERIC_EXTENSIONS:
            continue
        inventory_fields = list((inventory_files.get(relative.casefold()) or {}).get("fields", []))
        path = (root / relative).resolve() if root is not None else Path()
        if root is not None:
            try:
                path.relative_to(root)
            except ValueError:
                continue
        try:
            local_fields = _local_signal_fields(path, inventory_fields)
        except (OSError, ValueError):
            local_fields = infer_local_signal_fields(inventory_fields, max_dimensions=MAX_SIGNAL_DIMS)
        for field in local_fields:
            kind = str(field.get("kind") or "")
            field_name = str(field.get("field") or "")
            if kind not in {"joint", "action"} or not field_name:
                continue
            candidates.append({
                "kind": kind,
                "relative_path": relative,
                "field": field_name,
                "confidence": float(field.get("confidence") or 0.75),
                "role": str(field.get("role") or ""),
                "modality": str(field.get("modality") or ""),
                "representation": str(field.get("representation") or "unknown"),
                "dimension_names": [str(item) for item in field.get("dimension_names", [])],
                "gripper_indices": [int(item) for item in field.get("gripper_indices", [])],
                "embodiment_id": field.get("embodiment_id"),
                "side": str(
                    field.get("side_hint")
                    if str(field.get("side_hint") or "unknown") != "unknown"
                    else record.get("side") or "unknown"
                ),
                "members": [str(item) for item in field.get("members", [])],
                "extraction": str(field.get("extraction") or ""),
                "node_count": field.get("node_count"),
                "variant": str(
                    record.get("variant")
                    or ("raw" if re.search(r"_raw(?=\.[^./]+$)", relative, flags=re.IGNORECASE) else "primary")
                ),
                "source": "local_schema",
            })
    for candidate in candidates:
        relative = str(candidate.get("relative_path") or "").replace("\\", "/").casefold()
        field_leaf = str(candidate.get("field") or "").replace("\\", "/").rsplit("/", 1)[-1].casefold()
        if (
            "dexweaveg1" in relative
            and field_leaf == "skeleton"
            and int(candidate.get("node_count") or 20) == 20
        ):
            candidate["extraction"] = "nexus_dexweaveg1_20_to_mano21"
            candidate["embodiment_id"] = "dexweaveg1"
            candidate["dimension_names"] = [
                f"mano21_{node:02d}.{axis}"
                for node in range(21)
                for axis in ("x", "y", "z")
            ]
    aggregate_prefixes = {
        (item["relative_path"].casefold(), item["field"][:-1].casefold())
        for item in candidates
        if item.get("source") == "local_schema" and str(item.get("field") or "").endswith("/*")
    }
    candidates = [
        item
        for item in candidates
        if str(item.get("field") or "").endswith("/*")
        or not any(
            item["relative_path"].casefold() == relative and item["field"].casefold().startswith(prefix)
            for relative, prefix in aggregate_prefixes
        )
    ]
    native_skeleton_sources = {
        str(item.get("relative_path") or "").casefold()
        for item in candidates
        if str(item.get("extraction") or "") == "skeleton_xyz"
        or str(item.get("role") or "") == "hand_skeleton"
        and str(item.get("field") or "").replace("\\", "/").rsplit("/", 1)[-1].casefold() == "skeleton"
    }
    candidates = [
        item for item in candidates
        if not (
            str(item.get("relative_path") or "").casefold() in native_skeleton_sources
            and (
                str(item.get("role") or "") == "hand_orientation_auxiliary"
                or str(item.get("field") or "").replace("\\", "/").rsplit("/", 1)[-1].casefold() == "wrist_quat"
            )
        )
    ]
    # Prefer recorder-synchronized streams for cleaning.  Raw streams remain
    # indexed and available for audit, but do not get concatenated beside an
    # equivalent synchronized stream (which would double-count the same hand
    # at a different sample rate).
    def synchronized_key(item: dict) -> tuple[str, str, str]:
        relative = re.sub(
            r"_raw(?=\.[^./]+$)",
            "",
            str(item.get("relative_path") or "").replace("\\", "/"),
            flags=re.IGNORECASE,
        ).casefold()
        return str(item.get("kind") or ""), relative, str(item.get("field") or "").casefold()

    synchronized_sources = {
        synchronized_key(item)
        for item in candidates
        if str(item.get("variant") or "primary") != "raw"
    }
    candidates = [
        item for item in candidates
        if str(item.get("variant") or "primary") != "raw" or synchronized_key(item) not in synchronized_sources
    ]
    # Once a reviewed AlicePose correction is applied, S1 and the remaining
    # cleaning stages read its video-aligned 3D transforms.  The original
    # relative path remains the binding/source-signature identity so existing
    # repair and audit contracts continue to protect the immutable source.
    from .projection_correction import active_projection_source

    applied = active_projection_source(manifest, episode)
    if applied is not None:
        source_relative = str(applied.get("source_relative_path") or "").replace("\\", "/").casefold()
        if source_relative:
            for candidate in candidates:
                if candidate.get("kind") == "joint" and str(candidate.get("relative_path") or "").replace("\\", "/").casefold() == source_relative:
                    candidate["absolute_path"] = str(applied["path"])
                    candidate["derived_artifact_path"] = str(applied["path"])
                    candidate["source"] = (
                        "full_run_projection_correction"
                        if applied.get("activation_scope") == "full_run"
                        else "applied_projection_correction"
                    )
                    candidate["application_id"] = applied.get("application_id")
    deduplicated: dict[tuple[str, str, str], dict] = {}
    for candidate in candidates:
        key = (candidate["kind"], candidate["relative_path"].casefold(), candidate["field"].casefold())
        if key not in deduplicated or candidate["confidence"] > deduplicated[key]["confidence"]:
            deduplicated[key] = candidate
    return sorted(
        deduplicated.values(),
        key=lambda item: (
            str(item.get("variant") or "primary") == "raw",
            -item["confidence"],
            item["relative_path"],
            item["field"],
        ),
    )


def _nexus_mode_enabled(manifest: dict) -> bool:
    return dataset_mode(manifest)["family"] == "nexus_multimodal"


def _hand_visibility_capability(manifest: dict) -> tuple[bool, str | None]:
    """Apply the dataset-mode-specific camera contract for C3."""
    mode = dataset_mode(manifest)
    family = str(mode["family"])
    if mode["conflict"]:
        return False, "数据集格式声明互相冲突，C3 已安全停用"
    if mode["hand_visibility_backend"] == "egodex_embedded_camera_v1":
        return True, None
    if family in {"nexus_multimodal", "openxr"}:
        calibration, reason = external_hand_projection_calibration(manifest)
        if calibration is None:
            label = "Nexus" if family == "nexus_multimodal" else "OpenXR"
            return False, f"{label} 缺少严格有效的手部空间到 RGB 标定：{reason}"
        return True, None
    if family in {"lerobot", "alice_full"}:
        return False, "当前格式未声明可验证的 MANO21 到 RGB 投影契约"
    return False, "未识别数据集的关节坐标系与相机关系，C3 已安全停用"

def _normalized_hand_sides(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = [value]
    sides: list[str] = []
    for item in values:
        side = str(item or "").casefold().strip()
        if side in {"left", "right"} and side not in sides:
            sides.append(side)
    return sides


def _hand_side_text_hint(value: object) -> list[str]:
    text = str(value or "").casefold().replace("\\", "/")
    if not text:
        return []
    left = bool(re.search(r"(?<![a-z])(?:left|l_hand|l_arm)(?![a-z])", text))
    right = bool(re.search(r"(?<![a-z])(?:right|r_hand|r_arm)(?![a-z])", text))
    return [side for side, present in (("left", left), ("right", right)) if present]


def _hdf5_hand_side_hint(path: Path) -> list[str]:
    """Read only small HDF5 attributes that declare the active hand.

    EgoDex files commonly keep task metadata such as ``hand=right`` on the
    file (or the transforms group).  This is a bounded metadata read and does
    not load any trajectory samples.
    """
    try:
        import h5py

        with h5py.File(path, "r") as source:
            objects = [source]
            for name in ("transforms", "metadata", "task"):
                value = source.get(name)
                if value is not None:
                    objects.append(value)
            for obj in objects:
                for key, value in obj.attrs.items():
                    key_text = str(key).casefold()
                    if not (
                        any(token in key_text for token in ("hand", "side"))
                        or key_text in {"object", "task", "description", "llm_description", "llm_description2"}
                    ):
                        continue
                    sides = _normalized_hand_sides(value)
                    if sides:
                        return sides
                    sides = _hand_side_text_hint(value)
                    if sides:
                        return sides
    except (OSError, ValueError, ImportError):
        return []
    return []


def _required_hand_sides(
    manifest: dict,
    episode: dict,
    generated_s2: dict | None = None,
    generated_action_report: dict | None = None,
) -> list[str]:
    """Infer only explicitly supported hand sides for C3.

    An absent Action stream is not evidence that both hands are required.  If
    no trustworthy side is declared, return an empty list so C3 can report the
    limitation without manufacturing an all-frame rejection mask.
    """
    sides = _normalized_hand_sides((generated_s2 or {}).get("required_sides"))
    if sides:
        return sides

    report = generated_action_report or {}
    config = report.get("config") or {}
    profile = report.get("profile") or {}
    profile_sides = int(profile.get("sides") or 0) if str(profile.get("sides") or "").isdigit() else 0
    if profile_sides == 1:
        sides = _normalized_hand_sides(config.get("source_hand"))
        if sides:
            return sides

    candidates = _signal_candidates(manifest, episode)
    explicit: list[str] = []
    for item in candidates:
        for value in (item.get("side"), item.get("side_hint"), item.get("source_hand")):
            explicit.extend(_normalized_hand_sides(value))
        if not explicit:
            for value in (item.get("relative_path"), item.get("field"), *(item.get("dimension_names") or [])):
                explicit.extend(_hand_side_text_hint(value))
    explicit = list(dict.fromkeys(explicit))
    if explicit:
        return explicit

    for value in (episode.get("side"), episode.get("hand"), episode.get("hand_side")):
        sides = _normalized_hand_sides(value) or _hand_side_text_hint(value)
        if sides:
            return sides

    try:
        _, source_relative, _ = _find_transform_source(manifest, episode, prefer_applied=False)
        source_path = (Path(manifest["root_path"]).expanduser().resolve() / source_relative).resolve()
        return _hdf5_hand_side_hint(source_path)
    except (KeyError, OSError, RuntimeError, ValueError):
        return []


def _hand_visibility_skipped(frame_count: int, message: str, required_sides: list[str] | None = None) -> dict:
    sides = list(required_sides or [])
    return {
        "available": False,
        "skipped": True,
        "message": message,
        "invalid_mask": np.zeros(max(0, int(frame_count)), dtype=bool),
        "review_mask": np.zeros(max(0, int(frame_count)), dtype=bool),
        "metrics": {
            "available": False,
            "skipped": True,
            "required_sides": sides,
            "message": message,
        },
    }


def _nexus_pressure_side(record: dict) -> str:
    declared = str(record.get("side") or "").casefold()
    if declared in NEXUS_PRESSURE_EXPECTED_SIDES:
        return declared
    relative = str(record.get("relative_path") or "").replace("\\", "/").casefold()
    if any(token in relative for token in ("/left.", "/left_", "_left.", "left_pressure", "tactile/left")):
        return "left"
    if any(token in relative for token in ("/right.", "/right_", "_right.", "right_pressure", "tactile/right")):
        return "right"
    return "unknown"


def _nexus_pressure_records(manifest: dict, episode: dict) -> list[dict]:
    candidates: list[dict] = []
    for record in _episode_records(manifest, episode):
        relative = str(record.get("relative_path") or "").replace("\\", "/")
        text = f"{relative} {record.get('modality') or ''} {record.get('canonical_kind') or ''}".casefold()
        suffix = str(record.get("extension") or Path(relative).suffix).casefold()
        if suffix not in {".h5", ".hdf5", ".h5df"}:
            continue
        if not any(token in text for token in ("tactile", "pressure", "force_sensor", "contact_sensor")):
            continue
        raw = str(record.get("variant") or "").casefold() == "raw" or bool(re.search(r"_raw(?=\.[^./]+$)", relative, flags=re.IGNORECASE))
        candidates.append({**record, "relative_path": relative, "side": _nexus_pressure_side(record), "_raw": raw})
    selected: list[dict] = []
    for side in NEXUS_PRESSURE_EXPECTED_SIDES:
        matches = [item for item in candidates if item["side"] == side]
        if matches:
            selected.append(sorted(matches, key=lambda item: (item["_raw"], len(item["relative_path"]), item["relative_path"]))[0])
    unknown = [item for item in candidates if item["side"] == "unknown"]
    selected.extend(sorted(unknown, key=lambda item: (item["_raw"], item["relative_path"])))
    return selected


def _pressure_dataset_path(handle) -> str | None:
    import h5py

    candidates: list[tuple[int, str]] = []

    def visitor(name: str, value) -> None:
        if not isinstance(value, h5py.Dataset) or not value.shape or int(value.shape[0]) < 0:
            return
        leaf = name.replace("\\", "/").rsplit("/", 1)[-1].casefold()
        if leaf in {"partial", "source_seq", "frame_idx", "timestamps", "sensor_ts", "host_arrival_ts"}:
            return
        if leaf == "adc":
            score = 0
        elif any(token in name.casefold() for token in ("pressure", "tactile", "force")):
            score = 1
        else:
            return
        candidates.append((score, name))

    handle.visititems(visitor)
    return min(candidates, default=(99, None))[1]


def _pressure_source_validity(path: Path, *, include_features: bool = False) -> dict:
    import h5py

    with h5py.File(path, "r") as handle:
        field = _pressure_dataset_path(handle)
        if field is None:
            raise ValueError("pressure/tactile value field is missing")
        dataset = handle[field]
        count = int(dataset.shape[0])
        if count <= 0 or int(np.prod(dataset.shape[1:], dtype=np.int64)) <= 0:
            return {"field": field, "source_count": count, "valid_rows": np.zeros(max(0, count), dtype=bool)}
        valid_rows = np.ones(count, dtype=bool)
        features = np.zeros((count, 4), dtype=np.float64) if include_features else None
        if include_features and dataset.dtype.kind not in "biufc":
            raise ValueError("pressure/tactile values are not numeric")
        if dataset.dtype.kind in "biufc":
            chunk_size = max(1, min(4096, count))
            for start in range(0, count, chunk_size):
                end = min(count, start + chunk_size)
                values = np.asarray(dataset[start:end], dtype=np.float64).reshape(end - start, -1)
                if dataset.dtype.kind in "fc":
                    valid_rows[start:end] &= np.isfinite(values).any(axis=1)
                if features is not None:
                    finite_values = np.where(np.isfinite(values), values, 0.0)
                    active = finite_values > 0.0
                    totals = np.sum(finite_values, axis=1)
                    active_counts = np.sum(active, axis=1)
                    maximum = np.max(finite_values, axis=1)
                    active_mean = np.divide(
                        totals,
                        np.maximum(active_counts, 1),
                        out=np.zeros(end - start, dtype=np.float64),
                        where=active_counts > 0,
                    )
                    features[start:end] = np.column_stack((totals, active_counts, maximum, active_mean))
        elif dataset.dtype.kind in "OSU":
            chunk_size = max(1, min(4096, count))
            for start in range(0, count, chunk_size):
                end = min(count, start + chunk_size)
                values = np.asarray(dataset[start:end]).reshape(end - start, -1)
                present = np.asarray([
                    any(item is not None and str(item).strip() for item in row)
                    for row in values
                ], dtype=bool)
                valid_rows[start:end] &= present
        parent = field.rsplit("/", 1)[0] if "/" in field else ""
        for name in dict.fromkeys(filter(None, (
            f"{parent}/partial" if parent else "partial",
            "partial",
        ))):
            partial = handle.get(name)
            if isinstance(partial, h5py.Dataset) and partial.shape == (count,):
                valid_rows &= ~np.asarray(partial[()], dtype=bool).reshape(-1)
                break
        for leaf in ("source_seq", "frame_idx"):
            for name in dict.fromkeys(filter(None, (f"{parent}/{leaf}" if parent else leaf, leaf))):
                source_index = handle.get(name)
                if isinstance(source_index, h5py.Dataset) and source_index.shape == (count,):
                    values = np.asarray(source_index[()]).reshape(-1)
                    if values.dtype.kind in "iu":
                        valid_rows &= values >= 0
                    break
        for leaf in ("sensor_ts", "timestamps"):
            for name in dict.fromkeys(filter(None, (f"{parent}/{leaf}" if parent else leaf, leaf))):
                timestamp = handle.get(name)
                if isinstance(timestamp, h5py.Dataset) and timestamp.shape == (count,):
                    values = np.asarray(timestamp[()]).reshape(-1)
                    if values.dtype.kind in "fc":
                        valid_rows &= np.isfinite(values)
                    break
        return {
            "field": field,
            "source_count": count,
            "valid_rows": valid_rows,
            "features": features,
            "feature_names": ["pressure_sum", "active_taxel_count", "pressure_max", "active_pressure_mean"] if features is not None else [],
        }


def load_nexus_tactile_evidence(
    manifest: dict,
    episode: dict,
    alignment: dict,
    frame_count: int,
) -> dict:
    """Load video-aligned Nexus pressure features for post-VLM contact checks."""

    total = max(0, int(frame_count))
    empty = np.zeros(total, dtype=bool)
    if not _nexus_mode_enabled(manifest):
        return {
            "enabled": False,
            "contact": empty,
            "valid_mask": empty,
            "side_contact": {},
            "side_valid_mask": {},
            "side_features": {},
            "bindings": [],
            "errors": [],
        }
    root = Path(manifest["root_path"]).expanduser().resolve()
    side_contact: dict[str, np.ndarray] = {}
    side_valid: dict[str, np.ndarray] = {}
    side_features: dict[str, np.ndarray] = {}
    bindings: list[dict] = []
    errors: list[str] = []
    for record in _nexus_pressure_records(manifest, episode):
        side = str(record.get("side") or "unknown")
        relative = str(record.get("relative_path") or "")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
            source = _pressure_source_validity(path, include_features=True)
            source_count = int(source.get("source_count") or 0)
            features = source.get("features")
            if source_count <= 0 or features is None:
                raise ValueError("pressure/tactile feature rows are empty")
            aligned, _, aligned_valid = _resample_to_video(
                {
                    "values": features,
                    "row_indices": np.arange(source_count, dtype=np.int64),
                    "source_count": source_count,
                    "valid_rows": source["valid_rows"],
                },
                total,
                _alignment_stream(
                    alignment,
                    relative,
                    field=str(source.get("field") or "") or None,
                    source_count=source_count,
                ),
            )
            aligned = np.asarray(aligned, dtype=np.float64)
            valid = np.asarray(aligned_valid, dtype=bool)
            contact = (
                valid
                & np.isfinite(aligned[:, 0])
                & np.isfinite(aligned[:, 2])
                & (aligned[:, 0] > 0.0)
                & (aligned[:, 2] > 0.0)
            )
            side_features[side] = aligned
            side_valid[side] = valid
            side_contact[side] = contact
            bindings.append({
                "side": side,
                "status": "complete",
                "relative_path": relative,
                "field": source.get("field"),
                "source_rows": source_count,
                "feature_names": list(source.get("feature_names") or []),
                "contact_frame_count": int(contact.sum()),
            })
        except Exception as exc:
            errors.append(f"{side}: {str(exc)[:160]}")
            bindings.append({
                "side": side,
                "status": "unreadable_stream",
                "relative_path": relative,
                "error": str(exc)[:180],
            })
    contact = np.logical_or.reduce(list(side_contact.values())) if side_contact else empty.copy()
    valid_mask = np.logical_or.reduce(list(side_valid.values())) if side_valid else empty.copy()
    return {
        "enabled": bool(side_contact),
        "contact": contact,
        "valid_mask": valid_mask,
        "side_contact": side_contact,
        "side_valid_mask": side_valid,
        "side_features": side_features,
        "bindings": bindings,
        "errors": errors,
    }


def inspect_nexus_pressure_integrity(manifest: dict, episode: dict, alignment: dict, frame_count: int) -> dict:
    empty_mask = np.zeros(frame_count, dtype=bool)
    side_masks = {side: np.zeros(frame_count, dtype=bool) for side in NEXUS_PRESSURE_EXPECTED_SIDES}
    if not _nexus_mode_enabled(manifest):
        return {
            "enabled": False,
            "empty_mask": empty_mask,
            "side_masks": side_masks,
            "status": "skipped",
            "message": "仅在 Nexus 多传感器模式检查压力空值",
            "metrics": {"zero_is_valid": True, "empty_frame_count": 0},
            "bindings": [],
        }
    root = Path(manifest["root_path"]).expanduser().resolve()
    records = _nexus_pressure_records(manifest, episode)
    by_side = {str(item.get("side") or "unknown"): item for item in records}
    bindings: list[dict] = []
    for side in NEXUS_PRESSURE_EXPECTED_SIDES:
        record = by_side.get(side)
        if record is None:
            side_masks[side][:] = True
            bindings.append({"side": side, "status": "missing_stream", "relative_path": None})
            continue
        relative = str(record["relative_path"])
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
            source = _pressure_source_validity(path)
            source_count = int(source["source_count"])
            if source_count <= 0:
                aligned_valid = np.zeros(frame_count, dtype=bool)
            else:
                series = {
                    "values": np.zeros((source_count, 1), dtype=np.float64),
                    "row_indices": np.arange(source_count, dtype=np.int64),
                    "source_count": source_count,
                    "valid_rows": source["valid_rows"],
                }
                _, _, aligned_valid = _resample_to_video(
                    series,
                    frame_count,
                    _alignment_stream(
                        alignment,
                        relative,
                        field=str(source.get("field") or "") or None,
                        source_count=source_count,
                    ),
                )
            side_masks[side] = ~aligned_valid
            bindings.append({
                "side": side,
                "status": "complete" if aligned_valid.all() else "has_empty_rows",
                "relative_path": relative,
                "field": source["field"],
                "source_rows": source_count,
                "empty_source_row_count": int((~np.asarray(source["valid_rows"], dtype=bool)).sum()),
                "empty_video_frame_count": int((~aligned_valid).sum()),
            })
        except Exception as exc:
            side_masks[side][:] = True
            bindings.append({
                "side": side,
                "status": "unreadable_stream",
                "relative_path": relative,
                "error": str(exc)[:180],
            })
    empty_mask = np.logical_or.reduce([side_masks[side] for side in NEXUS_PRESSURE_EXPECTED_SIDES])
    empty_count = int(empty_mask.sum())
    metrics = {
        "detector": "nexus_pressure_empty_only_v1",
        "zero_is_valid": True,
        "expected_sides": list(NEXUS_PRESSURE_EXPECTED_SIDES),
        "stream_count": len(records),
        "missing_sides": [side for side in NEXUS_PRESSURE_EXPECTED_SIDES if by_side.get(side) is None],
        "empty_frame_count": empty_count,
        "empty_ranges": [
            {"start_frame": start, "end_frame": end}
            for start, end in _runs(empty_mask)
        ],
        "left_empty_frame_count": int(side_masks["left"].sum()),
        "right_empty_frame_count": int(side_masks["right"].sum()),
    }
    return {
        "enabled": True,
        "empty_mask": empty_mask,
        "side_masks": side_masks,
        "status": "warning" if empty_count else "completed",
        "message": f"检测到 {empty_count} 个压力空值帧；压力值为 0 不判错" if empty_count else "压力同步流完整；压力值为 0 仍视为有效",
        "metrics": metrics,
        "bindings": bindings,
    }


def detect_tactile_sudden_changes(
    matrix: np.ndarray,
    sigma: float,
    valid_mask: np.ndarray | None = None,
    max_spike_frames: int = NEXUS_TACTILE_SPIKE_MAX_FRAMES,
) -> dict:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("tactile feature matrix must be T x D")
    frame_count, dimensions = values.shape
    empty_mask = np.zeros(frame_count, dtype=bool)
    empty_dimensions = np.zeros((frame_count, dimensions), dtype=bool)
    if frame_count < 5 or dimensions == 0:
        return {
            "mask": empty_mask,
            "score": np.zeros(frame_count, dtype=np.float64),
            "dimension_mask": empty_dimensions,
            "event_count": 0,
            "range_count": 0,
            "ranges": [],
        }
    valid = np.ones(frame_count, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != (frame_count,):
        valid = np.zeros(frame_count, dtype=bool)
    prepared = np.where(valid[:, None], values, np.nan)
    filled = _filled(prepared)
    baseline = median_filter(filled, size=(5, 1), mode="nearest")
    residual = np.abs(filled - baseline)
    thresholds = np.maximum(_robust_threshold(residual[valid] if valid.any() else residual, sigma), 1.0)
    candidates = (residual > thresholds) & valid[:, None]
    # Feature order is pressure_sum, active_taxel_count, pressure_max,
    # active_pressure_mean. Count/mean can jitter during valid contact, so an
    # S1 reject must be anchored by total or peak pressure rather than those
    # auxiliary features alone.
    primary_dimensions = np.asarray([0, 2] if dimensions >= 3 else [0], dtype=np.int64)
    primary_candidates = candidates[:, primary_dimensions]
    dimension_mask = np.zeros_like(candidates, dtype=bool)
    for start, end in _mask_ranges(np.any(primary_candidates, axis=1)):
        if end - start + 1 > max(1, int(max_spike_frames)) or start == 0 or end >= frame_count - 1:
            continue
        if not (valid[start - 1] and valid[end + 1] and valid[start:end + 1].all()):
            continue
        before = filled[start - 1]
        after = filled[end + 1]
        local_baseline = (before + after) * 0.5
        excursion = np.max(np.abs(filled[start:end + 1] - local_baseline), axis=0)
        boundary_gap = np.abs(before - after)
        returned_to_baseline = boundary_gap <= np.maximum(thresholds, excursion * 0.35)
        relative_scale = np.maximum(np.abs(local_baseline), thresholds)
        strong_excursion = excursion >= np.maximum(
            thresholds,
            relative_scale * NEXUS_TACTILE_MIN_RELATIVE_EXCURSION,
        )
        selected = candidates[start:end + 1] & returned_to_baseline & strong_excursion
        dimension_mask[start:end + 1, primary_dimensions] = selected[:, primary_dimensions]
    flags = np.any(dimension_mask[:, primary_dimensions], axis=1)
    ratio = residual / np.maximum(thresholds, 1e-9)
    score = np.clip(np.max(np.where(dimension_mask, ratio, 0.0), axis=1) / 2.0, 0.0, 1.0)
    ranges = _mask_ranges(flags)
    return {
        "mask": flags,
        "score": score,
        "dimension_mask": dimension_mask,
        "event_count": int(flags.sum()),
        "range_count": len(ranges),
        "ranges": [{"start_frame": start, "end_frame": end} for start, end in ranges],
        "max_spike_frames": int(max_spike_frames),
        "min_relative_excursion": NEXUS_TACTILE_MIN_RELATIVE_EXCURSION,
    }


def inspect_nexus_tactile_sudden_changes(
    manifest: dict,
    episode: dict,
    alignment: dict,
    frame_count: int,
    sigma: float,
) -> dict:
    combined_mask = np.zeros(frame_count, dtype=bool)
    combined_score = np.zeros(frame_count, dtype=np.float64)
    side_masks = {side: np.zeros(frame_count, dtype=bool) for side in NEXUS_PRESSURE_EXPECTED_SIDES}
    side_scores = {side: np.zeros(frame_count, dtype=np.float64) for side in NEXUS_PRESSURE_EXPECTED_SIDES}
    if not _nexus_mode_enabled(manifest):
        return {
            "enabled": False,
            "mask": combined_mask,
            "score": combined_score,
            "side_masks": side_masks,
            "side_scores": side_scores,
            "status": "skipped",
            "message": "仅 Nexus 模式运行触觉突变 S1 检查",
            "metrics": {"zero_is_valid": True, "spike_frame_count": 0},
            "bindings": [],
        }
    root = Path(manifest["root_path"]).expanduser().resolve()
    records = _nexus_pressure_records(manifest, episode)
    by_side = {str(item.get("side") or "unknown"): item for item in records}
    bindings: list[dict] = []
    errors: list[str] = []
    feature_names = ["pressure_sum", "active_taxel_count", "pressure_max", "active_pressure_mean"]
    for side in NEXUS_PRESSURE_EXPECTED_SIDES:
        record = by_side.get(side)
        if record is None:
            continue
        relative = str(record["relative_path"])
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
            source = _pressure_source_validity(path, include_features=True)
            source_count = int(source["source_count"])
            features = source.get("features")
            if source_count <= 0 or features is None:
                raise ValueError("pressure/tactile feature rows are empty")
            aligned, _, aligned_valid = _resample_to_video(
                {
                    "values": features,
                    "row_indices": np.arange(source_count, dtype=np.int64),
                    "source_count": source_count,
                    "valid_rows": source["valid_rows"],
                },
                frame_count,
                _alignment_stream(
                    alignment,
                    relative,
                    field=str(source.get("field") or "") or None,
                    source_count=source_count,
                ),
            )
            detected = detect_tactile_sudden_changes(aligned, sigma, aligned_valid)
            side_masks[side] = np.asarray(detected["mask"], dtype=bool)
            side_scores[side] = np.asarray(detected["score"], dtype=np.float64)
            bindings.append({
                "side": side,
                "status": "has_spikes" if detected["event_count"] else "complete",
                "relative_path": relative,
                "field": source["field"],
                "source_rows": source_count,
                "feature_names": list(source.get("feature_names") or feature_names),
                "spike_frame_count": int(detected["event_count"]),
                "spike_ranges": detected["ranges"],
            })
        except Exception as exc:
            errors.append(f"{side}: {str(exc)[:160]}")
            bindings.append({
                "side": side,
                "status": "unreadable_stream",
                "relative_path": relative,
                "error": str(exc)[:180],
            })
    if side_masks:
        combined_mask = np.logical_or.reduce([side_masks[side] for side in NEXUS_PRESSURE_EXPECTED_SIDES])
        combined_score = np.maximum.reduce([side_scores[side] for side in NEXUS_PRESSURE_EXPECTED_SIDES])
    spike_count = int(combined_mask.sum())
    metrics = {
        "detector": "nexus_tactile_isolated_spike_v1",
        "zero_is_valid": True,
        "sustained_contact_step_is_valid": True,
        "max_spike_frames": NEXUS_TACTILE_SPIKE_MAX_FRAMES,
        "min_relative_excursion": NEXUS_TACTILE_MIN_RELATIVE_EXCURSION,
        "sigma": float(sigma),
        "feature_names": feature_names,
        "primary_feature_names": ["pressure_sum", "pressure_max"],
        "stream_count": len(records),
        "spike_frame_count": spike_count,
        "spike_ranges": [
            {"start_frame": start, "end_frame": end}
            for start, end in _mask_ranges(combined_mask)
        ],
        "left_spike_frame_count": int(side_masks["left"].sum()),
        "right_spike_frame_count": int(side_masks["right"].sum()),
        "errors": errors,
    }
    enabled = bool(records)
    return {
        "enabled": enabled,
        "mask": combined_mask,
        "score": combined_score,
        "side_masks": side_masks,
        "side_scores": side_scores,
        "status": "warning" if errors or spike_count else "completed" if enabled else "skipped",
        "message": (
            f"检测到 {spike_count} 个 Nexus 触觉孤立突变帧，已计入 S1"
            if spike_count else "Nexus 触觉未发现孤立突变；持续接触和数值 0 均视为有效"
        ) if enabled else "没有可读取的 Nexus 触觉流",
        "metrics": metrics,
        "bindings": bindings,
    }


def _coerce_matrix(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biufc":
        try:
            if array.dtype == object:
                array = np.asarray([np.asarray(item, dtype=np.float64).reshape(-1) for item in value], dtype=np.float64)
            else:
                array = array.astype(np.float64)
        except (TypeError, ValueError):
            raise ValueError("field is not a numeric time series")
    if array.ndim == 0:
        raise ValueError("field is scalar")
    if array.ndim == 1:
        array = array[:, None]
    else:
        array = array.reshape(array.shape[0], -1)
    if array.shape[0] < 4:
        raise ValueError("field has fewer than four rows")
    return np.asarray(array[:, :MAX_SIGNAL_DIMS], dtype=np.float64)


def _sample_matrix(array: np.ndarray, source_count: int | None = None) -> dict:
    source_count = int(source_count or array.shape[0])
    if array.shape[0] <= MAX_SIGNAL_ROWS:
        rows = np.arange(array.shape[0], dtype=np.int64)
        return {"values": array, "row_indices": rows, "source_count": source_count}
    rows = np.linspace(0, array.shape[0] - 1, MAX_SIGNAL_ROWS, dtype=np.int64)
    return {"values": array[rows], "row_indices": rows, "source_count": source_count}


def _read_hdf5(path: Path, field: str, descriptor: dict | None = None) -> dict:
    import h5py

    with h5py.File(path, "r") as handle:
        if field.endswith("/*"):
            group_name = field[:-2].rstrip("/")
            members = [str(item) for item in (descriptor or {}).get("members", [])]
            if not members:
                inferred = next(
                    (item for item in probe_local_signal_fields(path, max_dimensions=MAX_SIGNAL_DIMS) if item.get("field") == field),
                    None,
                )
                members = [str(item) for item in (inferred or {}).get("members", [])]
            if len(members) < 4 or any(not member.startswith(f"{group_name}/") for member in members):
                raise ValueError("skeletal transform group is incomplete")
            datasets = [handle[member] for member in members]
            count = min(int(dataset.shape[0]) for dataset in datasets)
            rows = np.arange(count, dtype=np.int64) if count <= MAX_SIGNAL_ROWS else np.linspace(0, count - 1, MAX_SIGNAL_ROWS, dtype=np.int64)
            values = np.concatenate([
                np.asarray(dataset[rows] if count > MAX_SIGNAL_ROWS else dataset[:count])[:, :3, 3]
                for dataset in datasets
            ], axis=1)
            return {
                "values": np.asarray(values, dtype=np.float64),
                "row_indices": rows,
                "source_count": count,
                "valid_rows": np.ones(rows.size, dtype=bool),
            }
        if field not in handle:
            raise KeyError(field)
        dataset = handle[field]
        if not dataset.shape:
            raise ValueError("field is scalar")
        count = int(dataset.shape[0])
        rows = np.arange(count, dtype=np.int64) if count <= MAX_SIGNAL_ROWS else np.linspace(0, count - 1, MAX_SIGNAL_ROWS, dtype=np.int64)
        raw_values = np.asarray(dataset[rows] if count > MAX_SIGNAL_ROWS else dataset[:])
        extraction = str((descriptor or {}).get("extraction") or "")
        if extraction == "nexus_dexweaveg1_20_to_mano21":
            wrist_field = f"{field.rsplit('/', 1)[0]}/wrist_quat" if "/" in field else "wrist_quat"
            wrist_dataset = handle.get(wrist_field)
            wrist_quaternion = None
            if wrist_dataset is not None and wrist_dataset.shape and int(wrist_dataset.shape[0]) == count:
                wrist_quaternion = np.asarray(wrist_dataset[rows] if count > MAX_SIGNAL_ROWS else wrist_dataset[:])
            points, joint_valid, _ = mano21_points_from_nexus_value(
                raw_values,
                wrist_quaternion=wrist_quaternion,
            )
            if points.ndim != 3 or points.shape[-2:] != (21, 3):
                raise ValueError("Nexus hand skeleton must resolve to T x 21 x 3")
            values = np.asarray(points.reshape(points.shape[0], -1), dtype=np.float64)
        elif extraction == "openxr_hand_26_to_mano21":
            parent = field.rsplit("/", 1)[0] if "/" in field else ""
            source_validity = None
            for leaf in ("location_flags", "flags", "validity", "valid", "tracked"):
                validity_field = f"{parent}/{leaf}" if parent else leaf
                candidate = handle.get(validity_field)
                if candidate is None or not candidate.shape or int(candidate.shape[0]) != count:
                    continue
                source_validity = np.asarray(candidate[rows] if count > MAX_SIGNAL_ROWS else candidate[:])
                if leaf in {"validity", "valid", "tracked"}:
                    source_validity = source_validity.astype(bool)
                break
            points, joint_valid = mano21_points_from_openxr_value(
                raw_values,
                validity=source_validity,
            )
            if points.ndim != 3 or points.shape[-2:] != (21, 3):
                raise ValueError("OpenXR hand stream must resolve to T x 21 x 3")
            values = np.asarray(points.reshape(points.shape[0], -1), dtype=np.float64)
        elif extraction == "skeleton_xyz":
            if raw_values.ndim != 3 or raw_values.shape[-1] < 3:
                raise ValueError("hand skeleton must have shape T x nodes x >=3")
            values = np.asarray(raw_values[..., :3].reshape(raw_values.shape[0], -1), dtype=np.float64)
            values = values[:, :MAX_SIGNAL_DIMS]
        else:
            values = _coerce_matrix(raw_values)
        parent = field.rsplit("/", 1)[0] if "/" in field else ""
        partial_field = f"{parent}/partial" if parent else "partial"
        valid_rows = np.ones(rows.size, dtype=bool)
        if extraction in {"openxr_hand_26_to_mano21", "nexus_dexweaveg1_20_to_mano21"}:
            valid_rows &= np.any(np.asarray(joint_valid, dtype=bool), axis=-1)
        if partial_field in handle:
            partial = handle[partial_field]
            if partial.shape and int(partial.shape[0]) == count:
                selected = np.asarray(partial[rows] if count > MAX_SIGNAL_ROWS else partial[:count], dtype=bool).reshape(-1)
                if selected.shape == valid_rows.shape:
                    valid_rows &= ~selected
        return {"values": values, "row_indices": rows, "source_count": count, "valid_rows": valid_rows}


def _read_parquet(path: Path, field: str) -> dict:
    import pyarrow.parquet as parquet

    source = parquet.ParquetFile(path)
    count = int(source.metadata.num_rows)
    stride = max(1, math.ceil(count / MAX_SIGNAL_ROWS))
    rows: list[int] = []
    values: list[Any] = []
    offset = 0
    for batch in source.iter_batches(batch_size=8192, columns=[field]):
        column = batch.column(0).to_pylist()
        for local, value in enumerate(column):
            absolute = offset + local
            if absolute % stride == 0:
                rows.append(absolute)
                values.append(value)
        offset += len(column)
    return {"values": _coerce_matrix(values), "row_indices": np.asarray(rows, dtype=np.int64), "source_count": count}


def _nested_json_value(payload: Any, field: str) -> Any:
    if field in {"", "$", "[]"}:
        return payload
    current = payload
    parts = [part for part in field.replace("[]", "").replace("/", ".").split(".") if part]
    for part in parts:
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, list):
            current = [item.get(part) if isinstance(item, dict) else None for item in current]
        else:
            raise KeyError(field)
    return current


def _read_json(path: Path, field: str) -> dict:
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("JSON exceeds the 64 MB curation limit")
    if path.suffix.casefold() == ".jsonl":
        payload = [json.loads(line) for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    return _sample_matrix(_coerce_matrix(_nested_json_value(payload, field)))


def _read_numpy(path: Path, field: str) -> dict:
    if path.suffix.casefold() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            selected = field if field in archive.files else (archive.files[0] if field in {"", "$"} and archive.files else None)
            if selected is None:
                raise KeyError(field)
            return _sample_matrix(_coerce_matrix(archive[selected]))
    return _sample_matrix(_coerce_matrix(np.load(path, mmap_mode="r", allow_pickle=False)))


def _read_table(path: Path, field: str) -> dict:
    delimiter = "\t" if path.suffix.casefold() == ".tsv" else ","
    values = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as source:
        for row in csv.DictReader(source, delimiter=delimiter):
            if field not in row:
                raise KeyError(field)
            values.append(float(row[field]))
            if len(values) >= MAX_SIGNAL_ROWS:
                break
    return _sample_matrix(_coerce_matrix(values))


def _read_numeric_series(path: Path, field: str, descriptor: dict | None = None) -> dict:
    suffix = path.suffix.casefold()
    if suffix in {".h5", ".hdf5", ".h5df"}:
        return _read_hdf5(path, field, descriptor)
    if suffix == ".parquet":
        return _read_parquet(path, field)
    if suffix in {".npy", ".npz"}:
        return _read_numpy(path, field)
    if suffix in {".json", ".jsonl"}:
        return _read_json(path, field)
    if suffix in {".csv", ".tsv"}:
        return _read_table(path, field)
    raise ValueError(f"unsupported numeric source: {suffix}")


def _alignment_stream(
    alignment: dict,
    relative_path: str,
    *,
    field: str | None = None,
    source_count: int | None = None,
) -> dict | None:
    return find_sensor_alignment_stream(
        alignment,
        relative_path,
        field=field,
        source_count=source_count,
    )


def _resample_to_video(series: dict, frame_count: int, alignment_stream: dict | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(series["values"], dtype=np.float64)
    row_indices = np.asarray(series["row_indices"], dtype=np.int64)
    source_count = int(series.get("source_count") or values.shape[0])
    if alignment_stream is None:
        return (
            np.full((frame_count, values.shape[1]), np.nan, dtype=np.float64),
            np.full(frame_count, -1, dtype=np.int64),
            np.zeros(frame_count, dtype=bool),
        )
    fractional_lookup = (alignment_stream or {}).get("frame_to_sensor_position")
    if isinstance(fractional_lookup, list):
        targets = np.full(frame_count, np.nan, dtype=np.float64)
        count = min(frame_count, len(fractional_lookup))
        targets[:count] = np.asarray(fractional_lookup[:count], dtype=np.float64)
        valid = np.isfinite(targets) & (targets >= 0.0) & (targets <= max(0, source_count - 1))
        sampled_valid = np.asarray(series.get("valid_rows", np.ones(values.shape[0], dtype=bool)), dtype=bool).reshape(-1)
        if sampled_valid.shape != (values.shape[0],):
            sampled_valid = np.ones(values.shape[0], dtype=bool)
        clipped = np.clip(targets, 0.0, max(0, source_count - 1))
        right = np.searchsorted(row_indices, clipped, side="left")
        right = np.clip(right, 0, max(0, row_indices.size - 1))
        left = np.clip(right - 1, 0, max(0, row_indices.size - 1))
        exact = row_indices[right] == clipped
        left = np.where(exact, right, left)
        left_rows = row_indices[left].astype(np.float64)
        right_rows = row_indices[right].astype(np.float64)
        span = right_rows - left_rows
        alpha = np.divide(
            clipped - left_rows,
            span,
            out=np.zeros_like(clipped),
            where=span > 0.0,
        )
        valid &= sampled_valid[left] & sampled_valid[right]
        output = np.full((frame_count, values.shape[1]), np.nan, dtype=np.float64)
        if valid.any():
            output[valid] = (
                values[left[valid]] * (1.0 - alpha[valid, None])
                + values[right[valid]] * alpha[valid, None]
            )
        source_rows = np.full(frame_count, -1, dtype=np.int64)
        source_rows[valid] = np.rint(clipped[valid]).astype(np.int64)
        return output, source_rows, valid
    targets = np.full(frame_count, -1, dtype=np.int64)
    lookup = (alignment_stream or {}).get("frame_to_sensor_index")
    if isinstance(lookup, list):
        count = min(frame_count, len(lookup))
        targets[:count] = np.asarray(lookup[:count], dtype=np.int64)
    elif alignment_stream.get("mode") in {
        "prealigned_master_clock",
        "paired_frame_index",
        "applied_projection_video_aligned",
    }:
        targets = np.arange(frame_count, dtype=np.int64)
    elif alignment_stream.get("index_multiplier") is not None:
        targets = np.rint(np.arange(frame_count) * float(alignment_stream["index_multiplier"])).astype(np.int64)
    else:
        return (
            np.full((frame_count, values.shape[1]), np.nan, dtype=np.float64),
            np.full(frame_count, -1, dtype=np.int64),
            np.zeros(frame_count, dtype=bool),
        )
    valid = (targets >= 0) & (targets < source_count)
    positions = np.searchsorted(row_indices, np.clip(targets, 0, max(0, source_count - 1)), side="left")
    positions = np.clip(positions, 0, max(0, row_indices.size - 1))
    left = np.clip(positions - 1, 0, max(0, row_indices.size - 1))
    use_left = np.abs(row_indices[left] - targets) <= np.abs(row_indices[positions] - targets)
    positions = np.where(use_left, left, positions)
    sampled_valid = np.asarray(series.get("valid_rows", np.ones(values.shape[0], dtype=bool)), dtype=bool).reshape(-1)
    if sampled_valid.shape != (values.shape[0],):
        sampled_valid = np.ones(values.shape[0], dtype=bool)
    valid &= sampled_valid[positions]
    output = np.full((frame_count, values.shape[1]), np.nan, dtype=np.float64)
    output[valid] = values[positions[valid]]
    source_rows = np.full(frame_count, -1, dtype=np.int64)
    source_rows[valid] = row_indices[positions[valid]]
    return output, source_rows, valid


def _load_signal_bundle(manifest: dict, episode: dict, alignment: dict, frame_count: int | None = None) -> dict:
    root = Path(manifest["root_path"]).expanduser().resolve()
    target_frame_count = int(frame_count if frame_count is not None else episode.get("frame_count") or 0)
    matrices: dict[str, list[np.ndarray]] = {"joint": [], "action": []}
    projection_raw_joint_matrices: list[np.ndarray] = []
    projection_correction_active = False
    bindings: list[dict] = []
    warnings: list[str] = []
    gripper_columns: dict[str, set[int]] = {"joint": set(), "action": set()}
    offsets = {"joint": 0, "action": 0}
    action_representations: set[str] = set()
    semantic_dimensions_known = True
    source_valid_masks: list[np.ndarray] = []
    for candidate in _signal_candidates(manifest, episode):
        if len(matrices[candidate["kind"]]) >= 6:
            continue
        absolute_path = str(candidate.get("absolute_path") or "").strip()
        path = Path(absolute_path).expanduser().resolve() if absolute_path else (root / candidate["relative_path"]).resolve()
        try:
            if not absolute_path:
                path.relative_to(root)
            elif not path.is_file():
                raise FileNotFoundError(path)
            series = _read_numeric_series(path, candidate["field"], candidate)
            if absolute_path:
                alignment_stream = {
                    "relative_path": candidate["relative_path"],
                    "field": candidate["field"],
                    "data_count": int(series["source_count"]),
                    "mode": "applied_projection_video_aligned",
                    "index_multiplier": 1.0,
                }
                alignment_retiming = alignment.get("retiming") or {}
                retimed_positions = np.asarray(
                    (alignment_retiming.get("source_frame_positions") or []),
                    dtype=np.float64,
                ).reshape(-1)
                projection_video_identity = bool(
                    candidate.get("source") in {"applied_projection_correction", "full_run_projection_correction"}
                    and target_frame_count == int(series["source_count"])
                    and str(alignment_retiming.get("mode") or "") == "derived_video_timeline"
                )
                if not projection_video_identity and (
                    retimed_positions.shape == (target_frame_count,)
                    and np.isfinite(retimed_positions).all()
                    and np.all(retimed_positions >= 0.0)
                    and np.all(retimed_positions <= int(series["source_count"]) - 1)
                ):
                    alignment_stream["frame_to_sensor_position"] = retimed_positions.tolist()
            else:
                alignment_stream = _alignment_stream(
                    alignment,
                    candidate["relative_path"],
                    field=candidate["field"],
                    source_count=int(series["source_count"]),
                )
            aligned, source_row_indices, source_valid = _resample_to_video(
                series,
                target_frame_count,
                alignment_stream,
            )
        except Exception as exc:
            warnings.append(f"{candidate['relative_path']} / {candidate['field']}: {str(exc)[:180]}")
            continue
        if aligned.shape[1] + offsets[candidate["kind"]] > MAX_SIGNAL_DIMS:
            aligned = aligned[:, : max(0, MAX_SIGNAL_DIMS - offsets[candidate["kind"]])]
        if not aligned.shape[1]:
            continue
        raw_projection_aligned: np.ndarray | None = None
        if candidate["kind"] == "joint" and candidate.get("source") in {"applied_projection_correction", "full_run_projection_correction"}:
            try:
                raw_path = (root / candidate["relative_path"]).resolve()
                raw_path.relative_to(root)
                raw_series = _read_numeric_series(raw_path, candidate["field"], candidate)
                raw_projection_aligned, _, _ = _resample_to_video(
                    raw_series,
                    target_frame_count,
                    _alignment_stream(
                        alignment,
                        candidate["relative_path"],
                        field=candidate["field"],
                        source_count=int(raw_series["source_count"]),
                    ),
                )
                raw_projection_aligned = raw_projection_aligned[:, : aligned.shape[1]]
                if raw_projection_aligned.shape != aligned.shape:
                    raw_projection_aligned = None
                else:
                    projection_correction_active = True
            except Exception as exc:
                warnings.append(f"Projection raw S1 guard unavailable for {candidate['relative_path']}: {str(exc)[:140]}")
        text = f"{candidate['field']} {candidate['role']} {candidate['modality']}".casefold()
        start = offsets[candidate["kind"]]
        end = start + aligned.shape[1]
        explicit_gripper = {
            start + int(value)
            for value in candidate.get("gripper_indices", [])
            if 0 <= int(value) < aligned.shape[1]
        }
        gripper_columns[candidate["kind"]].update(explicit_gripper)
        if "gripper" in text and not explicit_gripper:
            gripper_columns[candidate["kind"]].update(range(start, end))
        if candidate["kind"] == "action":
            representation = str(candidate.get("representation") or "unknown")
            if representation in {"absolute", "delta", "velocity"}:
                action_representations.add(representation)
        if aligned.shape[1] > 1 and len(candidate.get("dimension_names") or []) != aligned.shape[1] and "gripper" not in text:
            semantic_dimensions_known = False
        offsets[candidate["kind"]] = end
        matrices[candidate["kind"]].append(aligned)
        if candidate["kind"] == "joint":
            projection_raw_joint_matrices.append(raw_projection_aligned if raw_projection_aligned is not None else aligned)
        source_valid_masks.append(source_valid)
        bindings.append({
            **candidate,
            "dimensions": int(aligned.shape[1]),
            "column_start": start,
            "column_end": end,
            "source_rows": int(series["source_count"]),
            "_source_row_indices": source_row_indices,
            "_valid_mask": source_valid,
        })
    return {
        "joint": np.concatenate(matrices["joint"], axis=1) if matrices["joint"] else None,
        "projection_raw_joint": np.concatenate(projection_raw_joint_matrices, axis=1) if projection_correction_active else None,
        "projection_correction_active": projection_correction_active,
        "action": np.concatenate(matrices["action"], axis=1) if matrices["action"] else None,
        "bindings": bindings,
        "warnings": warnings,
        "gripper_columns": gripper_columns,
        "action_representation": next(iter(action_representations)) if len(action_representations) == 1 else "unknown",
        "semantic_dimensions_known": semantic_dimensions_known,
        "embodiment_ids": sorted({str(item.get("embodiment_id")) for item in bindings if item.get("embodiment_id")}),
        "valid_mask": np.logical_and.reduce(source_valid_masks) if source_valid_masks else np.ones(target_frame_count, dtype=bool),
    }


def _filled(matrix: np.ndarray) -> np.ndarray:
    output = np.asarray(matrix, dtype=np.float64).copy()
    x = np.arange(output.shape[0])
    for column in range(output.shape[1]):
        finite = np.isfinite(output[:, column])
        if finite.sum() < 2:
            output[:, column] = 0.0
        else:
            output[:, column] = np.interp(x, x[finite], output[finite, column])
    return output


def _smooth(matrix: np.ndarray) -> np.ndarray:
    filled = _filled(matrix)
    if filled.shape[0] < 5:
        return filled
    median = median_filter(filled, size=(3, 1), mode="nearest")
    median = median_filter(median, size=(5, 1), mode="nearest")
    window = min(21, filled.shape[0] if filled.shape[0] % 2 else filled.shape[0] - 1)
    return savgol_filter(median, window_length=max(5, window), polyorder=2, axis=0, mode="interp")


def _robust_threshold(values: np.ndarray, sigma: float) -> np.ndarray:
    median = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - median), axis=0) * 1.4826
    fallback = np.nanpercentile(values, 90, axis=0) * 0.1
    scale = np.maximum(mad, np.maximum(fallback, 1e-9))
    return median + sigma * scale


def detect_sudden_changes(matrix: np.ndarray, sigma: float) -> dict:
    smooth = _smooth(matrix)
    raw = _filled(matrix)
    residual = np.abs(raw - smooth)
    acceleration = np.abs(np.diff(raw, n=2, axis=0, prepend=np.repeat(raw[:1], 2, axis=0)))
    jerk = np.abs(np.diff(raw, n=3, axis=0, prepend=np.repeat(raw[:1], 3, axis=0)))
    residual_t = _robust_threshold(residual, sigma)
    acceleration_t = _robust_threshold(acceleration, sigma)
    jerk_t = _robust_threshold(jerk, sigma)
    dimension_flags = (residual > residual_t) & ((acceleration > acceleration_t) | (jerk > jerk_t))
    flags = np.any(dimension_flags, axis=1)
    if flags.size >= 6:
        flags[:3] = False
        flags[-3:] = False
    residual_ratio = residual / np.maximum(residual_t, 1e-9)
    dynamic_ratio = np.maximum(acceleration / np.maximum(acceleration_t, 1e-9), jerk / np.maximum(jerk_t, 1e-9))
    score = np.clip(np.nanmax(np.minimum(residual_ratio, dynamic_ratio), axis=1) / 2.0, 0.0, 1.0)
    return {
        "mask": flags,
        "score": score,
        "event_count": int(flags.sum()),
        "dimension_mask": dimension_flags,
    }


def _guard_projection_introduced_s1(
    detected: dict,
    projection_raw_joint: np.ndarray | None,
    action: np.ndarray | None,
    sigma: float,
) -> tuple[dict, np.ndarray, dict | None]:
    """A derived pose correction may fix raw S1, but may not create a hard reject."""
    mask = np.asarray(detected.get("mask"), dtype=bool)
    if projection_raw_joint is None or projection_raw_joint.shape[0] != mask.shape[0]:
        return detected, np.zeros(mask.shape, dtype=bool), None
    raw_parts = [projection_raw_joint]
    if action is not None:
        raw_parts.append(action)
    raw_detected = detect_sudden_changes(np.concatenate(raw_parts, axis=1), sigma)
    raw_mask = np.asarray(raw_detected["mask"], dtype=bool)
    introduced = mask & ~raw_mask
    if not introduced.any():
        return detected, introduced, raw_detected
    guarded = {**detected}
    guarded_mask = mask & raw_mask
    guarded["mask"] = guarded_mask
    guarded["event_count"] = int(guarded_mask.sum())
    guarded["score"] = np.where(guarded_mask, np.asarray(detected.get("score"), dtype=np.float64), 0.0)
    if detected.get("dimension_mask") is not None:
        dimension_mask = np.asarray(detected["dimension_mask"], dtype=bool).copy()
        dimension_mask[introduced] = False
        guarded["dimension_mask"] = dimension_mask
    return guarded, introduced, raw_detected


def _downgrade_sustained_motion_s1(
    detected: dict,
    fps: float,
    *,
    merge_gap_seconds: float = 0.40,
    minimum_span_seconds: float = 0.30,
) -> tuple[dict, np.ndarray]:
    """Keep isolated telemetry spikes red while treating repeated motion as review.

    A wiping stroke can contain many legitimate direction reversals.  When S1
    points repeat across a sustained time span, they are not an isolated sensor
    spike and must not remove the whole action before VLM can inspect it.
    """
    mask = np.asarray(detected.get("mask"), dtype=bool)
    frames = np.flatnonzero(mask)
    review = np.zeros(mask.shape, dtype=bool)
    if frames.size < 3:
        return detected, review
    maximum_gap = max(1, int(round(max(0.0, merge_gap_seconds) * max(0.01, fps))))
    minimum_span = max(3, int(round(max(0.0, minimum_span_seconds) * max(0.01, fps))))
    start = 0
    groups: list[np.ndarray] = []
    for offset in range(1, len(frames)):
        if int(frames[offset] - frames[offset - 1]) > maximum_gap:
            groups.append(frames[start:offset])
            start = offset
    groups.append(frames[start:])
    for group in groups:
        if len(group) >= 3 and int(group[-1] - group[0] + 1) >= minimum_span:
            review[group] = True
    if not review.any():
        return detected, review
    guarded = {**detected}
    guarded_mask = mask & ~review
    guarded["mask"] = guarded_mask
    guarded["event_count"] = int(guarded_mask.sum())
    guarded["score"] = np.where(guarded_mask, np.asarray(detected.get("score"), dtype=np.float64), 0.0)
    if detected.get("dimension_mask") is not None:
        dimension_mask = np.asarray(detected["dimension_mask"], dtype=bool).copy()
        dimension_mask[review] = False
        guarded["dimension_mask"] = dimension_mask
    return guarded, review


def _rotation_matrices_from_rot6d(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotations = np.asarray(values, dtype=np.float64)
    if rotations.ndim != 2 or rotations.shape[1] != 6:
        raise ValueError("rot6d signal must have shape T x 6")
    finite = np.isfinite(rotations).all(axis=1)
    first = rotations[:, :3]
    second = rotations[:, 3:]
    first_norm = np.linalg.norm(first, axis=1)
    first_unit = first / np.maximum(first_norm[:, None], 1e-12)
    second_orthogonal = second - np.sum(first_unit * second, axis=1, keepdims=True) * first_unit
    second_norm = np.linalg.norm(second_orthogonal, axis=1)
    second_unit = second_orthogonal / np.maximum(second_norm[:, None], 1e-12)
    third_unit = np.cross(first_unit, second_unit)
    valid = finite & (first_norm > 1e-6) & (second_norm > 1e-6)
    matrices = np.stack((first_unit, second_unit, third_unit), axis=2)
    matrices[~valid] = np.eye(3, dtype=np.float64)
    return matrices, valid


def detect_rot6d_jumps(
    values: np.ndarray,
    sigma: float = 6.0,
    absolute_threshold_degrees: float = ROT6D_ABSOLUTE_JUMP_DEGREES,
) -> dict:
    """Detect frame-to-frame orientation jumps in an end-pose rot6d signal."""
    matrices, valid = _rotation_matrices_from_rot6d(values)
    frame_count = int(matrices.shape[0])
    angles = np.zeros(frame_count, dtype=np.float64)
    pair_valid = valid & np.r_[False, valid[:-1]]
    if frame_count > 1:
        relative = np.einsum("fji,fjk->fik", matrices[:-1], matrices[1:])
        cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        angles[1:] = np.degrees(np.arccos(cosine))
    baseline = angles[pair_valid]
    if baseline.size:
        median = float(np.median(baseline))
        mad = float(np.median(np.abs(baseline - median))) * 1.4826
        robust_threshold = median + float(sigma) * max(mad, 0.25)
    else:
        robust_threshold = 0.0
    threshold = max(float(absolute_threshold_degrees), robust_threshold)
    invalid = ~valid
    jumps = pair_valid & (angles > threshold)
    mask = invalid | jumps
    score = np.maximum(
        np.clip(angles / max(threshold, 1e-9), 0.0, 2.0) / 2.0,
        invalid.astype(np.float64),
    )
    return {
        "mask": mask,
        "score": score,
        "event_count": int(mask.sum()),
        "jump_frame_count": int(jumps.sum()),
        "invalid_frame_count": int(invalid.sum()),
        "threshold_degrees": round(threshold, 6),
        "absolute_threshold_degrees": round(float(absolute_threshold_degrees), 6),
        "max_relative_degrees": round(float(angles.max(initial=0.0)), 6),
        "relative_degrees": angles,
    }


_ROT6D_DIMENSION_PATTERN = re.compile(r"^(.*?)(?:rot(?:ation)?[_ .-]?6d|r6d)[_.\[]?([0-5])\]?$", re.IGNORECASE)
_ENDPOSE_TOKENS = ("endpose", "end_pose", "end pose", "end-effector", "end_effector", "eef_pose", "tcp_pose")


def _rot6d_groups_from_bundle(bundle: dict) -> list[dict]:
    groups: list[dict] = []
    for binding in bundle.get("bindings") or []:
        kind = str(binding.get("kind") or "")
        matrix = bundle.get(kind)
        if matrix is None:
            continue
        start = int(binding.get("column_start") or 0)
        width = int(binding.get("dimensions") or 0)
        names = [str(item) for item in binding.get("dimension_names") or []]
        semantic_text = " ".join([
            str(binding.get("field") or ""),
            str(binding.get("role") or ""),
            str(binding.get("modality") or ""),
            *names,
        ]).casefold()
        explicit: dict[str, dict[int, int]] = {}
        for local_index, name in enumerate(names[:width]):
            match = _ROT6D_DIMENSION_PATTERN.match(name.strip())
            if match:
                explicit.setdefault(match.group(1).strip("_.[] -").casefold(), {})[int(match.group(2))] = local_index
        for prefix, members in explicit.items():
            if set(members) == set(range(6)):
                groups.append({
                    "name": prefix or str(binding.get("field") or "rot6d"),
                    "kind": kind,
                    "indices": [start + members[index] for index in range(6)],
                    "source_path": binding.get("relative_path"),
                    "field": binding.get("field"),
                })
        if explicit or not any(token in semantic_text for token in _ENDPOSE_TOKENS):
            continue
        if width == 6:
            local_indices = list(range(6))
        elif width >= 9:
            local_indices = list(range(3, 9))
        else:
            continue
        groups.append({
            "name": str(binding.get("field") or "endpose"),
            "kind": kind,
            "indices": [start + index for index in local_indices],
            "source_path": binding.get("relative_path"),
            "field": binding.get("field"),
        })
    deduplicated = {}
    for group in groups:
        key = (group["kind"], tuple(group["indices"]))
        deduplicated.setdefault(key, group)
    return list(deduplicated.values())


def inspect_rot6d_jumps(bundle: dict, sigma: float) -> dict:
    frame_count = max((int(np.asarray(bundle.get(kind)).shape[0]) for kind in ("joint", "action") if bundle.get(kind) is not None), default=0)
    combined_mask = np.zeros(frame_count, dtype=bool)
    combined_score = np.zeros(frame_count, dtype=np.float64)
    reports = []
    for group in _rot6d_groups_from_bundle(bundle):
        matrix = np.asarray(bundle[group["kind"]], dtype=np.float64)
        result = detect_rot6d_jumps(matrix[:, group["indices"]], sigma=sigma)
        combined_mask |= result["mask"]
        combined_score = np.maximum(combined_score, result["score"])
        reports.append({
            **{key: group.get(key) for key in ("name", "kind", "source_path", "field")},
            **{key: result[key] for key in ("event_count", "jump_frame_count", "invalid_frame_count", "threshold_degrees", "max_relative_degrees")},
        })
    return {
        "mask": combined_mask,
        "score": combined_score,
        "event_count": int(combined_mask.sum()),
        "group_count": len(reports),
        "groups": reports,
    }


def _mask_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _continuous_anchor_columns(values: np.ndarray, start: int, end: int) -> np.ndarray:
    left = values[start - 1]
    right = values[end + 1]
    finite = np.isfinite(left) & np.isfinite(right)
    before = np.abs(np.diff(values[max(0, start - 12):start], axis=0))
    after = np.abs(np.diff(values[end + 1:min(values.shape[0], end + 13)], axis=0))
    local_steps = np.concatenate([item for item in (before, after) if item.size], axis=0)
    if not local_steps.size:
        return np.zeros(values.shape[1], dtype=bool)
    median = np.nanmedian(local_steps, axis=0)
    mad = np.nanmedian(np.abs(local_steps - median), axis=0) * 1.4826
    allowed = (end - start + 2) * np.maximum(median + 6.0 * mad, 1e-7) * 2.0
    return finite & (np.abs(right - left) <= allowed)


def repair_isolated_spikes(
    matrix: np.ndarray,
    mask: np.ndarray,
    dimension_mask: np.ndarray,
    max_gap_frames: int = 5,
    protected_columns: set[int] | None = None,
) -> dict:
    original = np.asarray(matrix, dtype=np.float64)
    repaired = original.copy()
    cells = np.zeros(original.shape, dtype=bool)
    protected = protected_columns or set()
    ranges = []
    for start, end in _mask_ranges(mask):
        length = end - start + 1
        if start == 0 or end >= original.shape[0] - 1 or length > max_gap_frames:
            continue
        dimensions = np.any(dimension_mask[start:end + 1], axis=0)
        dimensions &= _continuous_anchor_columns(original, start, end)
        if protected:
            dimensions[list(protected)] = False
        selected = np.flatnonzero(dimensions)
        if not selected.size:
            continue
        anchor_indices = np.r_[
            np.arange(max(0, start - 6), start),
            np.arange(end + 1, min(original.shape[0], end + 7)),
        ]
        successful = []
        for column in selected:
            finite = np.isfinite(original[anchor_indices, column])
            usable = anchor_indices[finite]
            if (usable < start).sum() < 2 or (usable > end).sum() < 2:
                continue
            try:
                repaired[start:end + 1, column] = CubicSpline(
                    usable,
                    original[usable, column],
                )(np.arange(start, end + 1))
            except ValueError:
                continue
            successful.append(column)
        if not successful:
            continue
        cells[start:end + 1, successful] = True
        ranges.append((start, end))
    return {"values": repaired, "cell_mask": cells, "ranges": ranges}


def repair_rot6d_spikes(values: np.ndarray, mask: np.ndarray, max_gap_frames: int = 5) -> dict:
    original = np.asarray(values, dtype=np.float64)
    repaired = original.copy()
    cells = np.zeros(original.shape, dtype=bool)
    ranges = []
    matrices, valid = _rotation_matrices_from_rot6d(original)
    for start, end in _mask_ranges(mask):
        length = end - start + 1
        if start == 0 or end >= original.shape[0] - 1 or length > max_gap_frames:
            continue
        left_index, right_index = start - 1, end + 1
        if not (valid[left_index] and valid[right_index]):
            continue
        relative = matrices[left_index].T @ matrices[right_index]
        cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        if np.degrees(np.arccos(cosine)) > ROT6D_ABSOLUTE_JUMP_DEGREES:
            continue
        interpolation = Slerp(
            [float(left_index), float(right_index)],
            Rotation.from_matrix(matrices[[left_index, right_index]]),
        )(np.arange(start, end + 1, dtype=np.float64)).as_matrix()
        repaired[start:end + 1] = interpolation[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
        cells[start:end + 1] = True
        ranges.append((start, end))
    return {"values": repaired, "cell_mask": cells, "ranges": ranges}


def _combined_column_layout(bundle: dict) -> list[tuple[dict, int, int]]:
    joint_width = int(bundle["joint"].shape[1]) if bundle.get("joint") is not None else 0
    layout = []
    for binding in bundle.get("bindings") or []:
        offset = 0 if binding.get("kind") == "joint" else joint_width
        layout.append((binding, offset + int(binding["column_start"]), offset + int(binding["column_end"])))
    return layout


def _writable_repair_target(binding: dict, local_column: int) -> tuple[str, int] | None:
    suffix = Path(str(binding.get("relative_path") or "")).suffix.casefold()
    if suffix not in {".h5", ".hdf5", ".h5df"}:
        return None
    field = str(binding.get("field") or "").strip("/")
    if binding.get("extraction") in {
        "openxr_hand_26_to_mano21",
        "nexus_dexweaveg1_20_to_mano21",
    }:
        # These adapters reorder/drop source nodes and synthesize MANO0, so
        # canonical columns do not have a safe one-cell source writeback.
        return None
    if field.endswith("/*") and binding.get("extraction") == "matrix_translation_xyz":
        members = [str(item).strip("/") for item in binding.get("members") or []]
        member_index, axis = divmod(local_column, 3)
        if member_index >= len(members):
            return None
        return members[member_index], (3, 7, 11)[axis]
    if field and not field.endswith("/*"):
        return field, local_column
    return None


def _restrict_repairs_to_writable_sources(
    original: np.ndarray,
    candidate: np.ndarray,
    cell_mask: np.ndarray,
    bundle: dict,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(original, dtype=np.float64).copy()
    accepted = np.zeros(cell_mask.shape, dtype=bool)
    for binding, combined_start, combined_end in _combined_column_layout(bundle):
        source_rows = np.asarray(binding.get("_source_row_indices"), dtype=np.int64)
        if source_rows.shape != (values.shape[0],):
            continue
        unique_rows, counts = np.unique(source_rows[source_rows >= 0], return_counts=True)
        unique = set(unique_rows[counts == 1].tolist())
        for column in range(combined_start, combined_end):
            target = _writable_repair_target(binding, column - combined_start)
            if target is None:
                continue
            frames = np.flatnonzero(cell_mask[:, column])
            frames = np.asarray([frame for frame in frames if int(source_rows[frame]) in unique], dtype=np.int64)
            if not frames.size:
                continue
            values[frames, column] = candidate[frames, column]
            accepted[frames, column] = True
    return values, accepted


def _repair_patch_entries(original: np.ndarray, repaired: np.ndarray, cell_mask: np.ndarray, bundle: dict) -> list[dict]:
    grouped: dict[tuple[str, str], list[tuple[int, int, float]]] = {}
    for binding, combined_start, combined_end in _combined_column_layout(bundle):
        source_path = str(binding.get("relative_path") or "").replace("\\", "/")
        source_rows = np.asarray(binding.get("_source_row_indices"), dtype=np.int64)
        for column in range(combined_start, combined_end):
            target = _writable_repair_target(binding, column - combined_start)
            if target is None:
                continue
            dataset_path, flat_index = target
            for frame in np.flatnonzero(cell_mask[:, column]):
                value = float(repaired[frame, column])
                if source_rows[frame] >= 0 and np.isfinite(value) and value != original[frame, column]:
                    grouped.setdefault((source_path, dataset_path), []).append((int(source_rows[frame]), flat_index, value))
    return [
        {
            "source_path": source_path,
            "dataset_path": dataset_path,
            "source_rows": [item[0] for item in items],
            "flat_indices": [item[1] for item in items],
            "values": [item[2] for item in items],
        }
        for (source_path, dataset_path), items in sorted(grouped.items())
    ]


def apply_s1_repair_to_bundle(bundle: dict, repair: dict | None) -> None:
    if repair is None:
        return
    entries: dict[tuple[str, str, int], dict[int, float]] = {}
    for entry in repair.get("entries") or []:
        source_path = str(entry.get("source_path") or "").replace("\\", "/").casefold()
        dataset_path = str(entry.get("dataset_path") or "").strip("/").casefold()
        for row, flat_index, value in zip(
            entry.get("source_rows") or [],
            entry.get("flat_indices") or [],
            entry.get("values") or [],
        ):
            entries.setdefault((source_path, dataset_path, int(flat_index)), {})[int(row)] = float(value)
    for binding in bundle.get("bindings") or []:
        kind = str(binding.get("kind") or "")
        matrix = bundle.get(kind)
        source_rows = np.asarray(binding.get("_source_row_indices"), dtype=np.int64)
        if matrix is None or source_rows.shape != (matrix.shape[0],):
            continue
        source_path = str(binding.get("relative_path") or "").replace("\\", "/").casefold()
        output = np.asarray(matrix, dtype=np.float64).copy()
        start, end = int(binding["column_start"]), int(binding["column_end"])
        for column in range(start, end):
            target = _writable_repair_target(binding, column - start)
            if target is None:
                continue
            dataset_path, flat_index = target
            row_values = entries.get((source_path, dataset_path.casefold(), int(flat_index))) or {}
            for source_row, value in row_values.items():
                output[source_rows == source_row, column] = value
        bundle[kind] = output


def repair_s1_bundle(
    bundle: dict,
    sigma: float,
    max_gap_frames: int,
    blocked_repair_frames: np.ndarray | None = None,
) -> dict:
    parts = [item for item in (bundle.get("joint"), bundle.get("action")) if item is not None]
    original = np.concatenate(parts, axis=1)
    generic_before = detect_sudden_changes(original, sigma)
    rot_before = inspect_rot6d_jumps(bundle, sigma)
    rot_groups = _rot6d_groups_from_bundle(bundle)
    protected = {
        (0 if group["kind"] == "joint" else int(bundle["joint"].shape[1]) if bundle.get("joint") is not None else 0) + index
        for group in rot_groups
        for index in group["indices"]
    }
    protected.update(int(value) for value in (bundle.get("gripper_columns") or {}).get("joint", set()))
    action_offset = int(bundle["joint"].shape[1]) if bundle.get("joint") is not None else 0
    protected.update(
        action_offset + int(value)
        for value in (bundle.get("gripper_columns") or {}).get("action", set())
    )
    generic_candidate = repair_isolated_spikes(
        original,
        generic_before["mask"],
        generic_before["dimension_mask"],
        max_gap_frames,
        protected,
    )
    candidate = generic_candidate["values"]
    candidate_cells = generic_candidate["cell_mask"]
    for group in rot_groups:
        kind_offset = 0 if group["kind"] == "joint" else int(bundle["joint"].shape[1]) if bundle.get("joint") is not None else 0
        combined_indices = [kind_offset + index for index in group["indices"]]
        result = detect_rot6d_jumps(original[:, combined_indices], sigma=sigma)
        rot_candidate = repair_rot6d_spikes(original[:, combined_indices], result["mask"], max_gap_frames)
        candidate[:, combined_indices] = rot_candidate["values"]
        candidate_cells[:, combined_indices] = rot_candidate["cell_mask"]
    candidate, candidate_cells = _restrict_repairs_to_writable_sources(original, candidate, candidate_cells, bundle)
    if blocked_repair_frames is not None:
        blocked = np.asarray(blocked_repair_frames, dtype=bool)
        if blocked.shape != (original.shape[0],):
            raise ValueError("blocked S1 repair mask length does not match the signal timeline")
        candidate_cells[blocked] = False
    joint_width = int(bundle["joint"].shape[1]) if bundle.get("joint") is not None else 0

    def inspect(values: np.ndarray) -> tuple[dict, dict, np.ndarray]:
        candidate_bundle = {**bundle}
        if bundle.get("joint") is not None:
            candidate_bundle["joint"] = values[:, :joint_width]
        if bundle.get("action") is not None:
            candidate_bundle["action"] = values[:, joint_width:]
        generic = detect_sudden_changes(values, sigma)
        rot = inspect_rot6d_jumps(candidate_bundle, sigma)
        return generic, rot, generic["mask"] | rot["mask"]

    _, _, candidate_after_mask = inspect(candidate)
    accepted_cells = candidate_cells.copy()
    for start, end in _mask_ranges(np.any(candidate_cells, axis=1)):
        check_start, check_end = max(0, start - 3), min(original.shape[0] - 1, end + 3)
        if candidate_after_mask[check_start:check_end + 1].any():
            accepted_cells[start:end + 1] = False
    accepted_cells &= ~np.isclose(candidate, original, rtol=0.0, atol=0.0, equal_nan=True)
    while True:
        repaired = np.where(accepted_cells, candidate, original)
        generic_after, rot_after, after_mask = inspect(repaired)
        rejected = np.zeros(original.shape[0], dtype=bool)
        for start, end in _mask_ranges(np.any(accepted_cells, axis=1)):
            check_start, check_end = max(0, start - 3), min(original.shape[0] - 1, end + 3)
            if after_mask[check_start:check_end + 1].any():
                rejected[start:end + 1] = True
        if not rejected.any():
            break
        accepted_cells[rejected] = False
    repaired_frames = np.any(accepted_cells, axis=1)
    return {
        "values": repaired,
        "cell_mask": accepted_cells,
        "repaired_mask": repaired_frames,
        "before_mask": generic_before["mask"] | rot_before["mask"],
        "after_mask": after_mask,
        "generic_before": generic_before,
        "rot_before": rot_before,
        "generic_after": generic_after,
        "rot_after": rot_after,
        "entries": _repair_patch_entries(original, repaired, accepted_cells, bundle),
        "ranges": _mask_ranges(repaired_frames),
    }


def _aligned_pair(state: np.ndarray, action: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag > 0:
        return state[lag:], action[:-lag]
    if lag < 0:
        return state[:lag], action[-lag:]
    return state, action


def _canonical_dimension_name(value: object) -> str:
    tokens = [
        token
        for token in re.split(r"[^a-z0-9一-鿿]+", str(value or "").casefold())
        if token
    ]
    ignored = {
        "observation", "observations", "obs", "state", "states", "joint", "joints",
        "action", "actions", "command", "commands", "cmd", "target", "targets",
        "desired", "position", "positions", "value", "values",
    }
    normalized = [token for token in tokens if token not in ignored]
    return "/".join(normalized)


def _bundle_dimension_names(bundle: dict, kind: str, width: int) -> list[str]:
    output = [""] * max(0, int(width))
    for binding in bundle.get("bindings") or []:
        if str(binding.get("kind") or "") != kind:
            continue
        start = max(0, int(binding.get("column_start") or 0))
        end = min(len(output), int(binding.get("column_end") or start))
        names = [str(item) for item in binding.get("dimension_names") or []]
        if end > start and len(names) == end - start:
            output[start:end] = names
    return output


def _semantic_dimension_pairs(
    state_width: int,
    action_width: int,
    state_names: list[str] | None,
    action_names: list[str] | None,
    *,
    require_semantic_mapping: bool,
) -> list[tuple[int, int, str]]:
    if state_names is not None and action_names is not None:
        state_map: dict[str, list[int]] = {}
        action_map: dict[str, list[int]] = {}
        for index, value in enumerate(state_names[:state_width]):
            name = _canonical_dimension_name(value)
            if name:
                state_map.setdefault(name, []).append(index)
        for index, value in enumerate(action_names[:action_width]):
            name = _canonical_dimension_name(value)
            if name:
                action_map.setdefault(name, []).append(index)
        pairs = [
            (state_map[name][0], action_map[name][0], name)
            for name in sorted(state_map.keys() & action_map.keys())
            if len(state_map[name]) == 1 and len(action_map[name]) == 1
        ]
        if pairs:
            return pairs
    if require_semantic_mapping:
        raise ValueError("State/Action 缺少可唯一匹配的 dimension_names，S2 不执行位置猜测")
    dimensions = min(state_width, action_width)
    return [(index, index, f"column/{index}") for index in range(dimensions)]


def _retain_minimum_mask_runs(mask: np.ndarray, minimum_frames: int) -> np.ndarray:
    output = np.zeros(np.asarray(mask, dtype=bool).shape, dtype=bool)
    for start, end in _mask_ranges(np.asarray(mask, dtype=bool)):
        if end - start + 1 >= max(1, int(minimum_frames)):
            output[start:end + 1] = True
    return output


def _local_state_action_masks(
    state: np.ndarray,
    action: np.ndarray,
    lag_frames: int,
    fps: float,
    threshold: float,
    review_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    frame_count = int(state.shape[0])
    invalid = np.zeros(frame_count, dtype=bool)
    review = np.zeros(frame_count, dtype=bool)
    local_scores = np.full(frame_count, np.nan, dtype=np.float64)
    aligned_state, aligned_action = _aligned_pair(state, action, lag_frames)
    if aligned_state.shape[0] < 4:
        return invalid, review, local_scores, {"window_frames": 0, "evaluated_frame_count": 0}
    if lag_frames > 0:
        aligned_frames = np.arange(lag_frames, frame_count, dtype=np.int64)
    elif lag_frames < 0:
        aligned_frames = np.arange(0, frame_count + lag_frames, dtype=np.int64)
    else:
        aligned_frames = np.arange(frame_count, dtype=np.int64)
    state_diff = np.diff(aligned_state, axis=0)
    action_diff = np.diff(aligned_action, axis=0)
    stacked = np.concatenate((np.abs(state_diff), np.abs(action_diff)), axis=0)
    deadband = np.maximum(1e-5, np.nanpercentile(stacked, 25, axis=0) * 0.1)
    active = (np.abs(state_diff) > deadband) | (np.abs(action_diff) > deadband)
    agreement = np.sign(state_diff) == np.sign(action_diff)
    diff_frames = aligned_frames[1:]
    window_frames = max(5, int(math.ceil(S2_LOCAL_WINDOW_SECONDS * max(0.01, fps))))
    half_window = window_frames // 2
    minimum_comparisons = max(5, int(math.ceil(S2_LOCAL_MINIMUM_SPAN_SECONDS * max(0.01, fps))))
    for position, frame in enumerate(diff_frames):
        left = max(0, position - half_window)
        right = min(len(diff_frames), position + half_window + 1)
        selected = active[left:right]
        if int(selected.sum()) < minimum_comparisons:
            continue
        local_scores[frame] = float(np.mean(agreement[left:right][selected]))
    evaluated = np.isfinite(local_scores)
    raw_invalid = evaluated & (local_scores < review_floor)
    raw_review = evaluated & (local_scores >= review_floor) & (local_scores < threshold)
    minimum_run = max(2, int(math.ceil(S2_LOCAL_MINIMUM_SPAN_SECONDS * max(0.01, fps))))
    invalid = _retain_minimum_mask_runs(raw_invalid, minimum_run)
    review = _retain_minimum_mask_runs(raw_review | raw_invalid, minimum_run) & ~invalid
    return invalid, review, local_scores, {
        "window_frames": window_frames,
        "minimum_comparisons": minimum_comparisons,
        "minimum_run_frames": minimum_run,
        "evaluated_frame_count": int(evaluated.sum()),
    }


def estimate_state_action_alignment(
    state: np.ndarray,
    action: np.ndarray,
    fps: float,
    max_lag_seconds: float,
    da_threshold: float,
    action_representation: str,
    state_dimension_names: list[str] | None = None,
    action_dimension_names: list[str] | None = None,
    *,
    require_semantic_mapping: bool = False,
) -> dict:
    pairs = _semantic_dimension_pairs(
        int(state.shape[1]),
        int(action.shape[1]),
        state_dimension_names,
        action_dimension_names,
        require_semantic_mapping=require_semantic_mapping,
    )
    dimensions = len(pairs)
    if dimensions <= 0 or state.shape[0] < 12:
        raise ValueError("state/action shared dimensions are unavailable")
    state_values = _smooth(state[:, [item[0] for item in pairs]])
    action_values = _smooth(action[:, [item[1] for item in pairs]])
    if action_representation == "delta":
        action_values = np.cumsum(action_values, axis=0)
    elif action_representation == "velocity":
        action_values = np.cumsum(action_values, axis=0) / max(fps, 0.01)
    elif action_representation != "absolute":
        raise ValueError("Action 表示类型未知；需由 Qwen 或人工确认 absolute/delta/velocity")
    max_lag = max(1, min(int(round(max_lag_seconds * fps)), max(1, state.shape[0] // 4)))
    metrics = []
    for dimension in range(dimensions):
        s = state_values[:, dimension]
        a = action_values[:, dimension]
        # Estimate lag from motion rather than absolute position.  A local
        # offset or one bad segment must not shift the alignment of the whole
        # Episode.
        s_motion = np.diff(s, prepend=s[:1])
        a_motion = np.diff(a, prepend=a[:1])
        s_normalized = (s_motion - np.mean(s_motion)) / max(np.std(s_motion), 1e-9)
        a_normalized = (a_motion - np.mean(a_motion)) / max(np.std(a_motion), 1e-9)
        best_lag = 0
        best_correlation = -math.inf
        for lag in range(-max_lag, max_lag + 1):
            aligned_state, aligned_action = _aligned_pair(s_normalized, a_normalized, lag)
            if aligned_state.size < 8:
                continue
            correlation = float(np.mean(aligned_state * aligned_action))
            if correlation > best_correlation:
                best_correlation = correlation
                best_lag = lag
        aligned_state, aligned_action = _aligned_pair(s, a, best_lag)
        state_diff = np.diff(aligned_state)
        action_diff = np.diff(aligned_action)
        deadband = max(1e-5, float(np.nanpercentile(np.abs(np.concatenate([state_diff, action_diff])), 25)) * 0.1)
        active = (np.abs(state_diff) > deadband) | (np.abs(action_diff) > deadband)
        if active.sum() < 5:
            continue
        agreement = float(np.mean(np.sign(state_diff[active]) == np.sign(action_diff[active])))
        metrics.append({
            "dimension": dimension,
            "semantic_name": pairs[dimension][2],
            "state_dimension": pairs[dimension][0],
            "action_dimension": pairs[dimension][1],
            "lag_frames": best_lag,
            "correlation": round(best_correlation, 6),
            "directional_agreement": round(agreement, 6),
        })
    if not metrics:
        raise ValueError("state/action trajectories have insufficient variation")
    lag_frames = int(round(float(np.median([item["lag_frames"] for item in metrics]))))
    agreement = float(np.median([item["directional_agreement"] for item in metrics]))
    review_floor = min(da_threshold, 0.6)
    global_verdict = "pass" if agreement >= da_threshold else "review" if agreement >= review_floor else "reject_candidate"
    invalid_mask, review_mask, local_agreement, local_metrics = _local_state_action_masks(
        state_values,
        action_values,
        lag_frames,
        fps,
        da_threshold,
        review_floor,
    )
    verdict = "reject_candidate" if invalid_mask.any() else "review" if review_mask.any() else "pass"
    return {
        "lag_frames": lag_frames,
        "lag_seconds": round(lag_frames / max(fps, 0.01), 6),
        "directional_agreement": round(agreement, 6),
        "threshold": da_threshold,
        "passed": verdict == "pass",
        "verdict": verdict,
        "global_verdict": global_verdict,
        "shared_dimensions": dimensions,
        "semantic_mapping": [
            {"state_dimension": state_index, "action_dimension": action_index, "name": name}
            for state_index, action_index, name in pairs
        ],
        "action_representation": action_representation,
        "action_integrated": action_representation in {"delta", "velocity"},
        "dimensions": metrics[:32],
        "invalid_mask": invalid_mask,
        "review_mask": review_mask,
        "local_directional_agreement": local_agreement,
        "local_metrics": local_metrics,
        "invalid_frame_count": int(invalid_mask.sum()),
        "review_frame_count": int(review_mask.sum()),
        "invalid_ranges": [{"start_frame": start, "end_frame": end} for start, end in _mask_ranges(invalid_mask)],
        "review_ranges": [{"start_frame": start, "end_frame": end} for start, end in _mask_ranges(review_mask)],
    }


def detect_extreme_values(
    matrix: np.ndarray,
    alpha: float,
    exempt_columns: set[int] | None = None,
    reference_values: np.ndarray | None = None,
    reference_scope: str = "episode_limited",
    cohort_id: str | None = None,
) -> dict:
    values = np.asarray(matrix, dtype=np.float64)
    reference = np.asarray(reference_values if reference_values is not None else values, dtype=np.float64)
    dimensions = min(values.shape[1], reference.shape[1])
    if dimensions <= 0:
        return {"mask": np.zeros(values.shape[0], dtype=bool), "event_count": 0, "alpha": alpha, "q01": [], "q99": [], "exempt_columns": sorted(exempt_columns or set()), "reference_scope": reference_scope, "cohort_id": cohort_id}
    q01 = np.nanpercentile(reference[:, :dimensions], 1, axis=0)
    q99 = np.nanpercentile(reference[:, :dimensions], 99, axis=0)
    spread = np.maximum(q99 - q01, 1e-9)
    lower = q01 - alpha * spread
    upper = q99 + alpha * spread
    dimension_flags = np.zeros_like(values, dtype=bool)
    dimension_flags[:, :dimensions] = (values[:, :dimensions] < lower) | (values[:, :dimensions] > upper)
    for column in exempt_columns or set():
        if 0 <= column < dimension_flags.shape[1]:
            dimension_flags[:, column] = False
    flags = np.any(dimension_flags & np.isfinite(values), axis=1)
    return {
        "mask": flags,
        "event_count": int(flags.sum()),
        "alpha": alpha,
        "q01": [round(float(value), 8) for value in q01[:32]],
        "q99": [round(float(value), 8) for value in q99[:32]],
        "exempt_columns": sorted(exempt_columns or set()),
        "reference_scope": reference_scope,
        "cohort_id": cohort_id,
        "reference_frame_count": int(reference.shape[0]),
    }


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    if not values.size:
        return []
    changes = np.flatnonzero(np.diff(values.astype(np.int8)) != 0) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes - 1, values.size - 1]
    return [(int(start), int(end)) for start, end in zip(starts, ends) if values[start]]


def merge_dense_quality_marks(mask: np.ndarray, fps: float, gap_seconds: float = QUALITY_MARK_GAP_SECONDS) -> np.ndarray:
    """Join low-quality runs whose boundary timestamps are less than the threshold apart."""
    output = np.asarray(mask, dtype=bool).copy()
    rate = max(0.01, float(fps))
    threshold = max(0.0, float(gap_seconds))
    runs = _runs(output)
    for (_, left_end), (right_start, _) in zip(runs, runs[1:]):
        if (right_start - left_end) / rate < threshold:
            output[left_end + 1:right_start] = True
    return output


def inspect_video_quality(media: dict, action: np.ndarray | None, request: CurationJobRequest, progress: Callable[[float, str], None]) -> dict:
    frame_count = int(media.get("frame_count") or 0)
    fps = max(0.01, float(media.get("fps") or 30.0))
    requested_sample_fps = max(15.0, float(request.video_sample_fps))
    target_sample_fps = min(fps, requested_sample_fps)
    step = max(1, int(math.floor(fps / max(0.01, target_sample_fps) + 1e-9)))
    effective_sample_fps = min(fps, fps / step)
    indices = list(range(0, frame_count, step))
    if indices and indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    brightness: list[float] = []
    blur_scores: list[float] = []
    differences: list[float] = []
    corrupt: list[bool] = []
    previous = None
    capture = cv2.VideoCapture(str(media.get("path") or "")) if media.get("type") != "images" else None
    try:
        for position, frame_index in enumerate(indices):
            if media.get("type") == "images":
                frame = cv2.imread(media["frames"][frame_index])
                ok = frame is not None
            else:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
            if not ok or frame is None:
                brightness.append(0.0)
                blur_scores.append(0.0)
                differences.append(0.0)
                corrupt.append(True)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
            brightness.append(float(np.mean(small)))
            blur_scores.append(float(cv2.Laplacian(small, cv2.CV_64F).var()))
            differences.append(float(np.mean(cv2.absdiff(small, previous))) if previous is not None else math.inf)
            corrupt.append(False)
            previous = small
            if position % 20 == 0:
                progress(position / max(1, len(indices)) * 100, f"视频质量采样 {position + 1}/{len(indices)}")
    finally:
        if capture is not None:
            capture.release()
    brightness_array = np.asarray(brightness)
    blur_array = np.asarray(blur_scores)
    difference_array = np.asarray(differences)
    corrupt_mask = np.asarray(corrupt, dtype=bool)
    black_mask = brightness_array < request.black_level_threshold
    raw_blur = blur_array < request.blur_laplacian_threshold
    blur_mask = raw_blur & (np.convolve(raw_blur.astype(np.int8), np.ones(3, dtype=np.int8), mode="same") >= 2)
    low_motion = difference_array < request.static_difference_threshold
    static_mask = np.zeros(len(indices), dtype=bool)
    minimum_static_samples = max(2, int(math.ceil(request.static_duration_seconds * effective_sample_fps)))
    for start, end in _runs(low_motion):
        if end - start + 1 >= minimum_static_samples:
            static_mask[start:end + 1] = True
    protected = np.zeros(len(indices), dtype=bool)
    if action is not None and action.shape[0] > 2:
        action_values = _filled(action)
        motion = np.linalg.norm(np.diff(action_values, axis=0, prepend=action_values[:1]), axis=1)
        threshold = float(np.nanpercentile(motion, 95))
        if threshold > 0:
            for position, frame_index in enumerate(indices):
                left, right = max(0, frame_index - step), min(motion.size, frame_index + step + 1)
                protected[position] = bool(np.any(motion[left:right] >= threshold))
    static_confirmed = static_mask & ~protected if action is not None else np.zeros(len(indices), dtype=bool)
    blur_mask &= ~protected
    invalid_sample_mask = corrupt_mask | black_mask | blur_mask | static_confirmed
    invalid_frames = np.zeros(frame_count, dtype=bool)
    review_frames = np.zeros(frame_count, dtype=bool)
    findings = []
    for position, frame_index in enumerate(indices):
        left = max(0, frame_index - step // 2)
        right = min(frame_count - 1, frame_index + step // 2)
        issues = []
        if corrupt_mask[position]: issues.append("corrupt")
        if black_mask[position]: issues.append("black")
        if blur_mask[position]: issues.append("blur")
        if static_confirmed[position]: issues.append("static")
        if static_mask[position] and protected[position]: issues.append("protected_key_motion")
        if invalid_sample_mask[position]:
            invalid_frames[left:right + 1] = True
        elif static_mask[position]:
            review_frames[left:right + 1] = True
        if issues:
            findings.append({
                "frame": frame_index,
                "issues": issues,
                "brightness": round(float(brightness_array[position]), 4),
                "laplacian": round(float(blur_array[position]), 4),
                "frame_difference": None if not math.isfinite(float(difference_array[position])) else round(float(difference_array[position]), 4),
                "protected": bool(protected[position]),
            })
    return {
        "invalid_mask": invalid_frames,
        "review_mask": review_frames,
        "sample_indices": indices,
        "brightness": brightness_array,
        "blur": blur_array,
        "difference": difference_array,
        "findings": findings[:MAX_REPORT_FINDINGS],
        "metrics": {
            "sample_count": len(indices),
            "requested_sample_fps": round(requested_sample_fps, 6),
            "effective_sample_fps": round(effective_sample_fps, 6),
            "sample_step_frames": int(step),
            "black_sample_count": int(black_mask.sum()),
            "blur_sample_count": int(blur_mask.sum()),
            "static_sample_count": int(static_confirmed.sum()),
            "corrupt_sample_count": int(corrupt_mask.sum()),
            "protected_sample_count": int(protected.sum()),
        },
    }


def _stage(stage_id: str, status: str, message: str, metrics: dict | None = None) -> dict:
    name = dict(STAGE_DEFINITIONS)[stage_id]
    return {"id": stage_id, "name": name, "status": status, "message": message, "metrics": metrics or {}}


def _curation_time_sync(manifest: dict, episode: dict, media: dict) -> dict:
    alignment = scan_episode_sensor_alignment(
        manifest,
        episode,
        force=False,
        reference_media_file_id=str(media.get("file_id") or "") or None,
    )
    alignment = retime_sensor_alignment(alignment, media)
    alignment.setdefault("reference_video", {
        "file_id": media.get("file_id"),
        "stream_name": media.get("stream_name"),
        "relative_path": media.get("relative_path"),
        "fps": media.get("fps") or episode.get("fps") or 30.0,
        "frame_count": media.get("frame_count") or episode.get("frame_count") or 0,
        "duration": media.get("duration") or episode.get("duration"),
    })
    return validate_episode_time_sync(manifest, episode, alignment)


def _mask_findings(mask: np.ndarray, stage_id: str, severity: str, reason: str, fps: float, confidence: float) -> list[dict]:
    return [{
        "stage": stage_id,
        "severity": severity,
        "state": "invalid" if severity == "reject" else "uncertain",
        "start_frame": start,
        "end_frame": end,
        "start_time": round(start / fps, 6),
        "end_time": round(end / fps, 6),
        "reason": reason,
        "confidence": confidence,
    } for start, end in _runs(mask)]


def _retime_boolean_mask(mask: np.ndarray, source_positions: list[float]) -> np.ndarray:
    source = np.asarray(mask, dtype=bool).reshape(-1)
    positions = np.asarray(source_positions, dtype=np.float64).reshape(-1)
    if not source.size or not positions.size or not np.isfinite(positions).all():
        return source
    left = np.floor(positions).astype(np.int64)
    right = np.ceil(positions).astype(np.int64)
    valid = (left >= 0) & (right < source.size)
    output = np.ones(len(positions), dtype=bool)
    output[valid] = source[left[valid]] | source[right[valid]]
    return output


def _combined_segments(frame_count: int, fps: float, invalid: np.ndarray, review: np.ndarray, findings: list[dict]) -> list[dict]:
    state = np.zeros(frame_count, dtype=np.int8)
    state[review] = 1
    state[invalid] = 2
    changes = np.flatnonzero(np.diff(state) != 0) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes - 1, frame_count - 1]
    labels = {0: ("valid", "通过数据质量检查", 0.92), 1: ("uncertain", "需要人工复核的数据质量片段", 0.62), 2: ("invalid", "数据质量检查命中", 0.86)}
    segments = []
    for start, end in zip(starts, ends):
        value = int(state[start])
        name, default_reason, confidence = labels[value]
        reasons = sorted({item["reason"] for item in findings if item["start_frame"] <= end and item["end_frame"] >= start})
        segments.append({
            "start_frame": int(start),
            "end_frame": int(end),
            "start_time": round(int(start) / fps, 6),
            "end_time": round(int(end) / fps, 6),
            "state": name,
            "reason": "；".join(reasons) if reasons else default_reason,
            "confidence": confidence,
        })
    return segments


def _state_masks_from_segments(frame_count: int, segments: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    invalid = np.zeros(frame_count, dtype=bool)
    review = np.zeros(frame_count, dtype=bool)
    for segment in segments:
        if frame_count <= 0:
            break
        start = max(0, min(frame_count - 1, int(segment.get("start_frame") or 0)))
        end = max(start, min(frame_count - 1, int(segment.get("end_frame") or start)))
        state = str(segment.get("state") or "")
        if state == "invalid":
            invalid[start:end + 1] = True
        elif state == "uncertain":
            review[start:end + 1] = True
    review &= ~invalid
    return invalid, review


def resolve_post_vlm_review(precheck_review: np.ndarray, c1: dict, c2: dict) -> np.ndarray:
    # C1/C2 may add review evidence, but they do not evaluate or clear S3/C3
    # quality findings.  A review state can only be removed by a dedicated
    # resolver for the stage that created it.
    review = precheck_review.copy()
    for result in (c1, c2):
        mask = result.get("review_mask")
        if mask is not None:
            review |= np.asarray(mask, dtype=bool)
    return review


def curation_valid_ranges(report: dict, *, before_c2: bool = False) -> list[tuple[int, int]]:
    segments = report.get("pre_vlm_segments") if before_c2 else None
    if not isinstance(segments, list):
        segments = report.get("segments") or []
    return [
        (int(item.get("start_frame") or 0), int(item.get("end_frame") or item.get("start_frame") or 0))
        for item in segments
        if str(item.get("state") or "") == "valid"
    ]


def curation_vlm_ranges(report: dict) -> list[tuple[int, int]]:
    """Return every pre-C2 range except red/rejected segments."""
    segments = report.get("pre_vlm_segments") or report.get("segments") or []
    ranges: list[tuple[int, int]] = []
    for item in segments:
        if str(item.get("state") or "") == "invalid":
            continue
        start = int(item.get("start_frame") or 0)
        end = int(item.get("end_frame") or start)
        if ranges and start <= ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))
    return ranges


def behavior_matches_curation_ranges(behavior: dict | None, ranges: list[tuple[int, int]], media: dict | None = None) -> bool:
    if not behavior:
        return False
    stored = (behavior.get("sampling") or {}).get("allowed_ranges")
    if not isinstance(stored, list):
        return False
    normalized = [
        (int(item.get("start_frame") or 0), int(item.get("end_frame") or item.get("start_frame") or 0))
        for item in stored
        if isinstance(item, dict)
    ]
    if normalized != ranges:
        return False
    if media is not None:
        analysis_video = behavior.get("analysis_video") or {}
        if str(analysis_video.get("file_id") or "") != str(media.get("file_id") or ""):
            return False
        if int(analysis_video.get("frame_count") or 0) != int(media.get("frame_count") or 0):
            return False
        if not media_fingerprint_matches(analysis_video.get("fingerprint"), media):
            return False
    return True


def _motion_evidence(bundle: dict, frame_count: int) -> np.ndarray | None:
    scores: list[np.ndarray] = []
    for kind in ("joint", "action"):
        matrix = bundle.get(kind)
        if matrix is None or not np.asarray(matrix).size:
            continue
        values = _smooth(np.asarray(matrix, dtype=np.float64))
        if values.shape[0] != frame_count:
            continue
        if kind == "action" and bundle.get("action_representation") in {"delta", "velocity"}:
            centered = values
            scale = np.nanpercentile(np.abs(centered), 90, axis=0)
            scale = np.maximum(scale, 1e-9)
            score = np.sqrt(np.nanmean(np.square(centered / scale), axis=1))
        else:
            span = np.nanpercentile(values, 95, axis=0) - np.nanpercentile(values, 5, axis=0)
            span = np.maximum(np.abs(span), 1e-9)
            delta = np.diff(values, axis=0, prepend=values[:1])
            score = np.sqrt(np.nanmean(np.square(delta / span), axis=1))
        scores.append(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0))
    return np.maximum.reduce(scores) if scores else None


def _gripper_transition_evidence(bundle: dict, frame_count: int, eligible: np.ndarray) -> dict:
    scores: list[np.ndarray] = []
    source_kinds: list[str] = []
    for kind in ("joint", "action"):
        matrix = bundle.get(kind)
        if matrix is None or not np.asarray(matrix).size:
            continue
        values = np.asarray(matrix, dtype=np.float64)
        columns = sorted({
            int(index)
            for index in ((bundle.get("gripper_columns") or {}).get(kind) or set())
            if 0 <= int(index) < values.shape[1]
        })
        if not columns or values.shape[0] != frame_count:
            continue
        values = _smooth(values[:, columns])
        if kind == "action" and bundle.get("action_representation") in {"delta", "velocity"}:
            scale = np.nanpercentile(np.abs(values), 90, axis=0)
            scale = np.maximum(scale, 1e-9)
            score = np.sqrt(np.nanmean(np.square(values / scale), axis=1))
        else:
            span = np.nanpercentile(values, 95, axis=0) - np.nanpercentile(values, 5, axis=0)
            span = np.maximum(np.abs(span), 1e-9)
            delta = np.diff(values, axis=0, prepend=values[:1])
            score = np.sqrt(np.nanmean(np.square(delta / span), axis=1))
        scores.append(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0))
        source_kinds.append(kind)
    if not scores:
        return {
            "available": False,
            "score": None,
            "transition_mask": np.zeros(frame_count, dtype=bool),
            "threshold": None,
            "source_kinds": [],
        }
    score = np.maximum.reduce(scores)
    candidates = score[np.asarray(eligible, dtype=bool) & np.isfinite(score)]
    positive = candidates[candidates > 1e-9]
    threshold = max(1e-4, float(np.percentile(positive, 25)) * 0.5) if positive.size else 1e-4
    return {
        "available": True,
        "score": score,
        "transition_mask": score > threshold,
        "threshold": threshold,
        "source_kinds": source_kinds,
    }


def _contact_action_template(segment: dict) -> str | None:
    skill_text = str(segment.get("skill") or "").strip()
    skill = re.sub(r"[^a-z0-9]+", "", skill_text.casefold())
    if skill in {"grasp", "pinch", "clip", "suction", "catch", "takeover"}:
        return "grasp"
    if skill == "hold":
        return "hold"
    if skill == "release":
        return "release"
    if skill_text:
        return None
    phase = str(segment.get("phase_label") or segment.get("label") or "unknown").strip().casefold()
    phase = phase.replace("-", "_").replace(" ", "_")
    return phase if phase in {"grasp", "release"} else None


def inspect_behavior_state_consistency(
    behavior: dict | None,
    bundle: dict,
    preliminary_valid: np.ndarray,
    fps: float,
    tactile_evidence: dict | None = None,
) -> dict:
    frame_count = int(preliminary_valid.size)
    if not behavior:
        return {
            "status": "skipped",
            "message": "C2 需要先完成有效片段 VLM 标注",
            "review_mask": np.zeros(frame_count, dtype=bool),
            "findings": [],
            "metrics": {},
        }
    motion = _motion_evidence(bundle, frame_count)
    gripper = _gripper_transition_evidence(bundle, frame_count, preliminary_valid)
    tactile = tactile_evidence or {}
    tactile_contact = np.asarray(tactile.get("contact", np.zeros(frame_count, dtype=bool)), dtype=bool).reshape(-1)
    tactile_valid = np.asarray(tactile.get("valid_mask", np.zeros(frame_count, dtype=bool)), dtype=bool).reshape(-1)
    if tactile_contact.shape != (frame_count,) or tactile_valid.shape != (frame_count,):
        tactile_contact = np.zeros(frame_count, dtype=bool)
        tactile_valid = np.zeros(frame_count, dtype=bool)
    tactile_available = bool(tactile_valid.any())
    if motion is None and not tactile_available:
        return {
            "status": "skipped",
            "message": "C2 缺少可对齐的 State/Action 与触觉证据",
            "review_mask": np.zeros(frame_count, dtype=bool),
            "findings": [],
            "metrics": {
                "behavior_segment_count": len(behavior.get("segments") or []),
                "gripper_evidence_available": bool(gripper["available"]),
                "tactile_evidence_available": False,
                "tactile_contact_frame_count": 0,
            },
        }
    if motion is not None:
        candidates = motion[preliminary_valid & np.isfinite(motion)]
        positive = candidates[candidates > 1e-9]
        motion_threshold = max(1e-4, float(np.percentile(positive, 25)) * 0.5) if positive.size else 1e-4
    else:
        motion_threshold = None
    tactile_onset = np.zeros(frame_count, dtype=bool)
    tactile_offset = np.zeros(frame_count, dtype=bool)
    if frame_count > 1:
        valid_pairs = tactile_valid[1:] & tactile_valid[:-1]
        tactile_onset[1:] = valid_pairs & tactile_contact[1:] & ~tactile_contact[:-1]
        tactile_offset[1:] = valid_pairs & ~tactile_contact[1:] & tactile_contact[:-1]
    inactive_phases = {"idle", "observe", "reach", "withdraw", "unknown", "precheck_invalid"}
    minimum_frames = max(2, int(math.ceil(0.4 * fps)))
    review_mask = np.zeros(frame_count, dtype=bool)
    findings: list[dict] = []
    checked = 0
    mismatch_segments = 0
    contact_segments = 0
    supported_contact_segments = 0
    mismatch_contact_segments = 0
    insufficient_evidence_segments = 0
    ratios: list[float] = []
    evidence_audit: list[dict] = []
    for segment in behavior.get("segments") or []:
        if frame_count <= 0:
            break
        phase = str(segment.get("phase_label") or segment.get("label") or "unknown").strip().casefold().replace("-", "_").replace(" ", "_")
        template = _contact_action_template(segment)
        if template is None and phase in inactive_phases:
            continue
        start = max(0, min(frame_count - 1, int(segment.get("start_frame") or 0)))
        end = max(start, min(frame_count - 1, int(segment.get("end_frame") or start)))
        mask = preliminary_valid[start:end + 1]
        eligible_count = int(mask.sum())
        if eligible_count <= 0 or (template is None and eligible_count < minimum_frames):
            continue
        checked += 1
        active_ratio = None
        if motion is not None and motion_threshold is not None:
            active_ratio = float(np.mean(motion[start:end + 1][mask] > motion_threshold))
            ratios.append(active_ratio)
        audit = {
            "start_frame": start,
            "end_frame": end,
            "phase_label": phase,
            "skill": segment.get("skill"),
            "template": template or "motion",
            "eligible_frame_count": eligible_count,
            "active_motion_ratio": round(active_ratio, 6) if active_ratio is not None else None,
        }
        mismatch_reason: str | None = None
        outcome = "supported"
        if template is not None:
            contact_segments += 1
            tactile_slice_valid = tactile_valid[start:end + 1] & mask
            tactile_valid_count = int(tactile_slice_valid.sum())
            tactile_coverage = tactile_valid_count / max(1, eligible_count)
            minimum_tactile_count = min(eligible_count, max(1, int(math.ceil(eligible_count * 0.25))))
            segment_tactile_available = tactile_valid_count >= minimum_tactile_count
            contact_ratio = (
                float(np.mean(tactile_contact[start:end + 1][tactile_slice_valid]))
                if tactile_valid_count else None
            )
            onset_count = int((tactile_onset[start:end + 1] & mask).sum())
            offset_count = int((tactile_offset[start:end + 1] & mask).sum())
            gripper_transition_count = 0
            gripper_peak = None
            if gripper["available"]:
                gripper_transition_count = int((gripper["transition_mask"][start:end + 1] & mask).sum())
                eligible_scores = gripper["score"][start:end + 1][mask]
                gripper_peak = float(np.max(eligible_scores)) if eligible_scores.size else 0.0
            audit.update({
                "tactile_valid_frame_count": tactile_valid_count,
                "tactile_coverage": round(tactile_coverage, 6),
                "tactile_contact_ratio": round(contact_ratio, 6) if contact_ratio is not None else None,
                "tactile_onset_count": onset_count,
                "tactile_offset_count": offset_count,
                "gripper_transition_frame_count": gripper_transition_count,
                "gripper_transition_peak": round(gripper_peak, 8) if gripper_peak is not None else None,
            })
            if template == "grasp":
                if segment_tactile_available:
                    tactile_indices = np.flatnonzero(tactile_slice_valid)
                    end_contact = bool(tactile_contact[start + int(tactile_indices[-1])]) if tactile_indices.size else False
                    if onset_count > 0 or end_contact:
                        supported_contact_segments += 1
                        audit["support"] = "tactile_contact_established"
                    else:
                        mismatch_reason = "C2 Grasp 缺少触觉接触建立或片段末接触证据"
                elif gripper["available"]:
                    if gripper_transition_count > 0:
                        supported_contact_segments += 1
                        audit["support"] = "gripper_state_transition"
                    else:
                        mismatch_reason = "C2 Grasp 缺少夹爪状态变化证据"
                elif active_ratio is not None:
                    outcome = "insufficient_evidence"
                    audit["support"] = "legacy_motion_fallback" if active_ratio >= 0.08 else None
                    if active_ratio < 0.08:
                        mismatch_reason = "C2 Grasp 缺少夹爪/触觉证据，且同步 State/Action 运动不足"
                else:
                    outcome = "insufficient_evidence"
            elif template == "hold":
                if segment_tactile_available:
                    if contact_ratio is not None and contact_ratio >= 0.6:
                        supported_contact_segments += 1
                        audit["support"] = "sustained_tactile_contact"
                    else:
                        mismatch_reason = "C2 Hold 的有效触觉帧接触比例低于 60%"
                else:
                    outcome = "insufficient_evidence"
                    audit["support"] = "static_gripper_is_not_hold_proof" if gripper["available"] else None
            elif template == "release":
                if segment_tactile_available:
                    tactile_indices = np.flatnonzero(tactile_slice_valid)
                    first_contact = bool(tactile_contact[start + int(tactile_indices[0])]) if tactile_indices.size else False
                    end_contact = bool(tactile_contact[start + int(tactile_indices[-1])]) if tactile_indices.size else False
                    if offset_count > 0 or (first_contact and not end_contact):
                        supported_contact_segments += 1
                        audit["support"] = "tactile_contact_released"
                    else:
                        mismatch_reason = "C2 Release 缺少触觉接触消失证据"
                elif gripper["available"]:
                    if gripper_transition_count > 0:
                        supported_contact_segments += 1
                        audit["support"] = "gripper_state_transition"
                    else:
                        mismatch_reason = "C2 Release 缺少夹爪状态变化证据"
                elif active_ratio is not None:
                    outcome = "insufficient_evidence"
                    audit["support"] = "legacy_motion_fallback" if active_ratio >= 0.08 else None
                    if active_ratio < 0.08:
                        mismatch_reason = "C2 Release 缺少夹爪/触觉证据，且同步 State/Action 运动不足"
                else:
                    outcome = "insufficient_evidence"
        else:
            if active_ratio is None:
                outcome = "insufficient_evidence"
            elif active_ratio < 0.08:
                mismatch_reason = f"C2 VLM 阶段 {phase} 缺少同步 State/Action 运动证据"
        if mismatch_reason is not None:
            outcome = "mismatch"
            mismatch_segments += 1
            if template is not None:
                mismatch_contact_segments += 1
            segment_mask = np.zeros(frame_count, dtype=bool)
            segment_mask[start:end + 1] = preliminary_valid[start:end + 1]
            review_mask |= segment_mask
            findings.extend(_mask_findings(segment_mask, "c2", "review", mismatch_reason, fps, 0.72))
        elif outcome == "insufficient_evidence":
            insufficient_evidence_segments += 1
        audit["outcome"] = outcome
        evidence_audit.append(audit)
    metrics = {
        "behavior_segment_count": len(behavior.get("segments") or []),
        "checked_active_segment_count": checked,
        "mismatch_segment_count": mismatch_segments,
        "mismatch_frame_count": int(review_mask.sum()),
        "motion_threshold": round(motion_threshold, 8) if motion_threshold is not None else None,
        "mean_active_motion_ratio": round(float(np.mean(ratios)) if ratios else 0.0, 6),
        "signal_sources": [kind for kind in ("joint", "action") if bundle.get(kind) is not None],
        "contact_segment_count": contact_segments,
        "supported_contact_segment_count": supported_contact_segments,
        "mismatch_contact_segment_count": mismatch_contact_segments,
        "insufficient_evidence_segment_count": insufficient_evidence_segments,
        "gripper_evidence_available": bool(gripper["available"]),
        "gripper_evidence_sources": list(gripper["source_kinds"]),
        "gripper_transition_threshold": round(float(gripper["threshold"]), 8) if gripper["threshold"] is not None else None,
        "tactile_evidence_available": tactile_available,
        "tactile_contact_frame_count": int((tactile_contact & tactile_valid).sum()),
        "tactile_bindings": list(tactile.get("bindings") or []),
        "tactile_errors": list(tactile.get("errors") or []),
        "evidence_audit": evidence_audit,
    }
    has_warning = bool(findings) or insufficient_evidence_segments > 0
    if findings:
        message = f"发现 {mismatch_segments} 个视频-State 不一致片段"
        if insufficient_evidence_segments:
            message += f"；另有 {insufficient_evidence_segments} 个片段缺少动作特定证据"
    elif insufficient_evidence_segments:
        message = f"已复核 {checked} 个有效操作片段；{insufficient_evidence_segments} 个片段证据不足，未自动判为不一致"
    else:
        message = f"已复核 {checked} 个有效操作片段"
    return {
        "status": "warning" if has_warning else "completed",
        "message": message,
        "review_mask": review_mask,
        "findings": findings,
        "metrics": metrics,
    }


def inspect_instruction_consistency(
    behavior: dict | None,
    episode: dict,
    eligible: np.ndarray,
    fps: float,
) -> dict:
    frame_count = int(eligible.size)
    review_mask = np.zeros(frame_count, dtype=bool)
    if not behavior:
        return {"status": "skipped", "message": "C1 需要 VLM 任务标注", "review_mask": review_mask, "findings": [], "metrics": {}}
    task_label = str(behavior.get("task_label") or "other").strip().casefold()
    confidence = float(behavior.get("confidence") or 0.0)
    expected_text = " ".join(str(episode.get(key) or "") for key in ("name", "task", "instruction", "description"))
    ignored = {"episode", "ep", "trial", "demo", "task", "data", "dataset"}

    def tokens(value: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9\u4e00-\u9fff]+", value.casefold().replace("_", " ").replace("-", " "))
            if len(token) > 1 and token not in ignored and not token.isdigit()
        }

    expected_tokens = tokens(expected_text)
    actual_tokens = tokens(task_label)
    generic = task_label in {"", "other", "unknown"}
    explicit_mismatch = bool(expected_tokens and actual_tokens and expected_tokens.isdisjoint(actual_tokens))
    low_confidence = confidence < 0.35
    mismatch = generic or explicit_mismatch or low_confidence
    findings: list[dict] = []
    if mismatch:
        review_mask |= eligible
        reason = "C1 VLM 任务类别不明确" if generic else "C1 VLM 任务类别与 Episode 指令不一致" if explicit_mismatch else "C1 VLM 任务标注置信度不足"
        findings = _mask_findings(review_mask, "c1", "review", reason, fps, 0.74)
    overlap = sorted(expected_tokens & actual_tokens)
    metrics = {
        "task_label": task_label,
        "confidence": round(confidence, 6),
        "expected_tokens": sorted(expected_tokens),
        "task_tokens": sorted(actual_tokens),
        "matching_tokens": overlap,
        "review_frame_count": int(review_mask.sum()),
    }
    return {
        "status": "warning" if mismatch else "completed",
        "message": "VLM 任务与指令需要复核" if mismatch else "VLM 任务与 Episode 指令一致",
        "review_mask": review_mask,
        "findings": findings,
        "metrics": metrics,
    }


def curation_preflight(dataset_id: str, episode_id: str, media_file_id: str | None = None) -> dict:
    manifest = get_manifest(dataset_id)
    episode = next((item for item in manifest.get("episodes", []) if item.get("id") == episode_id), None)
    if episode is None:
        raise KeyError(episode_id)
    candidates = _signal_candidates(manifest, episode)
    kinds = {item["kind"] for item in candidates}
    action_representations = {
        str(item.get("representation") or "unknown")
        for item in candidates
        if item["kind"] == "action" and str(item.get("representation") or "unknown") != "unknown"
    }
    action_semantics_known = len(action_representations) == 1
    joint_dimension_names = [
        str(name)
        for item in candidates
        if item["kind"] == "joint"
        for name in item.get("dimension_names") or []
    ]
    action_dimension_names = [
        str(name)
        for item in candidates
        if item["kind"] == "action"
        for name in item.get("dimension_names") or []
    ]
    semantic_pairs_ready = bool(
        {
            _canonical_dimension_name(name)
            for name in joint_dimension_names
            if _canonical_dimension_name(name)
        }
        & {
            _canonical_dimension_name(name)
            for name in action_dimension_names
            if _canonical_dimension_name(name)
        }
    )
    generated_action = load_episode_action_mapping(dataset_id, episode_id, manifest=manifest)
    generated_action_ready = bool(generated_action and generated_action.get("artifact_path"))
    s2_ready = generated_action_ready or (
        {"joint", "action"} <= kinds
        and action_semantics_known
        and semantic_pairs_ready
    )
    try:
        _find_transform_source(manifest, episode)
        action_source_ready = True
    except (KeyError, OSError, RuntimeError, ValueError):
        action_source_ready = False
    behavior = load_behavior_annotation(dataset_id, episode_id)
    media = episode_media(episode, media_file_id or episode.get("primary_media_file_id"))
    nexus_mode = _nexus_mode_enabled(manifest)
    nexus_tactile_ready = nexus_mode and bool(_nexus_pressure_records(manifest, episode))
    s1_ready = bool(kinds or nexus_tactile_ready)
    hand_visibility_ready, hand_visibility_reason = _hand_visibility_capability(manifest)
    c3_status = "ready" if hand_visibility_ready else "warning"
    c3_target = media.get("stream_name") or media.get("relative_path") or "所选视频"
    c3_message = (
        f"将检查 {c3_target} 的画质与整手可见性"
        if hand_visibility_ready
        else f"将检查 {c3_target} 的视频画质；整手可见性已停用：{hand_visibility_reason}"
    )
    stages = [
        _stage(
            "t0",
            "ready",
            "任何清洗、VLM、Action 与导出开始前，先建立所选视频到各传感器行的统一时间映射",
        ),
        _stage(
            "p1",
            "ready" if nexus_mode else "skipped",
            "将检查左右同步压力流是否缺文件、缺行、partial 或空值；压力值为 0 不判错"
            if nexus_mode else "非 Nexus 模式，不运行压力空值错误检测",
        ),
        _stage(
            "s1",
            "ready" if s1_ready else "skipped",
            "将运行 Joint/Action Jerk、endpose rot6d 与 Nexus 触觉孤立突变检查"
            if kinds and nexus_tactile_ready else "将运行 Nexus 触觉孤立突变检查；持续接触与数值 0 不判错"
            if nexus_tactile_ready else "将运行通用 Jerk 与 endpose rot6d 相对旋转突变检查"
            if kinds else "没有已识别的 Joint/Action 或 Nexus 触觉数值流",
        ),
        _stage(
            "s2",
            "ready" if s2_ready else "pending" if action_source_ready else "skipped",
            "将校验已生成 Action 的源轨迹、预测帧和数值一致性"
            if generated_action_ready
            else "可运行"
            if s2_ready
            else "需要先生成 Action，或提供表示类型和 dimension_names 均明确且可匹配的 State/Joint 与 Action",
        ),
        _stage("s3", "ready" if kinds else "skipped", "可运行" if kinds else "没有已识别的数值流"),
        _stage("s4", "skipped", "当前版本未实现 FK 一致性计算；即使检测到 URDF/Pinocchio 也不会宣称可运行"),
        _stage("s5", "skipped", "当前版本仅记录坐标修正建议，不自动改写坐标；标定信息不会触发执行"),
        _stage("c3", c3_status, c3_message, {
            "video_quality": True,
            "can_hand_visibility": hand_visibility_ready,
            "hand_visibility_reason": hand_visibility_reason,
        }),
        _stage("c1", "pending", "初筛后将校验已有 VLM 是否匹配有效区间" if behavior else "将在 S1-S5/C3 初筛后标注有效片段"),
        _stage(
            "c2",
            "ready" if behavior and kinds else "pending" if kinds else "skipped",
            "将在有效片段 VLM 标注后运行" if kinds else "缺少可对齐的 State/Action 数值流",
        ),
    ]
    return {
        "dataset_id": dataset_id,
        "episode_id": episode_id,
        "media_file_id": media.get("file_id"),
        "signal_candidate_count": len(candidates),
        "stages": stages,
        "source_policy": "Preflight is read-only. Curation results are staged only in .alicePD.",
    }


def run_episode_curation(
    dataset_id: str,
    manifest: dict,
    episode: dict,
    media: dict,
    request: CurationJobRequest,
    progress: Callable[[float, str], None],
    s3_reference: dict | None = None,
    behavior_checks: bool | None = None,
    generated_s2_override: dict | None = None,
    run_id: str | None = None,
    timeline_id: str | None = None,
) -> dict:
    frame_count = int(media.get("frame_count") or episode.get("frame_count") or 0)
    fps = max(0.01, float(media.get("fps") or episode.get("fps") or 30.0))
    media_file_id = str(media.get("file_id") or "") or None
    if frame_count < 4:
        raise ValueError("Episode 帧数不足，无法运行数据清洗")
    root = Path(manifest["root_path"]).expanduser().resolve()
    generated_action_report = load_episode_action_mapping(dataset_id, str(episode["id"]), manifest=manifest)
    candidate_paths = {
        str(candidate.get("relative_path") or "").replace("\\", "/")
        for candidate in _signal_candidates(manifest, episode)
        if str(candidate.get("relative_path") or "")
    }
    candidate_paths.update(
        str(candidate.get("relative_path") or "").replace("\\", "/")
        for candidate in _nexus_pressure_records(manifest, episode)
        if str(candidate.get("relative_path") or "")
        and (root / str(candidate.get("relative_path") or "")).is_file()
    )
    media_relative = str(media.get("relative_path") or episode.get("relative_path") or "").replace("\\", "/")
    if media_relative:
        candidate_paths.add(media_relative)
    generated_source_relative = str((generated_action_report or {}).get("source", {}).get("relative_path") or "").replace("\\", "/")
    if generated_source_relative:
        candidate_paths.add(generated_source_relative)
    calibration_source_relative = str(
        (((manifest.get("camera_calibration") or {}).get("hand_projection") or {}).get("source_relative_path") or "")
    ).replace("\\", "/")
    if calibration_source_relative and (root / calibration_source_relative).is_file():
        candidate_paths.add(calibration_source_relative)
    try:
        _, visibility_source_relative, _ = _find_transform_source(manifest, episode)
        candidate_paths.add(visibility_source_relative.replace("\\", "/"))
    except RuntimeError:
        visibility_source_relative = ""
    source_paths = sorted(candidate_paths)
    source_signatures = [source_signature(root, relative) for relative in source_paths]
    reference_signatures = list((s3_reference or {}).get("source_signatures") or [])
    if reference_signatures:
        reference_matches, reference_changed = source_signatures_match(root, reference_signatures)
        if not reference_matches:
            raise RuntimeError(f"S3 cohort source changed before paper curation: {reference_changed or 'unknown source'}")
    progress(3, "T0 建立视频与传感器统一时间轴")
    alignment = _curation_time_sync(manifest, episode, media)
    progress(9, "读取 Joint / Action 数值流")
    bundle = _load_signal_bundle(manifest, episode, alignment, frame_count=frame_count)
    c3_signal_bundle = {
        **bundle,
        "joint": None if bundle.get("joint") is None else np.asarray(bundle["joint"], dtype=np.float64).copy(),
    }
    joint = bundle["joint"]
    action = bundle["action"]
    alignment_streams = list(alignment.get("streams") or [])
    alignment_skipped = list(alignment.get("skipped_files") or [])
    stages: list[dict] = [_stage(
        "t0",
        "completed",
        f"统一时间轴已建立：参考视频 {float((alignment.get('reference_video') or {}).get('fps') or fps):.3f} FPS，已映射 {len(alignment_streams)} 路数据流",
        {
            "reference_video": alignment.get("reference_video"),
            "reference_timestamp_source": alignment.get("reference_timestamp_source"),
            "stream_count": len(alignment_streams),
            "timestamp_aligned_count": sum(item.get("mode") == "timestamp_nearest" for item in alignment_streams),
            "rate_multiplier_count": sum(item.get("mode") == "rate_multiplier" for item in alignment_streams),
            "prealigned_count": sum(item.get("mode") in {"prealigned_master_clock", "paired_frame_index"} for item in alignment_streams),
            "rate_conflict_count": sum(bool(item.get("rate_conflict")) for item in alignment_streams),
            "skipped_file_count": len(alignment_skipped),
            "artifact_path": alignment.get("artifact_path"),
        },
    )]
    findings: list[dict] = []
    invalid = np.zeros(frame_count, dtype=bool)
    review = np.zeros(frame_count, dtype=bool)
    motion_score = np.zeros(frame_count, dtype=np.float64)
    bundle_valid = np.asarray(bundle.get("valid_mask", np.ones(frame_count, dtype=bool)), dtype=bool)
    if bundle_valid.shape != (frame_count,):
        bundle_valid = np.zeros(frame_count, dtype=bool)
        bundle["warnings"].append("Signal validity mask length did not match the selected video")
    source_invalid = ~bundle_valid
    s1_repair_patch: dict | None = None
    s1_repair_summary = {
        "enabled": bool(request.repair_s1_spikes),
        "method": "bounded_cubic_and_rot6d_slerp_v1",
        "tactile_policy": "detect_only_no_source_rewrite",
        "max_repair_frames": int(request.s1_max_repair_frames),
        "repaired_frame_count": 0,
        "repaired_range_count": 0,
        "artifact_path": None,
    }
    signal_parts = [item for item in (joint, action) if item is not None]
    progress(12, "P1 Nexus 压力空值错误检测")
    pressure_integrity = inspect_nexus_pressure_integrity(manifest, episode, alignment, frame_count)
    pressure_empty = np.asarray(pressure_integrity["empty_mask"], dtype=bool)
    if pressure_empty.shape != (frame_count,):
        pressure_empty = np.ones(frame_count, dtype=bool)
        pressure_integrity["status"] = "warning"
        pressure_integrity["message"] = "压力完整性结果长度错误，已将当前 Episode 标为异常"
    if pressure_integrity["enabled"]:
        invalid |= pressure_empty
        for side in NEXUS_PRESSURE_EXPECTED_SIDES:
            findings.extend(_mask_findings(
                np.asarray(pressure_integrity["side_masks"][side], dtype=bool),
                "p1",
                "reject",
                f"P1 Nexus {side} 压力传感器为空（缺行/partial/无值；数值 0 不属于错误）",
                fps,
                0.99,
            ))
    stages.append(_stage(
        "p1",
        str(pressure_integrity["status"]),
        str(pressure_integrity["message"]),
        {**pressure_integrity["metrics"], "bindings": pressure_integrity["bindings"]},
    ))
    progress(16, "S1 Nexus 触觉孤立突变检查")
    tactile_s1 = inspect_nexus_tactile_sudden_changes(
        manifest,
        episode,
        alignment,
        frame_count,
        request.sudden_change_sigma,
    )
    tactile_mask = np.asarray(tactile_s1["mask"], dtype=bool)
    tactile_score = np.asarray(tactile_s1["score"], dtype=np.float64)
    if tactile_mask.shape != (frame_count,) or tactile_score.shape != (frame_count,):
        tactile_mask = np.zeros(frame_count, dtype=bool)
        tactile_score = np.zeros(frame_count, dtype=np.float64)
        tactile_s1["status"] = "warning"
        tactile_s1["message"] = "Nexus 触觉突变结果长度错误，未将错误结果用于判废"
    if signal_parts or tactile_s1["enabled"]:
        progress(18, "S1 Joint/Action、rot6d 与 Nexus 触觉突变检查")
        if signal_parts:
            combined = np.concatenate(signal_parts, axis=1)
            s1_candidate_before = detect_sudden_changes(combined, request.sudden_change_sigma)
            s1_before, projection_introduced_before, _ = _guard_projection_introduced_s1(
                s1_candidate_before,
                bundle.get("projection_raw_joint"),
                action,
                request.sudden_change_sigma,
            )
            rot6d_before = inspect_rot6d_jumps(bundle, request.sudden_change_sigma)
            signal_before_mask = s1_before["mask"] | rot6d_before["mask"]
            if request.repair_s1_spikes and signal_before_mask.any():
                repair = repair_s1_bundle(
                    bundle,
                    request.sudden_change_sigma,
                    request.s1_max_repair_frames,
                    blocked_repair_frames=projection_introduced_before,
                )
                combined = repair["values"]
                joint_width = int(joint.shape[1]) if joint is not None else 0
                if joint is not None:
                    joint = combined[:, :joint_width]
                    bundle["joint"] = joint
                if action is not None:
                    action = combined[:, joint_width:]
                    bundle["action"] = action
                signal_parts = [item for item in (joint, action) if item is not None]
                s1 = repair["generic_after"]
                rot6d = repair["rot_after"]
                repaired_count = int(repair["repaired_mask"].sum())
                s1_repair_summary.update({
                    "repaired_frame_count": repaired_count,
                    "repaired_range_count": len(repair["ranges"]),
                    "repaired_ranges": [
                        {"start_frame": start, "end_frame": end}
                        for start, end in repair["ranges"]
                    ],
                })
                if repaired_count:
                    repair_path = s1_repair_path(dataset_id, str(episode["id"]), media_file_id, run_id)
                    s1_repair_summary["artifact_path"] = str(repair_path)
                    s1_repair_patch = {
                        "schema": S1_REPAIR_SCHEMA,
                        "dataset_id": dataset_id,
                        "episode_id": str(episode["id"]),
                        "full_run_id": run_id,
                        "timeline_id": timeline_id,
                        "created_at": _utc_now(),
                        "method": s1_repair_summary["method"],
                        "max_repair_frames": int(request.s1_max_repair_frames),
                        "repaired_frame_count": repaired_count,
                        "repaired_ranges": s1_repair_summary["repaired_ranges"],
                        "entries": repair["entries"],
                    }
            else:
                s1 = s1_before
                rot6d = rot6d_before
            s1, projection_introduced_after, _ = _guard_projection_introduced_s1(
                s1,
                bundle.get("projection_raw_joint"),
                action,
                request.sudden_change_sigma,
            )
            s1, sustained_motion_review = _downgrade_sustained_motion_s1(s1, fps)
            s1_review_mask = projection_introduced_after | sustained_motion_review
        else:
            signal_before_mask = np.zeros(frame_count, dtype=bool)
            projection_introduced_before = np.zeros(frame_count, dtype=bool)
            projection_introduced_after = np.zeros(frame_count, dtype=bool)
            sustained_motion_review = np.zeros(frame_count, dtype=bool)
            s1_review_mask = np.zeros(frame_count, dtype=bool)
            s1_candidate_before = {
                "mask": np.zeros(frame_count, dtype=bool),
                "score": np.zeros(frame_count, dtype=np.float64),
                "event_count": 0,
            }
            s1 = {
                "mask": np.zeros(frame_count, dtype=bool),
                "score": np.zeros(frame_count, dtype=np.float64),
                "event_count": 0,
            }
            rot6d = {
                "mask": np.zeros(frame_count, dtype=bool),
                "score": np.zeros(frame_count, dtype=np.float64),
                "event_count": 0,
                "group_count": 0,
                "groups": [],
            }
        before_mask = signal_before_mask | tactile_mask
        s1_mask = s1["mask"] | rot6d["mask"] | tactile_mask
        s1_reject_mask = s1_mask | source_invalid
        invalid |= s1_reject_mask
        review |= s1_review_mask & ~invalid
        motion_score = np.maximum(motion_score, np.maximum(np.maximum(s1["score"], rot6d["score"]), tactile_score))
        findings.extend(_mask_findings(s1["mask"], "s1", "reject", "S1 突变/加速度/Jerk 异常", fps, 0.88))
        findings.extend(_mask_findings(
            projection_introduced_after,
            "s1",
            "review",
            "S1 已应用投影归正引入新突变；原始轨迹未命中硬性异常",
            fps,
            0.76,
        ))
        findings.extend(_mask_findings(
            sustained_motion_review,
            "s1",
            "review",
            "S1 重复突变点跨越持续运动区间，降为待复核",
            fps,
            0.7,
        ))
        findings.extend(_mask_findings(rot6d["mask"], "s1", "reject", "S1 endpose rot6d 相对旋转突变或 6D 基向量无效", fps, 0.94))
        for side in NEXUS_PRESSURE_EXPECTED_SIDES:
            findings.extend(_mask_findings(
                np.asarray(tactile_s1["side_masks"][side], dtype=bool),
                "s1",
                "reject",
                f"S1 Nexus {side} 触觉孤立突变（持续接触与数值 0 不判错）",
                fps,
                0.92,
            ))
        findings.extend(_mask_findings(source_invalid, "s1", "reject", "S1 同步映射越界或源数据 partial/无效", fps, 0.97))
        stage_status = "warning" if s1_review_mask.any() or (tactile_s1["status"] == "warning" and not s1_reject_mask.any()) else "completed"
        stages.append(_stage(
            "s1",
            stage_status,
            f"检测到 {int(s1_reject_mask.sum())} 个坏帧、{int(s1_review_mask.sum())} 个待复核帧；其中触觉突变 {int(tactile_mask.sum())} 帧",
            {
            "flagged_frame_count_before_repair": int(before_mask.sum()),
            "flagged_frame_count": int(s1_reject_mask.sum()),
            "source_invalid_frame_count": int(source_invalid.sum()),
            "generic_candidate_frame_count_before_guard": int(np.asarray(s1_candidate_before["mask"], dtype=bool).sum()),
            "generic_jump_frame_count": s1["event_count"],
            "projection_correction_active": bool(bundle.get("projection_correction_active")),
            "projection_s1_policy": "derived_correction_may_not_create_hard_reject",
            "projection_introduced_review_frame_count": int(projection_introduced_after.sum()),
            "sustained_motion_review_frame_count": int(sustained_motion_review.sum()),
            "rot6d_jump_frame_count": rot6d["event_count"],
            "rot6d_group_count": rot6d["group_count"],
            "rot6d_groups": rot6d["groups"],
            "tactile_jump_frame_count": int(tactile_mask.sum()),
            "tactile": tactile_s1["metrics"],
            "tactile_bindings": tactile_s1["bindings"],
            "sigma": request.sudden_change_sigma,
                **s1_repair_summary,
            },
        ))
    else:
        stages.append(_stage("s1", "skipped", "没有可读取的 Joint/Action 或 Nexus 触觉数值流"))

    progress(32, "S2 State-Action 对齐与导出一致性")
    generated_s2 = generated_s2_override
    if generated_s2 is None and generated_action_report:
        generated_s2 = validate_episode_action_mapping(dataset_id, manifest, episode, frame_count)
    if generated_s2 is not None:
        s2_invalid = np.asarray(generated_s2["invalid_mask"], dtype=bool)
        if s2_invalid.shape != (frame_count,):
            s2_invalid = np.ones(frame_count, dtype=bool)
            generated_s2["verdict"] = "reject_candidate"
            generated_s2["error"] = "Action 校验结果帧数与视频不一致"
        invalid |= s2_invalid
        if s2_invalid.any():
            findings.extend(_mask_findings(
                s2_invalid,
                "s2",
                "reject",
                "S2 派生 Action 与源轨迹、预测目标帧或文件索引不一致",
                fps,
                0.96,
            ))
        s2_metrics = {key: value for key, value in generated_s2.items() if key != "invalid_mask"}
        if generated_s2["verdict"] == "pass":
            stages.append(_stage("s2", "completed", "派生 Action 与源轨迹及预测帧一致", s2_metrics))
        else:
            stages.append(_stage("s2", "warning", f"发现 {int(s2_invalid.sum())} 个 Action 不一致帧", s2_metrics))
    elif joint is not None and action is not None:
        try:
            s2 = estimate_state_action_alignment(
                joint,
                action,
                fps,
                request.max_lag_seconds,
                request.directional_agreement_threshold,
                bundle["action_representation"],
                _bundle_dimension_names(bundle, "joint", int(joint.shape[1])),
                _bundle_dimension_names(bundle, "action", int(action.shape[1])),
                require_semantic_mapping=True,
            )
            s2_invalid = np.asarray(s2["invalid_mask"], dtype=bool)
            s2_review = np.asarray(s2["review_mask"], dtype=bool) & ~s2_invalid
            invalid |= s2_invalid
            review |= s2_review & ~invalid
            findings.extend(_mask_findings(
                s2_invalid,
                "s2",
                "reject",
                f"S2 局部 State-Action 方向一致率低于 {min(request.directional_agreement_threshold, 0.6):.3f}",
                fps,
                0.94,
            ))
            findings.extend(_mask_findings(
                s2_review,
                "s2",
                "review",
                f"S2 局部 State-Action 方向一致率低于 {request.directional_agreement_threshold:.3f}",
                fps,
                0.72,
            ))
            stage_message = (
                "语义维度与局部趋势一致"
                if s2["verdict"] == "pass"
                else f"检测到 {int(s2_invalid.sum())} 个坏帧、{int(s2_review.sum())} 个待复核帧"
            )
            s2_metrics = {
                key: value
                for key, value in s2.items()
                if key not in {"invalid_mask", "review_mask", "local_directional_agreement"}
            }
            stages.append(_stage("s2", "completed" if s2["verdict"] == "pass" else "warning", stage_message, s2_metrics))
        except ValueError as exc:
            stages.append(_stage("s2", "skipped", str(exc)))
    else:
        stages.append(_stage("s2", "skipped", "需要同时存在 State/Joint 与 Action 数值流"))

    progress(45, "S3 分位极值检查")
    if signal_parts:
        combined = np.concatenate(signal_parts, axis=1)
        joint_dims = joint.shape[1] if joint is not None else 0
        exempt = set(bundle["gripper_columns"]["joint"])
        exempt.update(joint_dims + value for value in bundle["gripper_columns"]["action"])
        reference_values = (s3_reference or {}).get("matrix") if s3_reference else None
        reference_scope = str((s3_reference or {}).get("scope") or "episode_limited")
        cohort_id = (s3_reference or {}).get("cohort_id")
        s3 = detect_extreme_values(combined, request.outlier_alpha, exempt, reference_values, reference_scope, cohort_id)
        if bundle["semantic_dimensions_known"] and reference_scope == "cohort":
            invalid |= s3["mask"]
            findings.extend(_mask_findings(s3["mask"], "s3", "reject", "S3 超出 q01/q99 扩展区间", fps, 0.82))
            stage_status = "completed"
            stage_message = f"检测到 {s3['event_count']} 个极值帧"
        else:
            review |= s3["mask"]
            findings.extend(_mask_findings(s3["mask"], "s3", "review", "S3 极值候选；维度名/夹爪索引未知", fps, 0.62))
            stage_status = "warning"
            stage_message = f"检测到 {s3['event_count']} 个候选；仅当前 EP 统计或维度语义不足，未自动判废"
        stages.append(_stage("s3", stage_status, stage_message, {key: value for key, value in s3.items() if key != "mask"}))
    else:
        stages.append(_stage("s3", "skipped", "没有可读取的数值流"))

    progress(55, "检查 FK、坐标与跨模态先决条件")
    preflight = curation_preflight(dataset_id, str(episode["id"]), media.get("file_id"))
    preflight_stages = {item["id"]: item for item in preflight["stages"]}
    stages.append(_stage("s4", "skipped", preflight_stages["s4"]["message"]))
    stages.append(_stage("s5", "skipped", "检测到标定也只生成修正建议；当前版本不自动改写坐标" if preflight_stages["s5"]["status"] == "ready" else preflight_stages["s5"]["message"]))
    can_hand_visibility, hand_visibility_reason = _hand_visibility_capability(manifest)
    required_hand_sides = _required_hand_sides(manifest, episode, generated_s2, generated_action_report)
    progress(58, "C3 检查整手是否完整位于画面内")
    if not can_hand_visibility:
        hand_visibility = _hand_visibility_skipped(
            frame_count,
            f"整手可见性不可用：{hand_visibility_reason or '格式预检未启用该能力'}",
            required_hand_sides,
        )
    elif not required_hand_sides:
        hand_visibility = _hand_visibility_skipped(
            frame_count,
            "整手可见性未执行：未发现明确的 Action/元数据手侧，不能默认要求左右手",
        )
    else:
        hand_visibility = inspect_full_hand_visibility(
            manifest,
            episode,
            media,
            required_hand_sides,
            signal_bundle=c3_signal_bundle,
        )
    hand_invalid = np.asarray(hand_visibility["invalid_mask"], dtype=bool)
    hand_review = np.asarray(hand_visibility.get("review_mask", np.zeros(frame_count, dtype=bool)), dtype=bool)
    if hand_visibility.get("available") and hand_invalid.shape == (frame_count,):
        invalid |= hand_invalid
        if hand_review.shape == (frame_count,):
            review |= hand_review & ~invalid
        findings.extend(_mask_findings(
            hand_invalid,
            "c3",
            "reject",
            f"C3 MANO 超过 60% 关节不可见（检查 {'+'.join(required_hand_sides)}）",
            fps,
            0.92,
        ))
        findings.extend(_mask_findings(
            hand_review,
            "c3",
            "review",
            f"C3 MANO 仅部分关节可见，标黄待复核（检查 {'+'.join(required_hand_sides)}）",
            fps,
            0.72,
        ))
    progress(62, "C3 黑帧、模糊与长静止检查")
    quality = inspect_video_quality(media, action, request, lambda value, message: progress(62 + value * 0.25, message))
    invalid |= quality["invalid_mask"]
    review |= quality["review_mask"] & ~invalid
    findings.extend(_mask_findings(quality["invalid_mask"], "c3", "reject", "C3 黑帧/损坏/持续模糊/长静止", fps, 0.84))
    findings.extend(_mask_findings(quality["review_mask"] & ~quality["invalid_mask"], "c3", "review", "C3 静止片段缺少足够动作证据，需人工复核", fps, 0.62))
    invalid_before_gap_merge = invalid.copy()
    invalid = merge_dense_quality_marks(invalid, fps, request.quality_gap_merge_seconds)
    merged_invalid_gaps = invalid & ~invalid_before_gap_merge
    review &= ~invalid
    review_before_gap_merge = review.copy()
    review = merge_dense_quality_marks(review, fps, request.quality_gap_merge_seconds) & ~invalid
    merged_review_gaps = review & ~review_before_gap_merge
    findings.extend(_mask_findings(merged_invalid_gaps, "c3", "reject", f"低质量标记间隔小于 {request.quality_gap_merge_seconds:g} 秒，已合并为同一段", fps, 0.8))
    findings.extend(_mask_findings(merged_review_gaps, "c3", "review", f"待复核标记间隔小于 {request.quality_gap_merge_seconds:g} 秒，已合并为同一段", fps, 0.62))
    quality["metrics"].update({
        "quality_gap_merge_seconds": request.quality_gap_merge_seconds,
        "merged_invalid_gap_frame_count": int(merged_invalid_gaps.sum()),
        "merged_review_gap_frame_count": int(merged_review_gaps.sum()),
        "hand_visibility": hand_visibility["metrics"],
    })
    c3_message = f"采样 {quality['metrics']['sample_count']} 帧完成"
    if hand_visibility.get("available"):
        c3_message += f"；{hand_visibility['message']}"
    else:
        c3_message += "；整手可见性不可用"
    stages.append(_stage("c3", "completed" if hand_visibility.get("available") else "warning", c3_message, quality["metrics"]))

    preliminary_valid = ~invalid & ~review
    pre_vlm_segments = _combined_segments(frame_count, fps, invalid, review, findings)
    behavior = load_behavior_annotation(dataset_id, str(episode["id"])) if behavior_checks is not False else None
    if behavior_checks is True and behavior:
        stages.append(_stage("c1", "reused", "已复用有效片段 VLM 行为标注", {"task_label": behavior.get("task_label"), "confidence": behavior.get("confidence"), "segment_count": len(behavior.get("segments") or [])}))
        progress(88, "C2 基于 VLM、夹爪与触觉证据检查有效片段")
        tactile_evidence = load_nexus_tactile_evidence(manifest, episode, alignment, frame_count)
        c2 = inspect_behavior_state_consistency(
            behavior,
            bundle,
            preliminary_valid,
            fps,
            tactile_evidence=tactile_evidence,
        )
        review |= c2["review_mask"] & ~invalid
        findings.extend(c2["findings"])
        stages.append(_stage("c2", c2["status"], c2["message"], c2["metrics"]))
    elif behavior_checks is True:
        stages.append(_stage("c1", "skipped", "没有已有 VLM 行为标注；C2 不会提前执行"))
        stages.append(_stage("c2", "skipped", "需要先对 S1-S5/C3 有效片段完成 VLM 标注"))
    elif behavior_checks is False:
        stages.append(_stage("c1", "pending", "等待对 S1-S5/C3 有效片段进行 VLM 标注"))
        stages.append(_stage("c2", "pending", "将在有效片段 VLM 标注完成后执行"))
    elif behavior:
        stages.append(_stage("c1", "reused", "已复用现有 VLM 行为标注；未重复调用 Qwen", {"task_label": behavior.get("task_label"), "confidence": behavior.get("confidence"), "segment_count": len(behavior.get("segments") or [])}))
        stages.append(_stage("c2", "skipped", "此调用保持原有清洗行为，不自动执行 VLM 后 C2"))
    else:
        stages.append(_stage("c1", "skipped", "没有已有 VLM 行为标注"))
        stages.append(_stage("c2", "skipped", "此调用保持原有清洗行为，不自动执行 VLM 后 C2"))

    progress(90, "合并阶段结论并写入 .alicePD")
    findings = sorted(findings, key=lambda item: (item["start_frame"], item["stage"]))[:MAX_REPORT_FINDINGS]
    review &= ~invalid
    segments = _combined_segments(frame_count, fps, invalid, review, findings)
    sample_step = max(1, math.ceil(frame_count / 300))
    quality_by_frame = {frame: position for position, frame in enumerate(quality["sample_indices"])}
    samples = []
    for frame in range(0, frame_count, sample_step):
        nearest = min(quality_by_frame, key=lambda value: abs(value - frame)) if quality_by_frame else None
        quality_position = quality_by_frame.get(nearest) if nearest is not None else None
        issue_count = sum(1 for item in quality["findings"] if item["frame"] == nearest) if nearest is not None else 0
        samples.append({
            "frame": frame,
            "motion": round(float(motion_score[frame]), 6),
            "quality": round(max(0.0, 1.0 - min(1.0, issue_count / 2)), 6),
            "state": "invalid" if invalid[frame] else "uncertain" if review[frame] else "valid",
            "confidence": 0.86 if invalid[frame] else 0.62 if review[frame] else 0.92,
        })
    used_source_paths = sorted({
        *(item["relative_path"] for item in bundle["bindings"]),
        *(
            str(item.get("relative_path") or "")
            for item in pressure_integrity["bindings"]
            if str(item.get("relative_path") or "")
            and (root / str(item.get("relative_path") or "")).is_file()
        ),
        media_relative,
        generated_source_relative,
        str(hand_visibility.get("source_relative_path") or visibility_source_relative or ""),
        *(str(item.get("relative_path") or "") for item in reference_signatures),
    } - {""})
    signatures_by_path = {item["relative_path"]: item for item in source_signatures}
    signatures_by_path.update({str(item.get("relative_path") or ""): item for item in reference_signatures})
    missing_initial = [relative for relative in used_source_paths if relative not in signatures_by_path]
    if missing_initial:
        raise RuntimeError(f"Source set changed during paper curation before commit: {missing_initial[0]}")
    signatures = [signatures_by_path[relative] for relative in used_source_paths]
    allowed_paths = [str(item.get("relative_path") or "") for item in manifest.get("files", [])]
    matches, changed_path = source_signatures_match(root, signatures, allowed_paths=allowed_paths)
    if not matches:
        raise RuntimeError(f"Source changed during paper curation; report was not committed: {changed_path or 'unknown source'}")
    repair_path = s1_repair_path(dataset_id, str(episode["id"]), media_file_id, run_id)
    if s1_repair_patch is not None:
        s1_repair_patch["source_signatures"] = signatures
        _write_json_atomic(repair_path, s1_repair_patch)
    else:
        repair_path.unlink(missing_ok=True)
    summary = {
        "frame_count": frame_count,
        "valid_frame_count": int((~invalid & ~review).sum()),
        "review_frame_count": int(review.sum()),
        "invalid_frame_count": int(invalid.sum()),
        "invalid_segment_count": sum(item["state"] == "invalid" for item in segments),
        "stage_completed_count": sum(item["status"] in {"completed", "reused", "warning"} for item in stages),
        "stage_skipped_count": sum(item["status"] == "skipped" for item in stages),
        "recommendation": "exclude_episode" if invalid.all() else "review_and_apply" if invalid.any() or review.any() else "keep",
    }
    quality_evidence = build_quality_evidence(
        dataset_id=dataset_id,
        episode_id=str(episode["id"]),
        frame_count=frame_count,
        fps=fps,
        stages=stages,
        findings=findings,
        segments=segments,
        pipeline_version=CURATION_PIPELINE_VERSION,
        pipeline_schema=CURATION_SCHEMA,
        run_id=run_id,
        timeline_id=timeline_id,
        source_video=media,
        source_signatures=signatures,
        config={**request.model_dump(), "quality_gap_merge_rule": "strictly_less_than"},
        artifact_paths={
            "sensor_alignment": alignment.get("artifact_path"),
            "s1_repair": s1_repair_summary.get("artifact_path"),
            "smoothing": media.get("smoothing_artifact_path"),
        },
    )
    document = {
        "schema": CURATION_SCHEMA,
        "pipeline_version": CURATION_PIPELINE_VERSION,
        "pipeline_phase": "post_vlm" if behavior_checks is True and behavior else "pre_vlm" if behavior_checks is False else "legacy",
        "dataset_id": dataset_id,
        "episode_id": episode["id"],
        "full_run_id": run_id,
        "timeline_id": timeline_id,
        "full_run_stage": "curation_pre_vlm" if run_id else None,
        "episode_name": episode.get("name"),
        "created_at": _utc_now(),
        "source_policy": "Source dataset files are read-only. This report remains staged in .alicePD until reviewed and applied.",
        "paper_reference": "Qwen-RobotManip arXiv:2606.17846 sections 2.4 and 3.2",
        "source_video": {
            "file_id": media.get("file_id"),
            "stream_name": media.get("stream_name"),
            "relative_path": media.get("relative_path"),
            "fps": fps,
            "frame_count": frame_count,
            "source_frame_count": int(media.get("source_frame_count") or 0),
            "source_fps": float(media.get("source_fps") or 0.0),
            "source_frame_positions": list(media.get("source_frame_positions") or []),
            "smoothing_mode": media.get("smoothing_mode"),
            "smoothing_artifact_path": media.get("smoothing_artifact_path"),
            "smoothing_frame_audit_path": media.get("smoothing_frame_audit_path"),
            "pixel_transform_available": bool(media.get("pixel_transform_available")),
        },
        "source_signatures": signatures,
        "sensor_alignment": {
            "artifact_path": alignment.get("artifact_path"),
            "stream_count": len(alignment.get("streams") or []),
        },
        "pressure_integrity": {
            "enabled": bool(pressure_integrity["enabled"]),
            "status": pressure_integrity["status"],
            "message": pressure_integrity["message"],
            "metrics": pressure_integrity["metrics"],
            "bindings": pressure_integrity["bindings"],
        },
        "tactile_s1": {
            "enabled": bool(tactile_s1["enabled"]),
            "status": tactile_s1["status"],
            "message": tactile_s1["message"],
            "metrics": tactile_s1["metrics"],
            "bindings": tactile_s1["bindings"],
        },
        "s3_reference": {
            "scope": str((s3_reference or {}).get("scope") or "episode_limited"),
            "reference_policy": str((s3_reference or {}).get("reference_policy") or "episode_limited"),
            "cohort_id": (s3_reference or {}).get("cohort_id"),
            "episode_count": int((s3_reference or {}).get("episode_count") or 1),
            "cohort_episode_count": int((s3_reference or {}).get("cohort_episode_count") or 1),
            "reference_episode_ids": list((s3_reference or {}).get("reference_episode_ids") or []),
        },
        "stream_bindings": [
            {key: value for key, value in binding.items() if not key.startswith("_")}
            for binding in bundle["bindings"]
        ],
        "s1_repair": s1_repair_summary,
        "warnings": [
            *bundle["warnings"],
            *([] if hand_visibility.get("available") else [hand_visibility.get("message") or "整手可见性检查不可用"]),
        ],
        "config": {**request.model_dump(), "quality_gap_merge_rule": "strictly_less_than"},
        "stages": stages,
        "findings": findings,
        "pre_vlm_segments": pre_vlm_segments,
        "pre_vlm_valid_ranges": [
            {"start_frame": start, "end_frame": end}
            for start, end in curation_valid_ranges({"segments": pre_vlm_segments})
        ],
        "segments": segments,
        "samples": samples,
        "summary": summary,
        "quality_evidence": quality_evidence,
    }
    path = _write_curation_report(dataset_id, str(episode["id"]), media_file_id, document, run_id)
    staged_paths = [path, *([repair_path] if s1_repair_patch is not None else [])]
    change = record_change(
        dataset_id,
        "paper_curation",
        str(episode["id"]),
        f"Paper curation: {episode.get('name') or episode['id']}",
        staged_paths,
        summary,
        used_source_paths,
    )
    document["artifact_path"] = str(path)
    document["change"] = {"id": change["id"], "status": change["status"], "revision": change["revision"]}
    progress(100, "数据质量清洗报告已暂存")
    return document


def finalize_episode_curation(
    dataset_id: str,
    manifest: dict,
    episode: dict,
    behavior: dict | None,
    progress: Callable[[float, str], None],
    *,
    vlm_status: str = "completed",
    media_file_id: str | None = None,
    run_id: str | None = None,
    timeline_id: str | None = None,
    preliminary_report: dict | None = None,
) -> dict:
    report = preliminary_report or load_curation_report(dataset_id, str(episode["id"]), media_file_id, run_id)
    if report is None:
        raise RuntimeError("缺少 S1-S5/C3 初筛报告，不能执行 C2")
    frame_count = int((report.get("source_video") or {}).get("frame_count") or episode.get("frame_count") or 0)
    fps = max(0.01, float((report.get("source_video") or {}).get("fps") or episode.get("fps") or 30.0))
    pre_vlm_segments = report.get("pre_vlm_segments") or report.get("segments") or []
    invalid, precheck_review = _state_masks_from_segments(frame_count, pre_vlm_segments)
    preliminary_eligible = ~invalid
    progress(12, "读取 S1-S5/C3 有效片段与对齐数值流")
    source_video = report.get("source_video") or {}
    alignment = _curation_time_sync(manifest, episode, {
        "file_id": source_video.get("file_id"),
        "stream_name": source_video.get("stream_name"),
        "relative_path": source_video.get("relative_path"),
        "fps": source_video.get("fps") or fps,
        "frame_count": source_video.get("frame_count") or frame_count,
        "duration": source_video.get("duration"),
        "source_frame_count": source_video.get("source_frame_count"),
        "source_frame_positions": source_video.get("source_frame_positions") or [],
        "smoothing_mode": source_video.get("smoothing_mode"),
    })
    bundle = _load_signal_bundle(manifest, episode, alignment, frame_count=frame_count)
    apply_s1_repair_to_bundle(bundle, load_s1_repair(report))
    tactile_evidence = load_nexus_tactile_evidence(manifest, episode, alignment, frame_count)
    progress(45, "C2 基于有效片段 VLM、夹爪与触觉证据做一致性检查")
    c1 = inspect_instruction_consistency(behavior, episode, preliminary_eligible, fps)
    c2 = inspect_behavior_state_consistency(
        behavior,
        bundle,
        preliminary_eligible,
        fps,
        tactile_evidence=tactile_evidence,
    )
    review = resolve_post_vlm_review(precheck_review, c1, c2) & ~invalid
    findings = [item for item in report.get("findings") or [] if str(item.get("stage") or "") not in {"c1", "c2"}]
    findings.extend(c1["findings"])
    findings.extend(c2["findings"])
    findings = sorted(findings, key=lambda item: (item["start_frame"], item["stage"]))[:MAX_REPORT_FINDINGS]
    segments = _combined_segments(frame_count, fps, invalid, review, findings)
    stages = [item for item in report.get("stages") or [] if item.get("id") not in {"vlm", "c1", "c2"}]
    if behavior:
        sampling = behavior.get("sampling") or {}
        normalized_vlm_status = "reused" if vlm_status == "reused" else "completed"
        stages.append(_stage(
            "vlm",
            normalized_vlm_status,
            "已复用匹配当前非红片段的 VLM 标注，未请求 Qwen" if normalized_vlm_status == "reused" else "Qwen 已完成非红片段 VLM 标注",
            {
                "task_label": behavior.get("task_label"),
                "confidence": behavior.get("confidence"),
                "segment_count": len(behavior.get("segments") or []),
                "sampled_frame_count": len(sampling.get("frames") or []),
                "allowed_range_count": len(sampling.get("allowed_ranges") or []),
                "qwen_requested": normalized_vlm_status == "completed",
            },
        ))
        stages.append(_stage("c1", c1["status"], c1["message"], {
            **c1["metrics"],
            "segment_count": len(behavior.get("segments") or []),
            "sampled_frame_count": len(sampling.get("frames") or []),
        }))
    else:
        stages.append(_stage("vlm", "skipped", "S1-S5/C3 初筛后没有非红片段，未请求 Qwen", {"qwen_requested": False}))
        stages.append(_stage("c1", "skipped", "S1-S5/C3 初筛后没有有效片段，未调用 VLM"))
    stages.append(_stage("c2", c2["status"], c2["message"], c2["metrics"]))
    samples = []
    for sample in report.get("samples") or []:
        frame = max(0, min(frame_count - 1, int(sample.get("frame") or 0))) if frame_count else 0
        samples.append({
            **sample,
            "state": "invalid" if frame_count and invalid[frame] else "uncertain" if frame_count and review[frame] else "valid",
            "confidence": 0.86 if frame_count and invalid[frame] else 0.62 if frame_count and review[frame] else 0.92,
        })
    summary = {
        **(report.get("summary") or {}),
        "frame_count": frame_count,
        "valid_frame_count": int((~invalid & ~review).sum()),
        "review_frame_count": int(review.sum()),
        "invalid_frame_count": int(invalid.sum()),
        "invalid_segment_count": sum(item["state"] == "invalid" for item in segments),
        "stage_completed_count": sum(item["status"] in {"completed", "reused", "warning"} for item in stages),
        "stage_skipped_count": sum(item["status"] == "skipped" for item in stages),
        "recommendation": "exclude_episode" if invalid.all() else "review_and_apply" if invalid.any() or review.any() else "keep",
    }
    behavior_fingerprint = hashlib.sha256(json.dumps({
        "task_label": behavior.get("task_label"),
        "sampling": behavior.get("sampling"),
        "segments": behavior.get("segments"),
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest() if behavior else None
    document = {
        **report,
        "pipeline_version": CURATION_PIPELINE_VERSION,
        "pipeline_phase": "post_vlm",
        "updated_at": _utc_now(),
        "full_run_id": run_id or report.get("full_run_id"),
        "timeline_id": timeline_id or report.get("timeline_id"),
        "full_run_stage": "curation_post_vlm" if run_id else report.get("full_run_stage"),
        "behavior_fingerprint": behavior_fingerprint,
        "stages": stages,
        "findings": findings,
        "pre_vlm_segments": pre_vlm_segments,
        "segments": segments,
        "samples": samples,
        "summary": summary,
    }
    document["quality_evidence"] = build_quality_evidence(
        dataset_id=dataset_id,
        episode_id=str(episode["id"]),
        frame_count=frame_count,
        fps=fps,
        stages=stages,
        findings=findings,
        segments=segments,
        pipeline_version=CURATION_PIPELINE_VERSION,
        pipeline_schema=CURATION_SCHEMA,
        run_id=run_id or report.get("full_run_id"),
        timeline_id=timeline_id or report.get("timeline_id"),
        source_video=source_video,
        source_signatures=report.get("source_signatures") or [],
        config=report.get("config") or {},
        artifact_paths={
            "sensor_alignment": (report.get("sensor_alignment") or {}).get("artifact_path"),
            "s1_repair": (report.get("s1_repair") or {}).get("artifact_path"),
            "behavior": (behavior or {}).get("artifacts", {}).get("behavior"),
        },
    )
    resolved_media_file_id = str((report.get("source_video") or {}).get("file_id") or media_file_id or "") or None
    path = _write_curation_report(dataset_id, str(episode["id"]), resolved_media_file_id, document, run_id)
    source_paths = [str(item.get("relative_path") or "") for item in report.get("source_signatures") or []]
    change = record_change(
        dataset_id,
        "paper_curation",
        str(episode["id"]),
        f"Paper curation: {episode.get('name') or episode['id']}",
        [path],
        summary,
        source_paths,
    )
    document["artifact_path"] = str(path)
    document["change"] = {"id": change["id"], "status": change["status"], "revision": change["revision"]}
    progress(100, "C2 已合并到最终有效片段")
    return document


def _build_s3_references(
    manifest: dict,
    episodes: dict[str, dict],
    media_by_episode: dict[str, dict],
    episode_ids: list[str],
) -> dict[str, dict]:
    """Build Stage-3 quantile cohorts without mixing unknown embodiments."""
    if len(episode_ids) < 2:
        return {}
    groups: dict[tuple, list[dict]] = {}
    for episode_id in episode_ids:
        episode = episodes.get(episode_id)
        media = media_by_episode.get(episode_id)
        if not episode or not media:
            continue
        try:
            root = Path(manifest["root_path"]).expanduser().resolve()
            candidate_paths = sorted({
                str(item.get("relative_path") or "")
                for item in _signal_candidates(manifest, episode)
                if str(item.get("relative_path") or "")
            })
            initial_signatures = [source_signature(root, relative) for relative in candidate_paths]
            alignment = _curation_time_sync(manifest, episode, media)
            frame_count = int(media.get("frame_count") or episode.get("frame_count") or 0)
            bundle = _load_signal_bundle(manifest, episode, alignment, frame_count=frame_count)
            parts = [item for item in (bundle.get("joint"), bundle.get("action")) if item is not None]
            embodiment_ids = tuple(bundle.get("embodiment_ids") or ())
            if not parts or not embodiment_ids or not bundle.get("semantic_dimensions_known"):
                continue
            matrix = np.concatenate(parts, axis=1)
            names = tuple(
                name
                for item in bundle.get("bindings", [])
                for name in item.get("dimension_names", [])
            )
            key = (embodiment_ids, int(matrix.shape[1]), names)
            signature_map = {item["relative_path"]: item for item in initial_signatures}
            used_paths = sorted({str(item.get("relative_path") or "") for item in bundle.get("bindings", []) if item.get("relative_path")})
            used_signatures = [signature_map[path] for path in used_paths]
            matches, _ = source_signatures_match(root, used_signatures)
            if not matches:
                continue
            groups.setdefault(key, []).append({"episode_id": episode_id, "matrix": matrix, "source_signatures": used_signatures})
        except (OSError, RuntimeError, ValueError, KeyError):
            continue
    references: dict[str, dict] = {}
    for key, entries in groups.items():
        if len(entries) < 2:
            continue
        cohort_id = "cohort-" + hashlib.sha1(repr(key).encode("utf-8")).hexdigest()[:10]
        for target in entries:
            reference_entries = [entry for entry in entries if entry["episode_id"] != target["episode_id"]]
            if not reference_entries:
                continue
            capped: list[np.ndarray] = []
            for reference_entry in reference_entries:
                matrix = reference_entry["matrix"]
                if matrix.shape[0] > 20_000:
                    indices = np.linspace(0, matrix.shape[0] - 1, 20_000, dtype=np.int64)
                    matrix = matrix[indices]
                capped.append(matrix)
            reference = np.concatenate(capped, axis=0)
            reference_signatures = {
                str(signature.get("relative_path") or ""): signature
                for reference_entry in reference_entries
                for signature in reference_entry["source_signatures"]
            }
            episode_id = target["episode_id"]
            references[episode_id] = {
                "matrix": reference,
                "scope": "cohort",
                "reference_policy": "leave_one_episode_out",
                "cohort_id": cohort_id,
                "episode_count": len(reference_entries),
                "cohort_episode_count": len(entries),
                "reference_episode_ids": [entry["episode_id"] for entry in reference_entries],
                "source_signatures": list(reference_signatures.values()),
            }
    return references


def _full_action_config(request: CurationJobRequest | None) -> dict | None:
    if not request or not request.full_action_profile_id:
        return None
    requested = ActionMappingRequest(
        episode_ids=[request.episode_ids[0]],
        profile_id=request.full_action_profile_id,
        source_hand=request.full_action_source_hand,
        coordinate_frame=request.full_action_coordinate_frame,
        horizon_frames=request.full_action_horizon_frames,
    )
    return {
        "profile_id": requested.profile_id,
        "source_hand": requested.source_hand,
        "coordinate_frame": requested.coordinate_frame,
        "horizon_frames": requested.horizon_frames,
        "source": "full_pipeline_request",
    }


def _full_projection_config(manifest: dict, request: CurationJobRequest | None) -> dict:
    mode = dataset_mode(manifest)
    enabled = bool(
        request
        and request.full_pipeline
        and mode["projection_correction_backend"] == "egodex_mano_prior_v1"
    )
    return {
        "enabled": enabled,
        "backend": "mediapipe" if enabled else None,
        "adjustment_rate": FULL_EGODEX_PROJECTION_ADJUSTMENT_RATE if enabled else None,
        "adjustment_mode": "uniform" if enabled else None,
        "sample_fps": FULL_EGODEX_PROJECTION_SAMPLE_FPS if enabled else None,
        "wrist_point_source": "egodex" if enabled else None,
        "dataset_family": mode["family"],
        "nexus_isolation": mode["family"] == "nexus_multimodal",
    }


def _validate_curation_media(episode: dict, media: dict, *, full_pipeline: bool) -> None:
    episode_name = str(episode.get("name") or episode.get("id") or "Episode")
    stream_name = str(media.get("stream_name") or media.get("relative_path") or media.get("file_id") or "未命名媒体")
    modality = str(media.get("modality") or "").strip().casefold()
    media_type = str(media.get("type") or "").strip().casefold()
    is_depth = modality == "depth" or bool(media.get("is_depth_map")) or media_type == "raw_depth"
    # Older registered manifests predate modality-aware eligibility flags. A
    # decoded video/image is a safe RGB fallback; unknown binary streams never
    # receive this compatibility path.
    if not modality and media_type in {"video", "images"} and not is_depth:
        modality = "rgb"
    analysis_eligible = media.get("analysis_eligible")
    if analysis_eligible is None:
        analysis_eligible = modality == "rgb" and media_type in {"video", "images"}
    if modality != "rgb" or analysis_eligible is not True:
        if is_depth:
            raise ValueError(
                f"{episode_name} 选择的是原始 Depth 媒体 {stream_name}；"
                "数据质量清洗与 Full 流程只能使用 analysis_eligible=true 的 RGB 媒体。"
            )
        raise ValueError(
            f"{episode_name} 的媒体 {stream_name} 不具备分析资格；"
            "数据质量清洗与 Full 流程只能使用 modality=rgb 且 analysis_eligible=true 的媒体。"
        )
    if full_pipeline:
        missing = [
            f"{key}=true"
            for key in ("vlm_eligible", "smoothing_eligible")
            if media.get(key) is False or (media.get(key) is None and not analysis_eligible)
        ]
        if missing:
            raise ValueError(
                f"{episode_name} 的媒体 {stream_name} 不能运行 Full 流程；"
                f"所选 RGB 还必须满足 {', '.join(missing)}。"
            )


class CurationJobManager(CancellableJobMixin):
    def __init__(self, max_workers: int | None = None) -> None:
        if max_workers is None:
            try:
                max_workers = int(os.environ.get("ALICE_FULL_WORKERS", "2"))
            except ValueError:
                max_workers = 2
        self.max_workers = max(1, min(32, int(max_workers)))
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="alice-paper-curation")
        self._jobs: dict[str, dict] = {}
        self._reservations: dict[tuple[str, str], str] = {}
        self._lock = threading.RLock()
        self._init_cancellation()

    def runtime_config(self) -> dict:
        gpu_devices = [
            item.strip()
            for item in os.environ.get("ALICE_GPU_DEVICES", "").split(",")
            if item.strip()
        ]
        return {
            "workers": self.max_workers,
            "gpu_devices": gpu_devices,
            "video_encoder": os.environ.get("ALICE_VIDEO_ENCODER", "auto"),
            "video_accelerator": os.environ.get("ALICE_VIDEO_ACCELERATOR", "auto"),
            "opencv_threads": cv2.getNumThreads(),
            "episode_parallelism": "sharded_jobs",
        }

    def _on_cancelled_before_run(self, job_id: str) -> None:
        with self._lock:
            for key, owner in list(self._reservations.items()):
                if owner == job_id:
                    self._reservations.pop(key, None)

    def submit(self, dataset_id: str, request: CurationJobRequest) -> dict:
        manifest = get_manifest(dataset_id)
        episodes = {str(item["id"]): item for item in manifest.get("episodes", [])}
        episode_ids = list(dict.fromkeys(request.episode_ids))
        missing = [episode_id for episode_id in episode_ids if episode_id not in episodes]
        if missing:
            raise KeyError(missing[0])
        media_by_episode = {}
        for episode_id in episode_ids:
            media_id = request.media_file_ids.get(episode_id) or episodes[episode_id].get("primary_media_file_id")
            media = episode_media(episodes[episode_id], media_id)
            _validate_curation_media(episodes[episode_id], media, full_pipeline=request.full_pipeline)
            media_by_episode[episode_id] = media
        with self._lock:
            overlap = [episode_id for episode_id in episode_ids if (dataset_id, episode_id) in self._reservations]
            if overlap:
                raise RuntimeError(f"{episodes[overlap[0]]['name']} 已有数据清洗任务正在运行")
            job_id = uuid.uuid4().hex
            operation = "full_pipeline" if request.full_pipeline else "paper_curation"
            job = {
                "id": job_id,
                "run_id": job_id if request.full_pipeline else None,
                "kind": operation,
                "operation": operation,
                "dataset_id": dataset_id,
                "status": "queued",
                "progress": 0,
                "message": f"{'Full 流程' if request.full_pipeline else '数据质量清洗'}已排队 · {len(episode_ids)} Episodes",
                "episode_count": len(episode_ids),
                "completed_count": 0,
                "current_episode_id": None,
                "current_stage": "queued",
                "stages": [{"id": stage_id, "name": name, "status": "pending"} for stage_id, name in STAGE_DEFINITIONS],
                "result": None,
                "error": None,
            }
            self._jobs[job_id] = job
            self._register_cancellation(job_id)
            for episode_id in episode_ids:
                self._reservations[(dataset_id, episode_id)] = job_id
        self._executor.submit(self._run, job_id, dataset_id, episode_ids, media_by_episode, request)
        return dict(job)

    def get(self, job_id: str) -> dict:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return dict(self._jobs[job_id])

    def list(self, dataset_id: str, active_only: bool = False) -> dict:
        with self._lock:
            items = [dict(item) for item in self._jobs.values() if item.get("dataset_id") == dataset_id]
        if active_only:
            items = [item for item in items if item.get("status") in {"queued", "running", "cancelling"}]
        items.sort(key=lambda item: item.get("id", ""), reverse=True)
        return {"dataset_id": dataset_id, "items": items}

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(changes)

    def _run(self, job_id: str, dataset_id: str, episode_ids: list[str], media_by_episode: dict[str, dict], request: CurationJobRequest) -> None:
        results = []
        failures = []
        all_pairs: list[dict] = []
        vlm_requested_count = 0
        vlm_reused_count = 0
        vlm_skipped_count = 0
        total = len(episode_ids)
        run_id = job_id if request.full_pipeline else None
        timeline_ids: dict[str, str] = {}
        try:
            manifest = get_manifest(dataset_id)
            episodes = {str(item["id"]): item for item in manifest.get("episodes", [])}
            projection_documents: dict[str, dict | None] = {}
            projection_failures: dict[str, str] = {}
            media_by_episode = dict(media_by_episode)
            operation = "full_pipeline" if request.full_pipeline else "paper_curation"
            mode = dataset_mode(manifest)
            projection_config = _full_projection_config(manifest, request)
            automatic_egodex_projection = bool(projection_config["enabled"])
            pipeline_offset = 20.0 if automatic_egodex_projection else 0.0
            if run_id:
                request_payload = request.model_dump()
                request_payload["full_projection_correction"] = projection_config
                start_full_run(dataset_id, run_id, episode_ids, request_payload)
            output_root = Path(str(manifest["root_path"])).expanduser().resolve() / "output" / run_id if run_id else None
            output_setup_error = None
            if output_root is not None:
                try:
                    output_root.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    output_setup_error = str(exc)
            full_action = _full_action_config(request) if request.full_pipeline else None
            self._start_unless_cancelled(job_id, status="running", progress=1, message=f"{'Full' if request.full_pipeline else '后台清洗'}流程已启动 · 0/{total}")
            if automatic_egodex_projection:
                manifest[FULL_PROJECTION_SOURCE_OVERRIDES] = {}
                try:
                    with PROJECTION_RUNTIME_LOCK:
                        pose_status = registry.status().get("hand_pose") or {}
                        if not (registry.has_hand_pose and str(pose_status.get("backend") or "").casefold() == "mediapipe"):
                            self._update(
                                job_id,
                                progress=1.5,
                                current_stage="projection",
                                message="EgoDex Full 正在加载 MediaPipe 手部归正",
                            )
                            registry.configure_hand_pose(HandPoseModelConfig(kind="mediapipe", device="cpu", confidence=0.35))
                        for projection_position, episode_id in enumerate(episode_ids):
                            self._raise_if_cancelled(job_id)
                            projection_base = 2.0 + 18.0 * projection_position / max(1, total)
                            projection_span = 18.0 / max(1, total)

                            def projection_update(value: float, message: str, *, _episode_id: str = episode_id, _base: float = projection_base, _span: float = projection_span) -> None:
                                self._raise_if_cancelled(job_id)
                                self._update(
                                    job_id,
                                    progress=min(19.9, round(_base + _span * max(0.0, min(100.0, value)) / 100.0, 1)),
                                    current_episode_id=_episode_id,
                                    current_stage="projection",
                                    message=f"{episodes[_episode_id].get('name') or _episode_id} · EgoDex MediaPipe 手部归正 65% · {message}",
                                )

                            try:
                                projection_update(0.0, "T0 validating source video alignment")
                                _curation_time_sync(
                                    manifest,
                                    episodes[episode_id],
                                    media_by_episode[episode_id],
                                )
                                projection_document = run_projection_correction(
                                    dataset_id,
                                    manifest,
                                    episodes[episode_id],
                                    media_by_episode[episode_id],
                                    registry,
                                    projection_update,
                                    sample_fps=float(projection_config["sample_fps"]),
                                    adjustment_rate=float(projection_config["adjustment_rate"]),
                                    adjustment_mode=str(projection_config["adjustment_mode"]),
                                    wrist_point_source=str(projection_config["wrist_point_source"]),
                                    artifact_root=full_run_stage_dir(dataset_id, run_id, episode_id, "projection"),
                                    record_review_change=False,
                                    full_run_id=run_id,
                                )
                                projection_source = projection_source_from_document(projection_document)
                                if projection_source is None:
                                    raise RuntimeError("Full run projection artifact failed validation")
                                manifest[FULL_PROJECTION_SOURCE_OVERRIDES][episode_id] = projection_source
                                projected_media, _ = preferred_projection_media(
                                    manifest,
                                    episodes[episode_id],
                                    media_by_episode[episode_id],
                                )
                                media_by_episode[episode_id] = projected_media
                                projection_documents[episode_id] = projection_document
                            except JobCancelled:
                                raise
                            except Exception as exc:
                                projection_failures[episode_id] = str(exc)
                except JobCancelled:
                    raise
                except Exception as exc:
                    for episode_id in episode_ids:
                        projection_failures.setdefault(episode_id, str(exc))
            else:
                # Applied EgoDex corrections remain reusable for paper curation.
                # Nexus never enters either the automatic or applied projection path.
                for episode_id in episode_ids:
                    if mode["family"] == "nexus_multimodal":
                        projection_documents[episode_id] = None
                        continue
                    media_by_episode[episode_id], projection_documents[episode_id] = preferred_projection_media(
                        manifest,
                        episodes[episode_id],
                        media_by_episode[episode_id],
                    )

            if total > 1:
                self._update(job_id, progress=max(2.0, pipeline_offset), current_stage="s3", message=f"正在建立同 embodiment 的跨 EP S3 分位参考 · {total} Episodes")
            self._raise_if_cancelled(job_id)
            s3_episode_ids = [episode_id for episode_id in episode_ids if episode_id not in projection_failures]
            s3_references = _build_s3_references(manifest, episodes, media_by_episode, s3_episode_ids)
            for position, episode_id in enumerate(episode_ids):
                self._raise_if_cancelled(job_id)
                episode = episodes[episode_id]
                if run_id:
                    update_full_run_episode(
                        dataset_id,
                        run_id,
                        episode_id,
                        status="running",
                        media_file_id=str(media_by_episode[episode_id].get("file_id") or "") or None,
                    )
                base = pipeline_offset + position / max(1, total) * (100.0 - pipeline_offset)
                span = (100.0 - pipeline_offset) / max(1, total)

                def update(value: float, message: str) -> None:
                    self._raise_if_cancelled(job_id)
                    prefix = message.split(" ", 1)[0].casefold()
                    if prefix in {"t0", "p1", "s1", "s2", "s3", "s4", "s5", "c1", "c2", "c3"}:
                        stage_id = prefix
                    elif message.startswith("VLM"):
                        stage_id = "vlm"
                    elif message.startswith("视频平滑"):
                        stage_id = "smoothing"
                    elif message.startswith("导出"):
                        stage_id = "export"
                    else:
                        stage_id = "running"
                    self._update(
                        job_id,
                        progress=min(99, round(base + span * max(0, min(100, value)) / 100, 1)),
                        current_episode_id=episode_id,
                        current_stage=stage_id,
                        message=f"{episode.get('name') or episode_id} · {message} · {position + 1}/{total}",
                    )

                try:
                    if episode_id in projection_failures:
                        raise RuntimeError(f"EgoDex Full MediaPipe projection correction failed: {projection_failures[episode_id]}")
                    selected_media = media_by_episode[episode_id]
                    update(0.5, "T0 正在建立统一时间轴")
                    _curation_time_sync(manifest, episode, selected_media)
                    update(2.0, "T0 统一时间轴已就绪")
                    analysis_media = selected_media
                    smoothing_payload = None
                    action_stage_payload = None
                    action_stage_error = None
                    if request.full_pipeline:
                        def smoothing_update(value: float, message: str) -> None:
                            self._raise_if_cancelled(job_id)
                            update(max(0.0, min(100.0, value)) * 0.30, f"视频平滑 · {message}")

                        if full_action is not None:
                            def action_s2_work() -> dict:
                                self._raise_if_cancelled(job_id)
                                update(1.0, "S2 可选 Action 正在与视频平滑并行生成")
                                action_request = ActionMappingRequest(
                                    episode_ids=[episode_id],
                                    profile_id=str(full_action["profile_id"]),
                                    source_hand=str(full_action["source_hand"]),
                                    coordinate_frame=str(full_action["coordinate_frame"]),
                                    horizon_frames=int(full_action["horizon_frames"]),
                                    force=False,
                                )
                                action_report = generate_episode_action(
                                    dataset_id,
                                    manifest,
                                    episode,
                                    action_request,
                                    reference_media_file_id=str(selected_media.get("file_id") or "") or None,
                                )
                                self._raise_if_cancelled(job_id)
                                validation = validate_episode_action_mapping(
                                    dataset_id,
                                    manifest,
                                    episode,
                                    int(selected_media.get("frame_count") or episode.get("frame_count") or 0),
                                )
                                if validation is None:
                                    raise RuntimeError("S2 未生成可验证的 Action 结果")
                                update(29.0, f"S2 Action 校验完成：{validation.get('verdict') or 'unknown'}")
                                return {"report": action_report, "validation": validation}

                            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="alice-full-early") as early_executor:
                                smoothing_future = early_executor.submit(
                                    smooth_video,
                                    dataset_id,
                                    episode,
                                    selected_media,
                                    smoothing_update,
                                    run_id=run_id,
                                    mode=request.smoothing_mode,
                                    target_fps=request.smoothing_target_fps,
                                    motion_compensation=request.smoothing_motion_compensation,
                                )
                                action_future = early_executor.submit(action_s2_work)
                                smoothing_payload = smoothing_future.result()
                                try:
                                    action_stage_payload = action_future.result()
                                except JobCancelled:
                                    raise
                                except Exception as exc:
                                    action_stage_error = str(exc)
                                    update(29.0, f"S2 可选 Action 生成失败，继续质量清洗：{action_stage_error}")
                        else:
                            smoothing_payload = smooth_video(
                                dataset_id,
                                episode,
                                selected_media,
                                smoothing_update,
                                run_id=run_id,
                                mode=request.smoothing_mode,
                                target_fps=request.smoothing_target_fps,
                                motion_compensation=request.smoothing_motion_compensation,
                            )
                        smoothing_summary = smoothing_payload.get("summary") or {}
                        selected_frame_count = int(selected_media.get("frame_count") or 0)
                        smoothed_frame_count = int(smoothing_summary.get("frame_count") or 0)
                        source_frame_positions = list(smoothing_payload.get("source_frame_positions") or [])
                        retimed_output = bool(
                            smoothed_frame_count > 0
                            and len(source_frame_positions) == smoothed_frame_count
                        )
                        analysis_frame_count = smoothed_frame_count if retimed_output else (
                            min(selected_frame_count, smoothed_frame_count)
                            if selected_frame_count and smoothed_frame_count
                            else selected_frame_count or smoothed_frame_count
                        )
                        frame_audit = smoothing_payload.get("frame_audit") or {}
                        geometry_contract = smoothing_payload.get("geometry_contract") or {}
                        analysis_media = {
                            **selected_media,
                            "path": str(smoothing_payload["output_video"]),
                            # The smoothed file may retain physical MP4 tail
                            # frames that the dataset timestamp index excluded.
                            # All downstream stages must remain in the selected
                            # media's logical frame space.
                            "frame_count": analysis_frame_count,
                            "fps": float(smoothing_summary.get("fps") or selected_media.get("fps") or 30.0),
                            "width": int(smoothing_summary.get("width") or selected_media.get("width") or 0),
                            "height": int(smoothing_summary.get("height") or selected_media.get("height") or 0),
                            "duration": analysis_frame_count / max(0.01, float(smoothing_summary.get("fps") or selected_media.get("fps") or 30.0)),
                            "source_frame_count": int(smoothing_summary.get("source_frame_count") or selected_frame_count),
                            "source_fps": float(smoothing_summary.get("source_fps") or selected_media.get("fps") or 0.0),
                            "source_frame_positions": source_frame_positions if retimed_output else [],
                            "smoothing_mode": smoothing_summary.get("smoothing_mode") or request.smoothing_mode,
                            "smoothing_artifact_path": smoothing_payload.get("artifact_path"),
                            "smoothing_frame_audit_path": frame_audit.get("artifact_path"),
                            "pixel_transform_available": bool(geometry_contract.get("pixel_transform_available")),
                        }
                        if action_stage_payload is not None and retimed_output:
                            validation = dict((action_stage_payload.get("validation") or {}))
                            action_invalid = validation.get("invalid_mask")
                            validation["invalid_mask"] = _retime_boolean_mask(
                                np.asarray(action_invalid if action_invalid is not None else [], dtype=bool),
                                source_frame_positions,
                            )
                            validation["retimed_to_video_frame_count"] = analysis_frame_count
                            validation["retiming_mode"] = analysis_media["smoothing_mode"]
                            action_stage_payload = {**action_stage_payload, "validation": validation}

                    timeline_lock = None
                    timeline_id = None
                    if run_id:
                        if smoothing_payload is None:
                            raise RuntimeError("Full run did not produce its run-scoped smoothing artifact")
                        timeline_lock = write_full_timeline_lock(
                            dataset_id,
                            run_id,
                            episode,
                            selected_media,
                            analysis_media,
                            smoothing=smoothing_payload,
                            projection=projection_documents.get(episode_id),
                        )
                        timeline_id = str(timeline_lock["timeline_id"])
                        timeline_ids[episode_id] = timeline_id
                        smoothing_payload = write_stamped_artifact(
                            str(smoothing_payload["artifact_path"]),
                            smoothing_payload,
                            run_id,
                            timeline_id,
                            "smoothing",
                        )
                        analysis_media = {**analysis_media, "full_run_id": run_id, "timeline_id": timeline_id}
                        run_stage_artifacts = {
                            "smoothing": artifact_record(
                                dataset_id,
                                run_id,
                                smoothing_payload["artifact_path"],
                                stage="smoothing",
                                video=artifact_record(dataset_id, run_id, smoothing_payload["output_video"]),
                            ),
                        }
                        projection_document = projection_documents.get(episode_id)
                        if projection_document is not None and projection_document.get("activation_scope") == "full_run":
                            projection_document = write_stamped_artifact(
                                str(projection_document["artifact_path"]),
                                projection_document,
                                run_id,
                                timeline_id,
                                "projection",
                            )
                            projection_documents[episode_id] = projection_document
                            projection_record = artifact_record(
                                dataset_id,
                                run_id,
                                projection_document["artifact_path"],
                                stage="projection",
                                hdf5=artifact_record(dataset_id, run_id, projection_document["corrected_hdf5"]),
                            )
                            if projection_document.get("retimed_video"):
                                projection_record["video"] = artifact_record(dataset_id, run_id, projection_document["retimed_video"])
                            run_stage_artifacts["projection"] = projection_record
                            override = (manifest.get(FULL_PROJECTION_SOURCE_OVERRIDES) or {}).get(episode_id)
                            if isinstance(override, dict):
                                override["metadata"] = projection_document
                        update_full_run_episode(
                            dataset_id,
                            run_id,
                            episode_id,
                            status="running",
                            timeline=timeline_lock,
                            artifacts=run_stage_artifacts,
                        )

                    precheck_base = 30.0 if request.full_pipeline else 0.0
                    precheck_span = 30.0 if request.full_pipeline else 70.0
                    def preliminary_update(value: float, message: str) -> None:
                        self._raise_if_cancelled(job_id)
                        update(precheck_base + max(0.0, min(100.0, value)) * precheck_span / 100.0, message)

                    preliminary = run_episode_curation(
                        dataset_id,
                        manifest,
                        episode,
                        analysis_media,
                        request,
                        preliminary_update,
                        s3_reference=s3_references.get(episode_id),
                        behavior_checks=False,
                        generated_s2_override=(action_stage_payload or {}).get("validation"),
                        run_id=run_id,
                        timeline_id=timeline_id,
                    )
                    valid_ranges = curation_vlm_ranges(preliminary)
                    behavior = load_behavior_annotation(dataset_id, episode_id) if valid_ranges else None
                    reusable_behavior = bool(valid_ranges) and not request.force_vlm and behavior_matches_curation_ranges(
                        behavior, valid_ranges, analysis_media,
                    )
                    vlm_status = "skipped"
                    if valid_ranges and not reusable_behavior:
                        if not registry.has_vlm:
                            raise RuntimeError("S1-S5/C3 初筛已完成；继续有效片段标注与 C2 需要先配置 Qwen-VLM API")

                        def behavior_update(value: float, message: str) -> None:
                            self._raise_if_cancelled(job_id)
                            start, span_value = (60.0, 18.0) if request.full_pipeline else (70.0, 20.0)
                            update(start + max(0.0, min(100.0, value)) * span_value / 100.0, f"VLM 非红片段标注 · {message}")

                        vlm_requested_count += 1
                        behavior = annotate_episode_behavior(
                            dataset_id,
                            manifest,
                            episode,
                            BehaviorAnnotationRequest(
                                sample_count=request.vlm_sample_count,
                                media_file_id=str(selected_media.get("file_id") or "") or None,
                                force=True,
                            ),
                            behavior_update,
                            analysis_media_override=analysis_media,
                            analysis_source_kind="curation_non_rejected_segments",
                            analysis_frame_ranges=valid_ranges,
                            source_media_file_id=str(selected_media.get("file_id") or "") or None,
                            run_id=run_id,
                            timeline_id=timeline_id,
                            sampling_evidence=preliminary.get("samples"),
                        )
                        vlm_status = "completed"
                    elif reusable_behavior:
                        if run_id and behavior is not None and timeline_id is not None:
                            behavior = snapshot_behavior_annotation_for_run(
                                dataset_id, episode, behavior, run_id, timeline_id,
                            )
                        vlm_reused_count += 1
                        vlm_status = "reused"
                        update(78.0 if request.full_pipeline else 90.0, "VLM 已复用匹配当前非红片段的标注，未请求 Qwen")
                    else:
                        vlm_skipped_count += 1
                        update(78.0 if request.full_pipeline else 90.0, "VLM 已跳过：S1-S5/C3 后没有非红片段")
                    if run_id and behavior is not None:
                        behavior_path = str(((behavior.get("artifacts") or {}).get("behavior") or ""))
                        if not behavior_path:
                            raise RuntimeError("Full run VLM artifact was not written into its run directory")
                        update_full_run_episode(
                            dataset_id,
                            run_id,
                            episode_id,
                            artifacts={
                                "behavior": artifact_record(dataset_id, run_id, behavior_path, stage="vlm", reused=vlm_status == "reused"),
                            },
                        )
                    def finalize_update(value: float, message: str) -> None:
                        self._raise_if_cancelled(job_id)
                        start, span_value = (78.0, 10.0) if request.full_pipeline else (90.0, 10.0)
                        update(start + max(0.0, min(100.0, value)) * span_value / 100.0, message)

                    payload = finalize_episode_curation(
                        dataset_id,
                        manifest,
                        episode,
                        behavior,
                        finalize_update,
                        vlm_status=vlm_status,
                        media_file_id=str(analysis_media.get("file_id") or "") or None,
                        run_id=run_id,
                        timeline_id=timeline_id,
                        preliminary_report=preliminary,
                    )
                    if run_id:
                        update_full_run_episode(
                            dataset_id,
                            run_id,
                            episode_id,
                            artifacts={
                                "curation": artifact_record(dataset_id, run_id, payload["artifact_path"], stage="curation_post_vlm"),
                            },
                        )
                    export_result = None
                    export_error = None
                    if request.full_pipeline:
                        def export_update(value: float, message: str) -> None:
                            self._raise_if_cancelled(job_id)
                            update(88 + max(0.0, min(100.0, value)) * 0.12, message)

                        nexus_episode_package = (
                            dataset_mode(manifest)["family"] == "nexus_multimodal"
                            and request.full_output_format == EPISODE_LEROBOT_JSON_OUTPUT_FORMAT
                            and bool(((manifest.get("format_map") or {}).get("capabilities") or {}).get("can_nexus_mano21_adapter"))
                        )
                        export_supported = (
                            request.full_output_format == SUBTASK_JSON_OUTPUT_FORMAT
                            or nexus_episode_package
                            or ((manifest.get("format_map") or {}).get("capabilities") or {}).get("can_full_export") is not False
                        )
                        if output_setup_error:
                            export_error = f"无法创建 Full 输出目录，质量报告已保留：{output_setup_error}"
                        elif not export_supported:
                            export_error = (
                                "当前格式已完成 Full 的平滑、质量清洗与 VLM 标注，但不能安全导出固定 "
                                "MANO/LeRobot（format_map.capabilities.can_full_export=false）"
                            )
                        elif (
                            full_action is not None
                            and action_stage_payload is None
                            and request.full_output_format != SUBTASK_JSON_OUTPUT_FORMAT
                        ):
                            export_error = f"已请求机器人 Action，但 Action/S2 生成失败，已阻止输出不完整训练数据：{action_stage_error or '未知错误'}"
                        else:
                            try:
                                export_result = export_episode(
                                    output_root,
                                    manifest,
                                    episode,
                                    analysis_media,
                                    payload,
                                    behavior,
                                    export_update,
                                    output_format=request.full_output_format,
                                    run_id=run_id,
                                    timeline_id=timeline_id,
                                    action_report=(action_stage_payload or {}).get("report"),
                                ) if (
                                    behavior
                                    or request.full_output_format in {SUBTASK_JSON_OUTPUT_FORMAT, EPISODE_LEROBOT_JSON_OUTPUT_FORMAT}
                                ) else {
                                    "pairs": [],
                                    "filtering": {"retained_frame_count": 0, "removed_vlm_frame_count": 0},
                                    "transform_source": None,
                                    "category": None,
                                    "categories": [],
                                    "output_format": request.full_output_format,
                                    "full_run_id": run_id,
                                    "timeline_id": timeline_id,
                                }
                                all_pairs.extend(export_result["pairs"])
                            except JobCancelled:
                                raise
                            except Exception as exc:
                                export_error = str(exc)
                    item = {
                        "episode_id": episode_id,
                        "episode_name": episode.get("name"),
                        "status": "completed",
                        "media_file_id": str((payload.get("source_video") or {}).get("file_id") or selected_media.get("file_id") or "") or None,
                        "source_video": payload.get("source_video"),
                        "artifact_path": payload.get("artifact_path"),
                        "summary": payload.get("summary"),
                        "stages": payload.get("stages"),
                        "vlm_status": vlm_status,
                        "vlm_requested": vlm_status == "completed",
                        "vlm_reused": reusable_behavior,
                        "vlm_valid_ranges": [
                            {"start_frame": start, "end_frame": end}
                            for start, end in valid_ranges
                        ],
                    }
                    if smoothing_payload is not None:
                        item["smoothing"] = {"output_video": smoothing_payload.get("output_video"), "summary": smoothing_payload.get("summary")}
                    if projection_documents.get(episode_id) is not None:
                        projection_document = projection_documents[episode_id] or {}
                        retiming = projection_document.get("retiming") or {}
                        projection_summary = projection_document.get("summary") or {}
                        item["projection_correction"] = {
                            "status": "completed",
                            "backend": ((projection_document.get("model") or {}).get("backend") or "unknown"),
                            "adjustment_rate": float(projection_summary.get("adjustment_rate") or 0.0),
                            "adjustment_mode": projection_summary.get("adjustment_mode"),
                            "activation_scope": projection_document.get("activation_scope"),
                            "artifact_path": projection_document.get("artifact_path"),
                            "corrected_hdf5": projection_document.get("corrected_hdf5"),
                        }
                        item["projection_retiming"] = {
                            "inserted_frame_count": int(retiming.get("inserted_frame_count") or 0),
                            "source_frame_count": int(retiming.get("source_frame_count") or 0),
                            "output_frame_count": int(retiming.get("output_frame_count") or 0),
                            "remaining_projection_introduced_s1_frame_count": int(retiming.get("remaining_projection_introduced_s1_frame_count") or 0),
                        }
                    if action_stage_payload is not None:
                        action_report = action_stage_payload.get("report") or {}
                        action_validation = action_stage_payload.get("validation") or {}
                        item["action_s2"] = {
                            "status": "completed",
                            "profile": action_report.get("profile"),
                            "config": action_report.get("config"),
                            "summary": action_report.get("summary"),
                            "artifact_path": action_report.get("artifact_path"),
                            "reused": bool(action_report.get("reused")),
                            "validation": {key: value for key, value in action_validation.items() if key != "invalid_mask"},
                        }
                    elif full_action is not None:
                        item["action_s2"] = {
                            "status": "failed",
                            "config": full_action,
                            "error": action_stage_error or "未生成 Action",
                        }
                    if export_result is not None:
                        item["export"] = {"status": "completed", **export_result}
                        item["pair_count"] = len(export_result["pairs"])
                    elif request.full_pipeline:
                        item["export"] = {
                            "status": "failed",
                            "output_format": request.full_output_format,
                            "pairs": [],
                            "error": export_error or "未生成导出结果",
                        }
                        item["pair_count"] = 0
                    partial_error = export_error or action_stage_error
                    item["full_status"] = "partial" if request.full_pipeline and partial_error else "completed"
                    item["run_id"] = run_id
                    item["timeline_id"] = timeline_id
                    results.append(item)
                    if run_id:
                        run_artifacts: dict[str, dict] = {}
                        if export_result is not None:
                            run_artifacts["export"] = {
                                "stage": "export",
                                "output_root": str(output_root),
                                **export_result,
                            }
                        if action_stage_payload is not None:
                            action_path = str(((action_stage_payload.get("report") or {}).get("artifact_path") or ""))
                            run_artifacts["action_s2"] = {
                                "stage": "action_s2",
                                "artifact_path": action_path or None,
                                "validation": {key: value for key, value in (action_stage_payload.get("validation") or {}).items() if key != "invalid_mask"},
                            }
                        update_full_run_episode(
                            dataset_id,
                            run_id,
                            episode_id,
                            status=item["full_status"],
                            artifacts=run_artifacts,
                            summary={
                                "curation": payload.get("summary") or {},
                                "vlm_status": vlm_status,
                                "pair_count": int(item.get("pair_count") or 0),
                                "full_status": item["full_status"],
                            },
                            error=partial_error,
                        )
                        publish_full_run_episode(dataset_id, run_id, episode_id, item["media_file_id"])
                    if partial_error:
                        failures.append({
                            "episode_id": episode_id,
                            "episode_name": episode.get("name"),
                            "stage": "export" if export_error else "action_s2",
                            "error": partial_error,
                            "curation_artifact_path": payload.get("artifact_path"),
                            "media_file_id": item["media_file_id"],
                        })
                    self._update(job_id, stages=payload.get("stages"))
                except JobCancelled:
                    raise
                except Exception as exc:
                    failures.append({"episode_id": episode_id, "episode_name": episode.get("name"), "error": str(exc)})
                    if run_id:
                        update_full_run_episode(dataset_id, run_id, episode_id, status="failed", error=str(exc))
                self._raise_if_cancelled(job_id)
                self._update(
                    job_id,
                    completed_count=position + 1,
                    progress=round(pipeline_offset + (position + 1) / max(1, total) * (100.0 - pipeline_offset), 1),
                )
            self._raise_if_cancelled(job_id)
            index_path = None
            dataset_index_error = None
            if output_root is not None and output_setup_error is None:
                try:
                    index_path = write_dataset_index(
                        output_root,
                        manifest,
                        all_pairs,
                        failures,
                        output_format=request.full_output_format,
                        run_id=run_id,
                        timeline_ids=timeline_ids,
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    dataset_index_error = str(exc)
                    failures.append({
                        "episode_id": None,
                        "episode_name": None,
                        "stage": "dataset_index",
                        "error": dataset_index_error,
                    })
            result = {
                "dataset_id": dataset_id,
                "run_id": run_id,
                "operation": operation,
                "episode_count": total,
                "completed_count": len(results),
                "failure_count": len(failures),
                "items": results,
                "failures": failures,
                "pair_count": len(all_pairs),
                "vlm_requested_count": vlm_requested_count,
                "vlm_reused_count": vlm_reused_count,
                "vlm_skipped_count": vlm_skipped_count,
                "action_config": full_action,
                "projection_config": projection_config if request.full_pipeline else None,
                "output_format": request.full_output_format if request.full_pipeline else None,
                "output_root": str(output_root) if output_root is not None else None,
                "dataset_index": str(index_path) if index_path is not None else None,
                "dataset_index_error": dataset_index_error or output_setup_error,
            }
            if run_id:
                run_status = "failed" if failures and not results else "partial" if failures else "completed"
                finalize_full_run(dataset_id, run_id, run_status, {
                    "episode_count": total,
                    "completed_count": len(results),
                    "failure_count": len(failures),
                    "pair_count": len(all_pairs),
                    "output_root": str(output_root) if output_root is not None else None,
                    "dataset_index": str(index_path) if index_path is not None else None,
                    "timeline_ids": timeline_ids,
                })
            if failures and not results:
                self._update(job_id, status="failed", progress=100, current_episode_id=None, current_stage="failed", message=f"全部 {total} 个 Episode 清洗失败", result=result, error=failures[0]["error"])
            else:
                message = f"{'Full 数据集生成' if request.full_pipeline else '数据质量清洗'}完成 · {len(results)}/{total}"
                message += f" · VLM 请求 {vlm_requested_count} · 复用 {vlm_reused_count} · 跳过 {vlm_skipped_count}"
                if failures:
                    message += f" · {len(failures)} 个失败"
                self._update(job_id, status="complete", progress=100, current_episode_id=None, current_stage="complete", message=message, result=result)
        except JobCancelled:
            if run_id:
                try:
                    finalize_full_run(dataset_id, run_id, "cancelled", {
                        "episode_count": total,
                        "completed_count": len(results),
                        "failure_count": len(failures),
                        "pair_count": len(all_pairs),
                        "timeline_ids": timeline_ids,
                    })
                except RuntimeError:
                    passr
            self._mark_cancelled(job_id)
        except Exception as exc:
            if run_id:
                try:
                    finalize_full_run(dataset_id, run_id, "failed", {"error": str(exc), "timeline_ids": timeline_ids})
                except RuntimeError:
                    pass
            self._update(job_id, status="failed", progress=100, message=str(exc), error=str(exc), trace=traceback.format_exc(limit=8))
        finally:
            with self._lock:
                for episode_id in episode_ids:
                    if self._reservations.get((dataset_id, episode_id)) == job_id:
                        self._reservations.pop((dataset_id, episode_id), None)
            self._forget_cancellation(job_id)


curation_jobs = CurationJobManager()
