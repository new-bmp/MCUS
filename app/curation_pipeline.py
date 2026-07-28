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
from .behavior_annotator import annotate_episode_behavior, load_behavior_annotation, media_fingerprint_matches
from .full_export import _find_transform_source, export_episode, write_dataset_index
from .hand_visibility import inspect_full_hand_visibility
from .job_control import CancellableJobMixin, JobCancelled
from .models import registry
from .schema_profiler import infer_local_signal_fields, probe_local_signal_fields
from .schemas import ActionMappingRequest, BehaviorAnnotationRequest, CurationJobRequest
from .s1_repair import S1_REPAIR_SCHEMA, load_s1_repair
from .sensor_alignment import scan_episode_sensor_alignment
from .storage import dataset_artifact_dir, episode_media, get_manifest, record_change, slugify, storage_slug
from .video_smoothing import smooth_video


CURATION_SCHEMA = "alice/paper-curation/v1"
CURATION_PIPELINE_VERSION = 9
QUALITY_MARK_GAP_SECONDS = 0.3
ROT6D_ABSOLUTE_JUMP_DEGREES = 45.0
MAX_SIGNAL_ROWS = 120_000
MAX_SIGNAL_DIMS = 80
MAX_REPORT_FINDINGS = 2_000
NUMERIC_EXTENSIONS = {".h5", ".hdf5", ".h5df", ".parquet", ".npy", ".npz", ".json", ".jsonl", ".csv", ".tsv"}

STAGE_DEFINITIONS = [
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


def curation_report_path(dataset_id: str, episode_id: str) -> Path:
    return dataset_artifact_dir(dataset_id, "curation") / f"{storage_slug(episode_id)}.curation.alice"


def s1_repair_path(dataset_id: str, episode_id: str) -> Path:
    return dataset_artifact_dir(dataset_id, "curation-repairs") / f"{storage_slug(episode_id)}.s1-repair.alice"


def load_curation_report(dataset_id: str, episode_id: str) -> dict | None:
    preferred = curation_report_path(dataset_id, episode_id)
    legacy = preferred.with_name(f"{slugify(episode_id)}.curation.alice")
    path = next((candidate for candidate in dict.fromkeys((preferred, legacy)) if candidate.is_file()), None)
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema") != CURATION_SCHEMA:
        return None
    if str(payload.get("dataset_id") or "") != str(dataset_id) or str(payload.get("episode_id") or "") != str(episode_id):
        raise ValueError("Paper curation report identity does not match the requested Dataset/Episode")
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
                "members": [str(item) for item in field.get("members", [])],
                "extraction": str(field.get("extraction") or ""),
                "source": "local_schema",
            })
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
    deduplicated: dict[tuple[str, str, str], dict] = {}
    for candidate in candidates:
        key = (candidate["kind"], candidate["relative_path"].casefold(), candidate["field"].casefold())
        if key not in deduplicated or candidate["confidence"] > deduplicated[key]["confidence"]:
            deduplicated[key] = candidate
    return sorted(deduplicated.values(), key=lambda item: (-item["confidence"], item["relative_path"], item["field"]))


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
            return {"values": np.asarray(values, dtype=np.float64), "row_indices": rows, "source_count": count}
        if field not in handle:
            raise KeyError(field)
        dataset = handle[field]
        if not dataset.shape:
            raise ValueError("field is scalar")
        count = int(dataset.shape[0])
        rows = np.arange(count, dtype=np.int64) if count <= MAX_SIGNAL_ROWS else np.linspace(0, count - 1, MAX_SIGNAL_ROWS, dtype=np.int64)
        values = _coerce_matrix(dataset[rows] if count > MAX_SIGNAL_ROWS else dataset[:])
        return {"values": values, "row_indices": rows, "source_count": count}


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


def _alignment_stream(alignment: dict, relative_path: str) -> dict | None:
    wanted = relative_path.replace("\\", "/").casefold()
    return next(
        (item for item in alignment.get("streams", []) if str(item.get("relative_path") or "").replace("\\", "/").casefold() == wanted),
        None,
    )


def _resample_to_video(series: dict, frame_count: int, alignment_stream: dict | None) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(series["values"], dtype=np.float64)
    row_indices = np.asarray(series["row_indices"], dtype=np.int64)
    source_count = int(series.get("source_count") or values.shape[0])
    targets = np.full(frame_count, -1, dtype=np.int64)
    lookup = (alignment_stream or {}).get("frame_to_sensor_index")
    if isinstance(lookup, list):
        count = min(frame_count, len(lookup))
        targets[:count] = np.asarray(lookup[:count], dtype=np.int64)
    elif (alignment_stream or {}).get("mode") in {"prealigned_master_clock", "paired_frame_index"}:
        targets = np.arange(frame_count, dtype=np.int64)
    elif alignment_stream and alignment_stream.get("index_multiplier") is not None:
        targets = np.rint(np.arange(frame_count) * float(alignment_stream["index_multiplier"])).astype(np.int64)
    else:
        targets = np.rint(np.linspace(0, max(0, source_count - 1), frame_count)).astype(np.int64)
    valid = (targets >= 0) & (targets < source_count)
    positions = np.searchsorted(row_indices, np.clip(targets, 0, max(0, source_count - 1)), side="left")
    positions = np.clip(positions, 0, max(0, row_indices.size - 1))
    left = np.clip(positions - 1, 0, max(0, row_indices.size - 1))
    use_left = np.abs(row_indices[left] - targets) <= np.abs(row_indices[positions] - targets)
    positions = np.where(use_left, left, positions)
    output = np.full((frame_count, values.shape[1]), np.nan, dtype=np.float64)
    output[valid] = values[positions[valid]]
    source_rows = np.full(frame_count, -1, dtype=np.int64)
    source_rows[valid] = row_indices[positions[valid]]
    return output, source_rows


def _load_signal_bundle(manifest: dict, episode: dict, alignment: dict, frame_count: int | None = None) -> dict:
    root = Path(manifest["root_path"]).expanduser().resolve()
    target_frame_count = int(frame_count if frame_count is not None else episode.get("frame_count") or 0)
    matrices: dict[str, list[np.ndarray]] = {"joint": [], "action": []}
    bindings: list[dict] = []
    warnings: list[str] = []
    gripper_columns: dict[str, set[int]] = {"joint": set(), "action": set()}
    offsets = {"joint": 0, "action": 0}
    action_representations: set[str] = set()
    semantic_dimensions_known = True
    for candidate in _signal_candidates(manifest, episode):
        if len(matrices[candidate["kind"]]) >= 6:
            continue
        path = (root / candidate["relative_path"]).resolve()
        try:
            path.relative_to(root)
            series = _read_numeric_series(path, candidate["field"], candidate)
            aligned, source_row_indices = _resample_to_video(
                series,
                target_frame_count,
                _alignment_stream(alignment, candidate["relative_path"]),
            )
        except Exception as exc:
            warnings.append(f"{candidate['relative_path']} / {candidate['field']}: {str(exc)[:180]}")
            continue
        if aligned.shape[1] + offsets[candidate["kind"]] > MAX_SIGNAL_DIMS:
            aligned = aligned[:, : max(0, MAX_SIGNAL_DIMS - offsets[candidate["kind"]])]
        if not aligned.shape[1]:
            continue
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
        bindings.append({
            **candidate,
            "dimensions": int(aligned.shape[1]),
            "column_start": start,
            "column_end": end,
            "source_rows": int(series["source_count"]),
            "_source_row_indices": source_row_indices,
        })
    return {
        "joint": np.concatenate(matrices["joint"], axis=1) if matrices["joint"] else None,
        "action": np.concatenate(matrices["action"], axis=1) if matrices["action"] else None,
        "bindings": bindings,
        "warnings": warnings,
        "gripper_columns": gripper_columns,
        "action_representation": next(iter(action_representations)) if len(action_representations) == 1 else "unknown",
        "semantic_dimensions_known": semantic_dimensions_known,
        "embodiment_ids": sorted({str(item.get("embodiment_id")) for item in bindings if item.get("embodiment_id")}),
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


def repair_s1_bundle(bundle: dict, sigma: float, max_gap_frames: int) -> dict:
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


def estimate_state_action_alignment(
    state: np.ndarray,
    action: np.ndarray,
    fps: float,
    max_lag_seconds: float,
    da_threshold: float,
    action_representation: str,
) -> dict:
    dimensions = min(state.shape[1], action.shape[1])
    if dimensions <= 0 or state.shape[0] < 12:
        raise ValueError("state/action shared dimensions are unavailable")
    state_values = _smooth(state[:, :dimensions])
    action_values = _smooth(action[:, :dimensions])
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
        s = (s - np.mean(s)) / max(np.std(s), 1e-9)
        a = (a - np.mean(a)) / max(np.std(a), 1e-9)
        best_lag = 0
        best_correlation = -math.inf
        for lag in range(-max_lag, max_lag + 1):
            aligned_state, aligned_action = _aligned_pair(s, a, lag)
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
        metrics.append({"dimension": dimension, "lag_frames": best_lag, "correlation": round(best_correlation, 6), "directional_agreement": round(agreement, 6)})
    if not metrics:
        raise ValueError("state/action trajectories have insufficient variation")
    lag_frames = int(round(float(np.median([item["lag_frames"] for item in metrics]))))
    agreement = float(np.median([item["directional_agreement"] for item in metrics]))
    review_floor = min(da_threshold, 0.6)
    verdict = "pass" if agreement >= da_threshold else "review" if agreement >= review_floor else "reject_candidate"
    return {
        "lag_frames": lag_frames,
        "lag_seconds": round(lag_frames / max(fps, 0.01), 6),
        "directional_agreement": round(agreement, 6),
        "threshold": da_threshold,
        "passed": verdict == "pass",
        "verdict": verdict,
        "shared_dimensions": dimensions,
        "action_representation": action_representation,
        "action_integrated": action_representation in {"delta", "velocity"},
        "dimensions": metrics[:32],
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
    step = max(1, int(round(fps / request.video_sample_fps)))
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
    minimum_static_samples = max(2, int(math.ceil(request.static_duration_seconds * request.video_sample_fps)))
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
    evaluated = {"completed", "warning"}
    both_checks_ran = c1.get("status") in evaluated and c2.get("status") in evaluated
    review = np.zeros_like(precheck_review) if both_checks_ran else precheck_review.copy()
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


def inspect_behavior_state_consistency(
    behavior: dict | None,
    bundle: dict,
    preliminary_valid: np.ndarray,
    fps: float,
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
    if motion is None:
        return {
            "status": "skipped",
            "message": "C2 缺少可对齐的 State/Action 数值流",
            "review_mask": np.zeros(frame_count, dtype=bool),
            "findings": [],
            "metrics": {"behavior_segment_count": len(behavior.get("segments") or [])},
        }
    candidates = motion[preliminary_valid & np.isfinite(motion)]
    positive = candidates[candidates > 1e-9]
    motion_threshold = max(1e-4, float(np.percentile(positive, 25)) * 0.5) if positive.size else 1e-4
    inactive_phases = {"idle", "observe", "reach", "withdraw", "unknown", "precheck_invalid"}
    minimum_frames = max(2, int(math.ceil(0.4 * fps)))
    review_mask = np.zeros(frame_count, dtype=bool)
    findings: list[dict] = []
    checked = 0
    ratios: list[float] = []
    for segment in behavior.get("segments") or []:
        if frame_count <= 0:
            break
        phase = str(segment.get("phase_label") or segment.get("label") or "unknown").strip().casefold().replace("-", "_").replace(" ", "_")
        if phase in inactive_phases:
            continue
        start = max(0, min(frame_count - 1, int(segment.get("start_frame") or 0)))
        end = max(start, min(frame_count - 1, int(segment.get("end_frame") or start)))
        mask = preliminary_valid[start:end + 1]
        if int(mask.sum()) < minimum_frames:
            continue
        checked += 1
        active_ratio = float(np.mean(motion[start:end + 1][mask] > motion_threshold))
        ratios.append(active_ratio)
        if active_ratio >= 0.08:
            continue
        segment_mask = np.zeros(frame_count, dtype=bool)
        segment_mask[start:end + 1] = preliminary_valid[start:end + 1]
        review_mask |= segment_mask
        findings.extend(_mask_findings(
            segment_mask,
            "c2",
            "review",
            f"C2 VLM 阶段 {phase} 缺少同步 State/Action 运动证据",
            fps,
            0.72,
        ))
    metrics = {
        "behavior_segment_count": len(behavior.get("segments") or []),
        "checked_active_segment_count": checked,
        "mismatch_segment_count": len(findings),
        "mismatch_frame_count": int(review_mask.sum()),
        "motion_threshold": round(motion_threshold, 8),
        "mean_active_motion_ratio": round(float(np.mean(ratios)) if ratios else 0.0, 6),
        "signal_sources": [kind for kind in ("joint", "action") if bundle.get(kind) is not None],
    }
    return {
        "status": "warning" if findings else "completed",
        "message": f"发现 {len(findings)} 个视频-State 不一致片段" if findings else f"已复核 {checked} 个有效操作片段",
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
    generated_action = load_episode_action_mapping(dataset_id, episode_id, manifest=manifest)
    generated_action_ready = bool(generated_action and generated_action.get("artifact_path"))
    s2_ready = generated_action_ready or ({"joint", "action"} <= kinds and action_semantics_known)
    try:
        _find_transform_source(manifest, episode)
        action_source_ready = True
    except (KeyError, OSError, RuntimeError, ValueError):
        action_source_ready = False
    behavior = load_behavior_annotation(dataset_id, episode_id)
    media = episode_media(episode, media_file_id or episode.get("primary_media_file_id"))
    stages = [
        _stage("s1", "ready" if kinds else "skipped", "将运行通用 Jerk 与 endpose rot6d 相对旋转突变检查" if kinds else "没有已识别的 Joint/Action 数值流"),
        _stage(
            "s2",
            "ready" if s2_ready else "pending" if action_source_ready else "skipped",
            "将校验已生成 Action 的源轨迹、预测帧和数值一致性" if generated_action_ready else "可运行" if s2_ready else "需要先生成 Action，或提供语义明确的 State/Joint 与 Action",
        ),
        _stage("s3", "ready" if kinds else "skipped", "可运行" if kinds else "没有已识别的数值流"),
        _stage("s4", "skipped", "当前版本未实现 FK 一致性计算；即使检测到 URDF/Pinocchio 也不会宣称可运行"),
        _stage("s5", "skipped", "当前版本仅记录坐标修正建议，不自动改写坐标；标定信息不会触发执行"),
        _stage("c3", "ready", f"将检查 {media.get('stream_name') or media.get('relative_path') or '所选视频'} 的画质与整手可见性"),
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
) -> dict:
    frame_count = int(media.get("frame_count") or episode.get("frame_count") or 0)
    fps = max(0.01, float(media.get("fps") or episode.get("fps") or 30.0))
    if frame_count < 4:
        raise ValueError("Episode 帧数不足，无法运行数据清洗")
    root = Path(manifest["root_path"]).expanduser().resolve()
    generated_action_report = load_episode_action_mapping(dataset_id, str(episode["id"]), manifest=manifest)
    candidate_paths = {
        str(candidate.get("relative_path") or "").replace("\\", "/")
        for candidate in _signal_candidates(manifest, episode)
        if str(candidate.get("relative_path") or "")
    }
    media_relative = str(media.get("relative_path") or episode.get("relative_path") or "").replace("\\", "/")
    if media_relative:
        candidate_paths.add(media_relative)
    generated_source_relative = str((generated_action_report or {}).get("source", {}).get("relative_path") or "").replace("\\", "/")
    if generated_source_relative:
        candidate_paths.add(generated_source_relative)
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
    progress(3, "建立视频与传感器时间映射")
    alignment = scan_episode_sensor_alignment(manifest, episode, force=False)
    progress(9, "读取 Joint / Action 数值流")
    bundle = _load_signal_bundle(manifest, episode, alignment, frame_count=frame_count)
    joint = bundle["joint"]
    action = bundle["action"]
    stages: list[dict] = []
    findings: list[dict] = []
    invalid = np.zeros(frame_count, dtype=bool)
    review = np.zeros(frame_count, dtype=bool)
    motion_score = np.zeros(frame_count, dtype=np.float64)
    s1_repair_patch: dict | None = None
    s1_repair_summary = {
        "enabled": bool(request.repair_s1_spikes),
        "method": "bounded_cubic_and_rot6d_slerp_v1",
        "max_repair_frames": int(request.s1_max_repair_frames),
        "repaired_frame_count": 0,
        "repaired_range_count": 0,
        "artifact_path": None,
    }

    signal_parts = [item for item in (joint, action) if item is not None]
    if signal_parts:
        progress(18, "S1 突变、Jerk 与 endpose rot6d 相对旋转检查")
        combined = np.concatenate(signal_parts, axis=1)
        s1_before = detect_sudden_changes(combined, request.sudden_change_sigma)
        rot6d_before = inspect_rot6d_jumps(bundle, request.sudden_change_sigma)
        before_mask = s1_before["mask"] | rot6d_before["mask"]
        if request.repair_s1_spikes and before_mask.any():
            repair = repair_s1_bundle(bundle, request.sudden_change_sigma, request.s1_max_repair_frames)
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
                repair_path = s1_repair_path(dataset_id, str(episode["id"]))
                s1_repair_summary["artifact_path"] = str(repair_path)
                s1_repair_patch = {
                    "schema": S1_REPAIR_SCHEMA,
                    "dataset_id": dataset_id,
                    "episode_id": str(episode["id"]),
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
        s1_mask = s1["mask"] | rot6d["mask"]
        invalid |= s1_mask
        motion_score = np.maximum(motion_score, np.maximum(s1["score"], rot6d["score"]))
        findings.extend(_mask_findings(s1["mask"], "s1", "reject", "S1 突变/加速度/Jerk 异常", fps, 0.88))
        findings.extend(_mask_findings(rot6d["mask"], "s1", "reject", "S1 endpose rot6d 相对旋转突变或 6D 基向量无效", fps, 0.94))
        stages.append(_stage("s1", "completed", f"检测到 {int(s1_mask.sum())} 个异常帧", {
            "flagged_frame_count_before_repair": int(before_mask.sum()),
            "flagged_frame_count": int(s1_mask.sum()),
            "generic_jump_frame_count": s1["event_count"],
            "rot6d_jump_frame_count": rot6d["event_count"],
            "rot6d_group_count": rot6d["group_count"],
            "rot6d_groups": rot6d["groups"],
            "sigma": request.sudden_change_sigma,
            **s1_repair_summary,
        }))
    else:
        stages.append(_stage("s1", "skipped", "没有可读取的 Joint/Action 数值流"))

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
            s2 = estimate_state_action_alignment(joint, action, fps, request.max_lag_seconds, request.directional_agreement_threshold, bundle["action_representation"])
            if s2["verdict"] == "reject_candidate":
                invalid[:] = True
                findings.extend(_mask_findings(np.ones(frame_count, dtype=bool), "s2", "reject", f"S2 State-Action 方向一致率 {s2['directional_agreement']:.3f} 低于 {request.directional_agreement_threshold:.3f}", fps, 0.94))
            elif s2["verdict"] == "review":
                review[:] = True
                findings.extend(_mask_findings(np.ones(frame_count, dtype=bool), "s2", "review", f"S2 State-Action 方向一致率 {s2['directional_agreement']:.3f} 位于复核区间", fps, 0.72))
            stage_message = "趋势一致" if s2["verdict"] == "pass" else "建议人工复核" if s2["verdict"] == "review" else "建议排除整个 Episode"
            stages.append(_stage("s2", "completed" if s2["verdict"] == "pass" else "warning", stage_message, s2))
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
    required_hand_sides = list((generated_s2 or {}).get("required_sides") or ["left", "right"])
    progress(58, "C3 检查整手是否完整位于画面内")
    hand_visibility = inspect_full_hand_visibility(manifest, episode, media, required_hand_sides)
    hand_invalid = np.asarray(hand_visibility["invalid_mask"], dtype=bool)
    if hand_visibility.get("available") and hand_invalid.shape == (frame_count,):
        invalid |= hand_invalid
        findings.extend(_mask_findings(
            hand_invalid,
            "c3",
            "reject",
            f"C3 整手未完整位于画面内（检查 {'+'.join(required_hand_sides)}）",
            fps,
            0.92,
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
        progress(88, "C2 基于 VLM 与 State/Action 检查有效片段")
        c2 = inspect_behavior_state_consistency(behavior, bundle, preliminary_valid, fps)
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
    repair_path = s1_repair_path(dataset_id, str(episode["id"]))
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
    document = {
        "schema": CURATION_SCHEMA,
        "pipeline_version": CURATION_PIPELINE_VERSION,
        "pipeline_phase": "post_vlm" if behavior_checks is True and behavior else "pre_vlm" if behavior_checks is False else "legacy",
        "dataset_id": dataset_id,
        "episode_id": episode["id"],
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
        },
        "source_signatures": signatures,
        "sensor_alignment": {
            "artifact_path": alignment.get("artifact_path"),
            "stream_count": len(alignment.get("streams") or []),
        },
        "s3_reference": {
            "scope": str((s3_reference or {}).get("scope") or "episode_limited"),
            "cohort_id": (s3_reference or {}).get("cohort_id"),
            "episode_count": int((s3_reference or {}).get("episode_count") or 1),
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
    }
    path = curation_report_path(dataset_id, str(episode["id"]))
    _write_json_atomic(path, document)
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
) -> dict:
    report = load_curation_report(dataset_id, str(episode["id"]))
    if report is None:
        raise RuntimeError("缺少 S1-S5/C3 初筛报告，不能执行 C2")
    frame_count = int((report.get("source_video") or {}).get("frame_count") or episode.get("frame_count") or 0)
    fps = max(0.01, float((report.get("source_video") or {}).get("fps") or episode.get("fps") or 30.0))
    pre_vlm_segments = report.get("pre_vlm_segments") or report.get("segments") or []
    invalid, precheck_review = _state_masks_from_segments(frame_count, pre_vlm_segments)
    preliminary_eligible = ~invalid
    progress(12, "读取 S1-S5/C3 有效片段与对齐数值流")
    alignment = scan_episode_sensor_alignment(manifest, episode, force=False)
    bundle = _load_signal_bundle(manifest, episode, alignment, frame_count=frame_count)
    apply_s1_repair_to_bundle(bundle, load_s1_repair(report))
    progress(45, "C2 基于有效片段 VLM 与 State/Action 做一致性检查")
    c1 = inspect_instruction_consistency(behavior, episode, preliminary_eligible, fps)
    c2 = inspect_behavior_state_consistency(behavior, bundle, preliminary_eligible, fps)
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
        "behavior_fingerprint": behavior_fingerprint,
        "stages": stages,
        "findings": findings,
        "pre_vlm_segments": pre_vlm_segments,
        "segments": segments,
        "samples": samples,
        "summary": summary,
    }
    path = curation_report_path(dataset_id, str(episode["id"]))
    _write_json_atomic(path, document)
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
            alignment = scan_episode_sensor_alignment(manifest, episode, force=False)
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
        capped: list[np.ndarray] = []
        for entry in entries:
            matrix = entry["matrix"]
            if matrix.shape[0] > 20_000:
                indices = np.linspace(0, matrix.shape[0] - 1, 20_000, dtype=np.int64)
                matrix = matrix[indices]
            capped.append(matrix)
        reference = np.concatenate(capped, axis=0)
        cohort_id = "cohort-" + hashlib.sha1(repr(key).encode("utf-8")).hexdigest()[:10]
        cohort_signatures = {
            str(signature.get("relative_path") or ""): signature
            for entry in entries
            for signature in entry["source_signatures"]
        }
        for entry in entries:
            episode_id = entry["episode_id"]
            references[episode_id] = {
                "matrix": reference,
                "scope": "cohort",
                "cohort_id": cohort_id,
                "episode_count": len(entries),
                "source_signatures": list(cohort_signatures.values()),
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
            media_by_episode[episode_id] = episode_media(episodes[episode_id], media_id)
        with self._lock:
            overlap = [episode_id for episode_id in episode_ids if (dataset_id, episode_id) in self._reservations]
            if overlap:
                raise RuntimeError(f"{episodes[overlap[0]]['name']} 已有数据清洗任务正在运行")
            job_id = uuid.uuid4().hex
            operation = "full_pipeline" if request.full_pipeline else "paper_curation"
            job = {
                "id": job_id,
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
        try:
            manifest = get_manifest(dataset_id)
            episodes = {str(item["id"]): item for item in manifest.get("episodes", [])}
            operation = "full_pipeline" if request.full_pipeline else "paper_curation"
            output_root = Path(str(manifest["root_path"])).expanduser().resolve() / "output" if request.full_pipeline else None
            if output_root is not None:
                output_root.mkdir(parents=True, exist_ok=True)
            full_action = _full_action_config(request) if request.full_pipeline else None
            self._start_unless_cancelled(job_id, status="running", progress=1, message=f"{'Full' if request.full_pipeline else '后台清洗'}流程已启动 · 0/{total}")
            if total > 1:
                self._update(job_id, progress=2, current_stage="s3", message=f"正在建立同 embodiment 的跨 EP S3 分位参考 · {total} Episodes")
            self._raise_if_cancelled(job_id)
            s3_references = _build_s3_references(manifest, episodes, media_by_episode, episode_ids)
            for position, episode_id in enumerate(episode_ids):
                self._raise_if_cancelled(job_id)
                episode = episodes[episode_id]
                base = position / max(1, total) * 100
                span = 100 / max(1, total)

                def update(value: float, message: str) -> None:
                    self._raise_if_cancelled(job_id)
                    prefix = message.split(" ", 1)[0].casefold()
                    if prefix in {"s1", "s2", "s3", "s4", "s5", "c1", "c2", "c3"}:
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
                    selected_media = media_by_episode[episode_id]
                    analysis_media = selected_media
                    smoothing_payload = None
                    action_stage_payload = None
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
                                action_report = generate_episode_action(dataset_id, manifest, episode, action_request)
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
                                smoothing_future = early_executor.submit(smooth_video, dataset_id, episode, selected_media, smoothing_update)
                                action_future = early_executor.submit(action_s2_work)
                                smoothing_payload = smoothing_future.result()
                                action_stage_payload = action_future.result()
                        else:
                            smoothing_payload = smooth_video(dataset_id, episode, selected_media, smoothing_update)
                        smoothing_summary = smoothing_payload.get("summary") or {}
                        analysis_media = {
                            **selected_media,
                            "path": str(smoothing_payload["output_video"]),
                            "frame_count": int(smoothing_summary.get("frame_count") or selected_media.get("frame_count") or 0),
                            "fps": float(smoothing_summary.get("fps") or selected_media.get("fps") or 30.0),
                            "width": int(smoothing_summary.get("width") or selected_media.get("width") or 0),
                            "height": int(smoothing_summary.get("height") or selected_media.get("height") or 0),
                        }

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
                            BehaviorAnnotationRequest(sample_count=request.vlm_sample_count, force=True),
                            behavior_update,
                            analysis_media_override=analysis_media,
                            analysis_source_kind="curation_non_rejected_segments",
                            analysis_frame_ranges=valid_ranges,
                        )
                        vlm_status = "completed"
                    elif reusable_behavior:
                        vlm_reused_count += 1
                        vlm_status = "reused"
                        update(78.0 if request.full_pipeline else 90.0, "VLM 已复用匹配当前非红片段的标注，未请求 Qwen")
                    else:
                        vlm_skipped_count += 1
                        update(78.0 if request.full_pipeline else 90.0, "VLM 已跳过：S1-S5/C3 后没有非红片段")
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
                    )
                    export_result = None
                    if request.full_pipeline:
                        def export_update(value: float, message: str) -> None:
                            self._raise_if_cancelled(job_id)
                            update(88 + max(0.0, min(100.0, value)) * 0.12, message)

                        export_result = export_episode(
                            output_root,
                            manifest,
                            episode,
                            analysis_media,
                            payload,
                            behavior,
                            export_update,
                            output_format=request.full_output_format,
                        ) if behavior else {
                            "pairs": [],
                            "filtering": {"retained_frame_count": 0, "removed_vlm_frame_count": 0},
                            "transform_source": None,
                            "category": None,
                            "categories": [],
                            "output_format": request.full_output_format,
                        }
                        all_pairs.extend(export_result["pairs"])
                    item = {
                        "episode_id": episode_id,
                        "episode_name": episode.get("name"),
                        "status": "completed",
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
                    if action_stage_payload is not None:
                        action_report = action_stage_payload.get("report") or {}
                        action_validation = action_stage_payload.get("validation") or {}
                        item["action_s2"] = {
                            "profile": action_report.get("profile"),
                            "config": action_report.get("config"),
                            "summary": action_report.get("summary"),
                            "artifact_path": action_report.get("artifact_path"),
                            "reused": bool(action_report.get("reused")),
                            "validation": {key: value for key, value in action_validation.items() if key != "invalid_mask"},
                        }
                    if export_result is not None:
                        item["export"] = export_result
                        item["pair_count"] = len(export_result["pairs"])
                    results.append(item)
                    self._update(job_id, stages=payload.get("stages"))
                except JobCancelled:
                    raise
                except Exception as exc:
                    failures.append({"episode_id": episode_id, "episode_name": episode.get("name"), "error": str(exc)})
                self._raise_if_cancelled(job_id)
                self._update(job_id, completed_count=position + 1, progress=round((position + 1) / max(1, total) * 100, 1))
            self._raise_if_cancelled(job_id)
            index_path = write_dataset_index(
                output_root,
                manifest,
                all_pairs,
                failures,
                output_format=request.full_output_format,
            ) if output_root is not None else None
            result = {
                "dataset_id": dataset_id,
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
                "output_format": request.full_output_format if request.full_pipeline else None,
                "output_root": str(output_root) if output_root is not None else None,
                "dataset_index": str(index_path) if index_path is not None else None,
            }
            if failures and not results:
                self._update(job_id, status="failed", progress=100, current_episode_id=None, current_stage="failed", message=f"全部 {total} 个 Episode 清洗失败", result=result, error=failures[0]["error"])
            else:
                message = f"{'Full 数据集生成' if request.full_pipeline else '数据质量清洗'}完成 · {len(results)}/{total}"
                message += f" · VLM 请求 {vlm_requested_count} · 复用 {vlm_reused_count} · 跳过 {vlm_skipped_count}"
                if failures:
                    message += f" · {len(failures)} 个失败"
                self._update(job_id, status="complete", progress=100, current_episode_id=None, current_stage="complete", message=message, result=result)
        except JobCancelled:
            self._mark_cancelled(job_id)
        except Exception as exc:
            self._update(job_id, status="failed", progress=100, message=str(exc), error=str(exc), trace=traceback.format_exc(limit=8))
        finally:
            with self._lock:
                for episode_id in episode_ids:
                    if self._reservations.get((dataset_id, episode_id)) == job_id:
                        self._reservations.pop((dataset_id, episode_id), None)
            self._forget_cancellation(job_id)


curation_jobs = CurationJobManager()
