from __future__ import annotations

"""Detect and persist per-episode sensor/video timing relationships.

The generated documents are derived indices.  They live below ``.alicePD``
and never modify files in the source dataset.
"""

import csv
import json
import math
import threading
import traceback
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .storage import dataset_sidecar_root, get_manifest, slugify, storage_slug
from .job_control import CancellableJobMixin, JobCancelled


SENSOR_ALIGNMENT_SCHEMA = "alice/sensor-alignment/v4"
SENSOR_EXTENSIONS = {
    ".h5", ".hdf5", ".h5df", ".json", ".jsonl", ".parquet",
    ".npy", ".npz", ".csv", ".tsv",
}
RATE_KEYS = {
    "hz", "rate", "rate_hz", "fps", "frequency", "frequency_hz",
    "sample_rate", "sample_rate_hz", "sampling_rate", "sampling_rate_hz",
}
TIMESTAMP_NAMES = {
    "timestamp", "timestamps", "time", "times", "time_ns", "time_us",
    "time_ms", "ts", "sensor_ts", "master_ts", "wall_ts", "ts_wall",
    "pts", "pts_ns", "pts_us", "pts_ms",
}
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_TIMESTAMP_VALUES = 250_000
MAX_TABLE_COLUMNS = 512
SYNC_CAMERA_FIELDS = {
    "head": ("head_frame_idx", "head_frame_ts", "head_filled"),
    "wrist_left": ("wrist_left_frame_idx", "wrist_left_ts", "wrist_left_filled"),
    "wrist_right": ("wrist_right_frame_idx", "wrist_right_ts", "wrist_right_filled"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normal_key(value: Any) -> str:
    return str(value).strip().casefold().replace("-", "_").replace(" ", "_")


def _as_positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist() if value.size <= 64 else None
    return value


def _sidecar_root(manifest: dict) -> Path:
    configured = manifest.get("sidecar_path")
    if configured:
        return Path(str(configured)).expanduser().resolve()
    return dataset_sidecar_root(manifest["root_path"], manifest["id"])


def _alignment_reference_key(manifest: dict, episode_id: str, reference_media_file_id: str | None) -> str | None:
    requested = str(reference_media_file_id or "").strip()
    if not requested:
        return None
    episode = next((item for item in manifest.get("episodes", []) if str(item.get("id") or "") == str(episode_id)), None)
    primary_id = str((episode or {}).get("primary_media_file_id") or "").strip()
    return None if primary_id and requested == primary_id else requested


def sensor_alignment_path(
    manifest: dict,
    episode_id: str,
    reference_media_file_id: str | None = None,
) -> Path:
    reference_key = _alignment_reference_key(manifest, episode_id, reference_media_file_id)
    filename = f"{storage_slug(episode_id)}.alignment.alice"
    if reference_key:
        filename = f"{storage_slug(episode_id)}--{storage_slug(reference_key)}.alignment.alice"
    return (
        _sidecar_root(manifest)
        / "indices"
        / "sensor-alignment"
        / filename
    )


def _existing_sensor_alignment_path(
    manifest: dict,
    episode_id: str,
    reference_media_file_id: str | None = None,
) -> Path:
    reference_key = _alignment_reference_key(manifest, episode_id, reference_media_file_id)
    preferred = sensor_alignment_path(manifest, episode_id, reference_media_file_id)
    if reference_key:
        return preferred
    legacy = preferred.with_name(f"{slugify(episode_id)}.alignment.alice")
    return next((path for path in dict.fromkeys((preferred, legacy)) if path.is_file()), preferred)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_signature(path: Path) -> dict:
    stat = path.stat()
    return {"size_bytes": stat.st_size, "modified_ns": stat.st_mtime_ns}


def _episode_records(manifest: dict, episode: dict) -> list[dict]:
    episode_id = str(episode.get("id") or "")
    assignments = (manifest.get("episode_resolution") or {}).get("file_episode_assignments") or {}
    return [
        item
        for item in manifest.get("files", [])
        if str(assignments.get(str(item.get("id"))) or item.get("episode_id") or "") == episode_id
    ]


def _reference_video(episode: dict, media_file_id: str | None = None) -> dict:
    streams = list(episode.get("media_streams") or [])
    primary_id = episode.get("primary_media_file_id")
    requested_id = str(media_file_id or "").strip()
    selected = next((item for item in streams if str(item.get("file_id") or "") == requested_id), None) if requested_id else None
    if selected is None and requested_id and requested_id == str(primary_id or ""):
        selected = episode
    if selected is None and requested_id:
        raise KeyError(requested_id)
    selected = selected or next((item for item in streams if item.get("file_id") == primary_id), None)
    if selected is None:
        selected = next((item for item in streams if item.get("type") == "video"), None)
    selected = selected or episode
    fps = _as_positive_float(selected.get("fps")) or _as_positive_float(episode.get("fps")) or 30.0
    frame_count = int(selected.get("source_frame_count") or selected.get("frame_count") or episode.get("frame_count") or 0)
    duration = _as_positive_float(selected.get("duration"))
    if duration is None and frame_count:
        duration = frame_count / fps
    return {
        "file_id": selected.get("file_id") or primary_id,
        "stream_name": selected.get("stream_name") or Path(str(selected.get("relative_path") or "video")).name,
        "relative_path": selected.get("relative_path") or episode.get("relative_path"),
        "fps": round(fps, 9),
        "frame_count": frame_count,
        "duration": round(duration, 9) if duration is not None else None,
    }


def _episode_signature(root: Path, records: list[dict], reference: dict) -> dict:
    files = []
    for record in records:
        relative = str(record.get("relative_path") or "")
        if str(record.get("extension") or Path(relative).suffix).casefold() not in SENSOR_EXTENSIONS:
            continue
        path = root / relative
        if not path.is_file():
            continue
        files.append({"relative_path": record.get("relative_path"), **_source_signature(path)})
    files.sort(key=lambda item: str(item["relative_path"]).casefold())
    reference_path = root / str(reference.get("relative_path") or "")
    return {
        "reference_file_id": reference.get("file_id"),
        "reference_fps": reference.get("fps"),
        "reference_frame_count": reference.get("frame_count"),
        "reference_signature": _source_signature(reference_path) if reference_path.is_file() else None,
        "files": files,
    }


def _rate_candidate(value: Any, source: str, direct: bool = False) -> dict | None:
    hz = _as_positive_float(_json_value(value))
    if hz is None or not 0.001 <= hz <= 1_000_000:
        return None
    return {"hz": round(hz, 9), "source": source, "direct": direct}


def _dedupe_rate_candidates(candidates: Iterable[dict]) -> list[dict]:
    seen: set[tuple[float, str]] = set()
    result = []
    for candidate in candidates:
        key = (round(float(candidate["hz"]), 9), str(candidate["source"]))
        if key not in seen:
            seen.add(key)
            result.append(candidate)
    return result


def _timestamp_scale(values: np.ndarray, name: str, target_hz: float | None = None) -> tuple[float, str]:
    normalized = _normal_key(name)
    if normalized.endswith("_ns") or normalized == "ns":
        return 1e-9, "ns"
    if normalized.endswith("_us") or normalized == "us":
        return 1e-6, "us"
    if normalized.endswith("_ms") or normalized == "ms":
        return 1e-3, "ms"
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 1.0, "s"
    magnitude = float(np.nanmedian(np.abs(finite)))
    if magnitude >= 1e17:
        return 1e-9, "ns"
    if magnitude >= 1e14:
        return 1e-6, "us"
    if magnitude >= 1e11:
        return 1e-3, "ms"
    differences = np.diff(finite)
    positive = differences[differences > 0]
    if not positive.size:
        return 1.0, "s"
    raw_delta = float(np.median(positive))
    choices = [(1.0, "s"), (1e-3, "ms"), (1e-6, "us"), (1e-9, "ns")]
    plausible = [(scale, unit, 1.0 / (raw_delta * scale)) for scale, unit in choices if 0.001 <= 1.0 / (raw_delta * scale) <= 1_000_000]
    if not plausible:
        return 1.0, "s"
    target = target_hz if target_hz and target_hz > 0 else 30.0
    scale, unit, _ = min(plausible, key=lambda item: abs(math.log(max(item[2], 1e-12) / target)))
    return scale, unit


def _timestamp_analysis(values: Any, name: str, target_hz: float | None = None) -> tuple[dict | None, np.ndarray | None]:
    try:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None, None
    if array.size < 2:
        return None, None
    scale, unit = _timestamp_scale(array, name, target_hz)
    seconds = array * scale
    finite_mask = np.isfinite(seconds)
    finite = seconds[finite_mask]
    if finite.size < 2:
        return None, None
    differences = np.diff(finite)
    positive = differences[differences > 0]
    median_delta = float(np.median(positive)) if positive.size else None
    measured_hz = 1.0 / median_delta if median_delta and median_delta > 0 else None
    long_gap_threshold = median_delta * 3.0 if median_delta else None
    long_gaps = positive[positive > long_gap_threshold] if long_gap_threshold else np.asarray([], dtype=np.float64)
    summary = {
        "field": name,
        "count": int(array.size),
        "finite_count": int(finite.size),
        "unit": unit,
        "scale_to_seconds": scale,
        "measured_hz": round(measured_hz, 9) if measured_hz else None,
        "median_delta_seconds": round(median_delta, 12) if median_delta else None,
        "duplicate_count": int(np.count_nonzero(differences == 0)),
        "backward_count": int(np.count_nonzero(differences < 0)),
        "long_gap_count": int(long_gaps.size),
        "max_gap_seconds": round(float(np.max(positive)), 9) if positive.size else None,
        "start_seconds": round(float(finite[0]), 9),
        "end_seconds": round(float(finite[-1]), 9),
    }
    return summary, seconds


def _probe_numeric_fields(path: Path) -> list[dict]:
    """Return bounded field identities and row counts for one structured file.

    T0 keeps a file-level fallback stream for compatibility, but downstream
    consumers can now bind to the exact field (or HDF5 field group) they read.
    This prevents two arrays with different clocks in one container from
    silently sharing a single representative row count.
    """

    try:
        from .schema_profiler import probe_local_signal_fields

        descriptors = list(probe_local_signal_fields(path))
    except (ImportError, OSError, TypeError, ValueError):
        descriptors = []
    result = []
    seen: set[str] = set()
    for descriptor in descriptors:
        field = str(descriptor.get("field") or "").replace("\\", "/")
        shape = descriptor.get("source_shape") or descriptor.get("shape") or []
        try:
            count = int(shape[0]) if shape else 0
        except (TypeError, ValueError):
            count = 0
        if not field or count <= 1 or field.casefold() in seen:
            continue
        seen.add(field.casefold())
        result.append({
            "field": field,
            "data_count": count,
            "kind": str(descriptor.get("kind") or "sensor"),
            "role": str(descriptor.get("role") or ""),
            "modality": str(descriptor.get("modality") or ""),
            "members": [str(item).replace("\\", "/") for item in descriptor.get("members") or []],
            "evidence": str(descriptor.get("evidence") or "local_schema"),
        })
    return result


def _timestamp_priority(name: str) -> int:
    normalized = _normal_key(Path(name).name)
    if normalized in {"timestamps", "timestamp", "master_ts", "ts_wall"}:
        return 0
    if "master" in normalized or "wall" in normalized:
        return 1
    if normalized == "sensor_ts" or "sensor" in normalized:
        return 3
    return 2


def _representative_count(counts: list[int]) -> int:
    useful = [int(item) for item in counts if int(item) > 1]
    if not useful:
        return 0
    frequency = Counter(useful)
    return max(frequency, key=lambda value: (frequency[value], value))


def _probe_hdf5(path: Path, reference_fps: float) -> dict:
    import h5py

    rate_candidates: list[dict] = []
    data_counts: list[int] = []
    timestamp_entries: list[dict] = []
    partial_entries: list[dict] = []
    raw_numeric_fields: list[dict] = []
    attributes: dict[str, Any] = {}
    master_clock = None
    with h5py.File(path, "r") as handle:
        for key, value in handle.attrs.items():
            normalized = _normal_key(key)
            converted = _json_value(value)
            if converted is not None and not isinstance(converted, (list, dict)):
                attributes[str(key)] = converted
            if normalized in RATE_KEYS:
                candidate = _rate_candidate(value, f"hdf5:attrs.{key}", direct=True)
                if candidate:
                    rate_candidates.append(candidate)
            if normalized == "master_clock":
                master_clock = str(converted)

        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                base = _normal_key(Path(name).name)
                if obj.ndim and obj.shape[0] > 1:
                    if base == "partial" and obj.ndim == 1 and obj.shape[0] <= MAX_TIMESTAMP_VALUES:
                        try:
                            partial = np.asarray(obj[()], dtype=np.bool_).reshape(-1)
                            partial_entries.append({
                                "field": name,
                                "row_count": int(partial.size),
                                "partial_rows": np.flatnonzero(partial).astype(np.int64).tolist(),
                                "valid_rows": np.logical_not(partial).tolist(),
                            })
                        except (TypeError, ValueError, OSError):
                            pass
                    if base in TIMESTAMP_NAMES or "timestamp" in base or base.endswith("_ts"):
                        if obj.shape[0] <= MAX_TIMESTAMP_VALUES:
                            try:
                                values = np.asarray(obj[()], dtype=np.float64).reshape(-1)
                                summary, seconds = _timestamp_analysis(values, name, reference_fps)
                                if summary and seconds is not None:
                                    timestamp_entries.append({"summary": summary, "seconds": seconds})
                            except (TypeError, ValueError, OSError):
                                pass
                    else:
                        data_counts.append(int(obj.shape[0]))
                        if base != "partial" and obj.dtype.kind in "biufc":
                            raw_numeric_fields.append({
                                "field": name.replace("\\", "/"),
                                "data_count": int(obj.shape[0]),
                                "kind": "sensor",
                                "members": [],
                                "evidence": "hdf5_numeric_dataset",
                            })
                for key, value in obj.attrs.items():
                    if _normal_key(key) in RATE_KEYS:
                        candidate = _rate_candidate(value, f"hdf5:{name}.attrs.{key}", direct=True)
                        if candidate:
                            rate_candidates.append(candidate)

        handle.visititems(visitor)
    timestamp_entries.sort(key=lambda item: _timestamp_priority(item["summary"]["field"]))
    data_count = _representative_count(data_counts) or _representative_count([item["summary"]["count"] for item in timestamp_entries])
    partial_entry = next((item for item in partial_entries if int(item["row_count"]) == data_count), None)
    semantic_fields = _probe_numeric_fields(path)
    covered_members = {
        member.casefold()
        for item in semantic_fields
        for member in item.get("members") or []
    }
    covered_fields = {str(item.get("field") or "").casefold() for item in semantic_fields}
    grouped: dict[tuple[str, int], list[dict]] = {}
    top_level: list[dict] = []
    for item in raw_numeric_fields:
        field = str(item["field"])
        if field.casefold() in covered_members or field.casefold() in covered_fields:
            continue
        parent = field.rsplit("/", 1)[0] if "/" in field else ""
        if parent:
            grouped.setdefault((parent, int(item["data_count"])), []).append(item)
        else:
            top_level.append(item)
    numeric_fields = list(semantic_fields)
    numeric_fields.extend(top_level)
    parent_counts = Counter(parent for parent, _ in grouped)
    for (parent, count), members in grouped.items():
        if parent_counts[parent] == 1 and len(members) > 1:
            numeric_fields.append({
                "field": f"{parent}/*",
                "data_count": count,
                "kind": "sensor",
                "members": [str(item["field"]) for item in members],
                "evidence": "hdf5_common_parent_and_row_count",
            })
        else:
            numeric_fields.extend(members)
    return {
        "data_count": data_count,
        "rate_candidates": _dedupe_rate_candidates(rate_candidates),
        "timestamps": timestamp_entries,
        "master_clock": master_clock,
        "attributes": attributes,
        "partial_validity": partial_entry,
        "numeric_fields": numeric_fields,
    }


def _arrow_metadata_rates(metadata: dict[bytes, bytes] | None) -> list[dict]:
    candidates = []
    for raw_key, raw_value in (metadata or {}).items():
        key = raw_key.decode("utf-8", errors="replace") if isinstance(raw_key, bytes) else str(raw_key)
        if _normal_key(key) not in RATE_KEYS:
            continue
        value = raw_value.decode("utf-8", errors="replace") if isinstance(raw_value, bytes) else raw_value
        candidate = _rate_candidate(value, f"parquet:metadata.{key}", direct=True)
        if candidate:
            candidates.append(candidate)
    return candidates


def _probe_parquet(path: Path, reference_fps: float) -> dict:
    import pyarrow.parquet as parquet

    parquet_file = parquet.ParquetFile(path)
    names = list(parquet_file.schema_arrow.names)
    timestamp_entries = []
    for name in names:
        normalized = _normal_key(name)
        if normalized not in TIMESTAMP_NAMES and "timestamp" not in normalized and not normalized.endswith("_ts"):
            continue
        if parquet_file.metadata.num_rows > MAX_TIMESTAMP_VALUES:
            continue
        try:
            column = parquet_file.read(columns=[name]).column(0).combine_chunks()
            values = np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float64)
            summary, seconds = _timestamp_analysis(values, name, reference_fps)
            if summary and seconds is not None:
                timestamp_entries.append({"summary": summary, "seconds": seconds})
        except Exception:
            continue
    timestamp_entries.sort(key=lambda item: _timestamp_priority(item["summary"]["field"]))
    return {
        "data_count": int(parquet_file.metadata.num_rows),
        "rate_candidates": _dedupe_rate_candidates(_arrow_metadata_rates(parquet_file.schema_arrow.metadata)),
        "timestamps": timestamp_entries,
        "master_clock": None,
        "attributes": {},
        "columns": names,
        "numeric_fields": _probe_numeric_fields(path),
    }


def _walk_json_rates(value: Any, prefix: str = "json") -> list[dict]:
    result = []
    if isinstance(value, dict):
        for key, child in value.items():
            source = f"{prefix}.{key}"
            if _normal_key(key) in RATE_KEYS:
                candidate = _rate_candidate(child, source, direct=True)
                if candidate:
                    result.append(candidate)
            if isinstance(child, (dict, list)):
                result.extend(_walk_json_rates(child, source))
    elif isinstance(value, list):
        for index, child in enumerate(value[:20]):
            if isinstance(child, (dict, list)):
                result.extend(_walk_json_rates(child, f"{prefix}[{index}]"))
    return result


def _json_timestamp_arrays(value: Any, prefix: str = "$") -> list[tuple[str, Any]]:
    result = []
    if isinstance(value, dict):
        for key, child in value.items():
            field = f"{prefix}.{key}"
            normalized = _normal_key(key)
            if isinstance(child, list) and (normalized in TIMESTAMP_NAMES or "timestamp" in normalized or normalized.endswith("_ts")):
                result.append((field, child))
            elif isinstance(child, dict):
                result.extend(_json_timestamp_arrays(child, field))
    return result


def _json_data_count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if not isinstance(value, dict):
        return 0
    preferred = []
    fallback = []
    for key, child in value.items():
        if isinstance(child, list) and len(child) > 1:
            fallback.append(len(child))
            if _normal_key(key) not in TIMESTAMP_NAMES:
                preferred.append(len(child))
    return _representative_count(preferred or fallback)


def _probe_json(path: Path, reference_fps: float) -> dict:
    if path.suffix.casefold() == ".jsonl":
        timestamps: list[float] = []
        timestamp_name = None
        rate_candidates = []
        count = 0
        with path.open("r", encoding="utf-8", errors="replace") as source:
            for line in source:
                if not line.strip():
                    continue
                count += 1
                if count > MAX_TIMESTAMP_VALUES:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if count == 1:
                    rate_candidates.extend(_walk_json_rates(row, "jsonl"))
                if isinstance(row, dict):
                    for key, value in row.items():
                        normalized = _normal_key(key)
                        if normalized in TIMESTAMP_NAMES or "timestamp" in normalized or normalized.endswith("_ts"):
                            number = _as_positive_float(value)
                            if number is not None:
                                timestamps.append(number)
                                timestamp_name = key
                            break
        entries = []
        if timestamp_name and len(timestamps) > 1:
            summary, seconds = _timestamp_analysis(timestamps, timestamp_name, reference_fps)
            if summary and seconds is not None:
                entries.append({"summary": summary, "seconds": seconds})
        return {
            "data_count": count,
            "rate_candidates": _dedupe_rate_candidates(rate_candidates),
            "timestamps": entries,
            "master_clock": None,
            "attributes": {},
            "numeric_fields": _probe_numeric_fields(path),
        }
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("JSON file is too large for timing inspection")
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = []
    for name, values in _json_timestamp_arrays(payload):
        if len(values) > MAX_TIMESTAMP_VALUES:
            continue
        summary, seconds = _timestamp_analysis(values, name, reference_fps)
        if summary and seconds is not None:
            entries.append({"summary": summary, "seconds": seconds})
    entries.sort(key=lambda item: _timestamp_priority(item["summary"]["field"]))
    return {
        "data_count": _json_data_count(payload),
        "rate_candidates": _dedupe_rate_candidates(_walk_json_rates(payload)),
        "timestamps": entries,
        "master_clock": payload.get("master_clock") if isinstance(payload, dict) else None,
        "attributes": {},
        "payload": payload,
        "numeric_fields": _probe_numeric_fields(path),
    }


def _probe_numpy(path: Path, reference_fps: float) -> dict:
    del reference_fps
    arrays: list[tuple[str, np.ndarray]] = []
    if path.suffix.casefold() == ".npz":
        with np.load(path, mmap_mode="r", allow_pickle=False) as archive:
            for name in archive.files[:MAX_TABLE_COLUMNS]:
                array = archive[name]
                if array.ndim and int(array.shape[0]) > 1:
                    arrays.append((str(name), array))
    else:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim and int(array.shape[0]) > 1:
            arrays.append(("$", array))
    counts = [int(array.shape[0]) for _, array in arrays]
    fields = _probe_numeric_fields(path)
    if not fields:
        fields = [{"field": name, "data_count": int(array.shape[0]), "kind": "sensor", "members": []} for name, array in arrays]
    return {
        "data_count": _representative_count(counts),
        "rate_candidates": [],
        "timestamps": [],
        "master_clock": None,
        "attributes": {},
        "numeric_fields": fields,
    }


def _probe_table(path: Path, reference_fps: float) -> dict:
    delimiter = "\t" if path.suffix.casefold() == ".tsv" else ","
    count = 0
    fields: list[str] = []
    timestamp_values: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as source:
        reader = csv.DictReader(source, delimiter=delimiter)
        fields = [str(item) for item in (reader.fieldnames or [])[:MAX_TABLE_COLUMNS]]
        timestamp_names = [
            name for name in fields
            if _normal_key(name) in TIMESTAMP_NAMES or "timestamp" in _normal_key(name) or _normal_key(name).endswith("_ts")
        ]
        timestamp_values = {name: [] for name in timestamp_names}
        for row in reader:
            count += 1
            if count > MAX_TIMESTAMP_VALUES:
                continue
            for name in timestamp_names:
                try:
                    timestamp_values[name].append(float(row.get(name)))
                except (TypeError, ValueError):
                    timestamp_values[name].append(float("nan"))
    entries = []
    for name, values in timestamp_values.items():
        summary, seconds = _timestamp_analysis(values, name, reference_fps)
        if summary and seconds is not None:
            entries.append({"summary": summary, "seconds": seconds})
    entries.sort(key=lambda item: _timestamp_priority(item["summary"]["field"]))
    numeric_fields = _probe_numeric_fields(path)
    if not numeric_fields:
        numeric_fields = [
            {"field": name, "data_count": count, "kind": "sensor", "members": []}
            for name in fields
            if name not in timestamp_values
        ]
    return {
        "data_count": count,
        "rate_candidates": [],
        "timestamps": entries,
        "master_clock": None,
        "attributes": {},
        "columns": fields,
        "numeric_fields": numeric_fields,
    }


def _probe_file(path: Path, reference_fps: float) -> dict:
    suffix = path.suffix.casefold()
    if suffix in {".h5", ".hdf5", ".h5df"}:
        return _probe_hdf5(path, reference_fps)
    if suffix == ".parquet":
        return _probe_parquet(path, reference_fps)
    if suffix in {".json", ".jsonl"}:
        return _probe_json(path, reference_fps)
    if suffix in {".npy", ".npz"}:
        return _probe_numpy(path, reference_fps)
    if suffix in {".csv", ".tsv"}:
        return _probe_table(path, reference_fps)
    raise ValueError(f"Unsupported sensor format: {suffix}")


def _nested_get(value: Any, keys: tuple[str, ...]) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _json_file_leaves(value: Any, trail: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], str]]:
    leaves = []
    if isinstance(value, dict):
        for key, child in value.items():
            leaves.extend(_json_file_leaves(child, trail + (str(key),)))
    elif isinstance(value, str) and Path(value).suffix.casefold() in SENSOR_EXTENSIONS:
        leaves.append((trail, value.replace("\\", "/")))
    return leaves


def _metadata_rate_hints(root: Path, records: list[dict], reference_fps: float) -> dict[str, list[dict]]:
    """Link metadata ``sensors`` declarations to files in the same Episode."""
    known = [str(record.get("relative_path") or "").replace("\\", "/") for record in records]
    result: dict[str, list[dict]] = {}
    for record in records:
        relative = str(record.get("relative_path") or "")
        if Path(relative).suffix.casefold() != ".json":
            continue
        path = root / relative
        if not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("files"), dict):
            continue
        for trail, declared_path in _json_file_leaves(payload["files"]):
            target = next((item for item in known if item.casefold().endswith(declared_path.casefold())), None)
            if not target:
                continue
            sensor_metadata = _nested_get(payload.get("sensors"), trail)
            if not isinstance(sensor_metadata, dict):
                continue
            for key, value in sensor_metadata.items():
                if _normal_key(key) not in RATE_KEYS:
                    continue
                candidate = _rate_candidate(value, f"metadata:{relative}:sensors.{'.'.join(trail)}.{key}")
                if candidate:
                    result.setdefault(target, []).append(candidate)
    return {key: _dedupe_rate_candidates(value) for key, value in result.items()}


def _reference_camera_name(reference: dict) -> str:
    candidates = [reference.get("stream_name"), reference.get("relative_path")]
    normalized = []
    for candidate in candidates:
        if not candidate:
            continue
        stem = _normal_key(Path(str(candidate)).stem)
        normalized.append(stem)
        if "wrist_left" in stem:
            return "wrist_left"
        if "wrist_right" in stem:
            return "wrist_right"
        if "head_rgb" in stem or ("head" in stem and "depth" not in stem):
            return "head"
    return normalized[0] if normalized else "video"


def _strict_frame_index(value: Any) -> int | None:
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def _timeline_frame_period(timeline: np.ndarray, fps: float) -> float:
    finite_rows = np.flatnonzero(np.isfinite(timeline))
    if finite_rows.size >= 2:
        left, right = int(finite_rows[-2]), int(finite_rows[-1])
        delta = float(timeline[right] - timeline[left]) / max(1, right - left)
        if math.isfinite(delta) and delta > 0:
            return delta
    return 1.0 / max(0.001, fps)


def _video_timestamp_timeline(path: Path, relative: str, reference: dict, camera_name: str) -> tuple[np.ndarray, dict] | None:
    import pyarrow.compute as compute
    import pyarrow.parquet as parquet

    fps = float(reference["fps"])
    frame_count = int(reference.get("frame_count") or 0)
    parquet_file = parquet.ParquetFile(path)
    schema_names = parquet_file.schema_arrow.names
    if "frame_idx" not in schema_names or ("ts_wall" not in schema_names and "pts_us" not in schema_names):
        return None
    field = "ts_wall" if "ts_wall" in schema_names else "pts_us"
    columns = ["frame_idx", field]
    if "camera" in schema_names:
        columns.append("camera")
    table = parquet.read_table(path, columns=columns)
    if "camera" in columns:
        table = table.filter(compute.equal(table["camera"], camera_name))
    if table.num_rows < 2:
        return None
    raw_indices = table["frame_idx"].to_pylist()
    raw = np.asarray(table[field].to_numpy(zero_copy_only=False), dtype=np.float64)
    summary, seconds = _timestamp_analysis(raw, field, fps)
    if summary is None or seconds is None:
        return None
    parsed_indices = [_strict_frame_index(value) for value in raw_indices]
    nonnegative = [value for value in parsed_indices if value is not None and value >= 0]
    count = frame_count or ((max(nonnegative) + 1) if nonnegative else 0)
    if count <= 0:
        return None
    timeline = np.full(count, np.nan, dtype=np.float64)
    invalid_reasons = Counter()
    last_index = -1
    last_timestamp = -math.inf
    for frame_index, timestamp in zip(parsed_indices, seconds, strict=True):
        if frame_index is None:
            invalid_reasons["non_integer_index"] += 1
            continue
        if not 0 <= frame_index < count:
            invalid_reasons["out_of_media_range"] += 1
            continue
        if not math.isfinite(float(timestamp)):
            invalid_reasons["non_finite_timestamp"] += 1
            continue
        if frame_index <= last_index:
            invalid_reasons["duplicate_or_backward_index"] += 1
            continue
        if float(timestamp) <= last_timestamp:
            invalid_reasons["non_monotonic_timestamp"] += 1
            continue
        timeline[frame_index] = float(timestamp)
        last_index = frame_index
        last_timestamp = float(timestamp)
    accepted = int(np.count_nonzero(np.isfinite(timeline)))
    if accepted < 2:
        return None
    quality = dict(summary)
    quality.update({
        "accepted_frame_count": accepted,
        "invalid_frame_count": int(sum(invalid_reasons.values())),
        "invalid_reasons": dict(invalid_reasons),
        "media_frame_count": count,
    })
    return timeline, {
        "relative_path": relative,
        "field": summary["field"],
        "camera": camera_name,
        "method": "video_timestamps",
        "quality": quality,
        "repairs": [],
    }


def _sync_camera_rows(
    path: Path,
    relative: str,
    camera_name: str,
    frame_count: int,
    fps: float,
    reference_timeline: np.ndarray | None = None,
) -> tuple[list[dict], list[dict], dict] | None:
    import pyarrow.parquet as parquet

    fields = SYNC_CAMERA_FIELDS.get(camera_name)
    if fields is None or frame_count <= 0:
        return None
    index_field, timestamp_field, filled_field = fields
    schema_names = parquet.ParquetFile(path).schema_arrow.names
    if index_field not in schema_names or timestamp_field not in schema_names:
        return None
    columns = [index_field, timestamp_field]
    if filled_field in schema_names:
        columns.append(filled_field)
    if "partial" in schema_names:
        columns.append("partial")
    table = parquet.read_table(path, columns=columns)
    raw_indices = table[index_field].to_pylist()
    raw = np.asarray(table[timestamp_field].to_numpy(zero_copy_only=False), dtype=np.float64)
    summary, seconds = _timestamp_analysis(raw, timestamp_field, fps)
    if summary is None or seconds is None:
        return None
    filled = table[filled_field].to_pylist() if filled_field in columns else [False] * table.num_rows
    partial = table["partial"].to_pylist() if "partial" in columns else [False] * table.num_rows
    reference_period = _timeline_frame_period(reference_timeline, fps) if reference_timeline is not None else 1.0 / max(0.001, fps)
    match_tolerance = max(1e-6, reference_period * 0.35)
    entries = []
    legal = []
    rejected = Counter()
    last_index = -1
    last_timestamp = -math.inf
    for row_index, (raw_index, timestamp, is_filled, is_partial) in enumerate(zip(raw_indices, seconds, filled, partial, strict=True)):
        frame_index = _strict_frame_index(raw_index)
        entry = {
            "row": row_index,
            "original_frame_idx": frame_index,
            "timestamp": float(timestamp) if math.isfinite(float(timestamp)) else None,
            "filled": bool(is_filled),
            "partial": bool(is_partial),
            "legal": False,
        }
        entries.append(entry)
        if entry["filled"]:
            rejected["filled"] += 1
            continue
        if entry["partial"]:
            rejected["partial"] += 1
            continue
        if frame_index is None:
            rejected["non_integer_index"] += 1
            continue
        if not 0 <= frame_index < frame_count:
            rejected["out_of_media_range"] += 1
            continue
        if entry["timestamp"] is None:
            rejected["non_finite_timestamp"] += 1
            continue
        if frame_index <= last_index:
            rejected["duplicate_or_backward_index"] += 1
            continue
        if entry["timestamp"] <= last_timestamp:
            rejected["non_monotonic_timestamp"] += 1
            continue
        if reference_timeline is not None:
            reference_timestamp = float(reference_timeline[frame_index])
            if not math.isfinite(reference_timestamp):
                rejected["reference_timestamp_missing"] += 1
                continue
            if abs(reference_timestamp - entry["timestamp"]) > match_tolerance:
                rejected["reference_timestamp_mismatch"] += 1
                continue
        entry["legal"] = True
        legal.append(entry)
        last_index = frame_index
        last_timestamp = entry["timestamp"]
    validation = {
        "relative_path": relative,
        "camera": camera_name,
        "index_field": index_field,
        "timestamp_field": timestamp_field,
        "filled_field": filled_field if filled_field in schema_names else None,
        "row_count": int(table.num_rows),
        "legal_row_count": len(legal),
        "rejected_row_count": int(sum(rejected.values())),
        "rejected_reasons": dict(rejected),
        "partial_row_count": int(sum(bool(value) for value in partial)),
        "filled_row_count": int(sum(bool(value) for value in filled)),
        "match_tolerance_seconds": round(match_tolerance, 9),
        "timestamp_quality": summary,
    }
    return entries, legal, validation


def _repair_trailing_video_timestamp(
    timeline: np.ndarray,
    entries: list[dict],
    legal: list[dict],
    validation: dict,
) -> dict | None:
    finite_rows = np.flatnonzero(np.isfinite(timeline))
    if finite_rows.size < 2 or len(legal) < 2:
        return None
    previous_index, last_index = int(finite_rows[-2]), int(finite_rows[-1])
    candidate_index = last_index + 1
    if candidate_index >= timeline.size or math.isfinite(float(timeline[candidate_index])):
        return None
    table_period = float(timeline[last_index] - timeline[previous_index]) / max(1, last_index - previous_index)
    left, right = legal[-2], legal[-1]
    sync_index_delta = int(right["original_frame_idx"]) - int(left["original_frame_idx"])
    sync_time_delta = float(right["timestamp"]) - float(left["timestamp"])
    if table_period <= 0 or sync_index_delta <= 0 or sync_time_delta <= 0:
        return None
    sync_period = sync_time_delta / sync_index_delta
    expected_timestamp = float(timeline[last_index]) + table_period
    tolerance = max(0.002, min(table_period, sync_period) * 0.35)
    candidates = []
    for entry in entries:
        observed = entry.get("timestamp")
        original = entry.get("original_frame_idx")
        if entry.get("filled") or entry.get("partial") or observed is None:
            continue
        if not float(observed) > float(right["timestamp"]):
            continue
        if abs(float(observed) - expected_timestamp) > tolerance:
            continue
        original_is_invalid = original is None or not 0 <= int(original) < timeline.size
        if not original_is_invalid and int(original) != candidate_index:
            continue
        derived = int(round(int(right["original_frame_idx"]) + (float(observed) - float(right["timestamp"])) / sync_period))
        if derived != candidate_index:
            continue
        candidates.append((abs(float(observed) - expected_timestamp), entry, derived))
    if not candidates:
        return None
    _, selected, derived_index = min(candidates, key=lambda item: (item[0], int(item[1]["row"])))
    observed_timestamp = float(selected["timestamp"])
    timeline[candidate_index] = observed_timestamp
    return {
        "kind": "single_trailing_timestamp_extrapolation",
        "camera": validation.get("camera"),
        "frame_idx": candidate_index,
        "timestamp": observed_timestamp,
        "source_relative_path": validation.get("relative_path"),
        "source_row": int(selected["row"]),
        "source_timestamp_field": validation.get("timestamp_field"),
        "invalid_original_frame_idx": selected.get("original_frame_idx") if selected.get("original_frame_idx") != candidate_index else None,
        "derived_frame_idx": derived_index,
        "expected_timestamp": expected_timestamp,
        "timestamp_delta_ms": round((observed_timestamp - expected_timestamp) * 1000.0, 6),
        "media_frame_count": int(timeline.size),
        "basis": {
            "video_timestamp_indices": [previous_index, last_index],
            "sync_legal_indices": [int(left["original_frame_idx"]), int(right["original_frame_idx"])],
            "sync_legal_rows": [int(left["row"]), int(right["row"])],
        },
        "validity": {"filled": bool(selected.get("filled")), "partial": bool(selected.get("partial"))},
    }


def _sync_reference_timeline(path: Path, relative: str, reference: dict, camera_name: str) -> tuple[np.ndarray, dict] | None:
    frame_count = int(reference.get("frame_count") or 0)
    fps = float(reference["fps"])
    result = _sync_camera_rows(path, relative, camera_name, frame_count, fps)
    if result is None:
        return None
    _, legal, validation = result
    if len(legal) < 2:
        return None
    timeline = np.full(frame_count, np.nan, dtype=np.float64)
    for entry in legal:
        timeline[int(entry["original_frame_idx"])] = float(entry["timestamp"])
    return timeline, {
        "relative_path": relative,
        "field": validation["timestamp_field"],
        "camera": camera_name,
        "method": "sync_camera_index",
        "quality": validation["timestamp_quality"],
        "sync_validation": validation,
        "repairs": [],
    }


def _reference_timestamps(root: Path, records: list[dict], reference: dict) -> tuple[np.ndarray | None, dict | None]:
    fps = float(reference["fps"])
    frame_count = int(reference.get("frame_count") or 0)
    camera_name = _reference_camera_name(reference)
    parquet_records = [
        record
        for record in records
        if str(record.get("extension") or Path(str(record.get("relative_path") or "")).suffix).casefold() == ".parquet"
    ]
    video_records = [record for record in parquet_records if "video_timestamp" in str(record.get("relative_path") or "").casefold()]
    sync_records = [record for record in parquet_records if "sync" in Path(str(record.get("relative_path") or "")).name.casefold()]
    for record in video_records:
        relative = str(record.get("relative_path") or "")
        try:
            result = _video_timestamp_timeline(root / relative, relative, reference, camera_name)
            if result is None:
                continue
            timeline, source = result
            for sync_record in sync_records:
                sync_relative = str(sync_record.get("relative_path") or "")
                try:
                    sync_result = _sync_camera_rows(
                        root / sync_relative,
                        sync_relative,
                        camera_name,
                        int(timeline.size),
                        fps,
                        reference_timeline=timeline,
                    )
                except Exception:
                    continue
                if sync_result is None:
                    continue
                entries, legal, validation = sync_result
                source["sync_validation"] = validation
                repair = _repair_trailing_video_timestamp(timeline, entries, legal, validation)
                if repair:
                    source["repairs"].append(repair)
                break
            return timeline, source
        except Exception:
            continue
    for record in sync_records:
        relative = str(record.get("relative_path") or "")
        try:
            result = _sync_reference_timeline(root / relative, relative, reference, camera_name)
            if result is not None:
                return result
        except Exception:
            continue
    if frame_count > 0:
        return np.arange(frame_count, dtype=np.float64) / fps, {
            "relative_path": reference.get("relative_path"),
            "field": "decoded_frame_index/fps",
            "camera": camera_name,
            "method": "synthetic_frame_rate",
            "quality": None,
            "repairs": [],
        }
    return None, None


def _nearest_timestamp_lookup(reference_seconds: np.ndarray, sensor_seconds: np.ndarray, stored_hz: float | None) -> tuple[list[int] | None, dict]:
    reference_seconds = np.asarray(reference_seconds, dtype=np.float64).reshape(-1)
    sensor_seconds = np.asarray(sensor_seconds, dtype=np.float64).reshape(-1)
    valid_sensor = np.flatnonzero(np.isfinite(sensor_seconds))
    if valid_sensor.size < 2:
        return None, {"invalid_frame_count": int(reference_seconds.size), "reason": "insufficient_sensor_timestamps"}
    values = sensor_seconds[valid_sensor]
    valid_reference = reference_seconds[np.isfinite(reference_seconds)]
    origin_offset_seconds = None
    origin_alignment_mode = "preserved"
    if valid_reference.size:
        reference_origin = float(valid_reference[0])
        sensor_origin = float(values[0])
        # Preserve offsets when both streams use the same kind of clock.  Only
        # remove independent origins when one side is an epoch-like wall clock
        # and the other is a relative PTS/device timeline.
        reference_is_absolute = abs(reference_origin) >= 1_000_000.0
        sensor_is_absolute = abs(sensor_origin) >= 1_000_000.0
        reference_duration = float(valid_reference[-1] - valid_reference[0]) if valid_reference.size >= 2 else 0.0
        sensor_duration = float(values[-1] - values[0]) if values.size >= 2 else 0.0
        durations_compatible = (
            reference_duration > 0.0
            and sensor_duration > 0.0
            and abs(reference_duration - sensor_duration) <= max(0.1, 0.02 * max(reference_duration, sensor_duration))
        )
        origins_normalized = reference_is_absolute != sensor_is_absolute or (
            not reference_is_absolute
            and not sensor_is_absolute
            and durations_compatible
            and abs(reference_origin - sensor_origin) > max(0.05, 2.5 / stored_hz if stored_hz else 0.05)
        )
        if origins_normalized:
            origin_offset_seconds = sensor_origin - reference_origin
            origin_alignment_mode = "mixed_clock_start_repair" if reference_is_absolute != sensor_is_absolute else "relative_clock_start_repair"
            reference_seconds = reference_seconds - reference_origin
            values = values - sensor_origin
    else:
        origins_normalized = False
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_rows = valid_sensor[order]
    unique_values, unique_positions = np.unique(sorted_values, return_index=True)
    sorted_values = unique_values
    sorted_rows = sorted_rows[unique_positions]
    if sorted_values.size < 2:
        return None, {"invalid_frame_count": int(reference_seconds.size), "reason": "constant_sensor_timestamps"}
    positive = np.diff(sorted_values)
    median_delta = float(np.median(positive[positive > 0]))
    max_distance = max(2.5 * median_delta, 2.5 / stored_hz if stored_hz else 0.0)
    gap_threshold = 3.0 * median_delta
    mapping: list[int] = []
    invalid = 0
    long_gap_rejections = 0
    accepted_distances: list[float] = []
    for timestamp in reference_seconds:
        if not math.isfinite(float(timestamp)):
            mapping.append(-1)
            invalid += 1
            continue
        position = int(np.searchsorted(sorted_values, timestamp, side="left"))
        candidates = []
        if position < sorted_values.size:
            candidates.append(position)
        if position > 0:
            candidates.append(position - 1)
        nearest = min(candidates, key=lambda index: abs(float(sorted_values[index] - timestamp)))
        distance = abs(float(sorted_values[nearest] - timestamp))
        in_long_gap = (
            0 < position < sorted_values.size
            and float(sorted_values[position] - sorted_values[position - 1]) > gap_threshold
            and sorted_values[position - 1] < timestamp < sorted_values[position]
        )
        if distance > max_distance or in_long_gap:
            mapping.append(-1)
            invalid += 1
            long_gap_rejections += int(in_long_gap)
        else:
            mapping.append(int(sorted_rows[nearest]))
            accepted_distances.append(distance)
    distance_array = np.asarray(accepted_distances, dtype=np.float64)
    return mapping, {
        "invalid_frame_count": invalid,
        "valid_frame_count": len(mapping) - invalid,
        "valid_coverage": round((len(mapping) - invalid) / max(1, len(mapping)), 9),
        "long_gap_rejection_count": long_gap_rejections,
        "allowed_max_match_distance_seconds": round(max_distance, 9),
        "actual_max_match_error_seconds": round(float(distance_array.max()), 9) if distance_array.size else None,
        "p95_match_error_seconds": round(float(np.percentile(distance_array, 95)), 9) if distance_array.size else None,
        "origins_normalized": origins_normalized,
        "origin_alignment_mode": origin_alignment_mode,
        "origin_offset_seconds": round(float(origin_offset_seconds), 9) if origin_offset_seconds is not None else None,
    }


def _select_physical_hz(candidates: list[dict]) -> tuple[float | None, bool]:
    if not candidates:
        return None, False
    direct = [item for item in candidates if item.get("direct")]
    selected = float((direct or candidates)[0]["hz"])
    distinct = {round(float(item["hz"]), 6) for item in candidates}
    return selected, len(distinct) > 1


def _preserve_partial_validity(
    lookup: list[int] | None,
    mode: str,
    partial_entry: dict | None,
    video_count: int,
    data_count: int,
    multiplier: float,
) -> tuple[list[int] | None, list[int], int]:
    if not partial_entry:
        return lookup, [], 0
    valid_rows = np.asarray(partial_entry.get("valid_rows") or [], dtype=np.bool_).reshape(-1)
    if valid_rows.size != data_count:
        return lookup, [], 0
    if lookup is None:
        mapping_count = video_count or data_count
        if mode in {"prealigned_master_clock", "paired_frame_index"}:
            lookup = list(range(mapping_count))
        else:
            lookup = [int(round(frame * multiplier)) for frame in range(mapping_count)]
    partial_video_frames = []
    rejected = 0
    for video_frame, source_row in enumerate(lookup):
        row = int(source_row)
        if row < 0:
            continue
        if row >= data_count:
            lookup[video_frame] = -1
            continue
        if not bool(valid_rows[row]):
            lookup[video_frame] = -1
            partial_video_frames.append(video_frame)
            rejected += 1
    return lookup, partial_video_frames, rejected


def _mapping_quality(
    mode: str,
    lookup: list[int] | None,
    video_count: int,
    data_count: int,
    multiplier: float,
    timestamp_quality: dict | None,
) -> dict:
    if video_count <= 0:
        return {"mapped_count": 0, "invalid_frame_count": 0, "valid_coverage": 0.0}
    if isinstance(lookup, list):
        values = np.asarray(lookup[:video_count], dtype=np.int64)
        if values.size < video_count:
            values = np.pad(values, (0, video_count - values.size), constant_values=-1)
        valid = (values >= 0) & (values < max(0, data_count))
    elif mode in {"prealigned_master_clock", "paired_frame_index"}:
        values = np.arange(video_count, dtype=np.int64)
        valid = values < max(0, data_count)
    else:
        values = np.rint(np.arange(video_count) * float(multiplier or 1.0)).astype(np.int64)
        valid = (values >= 0) & (values < max(0, data_count))
    mapped = int(valid.sum())
    quality = dict(timestamp_quality or {})
    quality.update({
        "reference_frame_count": int(video_count),
        "mapped_count": mapped,
        "invalid_frame_count": int(video_count - mapped),
        "valid_coverage": round(mapped / max(1, video_count), 9),
    })
    return quality


def _field_clock_entries(probe: dict, field: str, data_count: int, members: list[str] | None = None) -> list[dict]:
    normalized_field = str(field or "").replace("\\", "/").casefold()
    normalized_members = {str(item).replace("\\", "/").casefold() for item in members or []}
    entries = []
    for entry in probe.get("timestamps") or []:
        summary = entry.get("summary") or {}
        timestamp_field = str(summary.get("field") or "").replace("\\", "/")
        normalized_timestamp = timestamp_field.casefold()
        timestamp_parent = normalized_timestamp.rsplit("/", 1)[0] if "/" in normalized_timestamp else ""
        field_parent = normalized_field[:-2].rstrip("/") if normalized_field.endswith("/*") else normalized_field.rsplit("/", 1)[0] if "/" in normalized_field else ""
        same_count = int(summary.get("count") or 0) == int(data_count)
        same_parent = bool(field_parent and timestamp_parent == field_parent)
        member_parent = any(member.rsplit("/", 1)[0] == timestamp_parent for member in normalized_members if "/" in member)
        if same_count and (same_parent or member_parent):
            entries.append(entry)
    if entries:
        return entries
    matching_count = [
        entry for entry in probe.get("timestamps") or []
        if int((entry.get("summary") or {}).get("count") or 0) == int(data_count)
    ]
    if len(matching_count) == 1:
        return matching_count
    return []


def _build_stream(
    relative_path: str,
    probe: dict,
    metadata_hints: list[dict],
    reference: dict,
    reference_seconds: np.ndarray | None,
    field_descriptor: dict | None = None,
) -> dict | None:
    descriptor = field_descriptor or {}
    field = str(descriptor.get("field") or "").replace("\\", "/") or None
    data_count = int(descriptor.get("data_count") or probe.get("data_count") or 0)
    if data_count <= 1:
        return None
    rate_candidates = _dedupe_rate_candidates([*(probe.get("rate_candidates") or []), *metadata_hints])
    physical_hz, rate_conflict = _select_physical_hz(rate_candidates)
    timestamp_entries = (
        _field_clock_entries(probe, field, data_count, list(descriptor.get("members") or []))
        if field
        else list(probe.get("timestamps") or [])
    )
    aligned_entry = next((item for item in timestamp_entries if _timestamp_priority(item["summary"]["field"]) <= 2), None)
    sensor_entry = next((item for item in timestamp_entries if "sensor" in _normal_key(item["summary"]["field"])), None)
    mapping_entry = aligned_entry or sensor_entry
    aligned_summary = (aligned_entry or {}).get("summary", {})
    aligned_hz = _as_positive_float((aligned_entry or {}).get("summary", {}).get("measured_hz"))
    measured_sensor_hz = _as_positive_float((sensor_entry or {}).get("summary", {}).get("measured_hz"))
    video_fps = float(reference["fps"])
    video_count = int(reference.get("frame_count") or 0)
    duration = _as_positive_float(reference.get("duration"))
    count_hz = data_count / duration if duration else None
    count_matches = video_count > 0 and abs(data_count - video_count) <= max(2, round(video_count * 0.01))
    timestamp_matches = aligned_hz is not None and abs(aligned_hz - video_fps) / video_fps <= 0.08
    timestamp_discontinuous = bool(
        int(aligned_summary.get("long_gap_count") or 0) > 0
        or int(aligned_summary.get("backward_count") or 0) > 0
        or int(aligned_summary.get("finite_count") or data_count) < int(aligned_summary.get("count") or data_count)
    )
    if not timestamp_entries and not rate_candidates and not count_matches:
        return None
    master_clock = str(probe.get("master_clock") or "").strip() or None
    repairs: list[dict] = []
    if count_matches and master_clock and not timestamp_discontinuous:
        mode = "prealigned_master_clock"
        stored_hz = aligned_hz or video_fps
        multiplier = 1.0
    else:
        stored_hz = aligned_hz or measured_sensor_hz or count_hz or physical_hz
        multiplier = stored_hz / video_fps if stored_hz else 1.0
        mode = "rate_multiplier"
    lookup = None
    lookup_quality = None
    if mode not in {"prealigned_master_clock", "paired_frame_index"} and reference_seconds is not None and mapping_entry is not None:
        lookup, lookup_quality = _nearest_timestamp_lookup(reference_seconds, mapping_entry["seconds"], stored_hz)
        if lookup is not None:
            mode = "timestamp_nearest"
            if lookup_quality.get("origins_normalized"):
                repairs.append({
                    "kind": str(lookup_quality.get("origin_alignment_mode") or "clock_start_repair"),
                    "offset_seconds": lookup_quality.get("origin_offset_seconds"),
                    "evidence": "compatible durations and sampling cadence",
                })
    if lookup is None and count_matches and aligned_entry is None:
        mode = "paired_frame_index"
        stored_hz = video_fps
        multiplier = 1.0
        repairs.append({
            "kind": "paired_frame_index_repair",
            "evidence": "row count matches reference frame count; no explicit timestamps",
        })
    partial_entry = probe.get("partial_validity")
    lookup, partial_video_frames, partial_rejections = _preserve_partial_validity(
        lookup,
        mode,
        partial_entry,
        video_count,
        data_count,
        float(multiplier or 1.0),
    )
    lookup_quality = _mapping_quality(
        mode,
        lookup,
        video_count,
        data_count,
        float(multiplier or 1.0),
        lookup_quality,
    )
    partial_summary = None
    if partial_entry:
        partial_summary = {
            "field": partial_entry.get("field"),
            "row_count": int(partial_entry.get("row_count") or 0),
            "partial_row_count": len(partial_entry.get("partial_rows") or []),
            "partial_rows": list(partial_entry.get("partial_rows") or []),
            "preserved_in_lookup": lookup is not None,
        }
        lookup_quality = dict(lookup_quality or {})
        lookup_quality.update({
            "partial_source_row_count": partial_summary["partial_row_count"],
            "partial_mapping_rejection_count": partial_rejections,
        })
    timestamp_summaries = [item["summary"] for item in timestamp_entries]
    return {
        "relative_path": relative_path.replace("\\", "/"),
        "field": field,
        "stream_key": f"{relative_path.replace('\\', '/')}::{field or '$'}",
        "kind": str(descriptor.get("kind") or "sensor"),
        "role": str(descriptor.get("role") or ""),
        "modality": str(descriptor.get("modality") or ""),
        "members": list(descriptor.get("members") or []),
        "data_count": data_count,
        "physical_hz": round(physical_hz, 9) if physical_hz else None,
        "stored_hz": round(stored_hz, 9) if stored_hz else None,
        "effective_stored_hz": round(stored_hz, 9) if stored_hz else None,
        "aligned_timestamp_hz": round(aligned_hz, 9) if aligned_hz else None,
        "measured_sensor_hz": round(measured_sensor_hz, 9) if measured_sensor_hz else None,
        "index_multiplier": round(multiplier, 12),
        "mode": mode,
        "master_clock": master_clock,
        "rate_candidates": rate_candidates,
        "rate_conflict": rate_conflict,
        "timestamp_quality": timestamp_summaries,
        "frame_to_sensor_index": lookup,
        "lookup_quality": lookup_quality,
        "partial_validity": partial_summary,
        "partial_video_frames": partial_video_frames,
        "repairs": repairs,
        "repair_applied": bool(repairs),
        "mapping_rule": (
            "explicit lookup; -1 marks timestamp gaps, out-of-range rows, or source partial rows"
            if lookup is not None and partial_summary is not None
            else "sensor_index = video_frame_index"
            if mode in {"prealigned_master_clock", "paired_frame_index"}
            else "nearest sensor timestamp; -1 marks gaps that must not be interpolated"
            if mode == "timestamp_nearest"
            else "sensor_index = round(video_frame_index * index_multiplier)"
        ),
    }


def scan_episode_sensor_alignment(
    manifest: dict,
    episode: dict,
    force: bool = False,
    reference_media_file_id: str | None = None,
) -> dict:
    """Inspect one Episode and persist its timing index under ``.alicePD``."""
    root = Path(manifest["root_path"]).expanduser().resolve()
    records = _episode_records(manifest, episode)
    reference = _reference_video(episode, reference_media_file_id)
    source_signature = _episode_signature(root, records, reference)
    artifact_path = sensor_alignment_path(manifest, str(episode["id"]), reference.get("file_id"))
    existing_path = _existing_sensor_alignment_path(manifest, str(episode["id"]), reference.get("file_id"))
    if not force and existing_path.is_file():
        try:
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            if existing.get("schema") == SENSOR_ALIGNMENT_SCHEMA and existing.get("source_signature") == source_signature:
                existing["artifact_path"] = str(existing_path)
                return existing
        except (OSError, json.JSONDecodeError):
            pass
    reference_seconds, reference_timestamp_source = _reference_timestamps(root, records, reference)
    metadata_hints = _metadata_rate_hints(root, records, float(reference["fps"]))
    streams = []
    skipped_files = []
    alignment_sources = []
    for record in records:
        relative = str(record.get("relative_path") or "")
        suffix = str(record.get("extension") or Path(relative).suffix).casefold()
        if suffix not in SENSOR_EXTENSIONS:
            continue
        path = root / relative
        if not path.is_file():
            skipped_files.append({"relative_path": relative, "reason": "source_missing"})
            continue
        lowered_name = path.name.casefold()
        if suffix == ".parquet" and ("sync" in lowered_name or "video_timestamp" in lowered_name):
            alignment_sources.append(relative.replace("\\", "/"))
            continue
        try:
            probe = _probe_file(path, float(reference["fps"]))
            field_descriptors = list(probe.get("numeric_fields") or [])
            built_streams = []
            for descriptor in field_descriptors or [None]:
                stream = _build_stream(
                    relative,
                    probe,
                    metadata_hints.get(relative.replace("\\", "/"), []),
                    reference,
                    reference_seconds,
                    field_descriptor=descriptor,
                )
                if stream:
                    stream["source_signature"] = _source_signature(path)
                    built_streams.append(stream)
                elif descriptor is not None:
                    skipped_files.append({
                        "relative_path": relative,
                        "field": descriptor.get("field"),
                        "kind": descriptor.get("kind"),
                        "data_count": int(descriptor.get("data_count") or 0),
                        "reason": "no_time_series",
                    })
            if built_streams:
                streams.extend(built_streams)
            elif not field_descriptors:
                skipped_files.append({"relative_path": relative, "reason": "no_time_series"})
        except Exception as exc:
            skipped_files.append({"relative_path": relative, "reason": "inspection_failed", "error": str(exc)[:240]})
    document = {
        "schema": SENSOR_ALIGNMENT_SCHEMA,
        "dataset_id": manifest["id"],
        "episode_id": episode["id"],
        "episode_name": episode.get("name"),
        "reference_media_file_id": reference.get("file_id"),
        "created_at": _utc_now(),
        "source_policy": "Source dataset files are read-only; this timing index is stored only in .alicePD.",
        "reference_video": reference,
        "reference_timestamp_source": reference_timestamp_source,
        "alignment_sources": alignment_sources,
        "streams": streams,
        "skipped_files": skipped_files,
        "source_signature": source_signature,
    }
    _write_json_atomic(artifact_path, document)
    document["artifact_path"] = str(artifact_path)
    return document


def validate_episode_time_sync(manifest: dict, episode: dict, document: dict) -> dict:
    """Validate an existing timing document as the T0 processing gate."""
    reference = document.get("reference_video") or {}
    if int(reference.get("frame_count") or 0) <= 0 or float(reference.get("fps") or 0.0) <= 0.01:
        raise RuntimeError("T0 时间同步失败：参考视频缺少有效帧数或帧率")

    records = {
        str(item.get("relative_path") or "").replace("\\", "/").casefold(): item
        for item in _episode_records(manifest, episode)
    }
    blocking = []
    streams_by_path: dict[str, list[dict]] = {}
    for stream in document.get("streams") or []:
        relative = str(stream.get("relative_path") or "").replace("\\", "/").casefold()
        streams_by_path.setdefault(relative, []).append(stream)
    critical_records = [
        record for record in records.values()
        if str(record.get("canonical_kind") or record.get("kind") or "") in {"joint", "action", "sensor"}
        and str(record.get("extension") or Path(str(record.get("relative_path") or "")).suffix).casefold() in SENSOR_EXTENSIONS
    ]
    critical_paths = {
        str(record.get("relative_path") or "").replace("\\", "/").casefold()
        for record in critical_records
    }
    for item in document.get("skipped_files") or []:
        reason = str(item.get("reason") or "")
        if reason not in {"source_missing", "inspection_failed", "no_time_series"}:
            continue
        relative = str(item.get("relative_path") or "").replace("\\", "/")
        normalized = relative.casefold()
        if normalized not in critical_paths:
            continue
        field = str(item.get("field") or "")
        dynamic_descriptor = (
            bool(field)
            and int(item.get("data_count") or 0) > 4
            and str(item.get("kind") or "sensor") in {"joint", "action", "sensor"}
        )
        file_unavailable = not field and not streams_by_path.get(normalized)
        if dynamic_descriptor or file_unavailable:
            blocking.append({
                "relative_path": relative,
                "field": field or None,
                "reason": reason,
                "error": item.get("error"),
            })
    repaired_stream_count = 0
    warning_streams = []
    validated_stream_keys: set[tuple[str, str]] = set()
    for record in critical_records:
        relative = str(record.get("relative_path") or "").replace("\\", "/")
        candidates = streams_by_path.get(relative.casefold()) or []
        if not candidates:
            blocking.append({"relative_path": relative, "reason": "critical_stream_unmapped", "error": None})
            continue
        for stream in candidates:
            if str(stream.get("kind") or "sensor") not in {"joint", "action", "sensor"}:
                continue
            validated_stream_keys.add((
                relative.casefold(),
                str(stream.get("stream_key") or stream.get("field") or "").casefold(),
            ))
            coverage = float((stream.get("lookup_quality") or {}).get("valid_coverage") or 0.0)
            repaired_stream_count += int(bool(stream.get("repair_applied")))
            if coverage <= 0.0:
                blocking.append({
                    "relative_path": relative,
                    "field": stream.get("field"),
                    "reason": "zero_valid_coverage",
                    "error": None,
                })
            elif coverage < 0.98:
                warning_streams.append({
                    "relative_path": relative,
                    "field": stream.get("field"),
                    "reason": "partial_coverage_after_repair",
                    "valid_coverage": round(coverage, 9),
                })
    # Older manifests may classify a structured file as ``other`` even though
    # the field-level probe identifies real joint/sensor timelines inside it.
    # T0 validates what it actually discovered so repair and coverage status do
    # not disappear merely because the cached file label is stale.
    for relative, candidates in streams_by_path.items():
        for stream in candidates:
            if str(stream.get("kind") or "sensor") not in {"joint", "action", "sensor"}:
                continue
            stream_key = (
                relative,
                str(stream.get("stream_key") or stream.get("field") or "").casefold(),
            )
            if stream_key in validated_stream_keys:
                continue
            coverage = float((stream.get("lookup_quality") or {}).get("valid_coverage") or 0.0)
            repaired_stream_count += int(bool(stream.get("repair_applied")))
            if coverage <= 0.0:
                blocking.append({
                    "relative_path": str(stream.get("relative_path") or relative),
                    "field": stream.get("field"),
                    "reason": "zero_valid_coverage",
                    "error": None,
                })
            elif coverage < 0.98:
                warning_streams.append({
                    "relative_path": str(stream.get("relative_path") or relative),
                    "field": stream.get("field"),
                    "reason": "partial_coverage_after_repair",
                    "valid_coverage": round(coverage, 9),
                })
    if blocking:
        first = blocking[0]
        raise RuntimeError(f"T0 时间同步失败：关键数据流无法读取 {first['relative_path']}（{first['reason']}）")

    document["gate"] = {
        "status": "ready",
        "quality_status": "ready_with_repairs" if repaired_stream_count or warning_streams else "ready",
        "blocking_stream_count": 0,
        "warning_stream_count": len(warning_streams),
        "warnings": warning_streams,
        "repaired_stream_count": repaired_stream_count,
        "reference_media_file_id": reference.get("file_id"),
        "reference_fps": reference.get("fps"),
        "reference_frame_count": reference.get("frame_count"),
    }
    return document


def ensure_episode_time_sync(
    manifest: dict,
    episode: dict,
    force: bool = False,
    reference_media_file_id: str | None = None,
) -> dict:
    """Build and validate the Episode time index before processing starts."""
    document = scan_episode_sensor_alignment(
        manifest,
        episode,
        force=force,
        reference_media_file_id=reference_media_file_id,
    )
    return validate_episode_time_sync(manifest, episode, document)


def get_valid_sensor_alignment(
    manifest: dict,
    episode: dict,
    reference_media_file_id: str | None = None,
    *,
    force: bool = False,
) -> dict:
    """Return a current, validated T0 document for one reference video."""

    return ensure_episode_time_sync(
        manifest,
        episode,
        force=force,
        reference_media_file_id=reference_media_file_id,
    )


def find_sensor_alignment_stream(
    document: dict,
    relative_path: str,
    *,
    field: str | None = None,
    source_count: int | None = None,
) -> dict | None:
    normalized_path = relative_path.replace("\\", "/").casefold()
    candidates = [
        item for item in document.get("streams") or []
        if str(item.get("relative_path") or "").replace("\\", "/").casefold() == normalized_path
    ]
    if not candidates:
        return None
    if field is not None:
        normalized_field = str(field).replace("\\", "/").casefold()
        exact = [item for item in candidates if str(item.get("field") or "").replace("\\", "/").casefold() == normalized_field]
        if exact:
            candidates = exact
        else:
            grouped = [
                item for item in candidates
                if normalized_field in {
                    str(member).replace("\\", "/").casefold()
                    for member in item.get("members") or []
                }
            ]
            if grouped:
                candidates = grouped
            else:
                return None
    if source_count is not None:
        matching_count = [item for item in candidates if int(item.get("data_count") or 0) == int(source_count)]
        if not matching_count:
            return None
        candidates = matching_count
    if len(candidates) == 1:
        return candidates[0]
    candidates.sort(key=lambda item: (
        str(item.get("kind") or "") == "joint",
        float((item.get("lookup_quality") or {}).get("valid_coverage") or 0.0),
        bool(item.get("field")),
    ), reverse=True)
    return candidates[0] if candidates else None


def aligned_sensor_rows(
    manifest: dict,
    episode: dict,
    relative_path: str,
    source_count: int,
    video_count: int,
    *,
    field: str | None = None,
    reference_media_file_id: str | None = None,
    alignment: dict | None = None,
    require_complete: bool = True,
) -> np.ndarray:
    """Resolve video frames to source rows only through validated T0."""

    document = alignment or get_valid_sensor_alignment(
        manifest,
        episode,
        reference_media_file_id=reference_media_file_id,
    )
    stream = find_sensor_alignment_stream(
        document,
        relative_path,
        field=field,
        source_count=source_count,
    )
    if stream is None:
        label = f"{relative_path}::{field}" if field else relative_path
        raise RuntimeError(f"T0 has no validated mapping for source: {label}")
    lookup = stream.get("frame_to_sensor_index")
    if isinstance(lookup, list):
        rows = np.full(video_count, -1, dtype=np.int64)
        count = min(video_count, len(lookup))
        if count:
            rows[:count] = np.asarray(lookup[:count], dtype=np.int64)
    elif stream.get("mode") in {"prealigned_master_clock", "paired_frame_index"}:
        rows = np.arange(video_count, dtype=np.int64)
    elif stream.get("index_multiplier") is not None:
        rows = np.rint(np.arange(video_count) * float(stream["index_multiplier"])).astype(np.int64)
    else:
        raise RuntimeError(f"T0 mapping has no row rule: {relative_path}")
    invalid = (rows < 0) | (rows >= int(source_count))
    if require_complete and invalid.any():
        raise RuntimeError(
            f"T0 mapping remains incomplete after repair: {relative_path} "
            f"({int(invalid.sum())}/{int(video_count)} frames unavailable)"
        )
    rows[invalid] = -1
    return rows


def retime_sensor_alignment(document: dict, media: dict) -> dict:
    """Project a validated raw-video T0 document onto a derived video timeline."""

    source_positions = np.asarray(media.get("source_frame_positions") or [], dtype=np.float64).reshape(-1)
    target_count = int(media.get("frame_count") or 0)
    if source_positions.size != target_count or target_count <= 0 or not np.isfinite(source_positions).all():
        return document
    output = deepcopy(document)
    for stream in output.get("streams") or []:
        lookup = stream.get("frame_to_sensor_position")
        if not isinstance(lookup, list):
            lookup = stream.get("frame_to_sensor_index")
        if isinstance(lookup, list) and lookup:
            original = np.asarray(lookup, dtype=np.float64)
            original[original < 0.0] = np.nan
            sensor_positions = np.full(target_count, -1.0, dtype=np.float64)
            left = np.floor(source_positions).astype(np.int64)
            right = np.ceil(source_positions).astype(np.int64)
            in_range = (left >= 0) & (right < original.size)
            valid = in_range.copy()
            if valid.any():
                valid_indices = np.flatnonzero(valid)
                valid[valid_indices] &= np.isfinite(original[left[valid_indices]]) & np.isfinite(original[right[valid_indices]])
            alpha = source_positions - left
            sensor_positions[valid] = (
                original[left[valid]] * (1.0 - alpha[valid])
                + original[right[valid]] * alpha[valid]
            )
        else:
            multiplier = float(stream.get("index_multiplier") or 1.0)
            sensor_positions = source_positions * multiplier
            data_count = int(stream.get("data_count") or 0)
            if data_count > 0:
                sensor_positions[(sensor_positions < 0.0) | (sensor_positions > data_count - 1)] = -1.0
        stream["frame_to_sensor_position"] = [round(float(value), 9) for value in sensor_positions]
        stream["frame_to_sensor_index"] = [int(round(value)) if value >= 0.0 else -1 for value in sensor_positions]
        stream["retimed_from_source_video"] = True
    output["reference_video"] = {
        **(output.get("reference_video") or {}),
        "file_id": media.get("file_id"),
        "stream_name": media.get("stream_name"),
        "relative_path": media.get("relative_path"),
        "fps": media.get("fps") or (output.get("reference_video") or {}).get("fps") or 30.0,
        "frame_count": target_count,
        "duration": target_count / max(0.01, float(media.get("fps") or 30.0)),
        "retimed_from_source_video": True,
    }
    fractional_count = int(np.count_nonzero(np.abs(source_positions - np.rint(source_positions)) > 1e-9))
    output["retiming"] = {
        "mode": str(media.get("smoothing_mode") or "derived_video_timeline"),
        "source_frame_count": int(media.get("source_frame_count") or (round(float(source_positions[-1])) + 1)),
        "output_frame_count": target_count,
        "fractional_source_frame_count": fractional_count,
        "source_frame_positions": [round(float(value), 9) for value in source_positions],
    }
    return output


def aligned_sensor_positions(
    manifest: dict,
    episode: dict,
    relative_path: str,
    source_count: int,
    video_count: int,
    *,
    source_frame_positions: list[float] | np.ndarray | None = None,
    field: str | None = None,
    reference_media_file_id: str | None = None,
    alignment: dict | None = None,
    require_complete: bool = True,
) -> np.ndarray:
    """Resolve derived video frames to fractional source rows through validated T0."""

    document = alignment or get_valid_sensor_alignment(
        manifest,
        episode,
        reference_media_file_id=reference_media_file_id,
    )
    if source_frame_positions is not None:
        document = retime_sensor_alignment(document, {
            "file_id": reference_media_file_id,
            "frame_count": video_count,
            "source_frame_positions": np.asarray(source_frame_positions, dtype=np.float64).reshape(-1).tolist(),
        })
    stream = find_sensor_alignment_stream(
        document,
        relative_path,
        field=field,
        source_count=source_count,
    )
    if stream is None:
        label = f"{relative_path}::{field}" if field else relative_path
        raise RuntimeError(f"T0 has no validated mapping for source: {label}")
    lookup = stream.get("frame_to_sensor_position")
    if isinstance(lookup, list):
        positions = np.full(video_count, -1.0, dtype=np.float64)
        count = min(video_count, len(lookup))
        if count:
            positions[:count] = np.asarray(lookup[:count], dtype=np.float64)
    else:
        rows = aligned_sensor_rows(
            manifest,
            episode,
            relative_path,
            source_count,
            video_count,
            field=field,
            reference_media_file_id=reference_media_file_id,
            alignment=document,
            require_complete=require_complete,
        )
        positions = rows.astype(np.float64)
    invalid = ~np.isfinite(positions) | (positions < 0.0) | (positions > int(source_count) - 1)
    if require_complete and invalid.any():
        raise RuntimeError(
            f"T0 fractional mapping remains incomplete: {relative_path} "
            f"({int(invalid.sum())}/{int(video_count)} frames unavailable)"
        )
    positions[invalid] = -1.0
    return positions


def load_sensor_alignment(
    manifest: dict,
    episode_id: str,
    reference_media_file_id: str | None = None,
) -> dict | None:
    path = _existing_sensor_alignment_path(manifest, episode_id, reference_media_file_id)
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if document.get("schema") != SENSOR_ALIGNMENT_SCHEMA:
        return None
    document["artifact_path"] = str(path)
    return document


def map_video_frame_to_sensor(
    manifest: dict,
    episode: dict,
    relative_path: str,
    video_frame: int,
    video_fps: float | None = None,
    alignment: dict | None = None,
    reference_media_file_id: str | None = None,
    field: str | None = None,
) -> tuple[int | None, dict]:
    """Map a video frame to a sensor row, detecting the Episode on demand."""
    document = alignment or get_valid_sensor_alignment(
        manifest,
        episode,
        reference_media_file_id=reference_media_file_id,
    )
    stream = find_sensor_alignment_stream(document, relative_path, field=field)
    if stream is None:
        raise KeyError(relative_path)
    requested = max(0, int(video_frame))
    count = max(0, int(stream.get("data_count") or 0))
    lookup = stream.get("frame_to_sensor_index")
    valid = True
    invalid_reason = None
    if isinstance(lookup, list):
        if requested >= len(lookup) or int(lookup[requested]) < 0:
            sensor_index = None
            valid = False
            partial_frames = {int(item) for item in stream.get("partial_video_frames") or []}
            invalid_reason = "source_partial" if requested in partial_frames else "timestamp_or_range_gap"
        else:
            sensor_index = int(lookup[requested])
    elif stream.get("mode") in {"prealigned_master_clock", "paired_frame_index"}:
        sensor_index = requested
    else:
        stored_hz = _as_positive_float(stream.get("stored_hz"))
        reference_fps = _as_positive_float((document.get("reference_video") or {}).get("fps"))
        requested_fps = _as_positive_float(video_fps) or reference_fps
        multiplier = stored_hz / requested_fps if stored_hz and requested_fps else _as_positive_float(stream.get("index_multiplier"))
        sensor_index = int(round(requested * (multiplier or 1.0)))
    if sensor_index is not None and count and not 0 <= sensor_index < count:
        sensor_index = None
        valid = False
        invalid_reason = "out_of_sensor_range"
    metadata = {
        "video_frame": requested,
        "sensor_index": sensor_index,
        "valid": valid,
        "invalid_reason": invalid_reason,
        "relative_path": stream.get("relative_path"),
        "mode": stream.get("mode"),
        "alignment_multiplier": round(multiplier, 12) if stream.get("mode") == "rate_multiplier" and multiplier else stream.get("index_multiplier"),
        "sensor_hz": stream.get("stored_hz"),
        "physical_hz": stream.get("physical_hz"),
        "artifact_path": document.get("artifact_path") or str(sensor_alignment_path(
            manifest,
            str(episode["id"]),
            (document.get("reference_video") or {}).get("file_id") or reference_media_file_id,
        )),
    }
    return sensor_index, metadata


class SensorAlignmentJobManager(CancellableJobMixin):
    """Run dataset-wide timing scans without blocking the API/UI thread."""

    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="alice-sensor-alignment")
        self._jobs: dict[str, dict] = {}
        self._dataset_jobs: dict[str, str] = {}
        self._lock = threading.RLock()
        self._init_cancellation()

    def submit(
        self,
        dataset_id: str,
        manifest: dict | None = None,
        episode_ids: list[str] | None = None,
        force: bool = False,
    ) -> dict:
        manifest = manifest or get_manifest(dataset_id)
        episodes = list(manifest.get("episodes") or [])
        if episode_ids is not None:
            wanted = set(episode_ids)
            episodes = [episode for episode in episodes if episode.get("id") in wanted]
            missing = wanted - {str(episode.get("id")) for episode in episodes}
            if missing:
                raise KeyError(sorted(missing)[0])
        with self._lock:
            existing_id = self._dataset_jobs.get(dataset_id)
            if existing_id and self._jobs.get(existing_id, {}).get("status") in {"queued", "running", "cancelling"}:
                return dict(self._jobs[existing_id])
            job_id = uuid.uuid4().hex
            job = {
                "id": job_id,
                "kind": "sensor_alignment",
                "dataset_id": dataset_id,
                "status": "queued",
                "progress": 0,
                "message": f"Sensor Hz scan queued: 0/{len(episodes)} Episodes",
                "episode_count": len(episodes),
                "completed_count": 0,
                "stream_count": 0,
                "failure_count": 0,
                "timestamp_aligned_count": 0,
                "rate_multiplier_count": 0,
                "current_episode_id": None,
                "result": None,
                "error": None,
            }
            self._jobs[job_id] = job
            self._register_cancellation(job_id)
            self._dataset_jobs[dataset_id] = job_id
        self._executor.submit(self._run, job_id, dict(manifest), episodes, force)
        return dict(job)

    def get(self, job_id: str) -> dict:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return dict(self._jobs[job_id])

    def status(self, dataset_id: str) -> dict:
        with self._lock:
            job_id = self._dataset_jobs.get(dataset_id)
            if job_id and job_id in self._jobs:
                return dict(self._jobs[job_id])
        return {
            "kind": "sensor_alignment",
            "dataset_id": dataset_id,
            "status": "idle",
            "progress": 0,
            "message": "Sensor Hz scan has not started",
        }

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(changes)

    def _run(self, job_id: str, manifest: dict, episodes: list[dict], force: bool) -> None:
        total = len(episodes)
        stream_count = 0
        multiplier_aligned_count = 0
        timestamp_aligned_count = 0
        rate_multiplier_count = 0
        conflict_count = 0
        prealigned_count = 0
        repaired_stream_count = 0
        warning_stream_count = 0
        failures = []
        artifacts = []
        try:
            self._start_unless_cancelled(job_id, status="running", progress=1 if total else 100, message=f"Detecting sensor Hz: 0/{total} Episodes")
            for position, episode in enumerate(episodes):
                self._raise_if_cancelled(job_id)
                episode_id = str(episode.get("id"))
                self._update(job_id, current_episode_id=episode_id, message=f"Detecting sensor Hz: {position}/{total} · {episode.get('name') or episode_id}")
                try:
                    document = ensure_episode_time_sync(manifest, episode, force=force)
                    streams = list(document.get("streams") or [])
                    stream_count += len(streams)
                    multiplier_aligned_count += sum(abs(float(item.get("index_multiplier") or 1.0) - 1.0) > 1e-6 for item in streams)
                    timestamp_aligned_count += sum(item.get("mode") == "timestamp_nearest" for item in streams)
                    rate_multiplier_count += sum(item.get("mode") == "rate_multiplier" for item in streams)
                    conflict_count += sum(bool(item.get("rate_conflict")) for item in streams)
                    prealigned_count += sum(item.get("mode") in {"prealigned_master_clock", "paired_frame_index"} for item in streams)
                    gate = document.get("gate") or {}
                    repaired_stream_count += int(gate.get("repaired_stream_count") or 0)
                    warning_stream_count += int(gate.get("warning_stream_count") or 0)
                    artifacts.append({
                        "episode_id": episode_id,
                        "stream_count": len(streams),
                        "multiplier_aligned_count": sum(abs(float(item.get("index_multiplier") or 1.0) - 1.0) > 1e-6 for item in streams),
                        "timestamp_aligned_count": sum(item.get("mode") == "timestamp_nearest" for item in streams),
                        "rate_multiplier_count": sum(item.get("mode") == "rate_multiplier" for item in streams),
                        "conflict_count": sum(bool(item.get("rate_conflict")) for item in streams),
                        "quality_status": gate.get("quality_status") or "ready",
                        "repaired_stream_count": int(gate.get("repaired_stream_count") or 0),
                        "warning_stream_count": int(gate.get("warning_stream_count") or 0),
                        "artifact_path": document.get("artifact_path") or str(sensor_alignment_path(manifest, episode_id)),
                    })
                except JobCancelled:
                    raise
                except Exception as exc:
                    failures.append({"episode_id": episode_id, "error": str(exc)})
                self._raise_if_cancelled(job_id)
                completed = position + 1
                self._update(
                    job_id,
                    completed_count=completed,
                    stream_count=stream_count,
                    multiplier_aligned_count=multiplier_aligned_count,
                    timestamp_aligned_count=timestamp_aligned_count,
                    rate_multiplier_count=rate_multiplier_count,
                    conflict_count=conflict_count,
                    repaired_stream_count=repaired_stream_count,
                    warning_stream_count=warning_stream_count,
                    failure_count=len(failures),
                    progress=round(completed / max(total, 1) * 100, 1),
                    message=f"Detecting sensor Hz: {completed}/{total} Episodes",
                )
            result = {
                "dataset_id": manifest.get("id"),
                "episode_count": total,
                "completed_count": total - len(failures),
                "stream_count": stream_count,
                "multiplier_aligned_count": multiplier_aligned_count,
                "timestamp_aligned_count": timestamp_aligned_count,
                "rate_multiplier_count": rate_multiplier_count,
                "conflict_count": conflict_count,
                "prealigned_count": prealigned_count,
                "repaired_stream_count": repaired_stream_count,
                "warning_stream_count": warning_stream_count,
                "failure_count": len(failures),
                "items": artifacts,
                "failures": failures,
            }
            # T0 is a dataset gate. A partial success must not unlock later
            # processing while any Episode lacks a trustworthy time mapping.
            final_status = "failed" if failures else "complete"
            self._update(
                job_id,
                status=final_status,
                progress=100,
                current_episode_id=None,
                message=f"Sensor Hz scan complete: {total - len(failures)}/{total} Episodes",
                result=result,
                error=failures[0]["error"] if final_status == "failed" else None,
            )
        except JobCancelled:
            self._mark_cancelled(job_id)
        except Exception as exc:
            self._update(job_id, status="failed", progress=100, message=str(exc), error=str(exc), trace=traceback.format_exc(limit=8))
        finally:
            self._forget_cancellation(job_id)


sensor_alignment_jobs = SensorAlignmentJobManager()
