from __future__ import annotations

"""Detect and persist per-episode sensor/video timing relationships.

The generated documents are derived indices.  They live below ``.alicePD``
and never modify files in the source dataset.
"""

import json
import math
import threading
import traceback
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .storage import dataset_sidecar_root, get_manifest, slugify, storage_slug
from .job_control import CancellableJobMixin, JobCancelled


SENSOR_ALIGNMENT_SCHEMA = "alice/sensor-alignment/v2"
SENSOR_EXTENSIONS = {".h5", ".hdf5", ".h5df", ".json", ".jsonl", ".parquet"}
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


def sensor_alignment_path(manifest: dict, episode_id: str) -> Path:
    return (
        _sidecar_root(manifest)
        / "indices"
        / "sensor-alignment"
        / f"{storage_slug(episode_id)}.alignment.alice"
    )


def _existing_sensor_alignment_path(manifest: dict, episode_id: str) -> Path:
    preferred = sensor_alignment_path(manifest, episode_id)
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


def _reference_video(episode: dict) -> dict:
    streams = list(episode.get("media_streams") or [])
    primary_id = episode.get("primary_media_file_id")
    selected = next((item for item in streams if item.get("file_id") == primary_id), None)
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
                for key, value in obj.attrs.items():
                    if _normal_key(key) in RATE_KEYS:
                        candidate = _rate_candidate(value, f"hdf5:{name}.attrs.{key}", direct=True)
                        if candidate:
                            rate_candidates.append(candidate)

        handle.visititems(visitor)
    timestamp_entries.sort(key=lambda item: _timestamp_priority(item["summary"]["field"]))
    return {
        "data_count": _representative_count(data_counts) or _representative_count([item["summary"]["count"] for item in timestamp_entries]),
        "rate_candidates": _dedupe_rate_candidates(rate_candidates),
        "timestamps": timestamp_entries,
        "master_clock": master_clock,
        "attributes": attributes,
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
        return {"data_count": count, "rate_candidates": _dedupe_rate_candidates(rate_candidates), "timestamps": entries, "master_clock": None, "attributes": {}}
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
    }


def _probe_file(path: Path, reference_fps: float) -> dict:
    suffix = path.suffix.casefold()
    if suffix in {".h5", ".hdf5", ".h5df"}:
        return _probe_hdf5(path, reference_fps)
    if suffix == ".parquet":
        return _probe_parquet(path, reference_fps)
    if suffix in {".json", ".jsonl"}:
        return _probe_json(path, reference_fps)
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


def _reference_timestamps(root: Path, records: list[dict], reference: dict) -> tuple[np.ndarray | None, dict | None]:
    fps = float(reference["fps"])
    frame_count = int(reference.get("frame_count") or 0)
    parquet_records = [record for record in records if str(record.get("extension") or "").casefold() == ".parquet"]
    parquet_records.sort(key=lambda record: 0 if "sync" in str(record.get("relative_path") or "").casefold() else 1)
    for record in parquet_records:
        relative = str(record.get("relative_path") or "")
        lowered = relative.casefold()
        if "sync" not in lowered and "video_timestamp" not in lowered:
            continue
        try:
            import pyarrow.compute as compute
            import pyarrow.parquet as parquet

            path = root / relative
            schema_names = parquet.ParquetFile(path).schema_arrow.names
            if "master_ts" in schema_names and "frame_idx" in schema_names:
                table = parquet.read_table(path, columns=["frame_idx", "master_ts"])
                frame_indices = np.asarray(table["frame_idx"].to_numpy(), dtype=np.int64)
                raw = np.asarray(table["master_ts"].to_numpy(), dtype=np.float64)
                summary, seconds = _timestamp_analysis(raw, "master_ts", fps)
            elif "frame_idx" in schema_names and ("ts_wall" in schema_names or "pts_us" in schema_names):
                columns = ["frame_idx", "ts_wall" if "ts_wall" in schema_names else "pts_us"]
                if "camera" in schema_names:
                    columns.append("camera")
                table = parquet.read_table(path, columns=columns)
                if "camera" in columns:
                    camera_name = Path(str(reference.get("stream_name") or "")).stem
                    table = table.filter(compute.equal(table["camera"], camera_name))
                frame_indices = np.asarray(table["frame_idx"].to_numpy(), dtype=np.int64)
                field = "ts_wall" if "ts_wall" in columns else "pts_us"
                raw = np.asarray(table[field].to_numpy(), dtype=np.float64)
                summary, seconds = _timestamp_analysis(raw, field, fps)
            else:
                continue
            if summary is None or seconds is None or not frame_indices.size:
                continue
            count = max(frame_count, int(np.max(frame_indices)) + 1)
            timeline = np.full(count, np.nan, dtype=np.float64)
            valid = (frame_indices >= 0) & (frame_indices < count)
            timeline[frame_indices[valid]] = seconds[valid]
            if np.count_nonzero(np.isfinite(timeline)) >= 2:
                return timeline, {"relative_path": relative, "field": summary["field"], "quality": summary}
        except Exception:
            continue
    if frame_count > 0:
        return np.arange(frame_count, dtype=np.float64) / fps, {"relative_path": reference.get("relative_path"), "field": "decoded_frame_index/fps", "quality": None}
    return None, None


def _nearest_timestamp_lookup(reference_seconds: np.ndarray, sensor_seconds: np.ndarray, stored_hz: float | None) -> tuple[list[int] | None, dict]:
    reference_seconds = np.asarray(reference_seconds, dtype=np.float64).reshape(-1)
    sensor_seconds = np.asarray(sensor_seconds, dtype=np.float64).reshape(-1)
    valid_sensor = np.flatnonzero(np.isfinite(sensor_seconds))
    if valid_sensor.size < 2:
        return None, {"invalid_frame_count": int(reference_seconds.size), "reason": "insufficient_sensor_timestamps"}
    values = sensor_seconds[valid_sensor]
    valid_reference = reference_seconds[np.isfinite(reference_seconds)]
    if valid_reference.size:
        reference_origin = float(valid_reference[0])
        sensor_origin = float(values[0])
        # Preserve offsets when both streams use the same kind of clock.  Only
        # remove independent origins when one side is an epoch-like wall clock
        # and the other is a relative PTS/device timeline.
        reference_is_absolute = abs(reference_origin) >= 1_000_000.0
        sensor_is_absolute = abs(sensor_origin) >= 1_000_000.0
        origins_normalized = reference_is_absolute != sensor_is_absolute
        if origins_normalized:
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
    return mapping, {
        "invalid_frame_count": invalid,
        "long_gap_rejection_count": long_gap_rejections,
        "max_match_distance_seconds": round(max_distance, 9),
        "origins_normalized": origins_normalized,
    }


def _select_physical_hz(candidates: list[dict]) -> tuple[float | None, bool]:
    if not candidates:
        return None, False
    direct = [item for item in candidates if item.get("direct")]
    selected = float((direct or candidates)[0]["hz"])
    distinct = {round(float(item["hz"]), 6) for item in candidates}
    return selected, len(distinct) > 1


def _build_stream(
    relative_path: str,
    probe: dict,
    metadata_hints: list[dict],
    reference: dict,
    reference_seconds: np.ndarray | None,
) -> dict | None:
    data_count = int(probe.get("data_count") or 0)
    if data_count <= 1:
        return None
    rate_candidates = _dedupe_rate_candidates([*(probe.get("rate_candidates") or []), *metadata_hints])
    physical_hz, rate_conflict = _select_physical_hz(rate_candidates)
    timestamp_entries = list(probe.get("timestamps") or [])
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
    if count_matches and (master_clock or timestamp_matches) and not timestamp_discontinuous:
        mode = "prealigned_master_clock"
        stored_hz = aligned_hz or video_fps
        multiplier = 1.0
    elif count_matches and aligned_entry is None:
        mode = "paired_frame_index"
        stored_hz = video_fps
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
    timestamp_summaries = [item["summary"] for item in timestamp_entries]
    return {
        "relative_path": relative_path.replace("\\", "/"),
        "kind": "sensor",
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
        "mapping_rule": (
            "sensor_index = video_frame_index"
            if mode in {"prealigned_master_clock", "paired_frame_index"}
            else "nearest sensor timestamp; -1 marks gaps that must not be interpolated"
            if mode == "timestamp_nearest"
            else "sensor_index = round(video_frame_index * index_multiplier)"
        ),
    }


def scan_episode_sensor_alignment(manifest: dict, episode: dict, force: bool = False) -> dict:
    """Inspect one Episode and persist its timing index under ``.alicePD``."""
    root = Path(manifest["root_path"]).expanduser().resolve()
    records = _episode_records(manifest, episode)
    reference = _reference_video(episode)
    source_signature = _episode_signature(root, records, reference)
    artifact_path = sensor_alignment_path(manifest, str(episode["id"]))
    existing_path = _existing_sensor_alignment_path(manifest, str(episode["id"]))
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
            stream = _build_stream(relative, probe, metadata_hints.get(relative.replace("\\", "/"), []), reference, reference_seconds)
            if stream:
                stream["source_signature"] = _source_signature(path)
                streams.append(stream)
            else:
                skipped_files.append({"relative_path": relative, "reason": "no_time_series"})
        except Exception as exc:
            skipped_files.append({"relative_path": relative, "reason": "inspection_failed", "error": str(exc)[:240]})
    document = {
        "schema": SENSOR_ALIGNMENT_SCHEMA,
        "dataset_id": manifest["id"],
        "episode_id": episode["id"],
        "episode_name": episode.get("name"),
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


def load_sensor_alignment(manifest: dict, episode_id: str) -> dict | None:
    path = _existing_sensor_alignment_path(manifest, episode_id)
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
) -> tuple[int | None, dict]:
    """Map a video frame to a sensor row, detecting the Episode on demand."""
    document = alignment or load_sensor_alignment(manifest, str(episode["id"]))
    if document is None:
        document = scan_episode_sensor_alignment(manifest, episode)
    normalized = relative_path.replace("\\", "/").casefold()
    stream = next((item for item in document.get("streams", []) if str(item.get("relative_path") or "").replace("\\", "/").casefold() == normalized), None)
    if stream is None:
        raise KeyError(relative_path)
    requested = max(0, int(video_frame))
    count = max(0, int(stream.get("data_count") or 0))
    lookup = stream.get("frame_to_sensor_index")
    valid = True
    if isinstance(lookup, list):
        if requested >= len(lookup) or int(lookup[requested]) < 0:
            sensor_index = None
            valid = False
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
    metadata = {
        "video_frame": requested,
        "sensor_index": sensor_index,
        "valid": valid,
        "relative_path": stream.get("relative_path"),
        "mode": stream.get("mode"),
        "alignment_multiplier": round(multiplier, 12) if stream.get("mode") == "rate_multiplier" and multiplier else stream.get("index_multiplier"),
        "sensor_hz": stream.get("stored_hz"),
        "physical_hz": stream.get("physical_hz"),
        "artifact_path": document.get("artifact_path") or str(sensor_alignment_path(manifest, str(episode["id"]))),
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
        failures = []
        artifacts = []
        try:
            self._start_unless_cancelled(job_id, status="running", progress=1 if total else 100, message=f"Detecting sensor Hz: 0/{total} Episodes")
            for position, episode in enumerate(episodes):
                self._raise_if_cancelled(job_id)
                episode_id = str(episode.get("id"))
                self._update(job_id, current_episode_id=episode_id, message=f"Detecting sensor Hz: {position}/{total} · {episode.get('name') or episode_id}")
                try:
                    document = scan_episode_sensor_alignment(manifest, episode, force=force)
                    streams = list(document.get("streams") or [])
                    stream_count += len(streams)
                    multiplier_aligned_count += sum(abs(float(item.get("index_multiplier") or 1.0) - 1.0) > 1e-6 for item in streams)
                    timestamp_aligned_count += sum(item.get("mode") == "timestamp_nearest" for item in streams)
                    rate_multiplier_count += sum(item.get("mode") == "rate_multiplier" for item in streams)
                    conflict_count += sum(bool(item.get("rate_conflict")) for item in streams)
                    prealigned_count += sum(item.get("mode") in {"prealigned_master_clock", "paired_frame_index"} for item in streams)
                    artifacts.append({
                        "episode_id": episode_id,
                        "stream_count": len(streams),
                        "multiplier_aligned_count": sum(abs(float(item.get("index_multiplier") or 1.0) - 1.0) > 1e-6 for item in streams),
                        "timestamp_aligned_count": sum(item.get("mode") == "timestamp_nearest" for item in streams),
                        "rate_multiplier_count": sum(item.get("mode") == "rate_multiplier" for item in streams),
                        "conflict_count": sum(bool(item.get("rate_conflict")) for item in streams),
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
                "failure_count": len(failures),
                "items": artifacts,
                "failures": failures,
            }
            final_status = "failed" if failures and len(failures) == total and total else "complete"
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
