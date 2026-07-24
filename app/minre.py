from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import h5py
import numpy as np

from .behavior_annotator import annotate_episode_behavior, load_behavior_annotation
from .curation_pipeline import load_curation_report, run_episode_curation
from .models import registry
from .schemas import BehaviorAnnotationRequest, CurationJobRequest, VLMModelConfig
from .sensor_alignment import load_sensor_alignment, scan_episode_sensor_alignment
from .storage import ROOT, episode_media, scan_dataset, slugify, storage_slug
from .video_smoothing import smooth_video


MINRE_STATE_SCHEMA = "alice/minre-state/v1"
MINRE_DATASET_SCHEMA = "alice/minre-dataset/v1"
MINRE_PAIR_SCHEMA = "alice/minre-pair/v1"
DEFAULT_DROP_PHASES = ("idle", "observe", "reach", "withdraw", "unknown")
HDF5_EXTENSIONS = {".h5", ".hdf5", ".h5df"}
MAX_STATIC_DATASET_BYTES = 16 * 1024 * 1024


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


def _file_signature(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _manifest_fingerprint(manifest: dict) -> str:
    rows = [
        f"{item.get('relative_path')}\0{int(item.get('size_bytes') or 0)}\0{item.get('modified_at')}"
        for item in manifest.get("files", [])
    ]
    return hashlib.sha256("\n".join(sorted(rows)).encode("utf-8")).hexdigest()


def _phase_key(value: Any) -> str:
    text = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    return re.sub(r"_+", "_", text).strip("_")


def _category_name(value: Any) -> str:
    text = str(value or "other").strip()
    return slugify(text.casefold().replace(" ", "_")) or "other"


def _episode_records(manifest: dict, episode: dict) -> list[dict]:
    episode_id = str(episode.get("id") or "")
    episode_key = str(episode.get("episode_key") or "")
    return [
        item
        for item in manifest.get("files", [])
        if str(item.get("episode_id") or "") == episode_id
        or (not item.get("episode_id") and str(item.get("episode_key") or "") == episode_key)
    ]


def _progress_printer(prefix: str, quiet: bool) -> Callable[[float, str], None]:
    last_bucket = -1

    def progress(value: float, message: str) -> None:
        nonlocal last_bucket
        bucket = int(max(0.0, min(100.0, float(value))) // 5)
        if quiet or (bucket == last_bucket and value < 100):
            return
        last_bucket = bucket
        print(f"[{prefix}] {min(100, max(0, int(round(value)))):3d}% {message}", flush=True)

    return progress


def _video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return 0
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    return count


def _mapped_rows(stream: dict | None, frame_indices: np.ndarray, video_fps: float) -> np.ndarray | None:
    if stream is None:
        return None
    count = max(0, int(stream.get("data_count") or 0))
    lookup = stream.get("frame_to_sensor_index")
    if isinstance(lookup, list):
        rows = np.full(frame_indices.shape, -1, dtype=np.int64)
        valid = frame_indices < len(lookup)
        if valid.any():
            rows[valid] = np.asarray([int(lookup[int(index)]) for index in frame_indices[valid]], dtype=np.int64)
    elif stream.get("mode") in {"prealigned_master_clock", "paired_frame_index"}:
        rows = frame_indices.astype(np.int64, copy=True)
    else:
        multiplier = stream.get("index_multiplier")
        if multiplier is None:
            stored_hz = float(stream.get("stored_hz") or 0.0)
            multiplier = stored_hz / max(0.01, video_fps) if stored_hz else 1.0
        rows = np.rint(frame_indices * float(multiplier)).astype(np.int64)
    if count:
        rows[(rows < 0) | (rows >= count)] = -1
    return rows


def _hdf5_timeline_counts(path: Path) -> set[int]:
    counts: set[int] = set()
    with h5py.File(path, "r") as source:
        def visitor(_: str, obj: h5py.Dataset | h5py.Group) -> None:
            if isinstance(obj, h5py.Dataset) and obj.ndim and int(obj.shape[0]) > 1:
                counts.add(int(obj.shape[0]))

        source.visititems(visitor)
    return counts


def _alignment_plan(
    manifest: dict,
    episode: dict,
    alignment: dict,
    frame_count: int,
    fps: float,
) -> tuple[np.ndarray, dict[str, dict]]:
    valid = np.ones(frame_count, dtype=bool)
    plans: dict[str, dict] = {}
    all_frames = np.arange(frame_count, dtype=np.int64)
    root = Path(manifest["root_path"]).expanduser().resolve()
    streams = {
        str(item.get("relative_path") or "").replace("\\", "/").casefold(): item
        for item in alignment.get("streams", [])
    }
    for record in _episode_records(manifest, episode):
        relative = str(record.get("relative_path") or "").replace("\\", "/")
        if Path(relative).suffix.casefold() not in HDF5_EXTENSIONS:
            continue
        path = (root / relative).resolve()
        stream = streams.get(relative.casefold())
        rows = _mapped_rows(stream, all_frames, fps)
        if rows is None:
            counts = _hdf5_timeline_counts(path)
            if frame_count in counts:
                rows = all_frames.copy()
                stream = {
                    "relative_path": relative,
                    "data_count": frame_count,
                    "mode": "paired_frame_index",
                    "mapping_rule": "sensor_index = video_frame_index",
                }
            elif counts:
                raise RuntimeError(f"HDF5 cannot be aligned reliably: {relative}")
            else:
                rows = np.full(frame_count, -1, dtype=np.int64)
                stream = {"relative_path": relative, "data_count": 0, "mode": "static_only"}
        if int(stream.get("data_count") or 0) > 1:
            valid &= rows >= 0
        plans[relative.casefold()] = {"record": record, "path": path, "stream": stream, "rows": rows}
    return valid, plans


def _clean_intervals(
    frame_count: int,
    curation: dict,
    behavior: dict,
    alignment_valid: np.ndarray,
    drop_phases: set[str],
    minimum_frames: int,
) -> tuple[list[tuple[int, int]], dict]:
    keep = np.zeros(frame_count, dtype=bool)
    for segment in curation.get("segments", []):
        if str(segment.get("state") or "") != "valid":
            continue
        start = max(0, min(frame_count - 1, int(segment.get("start_frame") or 0)))
        end = max(start, min(frame_count - 1, int(segment.get("end_frame") or start)))
        keep[start:end + 1] = True
    phase_removals: dict[str, int] = {}
    for segment in behavior.get("segments", []):
        phase = _phase_key(segment.get("phase_label") or segment.get("label"))
        if phase not in drop_phases:
            continue
        start = max(0, min(frame_count - 1, int(segment.get("start_frame") or 0)))
        end = max(start, min(frame_count - 1, int(segment.get("end_frame") or start)))
        phase_removals[phase] = phase_removals.get(phase, 0) + int(keep[start:end + 1].sum())
        keep[start:end + 1] = False
    keep &= alignment_valid
    padded = np.concatenate(([False], keep, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    intervals = [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if int(end) - int(start) + 1 >= minimum_frames
    ]
    retained = sum(end - start + 1 for start, end in intervals)
    return intervals, {
        "source_frame_count": frame_count,
        "retained_frame_count": retained,
        "discarded_frame_count": frame_count - retained,
        "alignment_invalid_frame_count": int((~alignment_valid).sum()),
        "phase_removed_frame_count": int(sum(phase_removals.values())),
        "phase_removals": phase_removals,
        "minimum_clip_frames": minimum_frames,
    }


def _phase_labels_for_frames(behavior: dict, frame_indices: np.ndarray) -> list[str]:
    labels = np.full(frame_indices.shape, "unknown", dtype=object)
    positions = {int(frame): index for index, frame in enumerate(frame_indices.tolist())}
    for segment in behavior.get("segments", []):
        start = int(segment.get("start_frame") or 0)
        end = int(segment.get("end_frame") or start)
        label = _phase_key(segment.get("phase_label") or segment.get("label")) or "unknown"
        for frame in range(max(start, int(frame_indices[0])), min(end, int(frame_indices[-1])) + 1):
            position = positions.get(frame)
            if position is not None:
                labels[position] = label
    return [str(value) for value in labels.tolist()]


def _copy_attributes(source: h5py.AttributeManager, target: h5py.AttributeManager) -> None:
    for key, value in source.items():
        try:
            target[key] = value
        except (TypeError, ValueError):
            target[key] = str(value)


def _dataset_bytes(dataset: h5py.Dataset) -> int:
    try:
        return int(dataset.size) * int(dataset.dtype.itemsize)
    except (TypeError, ValueError):
        return MAX_STATIC_DATASET_BYTES + 1


def _create_indexed_dataset(target: h5py.Group, name: str, source: h5py.Dataset, rows: np.ndarray) -> h5py.Dataset:
    shape = (int(rows.size), *source.shape[1:])
    options: dict[str, Any] = {}
    if shape and all(int(value) > 0 for value in shape):
        options["chunks"] = True
        if source.compression:
            options["compression"] = source.compression
            if source.compression_opts is not None:
                options["compression_opts"] = source.compression_opts
    try:
        output = target.create_dataset(name, shape=shape, dtype=source.dtype, **options)
    except (TypeError, ValueError):
        output = target.create_dataset(name, shape=shape, dtype=source.dtype, chunks=True)
    row_width = max(1, int(np.prod(source.shape[1:] or (1,))) * max(1, int(source.dtype.itemsize)))
    batch_size = max(1, min(512, (64 * 1024 * 1024) // row_width))
    for offset in range(0, int(rows.size), batch_size):
        selected = rows[offset:offset + batch_size]
        unique, inverse = np.unique(selected, return_inverse=True)
        values = source[unique.tolist()]
        output[offset:offset + selected.size] = values[inverse]
    _copy_attributes(source.attrs, output.attrs)
    return output


def _copy_hdf5_source(
    source_path: Path,
    target_root: h5py.Group,
    relative_path: str,
    source_rows: np.ndarray,
    source_data_count: int,
    video_frames: np.ndarray,
    source_video_frame_count: int,
) -> list[str]:
    skipped: list[str] = []
    key = f"{storage_slug(Path(relative_path).stem)}-{hashlib.sha1(relative_path.encode('utf-8')).hexdigest()[:8]}"
    file_root = target_root.require_group(key)
    file_root.attrs["source_relative_path"] = relative_path
    with h5py.File(source_path, "r") as source:
        _copy_attributes(source.attrs, file_root.attrs)

        def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            if not name:
                return
            parent_name, _, leaf = name.rpartition("/")
            parent = file_root.require_group(parent_name) if parent_name else file_root
            if isinstance(obj, h5py.Group):
                group = parent.require_group(leaf)
                _copy_attributes(obj.attrs, group.attrs)
                return
            if obj.ndim and source_data_count > 1 and int(obj.shape[0]) == source_data_count:
                _create_indexed_dataset(parent, leaf, obj, source_rows)
                return
            if obj.ndim and int(obj.shape[0]) == source_video_frame_count:
                _create_indexed_dataset(parent, leaf, obj, video_frames)
                return
            if obj.ndim == 0 or _dataset_bytes(obj) <= MAX_STATIC_DATASET_BYTES:
                source.copy(obj, parent, name=leaf)
                return
            skipped.append(name)

        source.visititems(visitor)
    return skipped


def _write_video_clip(source: Path, target: Path, start: int, end: int) -> dict:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot decode smoothed video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid video geometry: {source}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create video clip: {target}")
    written = 0
    try:
        for _ in range(start, end + 1):
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        capture.release()
        writer.release()
    expected = end - start + 1
    if written != expected:
        raise RuntimeError(f"Video clip stopped at {written}/{expected} frames")
    return {"frame_count": written, "fps": fps, "width": width, "height": height}


def _write_pair(
    manifest: dict,
    episode: dict,
    analysis_media: dict,
    behavior: dict,
    alignment_plans: dict[str, dict],
    category_root: Path,
    start: int,
    end: int,
) -> dict:
    frame_indices = np.arange(start, end + 1, dtype=np.int64)
    stem = f"{storage_slug(str(episode.get('name') or episode['id']))}-{start:08d}-{end:08d}"
    video_target = category_root / f"{stem}.mp4"
    hdf5_target = category_root / f"{stem}.h5"
    video_temporary = category_root / f".{stem}.{uuid.uuid4().hex}.part.mp4"
    hdf5_temporary = category_root / f".{stem}.{uuid.uuid4().hex}.part.h5"
    category_root.mkdir(parents=True, exist_ok=True)
    try:
        video_info = _write_video_clip(Path(str(analysis_media["path"])), video_temporary, start, end)
        skipped: dict[str, list[str]] = {}
        with h5py.File(hdf5_temporary, "w") as output:
            output.attrs["schema"] = MINRE_PAIR_SCHEMA
            output.attrs["dataset_id"] = str(manifest["id"])
            output.attrs["episode_id"] = str(episode["id"])
            output.attrs["episode_name"] = str(episode.get("name") or episode["id"])
            output.attrs["task_label"] = str(behavior.get("task_label") or "other")
            output.attrs["created_at"] = _utc_now()
            output.attrs["source_video"] = str(analysis_media.get("relative_path") or analysis_media.get("path") or "")
            metadata = output.require_group("minre")
            metadata.create_dataset("source_frame_index", data=frame_indices)
            metadata.create_dataset("timestamp", data=frame_indices.astype(np.float64) / max(0.01, float(video_info["fps"])))
            string_dtype = h5py.string_dtype(encoding="utf-8")
            metadata.create_dataset(
                "phase_label",
                data=np.asarray(_phase_labels_for_frames(behavior, frame_indices), dtype=object),
                dtype=string_dtype,
            )
            sources = output.require_group("sources")
            for plan in alignment_plans.values():
                relative = str(plan["record"].get("relative_path") or "").replace("\\", "/")
                rows = np.asarray(plan["rows"], dtype=np.int64)[frame_indices]
                if int(plan["stream"].get("data_count") or 0) > 1 and (rows < 0).any():
                    raise RuntimeError(f"Clip contains an unaligned HDF5 frame: {relative}")
                skipped_names = _copy_hdf5_source(
                    plan["path"],
                    sources,
                    relative,
                    rows,
                    int(plan["stream"].get("data_count") or 0),
                    frame_indices,
                    int(analysis_media.get("frame_count") or episode.get("frame_count") or 0),
                )
                if skipped_names:
                    skipped[relative] = skipped_names
            metadata.attrs["skipped_unaligned_datasets"] = json.dumps(skipped, ensure_ascii=False)
        verified_video_frames = _video_frame_count(video_temporary)
        with h5py.File(hdf5_temporary, "r") as output:
            verified_hdf5_frames = int(output["minre/source_frame_index"].shape[0])
            verified_schema = str(output.attrs.get("schema") or "")
        expected = int(frame_indices.size)
        if verified_schema != MINRE_PAIR_SCHEMA or verified_video_frames != expected or verified_hdf5_frames != expected:
            raise RuntimeError(
                f"Pair verification failed: video={verified_video_frames}, hdf5={verified_hdf5_frames}, expected={expected}"
            )
        video_temporary.replace(video_target)
        hdf5_temporary.replace(hdf5_target)
        return {
            "id": stem,
            "episode_id": episode["id"],
            "task_label": str(behavior.get("task_label") or "other"),
            "start_frame": start,
            "end_frame": end,
            "frame_count": expected,
            "fps": round(float(video_info["fps"]), 6),
            "mp4": str(video_target),
            "hdf5": str(hdf5_target),
            "skipped_unaligned_datasets": skipped,
        }
    finally:
        video_temporary.unlink(missing_ok=True)
        hdf5_temporary.unlink(missing_ok=True)


def _verify_pair(pair: dict) -> bool:
    video = Path(str(pair.get("mp4") or ""))
    hdf5 = Path(str(pair.get("hdf5") or ""))
    expected = int(pair.get("frame_count") or 0)
    if expected <= 0 or not video.is_file() or not hdf5.is_file() or _video_frame_count(video) != expected:
        return False
    try:
        with h5py.File(hdf5, "r") as source:
            return (
                str(source.attrs.get("schema") or "") == MINRE_PAIR_SCHEMA
                and int(source["minre/source_frame_index"].shape[0]) == expected
            )
    except (OSError, KeyError):
        return False


def _load_vlm_config(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Qwen config not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Qwen config is unreadable: {path}") from exc
    payload["verify"] = False
    registry.configure_vlm(VLMModelConfig.model_validate(payload))
    return registry.status().get("vlm") or {}


@dataclass(frozen=True)
class MinREOptions:
    source: Path
    output: Path
    vlm_config: Path
    sample_count: int = 18
    drop_phases: tuple[str, ...] = DEFAULT_DROP_PHASES
    min_clip_seconds: float = 0.3
    force_vlm: bool = False
    fail_fast: bool = False
    index_only: bool = False
    quiet: bool = False


class MinREPipeline:
    def __init__(self, options: MinREOptions) -> None:
        self.options = options
        self.state_path = options.output / "minre-state.json"
        self.index_path = options.output / "dataset.json"
        self.state: dict = {}

    def _save(self) -> None:
        self.state["updated_at"] = _utc_now()
        _write_json_atomic(self.state_path, self.state)
        self._write_index()

    def _write_index(self) -> None:
        episodes = self.state.get("episodes") or {}
        pairs = [pair for item in episodes.values() for pair in item.get("pairs", [])]
        categories: dict[str, int] = {}
        for pair in pairs:
            label = str(pair.get("task_label") or "other")
            categories[label] = categories.get(label, 0) + 1
        failures = [
            {"episode_id": episode_id, "episode_name": item.get("episode_name"), "error": item.get("error")}
            for episode_id, item in episodes.items()
            if item.get("status") == "failed"
        ]
        payload = {
            "schema": MINRE_DATASET_SCHEMA,
            "source_root": self.state.get("source_root"),
            "source_dataset_id": self.state.get("dataset_id"),
            "created_at": self.state.get("created_at"),
            "updated_at": self.state.get("updated_at"),
            "status": self.state.get("status"),
            "pipeline": [
                "index", "video_smoothing", "vlm_behavior", "eight_stage_curation",
                "drop_useless_vlm_phases", "aligned_cut", "vlm_classification", "paired_mp4_hdf5",
            ],
            "drop_phases": list(self.options.drop_phases),
            "sample_count": self.options.sample_count,
            "min_clip_seconds": self.options.min_clip_seconds,
            "episode_count": int(self.state.get("episode_count") or 0),
            "completed_episode_count": sum(item.get("status") == "complete" for item in episodes.values()),
            "failed_episode_count": len(failures),
            "pair_count": len(pairs),
            "retained_frame_count": sum(int(item.get("frame_count") or 0) for item in pairs),
            "categories": categories,
            "pairs": pairs,
            "failures": failures,
        }
        _write_json_atomic(self.index_path, payload)

    def _initial_state(self, manifest: dict, fingerprint: str) -> dict:
        return {
            "schema": MINRE_STATE_SCHEMA,
            "status": "indexed" if self.options.index_only else "running",
            "source_root": str(self.options.source),
            "output_root": str(self.options.output),
            "dataset_id": manifest["id"],
            "dataset_fingerprint": fingerprint,
            "episode_count": len(manifest.get("episodes", [])),
            "drop_phases": list(self.options.drop_phases),
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "episodes": {},
        }

    def _index(self) -> dict:
        self.options.output.mkdir(parents=True, exist_ok=True)
        prior = None
        if self.state_path.is_file():
            try:
                prior = json.loads(self.state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                prior = None
        digest = hashlib.sha1(str(self.options.source).casefold().encode("utf-8")).hexdigest()[:8]
        dataset_id = str((prior or {}).get("dataset_id") or f"minre-{slugify(self.options.source.name)}-{digest}")
        if not self.options.quiet:
            print(f"[minRE] indexing {self.options.source}", flush=True)
        manifest = scan_dataset(self.options.source, self.options.source.name, dataset_id=dataset_id)
        fingerprint = _manifest_fingerprint(manifest)
        if prior:
            if str(prior.get("source_root") or "").casefold() != str(self.options.source).casefold():
                raise RuntimeError("Existing minRE state belongs to a different source dataset")
            if prior.get("dataset_fingerprint") != fingerprint:
                raise RuntimeError("Source dataset changed after minRE started; use a new output directory")
            if tuple(prior.get("drop_phases") or ()) != tuple(self.options.drop_phases):
                raise RuntimeError("--drop-phases changed after minRE started; use a new output directory")
            if int(prior.get("sample_count") or self.options.sample_count) != self.options.sample_count:
                raise RuntimeError("--sample-count changed after minRE started; use a new output directory")
            if not math.isclose(
                float(prior.get("min_clip_seconds") or self.options.min_clip_seconds),
                self.options.min_clip_seconds,
            ):
                raise RuntimeError("--min-clip-seconds changed after minRE started; use a new output directory")
            self.state = prior
            self.state["episode_count"] = len(manifest.get("episodes", []))
            self.state["status"] = "indexed" if self.options.index_only else "running"
        else:
            self.state = self._initial_state(manifest, fingerprint)
        self._save()
        return manifest

    def _analysis_media(self, media: dict, smoothing: dict) -> dict:
        summary = smoothing.get("summary") or {}
        output = Path(str(smoothing.get("output_video") or ""))
        if not output.is_file():
            raise RuntimeError("Smoothed video is missing")
        expected_frames = int(summary.get("frame_count") or media.get("frame_count") or 0)
        if expected_frames <= 0 or _video_frame_count(output) != expected_frames:
            raise RuntimeError("Smoothed video failed frame-count verification")
        return {
            **media,
            "path": str(output),
            "frame_count": expected_frames,
            "fps": float(summary.get("fps") or media.get("fps") or 30.0),
            "width": int(summary.get("width") or media.get("width") or 0),
            "height": int(summary.get("height") or media.get("height") or 0),
        }

    def _process_episode(self, manifest: dict, episode: dict, position: int, total: int) -> None:
        episode_id = str(episode["id"])
        entry = self.state.setdefault("episodes", {}).setdefault(episode_id, {
            "episode_id": episode_id,
            "episode_name": episode.get("name"),
            "status": "pending",
            "stage": "pending",
            "pairs": [],
        })
        if entry.get("status") == "complete" and all(_verify_pair(pair) for pair in entry.get("pairs", [])):
            if not self.options.quiet:
                print(f"[minRE] {position}/{total} {episode.get('name')} already complete", flush=True)
            return
        media = episode_media(episode, episode.get("primary_media_file_id"))
        source_signature = _file_signature(Path(str(media["path"])))
        previous_signature = entry.get("source_video_signature")
        if previous_signature is not None and previous_signature != source_signature:
            raise RuntimeError("Primary video changed after this Episode started")
        entry.update({"status": "running", "source_video_signature": source_signature, "error": None})
        self._save()
        label = f"{position}/{total} {episode.get('name')}"

        smoothing = entry.get("smoothing") or {}
        smoothing_path = Path(str(smoothing.get("output_video") or ""))
        smoothing_signature = smoothing.get("output_signature")
        smoothing_reusable = smoothing_path.is_file()
        if smoothing_reusable and smoothing_signature is not None:
            smoothing_reusable = smoothing_signature == _file_signature(smoothing_path)
        if not smoothing_reusable:
            entry["stage"] = "video_smoothing"
            self._save()
            smoothing = smooth_video(
                manifest["id"], episode, media,
                _progress_printer(f"{label} smooth", self.options.quiet),
            )
            entry["smoothing"] = {
                "artifact_path": smoothing.get("artifact_path"),
                "output_video": smoothing.get("output_video"),
                "summary": smoothing.get("summary") or {},
                "output_signature": _file_signature(Path(str(smoothing.get("output_video") or ""))),
            }
            entry.pop("behavior_artifact", None)
            entry.pop("task_label", None)
            entry.pop("curation_artifact", None)
            entry.pop("curation_summary", None)
            entry.pop("curation_stages", None)
            entry["pairs"] = []
            self._save()
        elif smoothing_signature is None:
            entry["smoothing"]["output_signature"] = _file_signature(smoothing_path)
            self._save()
        analysis_media = self._analysis_media(media, entry["smoothing"])

        behavior = None
        if entry.get("behavior_artifact") and not self.options.force_vlm:
            behavior = load_behavior_annotation(manifest["id"], episode_id)
        if behavior is None:
            entry["stage"] = "vlm_behavior"
            self._save()
            behavior = annotate_episode_behavior(
                manifest["id"],
                manifest,
                episode,
                BehaviorAnnotationRequest(sample_count=self.options.sample_count, force=True),
                _progress_printer(f"{label} VLM", self.options.quiet),
                analysis_media_override=analysis_media,
                analysis_source_kind="minre_video_smoothing",
            )
            entry["behavior_artifact"] = str((behavior.get("artifacts") or {}).get("behavior") or "")
            entry["task_label"] = behavior.get("task_label") or "other"
            entry.pop("curation_artifact", None)
            entry.pop("curation_summary", None)
            entry.pop("curation_stages", None)
            entry["pairs"] = []
            self._save()

        curation = None
        report_path = Path(str(entry.get("curation_artifact") or ""))
        if report_path.is_file():
            curation = load_curation_report(manifest["id"], episode_id)
        if curation is None:
            entry["stage"] = "eight_stage_curation"
            self._save()
            request = CurationJobRequest(episode_ids=[episode_id], media_file_ids={episode_id: str(media.get("file_id") or "")})
            curation = run_episode_curation(
                manifest["id"], manifest, episode, analysis_media, request,
                _progress_printer(f"{label} clean", self.options.quiet),
            )
            entry["curation_artifact"] = str(curation.get("artifact_path") or "")
            entry["curation_summary"] = curation.get("summary") or {}
            entry["curation_stages"] = curation.get("stages") or []
            self._save()

        entry["stage"] = "aligned_cut"
        self._save()
        alignment = load_sensor_alignment(manifest, episode_id) or scan_episode_sensor_alignment(manifest, episode)
        frame_count = int(analysis_media.get("frame_count") or episode.get("frame_count") or 0)
        fps = max(0.01, float(analysis_media.get("fps") or episode.get("fps") or 30.0))
        alignment_valid, plans = _alignment_plan(manifest, episode, alignment, frame_count, fps)
        intervals, filtering = _clean_intervals(
            frame_count,
            curation,
            behavior,
            alignment_valid,
            {_phase_key(value) for value in self.options.drop_phases},
            max(1, int(math.ceil(self.options.min_clip_seconds * fps))),
        )
        entry["filtering"] = filtering
        category = _category_name(behavior.get("task_label"))
        category_root = self.options.output / category
        existing_pairs = {
            (int(pair.get("start_frame") or -1), int(pair.get("end_frame") or -1)): pair
            for pair in entry.get("pairs", [])
            if _verify_pair(pair)
        }
        pairs: list[dict] = []
        for clip_index, (start, end) in enumerate(intervals, start=1):
            existing = existing_pairs.get((start, end))
            if existing:
                pairs.append(existing)
                continue
            if not self.options.quiet:
                print(f"[minRE] {label} clip {clip_index}/{len(intervals)} frames {start}-{end}", flush=True)
            pair = _write_pair(
                manifest, episode, analysis_media, behavior, plans,
                category_root, start, end,
            )
            pairs.append(pair)
            entry["pairs"] = pairs
            self._save()
        entry.update({
            "status": "complete",
            "stage": "complete",
            "pairs": pairs,
            "pair_count": len(pairs),
            "completed_at": _utc_now(),
        })
        self._save()

    def run(self) -> dict:
        source = self.options.source
        output = self.options.output
        if not source.is_dir():
            raise RuntimeError(f"Dataset directory does not exist: {source}")
        try:
            output.relative_to(source)
        except ValueError:
            pass
        else:
            raise RuntimeError("minRE output must be outside the source dataset directory")
        manifest = self._index()
        if self.options.index_only:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        vlm = _load_vlm_config(self.options.vlm_config)
        self.state["vlm_model"] = vlm.get("model")
        self._save()
        episodes = list(manifest.get("episodes", []))
        for position, episode in enumerate(episodes, start=1):
            try:
                self._process_episode(manifest, episode, position, len(episodes))
            except Exception as exc:
                entry = self.state.setdefault("episodes", {}).setdefault(str(episode["id"]), {})
                entry.update({
                    "episode_id": episode["id"],
                    "episode_name": episode.get("name"),
                    "status": "failed",
                    "error": str(exc),
                    "trace": traceback.format_exc(limit=12),
                })
                self._save()
                print(f"[minRE] {position}/{len(episodes)} {episode.get('name')} failed: {exc}", file=sys.stderr, flush=True)
                if self.options.fail_fast:
                    break
        failures = [item for item in self.state.get("episodes", {}).values() if item.get("status") == "failed"]
        self.state["status"] = "complete_with_errors" if failures else "complete"
        self.state["completed_at"] = _utc_now()
        self._save()
        return json.loads(self.index_path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minRE",
        description="Minimal sequential VLA dataset refinement: smooth, VLM label, clean, align, cut, classify, pair MP4/HDF5.",
    )
    parser.add_argument("dataset", help="Source dataset directory (read-only)")
    parser.add_argument("--output", help="Output dataset directory; defaults to a sibling named <source>_minRE")
    parser.add_argument("--vlm-config", default=str(ROOT / ".vla_lens" / "vlm-config.json"))
    parser.add_argument("--sample-count", type=int, default=18, choices=range(6, 25), metavar="6..24")
    parser.add_argument("--drop-phases", default=",".join(DEFAULT_DROP_PHASES))
    parser.add_argument("--min-clip-seconds", type=float, default=0.3)
    parser.add_argument("--force-vlm", action="store_true", help="Ignore reusable VLM annotations")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--index-only", action="store_true", help="Create/update the index without processing Episodes")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print the final dataset index as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.dataset).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else source.with_name(f"{source.name}_minRE")
    drop_phases = tuple(dict.fromkeys(
        phase
        for value in str(args.drop_phases).split(",")
        if (phase := _phase_key(value))
    ))
    if args.min_clip_seconds <= 0:
        print("minRE: --min-clip-seconds must be greater than zero", file=sys.stderr)
        return 2
    options = MinREOptions(
        source=source,
        output=output,
        vlm_config=Path(args.vlm_config).expanduser().resolve(),
        sample_count=args.sample_count,
        drop_phases=drop_phases,
        min_clip_seconds=float(args.min_clip_seconds),
        force_vlm=bool(args.force_vlm),
        fail_fast=bool(args.fail_fast),
        index_only=bool(args.index_only),
        quiet=bool(args.quiet or args.json),
    )
    try:
        result = MinREPipeline(options).run()
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        print(f"minRE: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(
            f"[minRE] {result.get('status')} | episodes={result.get('completed_episode_count')}/{result.get('episode_count')} "
            f"pairs={result.get('pair_count')} output={output}",
            flush=True,
        )
    return 1 if result.get("failed_episode_count") else 0


if __name__ == "__main__":
    raise SystemExit(main())
