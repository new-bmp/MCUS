from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2

from .episode_resolver import build_local_episode_plan, episode_key, episode_token
from .schema_profiler import build_inventory, pending_profile


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".vla_lens"
MANIFESTS = RUNTIME / "datasets"
ANNOTATIONS = RUNTIME / "annotations"
CACHE = RUNTIME / "cache"
EXPORTS = ROOT / "exports"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUXILIARY_EXTENSIONS = {".alice", ".crc", ".checksum", ".sha1", ".sha256", ".md5", ".lock", ".tmp"}
AUXILIARY_NAMES = {".complete", ".ds_store", "thumbs.db", "desktop.ini"}
IGNORED_DIRECTORY_NAMES = {
    ".alicepd", ".git", ".hg", ".svn", ".venv", "venv", ".vla_lens",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules",
    "output",
}
ALICE_ANNOTATION_EXTENSION = ".alice"
ALICE_ANNOTATION_SCHEMA = "alice/annotation/v1"
CHANGE_RECORD_SCHEMA = "alice/change-record/v1"
CHANGE_CATALOG_SCHEMA = "alice/change-catalog/v1"
CHANGE_APPLICATION_SCHEMA = "alice/change-application/v1"
EXCLUSION_SCHEMA = "alice/dataset-exclusions/v1"
INVALID_BITMAP_MAGIC = b"ALPDINV1"
INVALID_BITMAP_HEADER_BYTES = len(INVALID_BITMAP_MAGIC) + 8

_lock = threading.RLock()
_detail_index_cache: dict[tuple[str, int], dict] = {}


def _paper_curation_version_status(payload: dict) -> tuple[int, bool]:
    """Return the report version and whether it must be rerun.

    Paper reports predate the version field, so a missing value is treated as
    version 1.  The import is intentionally local: curation_pipeline imports
    storage during application startup, and importing it at module scope here
    would create a circular import.
    """
    from .curation_pipeline import CURATION_PIPELINE_VERSION

    try:
        version = int(payload.get("pipeline_version") or 1)
    except (TypeError, ValueError):
        version = 0
    return version, version != CURATION_PIPELINE_VERSION


def ensure_runtime() -> None:
    for folder in (MANIFESTS, ANNOTATIONS, CACHE, EXPORTS):
        folder.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return slug[:64] or "dataset"


def storage_slug(value: str) -> str:
    """Return a bounded path key without discarding the unique tail of long IDs."""
    source = value.strip()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", source).strip("-._") or "dataset"
    if len(slug) <= 64 and slug == source:
        return slug
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"{slug[:53]}-{digest}"


def _sidecar_parent(root: Path) -> Path:
    return (root / ".alicePD") if root.parent == root else (root.parent / ".alicePD")


def dataset_sidecar_root(root: str | Path, dataset_id: str) -> Path:
    source = Path(root).expanduser().resolve()
    parent = _sidecar_parent(source)
    preferred = parent / storage_slug(dataset_id)
    legacy = parent / slugify(dataset_id)
    if preferred.exists() or legacy == preferred or not legacy.exists():
        return preferred
    manifest_path = legacy / "dataset.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        payload = {}
    return legacy if str(payload.get("id") or "") == str(dataset_id) else preferred


def _ensure_sidecar_layout(sidecar_root: Path) -> None:
    for category in ("annotations", "cache", "exports", "auxiliary", "exclusions", "indices/invalid", "changes/records", "changes/applied"):
        (sidecar_root / category).mkdir(parents=True, exist_ok=True)


def _migrate_legacy_artifacts(dataset_id: str, sidecar_root: Path) -> None:
    legacy_registry = MANIFESTS / f"{slugify(dataset_id)}.json"
    try:
        legacy_payload = json.loads(legacy_registry.read_text(encoding="utf-8")) if legacy_registry.is_file() else {}
    except (OSError, json.JSONDecodeError):
        legacy_payload = {}
    if legacy_payload and str(legacy_payload.get("id") or "") not in {"", str(dataset_id)}:
        return
    if len(str(dataset_id)) > 64 and not legacy_payload:
        return
    legacy_roots = {
        ANNOTATIONS / slugify(dataset_id): sidecar_root / "annotations",
        CACHE / slugify(dataset_id): sidecar_root / "cache",
    }
    for source_root, target_root in legacy_roots.items():
        if not source_root.is_dir():
            continue
        for source in source_root.rglob("*"):
            if not source.is_file():
                continue
            target = target_root / source.relative_to(source_root)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _manifest_sidecar_path(manifest: dict) -> Path:
    configured = manifest.get("sidecar_path")
    if configured:
        return Path(configured).expanduser().resolve()
    return dataset_sidecar_root(manifest["root_path"], manifest["id"])


def _exclusion_path(manifest: dict) -> Path:
    return _manifest_sidecar_path(manifest) / "exclusions" / "files.alice"


def _load_exclusions(manifest: dict) -> dict:
    path = _exclusion_path(manifest)
    if not path.is_file():
        return {"schema": EXCLUSION_SCHEMA, "dataset_id": manifest["id"], "items": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema": EXCLUSION_SCHEMA, "dataset_id": manifest["id"], "items": []}
    items = [item for item in payload.get("items", []) if isinstance(item, dict) and item.get("relative_path")]
    return {**payload, "schema": EXCLUSION_SCHEMA, "dataset_id": manifest["id"], "items": items}


def apply_exclusions(manifest: dict) -> dict:
    exclusions = _load_exclusions(manifest)
    excluded_paths = {str(item.get("relative_path", "")).replace("\\", "/") for item in exclusions.get("items", [])}
    if not excluded_paths:
        manifest["excluded_file_count"] = 0
        return manifest
    root = Path(manifest["root_path"]).resolve()
    files = [item for item in manifest.get("files", []) if item.get("relative_path") not in excluded_paths]
    kept_ids = {item.get("id") for item in files}
    kept_paths = {str(item.get("relative_path", "")).replace("\\", "/") for item in files}
    episodes = []
    for original in manifest.get("episodes", []):
        episode = dict(original)
        if original.get("type") == "images":
            frames = []
            for frame in original.get("frames", []):
                try:
                    relative = Path(frame).resolve().relative_to(root).as_posix()
                except ValueError:
                    continue
                if relative in kept_paths:
                    frames.append(frame)
            if not frames:
                continue
            episode["frames"] = frames
            episode["frame_count"] = len(frames)
            episode["media_files"] = [Path(frame).resolve().relative_to(root).as_posix() for frame in frames]
            image_stream = dict((original.get("media_streams") or [{}])[0])
            image_stream.update({"frames": frames, "frame_count": len(frames), "duration": round(len(frames) / max(0.01, float(episode.get("fps", 30.0))), 4)})
            episode["media_streams"] = [image_stream]
        elif original.get("media_streams"):
            streams = [stream for stream in original.get("media_streams", []) if stream.get("file_id") in kept_ids]
            if not streams:
                continue
            primary_id = original.get("primary_media_file_id") if original.get("primary_media_file_id") in {stream.get("file_id") for stream in streams} else streams[0].get("file_id")
            primary = next(stream for stream in streams if stream.get("file_id") == primary_id)
            episode.update({key: value for key, value in primary.items() if key not in {"file_id", "stream_name"}})
            episode["primary_media_file_id"] = primary_id
            episode["media_streams"] = streams
            episode["media_files"] = [stream.get("relative_path") for stream in streams]
        episodes.append(episode)
    manifest["files"] = files
    manifest["episodes"] = episodes
    valid_episode_ids = {episode.get("id") for episode in episodes}
    resolution = manifest.get("episode_resolution") or {}
    resolution["groups"] = [
        {**group, "file_ids": [file_id for file_id in group.get("file_ids", []) if file_id in kept_ids]}
        for group in resolution.get("groups", [])
        if any(file_id in kept_ids for file_id in group.get("file_ids", [])) and (not group.get("playable_episode_id") or group.get("playable_episode_id") in valid_episode_ids)
    ]
    resolution["shared_file_ids"] = [file_id for file_id in resolution.get("shared_file_ids", []) if file_id in kept_ids]
    resolution["unassigned_file_ids"] = [file_id for file_id in resolution.get("unassigned_file_ids", []) if file_id in kept_ids]
    resolution["file_episode_assignments"] = {file_id: episode_id for file_id, episode_id in resolution.get("file_episode_assignments", {}).items() if file_id in kept_ids and episode_id in valid_episode_ids}
    manifest["episode_resolution"] = resolution
    manifest["episode_count"] = len(episodes)
    manifest["frame_count"] = sum(int(item.get("frame_count", 0) or 0) for item in episodes)
    manifest["file_count"] = len(files)
    manifest["total_size"] = sum(int(item.get("size_bytes", 0) or 0) for item in files)
    manifest["type_counts"] = {kind: sum(item.get("kind") == kind for item in files) for kind in sorted({item.get("kind") for item in files})}
    manifest["excluded_file_count"] = len(excluded_paths)
    manifest["excluded_files"] = sorted(excluded_paths, key=str.casefold)
    inventory = (manifest.get("schema_profile") or {}).get("inventory")
    if isinstance(inventory, dict) and isinstance(inventory.get("files"), list):
        inventory["files"] = [item for item in inventory["files"] if item.get("path") in kept_paths]
        inventory["file_count"] = len(inventory["files"])
        inventory["field_count"] = sum(len(item.get("fields", [])) for item in inventory["files"])
    return manifest


def _is_sidecar_member(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return any(part.casefold() == ".alicepd" for part in relative.parts)


def _is_auxiliary_source_file(path: Path) -> bool:
    return path.name.casefold() in AUXILIARY_NAMES or path.suffix.casefold() in AUXILIARY_EXTENSIONS


def _auxiliary_record(path: Path, root: Path) -> dict:
    stat = path.stat()
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "name": path.name,
        "extension": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "reason": "alice_annotation" if path.suffix.lower() == ALICE_ANNOTATION_EXTENSION else "control_or_checksum_file",
    }


def _iter_dataset_files(root: Path):
    """Walk once with directory pruning and reuse DirEntry stat results."""
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            name = entry.name.casefold()
            try:
                if entry.is_dir(follow_symlinks=False):
                    if name not in IGNORED_DIRECTORY_NAMES:
                        stack.append(Path(entry.path))
                    continue
                if entry.is_file(follow_symlinks=False):
                    yield Path(entry.path), entry.stat(follow_symlinks=False)
            except OSError:
                continue


def discover_dataset_roots(path: str | Path) -> dict:
    """Discover immediate child datasets without scanning their contents."""
    root = Path(path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"数据集目录不存在: {root}")
    children: list[Path] = []
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                name = entry.name.casefold()
                if name in IGNORED_DIRECTORY_NAMES or name.startswith("."):
                    continue
                try:
                    if entry.is_dir(follow_symlinks=False):
                        children.append(Path(entry.path).resolve())
                except OSError:
                    continue
    except OSError as exc:
        raise ValueError(f"无法读取目录: {root}") from exc
    dataset_roots = sorted(children, key=lambda item: item.name.casefold()) if children else [root]
    mode = "collection" if children else "single"
    return {
        "mode": mode,
        "root_path": str(root),
        "dataset_count": len(dataset_roots),
        "datasets": [
            {
                "key": hashlib.sha1(str(item).casefold().encode("utf-8")).hexdigest()[:16],
                "name": item.name or str(item),
                "path": str(item),
                "status": "unloaded" if mode == "collection" else "loading",
            }
            for item in dataset_roots
        ],
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(payload)
    temp.replace(path)


def _episode_frame_count(manifest: dict, episode_id: str, payload: dict | None = None) -> int:
    payload_frame_count = int(((payload or {}).get("source_video") or {}).get("frame_count", 0) or 0)
    if payload_frame_count > 0:
        return payload_frame_count
    episode = next((item for item in manifest.get("episodes", []) if item.get("id") == episode_id), None)
    if episode:
        return max(0, int(episode.get("frame_count", 0) or 0))
    ends = [int(item.get("end_frame", -1)) for item in (payload or {}).get("segments", [])]
    frames = [int(item.get("frame", -1)) for item in (payload or {}).get("samples", [])]
    return max([-1, *ends, *frames]) + 1


def _invalid_intervals(payload: dict, frame_count: int) -> list[list[int]]:
    candidates: list[tuple[int, int]] = []
    for segment in payload.get("segments", []):
        if segment.get("state") != "invalid":
            continue
        start = max(0, int(segment.get("start_frame", 0) or 0))
        end = min(frame_count - 1, int(segment.get("end_frame", start) or start))
        if start <= end:
            candidates.append((start, end))
    for sample in payload.get("samples", []):
        if sample.get("state") != "invalid":
            continue
        raw_frame = sample.get("frame")
        if raw_frame is None:
            continue
        frame = int(raw_frame)
        if 0 <= frame < frame_count:
            candidates.append((frame, frame))
    merged: list[list[int]] = []
    for start, end in sorted(candidates):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _mark_bitmap_interval(bitmap: bytearray, start: int, end: int) -> None:
    start_byte, end_byte = start >> 3, end >> 3
    start_bit, end_bit = start & 7, end & 7
    if start_byte == end_byte:
        bitmap[start_byte] |= ((1 << (end_bit - start_bit + 1)) - 1) << start_bit
        return
    bitmap[start_byte] |= (0xFF << start_bit) & 0xFF
    if end_byte > start_byte + 1:
        bitmap[start_byte + 1:end_byte] = b"\xff" * (end_byte - start_byte - 1)
    bitmap[end_byte] |= (1 << (end_bit + 1)) - 1


def _write_invalid_frame_index(sidecar_root: Path, manifest: dict, episode_id: str, payload: dict) -> dict:
    frame_count = _episode_frame_count(manifest, episode_id, payload)
    intervals = _invalid_intervals(payload, frame_count)
    bitmap = bytearray((frame_count + 7) // 8)
    for start, end in intervals:
        _mark_bitmap_interval(bitmap, start, end)
    invalid_count = sum(end - start + 1 for start, end in intervals)
    index_root = sidecar_root / "indices" / "invalid"
    stem = storage_slug(episode_id)
    bitmap_name = f"{stem}.invalid.bin"
    bitmap_payload = INVALID_BITMAP_MAGIC + struct.pack("<Q", frame_count) + bytes(bitmap)
    _write_bytes_atomic(index_root / bitmap_name, bitmap_payload)
    index = {
        "schema": "alice/invalid-frame-index/v1",
        "dataset_id": manifest["id"],
        "episode_id": episode_id,
        "frame_count": frame_count,
        "invalid_frame_count": invalid_count,
        "invalid_intervals": intervals,
        "bitmap": bitmap_name,
        "bitmap_format": {
            "magic": INVALID_BITMAP_MAGIC.decode("ascii"),
            "header_bytes": INVALID_BITMAP_HEADER_BYTES,
            "frame_count_encoding": "uint64-little-endian",
            "bit_order": "lsb0",
            "meaning": "1=invalid, 0=not_marked_invalid",
        },
        "annotation_schema": payload.get("schema"),
        "annotation_created_at": payload.get("created_at"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    index_name = f"{stem}.invalid.alice"
    _write_json_atomic(index_root / index_name, index)
    catalog_path = index_root / "index.alice"
    try:
        legacy_catalog = index_root / "index.json"
        source_catalog = catalog_path if catalog_path.exists() else legacy_catalog
        catalog = json.loads(source_catalog.read_text(encoding="utf-8")) if source_catalog.exists() else {}
    except (OSError, json.JSONDecodeError):
        catalog = {}
    catalog.update({
        "schema": "alice/invalid-frame-catalog/v1",
        "dataset_id": manifest["id"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    catalog.setdefault("episodes", {})[episode_id] = {
        "index": index_name,
        "bitmap": bitmap_name,
        "frame_count": frame_count,
        "invalid_frame_count": invalid_count,
        "interval_count": len(intervals),
    }
    _write_json_atomic(catalog_path, catalog)
    return index


def _backfill_invalid_indices(manifest: dict, sidecar_root: Path) -> None:
    annotation_root = sidecar_root / "annotations"
    paths = {path.stem: path for path in annotation_root.glob("*.json")}
    paths.update({path.stem: path for path in annotation_root.glob(f"*{ALICE_ANNOTATION_EXTENSION}")})
    for path in paths.values():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            episode_id = str(payload.get("episode_id") or path.stem)
            _write_invalid_frame_index(sidecar_root, manifest, episode_id, payload)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue


def dataset_artifact_dir(dataset_id: str, category: str) -> Path:
    manifest = get_manifest(dataset_id)
    target = _manifest_sidecar_path(manifest) / slugify(category)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _artifact_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _change_record_id(kind: str, episode_id: str | None) -> str:
    key = f"{kind}:{episode_id or 'dataset'}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _read_artifact_schema(path: Path) -> str | None:
    if path.suffix.lower() != ALICE_ANNOTATION_EXTENSION:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("schema")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def record_change(
    dataset_id: str,
    kind: str,
    episode_id: str | None,
    title: str,
    artifacts: list[str | Path],
    summary: dict | None = None,
    source_paths: list[str] | None = None,
) -> dict:
    """Stage a change in .alicePD without mutating the source dataset."""
    manifest = get_manifest(dataset_id)
    sidecar_root = _manifest_sidecar_path(manifest)
    records_root = sidecar_root / "changes" / "records"
    records_root.mkdir(parents=True, exist_ok=True)
    artifact_records = []
    for value in artifacts:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(sidecar_root).as_posix()
        except ValueError as exc:
            raise ValueError("Change artifacts must remain inside .alicePD") from exc
        artifact_records.append({
            "relative_path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": _artifact_digest(path),
            "schema": _read_artifact_schema(path),
        })
    if not artifact_records:
        raise ValueError("Change record has no readable .alicePD artifacts")
    known_source_paths = {str(item.get("relative_path") or "") for item in manifest.get("files", [])}
    normalized_source_paths = sorted({str(value).replace("\\", "/") for value in (source_paths or []) if str(value).replace("\\", "/") in known_source_paths})
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        record_id = _change_record_id(kind, episode_id)
        path = records_root / f"{record_id}.alice"
        existing = {}
        if path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
        fingerprint_source = json.dumps(artifact_records, sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()
        changed = existing.get("fingerprint") != fingerprint
        document = {
            "schema": CHANGE_RECORD_SCHEMA,
            "id": record_id,
            "key": f"{kind}:{episode_id or 'dataset'}",
            "dataset_id": dataset_id,
            "episode_id": episode_id,
            "kind": kind,
            "title": title,
            "status": "pending" if changed else existing.get("status", "pending"),
            "revision": int(existing.get("revision", 0) or 0) + (1 if changed else 0),
            "created_at": existing.get("created_at") or now,
            "updated_at": now if changed else existing.get("updated_at", now),
            "fingerprint": fingerprint,
            "summary": summary or {},
            "source_paths": normalized_source_paths,
            "artifacts": artifact_records,
        }
        if not changed and existing.get("status") == "applied":
            document["applied_at"] = existing.get("applied_at")
            document["application_id"] = existing.get("application_id")
            document["snapshot_artifacts"] = existing.get("snapshot_artifacts", [])
        _write_json_atomic(path, document)
    return document


def _backfill_change_records(manifest: dict) -> None:
    dataset_id = manifest["id"]
    sidecar_root = _manifest_sidecar_path(manifest)
    annotation_root = sidecar_root / "annotations"
    for path in annotation_root.glob(f"*{ALICE_ANNOTATION_EXTENSION}"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            episode_id = str(payload.get("episode_id") or path.stem)
            stem = storage_slug(episode_id)
            artifacts: list[Path] = [path]
            invalid_root = sidecar_root / "indices" / "invalid"
            legacy_stem = slugify(episode_id)
            for candidate in dict.fromkeys((
                invalid_root / f"{stem}.invalid.alice",
                invalid_root / f"{stem}.invalid.bin",
                invalid_root / f"{legacy_stem}.invalid.alice",
                invalid_root / f"{legacy_stem}.invalid.bin",
            )):
                if candidate.is_file():
                    artifacts.append(candidate)
            episode = next((item for item in manifest.get("episodes", []) if item.get("id") == episode_id), {})
            record_change(dataset_id, "episode_annotation", episode_id, f"Episode annotation: {episode_id}", artifacts, payload.get("summary") or {}, [episode.get("relative_path", "")])
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    behavior_root = sidecar_root / "behavior-annotations"
    for path in behavior_root.glob("*.behavior.alice"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            episode_id = str(payload.get("episode_id") or path.name.removesuffix(".behavior.alice"))
            target = sidecar_root / "behavior-targets" / f"{slugify(episode_id)}.targets.alice"
            if target.is_file():
                target_payload = json.loads(target.read_text(encoding="utf-8"))
                if "primary_terms" not in target_payload:
                    target_payload["primary_terms"] = [str(item.get("name")) for item in target_payload.get("primary_targets", []) if item.get("name")]
                    _write_json_atomic(target, target_payload)
            artifacts = [path, target] if target.is_file() else [path]
            episode = next((item for item in manifest.get("episodes", []) if item.get("id") == episode_id), {})
            record_change(dataset_id, "vlm_behavior", episode_id, f"VLM behavior: {episode_id}", artifacts, {
                "task_label": payload.get("task_label"),
                "confidence": payload.get("confidence"),
                "target_count": len(payload.get("primary_targets", [])),
            }, [episode.get("relative_path", "")])
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    trim_root = sidecar_root / "no-action-trim"
    for path in trim_root.glob("*.trim.alice"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            episode_id = str(payload.get("episode_id") or path.name.removesuffix(".trim.alice"))
            episode = next((item for item in manifest.get("episodes", []) if item.get("id") == episode_id), {})
            record_change(dataset_id, "no_action_trim", episode_id, f"No-action trim: {episode_id}", [path], payload.get("summary") or {}, [episode.get("relative_path", "")])
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    qwen_trim_root = sidecar_root / "qwen-action-trim"
    for path in qwen_trim_root.glob("*.trim.alice"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            episode_id = str(payload.get("episode_id") or path.name.removesuffix(".trim.alice"))
            episode = next((item for item in manifest.get("episodes", []) if item.get("id") == episode_id), {})
            source_video = payload.get("source_video") or {}
            source_path = str(source_video.get("relative_path") or episode.get("relative_path") or "")
            record_change(
                dataset_id,
                "qwen_action_trim",
                episode_id,
                f"Qwen action trim: {episode_id}",
                [path],
                payload.get("summary") or {},
                [source_path],
            )
        except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
            continue
    pose_root = sidecar_root / "pose-recovery"
    for path in pose_root.glob("*.slam.alice"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            episode_id = str(payload.get("episode_id") or path.name.removesuffix(".slam.alice"))
            record_change(dataset_id, "pose_recovery", episode_id, f"Pose recovery: {episode_id}", [path], {
                "recovered_frame_count": payload.get("recovered_frame_count", 0),
                "method": payload.get("method", "slam_visual_odometry"),
            }, [str(item.get("source_path") or "") for item in payload.get("sides", [])])
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    curation_root = sidecar_root / "curation"
    for path in curation_root.glob("*.curation.alice"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") != "alice/paper-curation/v1":
                continue
            pipeline_version, requires_rerun = _paper_curation_version_status(payload)
            episode_id = str(payload.get("episode_id") or path.name.removesuffix(".curation.alice"))
            preferred = curation_root / f"{storage_slug(episode_id)}.curation.alice"
            if preferred.is_file() and path.resolve() != preferred.resolve():
                continue
            summary = {
                **(payload.get("summary") or {}),
                "pipeline_version": pipeline_version,
                "requires_rerun": requires_rerun,
            }
            record_change(
                dataset_id,
                "paper_curation",
                episode_id,
                f"Paper curation: {payload.get('episode_name') or episode_id}",
                [path],
                summary,
                [str(item.get("relative_path") or "") for item in payload.get("source_signatures", [])],
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue


def list_changes(dataset_id: str, backfill: bool = True) -> dict:
    manifest = get_manifest(dataset_id)
    if backfill:
        _backfill_change_records(manifest)
    sidecar_root = _manifest_sidecar_path(manifest)
    items = []
    for path in (sidecar_root / "changes" / "records").glob("*.alice"):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            if item.get("schema") == CHANGE_RECORD_SCHEMA:
                if item.get("kind") == "paper_curation":
                    artifact = next((candidate for candidate in item.get("artifacts", []) if candidate.get("schema") == "alice/paper-curation/v1"), None)
                    if artifact:
                        artifact_path = (sidecar_root / str(artifact.get("relative_path") or "")).resolve()
                        try:
                            artifact_path.relative_to(sidecar_root)
                            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                            pipeline_version, requires_rerun = _paper_curation_version_status(payload)
                        except (OSError, ValueError, json.JSONDecodeError, TypeError):
                            pipeline_version, requires_rerun = 0, True
                        item["pipeline_version"] = pipeline_version
                        item["requires_rerun"] = requires_rerun
                    else:
                        item["pipeline_version"] = 0
                        item["requires_rerun"] = True
                items.append(item)
        except (OSError, json.JSONDecodeError):
            continue
    items.sort(key=lambda item: (item.get("status") != "pending", item.get("updated_at", "")), reverse=False)
    pending = [item for item in items if item.get("status") == "pending"]
    applied = [item for item in items if item.get("status") == "applied"]
    runnable_pending = [item for item in pending if not item.get("requires_rerun")]
    return {
        "schema": CHANGE_CATALOG_SCHEMA,
        "dataset_id": dataset_id,
        "sidecar_path": str(sidecar_root / "changes"),
        "source_policy": "Source media and sensor files remain read-only. Apply activates reviewed .alicePD snapshots.",
        "pending_count": len(pending),
        "runnable_pending_count": len(runnable_pending),
        "requires_rerun_count": len(pending) - len(runnable_pending),
        "applied_count": len(applied),
        "items": pending + sorted(applied, key=lambda item: item.get("applied_at", ""), reverse=True),
    }


def change_is_applied(dataset_id: str, kind: str, episode_id: str | None) -> bool:
    manifest = get_manifest(dataset_id)
    path = _manifest_sidecar_path(manifest) / "changes" / "records" / f"{_change_record_id(kind, episode_id)}.alice"
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return record.get("status") == "applied" and record.get("fingerprint") is not None


def _verify_qwen_trim_source(manifest: dict, artifact_path: Path) -> None:
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        source_video = payload.get("source_video") or {}
        expected = source_video.get("fingerprint")
        relative_path = str(source_video.get("relative_path") or "").replace("\\", "/")
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        raise ValueError("Qwen trim artifact is not readable") from exc
    if not isinstance(expected, dict) or expected.get("schema") != "alice/source-video-fingerprint/v1":
        raise ValueError("Qwen trim result predates source-version locking; rerun Qwen trim before applying")
    known_paths = {
        str(item.get("relative_path") or "").replace("\\", "/").casefold(): str(item.get("relative_path") or "")
        for item in manifest.get("files", [])
    }
    canonical_relative = known_paths.get(relative_path.casefold())
    if not canonical_relative:
        raise ValueError(f"Qwen trim source is no longer part of the dataset: {relative_path}")
    dataset_root = Path(manifest["root_path"]).expanduser().resolve()
    source_path = (dataset_root / canonical_relative).resolve()
    try:
        source_path.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError("Qwen trim source escaped the dataset root") from exc
    try:
        from .qwen_trim import _source_fingerprints_match, _source_video_fingerprint

        current = _source_video_fingerprint({"path": str(source_path)})
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Cannot verify Qwen trim source: {exc}") from exc
    if not _source_fingerprints_match(expected, current):
        raise ValueError(f"Source video changed after Qwen review: {relative_path}")


def _verify_paper_curation_source(manifest: dict, artifact_path: Path, expected_episode_id: str | None = None) -> dict:
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        signatures = payload.get("source_signatures") or []
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        raise ValueError("Paper curation artifact is not readable") from exc
    if payload.get("schema") != "alice/paper-curation/v1" or not signatures:
        raise ValueError("Paper curation artifact has no source-version lock; rerun curation before applying")
    if str(payload.get("dataset_id") or "") != str(manifest.get("id") or ""):
        raise ValueError("Paper curation artifact belongs to a different dataset")
    if expected_episode_id is not None and str(payload.get("episode_id") or "") != str(expected_episode_id):
        raise ValueError("Paper curation artifact belongs to a different Episode")
    from .curation_pipeline import CURATION_PIPELINE_VERSION, source_signatures_match

    try:
        pipeline_version = int(payload.get("pipeline_version") or 1)
    except (TypeError, ValueError) as exc:
        raise ValueError("Paper curation report has an invalid pipeline version; rerun curation before applying") from exc
    if pipeline_version != CURATION_PIPELINE_VERSION:
        raise ValueError("Paper curation algorithm has changed; rerun curation before applying")

    root = Path(manifest["root_path"]).expanduser().resolve()
    allowed_paths = [str(item.get("relative_path") or "") for item in manifest.get("files", [])]
    matches, changed_path = source_signatures_match(root, signatures, allowed_paths=allowed_paths)
    if not matches:
        raise ValueError(f"Source changed after paper curation: {changed_path or 'unknown source'}")
    return payload


def apply_changes(dataset_id: str, change_ids: list[str]) -> dict:
    catalog = list_changes(dataset_id, backfill=True)
    selected_ids = set(change_ids)
    if not selected_ids:
        raise ValueError("Select at least one pending change")
    selected = [item for item in catalog["items"] if item.get("id") in selected_ids]
    if len(selected) != len(selected_ids):
        raise KeyError("Unknown change record")
    if any(item.get("status") != "pending" for item in selected):
        raise ValueError("Only pending changes can be applied")
    manifest = get_manifest(dataset_id)
    sidecar_root = _manifest_sidecar_path(manifest)
    for record in selected:
        kind = record.get("kind")
        if kind not in {"qwen_action_trim", "paper_curation"}:
            continue
        expected_schema = "alice/qwen-action-trim/v1" if kind == "qwen_action_trim" else "alice/paper-curation/v1"
        versioned_artifacts = [item for item in record.get("artifacts", []) if item.get("schema") == expected_schema]
        if len(versioned_artifacts) != 1:
            raise ValueError(f"{kind} change must contain exactly one versioned artifact")
        artifact = versioned_artifacts[0]
        artifact_path = (sidecar_root / artifact["relative_path"]).resolve()
        try:
            artifact_path.relative_to(sidecar_root)
        except ValueError as exc:
            raise ValueError("Change artifact escaped .alicePD") from exc
        if not artifact_path.is_file() or _artifact_digest(artifact_path) != artifact.get("sha256"):
            raise ValueError(f"Change artifact changed after review: {artifact.get('relative_path')}")
        if kind == "qwen_action_trim":
            _verify_qwen_trim_source(manifest, artifact_path)
        else:
            _verify_paper_curation_source(manifest, artifact_path, str(record.get("episode_id") or ""))
    now = datetime.now(timezone.utc)
    application_id = f"apply-{now.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    application_root = sidecar_root / "changes" / "applied" / application_id
    receipt_items = []
    with _lock:
        for record in selected:
            snapshot_artifacts = []
            for artifact in record.get("artifacts", []):
                source = (sidecar_root / artifact["relative_path"]).resolve()
                try:
                    source.relative_to(sidecar_root)
                except ValueError as exc:
                    raise ValueError("Change artifact escaped .alicePD") from exc
                if not source.is_file() or _artifact_digest(source) != artifact.get("sha256"):
                    raise ValueError(f"Change artifact changed after review: {artifact.get('relative_path')}")
                target = application_root / "artifacts" / artifact["relative_path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                snapshot_artifacts.append({**artifact, "snapshot_path": target.relative_to(sidecar_root).as_posix()})
            if record.get("kind") == "paper_curation":
                paper_artifact = next(item for item in record.get("artifacts", []) if item.get("schema") == "alice/paper-curation/v1")
                paper_path = (sidecar_root / paper_artifact["relative_path"]).resolve()
                payload = _verify_paper_curation_source(manifest, paper_path, str(record.get("episode_id") or ""))
                index = _write_invalid_frame_index(sidecar_root, manifest, str(record.get("episode_id") or ""), payload)
                generated = [
                    sidecar_root / "indices" / "invalid" / f"{storage_slug(str(record.get('episode_id') or ''))}.invalid.alice",
                    sidecar_root / "indices" / "invalid" / index["bitmap"],
                ]
                for source in generated:
                    if not source.is_file():
                        raise ValueError(f"Invalid-frame index was not generated: {source.name}")
                    target = application_root / "artifacts" / source.relative_to(sidecar_root)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    generated_record = {
                        "relative_path": source.relative_to(sidecar_root).as_posix(),
                        "size_bytes": source.stat().st_size,
                        "sha256": _artifact_digest(source),
                        "schema": _read_artifact_schema(source),
                    }
                    shutil.copy2(source, target)
                    snapshot_artifacts.append({**generated_record, "snapshot_path": target.relative_to(sidecar_root).as_posix()})
            applied_record = {**record, "status": "applied", "applied_at": now.isoformat(), "application_id": application_id, "snapshot_artifacts": snapshot_artifacts}
            _write_json_atomic(application_root / "records" / f"{record['id']}.alice", applied_record)
            _write_json_atomic(sidecar_root / "changes" / "records" / f"{record['id']}.alice", applied_record)
            receipt_items.append({"id": record["id"], "key": record["key"], "revision": record["revision"], "snapshot_artifacts": snapshot_artifacts})
        current_path = sidecar_root / "changes" / "current.alice"
        try:
            current = json.loads(current_path.read_text(encoding="utf-8")) if current_path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            current = {}
        current.update({"schema": "alice/applied-change-set/v1", "dataset_id": dataset_id, "updated_at": now.isoformat()})
        entries = current.setdefault("entries", {})
        for item in receipt_items:
            entries[item["key"]] = {"application_id": application_id, "change_id": item["id"], "revision": item["revision"], "artifacts": item["snapshot_artifacts"]}
        _write_json_atomic(current_path, current)
        receipt = {"schema": CHANGE_APPLICATION_SCHEMA, "id": application_id, "dataset_id": dataset_id, "applied_at": now.isoformat(), "source_mutated": False, "change_count": len(receipt_items), "changes": receipt_items}
        _write_json_atomic(application_root / "receipt.alice", receipt)
    return {"application": receipt, "catalog": list_changes(dataset_id, backfill=False)}


def dataset_cache_dir(dataset_id: str, episode_id: str | None = None) -> Path:
    target = dataset_artifact_dir(dataset_id, "cache")
    if episode_id:
        target = target / storage_slug(episode_id)
        target.mkdir(parents=True, exist_ok=True)
    return target


def _episode_id(group_key: str) -> str:
    stem = slugify(Path(group_key).name)
    digest = hashlib.sha1(group_key.encode("utf-8")).hexdigest()[:7]
    return f"{stem}-{digest}"


def _file_id(relative_path: str) -> str:
    return hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:16]


def _file_kind(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return "video", "vision"
    if suffix in IMAGE_EXTENSIONS:
        return "image", "vision"
    if suffix in {".h5", ".hdf5", ".h5df", ".npz", ".npy", ".parquet", ".csv", ".tsv"}:
        return "structured", "sensor"
    if suffix in {".json", ".jsonl", ".yaml", ".yml", ".toml", ".html", ".htm"}:
        return "metadata", "metadata"
    if suffix in {".txt", ".md"}:
        return "text", "metadata"
    return "file", "other"


def _primary_video(records: list[dict]) -> dict:
    def priority(record: dict) -> tuple[int, str]:
        name = Path(record["relative_path"]).stem.lower()
        if "head_rgb" in name or name in {"rgb", "main", "color"}:
            score = 0
        elif "rgb" in name or "color" in name:
            score = 1
        elif "wrist" in name:
            score = 3
        elif "depth" in name or "infrared" in name:
            score = 9
        else:
            score = 4
        return score, record["relative_path"].lower()

    return min(records, key=priority)


def _probe_video(path: Path, relative_path: str, group_key: str) -> dict | None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return None
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    fps = fps if fps > 0.01 else 30.0
    return {
        "id": _episode_id(group_key),
        "name": Path(group_key).name,
        "episode_key": group_key,
        "type": "video",
        "path": str(path.resolve()),
        "relative_path": relative_path,
        "fps": round(fps, 4),
        "frame_count": frame_count,
        "duration": round(frame_count / fps, 4) if frame_count else 0.0,
        "width": width,
        "height": height,
    }


def _parquet_timeline_length(path: Path) -> int | None:
    try:
        import pyarrow.parquet as parquet

        source = parquet.ParquetFile(path)
        if "frame_index" not in source.schema_arrow.names:
            return None
        return int(source.metadata.num_rows)
    except Exception:
        return None


def _natural_sort_key(value: str) -> list[tuple[int, object]]:
    return [
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", value)
        if part
    ]


def _probe_images(paths: list[Path], root: Path, group_key: str) -> dict | None:
    if not paths:
        return None
    paths = sorted(paths, key=lambda item: item.name.lower())
    image = cv2.imread(str(paths[0]))
    if image is None:
        return None
    relative_dir = paths[0].parent.relative_to(root).as_posix() or paths[0].parent.name
    height, width = image.shape[:2]
    fps = 30.0
    return {
        "id": _episode_id(group_key),
        "name": Path(group_key).name,
        "episode_key": group_key,
        "type": "images",
        "path": str(paths[0].parent.resolve()),
        "relative_path": relative_dir,
        "frames": [str(path.resolve()) for path in paths],
        "fps": fps,
        "frame_count": len(paths),
        "duration": round(len(paths) / fps, 4),
        "width": width,
        "height": height,
    }


def scan_dataset(path: str | Path, name: str | None = None, dataset_id: str | None = None) -> dict:
    ensure_runtime()
    root = Path(path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"数据集目录不存在: {root}")

    existing_sidecar = None
    if dataset_id:
        try:
            existing_sidecar = _manifest_sidecar_path(get_manifest(dataset_id))
        except KeyError:
            existing_sidecar = None
    dataset_id = dataset_id or f"{slugify(name or root.name)}-{uuid.uuid4().hex[:8]}"
    sidecar_root = existing_sidecar or dataset_sidecar_root(root, dataset_id)
    _ensure_sidecar_layout(sidecar_root)
    episodes: list[dict] = []
    indexed_files: list[dict] = []
    auxiliary_files: list[dict] = []
    video_groups: dict[str, list[dict]] = {}
    image_groups: dict[Path, list[Path]] = {}
    for item, stat in _iter_dataset_files(root):
        if RUNTIME in item.parents or _is_sidecar_member(item, root):
            continue
        if _is_auxiliary_source_file(item):
            auxiliary_files.append(_auxiliary_record(item, root))
            continue
        suffix = item.suffix.lower()
        relative = item.relative_to(root).as_posix()
        kind, category = _file_kind(item)
        image_sequence = suffix in IMAGE_EXTENSIONS and item.stem.isdigit()
        group_key = item.parent.relative_to(root).as_posix() if image_sequence else episode_key(item, root)
        if not group_key or group_key == ".":
            group_key = root.name
        record = {
            "id": _file_id(relative),
            "name": item.name,
            "relative_path": relative,
            "parent": item.parent.relative_to(root).as_posix(),
            "extension": suffix,
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "kind": kind,
            "category": category,
            "episode_key": group_key,
            "episode_token": episode_token(item, root),
        }
        indexed_files.append(record)
        if suffix in VIDEO_EXTENSIONS:
            video_groups.setdefault(group_key, []).append(record)
        elif suffix in IMAGE_EXTENSIONS:
            image_groups.setdefault(item.parent, []).append(item)

    video_results: dict[str, list[tuple[dict, dict]]] = {}

    def probe_video_record(payload: tuple[str, dict]) -> tuple[str, dict, dict | None]:
        group_key, record = payload
        stream = _probe_video(root / record["relative_path"], record["relative_path"], group_key)
        return group_key, record, stream

    probe_payloads = [
        (group_key, record)
        for group_key, records in video_groups.items()
        for record in records
    ]
    try:
        configured_workers = int(os.getenv("VLA_SCAN_WORKERS", str(min(8, os.cpu_count() or 4))))
    except ValueError:
        configured_workers = min(8, os.cpu_count() or 4)
    worker_count = max(1, min(8, configured_workers))
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="alice-media-probe") as executor:
        for group_key, record, stream in executor.map(probe_video_record, probe_payloads):
            if stream:
                video_results.setdefault(group_key, []).append((record, stream))

    for group_key, records in video_groups.items():
        media_streams = []
        readable_records = []
        for record, stream in video_results.get(group_key, []):
            stream["file_id"] = record["id"]
            stream["stream_name"] = Path(record["relative_path"]).name
            media_streams.append(stream)
            readable_records.append(record)
        if not media_streams:
            continue
        primary = _primary_video(readable_records)
        primary_stream = next(item for item in media_streams if item["file_id"] == primary["id"])
        episode = dict(primary_stream)
        episode["id"] = _episode_id(group_key)
        episode["name"] = Path(group_key).name
        episode["episode_key"] = group_key
        episode["primary_media_file_id"] = primary["id"]
        episode["media_files"] = [item["relative_path"] for item in media_streams]
        episode["media_streams"] = media_streams
        episodes.append(episode)

    for parent, paths in image_groups.items():
        group_key = parent.relative_to(root).as_posix() or parent.name
        episode = _probe_images(paths, root, group_key)
        if episode:
            episode["media_files"] = [path.relative_to(root).as_posix() for path in sorted(paths)]
            episode["primary_media_file_id"] = None
            episode["media_streams"] = [{
                key: value for key, value in episode.items()
                if key in {"type", "path", "relative_path", "frames", "fps", "frame_count", "duration", "width", "height"}
            }]
            episodes.append(episode)

    episodes_by_key = {episode["episode_key"]: episode for episode in episodes}
    parquet_lengths: dict[str, list[int]] = {}
    for record in indexed_files:
        if record["extension"] != ".parquet" or record["episode_key"] not in episodes_by_key:
            continue
        length = _parquet_timeline_length(root / record["relative_path"])
        if length is not None:
            parquet_lengths.setdefault(record["episode_key"], []).append(length)
    for key, lengths in parquet_lengths.items():
        episode = episodes_by_key[key]
        data_frame_count = max(lengths)
        media_frame_count = int(episode.get("frame_count", 0) or 0)
        logical_frame_count = min(data_frame_count, media_frame_count) if media_frame_count else data_frame_count
        episode["source_media_frame_count"] = media_frame_count
        episode["data_frame_count"] = data_frame_count
        episode["frame_count"] = logical_frame_count
        episode["duration"] = round(logical_frame_count / max(0.01, float(episode.get("fps", 30.0))), 4)
        episode["alignment"] = {
            "source": "parquet_frame_index",
            "logical_frame_count": logical_frame_count,
            "data_frame_count": data_frame_count,
            "source_media_frame_count": media_frame_count,
            "trimmed_media_frames": max(0, media_frame_count - logical_frame_count),
        }
        for stream in episode.get("media_streams", []):
            stream_count = int(stream.get("frame_count", 0) or 0)
            stream["source_frame_count"] = stream_count
            stream["frame_count"] = min(logical_frame_count, stream_count) if stream_count else logical_frame_count
            stream["duration"] = round(stream["frame_count"] / max(0.01, float(stream.get("fps", 30.0))), 4)

    episodes.sort(key=lambda item: _natural_sort_key(item["episode_key"]))
    if not episodes:
        raise ValueError("目录中未发现可读取的视频或图像序列")

    episode_ids = {episode["episode_key"]: episode["id"] for episode in episodes}
    for record in indexed_files:
        record["episode_id"] = episode_ids.get(record["episode_key"])
    indexed_files.sort(key=lambda item: item["relative_path"].lower())

    episode_resolution = build_local_episode_plan(indexed_files, episodes)
    manifest = {
        "id": dataset_id,
        "name": name or root.name,
        "root_path": str(root),
        "sidecar_path": str(sidecar_root),
        "sidecar_layout": {
            "manifest": "dataset.json",
            "annotations": "annotations",
            "annotation_extension": ALICE_ANNOTATION_EXTENSION,
            "annotation_schema": ALICE_ANNOTATION_SCHEMA,
            "cache": "cache",
            "exports": "exports",
            "auxiliary_index": "auxiliary/source-files.json",
            "invalid_frame_indices": "indices/invalid",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "episode_count": len(episodes),
        "frame_count": sum(item["frame_count"] for item in episodes),
        "file_count": len(indexed_files),
        "auxiliary_file_count": len(auxiliary_files),
        "auxiliary_total_size": sum(item["size_bytes"] for item in auxiliary_files),
        "total_size": sum(item["size_bytes"] for item in indexed_files),
        "type_counts": {
            kind: sum(item["kind"] == kind for item in indexed_files)
            for kind in sorted({item["kind"] for item in indexed_files})
        },
        "episodes": episodes,
        "files": indexed_files,
        "episode_resolution": episode_resolution,
        "schema_profile": pending_profile(build_inventory(root, episodes, {item["relative_path"] for item in indexed_files})),
    }
    manifest = apply_exclusions(manifest)
    _write_json_atomic(sidecar_root / "auxiliary" / "source-files.json", {
        "schema": "alicePD/source-auxiliary/v1",
        "source_root": str(root),
        "dataset_id": dataset_id,
        "policy": "Source remains read-only. Auxiliary files are excluded from the logical dataset and referenced in place.",
        "file_count": len(auxiliary_files),
        "total_size": sum(item["size_bytes"] for item in auxiliary_files),
        "files": sorted(auxiliary_files, key=lambda item: item["relative_path"].lower()),
    })
    save_manifest(manifest)
    return manifest


def save_manifest(manifest: dict) -> None:
    ensure_runtime()
    sidecar_root = _manifest_sidecar_path(manifest)
    _ensure_sidecar_layout(sidecar_root)
    _migrate_legacy_artifacts(manifest["id"], sidecar_root)
    manifest["sidecar_path"] = str(sidecar_root)
    sidecar_path = sidecar_root / "dataset.json"
    registry_path = MANIFESTS / f"{storage_slug(manifest['id'])}.json"
    registry = {
        "schema": "alicePD/registry-pointer/v1",
        "id": manifest["id"],
        "name": manifest["name"],
        "root_path": manifest["root_path"],
        "sidecar_path": str(sidecar_root),
        "manifest_path": str(sidecar_path),
        "created_at": manifest["created_at"],
    }
    with _lock:
        _write_json_atomic(sidecar_path, manifest)
        _write_json_atomic(registry_path, registry)
        _backfill_invalid_indices(manifest, sidecar_root)
        for key in [key for key in _detail_index_cache if key[0] == str(manifest["id"])]:
            _detail_index_cache.pop(key, None)


def manifest_registry_path(dataset_id: str) -> Path:
    preferred = MANIFESTS / f"{storage_slug(dataset_id)}.json"
    legacy = MANIFESTS / f"{slugify(dataset_id)}.json"
    for path in dict.fromkeys((preferred, legacy)):
        if not path.is_file():
            continue
        try:
            registry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(registry.get("id") or "") == str(dataset_id):
            return path
    return preferred


def get_manifest(dataset_id: str) -> dict:
    path = manifest_registry_path(dataset_id)
    if not path.exists():
        raise KeyError(dataset_id)
    registry = json.loads(path.read_text(encoding="utf-8"))
    if str(registry.get("id") or "") != str(dataset_id):
        raise KeyError(dataset_id)
    manifest_path = registry.get("manifest_path")
    if manifest_path:
        sidecar_manifest = Path(manifest_path).expanduser().resolve()
        if not sidecar_manifest.is_file():
            raise KeyError(dataset_id)
        manifest = json.loads(sidecar_manifest.read_text(encoding="utf-8"))
    else:
        manifest = registry
    if not manifest.get("episode_resolution") and manifest.get("files"):
        manifest["episode_resolution"] = build_local_episode_plan(
            manifest.get("files", []), manifest.get("episodes", [])
        )
        save_manifest(manifest)
    return manifest


def list_manifests() -> list[dict]:
    ensure_runtime()
    manifests: dict[str, dict] = {}
    for path in MANIFESTS.glob("*.json"):
        try:
            registry = json.loads(path.read_text(encoding="utf-8"))
            manifest_path = registry.get("manifest_path")
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8")) if manifest_path else registry
            summary = {
                key: manifest[key]
                for key in ("id", "name", "root_path", "created_at", "episode_count", "frame_count")
            }
            summary["file_count"] = manifest.get("file_count", len(manifest.get("files", [])))
            summary["auxiliary_file_count"] = manifest.get("auxiliary_file_count", 0)
            summary["sidecar_path"] = manifest.get("sidecar_path")
            summary["schema_status"] = manifest.get("schema_profile", {}).get("status", "not_profiled")
            current = manifests.get(str(summary["id"]))
            if current is None or str(summary.get("created_at") or "") >= str(current.get("created_at") or ""):
                manifests[str(summary["id"])] = summary
        except (KeyError, json.JSONDecodeError, OSError):
            continue
    return sorted(manifests.values(), key=lambda item: item["created_at"], reverse=True)


def get_episode(dataset_id: str, episode_id: str) -> tuple[dict, dict]:
    manifest = get_manifest(dataset_id)
    for episode in manifest["episodes"]:
        if episode["id"] == episode_id:
            return manifest, episode
    raise KeyError(episode_id)


def _manifest_detail_index(manifest: dict) -> dict:
    manifest_path = _manifest_sidecar_path(manifest) / "dataset.json"
    try:
        revision = manifest_path.stat().st_mtime_ns
    except OSError:
        revision = 0
    cache_key = (str(manifest["id"]), revision)
    with _lock:
        cached = _detail_index_cache.get(cache_key)
        if cached is not None:
            return cached
        inventory_files = (manifest.get("schema_profile") or {}).get("inventory", {}).get("files", [])
        resolution = manifest.get("episode_resolution") or {}
        groups_by_file = {}
        for group in resolution.get("groups", []):
            for file_id in group.get("file_ids", []):
                groups_by_file[str(file_id)] = group
        index = {
            "files": {str(item.get("id")): item for item in manifest.get("files", [])},
            "episodes": {str(item.get("id")): item for item in manifest.get("episodes", [])},
            "inventory": {str(item.get("path")): item for item in inventory_files},
            "groups": groups_by_file,
            "shared": set(map(str, resolution.get("shared_file_ids", []))),
            "unassigned": set(map(str, resolution.get("unassigned_file_ids", []))),
        }
        if len(_detail_index_cache) >= 12:
            _detail_index_cache.clear()
        _detail_index_cache[cache_key] = index
        return index


def get_dataset_file(dataset_id: str, file_id: str) -> tuple[dict, dict]:
    manifest = get_manifest(dataset_id)
    index = _manifest_detail_index(manifest)
    record = index["files"].get(str(file_id))
    if record is None:
        raise KeyError(file_id)
    detail = dict(record)
    profiled = index["inventory"].get(str(record["relative_path"]))
    if profiled:
        detail["fields"] = profiled.get("fields", [])
    if record.get("episode_id"):
        episode = index["episodes"].get(str(record["episode_id"]))
        if episode:
            detail["episode"] = {
                key: episode.get(key)
                for key in ("id", "name", "episode_key", "fps", "frame_count", "duration", "width", "height")
            }
    resolution = manifest.get("episode_resolution") or {}
    resolved_group = index["groups"].get(str(file_id))
    if resolved_group:
        detail["resolved_episode"] = {
            key: resolved_group.get(key)
            for key in ("group_id", "label", "source", "confidence", "evidence", "playable_episode_id")
        }
    elif str(file_id) in index["shared"]:
        detail["resolved_episode"] = {"label": "shared", "source": resolution.get("source")}
    elif str(file_id) in index["unassigned"]:
        detail["resolved_episode"] = {"label": "unassigned", "source": resolution.get("source")}
    return manifest, detail


def exclude_dataset_files(
    dataset_id: str,
    file_ids: list[str],
    reason: str = "manual_exclusion",
    scope_type: str = "file",
    scope_label: str | None = None,
) -> dict:
    manifest = get_manifest(dataset_id)
    requested = set(file_ids)
    records = {item.get("id"): item for item in manifest.get("files", [])}
    missing = sorted(requested - records.keys())
    if missing:
        raise KeyError(missing[0])
    current = _load_exclusions(manifest)
    existing_paths = {str(item.get("relative_path", "")).replace("\\", "/") for item in current.get("items", [])}
    new_items = [
        {
            "file_id": file_id,
            "relative_path": records[file_id]["relative_path"],
            "name": records[file_id].get("name"),
            "kind": records[file_id].get("kind"),
            "episode_id": records[file_id].get("episode_id"),
            "scope_type": scope_type,
            "scope_label": scope_label,
            "reason": reason.strip() or "manual_exclusion",
            "excluded_at": datetime.now(timezone.utc).isoformat(),
        }
        for file_id in sorted(requested)
        if records[file_id]["relative_path"] not in existing_paths
    ]
    if not new_items:
        return manifest
    payload = {
        "schema": EXCLUSION_SCHEMA,
        "dataset_id": dataset_id,
        "source_policy": "Excluded entries are removed from the logical dataset index; source files remain untouched.",
        "items": [*current.get("items", []), *new_items],
    }
    exclusion_path = _exclusion_path(manifest)
    _write_json_atomic(exclusion_path, payload)
    record_change(
        dataset_id,
        "dataset_exclusion",
        None,
        f"Manual {scope_type} exclusion: {scope_label or len(new_items)}",
        [exclusion_path],
        {
            "excluded_count": len(new_items),
            "total_excluded": len(payload["items"]),
            "reason": reason.strip() or "manual_exclusion",
            "scope_type": scope_type,
            "scope_label": scope_label,
        },
        [item["relative_path"] for item in new_items],
    )
    updated = apply_exclusions(manifest)
    save_manifest(updated)
    return updated


def get_dataset_file_path(dataset_id: str, file_id: str) -> tuple[dict, dict, Path]:
    manifest, detail = get_dataset_file(dataset_id, file_id)
    root = Path(manifest["root_path"]).resolve()
    path = (root / detail["relative_path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise KeyError(file_id) from exc
    if not path.is_file():
        raise KeyError(file_id)
    return manifest, detail, path


def read_frame(episode: dict, index: int):
    index = max(0, min(int(index), max(0, episode["frame_count"] - 1)))
    if episode["type"] == "images":
        return cv2.imread(episode["frames"][index])
    capture = cv2.VideoCapture(episode["path"])
    if not capture.isOpened():
        return None
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def episode_media(episode: dict, media_file_id: str | None = None) -> dict:
    if not media_file_id:
        return episode
    stream = next(
        (item for item in episode.get("media_streams", []) if item.get("file_id") == media_file_id),
        None,
    )
    if stream is None:
        raise KeyError(media_file_id)
    return {**episode, **stream, "id": episode["id"], "name": episode["name"]}


def annotation_path(dataset_id: str, episode_id: str) -> Path:
    path = dataset_artifact_dir(dataset_id, "annotations") / f"{storage_slug(episode_id)}{ALICE_ANNOTATION_EXTENSION}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_annotations(dataset_id: str, episode_id: str, payload: dict) -> None:
    path = annotation_path(dataset_id, episode_id)
    document = {
        **payload,
        "schema": ALICE_ANNOTATION_SCHEMA,
        "format": "alice",
        "format_version": 1,
        "dataset_id": dataset_id,
        "episode_id": episode_id,
    }
    with _lock:
        _write_json_atomic(path, document)
        manifest = get_manifest(dataset_id)
        sidecar_root = _manifest_sidecar_path(manifest)
        index = _write_invalid_frame_index(sidecar_root, manifest, episode_id, document)
        invalid_root = sidecar_root / "indices" / "invalid"
        source_video_path = str((document.get("source_video") or {}).get("relative_path") or "")
        fallback_path = next((item.get("relative_path", "") for item in manifest.get("episodes", []) if item.get("id") == episode_id), "")
        record_change(
            dataset_id,
            "episode_annotation",
            episode_id,
            f"Episode annotation: {episode_id}",
            [path, invalid_root / f"{storage_slug(episode_id)}.invalid.alice", invalid_root / index["bitmap"]],
            document.get("summary") or {},
            [source_video_path or fallback_path],
        )


def load_annotations(dataset_id: str, episode_id: str) -> dict | None:
    path = annotation_path(dataset_id, episode_id)
    if not path.exists():
        root = path.parent
        candidates = [
            root / f"{slugify(episode_id)}{ALICE_ANNOTATION_EXTENSION}",
            root / f"{storage_slug(episode_id)}.json",
            root / f"{slugify(episode_id)}.json",
        ]
        legacy = next((candidate for candidate in candidates if candidate.is_file()), None)
        if legacy is None:
            return None
        path = legacy
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {**payload, "schema": ALICE_ANNOTATION_SCHEMA, "format": "alice", "format_version": 1}


def load_invalid_frame_index(dataset_id: str, episode_id: str) -> dict | None:
    root = dataset_artifact_dir(dataset_id, "indices") / "invalid"
    path = root / f"{storage_slug(episode_id)}.invalid.alice"
    if not path.is_file():
        candidates = [
            root / f"{slugify(episode_id)}.invalid.alice",
            root / f"{storage_slug(episode_id)}.invalid.json",
            root / f"{slugify(episode_id)}.invalid.json",
        ]
        legacy = next((candidate for candidate in candidates if candidate.is_file()), None)
        if legacy is None:
            return None
        path = legacy
    return json.loads(path.read_text(encoding="utf-8"))


def is_frame_invalid(dataset_id: str, episode_id: str, frame: int) -> tuple[bool, dict]:
    index = load_invalid_frame_index(dataset_id, episode_id)
    if index is None:
        raise KeyError(episode_id)
    frame_count = int(index.get("frame_count", 0) or 0)
    resolved = max(0, min(int(frame), max(0, frame_count - 1)))
    bitmap_path = dataset_artifact_dir(dataset_id, "indices") / "invalid" / index["bitmap"]
    with bitmap_path.open("rb") as source:
        header = source.read(INVALID_BITMAP_HEADER_BYTES)
        if len(header) != INVALID_BITMAP_HEADER_BYTES or not header.startswith(INVALID_BITMAP_MAGIC):
            raise ValueError("无效帧 bitmap 格式损坏")
        source.seek(INVALID_BITMAP_HEADER_BYTES + (resolved >> 3))
        value = source.read(1)
    invalid = bool(value and (value[0] & (1 << (resolved & 7))))
    return invalid, {"frame": resolved, "frame_count": frame_count, "index": index}


def export_dataset(dataset_id: str, destination: Path, include_media: bool = False) -> Path:
    manifest = get_manifest(dataset_id)
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    export_root = destination / slugify(manifest["name"])
    export_root.mkdir(parents=True, exist_ok=True)

    clean_manifest = {**manifest}
    clean_manifest["episodes"] = [
        {key: value for key, value in episode.items() if key not in {"path", "frames"}}
        for episode in manifest["episodes"]
    ]
    (export_root / "manifest.json").write_text(
        json.dumps(clean_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sidecar_root = _manifest_sidecar_path(manifest)
    current_path = sidecar_root / "changes" / "current.alice"
    if current_path.is_file():
        try:
            current = json.loads(current_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
        for entry in current.get("entries", {}).values():
            for artifact in entry.get("artifacts", []):
                snapshot = (sidecar_root / artifact.get("snapshot_path", "")).resolve()
                try:
                    snapshot.relative_to(sidecar_root)
                except ValueError:
                    continue
                if not snapshot.is_file():
                    continue
                target = export_root / artifact["relative_path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(snapshot, target)
        change_target = export_root / "changes"
        change_target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current_path, change_target / "current.alice")

    if include_media:
        media_target = export_root / "media"
        media_target.mkdir(exist_ok=True)
        source_root = Path(manifest["root_path"])
        export_extensions = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS | {
            ".json", ".jsonl", ".parquet", ".csv", ".h5", ".hdf5", ".h5df", ".npz", ".npy"
        }
        for record in manifest.get("files", []):
            source = source_root / record["relative_path"]
            if not source.is_file() or source.suffix.lower() not in export_extensions:
                continue
            target = media_target / source.relative_to(source_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return export_root


def export_zip(dataset_id: str, include_media: bool = False) -> Path:
    ensure_runtime()
    staging = RUNTIME / "export-staging"
    export_root = export_dataset(dataset_id, staging, include_media=include_media)
    zip_path = dataset_artifact_dir(dataset_id, "exports") / f"{slugify(get_manifest(dataset_id)['name'])}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in export_root.rglob("*"):
            if item.is_file():
                archive.write(item, item.relative_to(export_root.parent))
    shutil.rmtree(export_root, ignore_errors=True)
    return zip_path


ensure_runtime()
