from __future__ import annotations

import hashlib
import json
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .storage import dataset_artifact_dir, storage_slug


FULL_RUN_SCHEMA = "alice/full-run/v1"
FULL_TIMELINE_SCHEMA = "alice/full-timeline-lock/v1"
_RUN_LOCK = threading.RLock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def full_run_root(dataset_id: str, run_id: str) -> Path:
    return dataset_artifact_dir(dataset_id, "full-runs") / storage_slug(run_id)


def full_run_manifest_path(dataset_id: str, run_id: str) -> Path:
    return full_run_root(dataset_id, run_id) / "run.alice"


def full_run_episode_dir(dataset_id: str, run_id: str, episode_id: str) -> Path:
    path = full_run_root(dataset_id, run_id) / "episodes" / storage_slug(episode_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def full_run_stage_dir(dataset_id: str, run_id: str, episode_id: str, stage: str) -> Path:
    path = full_run_episode_dir(dataset_id, run_id, episode_id) / storage_slug(stage)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _file_identity(path_value: Any) -> dict | None:
    path = Path(str(path_value or "")).expanduser()
    try:
        resolved = path.resolve()
        stat = resolved.stat()
    except (OSError, ValueError):
        return None
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "modified_ns": int(stat.st_mtime_ns),
    }


def _media_descriptor(media: dict) -> dict:
    positions = [float(value) for value in media.get("source_frame_positions") or []]
    positions_digest = hashlib.sha256(
        json.dumps(positions, separators=(",", ":")).encode("utf-8")
    ).hexdigest() if positions else None
    return {
        "file_id": media.get("file_id"),
        "stream_name": media.get("stream_name"),
        "relative_path": media.get("relative_path"),
        "frame_count": int(media.get("frame_count") or 0),
        "fps": float(media.get("fps") or 0.0),
        "width": int(media.get("width") or 0),
        "height": int(media.get("height") or 0),
        "duration": float(media.get("duration") or 0.0),
        "file_identity": _file_identity(media.get("path")),
        "source_frame_positions_count": len(positions),
        "source_frame_positions_sha256": positions_digest,
    }


def start_full_run(
    dataset_id: str,
    run_id: str,
    episode_ids: list[str],
    request_payload: dict,
) -> dict:
    manifest_path = full_run_manifest_path(dataset_id, run_id)
    if manifest_path.exists():
        raise FileExistsError(f"Full run already exists and cannot be overwritten: {run_id}")
    created_at = _utc_now()
    document = {
        "schema": FULL_RUN_SCHEMA,
        "run_id": run_id,
        "dataset_id": dataset_id,
        "operation": "full_pipeline",
        "status": "running",
        "created_at": created_at,
        "updated_at": created_at,
        "request": deepcopy(request_payload),
        "episode_order": list(episode_ids),
        "episodes": {
            episode_id: {
                "episode_id": episode_id,
                "status": "queued",
                "artifacts": {},
            }
            for episode_id in episode_ids
        },
    }
    _write_json_atomic(manifest_path, document)
    return document


def load_full_run(dataset_id: str, run_id: str) -> dict | None:
    payload = _read_json(full_run_manifest_path(dataset_id, run_id))
    if payload is None or payload.get("schema") != FULL_RUN_SCHEMA:
        return None
    if str(payload.get("dataset_id") or "") != str(dataset_id) or str(payload.get("run_id") or "") != str(run_id):
        return None
    return payload


def update_full_run_episode(
    dataset_id: str,
    run_id: str,
    episode_id: str,
    *,
    status: str | None = None,
    media_file_id: str | None = None,
    timeline: dict | None = None,
    artifacts: dict | None = None,
    summary: dict | None = None,
    error: str | None = None,
) -> dict:
    with _RUN_LOCK:
        document = load_full_run(dataset_id, run_id)
        if document is None:
            raise RuntimeError(f"Full run manifest is missing: {run_id}")
        entry = (document.setdefault("episodes", {})).setdefault(episode_id, {
            "episode_id": episode_id,
            "status": "queued",
            "artifacts": {},
        })
        if status is not None:
            entry["status"] = status
        if media_file_id is not None:
            entry["media_file_id"] = media_file_id
        if timeline is not None:
            entry["timeline"] = deepcopy(timeline)
            entry["timeline_id"] = timeline.get("timeline_id")
        if artifacts:
            entry.setdefault("artifacts", {}).update(deepcopy(artifacts))
        if summary is not None:
            entry["summary"] = deepcopy(summary)
        if error is not None:
            entry["error"] = str(error)
        entry["updated_at"] = _utc_now()
        document["updated_at"] = entry["updated_at"]
        _write_json_atomic(full_run_manifest_path(dataset_id, run_id), document)
        return deepcopy(entry)


def finalize_full_run(dataset_id: str, run_id: str, status: str, summary: dict) -> dict:
    with _RUN_LOCK:
        document = load_full_run(dataset_id, run_id)
        if document is None:
            raise RuntimeError(f"Full run manifest is missing: {run_id}")
        document["status"] = status
        document["summary"] = deepcopy(summary)
        document["updated_at"] = _utc_now()
        _write_json_atomic(full_run_manifest_path(dataset_id, run_id), document)
        return document


def write_full_timeline_lock(
    dataset_id: str,
    run_id: str,
    episode: dict,
    input_media: dict,
    analysis_media: dict,
    *,
    smoothing: dict | None = None,
    projection: dict | None = None,
) -> dict:
    core = {
        "schema": FULL_TIMELINE_SCHEMA,
        "run_id": run_id,
        "dataset_id": dataset_id,
        "episode_id": str(episode.get("id") or ""),
        "frame_space": "full_analysis_video",
        "input_video": _media_descriptor(input_media),
        "analysis_video": _media_descriptor(analysis_media),
        "logical_frame_count": int(analysis_media.get("frame_count") or 0),
        "fps": float(analysis_media.get("fps") or 0.0),
        "projection_application_id": (projection or {}).get("application_id"),
        "projection_retimed": bool((projection or {}).get("retiming")),
        "smoothing_artifact": str((smoothing or {}).get("artifact_path") or "") or None,
    }
    serialized = json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    document = {
        **core,
        "timeline_id": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "created_at": _utc_now(),
        "locked": True,
    }
    path = full_run_episode_dir(dataset_id, run_id, str(episode.get("id") or "")) / "timeline.alice"
    _write_json_atomic(path, document)
    document["artifact_path"] = str(path)
    return document


def stamp_full_run_artifact(payload: dict, run_id: str, timeline_id: str, stage: str) -> dict:
    stamped = deepcopy(payload)
    stamped["full_run_id"] = run_id
    stamped["timeline_id"] = timeline_id
    stamped["full_run_stage"] = stage
    return stamped


def write_stamped_artifact(path: str | Path, payload: dict, run_id: str, timeline_id: str, stage: str) -> dict:
    stamped = stamp_full_run_artifact(payload, run_id, timeline_id, stage)
    _write_json_atomic(Path(path), stamped)
    stamped["artifact_path"] = str(Path(path))
    return stamped


def artifact_record(dataset_id: str, run_id: str, path_value: str | Path, **metadata: Any) -> dict:
    root = full_run_root(dataset_id, run_id).resolve()
    path = Path(path_value).expanduser().resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Full run artifact must stay inside its immutable run directory") from exc
    return {"path": relative, **metadata}


def _resolve_artifact_path(dataset_id: str, run_id: str, record: dict | None) -> Path | None:
    relative = str((record or {}).get("path") or "")
    if not relative:
        return None
    root = full_run_root(dataset_id, run_id).resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    return path if path.is_file() else None


def publish_full_run_episode(dataset_id: str, run_id: str, episode_id: str, media_file_id: str | None) -> None:
    run = load_full_run(dataset_id, run_id)
    entry = ((run or {}).get("episodes") or {}).get(episode_id) or {}
    if entry.get("status") not in {"completed", "partial"}:
        raise RuntimeError("Only completed or partial Full episodes can become the latest published run")
    if media_file_id and entry.get("media_file_id") and str(entry.get("media_file_id")) != str(media_file_id):
        raise RuntimeError("Full run media identity changed before publication")
    root = dataset_artifact_dir(dataset_id, "full-runs") / "latest"
    media_suffix = f"--media-{storage_slug(media_file_id)}" if media_file_id else ""
    pointer = root / f"{storage_slug(episode_id)}{media_suffix}.alice"
    _write_json_atomic(pointer, {
        "schema": "alice/full-run-pointer/v1",
        "dataset_id": dataset_id,
        "episode_id": episode_id,
        "media_file_id": media_file_id,
        "run_id": run_id,
        "updated_at": _utc_now(),
    })


def latest_full_run_id(dataset_id: str, episode_id: str, media_file_id: str | None = None) -> str | None:
    root = dataset_artifact_dir(dataset_id, "full-runs")
    pointer_names = []
    if media_file_id:
        pointer_names.append(f"{storage_slug(episode_id)}--media-{storage_slug(media_file_id)}.alice")
    pointer_names.append(f"{storage_slug(episode_id)}.alice")
    for name in pointer_names:
        payload = _read_json(root / "latest" / name)
        if payload and str(payload.get("episode_id") or "") == str(episode_id):
            run_id = str(payload.get("run_id") or "")
            run = load_full_run(dataset_id, run_id) if run_id else None
            entry = ((run or {}).get("episodes") or {}).get(episode_id) or {}
            media_matches = not media_file_id or not entry.get("media_file_id") or str(entry.get("media_file_id")) == str(media_file_id)
            if run_id and entry.get("status") in {"completed", "partial"} and media_matches:
                return run_id
    candidates: list[tuple[str, str]] = []
    for manifest_path in root.glob("*/run.alice"):
        payload = _read_json(manifest_path)
        entry = ((payload or {}).get("episodes") or {}).get(episode_id) or {}
        if not payload or entry.get("status") not in {"completed", "partial"}:
            continue
        if media_file_id and entry.get("media_file_id") and str(entry.get("media_file_id")) != str(media_file_id):
            continue
        candidates.append((str(entry.get("updated_at") or payload.get("updated_at") or ""), str(payload.get("run_id") or "")))
    return max(candidates, default=("", ""))[1] or None


def load_full_run_episode_bundle(
    dataset_id: str,
    episode_id: str,
    media_file_id: str | None = None,
    run_id: str | None = None,
) -> dict | None:
    resolved_run_id = run_id or latest_full_run_id(dataset_id, episode_id, media_file_id)
    if not resolved_run_id:
        return None
    run = load_full_run(dataset_id, resolved_run_id)
    entry = ((run or {}).get("episodes") or {}).get(episode_id) or {}
    if not run or entry.get("status") not in {"completed", "partial"}:
        return None
    if media_file_id and entry.get("media_file_id") and str(entry.get("media_file_id")) != str(media_file_id):
        return None
    timeline = deepcopy(entry.get("timeline") or {})
    if not str(timeline.get("timeline_id") or ""):
        return None
    artifacts = entry.get("artifacts") or {}

    def load_document(name: str) -> dict | None:
        path = _resolve_artifact_path(dataset_id, resolved_run_id, artifacts.get(name))
        payload = _read_json(path) if path is not None else None
        if payload is None:
            return None
        if str(payload.get("full_run_id") or "") != resolved_run_id:
            return None
        if str(payload.get("timeline_id") or "") != str(entry.get("timeline_id") or ""):
            return None
        payload["artifact_path"] = str(path)
        return payload

    smoothing_record = artifacts.get("smoothing") or {}
    smoothing_document = load_document("smoothing")
    smoothing_video = _resolve_artifact_path(dataset_id, resolved_run_id, smoothing_record.get("video"))
    curation_document = load_document("curation")
    if smoothing_document is None or smoothing_video is None or curation_document is None:
        return None
    return {
        "schema": "alice/full-run-view/v1",
        "run_id": resolved_run_id,
        "dataset_id": dataset_id,
        "episode_id": episode_id,
        "media_file_id": entry.get("media_file_id"),
        "status": entry.get("status"),
        "timeline": timeline,
        "smoothing": smoothing_document,
        "smoothing_video": str(smoothing_video) if smoothing_video is not None else None,
        "curation": curation_document,
        "behavior": load_document("behavior"),
        "export": deepcopy(entry.get("export") or artifacts.get("export") or {}),
        "summary": deepcopy(entry.get("summary") or {}),
    }


def full_run_review_media(manifest: dict, episode: dict, media: dict, run_id: str | None) -> tuple[dict, dict | None]:
    if not run_id:
        return media, None
    bundle = load_full_run_episode_bundle(
        str(manifest.get("id") or ""),
        str(episode.get("id") or ""),
        str(media.get("file_id") or "") or None,
        run_id,
    )
    if bundle is None or not bundle.get("smoothing_video"):
        raise RuntimeError(f"Full run is incomplete or no longer readable: {run_id}")
    timeline = bundle.get("timeline") or {}
    analysis = timeline.get("analysis_video") or {}
    return {
        **media,
        "path": str(bundle["smoothing_video"]),
        "frame_count": int(timeline.get("logical_frame_count") or analysis.get("frame_count") or media.get("frame_count") or 0),
        "fps": float(timeline.get("fps") or analysis.get("fps") or media.get("fps") or 30.0),
        "width": int(analysis.get("width") or media.get("width") or 0),
        "height": int(analysis.get("height") or media.get("height") or 0),
        "duration": float(timeline.get("logical_frame_count") or 0) / max(0.01, float(timeline.get("fps") or 30.0)),
        "preview_variant": "full-run-smoothing",
        "full_run_id": run_id,
        "timeline_id": timeline.get("timeline_id"),
    }, bundle
