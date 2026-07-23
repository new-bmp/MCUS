from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


# Capture timestamped recorder folders such as ep_0001_20260710_141603 as one
# explicit episode boundary. The suffix is metadata, not another episode ID.
_EPISODE_PART = re.compile(r"^(?:ep|episode)[_-]?(\d+)(?:[_-].*)?$", re.IGNORECASE)
_EPISODE_STEM = re.compile(r"(?:^|[_-])(?:ep|episode)[_-]?(\d+)(?:$|[_-])", re.IGNORECASE)
_MODALITY_PARTS = {
    "camera", "cameras", "video", "videos", "rgb", "head", "head_rgb", "wrist",
    "left", "right", "tactile", "pressure", "sensor", "sensors", "state", "action",
    "joints", "pose", "poses", "mocap", "transforms", "metadata", "meta",
}
_STEM_NOISE = re.compile(
    r"(?:^|[_-])(left|right|head|wrist|rgb|camera|cam|video|tactile|pressure|state|action|joints?|pose|mocap|transforms?)(?=$|[_-])",
    re.IGNORECASE,
)


def _natural_key(value: str) -> list[tuple[int, object]]:
    return [
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", value)
        if part
    ]


def _relative(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.parent.name) / path.name


def episode_token(path: Path, root: Path) -> str | None:
    relative = _relative(path, root)
    for part in relative.parts[:-1]:
        match = _EPISODE_PART.fullmatch(part)
        if match:
            return str(int(match.group(1)))
    stem = path.stem
    if stem.isdigit():
        return str(int(stem))
    match = _EPISODE_STEM.search(stem)
    if match:
        return str(int(match.group(1)))
    digits = re.findall(r"\d+", _STEM_NOISE.sub("_", stem))
    return str(int(digits[0])) if len(digits) == 1 else None


def episode_key(path: Path, root: Path) -> str:
    relative = _relative(path, root)
    for index, part in enumerate(relative.parts[:-1]):
        if _EPISODE_PART.fullmatch(part):
            return Path(*relative.parts[: index + 1]).as_posix()

    token = episode_token(path, root)
    # LeRobot stores the same episode in separate data/ and videos/ trees.
    # An explicit episode_N filename is therefore a dataset-wide boundary.
    if token is not None and _EPISODE_PART.fullmatch(path.stem) and any(
        part.casefold() in {"data", "videos"} for part in relative.parts[:-1]
    ):
        return f"episode_{token}"
    if token is not None:
        parents = list(relative.parts[:-1])
        while parents and parents[-1].lower() in _MODALITY_PARTS:
            parents.pop()
        prefix = Path(*parents).as_posix() if parents else "."
        return f"{prefix}/episode_{token}" if prefix != "." else f"episode_{token}"

    return relative.with_suffix("").as_posix()


def build_episode_framework(files: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for record in files:
        relative = str(record.get("relative_path", ""))
        path = Path(relative)
        candidates.append({
            "file_id": str(record.get("id", "")),
            "path": relative,
            "kind": str(record.get("kind", "file")),
            "category": str(record.get("category", "other")),
            "episode_token": record.get("episode_token"),
            "deterministic_group": str(record.get("episode_key", relative)),
            "playable_episode_id": record.get("episode_id"),
            "native_numeric_episode": bool(
                path.stem.isdigit() and path.suffix.lower() in {".mp4", ".h5", ".hdf5", ".h5df"}
            ),
            "size": int(record.get("size_bytes", 0) or 0),
        })
    fingerprint = hashlib.sha256(
        "\n".join(sorted(item["path"] for item in candidates)).encode("utf-8")
    ).hexdigest()[:20]
    return {
        "schema_version": 1,
        "dataset_fingerprint": fingerprint,
        "rules": [
            "Use only supplied file_id values; never invent a path or ID.",
            "Never merge different non-null episode_token values.",
            "Matching task scope and numeric episode are strong synchronization evidence.",
            "Multi-camera, left/right hand, joint, tactile and action files may share an episode only with path/token evidence.",
            "Dataset-level metadata belongs in shared_file_ids, not an episode.",
            "Put uncertain files in unassigned_file_ids instead of guessing.",
        ],
        "output_schema": {
            "groups": [{
                "label": "short episode label",
                "file_ids": ["provided file_id"],
                "confidence": "0.0-1.0",
                "evidence": "short path/token evidence",
            }],
            "shared_file_ids": ["provided file_id"],
            "unassigned_file_ids": ["provided file_id"],
            "warnings": ["string"],
        },
        "files": candidates,
    }


def _episode_label(key: str, episodes: list[dict[str, Any]], playable_id: str | None = None) -> str:
    if playable_id:
        episode = next((item for item in episodes if item.get("id") == playable_id), None)
        if episode:
            return str(episode.get("name") or Path(key).name)
    return Path(key).name or key


def build_local_episode_plan(
    files: list[dict[str, Any]], episodes: list[dict[str, Any]]
) -> dict[str, Any]:
    framework = build_episode_framework(files)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    shared_ids: list[str] = []
    unassigned_ids: list[str] = []
    for record in files:
        if record.get("episode_id") or record.get("episode_token") is not None:
            grouped[str(record.get("episode_key"))].append(record)
        elif record.get("category") == "metadata" or record.get("kind") == "text":
            shared_ids.append(str(record["id"]))
        else:
            unassigned_ids.append(str(record["id"]))

    groups = []
    file_episode_assignments: dict[str, str] = {}
    for index, (key, records) in enumerate(sorted(grouped.items(), key=lambda item: _natural_key(item[0]))):
        playable = next((str(item["episode_id"]) for item in records if item.get("episode_id")), None)
        group_id = f"local:{index + 1}:{key}"
        for record in records:
            if playable:
                file_episode_assignments[str(record["id"])] = playable
        groups.append({
            "group_id": group_id,
            "label": _episode_label(key, episodes, playable),
            "file_ids": [str(item["id"]) for item in records],
            "episode_token": next((item.get("episode_token") for item in records if item.get("episode_token") is not None), None),
            "playable_episode_id": playable,
            "confidence": 1.0,
            "evidence": "Deterministic directory, filename, or decoded visual-stream boundary.",
            "source": "deterministic",
        })

    requires_api = bool(unassigned_ids)
    warnings = []
    if requires_api:
        warnings.append(
            f"{len(unassigned_ids)} files have no deterministic episode evidence; configure Qwen API to audit membership."
        )
    return {
        "schema_version": 1,
        "dataset_fingerprint": framework["dataset_fingerprint"],
        "status": "awaiting_qwen" if requires_api else "local_complete",
        "source": "deterministic",
        "model": None,
        "ai_confirmed": False,
        "requires_api": requires_api,
        "groups": groups,
        "shared_file_ids": shared_ids,
        "unassigned_file_ids": unassigned_ids,
        "file_episode_assignments": file_episode_assignments,
        "warnings": warnings,
    }


def validate_qwen_episode_plan(
    raw: Any,
    framework: dict[str, Any],
    files: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    model: str | None,
) -> dict[str, Any]:
    candidates = {item["file_id"]: item for item in framework.get("files", [])}
    records = {str(item["id"]): item for item in files}
    assigned: set[str] = set()
    groups: list[dict[str, Any]] = []
    warnings: list[str] = []
    values = raw.get("groups", []) if isinstance(raw, dict) else []
    for index, item in enumerate(values if isinstance(values, list) else []):
        if not isinstance(item, dict):
            continue
        ids = [
            str(value) for value in item.get("file_ids", [])
            if str(value) in candidates and str(value) not in assigned
        ]
        if not ids:
            continue
        tokens = {
            candidates[file_id]["episode_token"] for file_id in ids
            if candidates[file_id]["episode_token"] is not None
        }
        if len(tokens) > 1:
            warnings.append(f"Rejected Qwen group {index}: conflicting episode tokens {sorted(tokens)}")
            continue
        native_scopes = {
            candidates[file_id]["deterministic_group"] for file_id in ids
            if candidates[file_id].get("native_numeric_episode")
        }
        if len(native_scopes) > 1:
            warnings.append(f"Rejected Qwen group {index}: numeric episodes cross task scopes")
            continue
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < 0.72:
            warnings.append(f"Rejected Qwen group {index}: confidence below 0.72")
            continue
        playable = next(
            (str(records[file_id]["episode_id"]) for file_id in ids if records[file_id].get("episode_id")),
            None,
        )
        assigned.update(ids)
        groups.append({
            "group_id": f"qwen:{index + 1}",
            "label": str(item.get("label") or next(iter(tokens), f"episode_{index + 1}"))[:120],
            "file_ids": ids,
            "episode_token": next(iter(tokens), None),
            "playable_episode_id": playable,
            "confidence": round(confidence, 4),
            "evidence": str(item.get("evidence", ""))[:500],
            "source": "qwen",
        })

    def valid_bucket(name: str) -> list[str]:
        values = raw.get(name, []) if isinstance(raw, dict) else []
        result = []
        for value in values if isinstance(values, list) else []:
            file_id = str(value)
            if file_id in candidates and file_id not in assigned:
                assigned.add(file_id)
                result.append(file_id)
        return result

    shared_ids = valid_bucket("shared_file_ids")
    unassigned_ids = valid_bucket("unassigned_file_ids")

    # Preserve only strong deterministic groups for anything Qwen omitted or had rejected.
    local = build_local_episode_plan(files, episodes)
    for group in local["groups"]:
        remaining = [file_id for file_id in group["file_ids"] if file_id not in assigned]
        if not remaining:
            continue
        assigned.update(remaining)
        fallback = dict(group)
        fallback["group_id"] = f"fallback:{len(groups) + 1}"
        fallback["file_ids"] = remaining
        fallback["source"] = "deterministic_fallback"
        groups.append(fallback)
    for file_id in local["shared_file_ids"]:
        if file_id not in assigned:
            assigned.add(file_id)
            shared_ids.append(file_id)
    for file_id in candidates:
        if file_id not in assigned:
            unassigned_ids.append(file_id)

    file_episode_assignments: dict[str, str] = {}
    for group in groups:
        playable = group.get("playable_episode_id")
        if playable:
            for file_id in group["file_ids"]:
                file_episode_assignments[file_id] = str(playable)
    if isinstance(raw, dict) and isinstance(raw.get("warnings"), list):
        warnings.extend(str(value) for value in raw["warnings"][:30])
    return {
        "schema_version": 1,
        "dataset_fingerprint": framework["dataset_fingerprint"],
        "status": "qwen_complete",
        "source": "qwen_guarded",
        "model": model,
        "ai_confirmed": True,
        "requires_api": False,
        "groups": groups,
        "shared_file_ids": shared_ids,
        "unassigned_file_ids": unassigned_ids,
        "file_episode_assignments": file_episode_assignments,
        "warnings": warnings,
    }
