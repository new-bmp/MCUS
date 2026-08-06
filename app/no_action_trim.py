from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .models import OPEN_VOCAB_HAND_CLASSES, _open_vocab_proximity_classes, normalize_yoloe_object_terms, registry
from .storage import dataset_artifact_dir, dataset_sidecar_root, get_manifest, read_frame, record_change, require_media_eligibility, slugify


NO_ACTION_TRIM_SCHEMA = "alice/no-action-trim/v1"
NO_ACTION_TRIM_ARTIFACT_VERSION = 2


def _artifact_path(dataset_id: str, episode_id: str) -> Path:
    return dataset_artifact_dir(dataset_id, "no-action-trim") / f"{slugify(episode_id)}.trim.alice"


def _target_path(dataset_id: str, episode_id: str, manifest: dict | None = None) -> Path:
    dataset = manifest or get_manifest(dataset_id)
    configured = dataset.get("sidecar_path")
    sidecar = Path(configured).expanduser().resolve() if configured else dataset_sidecar_root(dataset["root_path"], dataset["id"])
    return sidecar / "behavior-targets" / f"{slugify(episode_id)}.targets.alice"


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_object_terms(dataset_id: str, episode_id: str, manifest: dict | None = None) -> tuple[list[str], Path]:
    path = _target_path(dataset_id, episode_id, manifest)
    if not path.is_file():
        raise ValueError("请先为该 Episode 运行 VLM 行为标注，生成主要目标词文件")
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Detection must use task-relevant targets first. ``primary_terms`` is an
    # audit inventory of every noun mentioned by the VLM and often contains
    # work surfaces, monitors, cables or the robot body; treating those as
    # interaction targets makes almost any nearby hand frame look valid.
    target_terms = [
        item.get("name")
        for item in payload.get("primary_targets", [])
        if isinstance(item, dict) and item.get("name")
    ]
    terms = target_terms or payload.get("primary_terms") or []
    terms = list(dict.fromkeys(str(value).strip() for value in terms if str(value or "").strip()))
    if not terms:
        raise ValueError("主要目标词文件中没有可用物体名词")
    return terms, path


def _fill_short_false_gaps(values: list[bool], max_gap: int) -> list[bool]:
    output = list(values)
    index = 0
    while index < len(output):
        if output[index]:
            index += 1
            continue
        start = index
        while index < len(output) and not output[index]:
            index += 1
        if start > 0 and index < len(output) and index - start <= max_gap:
            output[start:index] = [True] * (index - start)
    return output


def _drop_short_true_runs(values: list[bool], minimum: int) -> list[bool]:
    output = list(values)
    index = 0
    while index < len(output):
        if not output[index]:
            index += 1
            continue
        start = index
        while index < len(output) and output[index]:
            index += 1
        if index - start < minimum:
            output[start:index] = [False] * (index - start)
    return output


def _segments(samples: list[dict], frame_count: int, fps: float) -> list[dict]:
    if not samples:
        return []
    output = []
    start = 0
    for index in range(1, len(samples) + 1):
        if index < len(samples) and samples[index]["state"] == samples[start]["state"]:
            continue
        start_frame = int(samples[start]["frame"])
        end_frame = int(samples[index]["frame"] - 1) if index < len(samples) else max(0, frame_count - 1)
        group = samples[start:index]
        valid = samples[start]["state"] == "valid"
        output.append({
            "start_frame": start_frame,
            "end_frame": max(start_frame, end_frame),
            "start_time": round(start_frame / max(0.01, fps), 3),
            "end_time": round(max(start_frame, end_frame) / max(0.01, fps), 3),
            "state": "valid" if valid else "invalid",
            "confidence": round(sum(float(item.get("confidence", 0.0)) for item in group) / max(1, len(group)), 4),
            "reason": "目标物体与手/夹爪持续接近" if valid else "未检测到目标物体与手/夹爪接近",
        })
        start = index
    return output


def analyze_no_action_trim(
    dataset_id: str,
    manifest: dict,
    episode: dict,
    progress,
    media: dict,
    sample_fps: float = 4.0,
    proximity_threshold: float = 0.04,
    max_gap_seconds: float = 0.5,
    min_valid_seconds: float = 0.3,
) -> dict:
    require_media_eligibility(media, "no_action_trim")
    terms, terms_path = _load_object_terms(dataset_id, episode["id"])
    hand_keys = {value.casefold() for value in OPEN_VOCAB_HAND_CLASSES}
    detector_terms = [value for value in normalize_yoloe_object_terms(terms) if value.casefold() not in hand_keys]
    if not detector_terms:
        raise ValueError("主要目标词中没有可供 YOLOE 检测的物体名词")
    if not registry.has_local or registry.status().get("local", {}).get("family") != "YOLOE":
        raise RuntimeError("无动作剪切需要已加载的 YOLOE26X 分割模型")
    fps = float(media.get("fps", 30.0) or 30.0)
    frame_count = int(media.get("frame_count", 0) or 0)
    stride = max(1, int(round(fps / max(0.25, sample_fps))))
    indices = list(range(0, frame_count, stride))
    if indices and indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    samples = []
    progress(5, f"加载 {len(detector_terms)} 个归一化物体名词")
    for position, frame_index in enumerate(indices):
        frame = read_frame(media, frame_index)
        if frame is None:
            proximity = {"detections": [], "hand_count": 0, "object_count": 0, "nearest": None, "close": False}
        else:
            proximity = registry.infer_open_vocab_proximity(frame, detector_terms, proximity_threshold)
        nearest = proximity.get("nearest") or {}
        confidence = float(nearest.get("confidence", 0.0) or 0.0)
        samples.append({
            "frame": frame_index,
            "time": round(frame_index / max(0.01, fps), 3),
            "raw_valid": bool(proximity.get("close")),
            "state": "valid" if proximity.get("close") else "invalid",
            "confidence": round(confidence if proximity.get("close") else max(0.5, 1.0 - confidence), 4),
            "reason": "目标物体与手/夹爪接近" if proximity.get("close") else "物体与手/夹爪距离超过阈值或发生漏检",
            "distance": nearest,
            "hand_count": proximity.get("hand_count", 0),
            "object_count": proximity.get("object_count", 0),
            "detections": proximity.get("detections", []),
            "inference": proximity.get("inference"),
        })
        progress(8 + 82 * (position + 1) / max(1, len(indices)), f"YOLOE 开放词汇距离分析 {position + 1}/{len(indices)}")
    raw_values = [bool(item["raw_valid"]) for item in samples]
    max_gap_samples = max(0, int(round(max_gap_seconds * sample_fps)))
    minimum_valid_samples = max(1, int(round(min_valid_seconds * sample_fps)))
    filtered = _drop_short_true_runs(_fill_short_false_gaps(raw_values, max_gap_samples), minimum_valid_samples)
    for sample, valid in zip(samples, filtered):
        sample["state"] = "valid" if valid else "invalid"
        sample["filtered_gap_fill"] = bool(valid and not sample["raw_valid"])
        if sample["filtered_gap_fill"]:
            sample["reason"] = "时间滤波填补短暂单帧漏检"
    segments = _segments(samples, frame_count, fps)
    valid_frames = sum(item["end_frame"] - item["start_frame"] + 1 for item in segments if item["state"] == "valid")
    document = {
        "schema": NO_ACTION_TRIM_SCHEMA,
        "artifact_version": NO_ACTION_TRIM_ARTIFACT_VERSION,
        "dataset_id": dataset_id,
        "episode_id": episode["id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_policy": "Source media remains read-only; this file records proposed valid/invalid intervals.",
        "distance_method": "image_plane_proximity",
        "distance_limitations": "Metric 3D distance is not claimed without aligned depth and camera calibration.",
        "model": registry.status().get("local"),
        "source_video": {
            "file_id": media.get("file_id"),
            "stream_name": media.get("stream_name"),
            "relative_path": media.get("relative_path"),
            "frame_count": frame_count,
            "fps": fps,
            "width": media.get("width"),
            "height": media.get("height"),
        },
        "primary_terms": terms,
        "detector_terms": detector_terms,
        "prompt_classes": _open_vocab_proximity_classes(detector_terms),
        "prompt_policy": {
            "source": "primary_targets_then_primary_terms_fallback",
            "normalization": "core_object_nouns_without_color_quantity_size_material_position_or_action_modifiers",
            "fixed_effector_classes": OPEN_VOCAB_HAND_CLASSES,
        },
        "primary_terms_file": str(terms_path),
        "config": {
            "sample_fps": sample_fps,
            "stride": stride,
            "proximity_threshold": proximity_threshold,
            "max_gap_seconds": max_gap_seconds,
            "max_gap_samples": max_gap_samples,
            "min_valid_seconds": min_valid_seconds,
            "minimum_valid_samples": minimum_valid_samples,
        },
        "summary": {
            "sample_count": len(samples),
            "segment_count": len(segments),
            "invalid_count": sum(item["state"] == "invalid" for item in segments),
            "valid_frame_count": valid_frames,
            "invalid_frame_count": max(0, frame_count - valid_frames),
            "gap_filled_sample_count": sum(bool(item.get("filtered_gap_fill")) for item in samples),
        },
        "segments": segments,
        "samples": samples,
    }
    path = _artifact_path(dataset_id, episode["id"])
    _write_atomic(path, document)
    change = record_change(
        dataset_id,
        "no_action_trim",
        episode["id"],
        f"No-action trim: {episode['name']}",
        [path],
        document["summary"],
        [str(media.get("relative_path") or "")],
    )
    document["artifact_path"] = str(path)
    document["change"] = {"id": change["id"], "status": change["status"], "revision": change["revision"]}
    progress(98, "写入 .alicePD 无动作剪切记录")
    return document


def load_no_action_trim(dataset_id: str, episode_id: str) -> dict | None:
    path = _artifact_path(dataset_id, episode_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
