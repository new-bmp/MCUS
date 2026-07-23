from __future__ import annotations

import json
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .models import registry
from .schemas import AnalysisRequest
from .storage import ALICE_ANNOTATION_SCHEMA, dataset_cache_dir, get_episode, get_manifest, read_frame, save_annotations


class JobManager:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vla-analysis")
        self._jobs: dict[str, dict] = {}
        self._lock = threading.RLock()

    def submit(self, dataset_id: str, episode_id: str, config: AnalysisRequest) -> dict:
        schema_status = get_manifest(dataset_id).get("schema_profile", {}).get("status")
        if schema_status != "completed":
            raise RuntimeError("请先使用 Qwen-VLM 完成数据集结构理解和 vision/joint 映射")
        if not registry.has_local and not (config.use_vlm and registry.has_vlm):
            raise RuntimeError("请先加载 SAM/YOLO 模型，或配置并启用 Qwen-VLM")
        job_id = uuid.uuid4().hex
        job = {"id": job_id, "dataset_id": dataset_id, "episode_id": episode_id, "state": "queued", "progress": 0.0, "message": "等待处理", "result": None, "error": None}
        with self._lock:
            self._jobs[job_id] = job
        self._executor.submit(self._run, job_id, dataset_id, episode_id, config)
        return dict(job)

    def get(self, job_id: str) -> dict:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return dict(self._jobs[job_id])

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            self._jobs[job_id].update(changes)

    def _run(self, job_id: str, dataset_id: str, episode_id: str, config: AnalysisRequest) -> None:
        try:
            self._update(job_id, state="running", progress=0.01, message="正在读取 Episode")
            manifest, episode = get_episode(dataset_id, episode_id)
            result = analyze_episode(dataset_id, episode, config, lambda progress, message: self._update(job_id, progress=progress, message=message), manifest.get("schema_profile"))
            self._update(job_id, state="completed", progress=1.0, message="分析完成", result=result)
        except Exception as exc:
            self._update(job_id, state="failed", message="分析失败", error=str(exc), trace=traceback.format_exc(limit=5))


def _motion_features(previous: np.ndarray | None, frame: np.ndarray) -> tuple[float, list[int] | None, np.ndarray]:
    small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    if previous is None:
        return 0.0, None, gray
    diff = cv2.absdiff(previous, gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, binary = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    motion_score = float(np.count_nonzero(binary) / binary.size)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= 30]
    if not contours:
        return min(1.0, motion_score * 4.0), None, gray
    x1 = min(cv2.boundingRect(contour)[0] for contour in contours)
    y1 = min(cv2.boundingRect(contour)[1] for contour in contours)
    x2 = max(cv2.boundingRect(contour)[0] + cv2.boundingRect(contour)[2] for contour in contours)
    y2 = max(cv2.boundingRect(contour)[1] + cv2.boundingRect(contour)[3] for contour in contours)
    height, width = frame.shape[:2]
    box = [int(x1 * width / 320), int(y1 * height / 180), int(x2 * width / 320), int(y2 * height / 180)]
    return min(1.0, motion_score * 4.0), box, gray


def _draw_overlay(frame: np.ndarray, sample: dict) -> np.ndarray:
    output = frame.copy()
    state = sample["state"]
    color = (71, 184, 151) if state == "valid" else ((61, 170, 220) if state == "uncertain" else (72, 74, 220))
    for detection in sample.get("detections", []):
        x1, y1, x2, y2 = [int(value) for value in detection["box"]]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{detection['label']} {detection['confidence']:.2f}"
        cv2.putText(output, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.rectangle(output, (0, 0), (output.shape[1], 34), (18, 21, 25), -1)
    cv2.putText(output, f"{state.upper()}  motion {sample['motion']:.2f}  contact {sample['contact']:.2f}", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def _group_samples(samples: list[dict], fps: float, idle_duration: float) -> list[dict]:
    if not samples:
        return []
    groups: list[list[dict]] = [[samples[0]]]
    for sample in samples[1:]:
        if sample["state"] == groups[-1][-1]["state"]:
            groups[-1].append(sample)
        else:
            groups.append([sample])

    segments = []
    for group in groups:
        start = group[0]["frame"]
        end = group[-1]["frame"]
        state = group[0]["state"]
        duration = (end - start + 1) / max(0.01, fps)
        if state == "invalid" and duration < idle_duration:
            state = "uncertain"
        reasons = [item["reason"] for item in group]
        reason = max(set(reasons), key=reasons.count)
        segments.append({
            "start_frame": start,
            "end_frame": end,
            "start_time": round(start / fps, 3),
            "end_time": round(end / fps, 3),
            "state": state,
            "confidence": round(float(np.mean([item["confidence"] for item in group])), 4),
            "reason": reason,
            "motion": round(float(np.mean([item["motion"] for item in group])), 4),
            "contact": round(float(np.mean([item["contact"] for item in group])), 4),
        })
    return segments


def analyze_episode(dataset_id: str, episode: dict, config: AnalysisRequest, progress, schema_profile: dict | None = None) -> dict:
    fps = float(episode["fps"])
    stride = max(1, int(round(fps / config.sample_fps)))
    indices = list(range(0, episode["frame_count"], stride))
    if indices and indices[-1] != episode["frame_count"] - 1:
        indices.append(episode["frame_count"] - 1)
    samples: list[dict] = []
    frames_for_vlm: list[tuple[int, np.ndarray]] = []
    previous_gray = None
    cache_dir = dataset_cache_dir(dataset_id, episode["id"])

    for position, frame_index in enumerate(indices):
        frame = read_frame(episode, frame_index)
        if frame is None:
            continue
        motion, motion_box, previous_gray = _motion_features(previous_gray, frame)
        local = {"detections": [], "mask_coverage": 0.0, "interaction": 0.0}
        if registry.has_local:
            local = registry.infer_local(frame, motion_box)
        contact = float(local.get("interaction", 0.0))
        if registry.has_local:
            valid = motion >= config.motion_threshold and (contact >= config.contact_threshold or (registry.status()["local"]["kind"] == "sam" and local.get("mask_coverage", 0) > 0.01))
            confidence = min(0.99, 0.35 + motion * 0.35 + max(contact, local.get("mask_coverage", 0.0)) * 0.3)
        else:
            valid = motion >= config.motion_threshold
            confidence = 0.55 + min(0.25, motion * 0.25)
        state = "valid" if valid else "invalid"
        reason = "持续运动并与目标区域交互" if valid else ("手部或目标区域静止" if motion < config.motion_threshold else "未检测到物体接触")
        sample = {
            "frame": frame_index,
            "time": round(frame_index / fps, 3),
            "state": state,
            "confidence": round(confidence, 4),
            "reason": reason,
            "motion": round(motion, 4),
            "contact": round(contact, 4),
            "mask_coverage": local.get("mask_coverage", 0.0),
            "detections": local.get("detections", []),
        }
        samples.append(sample)
        if config.use_vlm and registry.has_vlm:
            frames_for_vlm.append((frame_index, frame.copy()))
        overlay = _draw_overlay(frame, sample)
        cv2.imwrite(str(cache_dir / f"{frame_index:08d}.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 84])
        progress(0.05 + 0.68 * (position + 1) / max(1, len(indices)), f"分割与运动分析 {position + 1}/{len(indices)}")

    if config.use_vlm and registry.has_vlm and frames_for_vlm:
        window_frames = max(1, int(round(config.vlm_window_seconds * config.sample_fps)))
        windows = [frames_for_vlm[index:index + window_frames] for index in range(0, len(frames_for_vlm), window_frames)]
        for index, window in enumerate(windows):
            schema_context = json.dumps((schema_profile or {}).get("understanding") or {}, ensure_ascii=False)[:5000]
            verdict = registry.judge_frames(
                [frame for _, frame in window],
                f"Episode {episode['name']}, {window[0][0] / fps:.2f}s-{window[-1][0] / fps:.2f}s. Validated dataset schema mapping: {schema_context}",
            )
            start_frame, end_frame = window[0][0], window[-1][0]
            for sample in samples:
                if start_frame <= sample["frame"] <= end_frame:
                    sample["state"] = verdict.get("state", "uncertain")
                    sample["confidence"] = float(verdict.get("confidence", sample["confidence"]))
                    sample["reason"] = str(verdict.get("reason", "Qwen-VLM 复核"))
                    sample["motion"] = float(verdict.get("motion", sample["motion"]))
                    sample["contact"] = float(verdict.get("contact", sample["contact"]))
            progress(0.74 + 0.2 * (index + 1) / len(windows), f"Qwen-VLM 复核 {index + 1}/{len(windows)}")

    segments = _group_samples(samples, fps, config.idle_duration)
    payload = {
        "schema": ALICE_ANNOTATION_SCHEMA,
        "dataset_id": dataset_id,
        "episode_id": episode["id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_status": registry.status(),
        "config": config.model_dump(),
        "dataset_schema": (schema_profile or {}).get("understanding"),
        "summary": {
            "segment_count": len(segments),
            "invalid_count": sum(item["state"] == "invalid" for item in segments),
            "uncertain_count": sum(item["state"] == "uncertain" for item in segments),
        },
        "segments": segments,
        "samples": samples,
    }
    save_annotations(dataset_id, episode["id"], payload)
    progress(0.98, "保存标注")
    return payload


jobs = JobManager()
