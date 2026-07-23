from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from .behavior_annotator import annotate_episode_behavior, behavior_annotation_status
from .models import registry
from .no_action_trim import analyze_no_action_trim
from .pose_recovery import pose_recovery_status, recover_episode_pose
from .schemas import BatchAnalysisRequest, BehaviorAnnotationRequest
from .storage import episode_media, get_manifest
from .video_smoothing import smooth_video


class BatchAnalysisJobManager:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="alice-batch-analysis")
        self._jobs: dict[str, dict] = {}
        self._lock = threading.RLock()

    def submit(self, dataset_id: str, request: BatchAnalysisRequest) -> dict:
        manifest = get_manifest(dataset_id)
        episodes = {item["id"]: item for item in manifest.get("episodes", [])}
        episode_ids = list(dict.fromkeys(request.episode_ids))
        if not episode_ids:
            raise ValueError("At least one Episode is required")
        missing = [episode_id for episode_id in episode_ids if episode_id not in episodes]
        if missing:
            raise KeyError(missing[0])
        reusable: dict[str, dict] = {}
        if request.operation == "vlm_behavior" and not request.force:
            reusable = {
                episode_id: behavior_annotation_status(dataset_id, manifest, episodes[episode_id])
                for episode_id in episode_ids
            }
            needs_api = [episode_id for episode_id, status in reusable.items() if not status.get("reusable")]
            if needs_api and not registry.has_vlm:
                raise RuntimeError("请先配置 Qwen-VLM API；已有有效行为标注的 Episode 可直接复用")
            if not needs_api:
                job_id = uuid.uuid4().hex
                items = [self._reused_behavior_item(episodes[episode_id], reusable[episode_id]) for episode_id in episode_ids]
                result = {
                    "dataset_id": dataset_id,
                    "operation": request.operation,
                    "episode_count": len(episode_ids),
                    "completed_count": 0,
                    "skipped_count": len(items),
                    "reused_count": len(items),
                    "failure_count": 0,
                    "items": items,
                    "failures": [],
                }
                job = {
                    "id": job_id,
                    "kind": "batch_analysis",
                    "operation": request.operation,
                    "status": "complete",
                    "progress": 100,
                    "message": f"已有有效 VLM 行为标注，已跳过 {len(items)} 个 Episode",
                    "episode_count": len(items),
                    "completed_count": 0,
                    "skipped_count": len(items),
                    "result": result,
                    "error": None,
                }
                with self._lock:
                    self._jobs[job_id] = job
                return dict(job)
        elif request.operation == "vlm_behavior" and not registry.has_vlm:
            raise RuntimeError("请先配置 Qwen-VLM API")
        if request.operation == "no_action_trim" and (not registry.has_local or registry.status().get("local", {}).get("family") != "YOLOE"):
            raise RuntimeError("无动作剪切需要已加载的 YOLOE26X 分割模型")
        if request.operation in {"video_smoothing", "no_action_trim"}:
            for episode_id in episode_ids:
                media_file_id = request.media_file_ids.get(episode_id)
                if not media_file_id:
                    raise ValueError(f"{episodes[episode_id]['name']} 必须指定一个视频流")
                try:
                    episode_media(episodes[episode_id], media_file_id)
                except KeyError as exc:
                    raise ValueError(f"{episodes[episode_id]['name']} 的视频流不存在: {media_file_id}") from exc
        job_id = uuid.uuid4().hex
        job = {
            "id": job_id,
            "kind": "batch_analysis",
            "operation": request.operation,
            "status": "queued",
            "progress": 0,
            "message": f"已提交后台线程 · {len(episode_ids)} Episodes",
            "episode_count": len(episode_ids),
            "completed_count": 0,
            "skipped_count": 0,
            "result": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
        self._executor.submit(self._run, job_id, dataset_id, episode_ids, request)
        return dict(job)

    @staticmethod
    def _reused_behavior_item(episode: dict, status: dict) -> dict:
        payload = status.get("payload") or {}
        return {
            "episode_id": episode["id"],
            "episode_name": episode["name"],
            "status": "skipped",
            "reason": "existing_valid_annotation",
            "reused": True,
            "validation": status.get("validation"),
            "artifact_path": status.get("artifact_path"),
            "target_path": status.get("target_path"),
            "task_label": payload.get("task_label"),
            "confidence": payload.get("confidence"),
            "target_count": len(payload.get("primary_targets") or []),
        }

    def get(self, job_id: str) -> dict:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return dict(self._jobs[job_id])

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            self._jobs[job_id].update(changes)

    def _run(self, job_id: str, dataset_id: str, episode_ids: list[str], request: BatchAnalysisRequest) -> None:
        try:
            manifest = get_manifest(dataset_id)
            episodes = {item["id"]: item for item in manifest.get("episodes", [])}
            results = []
            failures = []
            total = len(episode_ids)
            self._update(job_id, status="running", progress=1, message=f"后台线程已启动 · 0/{total}")
            for position, episode_id in enumerate(episode_ids):
                episode = episodes[episode_id]
                base = position / total * 100
                span = 1 / total * 100
                self._update(job_id, message=f"{episode['name']} · {position + 1}/{total}")
                try:
                    if request.operation == "video_smoothing":
                        selected_media = episode_media(episode, request.media_file_ids[episode_id])

                        def smoothing_progress(value: float, message: str) -> None:
                            self._update(
                                job_id,
                                progress=min(99, round(base + span * max(0, min(100, value)) / 100, 1)),
                                message=f"{episode['name']} · {message} · {position + 1}/{total}",
                            )

                        payload = smooth_video(dataset_id, episode, selected_media, smoothing_progress)
                        results.append({
                            "episode_id": episode_id,
                            "episode_name": episode["name"],
                            "status": "completed",
                            "media_file_id": selected_media.get("file_id"),
                            "stream_name": selected_media.get("stream_name"),
                            "frame_count": payload.get("summary", {}).get("frame_count", 0),
                            "artifact_path": payload.get("artifact_path"),
                        })
                    elif request.operation == "vlm_behavior":
                        behavior_request = BehaviorAnnotationRequest(sample_count=request.sample_count, force=request.force)

                        def progress(value: float, message: str) -> None:
                            self._update(
                                job_id,
                                progress=min(99, round(base + span * max(0, min(100, value)) / 100, 1)),
                                message=f"{episode['name']} · {message} · {position + 1}/{total}",
                            )

                        payload = annotate_episode_behavior(dataset_id, manifest, episode, behavior_request, progress)
                        reused = bool((payload.get("reuse") or {}).get("reused"))
                        results.append({
                            "episode_id": episode_id,
                            "episode_name": episode["name"],
                            "status": "skipped" if reused else "completed",
                            "reason": "existing_valid_annotation" if reused else None,
                            "reused": reused,
                            "task_label": payload.get("task_label"),
                            "confidence": payload.get("confidence"),
                            "target_count": len(payload.get("primary_targets", [])),
                        })
                    elif request.operation == "pose_recovery":
                        status = pose_recovery_status(dataset_id, manifest, episode)
                        if not status.get("available"):
                            results.append({"episode_id": episode_id, "episode_name": episode["name"], "status": "skipped", "reason": "no_mocap_source"})
                        elif status.get("artifact_exists") and not status.get("needed"):
                            results.append({"episode_id": episode_id, "episode_name": episode["name"], "status": "skipped", "reason": "existing_result", "artifact_path": status.get("artifact_path")})
                        else:
                            self._update(job_id, progress=min(99, round(base + span * 0.2, 1)), message=f"{episode['name']} · SLAM/VO 恢复中 · {position + 1}/{total}")
                            payload = recover_episode_pose(dataset_id, manifest, episode)
                            results.append({
                                "episode_id": episode_id,
                                "episode_name": episode["name"],
                                "status": "completed",
                                "recovered_frame_count": payload.get("recovered_frame_count", 0),
                                "artifact_path": payload.get("artifact_path"),
                            })
                    else:
                        selected_media = episode_media(episode, request.media_file_ids[episode_id])

                        def trim_progress(value: float, message: str) -> None:
                            self._update(
                                job_id,
                                progress=min(99, round(base + span * max(0, min(100, value)) / 100, 1)),
                                message=f"{episode['name']} · {message} · {position + 1}/{total}",
                            )

                        payload = analyze_no_action_trim(
                            dataset_id,
                            manifest,
                            episode,
                            trim_progress,
                            selected_media,
                            sample_fps=request.sample_fps,
                            proximity_threshold=request.proximity_threshold,
                            max_gap_seconds=request.max_gap_seconds,
                            min_valid_seconds=request.min_valid_seconds,
                        )
                        results.append({
                            "episode_id": episode_id,
                            "episode_name": episode["name"],
                            "status": "completed",
                            "media_file_id": selected_media.get("file_id"),
                            "stream_name": selected_media.get("stream_name"),
                            "valid_frame_count": payload.get("summary", {}).get("valid_frame_count", 0),
                            "invalid_frame_count": payload.get("summary", {}).get("invalid_frame_count", 0),
                            "artifact_path": payload.get("artifact_path"),
                        })
                except Exception as exc:
                    failures.append({"episode_id": episode_id, "episode_name": episode["name"], "error": str(exc)})
                completed = position + 1
                self._update(job_id, completed_count=completed, progress=round(completed / total * 100, 1), message=f"后台线程处理中 · {completed}/{total}")
            result = {
                "dataset_id": dataset_id,
                "operation": request.operation,
                "episode_count": total,
                "completed_count": len(results),
                "skipped_count": sum(item.get("status") == "skipped" for item in results),
                "reused_count": sum(bool(item.get("reused")) for item in results),
                "failure_count": len(failures),
                "items": results,
                "failures": failures,
            }
            if failures and not results:
                self._update(job_id, status="failed", progress=100, message=f"全部 {total} 个 Episode 处理失败", result=result, error=failures[0]["error"])
            else:
                message = f"后台任务完成 · {len(results)}/{total}"
                if failures:
                    message += f" · {len(failures)} 个失败"
                self._update(job_id, status="complete", progress=100, message=message, result=result)
        except Exception as exc:
            self._update(job_id, status="failed", message=str(exc), error=str(exc), trace=traceback.format_exc(limit=8))


batch_analysis_jobs = BatchAnalysisJobManager()
