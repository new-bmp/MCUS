from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from .behavior_annotator import annotate_episode_behavior, behavior_analysis_context, behavior_annotation_status
from .job_control import CancellableJobMixin, JobCancelled
from .models import registry
from .no_action_trim import analyze_no_action_trim
from .pose_recovery import pose_recovery_status, recover_episode_pose
from .projection_correction import projection_correction_status, run_projection_correction
from .schemas import BatchAnalysisRequest, BehaviorAnnotationRequest, HandPoseModelConfig
from .sensor_alignment import ensure_episode_time_sync
from .storage import episode_media, get_manifest, require_media_eligibility
from .video_smoothing import smooth_video


def _projection_result_matches_request(status: dict, request: BatchAnalysisRequest) -> bool:
    summary = status.get("summary") or {}
    return bool(
        status.get("available")
        and str(summary.get("wrist_point_source") or "") == str(request.wrist_point_source)
    )


class BatchAnalysisJobManager(CancellableJobMixin):
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="alice-batch-analysis")
        self._jobs: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._projection_lock = threading.Lock()
        self._init_cancellation()

    def submit(self, dataset_id: str, request: BatchAnalysisRequest) -> dict:
        manifest = get_manifest(dataset_id)
        episodes = {item["id"]: item for item in manifest.get("episodes", [])}
        episode_ids = list(dict.fromkeys(request.episode_ids))
        if not episode_ids:
            raise ValueError("At least one Episode is required")
        missing = [episode_id for episode_id in episode_ids if episode_id not in episodes]
        if missing:
            raise KeyError(missing[0])
        media_operation = {
            "video_smoothing": "video_smoothing",
            "no_action_trim": "no_action_trim",
            "vlm_behavior": "vlm_behavior",
            "projection_correction": "projection_correction",
        }.get(request.operation)
        if media_operation:
            for episode_id in episode_ids:
                episode = episodes[episode_id]
                media_file_id = request.media_file_ids.get(episode_id) or episode.get("primary_media_file_id")
                if not media_file_id:
                    raise ValueError(f"{episode['name']} 必须指定一个可用于 {request.operation} 的 RGB 视频流")
                try:
                    media = episode_media(episode, str(media_file_id))
                except KeyError as exc:
                    raise ValueError(f"{episode['name']} 的视频流不存在: {media_file_id}") from exc
                require_media_eligibility(media, media_operation)
        reusable: dict[str, dict] = {}
        if request.operation == "vlm_behavior" and not request.force:
            for episode_id in episode_ids:
                episode = episodes[episode_id]
                context = behavior_analysis_context(
                    dataset_id,
                    manifest,
                    episode,
                    request.media_file_ids.get(episode_id) or episode.get("primary_media_file_id"),
                )
                reusable[episode_id] = behavior_annotation_status(
                    dataset_id,
                    manifest,
                    episode,
                    source_media_file_id=str(context["source_media"].get("file_id") or "") or None,
                    analysis_media=context["analysis_media"],
                    analysis_frame_ranges=context["analysis_frame_ranges"],
                )
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
        hand_pose_status = registry.status().get("hand_pose", {})
        if (
            request.operation == "projection_correction"
            and request.hand_pose_backend == "current"
            and not registry.has_hand_pose
            and not hand_pose_status.get("loading")
        ):
            raise RuntimeError("Hand-pose detector is not loaded; configure MediaPipe or AlicePose first")
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
            self._register_cancellation(job_id)
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

    def _ensure_projection_backend(
        self,
        job_id: str,
        request: BatchAnalysisRequest,
        *,
        timeout_seconds: float = 180.0,
    ) -> dict:
        desired = str(request.hand_pose_backend or "current").casefold()
        deadline = time.monotonic() + max(1.0, float(timeout_seconds))
        requested = False
        while True:
            self._raise_if_cancelled(job_id)
            status = registry.status().get("hand_pose") or {}
            backend = str(status.get("backend") or "").casefold()
            loaded = bool(status.get("loaded")) and registry.has_hand_pose
            loading = bool(status.get("loading"))

            if desired == "current":
                if loaded:
                    return status
                if not loading:
                    raise RuntimeError(status.get("error") or "Hand-pose detector is not loaded")
            elif loaded and backend == desired:
                return status
            elif not loading:
                if requested and backend == desired and status.get("error"):
                    raise RuntimeError(f"{desired} failed to load: {status['error']}")
                model_path = request.hand_pose_model_path.strip() if desired == "alicepose" else ""
                registry.configure_hand_pose_async(
                    HandPoseModelConfig(
                        kind=desired,
                        model_path=model_path,
                        device=request.hand_pose_device if desired == "alicepose" else "cpu",
                        confidence=0.1 if desired == "alicepose" else 0.35,
                    )
                )
                requested = True
                status = registry.status().get("hand_pose") or {}
                loading = bool(status.get("loading"))

            label = "AlicePose" if desired == "alicepose" else "MediaPipe" if desired == "mediapipe" else "hand-pose model"
            self._update(job_id, progress=1, message=f"Loading {label} in the background")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for {label} after {timeout_seconds:g} seconds")
            time.sleep(0.1 if loading else 0.05)

    def _run(self, job_id: str, dataset_id: str, episode_ids: list[str], request: BatchAnalysisRequest) -> None:
        projection_lock_acquired = False
        try:
            manifest = get_manifest(dataset_id)
            episodes = {item["id"]: item for item in manifest.get("episodes", [])}
            results = []
            failures = []
            total = len(episode_ids)
            self._start_unless_cancelled(job_id, status="running", progress=1, message=f"后台线程已启动 · 0/{total}")
            if request.operation == "projection_correction":
                while not self._projection_lock.acquire(timeout=0.1):
                    self._raise_if_cancelled(job_id)
                    self._update(job_id, message="Waiting for the hand-pose correction task slot")
                projection_lock_acquired = True
                self._ensure_projection_backend(job_id, request)
            for position, episode_id in enumerate(episode_ids):
                self._raise_if_cancelled(job_id)
                episode = episodes[episode_id]
                base = position / total * 100
                span = 1 / total * 100
                self._update(job_id, message=f"{episode['name']} · {position + 1}/{total}")
                try:
                    selected_reference_id = request.media_file_ids.get(episode_id) or episode.get("primary_media_file_id")
                    self._update(job_id, message=f"{episode['name']} · T0 时间同步 · {position + 1}/{total}")
                    ensure_episode_time_sync(
                        manifest,
                        episode,
                        reference_media_file_id=str(selected_reference_id or "") or None,
                    )
                    if request.operation == "video_smoothing":
                        selected_media = episode_media(episode, request.media_file_ids[episode_id])

                        def smoothing_progress(value: float, message: str) -> None:
                            self._raise_if_cancelled(job_id)
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
                        selected_media_file_id = request.media_file_ids.get(episode_id) or episode.get("primary_media_file_id")
                        context = behavior_analysis_context(
                            dataset_id,
                            manifest,
                            episode,
                            selected_media_file_id,
                        )
                        behavior_request = BehaviorAnnotationRequest(
                            sample_count=request.sample_count,
                            media_file_id=str(context["source_media"].get("file_id") or "") or None,
                            force=request.force,
                        )

                        def progress(value: float, message: str) -> None:
                            self._raise_if_cancelled(job_id)
                            self._update(
                                job_id,
                                progress=min(99, round(base + span * max(0, min(100, value)) / 100, 1)),
                                message=f"{episode['name']} · {message} · {position + 1}/{total}",
                            )

                        payload = annotate_episode_behavior(
                            dataset_id,
                            manifest,
                            episode,
                            behavior_request,
                            progress,
                            analysis_media_override=context["analysis_media"],
                            analysis_source_kind=context["analysis_source_kind"],
                            analysis_frame_ranges=context["analysis_frame_ranges"],
                            source_media_file_id=str(context["source_media"].get("file_id") or "") or None,
                        )
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
                    elif request.operation == "projection_correction":
                        selected_media = episode_media(episode, request.media_file_ids[episode_id])
                        status = projection_correction_status(dataset_id, manifest, episode)
                        if _projection_result_matches_request(status, request) and not request.force:
                            results.append({
                                "episode_id": episode_id,
                                "episode_name": episode["name"],
                                "status": "skipped",
                                "reason": "existing_result",
                                "artifact_path": status.get("artifact_path"),
                                "summary": status.get("summary"),
                            })
                        else:
                            def correction_progress(value: float, message: str) -> None:
                                self._raise_if_cancelled(job_id)
                                self._update(
                                    job_id,
                                    progress=min(99, round(base + span * max(0, min(100, value)) / 100, 1)),
                                    message=f"{episode['name']} · {message} · {position + 1}/{total}",
                                )

                            payload = run_projection_correction(
                                dataset_id,
                                manifest,
                                episode,
                                selected_media,
                                registry,
                                correction_progress,
                                sample_fps=request.sample_fps,
                                maximum_interpolation_gap_seconds=request.max_gap_seconds,
                                adjustment_rate=request.adjustment_rate,
                                adjustment_mode=request.adjustment_mode,
                                wrist_point_source=request.wrist_point_source,
                                dynamic_low_confidence=request.dynamic_low_confidence,
                                dynamic_mid_confidence=request.dynamic_mid_confidence,
                                dynamic_low_multiplier=request.dynamic_low_multiplier,
                                dynamic_mid_multiplier=request.dynamic_mid_multiplier,
                                dynamic_high_multiplier=request.dynamic_high_multiplier,
                            )
                            results.append({
                                "episode_id": episode_id,
                                "episode_name": episode["name"],
                                "status": "completed",
                                "media_file_id": selected_media.get("file_id"),
                                "stream_name": selected_media.get("stream_name"),
                                "artifact_path": payload.get("artifact_path"),
                                "summary": payload.get("summary"),
                            })
                    else:
                        selected_media = episode_media(episode, request.media_file_ids[episode_id])

                        def trim_progress(value: float, message: str) -> None:
                            self._raise_if_cancelled(job_id)
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
                except JobCancelled:
                    raise
                except Exception as exc:
                    failures.append({"episode_id": episode_id, "episode_name": episode["name"], "error": str(exc)})
                self._raise_if_cancelled(job_id)
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
        except JobCancelled:
            self._mark_cancelled(job_id)
        except Exception as exc:
            self._update(job_id, status="failed", message=str(exc), error=str(exc), trace=traceback.format_exc(limit=8))
        finally:
            if projection_lock_acquired:
                self._projection_lock.release()
            self._forget_cancellation(job_id)


batch_analysis_jobs = BatchAnalysisJobManager()
