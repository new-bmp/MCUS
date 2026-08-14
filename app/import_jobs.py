from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from typing import Callable

from .dataset_format import inspect_dataset_format
from .job_control import CancellableJobMixin, JobCancelled
from .schemas import PathOpenRequest
from .storage import scan_dataset


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DatasetImportJobManager(CancellableJobMixin):
    """Serialize disk-heavy imports and expose cooperative scan progress."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="alice-dataset-import")
        self._jobs: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._init_cancellation()

    def submit(
        self,
        request: PathOpenRequest,
        confirmation_token: str,
        understand_manifest: Callable[[dict], dict] | None = None,
    ) -> dict:
        job_id = uuid.uuid4().hex
        job = {
            "id": job_id,
            "kind": "dataset_import",
            "status": "queued",
            "progress": 0.0,
            "stage": "queued",
            "message": "数据集导入已排队",
            "source_path": request.path,
            "file_count": 0,
            "video_count": 0,
            "episode_count": 0,
            "result": None,
            "error": None,
            "created_at": _now(),
        }
        with self._lock:
            self._jobs[job_id] = job
            self._register_cancellation(job_id)
        request_copy = request.model_copy(deep=True)
        self._executor.submit(
            self._run,
            job_id,
            request_copy,
            confirmation_token,
            understand_manifest,
        )
        return deepcopy(job)

    def get(self, job_id: str) -> dict:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return deepcopy(self._jobs[job_id])

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            self._jobs[job_id].update(changes)

    def _run(
        self,
        job_id: str,
        request: PathOpenRequest,
        confirmation_token: str,
        understand_manifest: Callable[[dict], dict] | None,
    ) -> None:
        try:
            self._start_unless_cancelled(
                job_id,
                status="running",
                progress=1.0,
                stage="preflight",
                message="正在重新确认数据格式与相机预设",
                started_at=_now(),
            )
            report = inspect_dataset_format(request.path, camera_profile_id=request.camera_profile_id)
            self._raise_if_cancelled(job_id)
            if report.get("confirmation_token") != confirmation_token:
                raise ValueError("数据集内容或相机预设已发生变化，请重新执行导入前确认")
            if report.get("root_mode") == "collection":
                raise ValueError("该目录包含多个独立数据集，请选择具体子数据集导入")
            if report.get("status") == "blocked" or not (report.get("capabilities") or {}).get("can_import"):
                raise ValueError("格式预检未通过，当前文件夹不能安全导入")

            def progress(value: float, message: str, metrics: dict) -> None:
                self._raise_if_cancelled(job_id)
                update = {
                    "progress": round(min(93.0, 5.0 + max(0.0, min(100.0, value)) * 0.88), 1),
                    "stage": "scan_dataset",
                    "message": message,
                }
                for key in ("file_count", "video_count", "probed_video_count", "episode_count"):
                    if key in metrics:
                        update[key] = metrics[key]
                self._update(job_id, **update)

            manifest = scan_dataset(
                request.path,
                request.name,
                camera_profile_id=request.camera_profile_id,
                progress=progress,
                check_cancelled=lambda: self._raise_if_cancelled(job_id),
            )
            self._raise_if_cancelled(job_id)
            if request.analyze_schema and understand_manifest is not None:
                self._update(
                    job_id,
                    progress=96.0,
                    stage="understand_schema",
                    message="正在理解数据结构与 Episode 归属",
                )
                manifest = understand_manifest(manifest)
                self._raise_if_cancelled(job_id)
            self._update(
                job_id,
                status="complete",
                progress=100.0,
                stage="complete",
                message="数据集导入完成",
                file_count=int(manifest.get("file_count") or 0),
                episode_count=int(manifest.get("episode_count") or 0),
                result=manifest,
                finished_at=_now(),
            )
        except JobCancelled:
            self._mark_cancelled(job_id, "数据集导入已终止")
        except Exception as exc:
            self._update(
                job_id,
                status="failed",
                stage="failed",
                message=str(exc),
                error=str(exc),
                trace=traceback.format_exc(limit=8),
                finished_at=_now(),
            )
        finally:
            self._forget_cancellation(job_id)


dataset_import_jobs = DatasetImportJobManager()
