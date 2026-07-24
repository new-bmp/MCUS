from __future__ import annotations

import threading
from copy import deepcopy
from datetime import datetime, timezone


class JobCancelled(Exception):
    """Raised inside a worker when the user requested a cooperative stop."""


class CancellableJobMixin:
    """Small, thread-safe cancellation protocol shared by background managers."""

    _TERMINAL_STATUSES = {"complete", "completed", "failed", "cancelled"}

    def _init_cancellation(self) -> None:
        self._cancel_events: dict[str, threading.Event] = {}

    def _register_cancellation(self, job_id: str) -> None:
        with self._lock:
            self._cancel_events[job_id] = threading.Event()

    def _event_for(self, job_id: str) -> threading.Event:
        with self._lock:
            return self._cancel_events.setdefault(job_id, threading.Event())

    @staticmethod
    def _status_field(job: dict) -> str:
        return "status" if "status" in job else "state"

    def cancel(self, job_id: str) -> dict:
        """Request a job stop, returning its new/current public state."""
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            job = self._jobs[job_id]
            field = self._status_field(job)
            status = str(job.get(field) or "")
            if status in self._TERMINAL_STATUSES:
                return deepcopy(job)
            event = self._cancel_events.setdefault(job_id, threading.Event())
            event.set()
            if status == "queued":
                job.update({field: "cancelled", "message": "任务已终止", "error": None, "finished_at": _now()})
            else:
                job.update({field: "cancelling", "message": "正在终止任务…", "error": None})
            result = deepcopy(job)
        if result.get(field) == "cancelled":
            hook = getattr(self, "_on_cancelled_before_run", None)
            if hook is not None:
                hook(job_id)
        return result

    def _start_unless_cancelled(self, job_id: str, **changes) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if self._cancel_events.setdefault(job_id, threading.Event()).is_set() or job.get(self._status_field(job)) == "cancelled":
                raise JobCancelled()
            job.update(changes)

    def _raise_if_cancelled(self, job_id: str) -> None:
        with self._lock:
            if self._cancel_events.setdefault(job_id, threading.Event()).is_set():
                raise JobCancelled()

    def _progress_update(self, job_id: str, update, **changes) -> None:
        self._raise_if_cancelled(job_id)
        update(**changes)

    def _mark_cancelled(self, job_id: str, message: str | None = None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            field = self._status_field(job)
            completed = job.get("completed_count")
            total = job.get("episode_count")
            if message is None:
                message = "任务已终止"
                if completed is not None and total is not None:
                    message += f" · 已完成 {completed}/{total} Episodes"
            job.update({field: "cancelled", "message": message, "error": None, "finished_at": _now()})

    def _forget_cancellation(self, job_id: str) -> None:
        with self._lock:
            self._cancel_events.pop(job_id, None)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
