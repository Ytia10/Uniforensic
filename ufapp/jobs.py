"""Background jobs with step-by-step progress (polled by the web page).

A job is a Python function `fn(job, **params) -> result dict`. It reports progress
with job.step("name") ... job.done_step(detail) and job.log(line). Heavy jobs run
one at a time (config_app.yaml max_parallel_jobs), so a laptop is not overloaded.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .settings import CFG


@dataclass
class Job:
    id: str
    kind: str
    params: dict
    status: str = "queued"                # queued | running | done | error
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    steps: list[dict] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    progress: float = 0.0                 # 0..1, set by jobs that know their length
    result: dict | None = None
    error: str | None = None

    # -- progress API used by job functions
    def step(self, name: str) -> None:
        self.steps.append({"name": name, "state": "running", "t0": time.time(), "seconds": None,
                           "detail": ""})

    def done_step(self, detail: str = "") -> None:
        s = self.steps[-1]
        s.update(state="done", seconds=round(time.time() - s["t0"], 2), detail=detail)

    def log(self, line: str) -> None:
        self.lines.append(line)
        del self.lines[:-400]             # keep the last 400 lines

    def public(self) -> dict:
        return {"id": self.id, "kind": self.kind, "params": self.params, "status": self.status,
                "progress": self.progress, "error": self.error,
                "seconds": round((self.finished or time.time()) - (self.started or self.created), 1),
                "steps": [{k: v for k, v in s.items() if k != "t0"} for s in self.steps],
                "lines": self.lines[-60:], "has_result": self.result is not None}


_jobs: dict[str, Job] = {}
_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=int(CFG.get("max_parallel_jobs", 1)))


def submit(kind: str, fn: Callable[..., dict], **params: Any) -> Job:
    job = Job(id=uuid.uuid4().hex[:12], kind=kind, params=params)
    with _lock:
        _jobs[job.id] = job

    def run():
        job.status, job.started = "running", time.time()
        try:
            job.result = fn(job, **params)
            job.status = "done"
            job.progress = 1.0
        except Exception as e:                       # noqa: BLE001  (shown to the user)
            if job.steps and job.steps[-1]["state"] == "running":
                job.steps[-1]["state"] = "error"
            job.error = f"{type(e).__name__}: {e}"
            job.log(traceback.format_exc())
            job.status = "error"
        finally:
            job.finished = time.time()

    _pool.submit(run)
    return job


def get(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)


def latest(kind: str) -> Job | None:
    with _lock:
        js = [j for j in _jobs.values() if j.kind == kind and j.status == "done"]
    return max(js, key=lambda j: j.created) if js else None
