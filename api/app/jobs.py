"""In-process background job runner.

Jobs run on a small thread pool inside the API worker that accepted the request.
All job state lives in Postgres (status / progress / error columns), so any worker
can report on any job. For multi-node deployments swap this for a real queue.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from .config import get_settings

log = logging.getLogger(__name__)
_executor: ThreadPoolExecutor | None = None


def _get() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=get_settings().job_workers, thread_name_prefix="job")
    return _executor


def submit(fn: Callable[..., Any], *args: Any) -> None:
    def run() -> None:
        try:
            fn(*args)
        except Exception:  # noqa: BLE001 - the job itself records failures; this is a backstop
            log.exception("background job %s%r crashed", getattr(fn, "__name__", fn), args)

    _get().submit(run)


def shutdown() -> None:
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None
