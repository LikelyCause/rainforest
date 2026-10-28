"""In-process background job runner.

Jobs run on a small thread pool inside the API worker that accepted the request.
All job state lives in Postgres (status / progress / error columns), so any worker
can report on any job. For multi-node deployments swap this for a real queue.

Ownership: every API process holds a session advisory lock on a random owner id for its lifetime and stamps
that id on the job rows it creates (`owner`). A job whose owner no longer holds the lock was interrupted, however
recently it reported progress, so `sweep()` can run whenever a worker starts without touching a sibling's jobs.

Concurrency: a job runs only while it holds one of JOB_SLOTS advisory-lock slots, shared by every process, and
its native thread pools are limited to cpu_count // JOB_SLOTS threads. Waiting jobs start oldest first: each takes a
ticket on submit and holds a lock on it (on the owner connection) until it gets a slot, and a job takes a free slot
only when no lower ticket is still held.
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import psycopg
from psycopg import sql
from threadpoolctl import threadpool_limits

from .config import get_settings
from .db import connection

log = logging.getLogger(__name__)

# Advisory-lock class ids (two-key form, a separate key space from db.py's single-key schema lock). Other two-key
# locks need their own class: experiments.TEAM_LOCK is 7_311_208.
OWNER_LOCK = 7_311_205
SLOT_LOCK = 7_311_206
TICKET_LOCK = 7_311_207
INTERRUPTED = "Interrupted by an API restart."
KEEPALIVE_S = 30.0
SLOT_POLL_S = 1.0

_executor: ThreadPoolExecutor | None = None
_owner: int | None = None
_owner_conn: psycopg.Connection | None = None
_owner_mutex = threading.Lock()
_stop = threading.Event()
_keeper: threading.Thread | None = None

_DEAD = f"""(owner IS NULL OR NOT EXISTS (
               SELECT 1 FROM pg_locks l
                WHERE l.locktype = 'advisory' AND l.granted AND l.classid = {OWNER_LOCK}::oid
                  AND l.objid = owner::oid AND l.objsubid = 2
                  AND l.database = (SELECT oid FROM pg_database WHERE datname = current_database())))"""

_AHEAD = f"""SELECT EXISTS (SELECT 1 FROM pg_locks l
                 WHERE l.locktype = 'advisory' AND l.granted AND l.classid = {TICKET_LOCK}::oid
                   AND l.objid::text::bigint < %s AND l.objsubid = 2
                   AND l.database = (SELECT oid FROM pg_database WHERE datname = current_database()))"""


def cpu_threads() -> int:
    """Threads one job may use, so JOB_SLOTS concurrent jobs share the CPUs instead of each taking all of them."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return max(1, n // max(1, get_settings().job_slots))


def owner_id() -> int:
    """This process's owner id; takes (or retakes, after a lost connection) its advisory lock."""
    global _owner, _owner_conn
    with _owner_mutex:
        if _owner_conn is None or _owner_conn.closed or _owner_conn.broken:
            fresh = _owner is None
            _owner_conn, _owner = _lock_owner(_owner)
            if fresh:
                sweep(owner=_owner)  # rows a dead process left under the same id
            _start_keeper()
        return _owner


def _lock_owner(prev: int | None) -> tuple[psycopg.Connection, int]:
    conn = psycopg.connect(get_settings().database_url, autocommit=True)
    cand = prev
    while True:
        cand = cand or secrets.randbelow(2**31 - 2) + 1
        if conn.execute("SELECT pg_try_advisory_lock(%s, %s)", (OWNER_LOCK, cand)).fetchone()[0]:
            return conn, cand
        cand = None


def _start_keeper() -> None:
    global _keeper
    if _keeper is not None and _keeper.is_alive():
        return
    _keeper = threading.Thread(target=_keep_owner_lock, name="job-owner", daemon=True)
    _keeper.start()


def _keep_owner_lock() -> None:
    while not _stop.wait(KEEPALIVE_S):
        try:
            assert _owner_conn is not None
            _owner_conn.execute("SELECT 1")
        except (psycopg.Error, AssertionError):
            if _stop.is_set():
                return
            log.warning("job owner connection lost; retaking the owner lock")
            try:
                owner_id()
            except psycopg.Error:
                log.exception("could not retake the job owner lock")


def sweep(owner: int | None = None) -> None:
    """Fail jobs whose owning process is gone, or with `owner`, every active job stamped with that id. Without
    `owner` (at startup), running batches then resume."""
    dead = sql.SQL(_DEAD) if owner is None else sql.SQL("owner = {}").format(sql.Literal(owner))
    with connection() as conn:
        # Before the datasets below change status: a pull is interrupted with the dataset job it belongs to.
        conn.execute(
            sql.SQL(
                """UPDATE meta.dataset_pulls SET status = 'failed', error = 'Interrupted by an API restart.',
                          finished_at = now()
                WHERE status = 'running' AND dataset_id IN (
                      SELECT id FROM meta.datasets
                       WHERE status IN ('pending', 'ingesting', 'profiling', 'correlating', 'refreshing') AND {})"""
            ).format(dead)
        )
        # A profiled dataset only loses the associations of targets chosen later; it stays usable.
        conn.execute(
            sql.SQL(
                """UPDATE meta.datasets
                  SET status = 'ready', progress = 1, status_message = 'Ready', updated_at = now(),
                      warnings = array_append(warnings, 'Associations for the new target(s) were interrupted '
                                              'by an API restart; choose the targets again to retry.')
                WHERE status = 'correlating' AND profiled_at IS NOT NULL AND {}"""
            ).format(dead)
        )
        # An interrupted refresh swapped nothing in (the swap is one transaction), so the old data is all there; its
        # leftover _new file and table are cleared by the next refresh. A refresh whose pull already ended, but
        # whose job couldn't set the dataset ready, is put back whoever owns it.
        conn.execute(
            sql.SQL(
                """UPDATE meta.datasets d SET status = 'ready', progress = 1, status_message = 'Ready', updated_at = now()
                WHERE status = 'refreshing' AND ({} OR NOT EXISTS (
                      SELECT 1 FROM meta.dataset_pulls p WHERE p.dataset_id = d.id AND p.status = 'running'))"""
            ).format(dead)
        )
        conn.execute(
            sql.SQL(
                """UPDATE meta.datasets
                  SET status = 'failed', updated_at = now(),
                      error = CASE WHEN file_format = 'mongodb'
                                   THEN 'Interrupted by an API restart; delete it and connect again.'
                                   ELSE 'Interrupted by an API restart; upload or generate it again.' END
                WHERE status IN ('pending', 'ingesting', 'profiling', 'correlating') AND {}"""
            ).format(dead)
        )
        for table in ("exports", "model_runs", "analyses"):
            # A batch's runs that it hasn't submitted yet have no owner and no job to lose: the batch resumes them.
            waiting = sql.SQL(" AND NOT (batch_id IS NOT NULL AND owner IS NULL)" if table == "model_runs" else "")
            conn.execute(
                sql.SQL(
                    """UPDATE meta.{} SET status = 'failed', error = {}, status_message = 'Failed',
                          finished_at = now(), updated_at = now()
                    WHERE status IN ('pending', 'running') AND {}{}"""
                ).format(sql.Identifier(table), sql.Literal(INTERRUPTED), dead, waiting)
            )
    if owner is None:
        from . import batches

        batches.resume_all()


def start() -> None:
    """Take this process's owner lock, then fail jobs left behind by processes that are gone."""
    _stop.clear()
    # Once per process, never restored: jobs share these native thread pools, so one ending mustn't widen them while
    # another runs.
    threadpool_limits(cpu_threads())
    owner_id()
    sweep()


def _take_ticket() -> int:
    """Join the queue: a ticket, locked on the owner connection until the job gets a slot (or the process ends)."""
    owner_id()
    with _owner_mutex:
        assert _owner_conn is not None
        ticket = _owner_conn.execute("SELECT nextval('meta.job_ticket')").fetchone()[0]
        _owner_conn.execute("SELECT pg_advisory_lock(%s, %s)", (TICKET_LOCK, ticket))
    return ticket


def _leave_queue(ticket: int) -> None:
    with _owner_mutex:
        if _owner_conn is None or _owner_conn.closed:
            return  # a lost connection already released it
        try:
            _owner_conn.execute("SELECT pg_advisory_unlock(%s, %s)", (TICKET_LOCK, ticket))
        except psycopg.Error:
            log.warning("could not release job ticket %s", ticket)


@contextmanager
def _slot(ticket: int) -> Iterator[bool]:
    """Hold one of JOB_SLOTS global slots for the duration, taken in ticket order; yields False if the process stops
    while waiting."""
    s = get_settings()
    queued = True
    try:
        with psycopg.connect(s.database_url, autocommit=True) as conn:
            while True:
                if not conn.execute(_AHEAD, (ticket,)).fetchone()[0]:
                    for k in range(max(1, s.job_slots)):
                        if conn.execute("SELECT pg_try_advisory_lock(%s, %s)", (SLOT_LOCK, k)).fetchone()[0]:
                            _leave_queue(ticket)
                            queued = False
                            yield True
                            return
                if _stop.wait(SLOT_POLL_S):
                    yield False
                    return
    finally:
        if queued:
            _leave_queue(ticket)


def _get() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=get_settings().job_workers, thread_name_prefix="job")
    return _executor


def submit(fn: Callable[..., Any], *args: Any) -> None:
    """Queue `fn(*args)`. Its job row must carry `owner = owner_id()` so a restart can tell it was interrupted."""

    def run() -> None:
        try:
            # Taken here, not at submit: a job waiting for one of this process's threads mustn't hold a place in the
            # global queue that it can't use.
            ticket = _take_ticket()
            with _slot(ticket) as ok:
                if ok:
                    fn(*args)
        except Exception:  # noqa: BLE001 - the job itself records failures; this is a backstop
            log.exception("background job %s%r crashed", getattr(fn, "__name__", fn), args)

    _get().submit(run)


def shutdown() -> None:
    global _executor, _owner, _owner_conn
    _stop.set()
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None
    with _owner_mutex:
        if _owner_conn is not None:
            _owner_conn.close()
        _owner_conn, _owner = None, None
