"""Job ownership (the restart sweep) and the global job slots. Needs RAINFOREST_TEST_DATABASE_URL."""
import os
import threading
import time

import psycopg
import pytest

from app import db, jobs
from app.config import get_settings

DB_URL = os.environ.get("RAINFOREST_TEST_DATABASE_URL")


@pytest.fixture(scope="module")
def database(tmp_path_factory):
    if not DB_URL:
        pytest.skip("set RAINFOREST_TEST_DATABASE_URL to run the job tests")
    mp = pytest.MonkeyPatch()
    mp.setenv("DATABASE_URL", DB_URL)
    mp.setenv("DATA_DIR", str(tmp_path_factory.mktemp("data")))
    get_settings.cache_clear()
    db.close_pool()
    db.bootstrap()
    jobs.start()
    yield
    jobs.shutdown()
    db.close_pool()
    mp.undo()
    get_settings.cache_clear()


@pytest.fixture
def sibling():
    """Another API process: a session holding its own owner lock."""
    conn = psycopg.connect(DB_URL, autocommit=True)
    conn.execute("SELECT pg_advisory_lock(%s, %s)", (jobs.OWNER_LOCK, 424_242))
    yield conn
    conn.close()


def _dataset(status: str = "ready", owner: int | None = None, profiled: bool = False) -> int:
    return db.fetch_one(
        """INSERT INTO meta.datasets (name, file_format, status, owner, profiled_at)
           VALUES ('jobs-test', 'generated', %s, %s, CASE WHEN %s THEN now() END) RETURNING id""",
        (status, owner, profiled),
    )["id"]


def _export(ds: int, status: str, owner: int | None) -> int:
    return db.fetch_one(
        "INSERT INTO meta.exports (dataset_id, status, owner, updated_at) VALUES (%s, %s, %s, now()) RETURNING id",
        (ds, status, owner),
    )["id"]


def _status(table: str, row_id: int) -> dict:
    return db.fetch_one(f"SELECT status, error, warnings FROM meta.{table} WHERE id = %s"
                        if table == "datasets" else f"SELECT status, error FROM meta.{table} WHERE id = %s",
                        (row_id,))


def test_sweep_fails_only_jobs_whose_owner_is_gone(database, sibling):
    gone = 515_151
    ds = _dataset()
    live = _export(ds, "running", 424_242)
    mine = _export(ds, "running", jobs.owner_id())
    restarted = _export(ds, "running", gone)  # updated a moment ago: a fast restart
    queued = _export(ds, "pending", gone)
    legacy = _export(ds, "pending", None)
    finished = _export(ds, "done", gone)
    profiling = _dataset("profiling", gone)
    retargeting = _dataset("correlating", gone, profiled=True)
    ingesting_live = _dataset("ingesting", 424_242)

    jobs.sweep()

    assert _status("exports", live)["status"] == "running"
    assert _status("exports", mine)["status"] == "running"
    for row in (restarted, queued, legacy):
        assert _status("exports", row) == {"status": "failed", "error": "Interrupted by an API restart."}
    assert _status("exports", finished)["status"] == "done"
    assert _status("datasets", profiling)["status"] == "failed"
    after = _status("datasets", retargeting)
    assert after["status"] == "ready" and "interrupted" in after["warnings"][-1]
    assert _status("datasets", ingesting_live)["status"] == "ingesting"

    sibling.close()  # the sibling process dies
    jobs.sweep()
    assert _status("exports", live)["status"] == "failed"
    assert _status("datasets", ingesting_live)["status"] == "failed"
    assert _status("exports", mine)["status"] == "running"


def test_sweep_by_owner_clears_rows_left_under_a_reused_id(database):
    ds = _dataset()
    stale = _export(ds, "running", 777)
    other = _export(ds, "running", 778)
    jobs.sweep(owner=777)
    assert _status("exports", stale)["status"] == "failed"
    assert _status("exports", other)["status"] == "running"


def test_owner_lock_is_retaken_after_a_lost_connection(database):
    before = jobs.owner_id()
    jobs._owner_conn.close()
    assert jobs.owner_id() == before
    probe = psycopg.connect(DB_URL, autocommit=True)
    try:
        held = probe.execute("SELECT pg_try_advisory_lock(%s, %s)", (jobs.OWNER_LOCK, before)).fetchone()[0]
        assert held is False
    finally:
        probe.close()


def _run_jobs(n: int, seconds: float) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    done = threading.Semaphore(0)

    def job() -> None:
        start = time.monotonic()
        time.sleep(seconds)
        spans.append((start, time.monotonic()))
        done.release()

    for _ in range(n):
        jobs.submit(job)
    for _ in range(n):
        assert done.acquire(timeout=30)
    return sorted(spans)


def test_slots_bound_concurrency_across_processes(database, monkeypatch):
    monkeypatch.setenv("JOB_SLOTS", "1")
    get_settings.cache_clear()
    try:
        spans = _run_jobs(2, 0.4)
        assert spans[1][0] >= spans[0][1], "with one slot the second job waits for the first"

        other = psycopg.connect(DB_URL, autocommit=True)  # another process holds the only slot
        other.execute("SELECT pg_advisory_lock(%s, 0)", (jobs.SLOT_LOCK,))
        started = threading.Event()
        jobs.submit(started.set)
        assert not started.wait(1.5)
        other.close()
        assert started.wait(10)
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


def test_cpu_threads_divides_cpus_between_slots(monkeypatch):
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    monkeypatch.setenv("JOB_SLOTS", "2")
    get_settings.cache_clear()
    try:
        assert jobs.cpu_threads() == max(1, cpus // 2)
        monkeypatch.setenv("JOB_SLOTS", str(cpus * 4))
        get_settings.cache_clear()
        assert jobs.cpu_threads() == 1
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()
