"""Refreshing a connected dataset: a new pull swapped in under the same fields, with drift against each finalized
baseline. Needs RAINFOREST_TEST_MONGO_URL and RAINFOREST_TEST_DATABASE_URL.
"""

import os
import random

import polars as pl
import pytest
from pymongo.errors import PyMongoError

from app import db, recipes, refresh
from app.analysis.runner import data_version, public
from app.config import get_settings
from app.routers.datasets import SELECT_DATASET
from app.routers.derived import DerivedIn, add_derived
from app.routers.fields import set_targets
from app.sources.mongodb import MongoSource
from app.sources.pull import run_connect

from .conftest import set_phase
from .mongo_helpers import seed, use_sources

DB_URL = os.environ.get("RAINFOREST_TEST_DATABASE_URL")


@pytest.fixture(scope="module")
def database(tmp_path_factory):
    if not DB_URL:
        pytest.skip("set RAINFOREST_TEST_DATABASE_URL to run the refresh tests")
    mp = pytest.MonkeyPatch()
    mp.setenv("DATABASE_URL", DB_URL)
    mp.setenv("DATA_DIR", str(tmp_path_factory.mktemp("data")))
    get_settings.cache_clear()
    db.close_pool()
    db.bootstrap()
    yield
    db.close_pool()
    mp.undo()
    get_settings.cache_clear()


@pytest.fixture
def source(database, mongo, mongo_url):
    """The test connection, and the seed data put back afterwards for the other tests sharing the server."""
    with use_sources(f"test={mongo_url}"):
        yield mongo
    seed(mongo)


def _user() -> int:
    return db.fetch_one("SELECT id FROM meta.users ORDER BY id LIMIT 1")["id"]


def _connect(mongo, *, target: str | None = None, derived: tuple[tuple[str, str], ...] = (), finalize=True) -> int:
    """curated + massive connected and ready, with a hand-made decision and note on `vehicle` (so it has history and
    user settings), optional targets and formulas, and a finalized Trees baseline."""
    with db.connection() as conn:
        ds = conn.execute(
            """INSERT INTO meta.datasets (name, file_format, status, created_by)
               VALUES ('connected', 'mongodb', 'pending', %s) RETURNING id""",
            (_user(),),
        ).fetchone()["id"]
        recipes.create_first(conn, ds, _user())
        conn.execute(
            """INSERT INTO meta.dataset_sources
                   (dataset_id, kind, connection, database, collections, add_source, dedupe, created_by)
               VALUES (%s, 'mongodb', 'test', %s, %s, true, true, %s)""",
            (ds, mongo.name, db.jsonb([{"name": "curated"}, {"name": "massive"}]), _user()),
        )
    run_connect(ds)
    row = db.fetch_one("SELECT status, error FROM meta.datasets WHERE id = %s", (ds,))
    assert row["status"] == "ready", row["error"]
    vid = _field(ds, "vehicle")["id"]
    with db.connection() as conn:
        conn.execute(
            """UPDATE meta.field_decisions SET notes = 'checked by hand', source = 'user', reviewed = true
                WHERE field_id = %s""",
            (vid,),
        )
        conn.execute(
            "UPDATE meta.field_settings SET source = 'user', params = '{\"chosen\": true}' WHERE field_id = %s",
            (vid,),
        )
        conn.execute(
            """INSERT INTO meta.decision_history (field_id, dataset_id, family, previous, current)
               VALUES (%s, %s, 'tree', '{}', '{"notes": "checked by hand"}')""",
            (vid, ds),
        )
        if target:
            set_targets(conn, ds, [target], _user())
    for name, expr in derived:
        add_derived(ds, DerivedIn(name=name, kind="formula", spec={"expr": expr}), None)
    if finalize:
        set_phase(ds, "finalized")
    return ds


def _refresh(ds: int) -> dict:
    """What the refresh endpoint does before queueing the job, then the job itself."""
    pull_id = db.fetch_one(
        "INSERT INTO meta.dataset_pulls (dataset_id, status, created_by) VALUES (%s, 'running', %s) RETURNING id",
        (ds, _user()),
    )["id"]
    db.execute("UPDATE meta.datasets SET status = 'refreshing' WHERE id = %s", (ds,))
    refresh.run_refresh(ds, pull_id)
    return db.fetch_one("SELECT * FROM meta.dataset_pulls WHERE id = %s", (pull_id,))


def _field(ds: int, name: str) -> dict | None:
    return db.fetch_one("SELECT * FROM meta.fields WHERE dataset_id = %s AND name = %s", (ds, name))


def _dataset(ds: int) -> dict:
    return db.fetch_one("SELECT * FROM meta.datasets WHERE id = %s", (ds,))


def _raw_rows(ds: int) -> int:
    return db.fetch_one(f"SELECT count(*) AS n FROM raw.ds_{ds}")["n"]


def _new_table(ds: int) -> bool:
    return db.fetch_one("SELECT to_regclass(%s) IS NOT NULL AS t", (f"raw.ds_{ds}_new",))["t"]


def _leftovers(ds: int) -> list[str]:
    base = get_settings().datasets_dir / str(ds)
    return sorted(p.name for p in base.iterdir() if p.name != "raw.parquet")


def _fresh_docs(n: int, null_share: float) -> list[dict]:
    """Documents with a new key, `fresh`, and `engine` and `rpm.max` missing from the first `null_share` of them."""
    rng = random.Random(3)
    return [
        {
            "vehicle": rng.choice(["sedan", "van", "coupe"]),
            "engine": None if i < n * null_share else rng.choice(["V6", "I4"]),
            "rpm": {
                "mean": round(rng.uniform(800, 3500), 1),
                "max": None if i < n * null_share else round(rng.uniform(4000, 7000), 1),
            },
            "tags": ["fleet"],
            "fresh": rng.randint(0, 9),
        }
        for i in range(n)
    ]


def _add_analysis(ds: int) -> None:
    params = {"data": "raw", "method": "pca", "kept": [], "target": None, "sample": 1000}
    with db.connection() as conn:
        version = data_version(conn, ds, params)
        conn.execute(
            """INSERT INTO meta.analyses (dataset_id, kind, params, params_key, data_version, status, result)
               VALUES (%s, 'map', %s, 'k', %s, 'done', '{}')""",
            (ds, db.jsonb(params), version),
        )


def _settings(ds: int, name: str, family: str) -> dict:
    return db.fetch_one(
        """SELECT s.* FROM meta.field_settings s JOIN meta.fields f ON f.id = s.field_id
            WHERE f.dataset_id = %s AND f.name = %s AND s.family = %s""",
        (ds, name, family),
    )


def _analyses(ds: int) -> list[dict]:
    with db.connection() as conn:
        rows = conn.execute("SELECT * FROM meta.analyses WHERE dataset_id = %s", (ds,)).fetchall()
        return [public(r, data_version(conn, ds, r["params"])) for r in rows]


def test_refresh_keeps_field_ids_and_reports_drift(source):
    ds = _connect(source)
    _add_analysis(ds)
    assert [a["stale"] for a in _analyses(ds)] == [False]
    before = _field(ds, "vehicle")
    settings_before = db.fetch_all("SELECT * FROM meta.field_settings WHERE field_id = %s", (before["id"],))
    max_before = _settings(ds, "rpm.max", "linear")
    source.massive.insert_many(_fresh_docs(200, 0.6))

    pull = _refresh(ds)

    assert pull["status"] == "done", pull["error"]
    assert pull["rows"] == 950 and pull["finished_at"] is not None
    assert pull["per_source"] == [
        {"label": "curated", "documents": 300, "removed_duplicates": 0},
        {"label": "massive", "documents": 700, "removed_duplicates": 50},
    ]
    assert "rpm.mean" in pull["flatten_notes"]["mixed"]
    row = _dataset(ds)
    assert row["status"] == "ready" and row["status_message"] == "Ready" and row["row_count"] == 950
    assert row["column_count"] == len(db.fetch_all("SELECT id FROM meta.fields WHERE dataset_id = %s", (ds,)))
    assert "Duplicates removed: massive 50" in row["warnings"]
    assert _raw_rows(ds) == 950

    after = _field(ds, "vehicle")
    assert after["id"] == before["id"] and after["row_count"] == 950
    decision = db.fetch_one("SELECT * FROM meta.field_decisions WHERE field_id = %s", (after["id"],))
    assert decision["notes"] == "checked by hand" and decision["source"] == "user"
    assert db.fetch_all("SELECT * FROM meta.decision_history WHERE field_id = %s", (after["id"],))
    assert db.fetch_all("SELECT * FROM meta.field_settings WHERE field_id = %s", (after["id"],)) == settings_before

    # Settings nobody chose follow the new stats: rpm.max has missing values now.
    max_after, maxed = _settings(ds, "rpm.max", "linear"), _field(ds, "rpm.max")
    assert maxed["null_count"] == 120 and max_after["source"] == "suggested"
    assert max_after["add_missing_indicator"] and not max_before["add_missing_indicator"]
    assert max_after["add_missing_indicator"] == maxed["suggestion"]["linear"]["add_missing_indicator"]

    fresh = _field(ds, "fresh")
    assert fresh is not None and fresh["null_count"] == 750
    assert db.fetch_one("SELECT 1 FROM meta.field_decisions WHERE field_id = %s", (fresh["id"],))
    ordinals = [r["ordinal"] for r in db.fetch_all("SELECT ordinal FROM meta.fields WHERE dataset_id = %s", (ds,))]
    assert sorted(ordinals) == list(range(len(ordinals)))

    reopened = {r["name"]: r for r in pull["drift"]["tree"]["reopened"]}
    assert reopened["engine"]["reason"] == "drift" and "null_pct" in reopened["engine"]["drift"]
    assert reopened["fresh"] == {"name": "fresh", "reason": "new"}
    assert "vehicle" in pull["drift"]["tree"]["carried"] and pull["drift"]["tree"]["missing"] == []
    assert [a["stale"] for a in _analyses(ds)] == [True]
    assert _leftovers(ds) == [] and not _new_table(ds)


def test_refresh_removes_vanished_fields_and_targets(source):
    ds = _connect(source, target="engine")
    assert _dataset(ds)["target_columns"] == ["engine"]
    for coll in (source.curated, source.massive):
        coll.update_many({}, {"$unset": {"engine": ""}})

    pull = _refresh(ds)

    assert pull["status"] == "done", pull["error"]
    assert _field(ds, "engine") is None and _field(ds, "vehicle") is not None
    row = _dataset(ds)
    assert row["target_columns"] == []
    assert "Target column no longer in the data: engine" in row["warnings"]
    assert pull["drift"]["tree"]["missing"] == ["engine"]
    assert not db.fetch_one("SELECT bool_or(is_target) AS t FROM meta.fields WHERE dataset_id = %s", (ds,))["t"]


def test_refresh_recomputes_derived_and_drops_orphans(source):
    ds = _connect(source, derived=(("rpm_k", "`rpm.mean` / 1000"), ("max_k", "`rpm.max` / 1000")))
    kept, orphan = _field(ds, "rpm_k"), _field(ds, "max_k")
    for coll in (source.curated, source.massive):
        coll.update_many({}, {"$unset": {"rpm.max": ""}})
    # Two new columns ahead of the derived ones: every later field moves up, past an ordinal still taken.
    source.massive.insert_one({"vehicle": "van", "engine": "V6", "rpm": {"mean": 1500.0}, "late": 1, "later": 2})

    pull = _refresh(ds)

    assert pull["status"] == "done", pull["error"]
    after = _field(ds, "rpm_k")
    assert after["id"] == kept["id"] and after["derived"] == kept["derived"] and after["row_count"] == 751
    assert after["ordinal"] > _field(ds, "later")["ordinal"]
    assert _field(ds, "max_k") is None and _field(ds, "rpm.max") is None
    assert db.fetch_one("SELECT 1 FROM meta.field_decisions WHERE field_id = %s", (orphan["id"],)) is None
    row = _dataset(ds)
    assert any(w.startswith("Derived field removed: max_k") and "rpm.max" in w for w in row["warnings"])
    df = pl.read_parquet(row["parquet_path"])
    assert "max_k" not in df.columns and df.height == 751
    expected = df["rpm.mean"].cast(pl.Float64, strict=False) / 1000
    assert df["rpm_k"].drop_nulls().len() > 700 and (df["rpm_k"] == expected).all()
    cols = {
        r["column_name"]
        for r in db.fetch_all(
            "SELECT column_name FROM information_schema.columns WHERE table_schema = 'raw' AND table_name = %s",
            (f"ds_{ds}",),
        )
    }
    assert after["db_column"] in cols


def test_refresh_failure_leaves_the_dataset_untouched(source, mongo_url, monkeypatch):
    ds = _connect(source)
    row = _dataset(ds)
    mtime = os.stat(row["parquet_path"]).st_mtime_ns
    fields = db.fetch_all(
        "SELECT id, name, null_pct, ordinal FROM meta.fields WHERE dataset_id = %s ORDER BY id", (ds,)
    )
    source.massive.insert_many(_fresh_docs(200, 0.6))
    real_pull = MongoSource.pull

    def fail_after_one_chunk(self, database, collections, out_dir, budget, timeout, progress):
        def progress_then_drop(label, n):
            assert list(out_dir.glob("chunk_*.parquet"))
            raise PyMongoError(f"connection reset by peer, Topology: {mongo_url}")

        return real_pull(self, database, collections, out_dir, budget, timeout, progress_then_drop)

    monkeypatch.setattr(MongoSource, "pull", fail_after_one_chunk)

    pull = _refresh(ds)

    assert pull["status"] == "failed" and pull["finished_at"] is not None
    # Redaction keeps the scheme (mongodb://<host>), never the address.
    assert pull["error"].startswith("test: ") and mongo_url not in pull["error"] and "127.0.0.1" not in pull["error"]
    after = _dataset(ds)
    assert after["status"] == "ready" and after["status_message"] == "Ready" and after["error"] is None
    assert after["warnings"] == row["warnings"] and after["row_count"] == row["row_count"]
    assert os.stat(row["parquet_path"]).st_mtime_ns == mtime
    assert _raw_rows(ds) == 750
    assert (
        db.fetch_all("SELECT id, name, null_pct, ordinal FROM meta.fields WHERE dataset_id = %s ORDER BY id", (ds,))
        == fields
    )
    assert _leftovers(ds) == [] and not _new_table(ds)


def test_refresh_failure_in_the_swap_rolls_back(source, monkeypatch):
    ds = _connect(source)
    row = _dataset(ds)
    mtime = os.stat(row["parquet_path"]).st_mtime_ns
    source.massive.insert_many(_fresh_docs(200, 0.6))

    def broken(*a, **kw):
        raise RuntimeError("drift broke")

    monkeypatch.setattr(refresh, "drift_by_family", broken)

    pull = _refresh(ds)

    assert pull["status"] == "failed" and pull["error"] == "drift broke"
    after = _dataset(ds)
    assert after["status"] == "ready" and after["row_count"] == 750
    assert os.stat(row["parquet_path"]).st_mtime_ns == mtime and _raw_rows(ds) == 750
    assert _field(ds, "fresh") is None
    assert _leftovers(ds) == [] and not _new_table(ds)


def test_refresh_without_a_finalized_baseline_has_no_drift(source):
    ds = _connect(source, finalize=False)
    source.massive.insert_many(_fresh_docs(200, 0.6))

    pull = _refresh(ds)

    assert pull["status"] == "done", pull["error"]
    assert db.fetch_one("SELECT drift IS NULL AS absent FROM meta.dataset_pulls WHERE id = %s", (pull["id"],))["absent"]
    shown = db.fetch_one(f"SELECT {SELECT_DATASET} FROM meta.datasets d {recipes.CURRENT_JOIN} WHERE d.id = %s", (ds,))
    assert shown["last_pull"]["has_drift"] is False and shown["last_pull"]["status"] == "done"
    assert _field(ds, "fresh") is not None


def test_a_second_refresh_swaps_again(source):
    ds = _connect(source, finalize=False)
    _add_analysis(ds)
    assert _refresh(ds)["status"] == "done"
    # The same data and decisions, but the data was replaced: still out of date.
    assert [a["stale"] for a in _analyses(ds)] == [True]
    source.massive.insert_many(_fresh_docs(10, 0))
    pull = _refresh(ds)
    assert pull["status"] == "done", pull["error"]
    assert _raw_rows(ds) == 760 and not _new_table(ds)


def test_a_restart_puts_a_refreshing_dataset_back_to_ready(database):
    from app import jobs

    ds = db.fetch_one(
        """INSERT INTO meta.datasets (name, file_format, status, created_by)
           VALUES ('connected', 'mongodb', 'refreshing', %s) RETURNING id""",
        (_user(),),
    )["id"]
    db.execute("INSERT INTO meta.dataset_pulls (dataset_id, status) VALUES (%s, 'running')", (ds,))
    jobs.sweep()
    row = _dataset(ds)
    assert row["status"] == "ready" and row["status_message"] == "Ready" and row["error"] is None
    pull = db.fetch_one("SELECT * FROM meta.dataset_pulls WHERE dataset_id = %s", (ds,))
    assert pull["status"] == "failed" and pull["finished_at"] is not None
