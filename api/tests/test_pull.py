"""Pulling MongoDB collections into a dataset's Parquet copy: chunks, assembly, deduplication and the connect job.

The fake-client tests need no server. The rest need RAINFOREST_TEST_MONGO_URL, and the job tests also
RAINFOREST_TEST_DATABASE_URL.
"""

import os
from datetime import datetime, timezone

import polars as pl
import pytest
from bson.objectid import ObjectId
from pymongo.errors import PyMongoError

from app import db
from app.config import get_settings
from app.sources import BudgetExceeded, CollectionSpec, PullTimeout, as_text, mongodb, pull
from app.sources.mongodb import MongoSource
from app.sources.pull import assemble, run_connect

from .mongo_helpers import use_sources

DB_URL = os.environ.get("RAINFOREST_TEST_DATABASE_URL")


class _Cursor:
    def __init__(self, docs):
        self.docs = docs
        self.closed = False

    def limit(self, n):
        return _Cursor(self.docs[:n])

    def __iter__(self):
        return iter(self.docs)

    def close(self):
        self.closed = True


class _Collection:
    def __init__(self, docs):
        self.docs = docs
        self.finds: list[dict] = []

    def find(self, flt=None, projection=None, **kw):
        self.finds.append(kw)
        return _Cursor(self.docs)

    def estimated_document_count(self):
        return len(self.docs)


class _Client:
    def __init__(self, colls):
        self.colls = colls

    def __getitem__(self, name):
        return self.colls


def _fake(**colls) -> MongoSource:
    src = MongoSource("vehicles", "mongodb://u:pw@db1.example.com:27017/")
    src._client = _Client({name: _Collection(docs) for name, docs in colls.items()})
    return src


def _chunks(path) -> list[str]:
    return sorted(p.name for p in path.glob("chunk_*.parquet"))


def test_pull_keeps_column_names_stable_across_documents_and_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(mongodb, "CHUNK_DOCS", 2)
    docs = [
        {"a": {"c.d": 1}},  # a.c_d, renamed from a.c.d
        {"a": {"c_d": 2}},  # a later literal key can't take the name a.c.d already has
        {"a": {"c.d": 3, "c_d": 4}},  # one document with both keeps the names they got before
        {"x_y": 5, "x.y": 6},  # both new in one document: the literal key keeps the base name
    ]
    r = _fake(c=docs).pull("d", [CollectionSpec("c")], tmp_path, 10**9, 60, None)
    assert _chunks(tmp_path) == ["chunk_c_0.parquet", "chunk_c_1.parquet"]
    df = pl.concat([pl.read_parquet(tmp_path / n) for n in _chunks(tmp_path)], how="diagonal_relaxed")
    assert df.columns == ["a.c_d", "a.c_d_2", "x_y", "x_y_2"]
    assert df["a.c_d"].to_list() == [1, None, 3, None]
    assert df["a.c_d_2"].to_list() == [None, 2, 4, None]
    assert df["x_y"].to_list()[3] == 5 and df["x_y_2"].to_list()[3] == 6
    assert r.flatten_notes["renamed"] == {"a.c.d": "a.c_d", "a.c_d": "a.c_d_2", "x.y": "x_y_2"}
    assert r.per_source == [{"label": "c", "documents": 4}]


def test_an_empty_key_is_one_column_across_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(mongodb, "CHUNK_DOCS", 2)
    docs = [{"": i, "x": i, "a": {"": -i}} for i in range(5)]
    r = _fake(c=docs).pull("d", [CollectionSpec("c")], tmp_path, 10**9, 60, None)
    dest = tmp_path / "raw.parquet"
    assemble(tmp_path, [CollectionSpec("c")], False, False, None, dest, identifiers=r.identifiers)
    df = pl.read_parquet(dest)
    assert df.columns == ["_empty", "x", "a._empty"]
    assert df["_empty"].to_list() == [0, 1, 2, 3, 4] and df["a._empty"].to_list() == [0, -1, -2, -3, -4]
    assert r.flatten_notes["renamed"] == {"": "_empty", "a.": "a._empty"}
    assert pull.warnings_for(r.flatten_notes, {}, False) == ['Keys renamed: "" → _empty, a. → a._empty']


def test_pull_detects_mixed_types_across_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(mongodb, "CHUNK_DOCS", 2)
    when = datetime(2025, 3, 1, 12, 30)
    docs = [
        {"v": 1.5, "ok": True, "when": when, "tags": [1, 2]},
        {"v": 2, "ok": False, "when": when},
        {"v": "n/a", "ok": True},
        {"v": None, "flag": True},
        {"flag": 1, "when": "soon"},  # a boolean among numbers is a mixed column too
    ]
    progress: list[tuple[str, int]] = []
    r = _fake(c=docs).pull("d", [CollectionSpec("c", "cars")], tmp_path, 10**9, 60, lambda *a: progress.append(a))
    assert r.flatten_notes["mixed"] == ["v", "when", "flag"] and r.flatten_notes["arrays"] == ["tags"]
    assert progress[-1] == ("cars", 5)
    dest = tmp_path / "raw.parquet"
    out = assemble(tmp_path, [CollectionSpec("c", "cars")], False, False, None, dest, identifiers=r.identifiers)
    df = pl.read_parquet(dest)
    assert out["rows"] == 5 and _chunks(tmp_path) == []
    assert df["v"].dtype == pl.String and df["v"].to_list() == ["1.5", "2", "n/a", None, None]
    assert df["flag"].to_list() == [None, None, None, "true", "1"]
    assert df["ok"].dtype == pl.Boolean
    # A datetime reads the same whether its chunk was written before or after the column turned out mixed.
    assert df["when"].to_list() == ["2025-03-01T12:30:00+00:00"] * 2 + [None, None, "soon"]
    assert as_text(when) == "2025-03-01T12:30:00+00:00"
    assert df["tags"][0] == "[1, 2]"


def test_pull_keeps_utc_datetimes_and_booleans(tmp_path):
    when = datetime(2025, 3, 1, 12, 30)
    src = _fake(c=[{"when": when, "ok": True}, {"when": None, "ok": False}])
    src.pull("d", [CollectionSpec("c")], tmp_path, 10**9, 60, None)
    assemble(tmp_path, [CollectionSpec("c")], False, False, None, tmp_path / "raw.parquet", identifiers=[])
    df = pl.read_parquet(tmp_path / "raw.parquet")
    assert df.schema == {"when": pl.Datetime("us", "UTC"), "ok": pl.Boolean}
    assert df["when"][0] == when.replace(tzinfo=timezone.utc)


def test_a_chunk_of_only_nulls_keeps_the_columns_type(tmp_path, monkeypatch):
    monkeypatch.setattr(mongodb, "CHUNK_DOCS", 2)
    when = datetime(2025, 3, 1, 12, 30)
    typed = {"w": when, "ok": True, "n": 3, "t": "x"}
    nulls = dict.fromkeys(typed)
    src = _fake(c=[typed, typed, nulls, nulls])
    src.pull("d", [CollectionSpec("c")], tmp_path, 10**9, 60, None)
    expected = {"w": pl.Datetime("us", "UTC"), "ok": pl.Boolean, "n": pl.Int64, "t": pl.String}
    assert pl.read_parquet_schema(tmp_path / "chunk_c_1.parquet") == expected
    assemble(tmp_path, [CollectionSpec("c")], False, False, None, tmp_path / "raw.parquet", identifiers=[])
    df = pl.read_parquet(tmp_path / "raw.parquet")
    assert df.schema == expected and df["w"].to_list()[2:] == [None, None]


def test_pull_timeout(tmp_path):
    src = _fake(c=[{"a": 1}])
    with pytest.raises(PullTimeout):
        src.pull("d", [CollectionSpec("c")], tmp_path, 10**9, 0, None)


def test_pull_reads_with_the_batch_size_and_a_server_time_limit(tmp_path):
    src = _fake(c=[{"a": 1}])
    src.pull("d", [CollectionSpec("c")], tmp_path, 10**9, 60, None)
    kw = src.client["d"]["c"].finds[-1]
    assert kw["batch_size"] == 5_000 and kw["no_cursor_timeout"] is False and 0 < kw["max_time_ms"] <= 60_000


def test_assemble_dedupe_keeps_the_first_collections_row_and_skips_identifiers(tmp_path):
    a = [{"_id": f"a{i}", "code_id": f"k{i}", "x": i % 3, "y": "s"} for i in range(600)]
    b = [{"_id": "b0", "code_id": "z", "x": 0, "y": "s", "only_b": 1}, {"_id": "b1", "code_id": "q", "x": 9, "y": "t"}]
    src = _fake(a=a, b=b)
    specs = [CollectionSpec("a"), CollectionSpec("b", "second")]
    r = src.pull("d", specs, tmp_path, 10**9, 60, None)
    out = assemble(tmp_path, specs, True, True, None, tmp_path / "raw.parquet", identifiers=r.identifiers)
    df = pl.read_parquet(tmp_path / "raw.parquet")
    # code_id has an identifier's name, so rows match on x and y alone: three in a, and b's (9, "t").
    assert out["compared_on"] == ["x", "y"] == src.preview("d", specs).compared_on
    assert out["removed_duplicates"] == {"a": 597, "second": 1}
    assert df.columns[0] == "source" and df["source"].to_list() == ["a", "a", "a", "second"]
    assert df["_id"].to_list() == ["a0", "a1", "a2", "b1"]


def test_the_pull_flags_identifiers_by_the_previews_rule(tmp_path):
    # Too few values for profiling to call either one an identifier; the preview decides by name and by BSON type.
    docs = [{"_id": ObjectId(), "serial": {"key": "abcd"[i % 4]}, "owner": ObjectId(), "x": i % 2} for i in range(40)]
    src = _fake(c=docs)
    specs = [CollectionSpec("c")]
    assert src.preview("d", specs).compared_on == ["x"]
    r = src.pull("d", specs, tmp_path, 10**9, 60, None)
    assert r.identifiers == ["_id", "serial.key", "owner"]
    out = assemble(tmp_path, specs, False, True, None, tmp_path / "raw.parquet", identifiers=r.identifiers)
    assert out["compared_on"] == ["x"] and out["rows"] == 2


def test_an_identifier_with_other_values_too_is_compared_as_the_preview_does(tmp_path, monkeypatch):
    monkeypatch.setattr(mongodb, "CHUNK_DOCS", 2)
    docs = [{"ref": ObjectId(), "x": 1}, {"ref": ObjectId(), "x": 1}, {"ref": "legacy", "x": 2}]
    src = _fake(c=docs)
    assert src.preview("d", [CollectionSpec("c")]).compared_on == ["ref", "x"]
    assert src.pull("d", [CollectionSpec("c")], tmp_path, 10**9, 60, None).identifiers == []


def test_a_collection_without_documents_does_not_empty_the_comparison(tmp_path):
    src = _fake(a=[{"_id": f"a{i}", "x": i % 2, "y": "s"} for i in range(6)], empty=[])
    specs = [CollectionSpec("a"), CollectionSpec("empty")]
    assert src.preview("d", specs).compared_on == ["x", "y"]
    r = src.pull("d", specs, tmp_path, 10**9, 60, None)
    out = assemble(tmp_path, specs, True, True, None, tmp_path / "raw.parquet", identifiers=r.identifiers)
    assert out["compared_on"] == ["x", "y"]


def test_assemble_match_on_must_name_columns(tmp_path):
    src = _fake(a=[{"x": 1}])
    src.pull("d", [CollectionSpec("a")], tmp_path, 10**9, 60, None)
    with pytest.raises(ValueError, match="nope"):
        assemble(tmp_path, [CollectionSpec("a")], False, True, ["x", "nope"], tmp_path / "raw.parquet", identifiers=[])


# ---- against a real MongoDB -------------------------------------------------------------------------------


def _source(mongo_url) -> MongoSource:
    return MongoSource("test", mongo_url)


SPECS = [CollectionSpec("curated"), CollectionSpec("massive")]


def test_pull_writes_chunks_and_notes_mixed_and_arrays(mongo, mongo_url, tmp_path):
    r = _source(mongo_url).pull(mongo.name, SPECS, tmp_path, 10**9, 60, None)
    assert "rpm.mean" in r.flatten_notes["mixed"] and "tags" in r.flatten_notes["arrays"]
    assert r.per_source == [{"label": "curated", "documents": 300}, {"label": "massive", "documents": 500}]
    assert _chunks(tmp_path) == ["chunk_curated_0.parquet", "chunk_massive_0.parquet"]


def test_assemble_adds_source_and_dedupes_on_shared_columns(mongo, mongo_url, tmp_path):
    r = _source(mongo_url).pull(mongo.name, SPECS, tmp_path, 10**9, 60, None)
    dest = tmp_path / "raw.parquet"
    out = assemble(tmp_path, SPECS, True, True, None, dest, identifiers=r.identifiers)
    df = pl.read_parquet(dest)
    assert df["source"].unique(maintain_order=True).to_list() == ["curated", "massive"]
    assert out["removed_duplicates"] == {"massive": 50} and out["rows"] == 750 == df.height
    assert "_id" not in out["compared_on"] and "extra.depth.key" not in out["compared_on"]
    assert set(out["compared_on"]) == {"vehicle", "engine", "rpm.mean", "rpm.max", "tags", "when"}
    assert df["rpm.mean"].dtype == pl.String and "extra.depth.key" in df.columns
    assert _source(mongo_url).preview(mongo.name, SPECS).compared_on == out["compared_on"]


def test_assemble_match_on_overrides(mongo, mongo_url, tmp_path):
    r = _source(mongo_url).pull(mongo.name, SPECS, tmp_path, 10**9, 60, None)
    dest = tmp_path / "raw.parquet"
    out = assemble(tmp_path, SPECS, True, True, ["vehicle", "engine"], dest, identifiers=r.identifiers)
    assert out["compared_on"] == ["vehicle", "engine"]
    assert out["rows"] == 25  # five vehicles by five engines, every pair seen in curated first
    assert sum(out["removed_duplicates"].values()) == 800 - 25


def test_budget_refuses_before_reading(mongo, mongo_url, tmp_path):
    with pytest.raises(BudgetExceeded) as e:
        _source(mongo_url).pull(mongo.name, SPECS, tmp_path, 10, 60, None)
    assert e.value.max == 10 and e.value.estimated > 10
    assert _chunks(tmp_path) == []


# ---- the connect job --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def database(tmp_path_factory):
    if not DB_URL:
        pytest.skip("set RAINFOREST_TEST_DATABASE_URL to run the pull job tests")
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


def _connected(database_name: str, collections: list[dict], **kw) -> int:
    from app import recipes

    with db.connection() as conn:
        user = conn.execute("SELECT id FROM meta.users ORDER BY id LIMIT 1").fetchone()["id"]
        ds = conn.execute(
            """INSERT INTO meta.datasets (name, file_format, status, created_by)
               VALUES ('connected', 'mongodb', 'pending', %s) RETURNING id""",
            (user,),
        ).fetchone()["id"]
        recipes.create_first(conn, ds, user)
        conn.execute(
            """INSERT INTO meta.dataset_sources
                   (dataset_id, kind, connection, database, collections, add_source, dedupe, match_on, created_by)
               VALUES (%s, 'mongodb', 'test', %s, %s, %s, %s, %s, %s)""",
            (
                ds,
                database_name,
                db.jsonb(collections),
                kw.get("add_source", True),
                kw.get("dedupe", True),
                kw.get("match_on"),
                user,
            ),
        )
    return ds


def _pulls(ds: int) -> list[dict]:
    return db.fetch_all("SELECT * FROM meta.dataset_pulls WHERE dataset_id = %s ORDER BY id", (ds,))


def test_run_connect_ends_ready_with_a_pull_row(database, mongo, mongo_url):
    ds = _connected(mongo.name, [{"name": "curated"}, {"name": "massive"}])
    with use_sources(f"test={mongo_url}"):
        run_connect(ds)
    row = db.fetch_one("SELECT * FROM meta.datasets WHERE id = %s", (ds,))
    assert row["status"] == "ready", row["error"]
    assert row["row_count"] == 750 and row["file_format"] == "mongodb"
    fields = [f["name"] for f in db.fetch_all("SELECT name FROM meta.fields WHERE dataset_id = %s", (ds,))]
    assert "source" in fields and "rpm.mean" in fields
    assert "Mixed types read as text: rpm.mean" in row["warnings"]
    assert "Arrays stored as JSON text: tags" in row["warnings"]
    assert "Duplicates removed: massive 50" in row["warnings"]
    pulls = _pulls(ds)
    assert len(pulls) == 1 and pulls[0]["status"] == "done" and pulls[0]["rows"] == 750
    assert pulls[0]["finished_at"] is not None and pulls[0]["created_by"] == row["created_by"]
    assert pulls[0]["per_source"] == [
        {"label": "curated", "documents": 300, "removed_duplicates": 0},
        {"label": "massive", "documents": 500, "removed_duplicates": 50},
    ]
    assert "rpm.mean" in pulls[0]["flatten_notes"]["mixed"]
    assert not (get_settings().datasets_dir / str(ds) / "pull").exists()


def test_run_connect_applies_filters_projections_and_labels(database, mongo, mongo_url):
    collections = [
        {"name": "curated", "label": "clean", "filter": {"vehicle": "van"}, "projection": {"vehicle": 1, "rpm": 1}},
        {"name": "massive", "label": "bulk", "filter": {"vehicle": "van"}, "projection": {"vehicle": 1, "rpm": 1}},
    ]
    ds = _connected(mongo.name, collections, dedupe=False)
    with use_sources(f"test={mongo_url}"):
        run_connect(ds)
    row = db.fetch_one("SELECT * FROM meta.datasets WHERE id = %s", (ds,))
    assert row["status"] == "ready", row["error"]
    df = pl.read_parquet(row["parquet_path"])
    assert df.columns == ["source", "_id", "vehicle", "rpm.mean", "rpm.max"]
    assert set(df["vehicle"].to_list()) == {"van"} and set(df["source"].to_list()) == {"clean", "bulk"}
    expected = mongo.curated.count_documents({"vehicle": "van"}) + mongo.massive.count_documents({"vehicle": "van"})
    assert row["row_count"] == expected


def test_run_connect_failure_marks_dataset_failed_without_uri(database, monkeypatch):
    uri = "mongodb://reader:s3cret@db1.example.com:27017/"

    def refuse(self, *a, **kw):
        raise PyMongoError(f"db1.example.com:27017: connection refused, Topology: {uri}")

    monkeypatch.setattr(MongoSource, "pull", refuse)
    ds = _connected("vehicles", [{"name": "curated"}])
    with use_sources(f"test={uri}"):
        run_connect(ds)
    row = db.fetch_one("SELECT * FROM meta.datasets WHERE id = %s", (ds,))
    pull = _pulls(ds)[0]
    assert row["status"] == "failed" and pull["status"] == "failed"
    assert row["error"] == pull["error"] and pull["error"].startswith("test: ")
    for secret in ("s3cret", "reader", "db1.example.com"):
        assert secret not in row["error"]
    assert pull["finished_at"] is not None


def test_run_connect_over_budget_fails_and_leaves_no_files(database, mongo, mongo_url, monkeypatch):
    monkeypatch.setenv("SOURCE_MAX_CELLS", "10")
    get_settings.cache_clear()
    ds = _connected(mongo.name, [{"name": "curated"}])
    with use_sources(f"test={mongo_url}"):
        run_connect(ds)
    monkeypatch.delenv("SOURCE_MAX_CELLS")
    get_settings.cache_clear()
    row = db.fetch_one("SELECT * FROM meta.datasets WHERE id = %s", (ds,))
    assert row["status"] == "failed" and "SOURCE_MAX_CELLS" in row["error"]
    assert _pulls(ds)[0]["status"] == "failed"
    assert not list((get_settings().datasets_dir / str(ds)).rglob("*.parquet"))


def test_run_connect_failure_after_the_pull_still_fails_both_rows(database, mongo, mongo_url, monkeypatch):
    def broken(*a, **kw):
        raise RuntimeError("warnings broke")

    monkeypatch.setattr(pull, "warnings_for", broken)
    ds = _connected(mongo.name, [{"name": "curated"}])
    with use_sources(f"test={mongo_url}"):
        run_connect(ds)
    row = db.fetch_one("SELECT * FROM meta.datasets WHERE id = %s", (ds,))
    assert row["status"] == "failed" and row["error"] == "warnings broke"
    assert _pulls(ds)[0]["status"] == "failed" and _pulls(ds)[0]["error"] == "warnings broke"


def test_run_connect_says_when_nothing_was_left_to_compare(database, mongo, mongo_url):
    collections = [
        {"name": "curated", "projection": {"vehicle": 1}},
        {"name": "massive", "projection": {"engine": 1}},
    ]
    ds = _connected(mongo.name, collections)
    with use_sources(f"test={mongo_url}"):
        run_connect(ds)
    row = db.fetch_one("SELECT * FROM meta.datasets WHERE id = %s", (ds,))
    assert row["status"] == "ready", row["error"]
    assert "Duplicates were not removed: no shared columns to compare on" in row["warnings"]
    assert row["row_count"] == 800 and _pulls(ds)[0]["flatten_notes"]["compared_on"] == []


@pytest.mark.parametrize("names", [["curated", "massive"], ["massive"]])
def test_the_previews_compared_on_is_what_the_pull_compares(database, mongo, mongo_url, names):
    # Alone, massive's sparse extra.depth.key has an ID-like name but too few values for profiling to call it one.
    ds = _connected(mongo.name, [{"name": n} for n in names])
    with use_sources(f"test={mongo_url}"):
        run_connect(ds)
    pulled = _pulls(ds)[0]
    assert pulled["status"] == "done", pulled["error"]
    previewed = _source(mongo_url).preview(mongo.name, [CollectionSpec(n) for n in names]).compared_on
    assert "extra.depth.key" not in previewed
    assert pulled["flatten_notes"]["compared_on"] == previewed


def test_warnings_for_dedupe_without_columns_only_when_asked():
    notes = {"arrays": [], "mixed": [], "renamed": {}, "compared_on": []}
    assert pull.warnings_for(notes, {}, dedupe=True) == ["Duplicates were not removed: no shared columns to compare on"]
    assert pull.warnings_for(notes, {}, dedupe=False) == []
    assert pull.warnings_for({**notes, "compared_on": ["x"]}, {}, dedupe=True) == []


def test_run_connect_unknown_connection_fails_with_its_name(database):
    ds = _connected("vehicles", [{"name": "curated"}])
    with use_sources(""):
        run_connect(ds)
    row = db.fetch_one("SELECT * FROM meta.datasets WHERE id = %s", (ds,))
    assert row["status"] == "failed" and "test" in row["error"]


def test_a_restart_fails_the_pull_of_an_interrupted_dataset(database):
    from app import jobs

    ds = _connected("vehicles", [{"name": "curated"}])
    db.execute("UPDATE meta.datasets SET status = 'ingesting', owner = NULL WHERE id = %s", (ds,))
    db.execute("INSERT INTO meta.dataset_pulls (dataset_id, status) VALUES (%s, 'running')", (ds,))
    jobs.sweep()
    assert _pulls(ds)[0]["status"] == "failed" and _pulls(ds)[0]["finished_at"] is not None
    row = db.fetch_one("SELECT status, error FROM meta.datasets WHERE id = %s", (ds,))
    assert row["status"] == "failed" and row["error"] == "Interrupted by an API restart; delete it and connect again."


def test_migration_constraints(database):
    defs = {
        r["conname"]: r["def"]
        for r in db.fetch_all(
            """SELECT conname, pg_get_constraintdef(oid) AS def FROM pg_constraint
                WHERE conrelid = 'meta.datasets'::regclass AND contype = 'c'"""
        )
    }
    assert "'mongodb'" in defs["datasets_file_format_check"] and "'refreshing'" in defs["datasets_status_check"]
    cols = {
        r["column_name"]
        for r in db.fetch_all(
            "SELECT column_name FROM information_schema.columns WHERE table_schema = 'meta' AND table_name = 'exports'"
        )
    }
    assert "pull_id" in cols
