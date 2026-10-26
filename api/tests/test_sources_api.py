"""The sources API against a real MongoDB.

Needs RAINFOREST_TEST_MONGO_URL=mongodb://host:27017 and, for the HTTP layer, RAINFOREST_TEST_DATABASE_URL.
"""

import os

import pytest

from .mongo_helpers import use_sources

DB_URL = os.environ.get("RAINFOREST_TEST_DATABASE_URL")


@pytest.fixture(scope="module")
def client(mongo, mongo_url, tmp_path_factory, login_as):
    if not DB_URL:
        pytest.skip("set RAINFOREST_TEST_DATABASE_URL to run the API tests")
    from fastapi.testclient import TestClient

    from app import db
    from app.config import get_settings

    mp = pytest.MonkeyPatch()
    mp.setenv("DATABASE_URL", DB_URL)
    mp.setenv("DATA_DIR", str(tmp_path_factory.mktemp("data")))
    get_settings.cache_clear()
    db.close_pool()
    from app.main import app

    with use_sources(f"test={mongo_url}"), TestClient(app) as c:
        yield login_as(c)
    db.close_pool()
    mp.undo()
    get_settings.cache_clear()


def _preview(client, mongo, *collections, name="test", **extra):
    return client.post(
        f"/api/sources/{name}/preview", json={"database": mongo.name, "collections": list(collections), **extra}
    )


def test_sources_lists_names_only(client):
    assert client.get("/api/sources").json() == [{"name": "test", "kind": "mongodb"}]


def test_databases(client, mongo):
    body = client.get("/api/sources/test/databases").json()
    assert body["listable"] is True and mongo.name in body["databases"]
    assert "admin" not in body["databases"]


def test_collections_and_counts(client, mongo):
    r = client.get(f"/api/sources/test/databases/{mongo.name}/collections")
    assert r.json() == [{"name": "curated", "documents": 300}, {"name": "massive", "documents": 500}]


def test_preview_flattens_and_estimates(client, mongo):
    r = _preview(client, mongo, {"name": "curated"}, {"name": "massive"}).json()
    names = {c["name"]: c for c in r["columns"]}
    assert (
        names["rpm.mean"]["kind"] == "mixed"
        and names["tags"]["kind"] == "json"
        and names["_id"]["kind"] == "identifier"
    )
    assert names["extra.depth.key"]["present_share"] < 1
    assert r["estimated_rows"] == 800 and r["budget"]["fits"] is True
    assert r["budget"]["max_cells"] == 600_000_000 and r["estimated_cells"] > 800
    assert [s["name"] for s in r["per_source"]] == ["curated", "massive"]


def test_preview_names_the_columns_deduplication_compares_on(client, mongo):
    r = _preview(client, mongo, {"name": "curated"}, {"name": "massive"}).json()
    # `_id` is an identifier and `extra.depth.key` is only in massive.
    assert r["compared_on"] == ["vehicle", "engine", "rpm.mean", "rpm.max", "tags", "when"]
    # Alone, massive has `extra.depth.key` everywhere, but the preview flags it as an identifier by its name.
    alone = _preview(client, mongo, {"name": "massive"}).json()
    assert alone["compared_on"] == r["compared_on"]


def test_preview_compares_on_match_on_when_given(client, mongo):
    r = _preview(client, mongo, {"name": "curated"}, {"name": "massive"}, match_on=[" engine", "vehicle", "engine"])
    assert r.status_code == 200, r.text
    assert r.json()["compared_on"] == ["engine", "vehicle"]
    assert _preview(client, mongo, {"name": "curated"}, match_on=[]).json()["compared_on"][0] == "vehicle"


def test_preview_refuses_a_match_on_column_the_sample_lacks(client, mongo):
    r = _preview(client, mongo, {"name": "curated"}, {"name": "massive"}, match_on=["vehicle", "nope"])
    assert r.status_code == 422
    assert r.json()["detail"] == "Match on column not found in a sample of up to 200 documents per collection: nope"


def test_preview_applies_a_filter_and_projection(client, mongo):
    spec = {"name": "curated", "filter": '{"vehicle": {"$exists": true}}', "projection": '{"vehicle": 1, "rpm": 1}'}
    r = _preview(client, mongo, spec).json()
    assert {c["name"] for c in r["columns"]} == {"_id", "vehicle", "rpm.mean", "rpm.max"}
    assert r["estimated_rows"] == 300


def test_preview_reads_extended_json_filters(client, mongo):
    spec = {"name": "curated", "filter": '{"when": {"$gte": {"$date": "2999-01-01T00:00:00Z"}}}'}
    assert _preview(client, mongo, spec).json()["estimated_rows"] == 0


def test_budget_does_not_fit_when_the_cells_exceed_it(client, mongo, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("SOURCE_MAX_CELLS", "100")
    get_settings.cache_clear()
    try:
        r = _preview(client, mongo, {"name": "curated"}).json()
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()
    assert r["budget"] == {"max_cells": 100, "fits": False}


@pytest.mark.parametrize("field", ["filter", "projection"])
@pytest.mark.parametrize(
    "value",
    [
        "{bad",
        "[1, 2]",
        "3",
        '"x"',
        "null",
        # Extended JSON wrappers whose decoders raise IndexError, InvalidId, InvalidOperation and TypeError
        '{"a": {"$date": "x"}}',
        '{"a": {"$oid": "zz"}}',
        '{"a": {"$numberDecimal": "q"}}',
        '{"a": {"$timestamp": 5}}',
    ],
)
def test_preview_rejects_a_bad_document(client, mongo, field, value):
    r = _preview(client, mongo, {"name": "curated"}, {"name": "massive", field: value})
    assert r.status_code == 422 and r.json()["detail"] == f"massive: {field} is not a JSON document"


def test_preview_rejects_a_bad_filter(client, mongo):
    r = _preview(client, mongo, {"name": "curated", "filter": "{bad"})
    assert r.status_code == 422 and "curated: filter" in r.json()["detail"]


def test_a_bad_document_is_refused_before_mongodb_is_touched(client, mongo):
    r = client.post(
        "/api/sources/test/preview",
        json={"database": "no_such_db_" + mongo.name, "collections": [{"name": "x"}, {"name": "y", "filter": "{"}]},
    )
    assert r.status_code == 422 and r.json()["detail"] == "y: filter is not a JSON document"


def test_preview_needs_a_collection(client, mongo):
    assert _preview(client, mongo).status_code == 422


def test_unknown_source_is_404(client, mongo):
    for r in (
        client.get("/api/sources/nope/databases"),
        client.get(f"/api/sources/nope/databases/{mongo.name}/collections"),
        _preview(client, mongo, {"name": "curated"}, name="nope"),
    ):
        assert r.status_code == 404 and r.json()["detail"] == "No source named nope"


def test_a_viewer_may_read(client, mongo, login_as):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as viewer:
        login_as(viewer, role="viewer")
        assert viewer.get("/api/sources").status_code == 200
        assert viewer.get(f"/api/sources/test/databases/{mongo.name}/collections").status_code == 200
        assert _preview(viewer, mongo, {"name": "curated"}).status_code == 200


def test_signing_in_is_required(client):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as anon:
        assert anon.get("/api/sources").status_code == 401
        body = {"database": "d", "collections": [{"name": "c"}]}
        assert anon.post("/api/sources/test/preview", json=body).status_code == 401


def test_errors_never_carry_the_uri(client, monkeypatch):
    monkeypatch.setattr("app.sources.mongodb.CONNECT_TIMEOUT_MS", 300)  # nothing listens on port 1
    with use_sources("test=mongodb://u:p@127.0.0.1:1/"):
        for r in (
            client.get("/api/sources/test/databases"),
            client.get("/api/sources/test/databases/d/collections"),
            client.post("/api/sources/test/preview", json={"database": "d", "collections": [{"name": "c"}]}),
        ):
            assert r.status_code == 502, r.text
            assert "u:p" not in r.text and "mongodb://u" not in r.text
            assert r.json()["detail"].startswith("test: ")
