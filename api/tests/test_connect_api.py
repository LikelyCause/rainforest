"""Creating a connected dataset over HTTP, against a real MongoDB and Postgres.

Needs RAINFOREST_TEST_MONGO_URL=mongodb://host:27017 and RAINFOREST_TEST_DATABASE_URL.
"""

import os
import time

import pytest

from .conftest import set_phase
from .mongo_helpers import use_sources

DB_URL = os.environ.get("RAINFOREST_TEST_DATABASE_URL")


@pytest.fixture(scope="module")
def clients(mongo, mongo_url, tmp_path_factory, login_as):
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

    with use_sources(f"test={mongo_url}"), TestClient(app) as lead:
        login_as(lead, "lead")
        member, viewer = TestClient(app), TestClient(app)
        login_as(member, "member")
        login_as(viewer, "viewer")
        yield {"lead": lead, "member": member, "viewer": viewer}
    db.close_pool()
    mp.undo()
    get_settings.cache_clear()


@pytest.fixture
def lead(clients):
    return clients["lead"]


def _body(mongo, *collections, name="Vehicles", **source):
    return {
        "name": name,
        "source": {
            "connection": "test",
            "database": mongo.name,
            "collections": [{"name": c} if isinstance(c, str) else c for c in collections],
            **source,
        },
    }


def _wait(client, ds_id: int, timeout: float = 120) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        ds = client.get(f"/api/datasets/{ds_id}").json()
        if ds["status"] in ("ready", "failed"):
            return ds
        assert time.monotonic() < deadline, f"timed out: {ds}"
        time.sleep(0.3)


@pytest.fixture(scope="module")
def connected(clients, mongo):
    """Two collections connected by a member, waited on until ready."""
    r = clients["member"].post(
        "/api/datasets/connect",
        json=_body(mongo, "curated", {"name": "massive", "label": "bulk"}, add_source=True, dedupe=True),
    )
    assert r.status_code == 202, r.text
    first = r.json()
    ds = _wait(clients["lead"], first["id"])
    yield first, ds
    clients["lead"].delete(f"/api/datasets/{first['id']}")


def test_connect_returns_the_row_with_its_source_and_no_uri(connected):
    first, _ = connected
    assert first["status"] in ("pending", "ingesting") and first["file_format"] == "mongodb"
    assert first["source"]["connection"] == "test" and first["source"]["kind"] == "mongodb"
    assert first["source"]["collections"] == [
        {"name": "curated", "label": "curated", "filter": None, "projection": None},
        {"name": "massive", "label": "bulk", "filter": None, "projection": None},
    ]
    assert first["source"]["add_source"] is True and first["source"]["dedupe"] is True
    assert first["source"]["match_on"] is None
    assert "mongodb://" not in str(first)


def test_connected_dataset_becomes_ready_with_a_done_pull(connected, lead):
    _, ds = connected
    assert ds["status"] == "ready", ds["error"]
    assert ds["row_count"] == 750 and ds["file_format"] == "mongodb"
    assert ds["last_pull"]["status"] == "done" and ds["last_pull"]["rows"] == 750
    assert ds["last_pull"]["has_drift"] is False and ds["last_pull"]["finished_at"] is not None
    assert "mongodb://" not in lead.get(f"/api/datasets/{ds['id']}").text
    listed = next(d for d in lead.get("/api/datasets").json() if d["id"] == ds["id"])
    assert listed["source"] == ds["source"] and listed["last_pull"] == ds["last_pull"]


def test_uploaded_datasets_have_no_source(lead):
    ds = lead.post("/api/datasets/sample", json={"rows": 1000, "fields": 30}).json()
    try:
        assert "source" in ds and ds["source"] is None and ds["last_pull"] is None
    finally:
        _wait(lead, ds["id"])
        lead.delete(f"/api/datasets/{ds['id']}")


def test_pulls_lists_the_pull_newest_first(connected, lead):
    _, ds = connected
    pulls = lead.get(f"/api/datasets/{ds['id']}/pulls").json()
    assert len(pulls) == 1
    p = pulls[0]
    assert p["status"] == "done" and p["rows"] == 750 and p["error"] is None and p["drift"] is None
    assert p["created_by_name"] == "Test member" and p["id"] == ds["last_pull"]["id"]
    assert [s["label"] for s in p["per_source"]] == ["curated", "bulk"]
    assert "compared_on" in p["flatten_notes"] and p["started_at"] and p["finished_at"]


def test_pulls_is_readable_by_a_viewer_and_404_for_an_unknown_dataset(connected, clients):
    _, ds = connected
    assert clients["viewer"].get(f"/api/datasets/{ds['id']}/pulls").status_code == 200
    assert clients["lead"].get("/api/datasets/999999/pulls").status_code == 404


def test_a_viewer_cannot_connect(clients, mongo):
    r = clients["viewer"].post("/api/datasets/connect", json=_body(mongo, "curated"))
    assert r.status_code == 403


def test_unknown_connection_is_404(lead, mongo):
    body = _body(mongo, "curated")
    body["source"]["connection"] = "nope"
    r = lead.post("/api/datasets/connect", json=body)
    assert r.status_code == 404 and r.json()["detail"] == "No source named nope"


@pytest.mark.parametrize("field", ["filter", "projection"])
def test_bad_filter_or_projection_is_422_naming_the_collection(lead, mongo, field):
    r = lead.post("/api/datasets/connect", json=_body(mongo, {"name": "curated", field: "[1, 2]"}))
    assert r.status_code == 422 and r.json()["detail"] == f"curated: {field} is not a JSON document"


def test_duplicate_labels_are_422(lead, mongo):
    r = lead.post("/api/datasets/connect", json=_body(mongo, "curated", {"name": "massive", "label": "curated"}))
    assert r.status_code == 422 and r.json()["detail"] == "Collection labels must be unique: curated"
    same = lead.post("/api/datasets/connect", json=_body(mongo, "curated", "curated"))
    assert same.status_code == 422 and "must be unique" in same.json()["detail"]


def test_the_reserved_label_is_422(lead, mongo):
    r = lead.post("/api/datasets/connect", json=_body(mongo, {"name": "curated", "label": "__source_rank"}))
    assert r.status_code == 422 and "__source_rank" in r.json()["detail"]


def test_several_collections_cannot_turn_the_source_column_off(lead, mongo):
    r = lead.post("/api/datasets/connect", json=_body(mongo, "curated", "massive", add_source=False))
    assert r.status_code == 422 and "source" in r.json()["detail"]


def test_over_the_budget_is_422_with_the_estimate(lead, mongo, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("SOURCE_MAX_CELLS", "10")
    get_settings.cache_clear()
    try:
        before = len(lead.get("/api/datasets").json())
        r = lead.post("/api/datasets/connect", json=_body(mongo, "curated"))
        assert r.status_code == 422
        detail = r.json()["detail"]
        assert detail["message"] == "Too large" and detail["max"] == 10 and detail["estimated"] > 10
        assert len(lead.get("/api/datasets").json()) == before, "no dataset is left behind"
    finally:
        monkeypatch.delenv("SOURCE_MAX_CELLS")
        get_settings.cache_clear()


def test_match_on_must_name_a_sampled_column(lead, mongo):
    before = len(lead.get("/api/datasets").json())
    r = lead.post("/api/datasets/connect", json=_body(mongo, "curated", "massive", match_on=["vehicle", "nope"]))
    assert r.status_code == 422
    assert r.json()["detail"] == "Match on column not found in a sample of up to 200 documents per collection: nope"
    assert len(lead.get("/api/datasets").json()) == before, "no dataset is left behind"


def test_match_on_is_stored_trimmed_and_once(lead, mongo):
    body = _body(mongo, "curated", "massive", name="Matched", match_on=[" engine", "vehicle", "engine", " "])
    r = lead.post("/api/datasets/connect", json=body)
    assert r.status_code == 202, r.text
    try:
        assert r.json()["source"]["match_on"] == ["engine", "vehicle"]
    finally:
        _wait(lead, r.json()["id"])
        lead.delete(f"/api/datasets/{r.json()['id']}")


def test_filters_round_trip_as_the_strings_sent(lead, mongo):
    flt = '{"vehicle": "van", "when": {"$gte": {"$date": "2020-01-01T00:00:00Z"}}}'
    r = lead.post(
        "/api/datasets/connect",
        json=_body(mongo, {"name": "curated", "filter": flt, "projection": '{"vehicle": 1}'}, name="Vans"),
    )
    assert r.status_code == 202, r.text
    try:
        c = r.json()["source"]["collections"][0]
        assert c["filter"] is not None and "$date" in c["filter"] and c["projection"] == '{"vehicle": 1}'
        assert r.json()["source"]["add_source"] is False
    finally:
        _wait(lead, r.json()["id"])
        lead.delete(f"/api/datasets/{r.json()['id']}")


def test_recipe_id_is_for_leads_and_needs_a_finalized_baseline(connected, clients, mongo):
    first, ds = connected
    body = {**_body(mongo, "curated", name="From recipe"), "recipe_id": None}
    recipe = set_phase(ds["id"], "finalized")
    body["recipe_id"] = recipe["id"]
    assert clients["member"].post("/api/datasets/connect", json=body).status_code == 403
    body["recipe_id"] = 999999
    assert clients["lead"].post("/api/datasets/connect", json=body).status_code == 404
    body["recipe_id"] = recipe["id"]
    r = clients["lead"].post("/api/datasets/connect", json=body)
    assert r.status_code == 202, r.text
    try:
        new = _wait(clients["lead"], r.json()["id"])
        assert new["status"] == "ready", new["error"]
        assert new["row_count"] == 300 and new["last_pull"]["status"] == "done"
    finally:
        clients["lead"].delete(f"/api/datasets/{r.json()['id']}")
