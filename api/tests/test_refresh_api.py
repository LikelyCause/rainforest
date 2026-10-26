"""Refreshing a connected dataset over HTTP: who may, when it is refused, the lock it holds while it runs, and the pull
an export records. Needs RAINFOREST_TEST_MONGO_URL and RAINFOREST_TEST_DATABASE_URL.
"""

import json
import os
import time
from contextlib import contextmanager

import pytest

from app import db, jobs
from app.modeling import mlflow_log

from .conftest import set_phase
from .mongo_helpers import seed, use_sources
from .test_mlflow import _model_dir, _run_row, file_store  # noqa: F401  (fixture)

DB_URL = os.environ.get("RAINFOREST_TEST_DATABASE_URL")
LOCKED = "A refresh of this dataset is running; try again once it finishes."


@pytest.fixture(scope="module")
def clients(mongo, mongo_url, tmp_path_factory, login_as):
    if not DB_URL:
        pytest.skip("set RAINFOREST_TEST_DATABASE_URL to run the API tests")
    from fastapi.testclient import TestClient

    from app.config import get_settings

    mp = pytest.MonkeyPatch()
    mp.setenv("DATABASE_URL", DB_URL)
    mp.setenv("DATA_DIR", str(tmp_path_factory.mktemp("data")))
    get_settings.cache_clear()
    db.close_pool()
    from app.main import app

    with use_sources(f"test={mongo_url}"), TestClient(app) as lead:
        login_as(lead, "lead")
        member = TestClient(app)
        login_as(member, "member")
        yield {"lead": lead, "member": member}
    db.close_pool()
    mp.undo()
    get_settings.cache_clear()
    seed(mongo)


@pytest.fixture
def lead(clients):
    return clients["lead"]


def _wait(client, ds_id: int, timeout: float = 120) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        ds = client.get(f"/api/datasets/{ds_id}").json()
        if ds["status"] in ("ready", "failed"):
            return ds
        assert time.monotonic() < deadline, f"timed out: {ds}"
        time.sleep(0.3)


def _wait_export(client, export_id: int, timeout: float = 120) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        e = client.get(f"/api/exports/{export_id}").json()
        if e["status"] in ("done", "failed"):
            return e
        assert time.monotonic() < deadline, f"timed out: {e}"
        time.sleep(0.3)


def _connected(lead, mongo, name: str, drop: tuple[str, ...] = ()) -> int:
    """`curated` connected by the lead, `vehicle` its target and `drop` dropped, with a finalized Trees baseline."""
    r = lead.post(
        "/api/datasets/connect",
        json={
            "name": name,
            "source": {"connection": "test", "database": mongo.name, "collections": [{"name": "curated"}]},
        },
    )
    assert r.status_code == 202, r.text
    ds_id = r.json()["id"]
    assert _wait(lead, ds_id)["status"] == "ready"
    assert lead.patch(f"/api/datasets/{ds_id}", json={"target_columns": ["vehicle"]}).status_code == 200
    assert _wait(lead, ds_id)["status"] == "ready"
    for n in drop:
        db.execute("UPDATE meta.field_decisions SET action = 'drop' WHERE field_id = %s", (_field(ds_id, n)["id"],))
    set_phase(ds_id, "finalized")
    return ds_id


@pytest.fixture(scope="module")
def ds(clients, mongo):
    ds_id = _connected(clients["lead"], mongo, "Refreshed")
    yield ds_id
    clients["lead"].delete(f"/api/datasets/{ds_id}")


@contextmanager
def _held(lead, ds_id: int):
    """A refresh started over HTTP with its job held back, so the dataset stays `refreshing` inside the block; the job
    then runs inline."""
    held = []
    mp = pytest.MonkeyPatch()
    mp.setattr(jobs, "submit", lambda fn, *args: held.append((fn, args)))
    try:
        r = lead.post(f"/api/datasets/{ds_id}/refresh")
        assert r.status_code == 202, r.text
        yield r.json()
    finally:
        mp.undo()
        for fn, args in held:
            fn(*args)


def _field(ds_id: int, name: str) -> dict:
    return db.fetch_one("SELECT * FROM meta.fields WHERE dataset_id = %s AND name = %s", (ds_id, name))


def _latest_pull(ds_id: int) -> dict:
    return db.fetch_one("SELECT * FROM meta.dataset_pulls WHERE dataset_id = %s ORDER BY id DESC LIMIT 1", (ds_id,))


def test_refresh_is_202_with_the_running_pull_and_locks_until_the_job_ends(lead, ds):
    # The pull that created the dataset finished before its baseline was finalized.
    assert not any(
        "consider a new version" in w for w in lead.get(f"/api/datasets/{ds}/export-preview").json()["warnings"]
    )
    with _held(lead, ds) as pull:
        assert pull["status"] == "running" and pull["created_by_name"] == "Admin" and pull["finished_at"] is None
        row = db.fetch_one("SELECT status, status_message, owner FROM meta.datasets WHERE id = %s", (ds,))
        assert row["status"] == "refreshing" and row["status_message"] == "Refreshing from test"
        assert row["owner"] == jobs.owner_id()
        assert lead.get(f"/api/datasets/{ds}/pulls").json()[0]["id"] == pull["id"]
    after = lead.get(f"/api/datasets/{ds}").json()
    assert after["status"] == "ready" and after["last_pull"]["id"] == pull["id"]
    assert after["last_pull"]["status"] == "done"


def test_a_member_cannot_refresh(clients, ds):
    assert clients["member"].post(f"/api/datasets/{ds}/refresh").status_code == 403


def test_a_dataset_that_is_not_connected_is_404(lead):
    up = db.fetch_one(
        "INSERT INTO meta.datasets (name, file_format, status) VALUES ('uploaded', 'csv', 'ready') RETURNING id"
    )["id"]
    try:
        r = lead.post(f"/api/datasets/{up}/refresh")
        assert r.status_code == 404 and r.json()["detail"] == "This dataset isn't connected to a source"
        assert lead.post("/api/datasets/999999/refresh").status_code == 404
    finally:
        db.execute("DELETE FROM meta.datasets WHERE id = %s", (up,))


def test_writes_exports_and_explore_are_409_while_refreshing(lead, ds):
    engine = _field(ds, "engine")
    recipe = lead.get(f"/api/datasets/{ds}").json()["baseline"]
    with _held(lead, ds):
        refused = {
            "decision": lead.put(
                f"/api/fields/{engine['id']}/decision",
                json={"action": "drop", "semantic_type": engine["inferred_type"]},
            ),
            "bulk": lead.post(
                f"/api/datasets/{ds}/decisions/bulk", json={"field_ids": [engine["id"]], "action": "drop"}
            ),
            "rename": lead.patch(f"/api/datasets/{ds}", json={"name": "Renamed"}),
            "export": lead.post(f"/api/datasets/{ds}/exports", json={}),
            "explore": lead.post(f"/api/datasets/{ds}/analyses", json={"kind": "map"}),
            "delete": lead.delete(f"/api/datasets/{ds}"),
            "derived": lead.post(
                f"/api/datasets/{ds}/derived", json={"name": "x2", "kind": "formula", "spec": {"expr": "1"}}
            ),
            "export rows": lead.post(f"/api/datasets/{ds}/export-rows", json={}),
            "new version": lead.post(f"/api/datasets/{ds}/recipes", json={}),
            "assessor": lead.post(f"/api/datasets/{ds}/assessors", json={"user_id": lead.user["id"]}),
            "finalize": lead.post(f"/api/datasets/{ds}/finalize", json={}),
            "baseline name": lead.patch(f"/api/datasets/{ds}/baselines/tree", json={"name": "Fleet"}),
            "experiment": lead.post(
                f"/api/datasets/{ds}/experiments", json={"name": "Try", "source": f"recipe:{recipe['id']}"}
            ),
        }
        again = lead.post(f"/api/datasets/{ds}/refresh")
        assert {k: (r.status_code, r.json()["detail"]) for k, r in refused.items()} == {
            k: (409, LOCKED) for k in refused
        }
        assert again.status_code == 409 and again.json()["detail"] == "A pull of this dataset is already running"
    assert lead.get(f"/api/datasets/{ds}").json()["name"] == "Refreshed"


def test_refresh_is_409_while_a_version_is_assessed(lead, ds):
    set_phase(ds, "assessing")
    try:
        r = lead.post(f"/api/datasets/{ds}/refresh")
        assert r.status_code == 409 and "is being assessed" in r.json()["detail"]
        assert db.fetch_one("SELECT status FROM meta.datasets WHERE id = %s", (ds,))["status"] == "ready"
    finally:
        set_phase(ds, "finalized")


def test_refresh_is_409_while_an_export_runs(lead, ds):
    exp = db.fetch_one("INSERT INTO meta.exports (dataset_id, status) VALUES (%s, 'running') RETURNING id", (ds,))["id"]
    try:
        r = lead.post(f"/api/datasets/{ds}/refresh")
        assert r.status_code == 409 and r.json()["detail"] == (
            "An export or model run of this dataset is still running; refresh once it finishes."
        )
    finally:
        db.execute("DELETE FROM meta.exports WHERE id = %s", (exp,))


def test_an_export_after_a_refresh_records_the_pull(lead, ds, mongo):
    mongo.curated.insert_many([{k: v for k, v in d.items() if k != "_id"} for d in mongo.curated.find().limit(40)])
    try:
        with _held(lead, ds) as pull:
            pass
        assert _latest_pull(ds)["status"] == "done"
        preview = lead.get(f"/api/datasets/{ds}/export-preview").json()
        assert f"Baseline finalized before pull #{pull['id']}; consider a new version" in preview["warnings"]
        assert preview["blocking"] == []
        r = lead.post(f"/api/datasets/{ds}/exports", json={})
        assert r.status_code == 202, r.text
        assert r.json()["pull_id"] == pull["id"] and r.json()["config"]["pull_id"] == pull["id"]
        done = _wait_export(lead, r.json()["id"])
        assert done["status"] == "done", done["error"]
        assert done["pull_id"] == pull["id"]
        assert any(e["pull_id"] == pull["id"] for e in lead.get(f"/api/datasets/{ds}/exports").json())
        manifest = json.loads(lead.get(f"/api/exports/{done['id']}/files/manifest").content)
        assert manifest["config"]["pull_id"] == pull["id"]
        with _held(lead, ds):
            run = lead.post(f"/api/exports/{done['id']}/runs", json={"model": "lightgbm", "cv": 0, "cv_folds": 0})
            assert run.status_code == 409 and run.json()["detail"] == LOCKED
    finally:
        seed(mongo)


def test_preview_blocks_on_kept_fields_and_targets_the_refresh_removed(lead, mongo):
    ds = _connected(lead, mongo, "Fields gone", drop=("rpm.max",))
    snap = db.fetch_one("SELECT snapshot FROM meta.recipes WHERE dataset_id = %s AND phase = 'finalized'", (ds,))
    actions = {n: f["decision"]["action"] for n, f in snap["snapshot"]["fields"].items()}
    assert (actions["rpm.max"], actions["engine"], actions["vehicle"]) == ("drop", "keep", "target")
    gone = "Field in the baseline is no longer in the data: "
    try:
        # The target goes last: the preview's checks on target values must not read a column that is gone.
        for unset, blocked in (("rpm.max", []), ("engine", ["engine"]), ("vehicle", ["engine", "vehicle"])):
            mongo.curated.update_many({}, {"$unset": {unset: ""}})
            with _held(lead, ds):
                pass
            assert _latest_pull(ds)["status"] == "done" and _field(ds, unset) is None
            r = lead.get(f"/api/datasets/{ds}/export-preview")
            assert r.status_code == 200, r.text
            assert sorted(b for b in r.json()["blocking"] if b.startswith(gone)) == [gone + n for n in blocked], unset
        r = lead.post(f"/api/datasets/{ds}/exports", json={})
        assert r.status_code == 422 and {gone + "engine", gone + "vehicle"} <= set(r.json()["detail"]["errors"])
    finally:
        seed(mongo)
        lead.delete(f"/api/datasets/{ds}")


def test_export_rows_reads_only_the_targets_a_refresh_left(lead, mongo):
    ds = _connected(lead, mongo, "Target gone")
    mongo.curated.update_many({}, {"$unset": {"vehicle": ""}})
    try:
        with _held(lead, ds):
            pass
        assert _latest_pull(ds)["status"] == "done" and _field(ds, "vehicle") is None
        r = lead.post(f"/api/datasets/{ds}/export-rows", json={})
        assert r.status_code == 200, r.text
        assert r.json()["rows"] == 300 and r.json()["missing_target"] == 0 and r.json()["split"] is None
    finally:
        seed(mongo)
        lead.delete(f"/api/datasets/{ds}")


def test_mlflow_logs_the_pull_an_export_was_built_from(file_store, tmp_path):  # noqa: F811
    row = {**_run_row(), "export_config": {"pull_id": 12}}
    model_dir = _model_dir(tmp_path)
    _, run_id = mlflow_log.log_run(row, "Vehicles", {"model": "lightgbm"}, model_dir, user="ana")
    assert file_store.get_run(run_id).data.params["rainforest.pull_id"] == "12"
    _, plain = mlflow_log.log_run(_run_row(2), "Vehicles", {"model": "lightgbm"}, model_dir)
    assert "rainforest.pull_id" not in file_store.get_run(plain).data.params
