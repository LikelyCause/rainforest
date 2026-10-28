"""Refreshing a connected dataset over HTTP: who may, when it is refused, the lock it holds while it runs, and the pull
an export records. Needs RAINFOREST_TEST_MONGO_URL and RAINFOREST_TEST_DATABASE_URL.
"""

import json
import os
import time
import uuid
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


def test_a_second_refresh_is_409_while_one_runs(lead, ds):
    with _held(lead, ds):
        again = lead.post(f"/api/datasets/{ds}/refresh")
        assert again.status_code == 409 and again.json()["detail"] == "A pull of this dataset is already running"


# Every route behind a check from routers/_lock (grep for `refuse_` and `REFRESHING`), each with a request that gets
# past whatever the route checks before the lock.
LOCKED_ROUTES = {
    "decision": lambda c, i: c.put(
        f"/api/fields/{i['field']}/decision", json={"action": "drop", "semantic_type": i["type"]}
    ),
    "bulk": lambda c, i: c.post(
        f"/api/datasets/{i['ds']}/decisions/bulk", json={"field_ids": [i["field"]], "action": "drop"}
    ),
    "bulk undo": lambda c, i: c.post(f"/api/datasets/{i['ds']}/decisions/bulk/{uuid.uuid4()}/undo"),
    "assign": lambda c, i: c.post(f"/api/datasets/{i['ds']}/assessors", json={"user_id": i["user"]}),
    "unassign": lambda c, i: c.delete(f"/api/datasets/{i['ds']}/assessors/{i['user']}"),
    "submit": lambda c, i: c.post(f"/api/datasets/{i['ds']}/assessment/submit"),
    "unsubmit": lambda c, i: c.post(f"/api/datasets/{i['ds']}/assessment/unsubmit"),
    "comment": lambda c, i: c.post(f"/api/fields/{i['field']}/comments", json={"body": "Looks right"}),
    "finalize": lambda c, i: c.post(f"/api/datasets/{i['ds']}/finalize", json={}),
    "experiment": lambda c, i: c.post(
        f"/api/datasets/{i['ds']}/experiments", json={"name": "Try", "source": f"recipe:{i['recipe']}"}
    ),
    "revert adjustments": lambda c, i: c.post(f"/api/experiments/{i['experiment']}/adjustments/revert", json={}),
    "experiment details": lambda c, i: c.patch(f"/api/experiments/{i['experiment']}", json={"name": "Renamed"}),
    "run note": lambda c, i: c.patch(f"/api/runs/{i['run']}", json={"note": "Promising"}),
    "start baseline": lambda c, i: c.post(
        f"/api/datasets/{i['ds']}/baselines", json={"family": "tree", "assessors": [i["user"]], "targets": ["vehicle"]}
    ),
    "baseline name": lambda c, i: c.patch(f"/api/datasets/{i['ds']}/baselines/tree", json={"name": "Fleet"}),
    "new version": lambda c, i: c.post(f"/api/datasets/{i['ds']}/recipes", json={}),
    "apply recipe": lambda c, i: c.post(f"/api/datasets/{i['ds']}/recipes/apply", json={"recipe_id": i["recipe"]}),
    "derived preview": lambda c, i: c.post(
        f"/api/datasets/{i['ds']}/derived/preview", json={"name": "x2", "kind": "formula", "spec": {"expr": "1"}}
    ),
    "add derived": lambda c, i: c.post(
        f"/api/datasets/{i['ds']}/derived", json={"name": "x2", "kind": "formula", "spec": {"expr": "1"}}
    ),
    "remove derived": lambda c, i: c.delete(f"/api/datasets/{i['ds']}/derived/{i['field']}"),
    "explore": lambda c, i: c.post(f"/api/datasets/{i['ds']}/analyses", json={"kind": "map"}),
    "correlations": lambda c, i: c.get(f"/api/datasets/{i['ds']}/explore/correlations"),
    "export rows": lambda c, i: c.post(f"/api/datasets/{i['ds']}/export-rows", json={}),
    "export": lambda c, i: c.post(f"/api/datasets/{i['ds']}/exports", json={}),
    "run": lambda c, i: c.post(f"/api/exports/{i['export']}/runs", json={"model": "lightgbm", "cv": 0, "cv_folds": 0}),
    "rename": lambda c, i: c.patch(f"/api/datasets/{i['ds']}", json={"name": "Renamed"}),
    "targets": lambda c, i: c.patch(f"/api/datasets/{i['ds']}", json={"target_columns": ["vehicle"]}),
    "delete": lambda c, i: c.delete(f"/api/datasets/{i['ds']}"),
}


@pytest.fixture(scope="module")
def ids(clients, ds) -> dict:
    """What the locked routes' requests name: a field, the finalized baseline, and a finished export with its
    experiment and a finished run."""
    lead = clients["lead"]
    r = lead.post(f"/api/datasets/{ds}/exports", json={})
    assert r.status_code == 202, r.text
    export = _wait_export(lead, r.json()["id"])
    assert export["status"] == "done", export["error"]
    engine = _field(ds, "engine")
    run = db.fetch_one(
        """INSERT INTO meta.model_runs (export_id, dataset_id, model, family, status, created_by, experiment_id)
           VALUES (%s, %s, 'lightgbm', 'tree', 'done', %s, %s) RETURNING id""",
        (export["id"], ds, lead.user["id"], export["experiment_id"]),
    )["id"]
    return {
        "ds": ds,
        "field": engine["id"],
        "type": engine["inferred_type"],
        "recipe": lead.get(f"/api/datasets/{ds}").json()["baseline"]["id"],
        "export": export["id"],
        "experiment": export["experiment_id"],
        "run": run,
        "user": lead.user["id"],
    }


@pytest.mark.parametrize("route", LOCKED_ROUTES)
def test_every_locked_route_is_409_while_refreshing(lead, ids, route):
    with _held(lead, ids["ds"]):
        r = LOCKED_ROUTES[route](lead, ids)
        assert (r.status_code, r.json()["detail"]) == (409, LOCKED)
    assert lead.get(f"/api/datasets/{ids['ds']}").json()["name"] == "Refreshed"


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


def _new_version_after_refresh(lead, mongo, name: str, change) -> tuple[int, object]:
    """A dataset refreshed after `change` to its collection, and the response to starting a new version on it."""
    ds = _connected(lead, mongo, name)
    change(mongo.curated)
    with _held(lead, ds):
        pass
    assert _latest_pull(ds)["status"] == "done", _latest_pull(ds)["error"]
    return ds, lead.post(f"/api/datasets/{ds}/recipes", json={})


def test_a_new_version_starts_once_a_refresh_removed_a_baseline_target(lead, mongo):
    ds, r = _new_version_after_refresh(
        lead, mongo, "Target unset", lambda c: c.update_many({}, {"$unset": {"vehicle": ""}})
    )
    try:
        assert r.status_code == 200, r.text
        assert lead.get(f"/api/datasets/{ds}").json()["target_columns"] == []
        engine = db.fetch_one("SELECT * FROM meta.field_decisions WHERE field_id = %s", (_field(ds, "engine")["id"],))
        assert engine["action"] == "keep" and engine["source"] == "user" and engine["reviewed"]
        summary = {
            "missing": ["vehicle"],
            "targets_not_carried": [{"name": "vehicle", "problem": "no longer in the data"}],
        }
        assert r.json()["recipe"]["start_summary"] == summary
        assert lead.get(f"/api/datasets/{ds}/recipes").json()[0]["start_summary"] == summary
    finally:
        seed(mongo)
        lead.delete(f"/api/datasets/{ds}")


def test_a_new_version_starts_once_profiling_turned_a_baseline_target_down(lead, mongo):
    def many_labels(c):
        c.insert_many([{"vehicle": f"V{i:03d}", "engine": "V6", "rpm": {"mean": 1000.0 + i}} for i in range(200)])

    ds, r = _new_version_after_refresh(lead, mongo, "Target rejected", many_labels)
    try:
        assert r.status_code == 200, r.text
        assert lead.get(f"/api/datasets/{ds}").json()["target_columns"] == []
        vehicle = _field(ds, "vehicle")
        assert (
            db.fetch_one("SELECT action FROM meta.field_decisions WHERE field_id = %s", (vehicle["id"],))["action"]
            != "target"
        )
        summary = r.json()["recipe"]["start_summary"]
        (dropped,) = summary["targets_not_carried"]
        assert summary["missing"] == [] and dropped["name"] == "vehicle" and "distinct values" in dropped["problem"]
    finally:
        seed(mongo)
        lead.delete(f"/api/datasets/{ds}")


def _experiments(lead, ds: int) -> dict[str, dict]:
    """A personal experiment on the lead's assessment of v2 and a team one made on v2 once finalized."""
    set_phase(ds, "assessing")
    rid = lead.get(f"/api/datasets/{ds}/team").json()["recipe"]["id"]
    r = lead.post(f"/api/datasets/{ds}/assessors", json={"user_id": lead.user["id"]})
    assert r.status_code == 200, r.text
    r = lead.post(
        f"/api/datasets/{ds}/experiments", json={"name": "Mine", "source": f"assessment:{rid}:{lead.user['id']}"}
    )
    assert r.status_code in (200, 201), r.text
    personal = r.json()
    set_phase(ds, "finalized")
    r = lead.post(f"/api/datasets/{ds}/experiments", json={"name": "Try", "source": f"recipe:{rid}"})
    assert r.status_code in (200, 201), r.text
    return {"team": r.json(), "personal": personal}


def test_experiment_exports_block_on_kept_fields_and_targets_the_refresh_removed(lead, mongo):
    ds = _connected(lead, mongo, "Experiment fields gone")
    xs = _experiments(lead, ds)
    gone = "Field in the baseline is no longer in the data: "
    try:
        mongo.curated.update_many({}, {"$unset": {"engine": "", "vehicle": ""}})
        with _held(lead, ds):
            pass
        assert _latest_pull(ds)["status"] == "done"
        for kind, x in xs.items():
            r = lead.get(f"/api/datasets/{ds}/export-preview", params={"experiment_id": x["id"]})
            assert r.status_code == 200, r.text
            assert sorted(b for b in r.json()["blocking"] if b.startswith(gone)) == [
                gone + "engine",
                gone + "vehicle",
            ], kind
            r = lead.post(f"/api/datasets/{ds}/exports", json={"experiment_id": x["id"]})
            assert r.status_code == 422 and {gone + "engine", gone + "vehicle"} <= set(r.json()["detail"]["errors"]), (
                kind
            )
    finally:
        seed(mongo)
        lead.delete(f"/api/datasets/{ds}")


def test_fields_a_refresh_removed_are_not_adjustments(lead, mongo):
    ds = _connected(lead, mongo, "Phantom adjustments", drop=("rpm.max",))
    rid = lead.get(f"/api/datasets/{ds}").json()["baseline"]["id"]
    r = lead.post(f"/api/datasets/{ds}/experiments", json={"name": "Try", "source": f"recipe:{rid}"})
    assert r.status_code in (200, 201), r.text
    x = r.json()

    def adjustments() -> dict:
        return lead.get(f"/api/experiments/{x['id']}").json()["adjustments"]

    try:
        mongo.curated.update_many({}, {"$unset": {"rpm.max": ""}})
        with _held(lead, ds) as first:
            pass
        gone = [{"name": "rpm.max", "pull_id": first["id"]}]
        a = adjustments()
        assert (a["count"], a["fields"], a["targets_changed"], a["removed_by_refresh"]) == (0, [], False, gone)
        row = next(r for r in lead.get(f"/api/datasets/{ds}/experiments").json() if r["id"] == x["id"])
        assert row["adjusted_count"] == 0 and not row["targets_changed"]
        r = lead.post(f"/api/datasets/{ds}/exports", json={"experiment_id": x["id"]})
        assert r.status_code == 202, r.text
        config = r.json()["config"]["adjustments"]
        assert (config["count"], config["fields"], config["removed_by_refresh"]) == (0, [], gone)
        assert _wait_export(lead, r.json()["id"])["status"] == "done"

        mongo.curated.update_many({}, {"$unset": {"vehicle": ""}})
        with _held(lead, ds) as second:
            pass
        a = adjustments()
        assert (a["count"], a["targets"], a["targets_changed"]) == (0, None, False)
        assert a["removed_by_refresh"] == [*gone, {"name": "vehicle", "pull_id": second["id"]}]
    finally:
        seed(mongo)
        lead.delete(f"/api/datasets/{ds}")


def _decide_in(lead, ds: int, kind: str, x: dict, name: str, action: str) -> None:
    """`name` set to `action` in the experiment's own copy (team) or in the assessment it follows (personal)."""
    fid = _field(ds, name)["id"]
    if kind == "team":
        db.execute(
            "UPDATE meta.experiment_decisions SET action = %s WHERE experiment_id = %s AND field_id = %s",
            (action, x["id"], fid),
        )
    else:
        db.execute(
            "UPDATE meta.assessments SET action = %s, source = 'user' WHERE recipe_id = %s AND user_id = %s "
            "AND field_id = %s",
            (action, x["recipe_id"], lead.user["id"], fid),
        )


@pytest.mark.parametrize("kind", ["team", "personal"])
@pytest.mark.parametrize(
    "name,baseline,own,blocked",
    [("engine", "keep", "drop", False), ("rpm.max", "drop", "keep", True)],
    ids=["dropped-by-the-experiment", "kept-by-the-experiment"],
)
def test_the_gone_field_check_follows_what_the_experiment_used(lead, mongo, kind, name, baseline, own, blocked):
    ds = _connected(lead, mongo, f"Gone {kind} {name}", drop=(name,) if baseline == "drop" else ())
    x = _experiments(lead, ds)[kind]
    _decide_in(lead, ds, kind, x, name, own)
    gone = "Field in the baseline is no longer in the data: "

    def blocking() -> list[str]:
        r = lead.get(f"/api/datasets/{ds}/export-preview", params={"experiment_id": x["id"]})
        assert r.status_code == 200, r.text
        return [b for b in r.json()["blocking"] if b.startswith(gone)]

    try:
        assert blocking() == []
        mongo.curated.update_many({}, {"$unset": {name: ""}})
        with _held(lead, ds) as pull:
            pass
        assert _latest_pull(ds)["status"] == "done"
        assert blocking() == ([gone + name] if blocked else [])
        r = lead.post(f"/api/datasets/{ds}/exports", json={"experiment_id": x["id"]})
        assert r.status_code == (422 if blocked else 202), r.text
        if not blocked:
            assert _wait_export(lead, r.json()["id"])["status"] == "done"
        notes = db.fetch_one("SELECT flatten_notes FROM meta.dataset_pulls WHERE id = %s", (pull["id"],))
        recorded = notes["flatten_notes"]["removed_in_use"]
        key = str(x["id"]) if kind == "team" else f"{x['recipe_id']}:{lead.user['id']}"
        assert recorded["experiments" if kind == "team" else "assessments"].get(key, []) == ([name] if blocked else [])
    finally:
        seed(mongo)
        lead.delete(f"/api/datasets/{ds}")


@pytest.mark.parametrize("kind", ["team", "personal"])
def test_an_experiment_made_after_the_refresh_blocks_on_what_its_baseline_used(lead, mongo, kind):
    ds = _connected(lead, mongo, f"Late {kind}")
    set_phase(ds, "assessing")
    rid = lead.get(f"/api/datasets/{ds}/team").json()["recipe"]["id"]
    assert lead.post(f"/api/datasets/{ds}/assessors", json={"user_id": lead.user["id"]}).status_code == 200
    set_phase(ds, "finalized")
    gone = "Field in the baseline is no longer in the data: engine"
    try:
        mongo.curated.update_many({}, {"$unset": {"engine": ""}})
        with _held(lead, ds) as pull:
            pass
        assert _latest_pull(ds)["status"] == "done"
        source = f"recipe:{rid}" if kind == "team" else f"assessment:{rid}:{lead.user['id']}"
        r = lead.post(f"/api/datasets/{ds}/experiments", json={"name": "Late", "source": source})
        assert r.status_code in (200, 201), r.text
        x = r.json()
        r = lead.get(f"/api/datasets/{ds}/export-preview", params={"experiment_id": x["id"]})
        assert gone in r.json()["blocking"]
        r = lead.post(f"/api/datasets/{ds}/exports", json={"experiment_id": x["id"]})
        assert r.status_code == 422 and gone in r.json()["detail"]["errors"]
        recorded = _latest_pull(ds)["flatten_notes"]["removed_in_use"]
        # Every assessor of a finalized version is recorded, an experiment or not; the late team one is not.
        assert recorded["assessments"] == {f"{rid}:{lead.user['id']}": ["engine"]}
        assert str(x["id"]) not in recorded["experiments"] and _latest_pull(ds)["id"] == pull["id"]
    finally:
        seed(mongo)
        lead.delete(f"/api/datasets/{ds}")


def test_the_in_use_record_reads_assessments_as_a_personal_export_does(lead, mongo):
    from app import refresh
    from app.assessment import personal_snapshot
    from app.recipes import get

    ds = _connected(lead, mongo, "In use record", drop=("rpm.max",))
    engine, rpm_max = _field(ds, "engine"), _field(ds, "rpm.max")
    try:
        r = lead.post(f"/api/datasets/{ds}/recipes", json={"reopen": [engine["id"]]})
        assert r.status_code == 200, r.text
        rid = r.json()["recipe"]["id"]
        assert lead.post(f"/api/datasets/{ds}/assessors", json={"user_id": lead.user["id"]}).status_code == 200
        # The reopened field follows the assessment; a carried one keeps the version's decision whatever the row says.
        for f, action in ((engine, "drop"), (rpm_max, "keep")):
            db.execute(
                "UPDATE meta.assessments SET action = %s WHERE recipe_id = %s AND user_id = %s AND field_id = %s",
                (action, rid, lead.user["id"], f["id"]),
            )
        set_phase(ds, "finalized")
        gone = {
            r["id"]: r["name"]
            for r in db.fetch_all("SELECT id, name FROM meta.fields WHERE dataset_id = %s ORDER BY ordinal", (ds,))
        }
        with db.connection() as conn:
            got = refresh._in_use(conn, ds, gone)
            snap = personal_snapshot(conn, get(conn, rid), lead.user["id"])
            conn.rollback()
        expected = [
            n for n in gone.values() if n in snap["fields"] and snap["fields"][n]["decision"]["action"] != "drop"
        ]
        assert got["assessments"][f"{rid}:{lead.user['id']}"] == expected
        assert "vehicle" in expected and "engine" not in expected and "rpm.max" not in expected
    finally:
        lead.delete(f"/api/datasets/{ds}")


def test_the_in_use_record_has_entries_only_when_the_refresh_deletes_fields(lead, mongo):
    ds = _connected(lead, mongo, "In use entries", drop=("rpm.max",))
    rid = lead.get(f"/api/datasets/{ds}").json()["baseline"]["id"]
    x = lead.post(f"/api/datasets/{ds}/experiments", json={"name": "Try", "source": f"recipe:{rid}"}).json()
    try:
        with _held(lead, ds):
            pass
        assert _latest_pull(ds)["flatten_notes"]["removed_in_use"] == {"experiments": {}, "assessments": {}}
        mongo.curated.update_many({}, {"$unset": {"rpm.max": ""}})
        with _held(lead, ds):
            pass
        # Deleting, it records every experiment, the ones that used none of the fields too.
        assert _latest_pull(ds)["flatten_notes"]["removed_in_use"]["experiments"] == {str(x["id"]): []}
    finally:
        seed(mongo)
        lead.delete(f"/api/datasets/{ds}")
