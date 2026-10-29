"""Deployments over HTTP: what may be deployed together, thresholds, keys, and the predict endpoint through a session
or a key. Needs RAINFOREST_TEST_DATABASE_URL."""

import json
import logging
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from app import db, scoring
from app.config import get_settings
from app.modeling.train import choose_threshold

from .team_helpers import app_, experiment, export, people, start, submit_and_finalize, upload, wait  # noqa: F401


def _csv(n=900, seed=5) -> str:
    """y: classes p, q, r and six rows of s (too few to train); v a number."""
    rng = np.random.default_rng(seed)
    x, z = rng.normal(size=n), rng.normal(size=n)
    s = x + 0.5 * rng.normal(size=n)
    y = np.where(s < -0.4, "p", np.where(s < 0.5, "q", "r"))
    y[rng.choice(n, 6, replace=False)] = "s"
    v = x + z + rng.normal(size=n)
    return "x,z,y,v\n" + "".join(f"{a:.3f},{b:.3f},{c},{d:.3f}\n" for a, b, c, d in zip(x, z, y, v))


BASE = {"model": "lightgbm", "cv": 0, "cv_folds": 0, "params": {"n_estimators": 15}}


def _run(client, e, **body) -> dict:
    r = client.post(f"/api/exports/{e['id']}/runs", json={**BASE, **body})
    assert r.status_code == 202, r.text
    run = wait(lambda: (x := client.get(f"/api/runs/{r.json()['id']}").json())["status"] in ("done", "failed") and x)
    assert run["status"] == "done", run.get("error")
    return run


@pytest.fixture(scope="module")
def setup(people):
    lead = people["lead"]
    ds = upload(lead, _csv(), targets="y,v")
    start(lead, ds, "tree", [lead], targets=("y", "v"))
    rid = submit_and_finalize(lead, ds)["recipe"]["id"]
    x = experiment(lead, ds, rid, "deployments")
    e = export(lead, ds, x)
    r = lead.post(f"/api/exports/{e['id']}/batches", json={**BASE, "kind": "one_vs_rest", "target": "y"})
    assert r.status_code == 202, r.text
    batch_id = r.json()["batch"]["id"]
    batch = wait(lambda: (b := lead.get(f"/api/batches/{batch_id}").json())["status"] != "running" and b, 240)
    assert batch["status"] == "done", batch
    ovr = {row["positive_class"]: row["run_id"] for row in batch["rows"]}
    return {
        "lead": lead,
        "ds": ds,
        "x": x,
        "e": e,
        "batch": batch,
        "ovr": ovr,
        "multi": _run(lead, e, order=["y"])["id"],
        "number": _run(lead, e, order=["v"])["id"],
        "chain": _run(lead, e, order=["y", "v"])["id"],
        "p_again": _run(lead, e, target="y", positive_class="p")["id"],
    }


def _deploy(client, x, expect=201, **body) -> dict:
    r = client.post(f"/api/experiments/{x['id']}/deployments", json={"name": f"d {uuid.uuid4().hex[:6]}", **body})
    assert r.status_code == expect, r.text
    return r.json()


def _records(setup, n=3) -> list[dict]:
    path = db.fetch_one("SELECT parquet_path FROM meta.datasets WHERE id = %s", (setup["ds"]["id"],))["parquet_path"]
    return pl.read_parquet(path, columns=["x", "z"], n_rows=n).to_dicts()


def _predict(client, dep_id, records, expect=200, **kw):
    r = client.post(f"/api/deployments/{dep_id}/predict", json={"records": records}, **kw)
    assert r.status_code == expect, r.text
    return r.json()


def _report(run_id: int) -> dict:
    return db.fetch_one("SELECT report FROM meta.model_runs WHERE id = %s", (run_id,))["report"]


@pytest.fixture(scope="module")
def deployed(setup):
    """A deployment of the one-vs-rest batch, and a key of it."""
    dep = _deploy(setup["lead"], setup["x"], batch_id=setup["batch"]["id"])
    key = setup["lead"].post(f"/api/deployments/{dep['id']}/keys", json={"name": "script"})
    assert key.status_code == 201, key.text
    return {"dep": dep, "key": key.json()}


def _keyed(app_, key: str) -> TestClient:
    c = TestClient(app_)
    c.headers["Authorization"] = f"Bearer {key}"
    return c


@contextmanager
def _config(export_id: int, change):
    before = db.fetch_one("SELECT config FROM meta.exports WHERE id = %s", (export_id,))["config"]
    db.execute("UPDATE meta.exports SET config = %s WHERE id = %s", (db.jsonb(change(dict(before))), export_id))
    try:
        yield
    finally:
        db.execute("UPDATE meta.exports SET config = %s WHERE id = %s", (db.jsonb(before), export_id))


# ------------------------------------------------------------------------------------------- creating
def test_a_deployment_of_a_one_vs_rest_batch_keeps_each_classs_f1_threshold(setup, deployed):
    dep = deployed["dep"]
    assert dep["status"] == "active" and dep["export_id"] == setup["e"]["id"]
    assert dep["batch_id"] == setup["batch"]["id"] and dep["dataset_id"] == setup["ds"]["id"]
    assert [m["positive_class"] for m in dep["models"]] == ["p", "q", "r"]
    for m in dep["models"]:
        assert m["run_id"] == setup["ovr"][m["positive_class"]] and m["kind"] == "binary"
        assert m["label"] == m["positive_class"] and m["threshold_rule"] == {"kind": "f1"}
        assert m["threshold"] == _report(m["run_id"])["f1_threshold"]["threshold"]
    assert dep["can_manage"] is True
    assert [f["name"] for f in dep["fields"]] == ["x", "z"]
    assert len(dep["example"]) == 2 and set(dep["example"][0]) == {"x", "z"}


def test_thresholds_follow_the_rule_for_all_and_a_rule_per_run(setup):
    p, q = setup["ovr"]["p"], setup["ovr"]["q"]
    dep = _deploy(
        setup["lead"],
        setup["x"],
        run_ids=[p, q],
        threshold_rule={"kind": "precision", "min": 0.6},
        thresholds={str(q): {"kind": "fixed", "value": 0.3}},
    )
    by_run = {m["run_id"]: m for m in dep["models"]}
    want, _ = choose_threshold(_report(p)["threshold_curve"]["valid"], {"kind": "precision", "min": 0.6})
    assert by_run[p]["threshold"] == want and by_run[p]["threshold_rule"] == {"kind": "precision", "min": 0.6}
    assert by_run[q]["threshold"] == 0.3 and by_run[q]["threshold_rule"] == {"kind": "fixed", "value": 0.3}


def test_a_precision_rule_that_cant_be_met_falls_back_to_f1_with_a_note(setup):
    p = setup["ovr"]["p"]
    dep = _deploy(setup["lead"], setup["x"], run_ids=[p], threshold_rule={"kind": "precision", "min": 1.0})
    m = dep["models"][0]
    curve = _report(p)["threshold_curve"]["valid"]
    if any(pt["precision"] >= 1.0 for pt in curve):
        pytest.skip("this run reaches precision 1 on validation")
    assert m["threshold"] == _report(p)["f1_threshold"]["threshold"] and "F1 choice" in m["threshold_note"]


def test_single_runs_of_one_target_deploy_alone(setup):
    multi = _deploy(setup["lead"], setup["x"], run_ids=[setup["multi"]])
    assert multi["models"][0]["kind"] == "multiclass" and multi["models"][0]["threshold"] is None
    assert multi["models"][0]["label"] == "y"
    number = _deploy(setup["lead"], setup["x"], run_ids=[setup["number"]])
    assert number["models"][0]["kind"] == "regression" and number["models"][0]["label"] == "v"


@pytest.mark.parametrize(
    "runs, words",
    [
        (lambda s: [s["chain"]], ["chains"]),
        (lambda s: [s["ovr"]["p"], s["multi"]], ["one-vs-rest", "multiclass"]),
        (lambda s: [s["multi"], s["number"]], ["one run"]),
        (lambda s: [s["ovr"]["p"], s["p_again"]], ["'p'", "two runs"]),
    ],
    ids=["chain", "mixed set", "two single runs", "duplicate class"],
)
def test_what_may_not_share_a_deployment_is_refused_naming_the_conflict(setup, runs, words):
    detail = _deploy(setup["lead"], setup["x"], 422, run_ids=runs(setup))["detail"]
    for w in words:
        assert w in detail, detail


def test_runs_of_two_exports_are_refused(setup):
    lead = setup["lead"]
    other = export(lead, setup["ds"], setup["x"])
    run = _run(lead, other, target="y", positive_class="q")
    detail = _deploy(lead, setup["x"], 422, run_ids=[setup["ovr"]["p"], run["id"]])["detail"]
    assert "export" in detail and str(other["id"]) in detail, detail


def test_a_threshold_for_a_run_that_has_none_is_refused(setup):
    m = setup["multi"]
    detail = _deploy(setup["lead"], setup["x"], 422, run_ids=[m], thresholds={str(m): {"kind": "fixed", "value": 0.5}})
    assert "threshold" in detail["detail"] and str(m) in detail["detail"], detail
    bad = _deploy(
        setup["lead"], setup["x"], 422, run_ids=[setup["ovr"]["p"]], threshold_rule={"kind": "fixed", "value": 2}
    )
    assert "0 to 1" in bad["detail"]
    stray = _deploy(
        setup["lead"],
        setup["x"],
        422,
        run_ids=[setup["ovr"]["p"]],
        thresholds={str(setup["ovr"]["q"]): {"kind": "f1"}},
    )
    assert str(setup["ovr"]["q"]) in stray["detail"]


def test_a_search_batch_is_deployed_by_choosing_its_rows(setup):
    lead = setup["lead"]
    r = lead.post(
        f"/api/exports/{setup['e']['id']}/batches",
        json={**BASE, "kind": "search", "target": "y", "search_classes": ["p"], "space": {"n_estimators": [5, 8]}},
    )
    assert r.status_code == 202, r.text
    bid = r.json()["batch"]["id"]
    batch = wait(lambda: (b := lead.get(f"/api/batches/{bid}").json())["status"] != "running" and b, 240)
    detail = _deploy(lead, setup["x"], 422, batch_id=bid)["detail"]
    assert "choosing its rows" in detail, detail
    row = batch["rows"][0]["run_id"]
    dep = _deploy(lead, setup["x"], batch_id=bid, run_ids=[row])
    assert dep["batch_id"] == bid and [m["run_id"] for m in dep["models"]] == [row]
    assert _deploy(lead, setup["x"], 422, batch_id=bid, run_ids=[setup["multi"]])["detail"]


def test_an_export_made_before_inputs_were_frozen_is_refused(setup):
    def strip(cfg):
        cfg.pop("derived", None)
        cfg.pop("input_schema", None)
        return cfg

    with _config(setup["e"]["id"], strip):
        detail = _deploy(setup["lead"], setup["x"], 422, run_ids=[setup["multi"]])["detail"]
    assert detail == scoring.REEXPORT


def test_a_model_that_doesnt_reproduce_its_training_scores_is_refused(setup):
    """The features the runs trained on no longer match what the export's preprocessing makes of the raw rows."""
    out = Path(db.fetch_one("SELECT output_dir FROM meta.exports WHERE id = %s", (setup["e"]["id"],))["output_dir"])
    data = out / "model_ready.parquet"
    feats = json.loads((out / "manifest.json").read_text())["feature_names"]
    saved = data.read_bytes()
    df = pl.read_parquet(data)
    df.with_columns(pl.col(f) + 1.0 for f in feats if df[f].dtype.is_float()).write_parquet(data)
    try:
        detail = _deploy(setup["lead"], setup["x"], 422, run_ids=[setup["ovr"]["p"]])["detail"]
    finally:
        data.write_bytes(saved)
    assert "'p'" in detail and "row" in detail, detail


def test_an_unfinished_run_or_a_run_of_another_experiment_is_refused(setup):
    lead = setup["lead"]
    other = experiment(lead, setup["ds"], db.fetch_one(
        "SELECT recipe_id FROM meta.experiments WHERE id = %s", (setup["x"]["id"],))["recipe_id"], "other")  # fmt: skip
    assert _deploy(lead, other, 422, run_ids=[setup["multi"]])["detail"]
    db.execute("UPDATE meta.model_runs SET status = 'failed' WHERE id = %s", (setup["number"],))
    try:
        detail = _deploy(lead, setup["x"], 422, run_ids=[setup["number"]])["detail"]
    finally:
        db.execute("UPDATE meta.model_runs SET status = 'done' WHERE id = %s", (setup["number"],))
    assert "finished" in detail or "done" in detail, detail
    assert _deploy(lead, setup["x"], 422)["detail"]  # neither runs nor a batch


def test_creating_is_409_while_the_dataset_is_refreshing(setup):
    ds = setup["ds"]["id"]
    db.execute("UPDATE meta.datasets SET status = 'refreshing' WHERE id = %s", (ds,))
    try:
        _deploy(setup["lead"], setup["x"], 409, run_ids=[setup["multi"]])
    finally:
        db.execute("UPDATE meta.datasets SET status = 'ready' WHERE id = %s", (ds,))


# ------------------------------------------------------------------------------------------- predict
def _scored(dep: dict, records: list[dict]) -> scoring.ScoreResult:
    with db.connection() as conn:
        s = scoring.load(conn, {"export_id": dep["export_id"], "models": dep["models"]})
    return s.score(pl.DataFrame(records))


def test_predict_through_the_session_scores_each_record_in_order(setup, deployed):
    dep, recs = deployed["dep"], _records(setup)
    out = _predict(setup["lead"], dep["id"], recs)
    assert out["deployment"] == dep["id"] and out["models"] == ["p", "q", "r"] and out["warnings"] == {}
    want = _scored(dep, recs)
    assert len(out["results"]) == 3
    for k, res in enumerate(out["results"]):
        assert res["scores"] == {c: float(want.scores[c][k]) for c in "pqr"}
        assert res["top"] == {"label": want.top[k][0], "score": want.top[k][1]}
        assert res["matches"] == want.matches[k] and res["none_matched"] == want.none_matched[k]


def test_a_multiclass_deployment_has_top_but_no_matches(setup):
    dep = _deploy(setup["lead"], setup["x"], run_ids=[setup["multi"]])
    res = _predict(setup["lead"], dep["id"], _records(setup, 1))["results"][0]
    assert list(res["scores"]) == _report(setup["multi"])["target_classes"]["y"]
    assert res["top"]["label"] in res["scores"] and res["matches"] is None and res["none_matched"] is None


def test_values_are_read_by_the_fields_type_not_the_first_record(setup, deployed):
    dep = deployed["dep"]
    recs = _records(setup, 2)
    as_text = [{k: str(v) for k, v in r.items()} for r in recs]
    mixed = [as_text[0], recs[1]]  # text first, numbers second: neither decides the column's type
    base = _predict(setup["lead"], dep["id"], recs)["results"]
    assert _predict(setup["lead"], dep["id"], as_text)["results"] == base
    assert _predict(setup["lead"], dep["id"], mixed)["results"] == base
    ints = [{"x": 1, "z": 2}, {"x": 1.5, "z": 2.25}]
    floats = [{"x": 1.0, "z": 2.0}, {"x": 1.5, "z": 2.25}]
    assert _predict(setup["lead"], dep["id"], ints)["results"] == _predict(setup["lead"], dep["id"], floats)["results"]
    # Inferred from the leading records, 2.5 after a hundred whole numbers would be read as 2, or refused after nulls.
    alone = _predict(setup["lead"], dep["id"], [{"x": 2.5, "z": 0.5}])["results"][0]
    for lead_x in (1, None):
        many = [{"x": lead_x, "z": 0.5}] * 100 + [{"x": 2.5, "z": 0.5}]
        assert _predict(setup["lead"], dep["id"], many)["results"][-1] == alone, lead_x


def test_a_value_that_doesnt_convert_is_missing_and_counted_in_warnings(setup, deployed):
    recs = [{"x": "abc", "z": 0.1}, {"x": None, "z": "NA"}, {"x": 0.2, "z": 0.3}]
    out = _predict(setup["lead"], deployed["dep"]["id"], recs)
    assert out["warnings"] == {"x": 1}


def test_a_missing_field_or_a_nested_value_is_422_naming_it(setup, deployed):
    dep_id = deployed["dep"]["id"]
    lead = setup["lead"]
    assert "'z'" in _predict(lead, dep_id, [{"x": 1}], 422)["detail"]
    detail = _predict(lead, dep_id, [{"x": 1, "z": 2}, {"x": 1}], 422)["detail"]
    assert "'z'" in detail and "2" in detail, detail
    assert "'x'" in _predict(lead, dep_id, [{"x": [1, 2], "z": 2}], 422)["detail"]
    assert "'z'" in _predict(lead, dep_id, [{"x": 1, "z": {"a": 1}}], 422)["detail"]
    for body in ({"records": "x"}, {"rows": []}, {"records": [1, 2]}, []):
        r = lead.post(f"/api/deployments/{dep_id}/predict", json=body)
        assert r.status_code == 422, (body, r.text)
    r = lead.post(f"/api/deployments/{dep_id}/predict", content=b"{nope", headers={"content-type": "application/json"})
    assert r.status_code == 422
    r = lead.post(
        f"/api/deployments/{dep_id}/predict", content=b'{"records": [{"x": NaN, "z": 1}]}',
        headers={"content-type": "application/json"},
    )  # fmt: skip
    assert r.status_code == 422
    assert _predict(lead, dep_id, [])["results"] == []


def test_too_many_records_or_too_large_a_body_is_413(setup, deployed, monkeypatch):
    dep_id = deployed["dep"]["id"]
    lead = setup["lead"]
    monkeypatch.setenv("PREDICT_MAX_RECORDS", "2")
    get_settings.cache_clear()
    try:
        detail = _predict(lead, dep_id, [{"x": 1, "z": 2}] * 3, 413)["detail"]
        assert "2" in detail and "PREDICT_MAX_RECORDS" in detail
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()
    big = json.dumps({"records": [{"x": "1" * (5 * 1024 * 1024), "z": 1}]}).encode()
    r = lead.post(f"/api/deployments/{dep_id}/predict", content=big, headers={"content-type": "application/json"})
    assert r.status_code == 413, r.text

    def chunks():
        yield b'{"records": [{"x": "'
        for _ in range(6):
            yield b"1" * (1024 * 1024)
        yield b'", "z": 1}]}'

    r = lead.post(f"/api/deployments/{dep_id}/predict", content=chunks(), headers={"content-type": "application/json"})
    assert r.status_code == 413, r.text


# ------------------------------------------------------------------------------------------- keys
def test_a_key_is_shown_once_and_stored_only_as_a_hash_and_prefix(setup, deployed):
    key = deployed["key"]
    assert key["key"].startswith("rf_") and len(key["key"]) == 35 and key["prefix"] == key["key"][:8]
    assert set(key["key"][3:]) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
    row = db.fetch_one("SELECT * FROM meta.deployment_keys WHERE id = %s", (key["id"],))
    import hashlib

    assert row["hash"] == hashlib.sha256(key["key"].encode()).hexdigest() and row["prefix"] == key["prefix"]
    assert key["key"] not in json.dumps(row, default=str)
    listed = setup["lead"].get(f"/api/deployments/{deployed['dep']['id']}/keys").json()
    assert [k["id"] for k in listed] == [key["id"]]
    assert "key" not in listed[0] and "hash" not in listed[0] and listed[0]["prefix"] == key["prefix"]


def test_predict_through_a_key_without_a_session_and_from_another_origin(setup, deployed, app_):
    c = _keyed(app_, deployed["key"]["key"])
    recs = _records(setup, 2)
    out = _predict(c, deployed["dep"]["id"], recs, headers={"Origin": "https://elsewhere.example"})
    assert out["results"] == _predict(setup["lead"], deployed["dep"]["id"], recs)["results"]
    # A session from another origin is still refused.
    r = setup["lead"].post(
        f"/api/deployments/{deployed['dep']['id']}/predict",
        json={"records": recs},
        headers={"Origin": "https://elsewhere.example"},
    )
    assert r.status_code == 403


def test_a_key_authorises_only_its_own_deployments_predict(setup, deployed, app_):
    c = _keyed(app_, deployed["key"]["key"])
    other = _deploy(setup["lead"], setup["x"], run_ids=[setup["multi"]])
    assert c.post(f"/api/deployments/{other['id']}/predict", json={"records": []}).status_code == 401
    dep_id = deployed["dep"]["id"]
    for method, path in (
        ("GET", f"/api/deployments/{dep_id}"),
        ("GET", f"/api/deployments/{dep_id}/keys"),
        ("POST", f"/api/deployments/{dep_id}/retire"),
        ("GET", "/api/auth/me"),
        ("GET", f"/api/datasets/{setup['ds']['id']}/deployments"),
    ):
        assert c.request(method, path).status_code == 401, path
    assert TestClient(app_).post(f"/api/deployments/{dep_id}/predict", json={"records": []}).status_code == 401
    for bad in ("Bearer rf_nottherightkeyatall000000000000", "Basic abc", "Bearer "):
        r = TestClient(app_).post(
            f"/api/deployments/{dep_id}/predict", json={"records": []}, headers={"Authorization": bad}
        )
        assert r.status_code == 401, bad


def test_a_revoked_key_stops_at_once(setup, app_):
    dep = _deploy(setup["lead"], setup["x"], run_ids=[setup["multi"]])
    key = setup["lead"].post(f"/api/deployments/{dep['id']}/keys", json={"name": "short-lived"}).json()
    c = _keyed(app_, key["key"])
    _predict(c, dep["id"], _records(setup, 1))
    r = setup["lead"].delete(f"/api/deployments/{dep['id']}/keys/{key['id']}")
    assert r.status_code == 200 and r.json()["revoked_at"], r.text
    assert c.post(f"/api/deployments/{dep['id']}/predict", json={"records": []}).status_code == 401
    listed = setup["lead"].get(f"/api/deployments/{dep['id']}/keys").json()
    assert listed[0]["revoked_at"]


def test_last_used_is_recorded_at_most_once_a_minute(setup, deployed, app_):
    kid = deployed["key"]["id"]
    c = _keyed(app_, deployed["key"]["key"])
    db.execute("UPDATE meta.deployment_keys SET last_used_at = now() - interval '30 seconds' WHERE id = %s", (kid,))
    before = db.fetch_one("SELECT last_used_at FROM meta.deployment_keys WHERE id = %s", (kid,))["last_used_at"]
    _predict(c, deployed["dep"]["id"], [])
    assert db.fetch_one("SELECT last_used_at FROM meta.deployment_keys WHERE id = %s", (kid,))["last_used_at"] == before
    db.execute("UPDATE meta.deployment_keys SET last_used_at = now() - interval '2 minutes' WHERE id = %s", (kid,))
    _predict(c, deployed["dep"]["id"], [])
    after = db.fetch_one("SELECT last_used_at FROM meta.deployment_keys WHERE id = %s", (kid,))["last_used_at"]
    assert after > before


def test_a_key_appears_in_no_later_response_and_no_log_line(setup, app_, caplog):
    caplog.set_level(logging.DEBUG)
    lead = setup["lead"]
    dep = _deploy(lead, setup["x"], run_ids=[setup["multi"]])
    created = lead.post(f"/api/deployments/{dep['id']}/keys", json={"name": "logged?"})
    key = created.json()["key"]
    digest = db.fetch_one("SELECT hash FROM meta.deployment_keys WHERE id = %s", (created.json()["id"],))["hash"]
    c = _keyed(app_, key)
    later = [
        c.post(f"/api/deployments/{dep['id']}/predict", json={"records": _records(setup, 1)}),
        c.post(f"/api/deployments/{dep['id']}/predict", json={"records": [{"x": 1}]}),
        c.get(f"/api/deployments/{dep['id']}"),
        lead.get(f"/api/deployments/{dep['id']}/keys"),
        lead.get(f"/api/deployments/{dep['id']}"),
        lead.get(f"/api/datasets/{setup['ds']['id']}/deployments"),
        lead.delete(f"/api/deployments/{dep['id']}/keys/{created.json()['id']}"),
        c.post(f"/api/deployments/{dep['id']}/predict", json={"records": []}),
    ]
    assert [r.status_code for r in later] == [200, 422, 401, 200, 200, 200, 200, 401]
    for r in later:
        assert key not in r.text and digest not in r.text
    assert caplog.records, "nothing was captured"
    for rec in caplog.records:
        line = rec.getMessage() + repr(rec.args) + (rec.exc_text or "")
        assert key not in line and key[8:] not in line and digest not in line, rec.getMessage()


# ------------------------------------------------------------------------------------------- lifecycle
def test_a_retired_deployment_answers_410_until_reactivated(setup, app_, monkeypatch):
    forgotten = []
    real = scoring.forget
    monkeypatch.setattr(scoring, "forget", lambda i: (forgotten.append(i), real(i)))
    lead = setup["lead"]
    dep = _deploy(lead, setup["x"], run_ids=[setup["multi"]])
    key = lead.post(f"/api/deployments/{dep['id']}/keys", json={"name": "k"}).json()["key"]
    c = _keyed(app_, key)
    recs = _records(setup, 1)
    _predict(c, dep["id"], recs)
    retired = lead.post(f"/api/deployments/{dep['id']}/retire")
    assert retired.status_code == 200 and retired.json()["status"] == "retired" and retired.json()["retired_at"]
    assert forgotten == [dep["id"]]
    _predict(c, dep["id"], recs, 410)
    _predict(lead, dep["id"], recs, 410)
    assert lead.post(f"/api/deployments/{dep['id']}/retire").status_code == 409
    active = lead.post(f"/api/deployments/{dep['id']}/activate")
    assert active.status_code == 200 and active.json()["status"] == "active" and active.json()["retired_at"] is None
    assert forgotten == [dep["id"], dep["id"]]
    _predict(c, dep["id"], recs)
    assert lead.post(f"/api/deployments/{dep['id']}/activate").status_code == 409


def test_lists_and_reads_follow_the_experiments_visibility(setup, deployed):
    lead = setup["lead"]
    listed = lead.get(f"/api/datasets/{setup['ds']['id']}/deployments").json()
    assert deployed["dep"]["id"] in [d["id"] for d in listed]
    row = next(d for d in listed if d["id"] == deployed["dep"]["id"])
    assert row["experiment_name"] == "deployments" and row["keys"] == 1 and row["created_by_name"]
    assert lead.get("/api/deployments/999999999").status_code == 404
    assert lead.post("/api/deployments/999999999/predict", json={"records": []}).status_code == 404


def test_a_viewer_reads_deployments_and_predicts_only_with_a_key(setup, deployed, app_, login_as):  # noqa: F811
    viewer = login_as(TestClient(app_), "viewer", name=f"Viewer {uuid.uuid4().hex[:6]}")
    dep_id = deployed["dep"]["id"]
    assert viewer.get(f"/api/datasets/{setup['ds']['id']}/deployments").status_code == 200
    got = viewer.get(f"/api/deployments/{dep_id}")
    assert got.status_code == 200 and got.json()["can_manage"] is False
    assert viewer.get(f"/api/deployments/{dep_id}/keys").status_code == 200
    r = viewer.post(f"/api/experiments/{setup['x']['id']}/deployments", json={"name": "n", "run_ids": [setup["multi"]]})
    assert r.status_code == 403
    assert viewer.post(f"/api/deployments/{dep_id}/retire").status_code == 403
    assert viewer.post(f"/api/deployments/{dep_id}/keys", json={"name": "v"}).status_code == 403
    assert viewer.delete(f"/api/deployments/{dep_id}/keys/{deployed['key']['id']}").status_code == 403
    assert viewer.post(f"/api/deployments/{dep_id}/predict", json={"records": []}).status_code == 403
    viewer.headers["Authorization"] = f"Bearer {deployed['key']['key']}"
    assert viewer.post(f"/api/deployments/{dep_id}/predict", json={"records": []}).status_code == 200


def test_only_the_creator_and_leads_retire_and_manage_keys(setup, people):
    member, other = people["a"], people["b"]
    dep = _deploy(member, setup["x"], run_ids=[setup["multi"]])
    assert dep["can_manage"] is True and other.get(f"/api/deployments/{dep['id']}").json()["can_manage"] is False
    assert other.post(f"/api/deployments/{dep['id']}/retire").status_code == 403
    assert other.post(f"/api/deployments/{dep['id']}/keys", json={"name": "x"}).status_code == 403
    key = member.post(f"/api/deployments/{dep['id']}/keys", json={"name": "mine"})
    assert key.status_code == 201
    assert other.delete(f"/api/deployments/{dep['id']}/keys/{key.json()['id']}").status_code == 403
    assert other.post(f"/api/deployments/{dep['id']}/predict", json={"records": []}).status_code == 200
    assert setup["lead"].delete(f"/api/deployments/{dep['id']}/keys/{key.json()['id']}").status_code == 200
    assert setup["lead"].post(f"/api/deployments/{dep['id']}/retire").status_code == 200
    assert member.post(f"/api/deployments/{dep['id']}/activate").status_code == 200


def test_deleting_the_dataset_removes_its_deployments_first(setup, deployed, monkeypatch):
    forgotten = []
    real = scoring.forget
    monkeypatch.setattr(scoring, "forget", lambda i: (forgotten.append(i), real(i)))
    ids = [d["id"] for d in setup["lead"].get(f"/api/datasets/{setup['ds']['id']}/deployments").json()]
    assert deployed["dep"]["id"] in ids
    folder = get_settings().deployments_dir / str(deployed["dep"]["id"])
    folder.mkdir(parents=True, exist_ok=True)
    r = setup["lead"].delete(f"/api/datasets/{setup['ds']['id']}")
    assert r.status_code == 200, r.text
    assert r.json()["deployments"] == len(ids)
    assert db.fetch_all("SELECT id FROM meta.deployments WHERE id = ANY(%s)", (ids,)) == []
    assert set(forgotten) == set(ids) and not folder.exists()
