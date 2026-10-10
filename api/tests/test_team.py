"""Blind assessment, team review and recipes. The adoption rule is tested on its own; the rest runs against a real
Postgres (RAINFOREST_TEST_DATABASE_URL): phases and their transitions, role checks, that no read path shows one
assessor another's choices while assessing, finalizing, exports, the migration of existing datasets and the
console role's grants."""
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.assessment import adopt, settings_key

DB_URL = os.environ.get("RAINFOREST_TEST_DATABASE_URL")
KEYS = ("action", "semantic_type", "imputation", "impute_value", "add_missing_indicator", "transform", "scaling",
        "encoding", "params", "notes", "reviewed")


# ------------------------------------------------------------------------------------------ adoption rule
def _s(imputation="none", value=None, indicator=False):
    return {"imputation": imputation, "impute_value": value, "add_missing_indicator": indicator, "transform": "none",
            "scaling": "none", "encoding": "none", "params": {}}


def _a(action="keep", t="numeric_continuous", **settings):
    return {"action": action, "semantic_type": t, "settings": settings or {"tree": _s()}}


def _sugg(t, family):
    return _s("median")


def test_disagreement_on_action_or_type_is_not_adopted():
    assert adopt([_a("keep"), _a("drop")], _sugg) is None
    assert adopt([_a(t="numeric_continuous"), _a(t="numeric_discrete")], _sugg) is None
    assert adopt([], _sugg) is None


def test_agreement_takes_the_most_common_settings_per_family():
    out = adopt([_a(tree=_s("mean"), linear=_s("mean")), _a(tree=_s("mean"), linear=_s("constant", "0")),
                 _a(tree=_s("constant", 0), linear=_s("constant", "0"))], _sugg)
    assert (out["action"], out["semantic_type"]) == ("keep", "numeric_continuous")
    assert out["settings"]["tree"]["imputation"] == "mean"
    # 0 and "0" are the same fill value
    assert out["settings"]["linear"]["imputation"] == "constant"


def test_a_tie_takes_the_suggestion():
    out = adopt([_a(tree=_s("mean")), _a(tree=_s("constant", "1"))], _sugg)
    assert out["settings"]["tree"] == _s("median")
    one = adopt([_a(tree=_s("mean"))], _sugg)
    assert one["settings"]["tree"]["imputation"] == "mean", "a single assessor's choice is the most common"


def test_settings_key_normalises_values():
    assert settings_key(_s("constant", 1)) == settings_key(_s("constant", "1"))
    assert settings_key({**_s(), "params": {"a": 1, "b": 2}}) == settings_key({**_s(), "params": {"b": 2, "a": 1}})
    assert settings_key(_s(indicator=True)) != settings_key(_s())


# --------------------------------------------------------------------------------------------- API tests
@pytest.fixture(scope="module")
def app_(tmp_path_factory, login_as):
    if not DB_URL:
        pytest.skip("set RAINFOREST_TEST_DATABASE_URL to run the API tests")
    from app import db
    from app.config import get_settings

    mp = pytest.MonkeyPatch()
    mp.setenv("DATABASE_URL", DB_URL)
    mp.setenv("DATA_DIR", str(tmp_path_factory.mktemp("data")))
    get_settings.cache_clear()
    db.close_pool()
    from app.main import app

    with TestClient(app):
        yield app
    db.close_pool()
    mp.undo()
    get_settings.cache_clear()


@pytest.fixture(scope="module")
def people(app_, login_as):
    tag = uuid.uuid4().hex[:6]
    return {
        "lead": login_as(TestClient(app_)),
        "a": login_as(TestClient(app_), "member", name=f"Assessor A {tag}"),
        "b": login_as(TestClient(app_), "member", name=f"Assessor B {tag}"),
        "c": login_as(TestClient(app_), "member", name=f"Bystander C {tag}"),
    }


def _csv(n=400, seed=3) -> str:
    rng = np.random.default_rng(seed)
    return "x,z,w,c,y\n" + "".join(
        f"{'' if i % 9 == 0 else f'{a:.3f}'},{b:.3f},{c:.3f},{'pqr'[i % 3]},{'ab'[i % 2]}\n"
        for i, (a, b, c) in enumerate(zip(rng.normal(size=n), rng.normal(size=n), rng.normal(size=n))))


def _wait(fn, timeout=120.0):
    deadline = time.monotonic() + timeout
    while not (out := fn()):
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.3)
    return out


def _upload(client, csv=None) -> dict:
    ds = client.post("/api/datasets/upload", files={"file": ("t.csv", csv or _csv(), "text/csv")},
                     data={"targets": "y"}).json()
    ds = _wait(lambda: (d := client.get(f"/api/datasets/{ds['id']}").json())["status"] in ("ready", "failed") and d)
    assert ds["status"] == "ready", ds
    return ds


def _fields(client, ds) -> dict[str, dict]:
    return {f["name"]: f for f in client.get(f"/api/datasets/{ds['id']}/fields").json()}


def _save(client, fid, family="tree", **patch):
    cur = client.get(f"/api/fields/{fid}?family={family}").json()["decision"]
    r = client.put(f"/api/fields/{fid}/decision", json={**{k: cur[k] for k in KEYS}, "family": family, **patch})
    return r


def _assign(lead, ds, *users):
    for u in users:
        r = lead.post(f"/api/datasets/{ds['id']}/assessors", json={"user_id": u.user["id"]})
        assert r.status_code == 200, r.text
    return r.json()


def _review_all(client, ds):
    r = client.post(f"/api/datasets/{ds['id']}/decisions/bulk", json={"filters": [], "reviewed": True})
    assert r.status_code == 200, r.text


def _phase(client, ds) -> str:
    return client.get(f"/api/datasets/{ds['id']}").json()["phase"]


def test_new_dataset_starts_assessing_with_assessors_seeded_from_suggestions(people):
    lead, a = people["lead"], people["a"]
    ds = _upload(lead)
    try:
        assert (ds["phase"], ds["recipe_version"]) == ("assessing", 1)
        team = _assign(lead, ds, a)
        assert [x["user_id"] for x in team["assessors"]] == [a.user["id"]]
        assert team["assessors"][0]["reviewed"] == 0 and team["assessors"][0]["total"] == 5
        mine = _fields(a, ds)
        assert mine["y"]["action"] == "target" and not any(f["reviewed"] for f in mine.values())
        listed = next(d for d in a.get("/api/datasets").json() if d["id"] == ds["id"])
        assert listed["phase"] == "assessing" and listed["my_assessment"] == {"reviewed": 0, "submitted": False}
        assert next(d for d in lead.get("/api/datasets").json() if d["id"] == ds["id"])["my_assessment"] is None
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")


def test_members_get_403_on_lead_actions(people):
    lead, a, c = people["lead"], people["a"], people["c"]
    ds = _upload(lead)
    try:
        assert c.post(f"/api/datasets/{ds['id']}/assessors", json={"user_id": a.user["id"]}).status_code == 403
        _assign(lead, ds, a)
        assert c.delete(f"/api/datasets/{ds['id']}/assessors/{a.user['id']}").status_code == 403
        assert a.post(f"/api/datasets/{ds['id']}/finalize", json={}).status_code == 403
        fid = _fields(c, ds)["x"]["id"]
        r = _save(c, fid, action="drop")
        assert r.status_code == 403 and "assessors" in r.json()["detail"], "non-assessors can't decide while assessing"
        _review_all(a, ds)
        assert a.post(f"/api/datasets/{ds['id']}/assessment/submit").status_code == 200
        assert _phase(lead, ds) == "review"
        assert _save(a, fid, action="drop").status_code == 403, "members don't set the team decision"
        assert a.post(f"/api/datasets/{ds['id']}/decisions/bulk",
                      json={"filters": [], "reviewed": True, "dry_run": True}).status_code == 403
        assert a.patch(f"/api/datasets/{ds['id']}", json={"target_columns": ["y", "c"]}).status_code == 403
        assert c.post(f"/api/datasets/{ds['id']}/finalize", json={"accept_open": True}).status_code == 403
        assert _save(lead, fid, action="drop").status_code == 200
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")


def test_targets_are_fixed_once_assessors_are_assigned(people):
    lead, a = people["lead"], people["a"]
    ds = _upload(lead)
    try:
        assert a.patch(f"/api/datasets/{ds['id']}", json={"target_columns": ["y", "c"]}).status_code == 200
        assert _assign(lead, ds, a)
        assert _fields(a, ds)["c"]["action"] == "target", "assessors start from the dataset's targets"
        r = lead.patch(f"/api/datasets/{ds['id']}", json={"target_columns": ["y"]})
        assert r.status_code == 409 and "fixed" in r.text
        assert lead.patch(f"/api/datasets/{ds['id']}", json={"name": "renamed"}).status_code == 200
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")


def test_an_assessors_bulk_edit_and_undo_touch_only_their_assessment(people):
    lead, a, b = people["lead"], people["a"], people["b"]
    ds = _upload(lead)
    try:
        _assign(lead, ds, a, b)
        w = _fields(a, ds)["w"]["id"]
        r = a.post(f"/api/datasets/{ds['id']}/decisions/bulk",
                   json={"filters": [{"column": "name", "op": "=", "value": "w"}], "action": "drop"})
        assert r.status_code == 200 and r.json()["changed"] == 1
        assert _fields(a, ds)["w"]["action"] == "drop"
        assert _fields(b, ds)["w"]["action"] == "keep" and _fields(lead, ds)["w"]["action"] == "keep"
        assert a.post(f"/api/datasets/{ds['id']}/fields/query", json={
            "filters": [{"column": "action", "op": "=", "value": "drop"}]}).json()[0]["field_id"] == w
        assert b.post(f"/api/datasets/{ds['id']}/decisions/bulk/{r.json()['batch_id']}/undo").status_code == 404
        assert a.post(f"/api/datasets/{ds['id']}/decisions/bulk/{r.json()['batch_id']}/undo").json()["restored"] == 1
        assert _fields(a, ds)["w"]["action"] == "keep"
        assert a.post(f"/api/datasets/{ds['id']}/decisions/bulk/{r.json()['batch_id']}/undo").status_code == 409
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")


def test_submit_needs_every_field_reviewed_and_can_be_undone_until_review(people):
    lead, a, b = people["lead"], people["a"], people["b"]
    ds = _upload(lead)
    try:
        _assign(lead, ds, a, b)
        r = a.post(f"/api/datasets/{ds['id']}/assessment/submit")
        assert r.status_code == 409 and "5 left" in r.text
        _review_all(a, ds)
        team = a.post(f"/api/datasets/{ds['id']}/assessment/submit").json()
        assert team["me"] == {"submitted": True, "reviewed": 5, "total": 5} and team["recipe"]["phase"] == "assessing"
        assert _save(a, _fields(a, ds)["x"]["id"], action="drop").status_code == 409, "submitted is locked"
        assert a.post(f"/api/datasets/{ds['id']}/assessment/unsubmit").json()["me"]["submitted"] is False
        assert _save(a, _fields(a, ds)["x"]["id"], action="drop").status_code == 200
        assert a.post(f"/api/datasets/{ds['id']}/assessment/submit").status_code == 200
        assert people["c"].post(f"/api/datasets/{ds['id']}/assessment/submit").status_code == 403
        progress = {x["user_id"]: x for x in lead.get(f"/api/datasets/{ds['id']}/team").json()["assessors"]}
        assert progress[a.user["id"]]["submitted_at"] and progress[b.user["id"]]["submitted_at"] is None
        assert progress[b.user["id"]]["reviewed"] == 0
        _review_all(b, ds)
        assert b.post(f"/api/datasets/{ds['id']}/assessment/submit").json()["recipe"]["phase"] == "review"
        assert a.post(f"/api/datasets/{ds['id']}/assessment/unsubmit").status_code == 409
        assert lead.delete(f"/api/datasets/{ds['id']}/assessors/{a.user['id']}").status_code == 409
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")


def test_unassigning_the_last_one_who_hasnt_submitted_opens_review(people):
    lead, a, b = people["lead"], people["a"], people["b"]
    ds = _upload(lead)
    try:
        _assign(lead, ds, a, b)
        _review_all(a, ds)
        a.post(f"/api/datasets/{ds['id']}/assessment/submit")
        assert _phase(lead, ds) == "assessing"
        team = lead.delete(f"/api/datasets/{ds['id']}/assessors/{b.user['id']}").json()
        assert team["recipe"]["phase"] == "review" and [x["user_id"] for x in team["assessors"]] == [a.user["id"]]
        assert team["review"] == {"fields": 5, "adopted": 5, "disagreements": 0, "open": 0}
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")


def test_unassigning_everyone_leaves_it_assessing(people):
    lead, a = people["lead"], people["a"]
    ds = _upload(lead)
    try:
        _assign(lead, ds, a)
        assert lead.delete(f"/api/datasets/{ds['id']}/assessors/{a.user['id']}").json()["recipe"]["phase"] == "assessing"
        assert lead.delete(f"/api/datasets/{ds['id']}/assessors/{a.user['id']}").status_code == 404
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")


def test_review_adopts_agreement_queues_disagreement_and_finalize_freezes(people):
    from app.db import fetch_one

    lead, a, b, c = people["lead"], people["a"], people["b"], people["c"]
    ds = _upload(lead)
    try:
        _assign(lead, ds, a, b)
        f = _fields(a, ds)
        # Agreement on z with different settings: they tie, so the suggestion wins (a baseline has one model type).
        assert _save(a, f["z"]["id"], imputation="mean", notes="z looks fine").status_code == 200
        assert _save(b, f["z"]["id"], imputation="constant", impute_value="0").status_code == 200
        cur = a.get(f"/api/fields/{f['z']['id']}").json()["decision"]
        r = a.put(f"/api/fields/{f['z']['id']}/decision", json={**{k: cur[k] for k in KEYS}, "family": "linear"})
        assert r.status_code == 422, "a baseline has one model type"
        # Disagreement on x (keep vs drop) and on w's type.
        assert _save(b, f["x"]["id"], action="drop", notes="x is noise").status_code == 200
        assert _save(a, f["w"]["id"], semantic_type="numeric_discrete").status_code == 200
        r = lead.get(f"/api/datasets/{ds['id']}/review")
        assert r.status_code == 409, "no review while assessing"
        for who in (a, b):
            _review_all(who, ds)
            assert who.post(f"/api/datasets/{ds['id']}/assessment/submit").status_code == 200
        assert _phase(lead, ds) == "review"

        team = {x["name"]: x for x in lead.get(f"/api/datasets/{ds['id']}/fields").json()}
        assert team["z"]["reviewed"] and team["c"]["reviewed"] and not team["x"]["reviewed"]
        z_tree = lead.get(f"/api/fields/{f['z']['id']}?family=tree").json()["decision"]
        assert z_tree["imputation"] == "none" and z_tree["settings_source"] == "suggested"
        assert z_tree["notes"] == f"{a.user['name']}: z looks fine"
        hist = lead.get(f"/api/fields/{f['z']['id']}/history").json()
        assert hist and hist[0]["user_id"] == b.user["id"], "adoption is recorded, by whoever opened review"

        queue = c.get(f"/api/datasets/{ds['id']}/review").json()
        assert queue["summary"] == {"fields": 5, "adopted": 3, "disagreements": 2, "open": 2}
        x = next(q for q in queue["queue"] if q["name"] == "x")
        assert {q["name"] for q in queue["queue"]} == {"x", "w"}
        assert {(v["user_id"], v["action"], v["notes"]) for v in x["assessments"]} == {
            (a.user["id"], "keep", None), (b.user["id"], "drop", "x is noise")}
        detail = c.get(f"/api/fields/{f['x']['id']}/review").json()
        assert detail["disagreement"] and set(detail["assessments"][0]["settings"]) == {"tree", "linear", "neural"}

        r = c.post(f"/api/fields/{f['x']['id']}/comments", json={"body": "  I'd drop it  "})
        assert r.status_code == 201 and r.json()[-1]["body"] == "I'd drop it"
        assert c.post(f"/api/fields/{f['x']['id']}/comments", json={"body": "   "}).status_code == 422
        assert c.get(f"/api/fields/{f['x']['id']}/review").json()["comments"][0]["user_id"] == c.user["id"]

        r = lead.post(f"/api/datasets/{ds['id']}/finalize", json={})
        assert r.status_code == 409 and r.json()["detail"]["open"] == 2
        assert _save(lead, f["x"]["id"], action="drop", reviewed=True).status_code == 200
        assert lead.get(f"/api/datasets/{ds['id']}/review").json()["summary"]["open"] == 1
        r = lead.post(f"/api/datasets/{ds['id']}/finalize", json={"accept_open": True})
        assert r.status_code == 200, r.text
        assert r.json()["recipe"]["phase"] == "finalized"
        assert r.json()["recipe"]["finalized_by_name"] == lead.user["name"]

        rec = fetch_one("SELECT * FROM meta.recipes WHERE id = %s", (r.json()["recipe"]["id"],))
        snap = rec["snapshot"]
        assert snap["target_order"] == ["y"] and set(snap["fields"]) == {"x", "z", "w", "c", "y"}
        assert snap["fields"]["x"]["decision"] == {"action": "drop", "semantic_type": "numeric_continuous",
                                                   "notes": None, "target_task": None, "target_bins": None}
        assert set(snap["fields"]["z"]["settings"]) == {"tree"}  # a baseline keeps its own model type's settings
        assert snap["fields"]["x"]["inferred_type"] == "numeric_continuous"
        assert snap["fields"]["x"]["null_pct"] > 0.1 and snap["fields"]["c"]["n_unique"] == 3
        assert rec["schema_fingerprint"] and rec["finalized_by"] == lead.user["id"]

        r = _save(lead, f["z"]["id"], action="drop")
        assert r.status_code == 409 and "finalized" in r.text
        assert c.post(f"/api/fields/{f['x']['id']}/comments", json={"body": "late"}).status_code == 409
        with pytest.raises(Exception, match="finalized"):
            from app.db import execute
            execute("UPDATE meta.recipes SET snapshot = '{}' WHERE id = %s", (rec["id"],))
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")


def test_team_exports_wait_for_a_finalized_recipe(people):
    lead, a = people["lead"], people["a"]
    ds = _upload(lead)
    try:
        r = lead.post(f"/api/datasets/{ds['id']}/exports", json={})
        assert r.status_code == 409 and "Trees baseline v1 is being assessed" in r.json()["detail"]
        pv = lead.get(f"/api/datasets/{ds['id']}/export-preview").json()
        assert pv["recipe"]["phase"] == "assessing" and "finalizes" in pv["blocking"][0]
        assert lead.post(f"/api/datasets/{ds['id']}/exports", json={"source": "assessment"}).status_code == 422
        _assign(lead, ds, a)
        _review_all(a, ds)
        a.post(f"/api/datasets/{ds['id']}/assessment/submit")
        assert "team review" in lead.post(f"/api/datasets/{ds['id']}/exports", json={}).json()["detail"]
        team = lead.post(f"/api/datasets/{ds['id']}/finalize", json={}).json()
        exp = lead.post(f"/api/datasets/{ds['id']}/exports", json={})
        assert exp.status_code == 202, exp.text
        exp = _wait(lambda: (x := lead.get(f"/api/exports/{exp.json()['id']}").json())["status"] in ("done", "failed")
                    and x)
        assert exp["status"] == "done", exp["error"]
        assert exp["recipe_id"] == team["recipe"]["id"]
        manifest = json.loads((Path(exp["output_dir"]) / "manifest.json").read_text())
        assert (manifest["config"]["recipe_id"], manifest["config"]["recipe_version"]) == (team["recipe"]["id"], 1)
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")


# ------------------------------------------------------------------------------------------- blindness
SECRET_NOTE = f"B-secret-{uuid.uuid4().hex}"
SECRET_VALUE = "987654.321"


@pytest.fixture(scope="module")
def blind(people):
    """A dataset being assessed by A and B, where B has made choices nobody else may see: a note, a constant fill
    value, dropping w and retyping z, by single saves and a bulk edit, and a "Kept only" Explore result; and a
    personal experiment on that assessment (named with the secret), with an export, a run and an Explore result on
    the export."""
    lead, a, b = people["lead"], people["a"], people["b"]
    ds = _upload(lead)
    _assign(lead, ds, a, b)
    f = _fields(b, ds)
    assert _save(b, f["x"]["id"], imputation="constant", impute_value=SECRET_VALUE, notes=SECRET_NOTE).status_code == 200
    assert _save(b, f["z"]["id"], semantic_type="numeric_discrete").status_code == 200
    r = b.post(f"/api/datasets/{ds['id']}/decisions/bulk", json={"field_ids": [f["w"]["id"]], "action": "drop"})
    assert r.status_code == 200 and r.json()["changed"] == 1
    an = b.post(f"/api/datasets/{ds['id']}/analyses",
                json={"kind": "importance", "params": {"target": "y", "kept": True, "sample": 1000}}).json()
    an = _wait(lambda: (x := b.get(f"/api/analyses/{an['id']}").json())["status"] in ("done", "failed") and x)
    assert an["status"] == "done", an["error"]
    assert an["params"]["decisions"].startswith("assessment:")
    # A has made a (different) choice of their own, so A reads their assessment rather than the team's.
    assert _save(a, f["c"]["id"], notes="A's own note", reviewed=True).status_code == 200
    x = b.post(f"/api/datasets/{ds['id']}/experiments", json={
        "name": f"B's {SECRET_NOTE}", "source": f"assessment:{ds['recipe_id']}:{b.user['id']}"}).json()
    exp = b.post(f"/api/datasets/{ds['id']}/exports", json={"experiment_id": x["id"]}).json()
    exp = _wait(lambda: (e := b.get(f"/api/exports/{exp['id']}").json())["status"] in ("done", "failed") and e)
    assert exp["status"] == "done", exp["error"]
    run = b.post(f"/api/exports/{exp['id']}/runs", json={"model": "lightgbm", "cv": 0, "params": {"n_estimators": 5}}
                 ).json()
    run = _wait(lambda: (e := b.get(f"/api/runs/{run['id']}").json())["status"] in ("done", "failed") and e)
    assert run["status"] == "done", run["error"]
    xan = b.post(f"/api/datasets/{ds['id']}/analyses",
                 json={"kind": "importance", "params": {"data": f"export:{exp['id']}", "target": "y", "sample": 1000}}
                 ).json()
    assert "id" in xan, xan
    yield {"ds": ds, "fields": f, "b_analysis": an, "b_batch": r.json()["batch_id"], "b_xp": x["id"],
           "b_export": exp["id"], "b_run": run["id"], "b_export_analysis": xan["id"]}
    lead.delete(f"/api/datasets/{ds['id']}")


def _sql(client, q):
    return client.post("/api/sql", json={"sql": q})


# Every read endpoint, as (name, call). Each returns a Response; the check is on its whole JSON text, plus the
# structural checks in test_no_read_path_shows_another_assessors_choices.
READS = {
    "datasets list": lambda c, d, f, x: c.get("/api/datasets"),
    "dataset": lambda c, d, f, x: c.get(f"/api/datasets/{d}"),
    "fields list": lambda c, d, f, x: c.get(f"/api/datasets/{d}/fields"),
    "field detail tree": lambda c, d, f, x: c.get(f"/api/fields/{f['x']['id']}?family=tree"),
    "field detail linear": lambda c, d, f, x: c.get(f"/api/fields/{f['x']['id']}?family=linear"),
    "field detail w": lambda c, d, f, x: c.get(f"/api/fields/{f['w']['id']}"),
    "field detail z": lambda c, d, f, x: c.get(f"/api/fields/{f['z']['id']}"),
    "field correlations": lambda c, d, f, x: c.get(f"/api/fields/{f['x']['id']}/correlations"),
    "field history x": lambda c, d, f, x: c.get(f"/api/fields/{f['x']['id']}/history"),
    "field history w": lambda c, d, f, x: c.get(f"/api/fields/{f['w']['id']}/history"),
    "finder": lambda c, d, f, x: c.post(f"/api/datasets/{d}/fields/query", json={}),
    "finder on notes": lambda c, d, f, x: c.post(f"/api/datasets/{d}/fields/query", json={
        "filters": [{"column": "notes", "op": "contains", "value": "secret"}]}),
    "finder on drops": lambda c, d, f, x: c.post(f"/api/datasets/{d}/fields/query", json={
        "filters": [{"column": "action", "op": "=", "value": "drop"}]}),
    "bulk preview": lambda c, d, f, x: c.post(f"/api/datasets/{d}/decisions/bulk", json={
        "filters": [], "accept_suggestion": True, "dry_run": True}),
    "bulk preview on notes": lambda c, d, f, x: c.post(f"/api/datasets/{d}/decisions/bulk", json={
        "filters": [{"column": "notes", "op": "contains", "value": "secret"}], "reviewed": True, "dry_run": True}),
    "undo another's batch": lambda c, d, f, x: c.post(f"/api/datasets/{d}/decisions/bulk/{x['b_batch']}/undo"),
    "export preview": lambda c, d, f, x: c.get(f"/api/datasets/{d}/export-preview"),
    "exports": lambda c, d, f, x: c.get(f"/api/datasets/{d}/exports"),
    "team": lambda c, d, f, x: c.get(f"/api/datasets/{d}/team"),
    "review queue": lambda c, d, f, x: c.get(f"/api/datasets/{d}/review"),
    "field review": lambda c, d, f, x: c.get(f"/api/fields/{f['x']['id']}/review"),
    "recipe versions": lambda c, d, f, x: c.get(f"/api/datasets/{d}/recipes"),
    "recipe diff": lambda c, d, f, x: c.get(
        f"/api/datasets/{d}/recipes/diff?a={x['ds']['recipe_id']}&b={x['ds']['recipe_id']}"),
    "finalized recipes": lambda c, d, f, x: c.get("/api/recipes"),
    "explore correlations kept": lambda c, d, f, x: c.get(f"/api/datasets/{d}/explore/correlations?kept=true"),
    "explore correlations": lambda c, d, f, x: c.get(f"/api/datasets/{d}/explore/correlations"),
    "explore kept-only analysis": lambda c, d, f, x: c.post(f"/api/datasets/{d}/analyses", json={
        "kind": "importance", "params": {"target": "y", "kept": True, "sample": 1000}}),
    "another's analysis": lambda c, d, f, x: c.get(f"/api/analyses/{x['b_analysis']['id']}"),
    "sql assessments": lambda c, d, f, x: _sql(c, "SELECT * FROM meta.assessments"),
    "sql assessment settings": lambda c, d, f, x: _sql(c, "SELECT * FROM meta.assessment_settings"),
    "sql assessment batches": lambda c, d, f, x: _sql(c, "SELECT * FROM meta.assessment_batches"),
    "sql assessors": lambda c, d, f, x: _sql(c, "SELECT * FROM meta.dataset_assessors"),
    "sql comments": lambda c, d, f, x: _sql(c, "SELECT * FROM meta.review_comments"),
    "sql v_fields": lambda c, d, f, x: _sql(c, f"SELECT * FROM meta.v_fields WHERE dataset_id = {d}"),
    "sql decisions": lambda c, d, f, x: _sql(c, "SELECT * FROM meta.field_decisions"),
    "sql settings": lambda c, d, f, x: _sql(c, "SELECT * FROM meta.v_field_settings"),
    "sql history": lambda c, d, f, x: _sql(c, "SELECT * FROM meta.decision_history"),
    "sql bulk batches": lambda c, d, f, x: _sql(c, "SELECT * FROM meta.bulk_batches"),
    "sql analyses": lambda c, d, f, x: _sql(c, "SELECT * FROM meta.analyses"),
    "sql recipes": lambda c, d, f, x: _sql(c, "SELECT * FROM meta.recipes"),
    # B's personal experiment, its export and run, and anything built on them.
    "experiments": lambda c, d, f, x: c.get(f"/api/datasets/{d}/experiments"),
    "experiment": lambda c, d, f, x: c.get(f"/api/experiments/{x['b_xp']}"),
    "rename experiment": lambda c, d, f, x: c.patch(f"/api/experiments/{x['b_xp']}", json={"notes": "mine"}),
    "exports of experiment": lambda c, d, f, x: c.get(f"/api/datasets/{d}/exports?experiment_id={x['b_xp']}"),
    "export": lambda c, d, f, x: c.get(f"/api/exports/{x['b_export']}"),
    "export data": lambda c, d, f, x: c.get(f"/api/exports/{x['b_export']}/files/data"),
    "export pipeline": lambda c, d, f, x: c.get(f"/api/exports/{x['b_export']}/files/preprocessor"),
    "export manifest": lambda c, d, f, x: c.get(f"/api/exports/{x['b_export']}/files/manifest"),
    "export models": lambda c, d, f, x: c.get(f"/api/exports/{x['b_export']}/models"),
    "export preview of experiment": lambda c, d, f, x: c.get(f"/api/datasets/{d}/export-preview?experiment_id={x['b_xp']}"),
    "export into experiment": lambda c, d, f, x: c.post(f"/api/datasets/{d}/exports", json={"experiment_id": x["b_xp"]}),
    "train on export": lambda c, d, f, x: c.post(f"/api/exports/{x['b_export']}/runs", json={"cv": 0}),
    "runs": lambda c, d, f, x: c.get(f"/api/datasets/{d}/runs"),
    "runs of experiment": lambda c, d, f, x: c.get(f"/api/datasets/{d}/runs?experiment_id={x['b_xp']}"),
    "run": lambda c, d, f, x: c.get(f"/api/runs/{x['b_run']}"),
    "run model": lambda c, d, f, x: c.get(f"/api/runs/{x['b_run']}/files/model"),
    "run report": lambda c, d, f, x: c.get(f"/api/runs/{x['b_run']}/files/report"),
    "explore on export": lambda c, d, f, x: c.post(f"/api/datasets/{d}/analyses", json={
        "kind": "importance", "params": {"data": f"export:{x['b_export']}", "target": "y", "sample": 1000}}),
    "another's export analysis": lambda c, d, f, x: c.get(f"/api/analyses/{x['b_export_analysis']}"),
    "sql experiments": lambda c, d, f, x: _sql(c, f"SELECT * FROM meta.experiments WHERE dataset_id = {d}"),
    "sql exports": lambda c, d, f, x: _sql(c, f"SELECT * FROM meta.exports WHERE dataset_id = {d}"),
    "sql model runs": lambda c, d, f, x: _sql(c, f"SELECT * FROM meta.model_runs WHERE dataset_id = {d}"),
    "sql model metrics": lambda c, d, f, x: _sql(c, f"SELECT * FROM meta.v_model_metrics WHERE dataset_id = {d}"),
}


# Reads that must be refused outright, whatever they would contain.
REFUSED = {"undo another's batch": 404, "field detail linear": 422, "another's analysis": 404, "review queue": 409, "field review": 409,
           **{name: 404 for name in ("experiment", "rename experiment", "export", "export data", "export pipeline",
                                     "export manifest", "export models", "export preview of experiment",
                                     "export into experiment", "train on export", "run", "run model", "run report",
                                     "explore on export", "another's export analysis")},
           **{name: 400 for name in ("sql assessments", "sql assessment settings", "sql assessment batches",
                                     "sql assessors", "sql comments")}}


@pytest.mark.parametrize("viewer", ["a", "lead", "c"])
@pytest.mark.parametrize("read", list(READS))
def test_no_read_path_shows_another_assessors_choices(people, blind, viewer, read):
    client = people[viewer]
    ds, f = blind["ds"]["id"], blind["fields"]
    r = READS[read](client, ds, f, blind)
    text = r.text
    assert SECRET_NOTE not in text and SECRET_VALUE not in text, f"{read} leaked B's choices to {viewer}"
    refused = REFUSED.get(read) or (403 if read.startswith("bulk") and viewer != "a" else None)
    if refused:
        assert r.status_code == refused, (read, r.status_code, text)
        return
    assert r.status_code == 200, (read, text)
    body = r.json()
    if read == "fields list":
        mine = {x["name"]: x for x in body}
        assert mine["w"]["action"] == "keep" and mine["z"]["semantic_type"] == "numeric_continuous"
        assert mine["c"]["reviewed"] is (viewer == "a")
    elif read.startswith("field detail"):
        assert body["decision"]["action"] == "keep" and body["decision"]["semantic_type"] == "numeric_continuous"
        assert body["decision"]["imputation"] != "constant"
    elif read == "finder on drops":
        assert body == []
    elif read == "bulk preview":
        assert all(ch["before"]["action"] != "drop" for ch in body["changes"])
    elif read == "bulk preview on notes":
        assert body["matched"] == 0
    elif read.startswith("explore correlations"):
        assert {x["name"]: x["action"] for x in body["fields"]}["w"] == "keep"
    elif read == "explore kept-only analysis":
        assert body["id"] != blind["b_analysis"]["id"]
    elif read == "recipe diff":
        assert body["fields"] == []
    elif read == "sql analyses":
        idx = body["columns"].index("id")
        ids = [row[idx] for row in body["rows"]]
        assert blind["b_analysis"]["id"] not in ids and blind["b_export_analysis"] not in ids
    elif read == "datasets list":
        mine = next(x for x in body if x["id"] == ds)
        assert (mine["export_count"], mine["run_count"], mine["last_export"], mine["last_run"]) == (0, 0, None, None)
    elif read in ("experiments", "exports of experiment", "runs", "runs of experiment"):
        assert body == []
    elif read.startswith("sql ") and read[4:] in ("experiments", "exports", "model runs", "model metrics"):
        assert body["rows"] == [], read


def test_the_console_role_has_no_access_to_private_tables(app_, people):
    from app import db
    from app.config import get_settings

    role = get_settings().readonly_role
    db.bootstrap()  # grants are re-applied on every start; the exclusions must survive that
    for table in db.PRIVATE_TABLES:
        assert db.fetch_one("SELECT has_table_privilege(%s, %s, 'SELECT') AS ok", (role, f"meta.{table}"))["ok"] is False
        r = _sql(people["lead"], f"SELECT count(*) FROM meta.{table}")
        assert r.status_code == 400 and "permission denied" in r.json()["detail"]["message"]
    assert db.fetch_one("SELECT has_table_privilege(%s, 'meta.recipes', 'SELECT') AS ok", (role,))["ok"]


def test_existing_datasets_migrate_to_a_finalized_v1_and_keep_exporting(app_, people):
    from app import recipes
    from app.db import MIGRATIONS_DIR, connection, fetch_one

    lead = people["lead"]
    ds = _upload(lead)
    try:
        fid = _fields(lead, ds)["x"]["id"]
        with connection() as conn:
            # As the data was before recipes: no recipe, an export made from the decisions, a customised field.
            conn.execute("UPDATE meta.field_decisions SET action = 'drop', notes = 'old note' WHERE field_id = %s",
                         (fid,))
            conn.execute("DELETE FROM meta.recipes WHERE dataset_id = %s", (ds["id"],))
            old = conn.execute("""INSERT INTO meta.exports (dataset_id, status, model_family, config)
                                  VALUES (%s, 'done', 'tree', '{}') RETURNING id""", (ds["id"],)).fetchone()
            # 008 and the migrations after it, which redefine some of what it creates.
            for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
                if path.name >= "008":
                    conn.execute(path.read_text())
            recipes.backfill_snapshots(conn)
        rec = fetch_one("SELECT * FROM meta.recipes WHERE dataset_id = %s", (ds["id"],))
        first_lead = fetch_one("SELECT id FROM meta.users WHERE role = 'lead' ORDER BY id LIMIT 1")["id"]
        assert (rec["version"], rec["phase"], rec["created_by"], rec["finalized_by"]) == (1, "finalized", first_lead,
                                                                                          first_lead)
        assert rec["snapshot"]["fields"]["x"]["decision"] == {"action": "drop", "semantic_type": "numeric_continuous",
                                                              "notes": "old note", "target_task": None,
                                                              "target_bins": None}
        assert fetch_one("SELECT recipe_id FROM meta.exports WHERE id = %s", (old["id"],))["recipe_id"] == rec["id"]
        exp = lead.post(f"/api/datasets/{ds['id']}/exports", json={})
        assert exp.status_code == 202, exp.text
        exp = _wait(lambda: (x := lead.get(f"/api/exports/{exp.json()['id']}").json())["status"] in ("done", "failed")
                    and x)
        assert exp["status"] == "done" and exp["recipe_id"] == rec["id"]
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")
