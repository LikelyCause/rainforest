"""Recipe versions and reuse. The matching, drift and diff rules are tested on their own; the rest runs against a real
Postgres (RAINFOREST_TEST_DATABASE_URL): new versions with and without reopened fields, locked carried fields, the
reopened-only assessment, version history and diffs, reusing a recipe on new data, provenance, and exports built from
a version's snapshot."""
import json
import uuid
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import recipes
from app.reuse import drift, match

from .test_team import _assign, _csv, _fields, _review_all, _save, _upload, _wait, app_, people  # noqa: F401


# ------------------------------------------------------------------------------------------- pure rules
def _stats(null_pct=0.0, n_unique=10, t="numeric_continuous"):
    return {"inferred_type": t, "null_pct": null_pct, "n_unique": n_unique}


@pytest.mark.parametrize("before,after,drifted", [
    (0.0, 0.10, False), (0.7, 0.8, False), (0.8, 0.7, False), (0.2, 0.3001, True), (0.5, 0.39, True),
])
def test_null_share_drifts_past_ten_points(before, after, drifted):
    out = drift(_stats(null_pct=before), _stats(null_pct=after))
    assert (out is not None) is drifted
    if drifted:
        assert out == {"null_pct": [before, after]}


@pytest.mark.parametrize("before,after,drifted", [
    (10, 20, False), (10, 5, False), (10, 21, True), (10, 4, True), (0, 0, False), (0, 3, True), (3, 0, True),
    (3, 6, False), (3, 7, True),
])
def test_distinct_values_drift_outside_half_to_double(before, after, drifted):
    out = drift(_stats(n_unique=before), _stats(n_unique=after))
    assert (out is not None) is drifted
    if drifted:
        assert out == {"n_unique": [before, after]}


def test_match_carries_unchanged_fields_and_reopens_the_rest():
    snap = {"fields": {"a": _stats(), "b": _stats(), "c": _stats(0.0, 10), "gone": _stats()}}
    new = [{"name": "a", **_stats()}, {"name": "b", **_stats(t="categorical")}, {"name": "c", **_stats(0.3, 10)},
           {"name": "new", **_stats()}]
    m = match(snap, new)
    assert m["carried"] == ["a"] and m["missing"] == ["gone"]
    assert m["reopened"] == [
        {"name": "b", "reason": "type_changed", "before": "numeric_continuous", "after": "categorical"},
        {"name": "c", "reason": "drift", "drift": {"null_pct": [0.0, 0.3]}},
        {"name": "new", "reason": "new"},
    ]


def _entry(action="keep", t="numeric_continuous", notes=None, **tree):
    settings = {"imputation": "none", "impute_value": None, "add_missing_indicator": False, "transform": "none",
                "scaling": "none", "encoding": "none", "params": {}}
    return {"decision": {"action": action, "semantic_type": t, "notes": notes},
            "settings": {"tree": {**settings, **tree}, "linear": settings}}


def test_diff_lists_changed_decisions_settings_and_targets():
    a = {"fields": {"x": _entry(), "y": _entry("target", "binary"), "z": _entry(impute_value=0), "old": _entry()},
         "target_order": ["y"]}
    b = {"fields": {"x": _entry("drop"), "y": _entry("target", "binary"), "z": _entry(impute_value="0"),
                    "w": _entry(imputation="mean"), "new": _entry()}, "target_order": ["y"]}
    b["fields"]["y"]["settings"]["linear"] = {**b["fields"]["y"]["settings"]["linear"], "scaling": "standard"}
    out = recipes.diff(a, b)
    by = {f["name"]: f for f in out["fields"]}
    assert set(by) == {"x", "y", "new", "old", "w"}, "0 and '0' are the same fill value"
    assert by["x"]["changes"] == [{"key": "action", "family": None, "before": "keep", "after": "drop"}]
    assert by["y"]["changes"] == [{"key": "scaling", "family": "linear", "before": "none", "after": "standard"}]
    assert (by["new"]["change"], by["old"]["change"], by["w"]["change"]) == ("added", "removed", "added")
    assert out["targets"] is None
    assert [f["name"] for f in recipes.diff(a, b, family="tree")["fields"]] == ["x", "w", "new", "old"]
    b["fields"]["x"]["decision"]["action"] = "target"
    assert recipes.diff(a, b)["targets"] == {"before": ["y"], "after": ["y", "x"]}


# --------------------------------------------------------------------------------------------- helpers
def _finalize_v1(people, ds, edits=None):
    """Take a fresh dataset through one assessor and review to a finalized v1; `edits` (name -> patch) are set by the
    lead in review."""
    lead, a = people["lead"], people["a"]
    _assign(lead, ds, a)
    _review_all(a, ds)
    assert a.post(f"/api/datasets/{ds['id']}/assessment/submit").json()["recipe"]["phase"] == "review"
    f = _fields(lead, ds)
    for name, patch in (edits or {}).items():
        assert _save(lead, f[name]["id"], **patch).status_code == 200
    r = lead.post(f"/api/datasets/{ds['id']}/finalize", json={"accept_open": True})
    assert r.status_code == 200, r.text
    return r.json()["recipe"]


def _snapshot(recipe_id):
    from app.db import fetch_one
    return fetch_one("SELECT snapshot FROM meta.recipes WHERE id = %s", (recipe_id,))["snapshot"]


def _export(client, ds, **body):
    r = client.post(f"/api/datasets/{ds['id']}/exports", json=body)
    assert r.status_code == 202, r.text
    exp = _wait(lambda: (x := client.get(f"/api/exports/{r.json()['id']}").json())["status"] in ("done", "failed")
                and x)
    assert exp["status"] == "done", exp["error"]
    return exp, json.loads((Path(exp["output_dir"]) / "manifest.json").read_text())


# ------------------------------------------------------------------------------------------ new versions
def test_a_new_version_reopens_only_the_chosen_fields(people):
    lead, a, b, c = people["lead"], people["a"], people["b"], people["c"]
    ds = _upload(lead)
    try:
        v1 = _finalize_v1(people, ds)
        f = _fields(lead, ds)
        ids = {n: f[n]["id"] for n in f}
        r = _save(lead, ids["z"], action="drop")
        assert r.status_code == 409 and "new version" in r.json()["detail"], "a finalized version says how to change"
        assert a.post(f"/api/datasets/{ds['id']}/recipes", json={"reopen": [ids["x"]]}).status_code == 403
        r = lead.post(f"/api/datasets/{ds['id']}/recipes", json={"reopen": [ids["x"], 10 ** 12]})
        assert r.status_code == 422
        team = lead.post(f"/api/datasets/{ds['id']}/recipes", json={"reopen": [ids["x"], ids["w"]]}).json()
        assert (team["recipe"]["version"], team["recipe"]["phase"], team["recipe"]["scope"]) == (2, "assessing",
                                                                                                "reopened")
        assert team["recipe"]["reopened"] == 2 and team["recipe"]["parent_id"] == v1["id"]
        assert lead.post(f"/api/datasets/{ds['id']}/recipes", json={}).status_code == 409, "one open version at most"

        team = _assign(lead, ds, a, b)
        assert {x["total"] for x in team["assessors"]} == {2}
        mine = _fields(a, ds)
        assert mine["z"]["locked"] and mine["z"]["reviewed"] and not mine["x"]["locked"] and not mine["x"]["reviewed"]
        detail = a.get(f"/api/fields/{ids['z']}").json()
        assert "carried over" in detail["locked"] and a.get(f"/api/fields/{ids['x']}").json()["locked"] is None
        r = _save(a, ids["z"], action="drop")
        assert r.status_code == 409 and "z: carried over into recipe v2" in r.json()["detail"]
        r = a.post(f"/api/datasets/{ds['id']}/assessment/submit")
        assert r.status_code == 409 and "2 left" in r.text
        r = a.post(f"/api/datasets/{ds['id']}/decisions/bulk", json={"filters": [], "action": "drop", "dry_run": True})
        assert r.json()["matched"] == 2, "bulk edit passes over carried fields"
        assert _save(a, ids["x"], action="drop", notes="x drifts").status_code == 200
        for who in (a, b):
            _review_all(who, ds)
            assert who.post(f"/api/datasets/{ds['id']}/assessment/submit").status_code == 200
        team = lead.get(f"/api/datasets/{ds['id']}/team").json()
        assert team["recipe"]["phase"] == "review", "review opens by itself"
        assert team["review"] == {"fields": 2, "adopted": 1, "disagreements": 1, "open": 1}
        queue = c.get(f"/api/datasets/{ds['id']}/review").json()["queue"]
        assert [q["name"] for q in queue] == ["x"]
        assert c.get(f"/api/fields/{ids['z']}/review").json()["assessments"] == [], "carried fields aren't reviewed"
        assert _save(lead, ids["z"], action="drop").status_code == 409, "carried fields stay locked in review"
        assert lead.patch(f"/api/datasets/{ds['id']}", json={"target_columns": ["y", "c"]}).status_code == 409
        assert _save(lead, ids["x"], action="drop", reviewed=True).status_code == 200
        v2 = lead.post(f"/api/datasets/{ds['id']}/finalize", json={}).json()["recipe"]
        assert v2["phase"] == "finalized"

        s1, s2 = _snapshot(v1["id"]), _snapshot(v2["id"])
        assert s1["fields"]["x"]["decision"]["action"] == "keep", "v1 is untouched"
        assert s2["fields"]["x"]["decision"]["action"] == "drop" and s2["fields"]["z"] == s1["fields"]["z"]
        d = c.get(f"/api/datasets/{ds['id']}/recipes/diff?a=1&b=2").json()
        assert [x["name"] for x in d["fields"]] == ["x"]
        assert {"key": "action", "family": None, "before": "keep", "after": "drop"} in d["fields"][0]["changes"]
        assert c.get(f"/api/datasets/{ds['id']}/recipes/diff?a=1&b=9").status_code == 404

        versions = c.get(f"/api/datasets/{ds['id']}/recipes").json()
        assert [(v["version"], v["phase"], v["reopened"]) for v in versions] == [(2, "finalized", 2),
                                                                                (1, "finalized", None)]
        assert versions[0]["created_by_name"] == lead.user["name"] and versions[0]["finalized_at"]
        assert versions[0]["parent_version"] == 1
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")


def test_a_new_version_with_nothing_reopened_is_edited_directly(people):
    lead, a = people["lead"], people["a"]
    ds = _upload(lead)
    try:
        _finalize_v1(people, ds)
        team = lead.post(f"/api/datasets/{ds['id']}/recipes", json={"reopen": []}).json()
        assert (team["recipe"]["phase"], team["recipe"]["scope"], team["recipe"]["reopened"]) == ("review", "all",
                                                                                                 None)
        assert team["review"]["open"] == 0
        z = _fields(lead, ds)["z"]["id"]
        assert _save(a, z, action="drop").status_code == 403
        assert _save(lead, z, imputation="mean").status_code == 200
        hist = lead.get(f"/api/fields/{z}/history").json()
        assert hist[0]["user_id"] == lead.user["id"], "direct edits are in the history"
        assert lead.post(f"/api/datasets/{ds['id']}/finalize", json={}).json()["recipe"]["version"] == 2
        d = lead.get(f"/api/datasets/{ds['id']}/recipes/diff?a=1&b=2").json()
        assert d["fields"][0]["changes"] == [{"key": "imputation", "family": "tree", "before": "none",
                                              "after": "mean"}]
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")


def test_versions_and_diffs_show_no_assessment_while_assessing(people):
    lead, a, b, c = people["lead"], people["a"], people["b"], people["c"]
    secret = f"secret-{uuid.uuid4().hex}"
    ds = _upload(lead)
    try:
        _finalize_v1(people, ds)
        x = _fields(lead, ds)["x"]["id"]
        lead.post(f"/api/datasets/{ds['id']}/recipes", json={"reopen": [x]})
        _assign(lead, ds, a, b)
        assert _save(b, x, action="drop", imputation="constant", impute_value="13579.2468", notes=secret
                     ).status_code == 200
        for who in (a, lead, c):
            for path in (f"/api/datasets/{ds['id']}/recipes", f"/api/datasets/{ds['id']}/recipes/diff?a=1&b=2",
                         f"/api/datasets/{ds['id']}/recipes/diff?a=2&b=1", "/api/recipes",
                         f"/api/datasets/{ds['id']}/team", f"/api/datasets/{ds['id']}/export-preview"):
                r = who.get(path)
                assert r.status_code == 200, (path, r.text)
                assert secret not in r.text and "13579.2468" not in r.text, f"{path} leaked to {who.user['name']}"
            assert who.get(f"/api/datasets/{ds['id']}/recipes/diff?a=1&b=2").json()["fields"] == []
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")


def test_exports_are_built_from_a_versions_snapshot(people):
    lead = people["lead"]
    ds = _upload(lead)
    other = None
    try:
        v1 = _finalize_v1(people, ds)
        e1, m1 = _export(lead, ds)
        assert (e1["recipe_id"], e1["recipe_version"]) == (v1["id"], 1)
        assert (m1["config"]["recipe_id"], m1["config"]["recipe_version"]) == (v1["id"], 1)

        lead.post(f"/api/datasets/{ds['id']}/recipes", json={})
        f = _fields(lead, ds)
        assert _save(lead, f["x"]["id"], action="drop").status_code == 200
        assert _save(lead, f["z"]["id"], imputation="mean").status_code == 200
        pv = lead.get(f"/api/datasets/{ds['id']}/export-preview").json()
        assert pv["recipe"]["version"] == 1 and not pv["blocking"], "the latest finalized version exports meanwhile"
        assert pv["changed_since_export"] == 0
        e2, m2 = _export(lead, ds)
        assert e2["recipe_version"] == 1 and m2["fields"] == m1["fields"], "editing v2 doesn't change v1's export"

        v2 = lead.post(f"/api/datasets/{ds['id']}/finalize", json={}).json()["recipe"]
        pv = lead.get(f"/api/datasets/{ds['id']}/export-preview").json()
        assert pv["recipe"]["version"] == 2 and pv["changed_since_export"] == 2
        assert lead.get(f"/api/datasets/{ds['id']}/export-preview?recipe_id={v1['id']}").json()["kept"] == pv["kept"] + 1
        e3, m3 = _export(lead, ds, recipe_id=v1["id"])
        assert e3["recipe_version"] == 1 and m3["fields"] == m1["fields"], "v1 re-exports the same way"
        e4, m4 = _export(lead, ds)
        assert e4["recipe_id"] == v2["id"] and "x" not in {s["name"] for s in m4["fields"]}
        assert next(s for s in m4["fields"] if s["name"] == "z")["imputation"] == "mean"
        assert [e["recipe_version"] for e in lead.get(f"/api/datasets/{ds['id']}/exports").json()] == [2, 1, 1, 1]

        lead.post(f"/api/datasets/{ds['id']}/recipes", json={"reopen": [f["x"]["id"]]})
        cur = lead.get(f"/api/datasets/{ds['id']}/team").json()["recipe"]
        r = lead.post(f"/api/datasets/{ds['id']}/exports", json={"recipe_id": cur["id"]})
        assert r.status_code == 409 and "being assessed" in r.json()["detail"]
        other = _upload(lead)
        r = lead.post(f"/api/datasets/{other['id']}/exports", json={"recipe_id": v1["id"]})
        assert r.status_code == 404
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")
        if other:
            lead.delete(f"/api/datasets/{other['id']}")


# --------------------------------------------------------------------------------------------------- reuse
def _changed_csv(n=400, seed=3) -> str:
    """The _csv dataset with x retyped (labels), z mostly missing (drift), w renamed to w2 and a new column n."""
    rng = np.random.default_rng(seed)
    return "x,z,w2,c,y,n\n" + "".join(
        f"v{i % 5},{'' if i % 3 == 0 else f'{b:.3f}'},{c:.3f},{'pqr'[i % 3]},{'ab'[i % 2]},{i % 50}\n"
        for i, (b, c) in enumerate(zip(rng.normal(size=n), rng.normal(size=n))))


def test_a_new_upload_starts_from_a_recipe(people):
    from app.db import fetch_one

    lead, a = people["lead"], people["a"]
    src = _upload(lead)
    new = other = None
    try:
        v1 = _finalize_v1(people, src, {"c": {"notes": "keep c as is", "imputation": "most_frequent"}})
        listed = {r["id"]: r for r in a.get("/api/recipes").json()}
        assert listed[v1["id"]]["dataset_name"] == src["name"] and listed[v1["id"]]["field_count"] == 5
        assert listed[v1["id"]]["finalized_by_name"] == lead.user["name"]

        new = _upload(lead, _changed_csv())
        assert a.post(f"/api/datasets/{new['id']}/recipes/apply", json={"recipe_id": v1["id"]}).status_code == 403
        assert lead.post(f"/api/datasets/{new['id']}/recipes/apply", json={"recipe_id": 10 ** 12}).status_code == 404
        team = lead.post(f"/api/datasets/{new['id']}/recipes/apply", json={"recipe_id": v1["id"]}).json()
        rec = team["recipe"]
        assert (rec["version"], rec["phase"], rec["scope"], rec["parent_id"]) == (1, "assessing", "reopened",
                                                                                 v1["id"])
        reuse = rec["reuse"]
        assert reuse["source"] == {"recipe_id": v1["id"], "dataset_id": src["id"], "dataset_name": src["name"],
                                   "version": 1}
        assert reuse["carried"] == ["c", "y"] and reuse["missing"] == ["w"] and reuse["targets"] == ["y"]
        why = {r["name"]: r for r in reuse["reopened"]}
        assert [r["name"] for r in reuse["reopened"]] == ["x", "z", "w2", "n"]
        assert why["x"]["reason"] == "type_changed" and why["x"]["before"] == "numeric_continuous"
        assert why["z"]["reason"] == "drift" and why["z"]["drift"]["null_pct"][0] == 0
        assert why["z"]["drift"]["null_pct"][1] == pytest.approx(134 / 400)
        assert why["w2"]["reason"] == "new" and why["n"]["reason"] == "new"
        assert rec["reopened"] == 4

        f = _fields(lead, new)
        assert f["y"]["action"] == "target" and f["c"]["locked"] and not f["x"]["locked"]
        c_detail = lead.get(f"/api/fields/{f['c']['id']}?family=tree").json()["decision"]
        assert c_detail["imputation"] == "most_frequent" and c_detail["notes"] == "keep c as is"
        assert _save(lead, f["c"]["id"], action="drop").status_code in (403, 409)
        r = lead.post(f"/api/datasets/{new['id']}/recipes/apply", json={"recipe_id": v1["id"]})
        assert r.status_code == 409, "a recipe is applied once"
        team = _assign(lead, new, a)
        assert team["assessors"][0]["total"] == 4

        # Deleting the source dataset keeps the provenance in the summary.
        lead.delete(f"/api/datasets/{src['id']}")
        src = None
        row = fetch_one("SELECT parent_id, reuse FROM meta.recipes WHERE id = %s", (rec["id"],))
        assert row["parent_id"] is None and row["reuse"]["source"]["recipe_id"] == v1["id"]

        other = _upload(lead)
        _assign(lead, other, a)
        lead.delete(f"/api/datasets/{other['id']}/assessors/{a.user['id']}")
        r = lead.post(f"/api/datasets/{other['id']}/recipes/apply", json={"recipe_id": v1["id"]})
        assert r.status_code == 409 and "Assessment has started" in r.json()["detail"]
    finally:
        for d in (src, new, other):
            if d:
                lead.delete(f"/api/datasets/{d['id']}")


def test_reuse_with_nothing_to_assess_is_confirmed_by_finalizing(people):
    lead, a = people["lead"], people["a"]
    src = _upload(lead)
    new = None
    try:
        v1 = _finalize_v1(people, src, {"z": {"imputation": "mean"}})
        r = a.post("/api/datasets/upload", files={"file": ("t.csv", _csv(), "text/csv")},
                   data={"recipe_id": str(v1["id"])})
        assert r.status_code == 403
        r = lead.post("/api/datasets/upload", files={"file": ("t.csv", _csv(), "text/csv")},
                      data={"recipe_id": str(v1["id"])})
        assert r.status_code == 202, r.text
        new = _wait(lambda: (d := lead.get(f"/api/datasets/{r.json()['id']}").json())["status"] in ("ready", "failed")
                    and d)
        assert new["status"] == "ready" and new["phase"] == "review", "applied while processing; nothing to assess"
        team = lead.get(f"/api/datasets/{new['id']}/team").json()
        assert team["recipe"]["reuse"]["carried"] == ["x", "z", "w", "c", "y"] and team["recipe"]["reopened"] == 0
        assert team["review"] == {"fields": 0, "adopted": 0, "disagreements": 0, "open": 0}
        assert a.post(f"/api/datasets/{new['id']}/finalize", json={}).status_code == 403
        done = lead.post(f"/api/datasets/{new['id']}/finalize", json={}).json()["recipe"]
        assert done["phase"] == "finalized" and done["parent_id"] == v1["id"]
        assert recipes.diff(_snapshot(v1["id"]), _snapshot(done["id"]))["fields"] == []
        # The source can still be deleted; the finalized copy only loses the link, not its provenance.
        assert lead.delete(f"/api/datasets/{src['id']}").status_code == 200
        src = None
        done = lead.get(f"/api/datasets/{new['id']}/recipes").json()[0]
        assert done["parent_id"] is None and done["reuse"]["source"]["recipe_id"] == v1["id"]
    finally:
        for d in (src, new):
            if d:
                lead.delete(f"/api/datasets/{d['id']}")


def _levels_csv(levels: int, n=2000) -> str:
    rng = np.random.default_rng(5)
    return "k,v\n" + "".join(f"L{i % levels},{x:.3f}\n" for i, x in enumerate(rng.normal(size=n)))


def test_targets_carry_over_only_while_still_valid(people):
    lead = people["lead"]
    src = _upload(lead, _levels_csv(60))
    new = None
    try:
        assert lead.patch(f"/api/datasets/{src['id']}", json={"target_columns": ["k"]}).status_code == 200
        v1 = _finalize_v1(people, src)
        assert _snapshot(v1["id"])["fields"]["k"]["decision"]["action"] == "target"
        new = _upload(lead, _levels_csv(110))
        f = _fields(lead, new)
        if f["k"]["inferred_type"] != _snapshot(v1["id"])["fields"]["k"]["inferred_type"]:
            pytest.skip("profiling typed the wider column differently")
        reuse = lead.post(f"/api/datasets/{new['id']}/recipes/apply",
                          json={"recipe_id": v1["id"]}).json()["recipe"]["reuse"]
        assert reuse["targets"] == [] and [t["name"] for t in reuse["targets_not_carried"]] == ["k"]
        assert "at most 100" in reuse["targets_not_carried"][0]["problem"]
        assert {r["name"]: r["reason"] for r in reuse["reopened"]} == {"k": "target"}
        assert _fields(lead, new)["k"]["action"] != "target"
    finally:
        for d in (src, new):
            if d:
                lead.delete(f"/api/datasets/{d['id']}")
