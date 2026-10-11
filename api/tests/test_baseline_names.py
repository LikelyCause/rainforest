"""Baseline names: one optional name per baseline (model type) of a dataset, shared by its versions, shown in every
label. Needs RAINFOREST_TEST_DATABASE_URL."""
import pytest

from .team_helpers import app_, people  # noqa: F401  (fixtures)
from .team_helpers import finalized_trees, start, upload


def _name(ds_id: int, family: str, name: str) -> None:
    from app.db import connection

    with connection() as conn:
        conn.execute("INSERT INTO meta.baseline_names (dataset_id, family, name) VALUES (%s, %s, %s)",
                     (ds_id, family, name))


def test_labels_use_the_name(people):
    from app import recipes

    assert recipes.label({"family": "tree", "version": 2, "name": None}) == "Trees baseline v2"
    assert recipes.label({"family": "tree", "version": 2, "name": "Dropout risk"}) == "Dropout risk v2 (Trees)"
    assert recipes.short_label("linear", 1, "Churn") == "Churn v1"
    assert recipes.short_label("linear", 1, None) == "Linear v1"


def test_label_needs_the_name_key(people):
    from app import recipes

    with pytest.raises(KeyError):
        recipes.label({"family": "tree", "version": 1})


def test_name_rules(people):
    from fastapi import HTTPException

    from app import recipes

    assert recipes.clean_name("  Dropout risk ") == "Dropout risk" and recipes.clean_name("   ") is None
    assert recipes.clean_name(None) is None and recipes.clean_name("x" * 60) == "x" * 60
    with pytest.raises(HTTPException) as e:
        recipes.clean_name("x" * 61)
    assert e.value.status_code == 422


def test_recipe_rows_carry_the_name(people):
    from app import recipes
    from app.db import connection

    ds = upload(people["lead"])
    _name(ds["id"], "tree", "Dropout risk")
    with connection() as conn:
        cur = recipes.current(conn, ds["id"])
        assert cur["name"] == "Dropout risk" and recipes.label(cur) == "Dropout risk v1 (Trees)"


def test_every_baseline_payload_shows_the_name(people):
    lead = people["lead"]
    ds, rid = finalized_trees(lead)
    _name(ds["id"], "tree", "Dropout risk")
    i = ds["id"]
    row = next(x for x in lead.get("/api/datasets").json() if x["id"] == i)
    assert row["baseline"]["label"] == "Dropout risk v1 (Trees)" and row["baseline"]["name"] == "Dropout risk"
    assert row["baselines"][0]["name"] == "Dropout risk" and row["baselines"][0]["label"] == "Dropout risk v1 (Trees)"
    assert lead.get(f"/api/datasets/{i}").json()["baseline"]["name"] == "Dropout risk"
    assert lead.get(f"/api/datasets/{i}/team").json()["recipe"]["label"] == "Dropout risk v1 (Trees)"
    line = lead.get(f"/api/datasets/{i}/baselines").json()[0]
    assert line["name"] == "Dropout risk" and line["versions"][0]["label"] == "Dropout risk v1 (Trees)"
    assert next(r for r in lead.get("/api/recipes").json() if r["id"] == rid)["name"] == "Dropout risk"
    assert lead.get(f"/api/datasets/{i}/recipes").json()[0]["label"] == "Dropout risk v1 (Trees)"
    assert lead.get(f"/api/datasets/{i}/recipes/diff?a={rid}&b={rid}").json()["a"]["label"] == "Dropout risk v1 (Trees)"
    pv = lead.get(f"/api/datasets/{i}/export-preview").json()["recipe"]
    assert pv["label"] == "Dropout risk v1 (Trees)" and pv["name"] == "Dropout risk"
    r = lead.post(f"/api/datasets/{i}/exports", json={})
    assert r.status_code == 202, r.text
    assert r.json()["experiment_name"] == "Team · Dropout risk v1"
    assert next(e for e in lead.get(f"/api/datasets/{i}/exports").json() if e["id"] == r.json()["id"])["recipe_name"] == "Dropout risk"
    team_x = next(x for x in lead.get(f"/api/datasets/{i}/experiments").json() if x["is_default"])
    assert team_x["recipe_name"] == "Dropout risk"
    other = upload(lead)
    st = start(lead, other, "tree", [lead], start="recipe", source=rid)
    assert st["recipe"]["reuse"]["source"]["name"] == "Dropout risk"
    v = next(v for v in lead.get(f"/api/datasets/{other['id']}/recipes").json() if v["reuse"])
    assert v["reuse"]["source"]["name"] == "Dropout risk"
    line = lead.get(f"/api/datasets/{other['id']}/baselines").json()[0]["versions"][0]
    assert line["reuse"]["source"]["name"] == "Dropout risk" and line["parent_dataset_name"] == ds["name"]
    assert "reopened" in line


def _start(lead, ds, family, name=None, expect=200):
    body = {"family": family, "start": "scratch", "targets": ["y"], "assessors": [lead.user["id"]]}
    if name is not None:
        body["name"] = name
    r = lead.post(f"/api/datasets/{ds['id']}/baselines", json=body)
    assert r.status_code == expect, r.text
    return r.json()


def _names(lead, ds) -> dict:
    return {b["family"]: b["name"] for b in lead.get(f"/api/datasets/{ds['id']}/baselines").json()}


def test_a_name_given_at_setup_goes_with_the_chosen_model_type(people):
    from .team_helpers import submit_and_finalize

    lead = people["lead"]
    ds = upload(lead)
    st = _start(lead, ds, "linear", "  Dropout risk ")
    assert st["recipe"]["label"] == "Dropout risk v1 (Linear)"
    assert _names(lead, ds) == {"linear": "Dropout risk"}
    submit_and_finalize(lead, ds)
    _start(lead, ds, "tree", "dropout RISK", expect=422)
    _start(lead, ds, "tree", "x" * 61, expect=422)
    _start(lead, ds, "tree", "   ")
    assert _names(lead, ds) == {"linear": "Dropout risk", "tree": None}


def test_an_existing_baseline_keeps_its_name_when_a_version_starts(people):
    from .team_helpers import submit_and_finalize

    lead = people["lead"]
    ds = upload(lead)
    _start(lead, ds, "tree", "Dropout risk")
    submit_and_finalize(lead, ds)
    _start(lead, ds, "tree", "Something else", expect=422)
    st = _start(lead, ds, "tree", "Dropout risk")
    assert st["recipe"]["label"] == "Dropout risk v2 (Trees)"


def test_a_lead_renames_a_finalized_baseline_without_touching_its_snapshot(people, login_as, app_):
    from fastapi.testclient import TestClient

    from app.db import connection

    from .team_helpers import submit_and_finalize

    lead, member = people["lead"], people["a"]
    ds = upload(lead)
    _start(lead, ds, "tree")
    rid = submit_and_finalize(lead, ds)["recipe"]["id"]
    with connection() as conn:
        before = conn.execute("SELECT snapshot::text AS s FROM meta.recipes WHERE id = %s", (rid,)).fetchone()["s"]
    url = f"/api/datasets/{ds['id']}/baselines/tree"
    r = lead.patch(url, json={"name": "Dropout risk"})
    assert r.status_code == 200, r.text
    assert r.json()[0]["name"] == "Dropout risk" and r.json()[0]["versions"][0]["label"] == "Dropout risk v1 (Trees)"
    with connection() as conn:
        after = conn.execute("SELECT snapshot::text AS s FROM meta.recipes WHERE id = %s", (rid,)).fetchone()["s"]
    assert after == before
    assert member.patch(url, json={"name": "Mine"}).status_code == 403
    viewer = login_as(TestClient(app_), "viewer")
    assert viewer.patch(url, json={"name": "Mine"}).status_code == 403
    assert lead.patch(f"/api/datasets/{ds['id']}/baselines/linear", json={"name": "Nope"}).status_code == 404
    assert lead.patch(url, json={"name": "y" * 61}).status_code == 422
    assert lead.patch(url, json={"name": ""}).status_code == 200 and _names(lead, ds) == {"tree": None}


def test_deleting_a_dataset_deletes_its_names(people):
    from app.db import connection

    lead = people["lead"]
    ds = upload(lead)
    _start(lead, ds, "tree", "Dropout risk")
    assert _names(lead, ds) == {"tree": "Dropout risk"}
    assert lead.delete(f"/api/datasets/{ds['id']}").status_code == 200
    with connection() as conn:
        assert conn.execute("SELECT 1 FROM meta.baseline_names WHERE dataset_id = %s", (ds["id"],)).fetchone() is None


def test_a_names_case_survives_in_lowercased_messages(people):
    from app import recipes

    assert recipes.label({"family": "tree", "version": 2, "name": "Dropout risk"}, lower=True) == "Dropout risk v2 (Trees)"
    assert recipes.label({"family": "tree", "version": 2, "name": None}, lower=True) == "trees baseline v2"


def test_a_taken_name_is_reported_as_stored_and_patch_needs_a_name(people):
    from .team_helpers import submit_and_finalize

    lead = people["lead"]
    ds = upload(lead)
    _start(lead, ds, "linear", "Dropout risk")
    submit_and_finalize(lead, ds)
    r = lead.post(f"/api/datasets/{ds['id']}/baselines", json={
        "family": "tree", "start": "scratch", "targets": ["y"], "assessors": [lead.user["id"]], "name": "dropout RISK"})
    assert r.status_code == 422 and "'Dropout risk'" in r.json()["detail"]
    assert lead.patch(f"/api/datasets/{ds['id']}/baselines/linear", json={}).status_code == 422
    assert _names(lead, ds) == {"linear": "Dropout risk"}


def test_a_name_left_by_a_model_type_without_versions_never_blocks(people):
    lead = people["lead"]
    ds = upload(lead)
    _start(lead, ds, "tree")
    _name(ds["id"], "neural", "Dropout risk")
    r = lead.patch(f"/api/datasets/{ds['id']}/baselines/tree", json={"name": "Dropout risk"})
    assert r.status_code == 200, r.text
    from app.db import connection

    with connection() as conn:
        rows = conn.execute("SELECT family FROM meta.baseline_names WHERE dataset_id = %s", (ds["id"],)).fetchall()
    assert [r["family"] for r in rows] == ["tree"]


def test_system_says_whether_mlflow_is_on(people):
    assert isinstance(people["lead"].get("/api/system").json()["mlflow"], bool)
