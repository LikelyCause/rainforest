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
    assert lead.get(f"/api/datasets/{i}/export-preview").json()["recipe"]["label"] == "Dropout risk v1 (Trees)"
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
