"""Baselines: one per model type, each assessed and finalized like a recipe version, started from scratch, from
another baseline, or from another dataset's. Needs RAINFOREST_TEST_DATABASE_URL."""
from .conftest import set_phase
from .team_helpers import app_, people  # noqa: F401  (fixtures)
from .team_helpers import fields, save, start, submit_and_finalize, upload


def test_recipes_have_a_family_and_versions_count_per_family(app_):
    from app.db import connection

    with connection() as conn:
        cols = {r["column_name"] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_schema='meta' AND table_name='recipes'")}
        assert "family" in cols
        idx = conn.execute("SELECT indexdef FROM pg_indexes WHERE schemaname='meta' AND tablename='recipes'").fetchall()
        assert any("(dataset_id, family, version)" in r["indexdef"] for r in idx)
        xcols = {r["column_name"] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_schema='meta' AND table_name='experiments'")}
        assert "is_default" in xcols


def test_labels_and_per_family_versions(people):
    from app import recipes
    from app.db import connection

    ds = upload(people["lead"])
    with connection() as conn:
        r = recipes.current(conn, ds["id"])
        assert r["family"] == "tree" and recipes.label(r) == "Trees baseline v1"
        assert recipes.next_version(conn, ds["id"], "linear") == 1
        assert recipes.next_version(conn, ds["id"], "tree") == 2


def test_snapshot_holds_only_the_baselines_family(people):
    from app import recipes
    from app.db import connection

    ds = upload(people["lead"])
    set_phase(ds["id"], "finalized")
    with connection() as conn:
        snap = recipes.snapshot_of(conn, recipes.current(conn, ds["id"]))
    assert snap["fields"] and all(set(f["settings"]) == {"tree"} for f in snap["fields"].values())


def test_a_new_version_reproduces_its_parent_exactly(people):
    """Retyped and customised fields carry over as they were, not re-derived from suggestions."""
    from app import recipes
    from app.db import connection

    lead = people["lead"]
    ds = upload(lead)
    start(lead, ds, "linear", [lead])
    f = fields(lead, ds)
    assert save(lead, f["z"]["id"], semantic_type="numeric_discrete", reviewed=True).status_code == 200
    assert save(lead, f["x"]["id"], imputation="constant", impute_value="7", reviewed=True).status_code == 200
    st = submit_and_finalize(lead, ds)
    with connection() as conn:
        parent = recipes.get(conn, st["recipe"]["id"])
        child = recipes.start_version(conn, ds["id"], lead.user["id"], "linear", "review")
        assert child["family"] == "linear" and child["version"] == 2
        now = recipes.snapshot(conn, ds["id"], family="linear")
        assert recipes.diff(recipes.snapshot_of(conn, parent), now, "linear")["fields"] == []


def test_new_version_loads_its_own_parent_not_the_last_worked_baseline(people):
    lead = people["lead"]
    ds = upload(lead)
    f = fields(lead, ds)
    start(lead, ds, "tree", [lead])
    assert save(lead, f["x"]["id"], action="drop", reviewed=True).status_code == 200
    submit_and_finalize(lead, ds)
    start(lead, ds, "linear", [lead])
    assert fields(lead, ds)["x"]["action"] == "keep"
    submit_and_finalize(lead, ds)
    r = lead.post(f"/api/datasets/{ds['id']}/recipes", json={"family": "tree", "reopen": []})
    assert r.status_code == 200, r.text
    assert (r.json()["recipe"]["family"], r.json()["recipe"]["version"]) == ("tree", 2)
    assert fields(lead, ds)["x"]["action"] == "drop"


def test_start_sets_up_the_first_baseline_in_place(people):
    lead, a = people["lead"], people["a"]
    ds = upload(lead)
    st = start(lead, ds, "linear", [lead, a])
    assert (st["recipe"]["family"], st["recipe"]["version"], st["recipe"]["phase"]) == ("linear", 1, "assessing")
    assert {x["user_id"] for x in st["assessors"]} == {lead.user["id"], a.user["id"]}
    d = lead.get(f"/api/datasets/{ds['id']}").json()
    assert d["model_family"] == "linear" and d["recipe_id"] == st["recipe"]["id"]
    assert fields(a, ds)["y"]["action"] == "target"
    start(lead, ds, "tree", [lead], expect=409)  # fixed once assessment started


def test_start_validates_its_input(people):
    lead, a = people["lead"], people["a"]
    ds = upload(lead)
    base = {"family": "tree", "start": "scratch", "targets": ["y"], "assessors": [lead.user["id"]]}
    for patch, code in (({"family": "svm"}, 422), ({"assessors": []}, 422), ({"assessors": [999999]}, 422),
                        ({"targets": []}, 422), ({"targets": ["z"]}, 422), ({"targets": ["nope"]}, 422),
                        ({"start": "baseline", "source_recipe_id": 999999}, 404),
                        ({"start": "baseline"}, 422)):
        r = lead.post(f"/api/datasets/{ds['id']}/baselines", json={**base, **patch})
        assert r.status_code == code, (patch, r.text)
    assert a.post(f"/api/datasets/{ds['id']}/baselines", json=base).status_code == 403
    assert lead.get(f"/api/datasets/{ds['id']}").json()["baseline"]["phase"] == "assessing"  # nothing half-done
    assert lead.get(f"/api/datasets/{ds['id']}/team").json()["assessors"] == []


def test_populate_copies_decisions_and_only_valid_settings(people):
    lead = people["lead"]
    ds = upload(lead)
    f = fields(lead, ds)
    start(lead, ds, "tree", [lead])
    assert save(lead, f["w"]["id"], action="drop", reviewed=True).status_code == 200
    trees = submit_and_finalize(lead, ds)["recipe"]["id"]
    start(lead, ds, "linear", [lead], start="baseline", source=trees)
    now = fields(lead, ds)
    assert now["w"]["action"] == "drop" and not now["w"]["reviewed"]
    x = lead.get(f"/api/fields/{f['x']['id']}").json()["decision"]
    assert x["family"] == "linear" and x["imputation"] != "none"  # trees' "leave missing" isn't valid for linear


def test_one_baseline_in_progress_at_a_time(people):
    lead = people["lead"]
    ds = upload(lead)
    start(lead, ds, "tree", [lead])
    submit_and_finalize(lead, ds)
    start(lead, ds, "linear", [lead])
    r = lead.post(f"/api/datasets/{ds['id']}/baselines", json={"family": "neural", "start": "scratch",
                  "targets": ["y"], "assessors": [lead.user["id"]]})
    assert r.status_code == 409 and "finalize" in r.json()["detail"]
    assert lead.post(f"/api/datasets/{ds['id']}/recipes", json={"family": "tree", "reopen": []}).status_code == 409


def test_baselines_are_listed_per_family(people):
    lead = people["lead"]
    ds = upload(lead)
    start(lead, ds, "tree", [lead])
    submit_and_finalize(lead, ds)
    start(lead, ds, "linear", [lead])
    out = lead.get(f"/api/datasets/{ds['id']}/baselines").json()
    assert [b["family"] for b in out] == ["tree", "linear"]
    assert out[0]["label"] == "Trees" and out[0]["latest_finalized_id"] == out[0]["versions"][0]["id"]
    assert out[1]["latest_finalized_id"] is None and out[1]["versions"][0]["label"] == "Linear baseline v1"


def test_start_from_another_datasets_baseline(people):
    lead, a = people["lead"], people["a"]
    src = upload(lead)
    start(lead, src, "linear", [lead])
    rid = submit_and_finalize(lead, src)["recipe"]["id"]
    ds = upload(lead)
    st = start(lead, ds, "tree", [lead, a], start="recipe", source=rid)
    assert st["recipe"]["family"] == "linear" and st["recipe"]["reuse"]["source"]["recipe_id"] == rid
    assert st["recipe"]["phase"] in ("assessing", "review")


def test_diff_compares_by_id_and_across_families_by_decision_only(people):
    lead = people["lead"]
    ds = upload(lead)
    f = fields(lead, ds)
    start(lead, ds, "tree", [lead])
    trees = submit_and_finalize(lead, ds)["recipe"]["id"]
    start(lead, ds, "linear", [lead], start="baseline", source=trees)
    assert save(lead, f["w"]["id"], action="drop", reviewed=True).status_code == 200
    linear = submit_and_finalize(lead, ds)["recipe"]["id"]
    d = lead.get(f"/api/datasets/{ds['id']}/recipes/diff?a={trees}&b={linear}").json()
    assert [x["name"] for x in d["fields"]] == ["w"]
    assert all(c["family"] is None for c in d["fields"][0]["changes"])
    other = upload(lead)
    assert lead.get(f"/api/datasets/{other['id']}/recipes/diff?a={trees}&b={linear}").status_code == 404


def test_dataset_shows_its_baselines_and_refuses_family_switch(people):
    lead = people["lead"]
    ds = upload(lead)
    d = lead.get(f"/api/datasets/{ds['id']}").json()
    assert d["baseline"]["label"] == "Trees baseline v1" and d["baseline"]["setup"] is True
    assert [b["family"] for b in d["baselines"]] == ["tree"]
    r = lead.patch(f"/api/datasets/{ds['id']}", json={"model_family": "linear"})
    assert r.status_code == 422 and "baseline" in r.json()["detail"]
    start(lead, ds, "tree", [lead])
    submit_and_finalize(lead, ds)
    start(lead, ds, "linear", [lead])
    row = next(x for x in lead.get("/api/datasets").json() if x["id"] == ds["id"])
    assert row["baseline"]["label"] == "Linear baseline v1" and row["baseline"]["setup"] is False
    assert {b["family"]: (b["version"], b["phase"], b["latest_finalized_id"] is not None)
            for b in row["baselines"]} == {"tree": (1, "finalized", True), "linear": (1, "assessing", False)}
