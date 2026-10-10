"""Experiment adjustments: a team experiment's own copy of its baseline, changed through view=experiment:<id>,
exported and logged as baseline + adjustments. Needs RAINFOREST_TEST_DATABASE_URL."""
from .team_helpers import app_, people  # noqa: F401  (fixtures)
from .team_helpers import adjust, experiment, export, fields, finalized_trees, save, upload


def test_experiment_copies_exist_and_go_with_their_experiment(people):
    from app.db import connection

    lead = people["lead"]
    ds, rid = finalized_trees(lead)
    x = experiment(lead, ds, rid)
    with connection() as conn:
        n = conn.execute("SELECT count(*) AS n FROM meta.experiment_decisions WHERE experiment_id = %s",
                         (x["id"],)).fetchone()["n"]
        assert n == len(fields(lead, ds))
        conn.execute("DELETE FROM meta.experiments WHERE id = %s", (x["id"],))
        assert conn.execute("SELECT count(*) AS n FROM meta.experiment_settings WHERE experiment_id = %s",
                            (x["id"],)).fetchone()["n"] == 0


def test_owner_adjusts_through_the_experiment_view(people):
    lead, a, b = people["lead"], people["a"], people["b"]
    ds, rid = finalized_trees(lead)
    x = experiment(a, ds, rid, "A tries")
    view = f"experiment:{x['id']}"
    f = fields(a, ds, view)
    cur = a.get(f"/api/fields/{f['w']['id']}?view={view}").json()
    assert cur["family"] == "tree" and cur["baseline"]["action"] == "keep"
    assert save(a, f["w"]["id"], view, action="drop").status_code == 200
    assert fields(a, ds, view)["w"]["action"] == "drop"
    assert fields(b, ds, view)["w"]["action"] == "drop"  # team experiments are visible to everyone
    assert fields(lead, ds, f"baseline:{rid}")["w"]["action"] == "keep"  # the baseline's snapshot didn't move
    assert fields(lead, ds)["w"]["action"] == "keep"  # nor did the working copy
    assert save(b, f["w"]["id"], view, action="keep").status_code == 403
    assert save(lead, f["w"]["id"], view, action="keep").status_code == 403  # owners only, leads included
    r = a.post(f"/api/datasets/{ds['id']}/decisions/bulk?view={view}", json={"field_ids": [f["x"]["id"]], "action": "drop"})
    assert r.status_code == 200 and r.json()["changed"] == 1
    assert fields(a, ds, view)["x"]["action"] == "drop"
    assert a.post(f"/api/datasets/{ds['id']}/decisions/bulk/{r.json()['batch_id']}/undo?view={view}").status_code == 200
    assert fields(a, ds, view)["x"]["action"] == "keep"
    q = a.post(f"/api/datasets/{ds['id']}/fields/query?view={view}", json={"filters": [], "order_by": "name"})
    assert {r["name"]: r["action"] for r in q.json()}["w"] == "drop"


def test_views_are_refused_where_they_must_be(people):
    lead, a = people["lead"], people["a"]
    ds, rid = finalized_trees(lead)
    other, _ = finalized_trees(lead)
    x = experiment(a, ds, rid)
    assert lead.get(f"/api/datasets/{ds['id']}/fields?view=nonsense").status_code == 422
    assert lead.get(f"/api/datasets/{ds['id']}/fields?view=experiment:999999").status_code == 404
    assert lead.get(f"/api/datasets/{other['id']}/fields?view=experiment:{x['id']}").status_code == 404
    fid = fields(lead, ds)["w"]["id"]
    assert a.get(f"/api/fields/{fid}?view=experiment:{x['id']}&family=linear").status_code == 422
    team = lead.post(f"/api/datasets/{ds['id']}/exports", json={}).json()
    default = team["experiment_id"]
    assert save(lead, fid, f"experiment:{default}", action="drop").status_code == 403  # default team experiments
    a.patch(f"/api/experiments/{x['id']}", json={"archived": True})
    assert save(a, fid, f"experiment:{x['id']}", action="drop").status_code == 409


def test_a_baseline_only_takes_its_own_model_type(people):
    lead = people["lead"]
    ds = upload(lead)
    from .team_helpers import start
    start(lead, ds, "tree", [lead])
    fid = fields(lead, ds)["x"]["id"]
    assert lead.get(f"/api/fields/{fid}?family=linear").status_code == 422
    assert save(lead, fid, family="linear").status_code == 422
    assert lead.get(f"/api/fields/{fid}").json()["family"] == "tree"


def test_experiment_view_shows_its_own_targets(people):
    a = people["a"]
    ds, rid = finalized_trees(people["lead"])
    x = experiment(a, ds, rid)
    view = f"experiment:{x['id']}"
    adjust(a, ds, x, "y", action="keep")
    adjust(a, ds, x, "c", action="target")
    f = fields(a, ds, view)
    assert f["c"]["is_target"] and not f["y"]["is_target"]
    assert fields(a, ds)["y"]["is_target"]


def test_adjustments_stay_in_their_experiment_and_exports_freeze_them(people):
    lead, a = people["lead"], people["a"]
    ds, rid = finalized_trees(lead)
    x1, x2 = experiment(a, ds, rid, "one"), experiment(a, ds, rid, "two")
    adjust(a, ds, x1, "w", action="drop")
    e = export(a, ds, x1)
    adjust(a, ds, x1, "x", action="drop")  # after the export
    man = a.get(f"/api/exports/{e['id']}/files/manifest").json()
    names = {f["name"] for f in man["fields"]}
    assert "w" not in names and "x" in names
    assert e["config"]["adjustments"]["count"] == 1 and e["config"]["adjustments"]["fields"] == ["w"]
    assert e["config"]["baseline"] == {"recipe_id": rid, "family": "tree", "version": 1, "label": "Trees baseline v1"}
    info = a.get(f"/api/experiments/{x1['id']}").json()
    assert info["adjustments"]["count"] == 2 and info["can_adjust"]
    assert {f["name"] for f in info["adjustments"]["fields"]} == {"w", "x"}
    assert a.get(f"/api/experiments/{x2['id']}").json()["adjustments"]["count"] == 0
    assert not lead.get(f"/api/experiments/{x1['id']}").json()["can_adjust"]
    row = next(r for r in a.get(f"/api/datasets/{ds['id']}/experiments").json() if r["id"] == x1["id"])
    assert row["adjusted_count"] == 2 and row["targets_changed"] is False
    xid = fields(a, ds)["x"]["id"]
    r = a.post(f"/api/experiments/{x1['id']}/adjustments/revert", json={"field_ids": [xid]})
    assert r.status_code == 200 and r.json()["adjustments"]["count"] == 1
    assert lead.post(f"/api/experiments/{x1['id']}/adjustments/revert", json={"field_ids": None}).status_code == 403
    r = a.post(f"/api/experiments/{x1['id']}/adjustments/revert", json={"field_ids": None})
    assert r.status_code == 200 and r.json()["adjustments"]["count"] == 0


def test_retype_in_experiment_resets_settings_to_valid_ones(people):
    lead = people["lead"]
    ds, rid = finalized_trees(lead)
    x = experiment(lead, ds, rid, "retype")
    view = f"experiment:{x['id']}"
    fid = fields(lead, ds)["z"]["id"]
    assert save(lead, fid, view, semantic_type="categorical", encoding="ordinal").status_code == 200
    pv = lead.get(f"/api/datasets/{ds['id']}/export-preview?experiment_id={x['id']}").json()
    assert pv["kept_by_type"].get("categorical", 0) >= 2  # c and the retyped z
    assert not any("aren't valid" in b for b in pv["blocking"])


def test_experiment_targets_are_validated(people):
    lead = people["lead"]
    ds, rid = finalized_trees(lead)
    x = experiment(lead, ds, rid, "targets")
    adjust(lead, ds, x, "x", expect=422, action="target")  # continuous: not a class label
    adjust(lead, ds, x, "y", action="keep")
    pv = lead.get(f"/api/datasets/{ds['id']}/export-preview?experiment_id={x['id']}").json()
    assert "No target selected" in pv["blocking"]
    adjust(lead, ds, x, "c", action="target")
    info = lead.get(f"/api/experiments/{x['id']}").json()["adjustments"]
    assert info["targets_changed"] and info["targets"] == {"before": ["y"], "after": ["c"]}
    pv = lead.get(f"/api/datasets/{ds['id']}/export-preview?experiment_id={x['id']}").json()
    assert pv["targets"] == ["c"] and any("Targets differ" in w for w in pv["warnings"])


def test_an_experiments_leak_warnings_follow_its_targets(people):
    from .test_baselines import LEAKY

    lead = people["lead"]
    ds, rid = finalized_trees(lead, LEAKY)
    x = experiment(lead, ds, rid, "leaks")
    adjust(lead, ds, x, "y", action="keep")
    adjust(lead, ds, x, "c", action="target")
    base = lead.get(f"/api/datasets/{ds['id']}/export-preview").json()
    mine = lead.get(f"/api/datasets/{ds['id']}/export-preview?experiment_id={x['id']}").json()
    assert not any("cc" in w for w in base["warnings"]) and any("leakage: cc" in w for w in mine["warnings"])


def test_bulk_accept_in_a_fresh_experiment_changes_nothing(people):
    a = people["a"]
    ds, rid = finalized_trees(people["lead"])
    x = experiment(a, ds, rid, "bulk")
    r = a.post(f"/api/datasets/{ds['id']}/decisions/bulk?view=experiment:{x['id']}",
               json={"filters": [], "accept_suggestion": True, "dry_run": True})
    assert r.status_code == 200 and r.json()["changed"] == 0, r.json()


def test_a_note_alone_isnt_an_adjustment(people):
    a = people["a"]
    ds, rid = finalized_trees(people["lead"])
    x = experiment(a, ds, rid, "notes")
    adjust(a, ds, x, "w", notes="just a thought")
    assert a.get(f"/api/experiments/{x['id']}").json()["adjustments"]["count"] == 0


def test_views_of_private_or_unfinished_sources_are_refused(people):
    lead, a, b = people["lead"], people["a"], people["b"]
    ds = upload(lead)
    from .team_helpers import start
    st = start(lead, ds, "tree", [a, b])
    rid = st["recipe"]["id"]
    x = a.post(f"/api/datasets/{ds['id']}/experiments", json={
        "name": "private", "source": f"assessment:{rid}:{a.user['id']}"}).json()
    assert a.get(f"/api/datasets/{ds['id']}/fields?view=experiment:{x['id']}").status_code == 409
    assert b.get(f"/api/datasets/{ds['id']}/fields?view=experiment:{x['id']}").status_code == 404
    assert lead.get(f"/api/datasets/{ds['id']}/fields?view=baseline:{rid}").status_code == 404
