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
    assert fields(lead, ds)["w"]["action"] == "keep"  # the baseline didn't move
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
