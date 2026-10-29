"""Experiments against a real Postgres (RAINFOREST_TEST_DATABASE_URL): who sees a personal experiment in each phase
(API and SQL console), what a personal export contains and that it rebuilds the same way, the default Team
experiment, role checks, the Imported migration, and MLflow logging deferred until review opens."""

import json
from pathlib import Path

import pytest

from .test_team import _assign, _fields, _review_all, _save, _upload, _wait, app_, people  # noqa: F401


def _xp(client, ds, name, source=None):
    if source is None:
        recipe = client.get(f"/api/datasets/{ds['id']}").json()["recipe_id"]
        source = f"assessment:{recipe}:{client.user['id']}"
    r = client.post(f"/api/datasets/{ds['id']}/experiments", json={"name": name, "source": source})
    assert r.status_code == 201, r.text
    return r.json()


def _export(client, ds, xid=None, **body):
    r = client.post(f"/api/datasets/{ds['id']}/exports", json={**body, "experiment_id": xid})
    assert r.status_code == 202, r.text
    exp = _wait(
        lambda: (x := client.get(f"/api/exports/{r.json()['id']}").json())["status"] in ("done", "failed") and x
    )
    assert exp["status"] == "done", exp["error"]
    return exp, json.loads((Path(exp["output_dir"]) / "manifest.json").read_text())


def _train(client, exp):
    r = client.post(
        f"/api/exports/{exp['id']}/runs", json={"model": "lightgbm", "cv": 0, "params": {"n_estimators": 10}}
    )
    assert r.status_code == 202, r.text
    run = _wait(lambda: (x := client.get(f"/api/runs/{r.json()['id']}").json())["status"] in ("done", "failed") and x)
    assert run["status"] == "done", run["error"]
    return run


def _kept(manifest) -> set[str]:
    return {f["name"] for f in manifest["fields"]}


def _sees(client, ds, xid, exp_id, run_id) -> dict[str, bool]:
    """Whether `client` can see the experiment, its export and its run, on every read path."""
    d = ds["id"]
    listed = next(x for x in client.get("/api/datasets").json() if x["id"] == d)
    sql = client.post("/api/sql", json={"sql": f"SELECT id FROM meta.experiments WHERE id = {xid}"}).json()
    sql_exp = client.post("/api/sql", json={"sql": f"SELECT id FROM meta.exports WHERE id = {exp_id}"}).json()
    sql_run = client.post(
        "/api/sql", json={"sql": f"SELECT run_id FROM meta.v_model_metrics WHERE run_id = {run_id}"}
    ).json()
    return {
        "list": xid in [x["id"] for x in client.get(f"/api/datasets/{d}/experiments").json()],
        "detail": client.get(f"/api/experiments/{xid}").status_code == 200,
        "exports": exp_id in [e["id"] for e in client.get(f"/api/datasets/{d}/exports").json()],
        "export": client.get(f"/api/exports/{exp_id}").status_code == 200,
        "manifest": client.get(f"/api/exports/{exp_id}/files/manifest").status_code == 200,
        "script": client.get(f"/api/exports/{exp_id}/files/code").status_code == 200,
        "models": client.get(f"/api/exports/{exp_id}/models").status_code == 200,
        "runs": run_id in [r["id"] for r in client.get(f"/api/datasets/{d}/runs").json()],
        "run": client.get(f"/api/runs/{run_id}").status_code == 200,
        "run file": client.get(f"/api/runs/{run_id}/files/report").status_code == 200,
        "counts": listed["export_count"] > 0 and listed["run_count"] > 0,
        "sql experiments": bool(sql["rows"]),
        "sql exports": bool(sql_exp["rows"]),
        "sql metrics": bool(sql_run["rows"]),
    }


def test_a_personal_experiment_is_private_while_assessing_then_shared(people):
    lead, a, b, c = people["lead"], people["a"], people["b"], people["c"]
    ds = _upload(lead)
    try:
        _assign(lead, ds, a, b)
        x = _xp(a, ds, "A's trees")
        assert x["private"] and x["source"] == f"assessment:{ds['recipe_id']}:{a.user['id']}"
        exp, _ = _export(a, ds, x["id"])
        assert exp["experiment_id"] == x["id"] and exp["config"]["source"] == "assessment"
        run = _train(a, exp)
        assert run["experiment_id"] == x["id"]

        def matrix():
            return {
                who: _sees(cl, ds, x["id"], exp["id"], run["id"])
                for who, cl in (("owner", a), ("member", c), ("lead", lead))
            }

        seen = matrix()
        # The console has no notion of who asks, so it shows a private experiment to nobody, its owner included.
        assert {k for k, v in seen["owner"].items() if not v} == {"sql experiments", "sql exports", "sql metrics"}
        assert not any(seen["member"].values()), seen["member"]
        assert not any(seen["lead"].values()), seen["lead"]
        assert all(v is False for k, v in _sees(b, ds, x["id"], exp["id"], run["id"]).items()), "nor a co-assessor"

        _review_all(a, ds)
        a.post(f"/api/datasets/{ds['id']}/assessment/submit")
        assert not any(matrix()["member"].values()), "one submission doesn't open review"
        _review_all(b, ds)
        assert b.post(f"/api/datasets/{ds['id']}/assessment/submit").json()["recipe"]["phase"] == "review"
        seen = matrix()
        for who in ("owner", "member", "lead"):
            assert all(seen[who].values()), (who, seen[who])
        assert c.get(f"/api/experiments/{x['id']}").json()["private"] is False

        assert lead.post(f"/api/datasets/{ds['id']}/finalize", json={"accept_open": True}).status_code == 200
        seen = matrix()
        for who in ("owner", "member", "lead"):
            assert all(seen[who].values()), (who, seen[who])
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")


def test_a_personal_export_is_the_owners_assessment_and_rebuilds_the_same(people):
    from app.db import fetch_one
    from app.routers import exports

    lead, a, b = people["lead"], people["a"], people["b"]
    ds = _upload(lead)
    try:
        _assign(lead, ds, a, b)
        fa, fb = _fields(a, ds), _fields(b, ds)
        assert _save(a, fa["w"]["id"], action="drop").status_code == 200
        assert _save(a, fa["z"]["id"], imputation="constant", impute_value="7").status_code == 200
        assert _save(b, fb["x"]["id"], action="drop").status_code == 200
        x = _xp(a, ds, "A")
        pv = a.get(f"/api/datasets/{ds['id']}/export-preview?experiment_id={x['id']}").json()
        assert pv["source"] == "assessment" and not pv["blocking"], pv
        exp, m1 = _export(a, ds, x["id"])
        assert _kept(m1) == {"x", "z", "c"}, "A's drop of w counts, B's drop of x doesn't"
        z = next(f for f in m1["fields"] if f["name"] == "z")
        assert z["imputation"] == "constant" and str(z["impute_value"]) == "7"
        assert m1["config"]["assessor_id"] == a.user["id"] and m1["config"]["experiment_id"] == x["id"]

        # Later edits don't change what the export was built from, and rebuilding it gives the same fields.
        assert _save(a, fa["x"]["id"], action="drop").status_code == 200
        frozen = fetch_one("SELECT snapshot FROM meta.exports WHERE id = %s", (exp["id"],))["snapshot"]
        assert frozen["fields"]["x"]["decision"]["action"] == "keep"
        exports._run(exp["id"])
        rebuilt = json.loads((Path(exp["output_dir"]) / "manifest.json").read_text())
        assert rebuilt["fields"] == m1["fields"]
        pv = a.get(f"/api/datasets/{ds['id']}/export-preview?experiment_id={x['id']}").json()
        assert pv["last_export"]["id"] == exp["id"] and pv["changed_since_export"] == 1
        _, m2 = _export(a, ds, x["id"])
        assert _kept(m2) == {"z", "c"}
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")


def test_carried_fields_of_a_reopened_version_take_the_team_values(people):
    from app.db import execute

    lead, a = people["lead"], people["a"]
    ds = _upload(lead)
    try:
        _assign(lead, ds, a)
        _review_all(a, ds)
        a.post(f"/api/datasets/{ds['id']}/assessment/submit")
        f = _fields(lead, ds)
        assert _save(lead, f["x"]["id"], action="drop").status_code == 200
        lead.post(f"/api/datasets/{ds['id']}/finalize", json={"accept_open": True})
        v2 = lead.post(f"/api/datasets/{ds['id']}/recipes", json={"reopen": [f["z"]["id"]]}).json()["recipe"]
        _assign(lead, ds, a)
        assert _save(a, f["z"]["id"], action="drop").status_code == 200
        # Even if an assessment row of a locked field differed, the carried team value is what exports.
        execute(
            "UPDATE meta.assessments SET action = 'keep' WHERE recipe_id = %s AND user_id = %s AND field_id = %s",
            (v2["id"], a.user["id"], f["x"]["id"]),
        )
        x = _xp(a, ds, "A on v2")
        assert x["recipe_version"] == 2
        _, m = _export(a, ds, x["id"])
        assert _kept(m) == {"w", "c"}, "x stays dropped (carried), z is dropped by A"
        assert m["config"]["recipe_version"] == 2
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")


def test_team_exports_default_to_the_versions_team_experiment(people):
    lead, a, c = people["lead"], people["a"], people["c"]
    ds = _upload(lead)
    try:
        _assign(lead, ds, a)
        _review_all(a, ds)
        a.post(f"/api/datasets/{ds['id']}/assessment/submit")
        recipe = lead.post(f"/api/datasets/{ds['id']}/finalize", json={"accept_open": True}).json()["recipe"]
        e1, _ = _export(c, ds)
        e2, _ = _export(a, ds)
        assert e1["experiment_id"] == e2["experiment_id"] and e1["experiment_name"] == "Team · Trees v1"
        xs = c.get(f"/api/datasets/{ds['id']}/experiments").json()
        team = next(x for x in xs if x["id"] == e1["experiment_id"])
        assert team["source"] == f"recipe:{recipe['id']}" and team["owner"] is None and not team["private"]
        assert team["export_count"] == 2 and team["run_count"] == 0 and team["best"] is None
        run = _train(c, e1)
        team = next(x for x in c.get(f"/api/datasets/{ds['id']}/experiments").json() if x["id"] == team["id"])
        assert team["run_count"] == 1 and team["best"]["run_id"] == run["id"] and team["best"]["metric"] == "roc_auc"
        detail = a.get(f"/api/experiments/{team['id']}").json()
        assert [e["id"] for e in detail["exports"]] == [e2["id"], e1["id"]] and detail["runs"][0]["id"] == run["id"]

        mine = _xp(c, ds, "C's own", f"recipe:{recipe['id']}")
        assert mine["owner"] == c.user["id"] and mine["owner_name"] == c.user["name"]
        e3, _ = _export(a, ds, mine["id"])
        assert e3["experiment_id"] == mine["id"], "anyone may add to a team experiment"
        r = lead.post(f"/api/datasets/{ds['id']}/exports", json={"experiment_id": mine["id"], "recipe_id": 999999})
        assert r.status_code == 422
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")


def test_role_checks(people):
    lead, a, b, c = people["lead"], people["a"], people["b"], people["c"]
    ds = _upload(lead)
    try:
        rid = ds["recipe_id"]
        _assign(lead, ds, a, b)
        post = lambda cl, body: cl.post(f"/api/datasets/{ds['id']}/experiments", json=body)  # noqa: E731
        assert post(c, {"name": "x", "source": f"assessment:{rid}:{c.user['id']}"}).status_code == 403, "not assessor"
        assert post(c, {"name": "x", "source": f"assessment:{rid}:{a.user['id']}"}).status_code == 403, "not mine"
        assert post(lead, {"name": "x", "source": f"recipe:{rid}"}).status_code == 409, "not finalized"
        assert post(a, {"name": "x", "source": "assessment:1"}).status_code == 422
        assert post(a, {"name": " ", "source": f"assessment:{rid}:{a.user['id']}"}).status_code == 422
        assert lead.post(f"/api/datasets/{ds['id']}/exports", json={"source": "assessment"}).status_code == 422
        x = _xp(a, ds, "A")
        assert (
            a.post(f"/api/datasets/{ds['id']}/exports", json={"experiment_id": x["id"], "source": "recipe"}).status_code
            == 422
        )
        exp, _ = _export(a, ds, x["id"])

        for who in (a, b):
            _review_all(who, ds)
            who.post(f"/api/datasets/{ds['id']}/assessment/submit")
        # Visible to all from review on, but only A adds to it, and only A or a lead changes it.
        r = b.post(f"/api/datasets/{ds['id']}/exports", json={"experiment_id": x["id"]})
        assert r.status_code == 403
        assert b.post(f"/api/exports/{exp['id']}/runs", json={"cv": 0}).status_code == 403
        assert c.patch(f"/api/experiments/{x['id']}", json={"name": "C's now"}).status_code == 403
        r = a.patch(f"/api/experiments/{x['id']}", json={"name": "A renamed", "notes": "tried trees"})
        assert r.status_code == 200 and (r.json()["name"], r.json()["notes"]) == ("A renamed", "tried trees")
        r = lead.patch(f"/api/experiments/{x['id']}", json={"archived": True})
        assert r.status_code == 200 and r.json()["archived_at"]
        assert a.post(f"/api/datasets/{ds['id']}/exports", json={"experiment_id": x["id"]}).status_code == 409
        assert a.post(f"/api/exports/{exp['id']}/runs", json={"cv": 0}).status_code == 409
        assert a.post(f"/api/exports/{exp['id']}/runs", json={"cv": 0, "experiment_id": x["id"] + 1}).status_code == 422
        listed = a.get(f"/api/datasets/{ds['id']}/experiments").json()
        assert listed[-1]["id"] == x["id"], "archived experiments sort last"
        assert a.patch(f"/api/experiments/{x['id']}", json={"archived": False}).json()["archived_at"] is None
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")


def test_existing_exports_and_runs_move_into_one_imported_experiment(people):
    from app import recipes
    from app.db import MIGRATIONS_DIR, connection, fetch_all, fetch_one

    lead, a = people["lead"], people["a"]
    ds = _upload(lead)
    other = _upload(lead)
    try:
        with connection() as conn:
            rec = recipes.finalize(conn, recipes.current(conn, ds["id"]), lead.user["id"])
            # As before experiments: two exports and a run with no experiment.
            ids = [
                conn.execute(
                    """INSERT INTO meta.exports (dataset_id, status, model_family, config, recipe_id)
                                   VALUES (%s, 'done', 'tree', '{}', %s) RETURNING id""",
                    (ds["id"], rec["id"]),
                ).fetchone()["id"]
                for _ in range(2)
            ]
            run = conn.execute(
                """INSERT INTO meta.model_runs (export_id, dataset_id, model, family, status)
                                  VALUES (%s, %s, 'lightgbm', 'tree', 'done') RETURNING id""",
                (ids[0], ds["id"]),
            ).fetchone()["id"]
            for _ in range(2):  # idempotent
                conn.execute((MIGRATIONS_DIR / "010_experiments.sql").read_text())
        made = fetch_all("SELECT * FROM meta.experiments WHERE dataset_id = %s", (ds["id"],))
        assert len(made) == 1
        imp = made[0]
        assert (imp["name"], imp["source"], imp["owner"], imp["assessor_id"]) == (
            "Imported",
            f"recipe:{rec['id']}",
            None,
            None,
        )
        assert {
            r["experiment_id"] for r in fetch_all("SELECT experiment_id FROM meta.exports WHERE id = ANY(%s)", (ids,))
        } == {imp["id"]}
        assert (
            fetch_one("SELECT experiment_id FROM meta.model_runs WHERE id = %s", (run,))["experiment_id"] == imp["id"]
        )
        assert fetch_all("SELECT * FROM meta.experiments WHERE dataset_id = %s", (other["id"],)) == []
        xs = a.get(f"/api/datasets/{ds['id']}/experiments").json()
        assert [(x["name"], x["export_count"], x["run_count"]) for x in xs] == [("Imported", 2, 1)]
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")
        lead.delete(f"/api/datasets/{other['id']}")


@pytest.fixture
def mlflow_store(tmp_path, monkeypatch):
    from mlflow import MlflowClient

    from app.config import get_settings

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    get_settings.cache_clear()
    yield MlflowClient(tracking_uri=uri)
    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    get_settings.cache_clear()


def test_personal_runs_reach_mlflow_only_when_review_opens(people, mlflow_store):
    lead, a, b = people["lead"], people["a"], people["b"]
    ds = _upload(lead)
    try:
        _assign(lead, ds, a, b)
        x = _xp(a, ds, "A's MLflow")
        exp, _ = _export(a, ds, x["id"])
        run = _train(a, exp)
        assert run["report"]["mlflow_pending"] is True and run["mlflow_url"] is None
        assert mlflow_store.search_experiments(filter_string=f"tags.`rainforest.dataset_id` = '{ds['id']}'") == [], (
            "nothing in MLflow while private"
        )
        for who in (a, b):
            _review_all(who, ds)
            who.post(f"/api/datasets/{ds['id']}/assessment/submit")
        run = _wait(lambda: (r := lead.get(f"/api/runs/{run['id']}").json())["mlflow_run_id"] and r, timeout=60)
        assert "mlflow_pending" not in run["report"] and run["mlflow_url"]
        tags = mlflow_store.get_run(run["mlflow_run_id"]).data.tags
        assert tags["rainforest.experiment_id"] == str(x["id"]) and tags["rainforest.experiment_name"] == "A's MLflow"
        assert tags["rainforest.user"].startswith(a.user["name"])
        # A run in the now-visible experiment logs at once.
        later = _train(a, exp)
        assert later["mlflow_run_id"] and "mlflow_pending" not in later["report"]
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")
