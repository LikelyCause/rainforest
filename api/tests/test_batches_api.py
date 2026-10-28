"""Batches over HTTP: one-vs-rest and search batches, the feeder that submits their runs one at a time, cancel, and
resuming after a restart. Needs RAINFOREST_TEST_DATABASE_URL."""

import time
import uuid
from contextlib import contextmanager

import numpy as np
import psycopg
import pytest
from fastapi.testclient import TestClient

from app import batches, db, experiments, jobs
from app.config import get_settings
from app.routers.datasets import best_by_dataset

from .team_helpers import DB_URL, app_, experiment, export, people, start, submit_and_finalize, upload, wait  # noqa: F401

DEAD = 515_152


def _csv(n=900, seed=5) -> str:
    """y: classes p, q, r and six rows of s (too few to train); v a number."""
    rng = np.random.default_rng(seed)
    x, z = rng.normal(size=n), rng.normal(size=n)
    s = x + 0.5 * rng.normal(size=n)
    y = np.where(s < -0.4, "p", np.where(s < 0.5, "q", "r"))
    y[rng.choice(n, 6, replace=False)] = "s"
    v = 3 * x + rng.normal(scale=0.3, size=n)
    return "x,z,y,v\n" + "".join(f"{a:.3f},{b:.3f},{c},{d:.3f}\n" for a, b, c, d in zip(x, z, y, v))


@pytest.fixture(scope="module")
def setup(people):
    lead = people["lead"]
    ds = upload(lead, _csv(), targets="y,v")
    start(lead, ds, "tree", [lead], targets=("y", "v"))
    rid = submit_and_finalize(lead, ds)["recipe"]["id"]
    x = experiment(lead, ds, rid, "batches")
    return {"lead": lead, "ds": ds, "rid": rid, "x": x, "e": export(lead, ds, x)}


def _post(client, e, expect=202, **body):
    r = client.post(
        f"/api/exports/{e['id']}/batches",
        json={"model": "lightgbm", "cv": 0, "cv_folds": 0, "params": {"n_estimators": 15}, **body},
    )
    assert r.status_code == expect, r.text
    return r.json()


def _finished(client, batch_id, timeout=240.0) -> dict:
    return wait(lambda: (b := client.get(f"/api/batches/{batch_id}").json())["status"] != "running" and b, timeout)


def _runs(batch_id) -> list[dict]:
    return db.fetch_all("SELECT * FROM meta.model_runs WHERE batch_id = %s ORDER BY id", (batch_id,))


@contextmanager
def _slot_held():
    """Another process holds the only job slot, so nothing trains inside the block."""
    other = psycopg.connect(DB_URL, autocommit=True)
    other.execute("SELECT pg_advisory_lock(%s, 0)", (jobs.SLOT_LOCK,))
    try:
        yield
    finally:
        other.close()


def test_a_one_vs_rest_batch_trains_one_run_per_class_one_at_a_time(setup):
    lead, e = setup["lead"], setup["e"]
    # Weighting settings at their defaults don't clash with balance.
    out = _post(
        lead,
        e,
        kind="one_vs_rest",
        target="y",
        balance=True,
        params={"n_estimators": 15, "is_unbalance": False},
    )
    batch = out["batch"]
    assert batch["kind"] == "one_vs_rest" and batch["status"] == "running"
    assert batch["name"] == "LightGBM · one model per class of y"
    assert len(out["run_ids"]) == 3 and out["redraws"] == 0
    assert out["skipped"] == [{"class": "s", "reason": "fewer than 10 positives in training"}]
    assert batch["counts"]["pending"] + batch["counts"]["running"] == 3
    # An export asked for after the batch queues between its runs rather than behind all of them.
    later = lead.post(f"/api/datasets/{setup['ds']['id']}/exports", json={"experiment_id": setup["x"]["id"]})
    assert later.status_code == 202, later.text

    def one_at_a_time():
        runs = _runs(batch["id"])
        active = [r for r in runs if r["status"] == "running" or (r["status"] == "pending" and r["owner"] is not None)]
        assert len(active) <= 1, [(r["id"], r["status"]) for r in active]
        b = lead.get(f"/api/batches/{batch['id']}").json()
        return b["status"] != "running" and b

    done = wait(one_at_a_time, 240)
    assert done["status"] == "done" and done["finished_at"] and done["counts"]["done"] == 3, done
    assert done["spec"]["classes"] == ["p", "q", "r"] and done["skipped"] == out["skipped"]
    rows = done["rows"]
    assert [r["label"] for r in rows] == [f"LightGBM · y = {c} vs rest" for c in "pqr"]
    for r in rows:
        assert r["status"] == "done" and r["positive_class"] in "pqr" and r["retried"] is False
        assert set(r["positives"]) == {"train", "valid", "test"} and r["positives"]["train"] > 10
        for split in ("valid", "test"):
            assert 0 <= r[split]["pr_auc"] <= 1 and 0 <= r[split]["roc_auc"] <= 1 and 0 <= r[split]["f1"] <= 1
        assert 0 <= r["threshold"] <= 1
        assert r["weak"] == (r["valid"]["pr_auc"] < 0.5 or r["positives"]["test"] < 30)
    run = lead.get(f"/api/runs/{rows[0]['run_id']}").json()
    assert run["batch_id"] == batch["id"] and run["config"]["balance"] is True

    later_export = lead.get(f"/api/exports/{later.json()['id']}").json()
    assert later_export["status"] == "done"
    last = max(r["finished_at"] for r in _runs(batch["id"]))
    assert (
        db.fetch_one("SELECT finished_at FROM meta.exports WHERE id = %s", (later_export["id"],))["finished_at"] < last
    )

    listed = lead.get(f"/api/experiments/{setup['x']['id']}/batches").json()
    assert listed[0]["id"] == batch["id"] and listed[0]["counts"]["done"] == 3


def test_unticked_classes_are_listed_as_not_selected(setup):
    with _slot_held():
        out = _post(setup["lead"], setup["e"], kind="one_vs_rest", target="y", classes=["q"])
        assert len(out["run_ids"]) == 1
        assert {s["class"]: s["reason"] for s in out["skipped"]} == {
            "p": "not selected",
            "r": "not selected",
            "s": "not selected",
        }
        setup["lead"].post(f"/api/batches/{out['batch']['id']}/cancel")


def test_a_random_search_runs_n_valid_combinations_and_ranks_them(setup):
    lead = setup["lead"]
    space = {"n_estimators": [10, 0, 20, 30], "learning_rate": {"low": 0.05, "high": 0.3, "log": True}}
    out = _post(lead, setup["e"], kind="search", space=space, sampler={"kind": "random", "n": 4, "seed": 3})
    assert len(out["run_ids"]) == 4 and out["redraws"] > 0, "n_estimators 0 is refused and redrawn"
    done = _finished(lead, out["batch"]["id"])
    assert done["status"] == "done" and done["counts"]["done"] == 4
    assert done["spec"]["ranking"] == "f1_macro" and done["spec"]["ranked_target"] == "y"
    assert all(r["settings"]["n_estimators"] in (10, 20, 30) for r in done["rows"])
    board = done["leaderboard"]
    assert [b["rank"] for b in board] == [1, 2, 3, 4]
    values = [b["value"] for b in board]
    assert values == sorted(values, reverse=True) and all(0 <= v <= 1 for v in values)
    assert all(len(b["run_ids"]) == 1 and b["classes"] == {} for b in board)


def test_a_one_vs_rest_search_trains_each_combination_per_class_and_ranks_by_the_mean(setup):
    lead = setup["lead"]
    out = _post(
        lead,
        setup["e"],
        kind="search",
        target="y",
        search_classes=["p", "r"],
        space={"n_estimators": [5, 25]},
        sampler={"kind": "grid"},
    )
    assert len(out["run_ids"]) == 4
    done = _finished(lead, out["batch"]["id"])
    assert done["status"] == "done" and done["spec"]["ranking"] == "pr_auc"
    assert [r["label"] for r in done["rows"]] == [f"LightGBM · y = {c} vs rest" for c in "prpr"]
    board = done["leaderboard"]
    assert len(board) == 2 and {b["settings"]["n_estimators"] for b in board} == {5, 25}
    for b in board:
        assert set(b["classes"]) == {"p", "r"}
        assert b["value"] == pytest.approx(np.mean(list(b["classes"].values())))
    assert board[0]["value"] >= board[1]["value"] and board[0]["rank"] == 1


def test_a_grid_with_a_refused_combination_starts_nothing(setup):
    before = db.fetch_one("SELECT count(*) AS n FROM meta.run_batches")["n"]
    r = setup["lead"].post(
        f"/api/exports/{setup['e']['id']}/batches",
        json={"kind": "search", "model": "lightgbm", "cv": 0, "cv_folds": 0, "space": {"n_estimators": [10, 0]}},
    )
    assert r.status_code == 422 and "n_estimators" in r.text, r.text
    assert db.fetch_one("SELECT count(*) AS n FROM meta.run_batches")["n"] == before
    bad = {
        "a number target": {"kind": "one_vs_rest", "target": "v"},
        "a refused setting": {"kind": "one_vs_rest", "target": "y", "params": {"n_estimators": 0}},
        "a planned model": {"kind": "one_vs_rest", "target": "y", "model": "nope"},
        "an unknown class": {"kind": "one_vs_rest", "target": "y", "classes": ["zzz"]},
        "a class too rare to tune on": {
            "kind": "search",
            "target": "y",
            "search_classes": ["s"],
            "space": {"n_estimators": [5]},
        },
        "a range in a grid": {"kind": "search", "space": {"learning_rate": {"low": 0.1, "high": 0.2}}},
        "a string for a list": {"kind": "search", "space": {"boosting_type": "gbdt"}},
        "not a setting": {"kind": "search", "space": {"bogus": [1, 2]}},
        "a metric of another kind": {"kind": "search", "space": {"n_estimators": [5]}, "ranking": "r2"},
        "balance without classes": {"kind": "search", "space": {"n_estimators": [5]}, "balance": True},
        "a space on one-vs-rest": {"kind": "one_vs_rest", "target": "y", "space": {"n_estimators": [5]}},
        "a random search without n": {"kind": "search", "space": {"n_estimators": [5]}, "sampler": {"kind": "random"}},
    }
    for why, body in bad.items():
        r = setup["lead"].post(f"/api/exports/{setup['e']['id']}/batches", json={"cv": 0, "cv_folds": 0, **body})
        assert r.status_code == 422, (why, r.status_code, r.text)
    assert db.fetch_one("SELECT count(*) AS n FROM meta.run_batches")["n"] == before


def test_too_many_runs_is_422_with_the_count(setup, monkeypatch):
    monkeypatch.setenv("BATCH_MAX_RUNS", "2")
    get_settings.cache_clear()
    try:
        r = setup["lead"].post(
            f"/api/exports/{setup['e']['id']}/batches", json={"kind": "one_vs_rest", "target": "y", "cv_folds": 0}
        )
        assert r.status_code == 422 and "3 runs" in r.json()["detail"] and "2" in r.json()["detail"], r.text
        grid = {"kind": "search", "space": {"n_estimators": list(range(1, 11)), "max_depth": list(range(2, 12))}}
        r = setup["lead"].post(f"/api/exports/{setup['e']['id']}/batches", json=grid)
        assert r.status_code == 422 and "100 runs" in r.json()["detail"], r.text
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


def test_cancel_cancels_the_waiting_runs_and_the_batch(setup):
    lead = setup["lead"]
    with _slot_held():
        out = _post(lead, setup["e"], kind="one_vs_rest", target="y")
        b = lead.post(f"/api/batches/{out['batch']['id']}/cancel")
        assert b.status_code == 200, b.text
        b = b.json()
        assert b["status"] == "cancelled" and b["finished_at"] and b["counts"]["cancelled"] == 3
        assert {r["status"] for r in b["rows"]} == {"cancelled"}
        again = lead.post(f"/api/batches/{out['batch']['id']}/cancel")
        assert again.status_code == 409 and "cancelled" in again.json()["detail"]
    # The run that had been submitted reaches the slot, finds itself cancelled, and doesn't train.
    time.sleep(2)
    assert {r["status"] for r in _runs(out["batch"]["id"])} == {"cancelled"}


def test_a_restart_queues_the_interrupted_run_again_once_and_the_batch_carries_on(setup):
    lead = setup["lead"]
    with _slot_held():
        out = _post(lead, setup["e"], kind="one_vs_rest", target="y")
        first, *rest = out["run_ids"]
        # The process training the first run dies; the API starts again.
        db.execute("UPDATE meta.model_runs SET status = 'running', owner = %s WHERE id = %s", (DEAD, first))
        jobs.sweep()
        runs = {r["id"]: r for r in _runs(out["batch"]["id"])}
        assert runs[first]["status"] == "pending" and runs[first]["report"]["retried"] is True
        assert runs[first]["owner"] == jobs.owner_id(), "queued again"
        assert all(runs[i]["status"] == "pending" and runs[i]["owner"] is None for i in rest), "still waiting"
        batches.feed(out["batch"]["id"])  # another caller (a second process starting) submits nothing more
        assert all(r["owner"] is None for r in _runs(out["batch"]["id"]) if r["id"] in rest)

        # Interrupted a second time, it stays failed and the batch moves on.
        db.execute("UPDATE meta.model_runs SET status = 'running', owner = %s WHERE id = %s", (DEAD, first))
        jobs.sweep()
        runs = {r["id"]: r for r in _runs(out["batch"]["id"])}
        assert runs[first]["status"] == "failed" and runs[first]["error"] == jobs.INTERRUPTED
        assert runs[rest[0]]["owner"] == jobs.owner_id() and runs[rest[1]]["owner"] is None
    done = _finished(lead, out["batch"]["id"])
    assert done["status"] == "done"
    rows = {r["run_id"]: r for r in done["rows"]}
    assert rows[first]["status"] == "failed" and rows[first]["retried"] is True
    assert all(rows[i]["status"] == "done" and not rows[i]["retried"] for i in rest)


def test_a_retried_run_that_finishes_keeps_its_retried_mark(setup):
    lead = setup["lead"]
    with _slot_held():
        out = _post(lead, setup["e"], kind="one_vs_rest", target="y", classes=["q"])
        (only,) = out["run_ids"]
        db.execute("UPDATE meta.model_runs SET status = 'running', owner = %s WHERE id = %s", (DEAD, only))
        jobs.sweep()
    done = _finished(lead, out["batch"]["id"])
    assert done["status"] == "done" and done["rows"][0]["status"] == "done" and done["rows"][0]["retried"] is True


def test_starting_a_batch_is_409_while_refreshing(setup):
    ds = setup["ds"]["id"]
    db.execute("UPDATE meta.datasets SET status = 'refreshing' WHERE id = %s", (ds,))
    try:
        r = setup["lead"].post(
            f"/api/exports/{setup['e']['id']}/batches", json={"kind": "one_vs_rest", "target": "y", "cv_folds": 0}
        )
        assert r.status_code == 409 and "refresh" in r.json()["detail"]
    finally:
        db.execute("UPDATE meta.datasets SET status = 'ready' WHERE id = %s", (ds,))


def test_a_viewer_reads_batches_but_cannot_start_or_cancel_one(setup, app_, login_as):  # noqa: F811
    viewer = login_as(TestClient(app_), "viewer", name=f"Viewer {uuid.uuid4().hex[:6]}")
    r = viewer.post(f"/api/exports/{setup['e']['id']}/batches", json={"kind": "one_vs_rest", "target": "y"})
    assert r.status_code == 403
    listed = viewer.get(f"/api/experiments/{setup['x']['id']}/batches")
    assert listed.status_code == 200 and listed.json()
    batch_id = listed.json()[0]["id"]
    assert viewer.get(f"/api/batches/{batch_id}").status_code == 200
    assert viewer.post(f"/api/batches/{batch_id}/cancel").status_code == 403
    assert setup["lead"].get("/api/batches/999999999").status_code == 404


def test_one_vs_rest_runs_are_never_an_experiments_or_datasets_best(setup):
    lead, ds, e = setup["lead"], setup["ds"], setup["e"]
    x = experiment(lead, ds, setup["rid"], "best")

    def insert(config, report, valid):
        return db.fetch_one(
            """INSERT INTO meta.model_runs (export_id, dataset_id, model, family, status, config, report, metrics,
                                            created_by, experiment_id)
               VALUES (%s, %s, 'lightgbm', 'tree', 'done', %s, %s, %s, %s, %s) RETURNING id""",
            (
                e["id"],
                ds["id"],
                db.jsonb(config),
                db.jsonb(report),
                db.jsonb({"valid": valid}),
                lead.user["id"],
                x["id"],
            ),
        )["id"]

    plain = insert({}, {"order": ["y"]}, {"y": {"roc_auc_ovr_macro": 0.6}})
    ovr = insert(
        {"positive_class": "p", "target": "y"},
        {"order": ["y"], "chain_kind": "one_vs_rest"},
        {"y": {"roc_auc": 0.99}},
    )
    with db.connection() as conn:
        best = experiments.best_by_experiment(conn, [x["id"]])[x["id"]]
    assert best["run_id"] == plain and best["metric"] == "roc_auc_ovr_macro"
    by_ds = best_by_dataset([ds["id"]], experiments.visible_ids(lead.user["id"]))[ds["id"]]
    assert by_ds["run_id"] != ovr and by_ds["metric"] != "roc_auc"
