"""Testing a deployment with an uploaded file: the job that scores it in chunks, the metrics a labelled file gets, the
results and their download, warnings, retention, and what refuses a test. Needs RAINFOREST_TEST_DATABASE_URL."""

import io
import shutil
import uuid
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import average_precision_score

from app import db, deployments, jobs, scoring
from app.config import get_settings

from .team_helpers import app_, experiment, export, people, start, submit_and_finalize, upload, wait  # noqa: F401


def _csv(n=900, seed=5) -> str:
    """code a unique key; y: classes p, q, r and six rows of s (too few to train); v a number."""
    rng = np.random.default_rng(seed)
    x, z = rng.normal(size=n), rng.normal(size=n)
    s = x + 0.5 * rng.normal(size=n)
    y = np.where(s < -0.4, "p", np.where(s < 0.5, "q", "r"))
    y[rng.choice(n, 6, replace=False)] = "s"
    v = x + z + rng.normal(size=n)
    rows = (f"K{i:05d},{a:.3f},{b:.3f},{c},{d:.3f}\n" for i, (a, b, c, d) in enumerate(zip(x, z, y, v)))
    return "code,x,z,y,v\n" + "".join(rows)


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
    x = experiment(lead, ds, rid, "deployment tests")
    e = export(lead, ds, x)
    r = lead.post(f"/api/exports/{e['id']}/batches", json={**BASE, "kind": "one_vs_rest", "target": "y"})
    assert r.status_code == 202, r.text
    bid = r.json()["batch"]["id"]
    batch = wait(lambda: (b := lead.get(f"/api/batches/{bid}").json())["status"] != "running" and b, 240)
    assert batch["status"] == "done", batch
    ovr = {row["positive_class"]: row["run_id"] for row in batch["rows"]}
    assert set(ovr) == {"p", "q", "r"}, ovr  # s has too few rows for a model
    # The team decides `code` is the record's key, so results show it instead of the row number.
    db.execute(
        """UPDATE meta.field_decisions d SET semantic_type = 'identifier' FROM meta.fields f
            WHERE f.id = d.field_id AND f.dataset_id = %s AND f.name = 'code'""",
        (ds["id"],),
    )
    path = db.fetch_one("SELECT parquet_path FROM meta.datasets WHERE id = %s", (ds["id"],))["parquet_path"]
    return {
        "lead": lead,
        "ds": ds,
        "x": x,
        "e": e,
        "batch": batch,
        "ovr": ovr,
        "multi": _run(lead, e, order=["y"])["id"],
        "number": _run(lead, e, order=["v"])["id"],
        "rows": pl.read_parquet(path, columns=["code", "x", "z", "y", "v"]),
    }


def _deploy(client, x, **body) -> dict:
    r = client.post(f"/api/experiments/{x['id']}/deployments", json={"name": f"d {uuid.uuid4().hex[:6]}", **body})
    assert r.status_code == 201, r.text
    return r.json()


@pytest.fixture(scope="module")
def ovr_dep(setup):
    return _deploy(setup["lead"], setup["x"], batch_id=setup["batch"]["id"])


def _post(client, dep_id, name, content, expect=202):
    r = client.post(f"/api/deployments/{dep_id}/tests", files={"file": (name, content, "application/octet-stream")})
    assert r.status_code == expect, r.text
    return r.json()


def _finished(client, test_id) -> dict:
    return wait(
        lambda: (t := client.get(f"/api/deployment-tests/{test_id}").json())["status"] in ("done", "failed") and t, 120
    )


def _test(client, dep_id, name, content) -> dict:
    t = _post(client, dep_id, name, content)
    assert t["status"] == "pending" and t["filename"] == name, t
    return _finished(client, t["id"])


def _csv_of(df: pl.DataFrame) -> bytes:
    return df.write_csv().encode()


def _predicted(client, dep_id, df: pl.DataFrame) -> list[dict]:
    r = client.post(f"/api/deployments/{dep_id}/predict", json={"records": df.select("x", "z").to_dicts()})
    assert r.status_code == 200, r.text
    return r.json()["results"]


def _labelled_file(setup) -> pl.DataFrame:
    """300 of the dataset's rows, with five more whose label has no model and four whose label is missing."""
    rows = setup["rows"].head(300).select("code", "x", "z", "y")
    extra = (
        setup["rows"]
        .slice(300, 9)
        .select("code", "x", "z")
        .with_columns(y=pl.Series(["s"] * 5 + [None] * 4, dtype=pl.String))
    )
    return pl.concat([rows, extra])


# ------------------------------------------------------------------------------------------- labelled files
def test_a_labelled_file_reports_accuracy_per_class_metrics_and_the_nothing_matched_share(setup, ovr_dep):
    lead, dep = setup["lead"], ovr_dep
    df = _labelled_file(setup)
    t = _test(lead, dep["id"], "labelled.csv", _csv_of(df))
    assert t["status"] == "done", t["error"]
    assert t["rows"] == df.height and t["has_labels"] is True and t["progress"] == 1
    m = t["metrics"]
    want = _predicted(lead, dep["id"], df)
    labels = df["y"].to_list()
    labelled = [i for i, y in enumerate(labels) if y is not None]
    assert m["labelled_rows"] == len(labelled) and m["rows_without_label"] == 4
    hits = sum(want[i]["top"]["label"] == labels[i] for i in labelled)
    assert m["top1_accuracy"] == pytest.approx(hits / len(labelled))
    assert m["rows_label_without_model"] == sum(labels[i] == "s" for i in labelled) >= 5
    # Over all rows, the unlabelled ones too.
    assert m["nothing_matched_share"] == pytest.approx(np.mean([r["none_matched"] for r in want]))
    thresholds = {mm["label"]: mm["threshold"] for mm in dep["models"]}
    by_class = {c["label"]: c for c in m["classes"]}
    assert set(by_class) == {"p", "q", "r"}  # no row for s, which has no model
    for c, got in by_class.items():
        y = np.array([labels[i] == c for i in labelled])
        score = np.array([want[i]["scores"][c] for i in labelled])
        pred = score >= thresholds[c]
        tp = int((y & pred).sum())
        assert got["support"] == int(y.sum()) and got["threshold"] == thresholds[c]
        assert got["precision"] == pytest.approx(tp / pred.sum())
        assert got["recall"] == pytest.approx(tp / y.sum())
        p, r = tp / pred.sum(), tp / y.sum()
        assert got["f1"] == pytest.approx(2 * p * r / (p + r))
        assert got["pr_auc"] == pytest.approx(average_precision_score(y, score))
    conf = m["confusion"]
    assert set(conf["labels"]) == {"p", "q", "r", "s"} and conf["labels"][0] == max(
        "pqrs", key=lambda c: labels.count(c)
    )
    assert sum(map(sum, conf["counts"])) == len(labelled)
    i, j = conf["labels"].index("s"), conf["labels"].index(want[300]["top"]["label"])
    assert conf["counts"][i][j] >= 1


def test_thresholds_of_zero_match_everything_so_nothing_matched_is_zero(setup):
    lead = setup["lead"]
    dep = _deploy(lead, setup["x"], batch_id=setup["batch"]["id"], threshold_rule={"kind": "fixed", "value": 0})
    df = _labelled_file(setup)
    m = _test(lead, dep["id"], "zero.csv", _csv_of(df))["metrics"]
    assert m["nothing_matched_share"] == 0
    labelled = df.filter(pl.col("y").is_not_null())
    for c in m["classes"]:
        support = int((labelled["y"] == c["label"]).sum())
        assert c["recall"] == 1 and c["precision"] == pytest.approx(support / labelled.height)


def test_a_multiclass_deployment_judges_each_class_by_top(setup):
    lead = setup["lead"]
    dep = _deploy(lead, setup["x"], run_ids=[setup["multi"]])
    df = _labelled_file(setup)
    m = _test(lead, dep["id"], "multi.csv", _csv_of(df))["metrics"]
    want = _predicted(lead, dep["id"], df)
    labels = df["y"].to_list()
    labelled = [i for i, y in enumerate(labels) if y is not None]
    assert m["nothing_matched_share"] is None
    by_class = {c["label"]: c for c in m["classes"]}
    for c, got in by_class.items():
        y = np.array([labels[i] == c for i in labelled])
        pred = np.array([want[i]["top"]["label"] == c for i in labelled])
        assert got["threshold"] is None and got["support"] == int(y.sum())
        if pred.any():
            assert got["precision"] == pytest.approx((y & pred).sum() / pred.sum())


def test_a_number_deployment_reports_errors_against_the_target(setup):
    lead = setup["lead"]
    dep = _deploy(lead, setup["x"], run_ids=[setup["number"]])
    df = setup["rows"].head(200).select("x", "z", "v")
    m = _test(lead, dep["id"], "number.csv", _csv_of(df))["metrics"]
    got = np.array([r["scores"]["v"] for r in _predicted(lead, dep["id"], df)])
    err = got - df["v"].to_numpy()
    assert m["labelled_rows"] == 200
    assert m["mae"] == pytest.approx(np.abs(err).mean(), rel=1e-6)
    assert m["rmse"] == pytest.approx(np.sqrt((err**2).mean()), rel=1e-6)


# ------------------------------------------------------------------------------------------- results
def test_an_unlabelled_file_gives_results_only_and_shows_the_first_500(setup, ovr_dep):
    lead, dep = setup["lead"], ovr_dep
    df = setup["rows"].head(600).select("code", "x", "z")
    t = _test(lead, dep["id"], "plain.csv", _csv_of(df))
    assert t["status"] == "done" and t["has_labels"] is False and t["metrics"] is None
    assert t["rows"] == 600 and t["id_field"] == "code" and t["warnings"] == {}
    shown = t["results"]
    assert len(shown) == 500 and t["labels"] == ["p", "q", "r"]
    want = _predicted(lead, dep["id"], df)
    for k in (0, 1, 499):
        assert shown[k]["row"] == k + 1 and shown[k]["id"] == df["code"][k] and shown[k]["label"] is None
        assert shown[k]["scores"] == pytest.approx(want[k]["scores"])
        assert shown[k]["top"]["label"] == want[k]["top"]["label"] and shown[k]["matches"] == want[k]["matches"]
        assert shown[k]["none_matched"] == want[k]["none_matched"]

    r = lead.get(f"/api/deployment-tests/{t['id']}/download?format=parquet")
    assert r.status_code == 200, r.text
    full = pl.read_parquet(io.BytesIO(r.content))
    assert full.height == 600 and full["code"].to_list() == df["code"].to_list()
    assert full["score_p"].to_list() == pytest.approx([w["scores"]["p"] for w in want])
    assert full["matches"].to_list() == [w["matches"] for w in want]
    r = lead.get(f"/api/deployment-tests/{t['id']}/download?format=csv")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    text = pl.read_csv(io.BytesIO(r.content))
    assert text.height == 600 and text.columns == full.columns
    assert text["top"].to_list() == full["top"].to_list()
    assert lead.get(f"/api/deployment-tests/{t['id']}/download?format=xlsx").status_code == 422

    listed = lead.get(f"/api/deployments/{dep['id']}/tests").json()
    assert listed[0]["id"] == t["id"] and "results" not in listed[0]


def test_without_an_identifier_column_the_row_number_identifies_a_record(setup, ovr_dep):
    df = setup["rows"].head(5).select("x", "z")
    t = _test(setup["lead"], ovr_dep["id"], "noid.csv", _csv_of(df))
    assert t["id_field"] is None and [r["id"] for r in t["results"]] == [None] * 5
    assert [r["row"] for r in t["results"]] == [1, 2, 3, 4, 5]


def test_scoring_in_chunks_gives_what_scoring_at_once_does(setup, ovr_dep, monkeypatch):
    lead, dep = setup["lead"], ovr_dep
    content = _csv_of(_labelled_file(setup))
    whole = _test(lead, dep["id"], "whole.csv", content)
    monkeypatch.setattr(deployments, "TEST_CHUNK", 64)
    parts = _test(lead, dep["id"], "parts.csv", content)
    assert parts["metrics"] == whole["metrics"]

    def frame(t):
        return pl.read_parquet(io.BytesIO(lead.get(f"/api/deployment-tests/{t['id']}/download?format=parquet").content))

    assert frame(parts).equals(frame(whole))


def test_parquet_and_tab_separated_files_go_through_the_upload_reader(setup, ovr_dep):
    lead, dep = setup["lead"], ovr_dep
    df = setup["rows"].head(50).select("code", "x", "z", "y")
    base = _test(lead, dep["id"], "t.csv", _csv_of(df))
    buf = io.BytesIO()
    df.write_parquet(buf)
    pq = _test(lead, dep["id"], "t.parquet", buf.getvalue())
    tsv = _test(lead, dep["id"], "t.tsv", df.write_csv(separator="\t").encode())
    for t in (pq, tsv):
        assert t["status"] == "done", t["error"]
        assert t["metrics"] == base["metrics"] and t["results"] == base["results"]
    assert any("tabs" in n for n in tsv["notes"]), tsv["notes"]


def test_values_that_dont_convert_are_counted_per_field_and_a_mostly_failing_field_is_flagged(setup, ovr_dep):
    df = setup["rows"].head(40).select("x", "z")
    df = df.with_columns(
        x=pl.Series([("abc" if i < 4 else str(v)) for i, v in enumerate(df["x"])]),
        z=pl.Series([("??z" if i < 30 else str(v)) for i, v in enumerate(df["z"])]),
    )
    t = _test(setup["lead"], ovr_dep["id"], "bad.csv", _csv_of(df))
    assert t["status"] == "done", t["error"]
    assert t["warnings"] == {
        "x": {"failed": 4, "values": 40, "flagged": False},
        "z": {"failed": 30, "values": 40, "flagged": True},
    }


def test_a_label_reads_as_the_class_text_training_saw(setup):
    codes = scoring.Input("y", pl.Int64, None, None, integer_codes=True)
    got = deployments._label_values(pl.Series("y", ["7.0", "07", " 7", "7.5", "NA", None]), codes)
    assert got.to_list() == ["7", "7", "7", None, None, None]
    flags = scoring.Input("y", pl.Boolean, None, None)
    assert deployments._label_values(pl.Series("y", ["true", "0"]), flags).to_list() == ["True", "False"]
    assert deployments._label_values(pl.Series("y", ["p", ""]), None).to_list() == ["p", None]
    with db.connection() as conn:  # the export froze the target's type, so a file's labels are read by it
        frozen = deployments._label_input(conn, setup["e"]["id"], "y", "binary")
    assert frozen is not None and frozen.dtype == pl.String


# ------------------------------------------------------------------------------------------- failing and refusing
def test_a_file_missing_a_field_fails_naming_it(setup, ovr_dep):
    t = _test(setup["lead"], ovr_dep["id"], "nox.csv", _csv_of(setup["rows"].head(5).select("z")))
    assert t["status"] == "failed" and "'x'" in t["error"], t


def test_a_model_changed_after_deploying_fails_the_test_with_the_reason(setup):
    lead, p, q = setup["lead"], setup["ovr"]["p"], setup["ovr"]["q"]
    dep = _deploy(lead, setup["x"], run_ids=[p])
    model_dir = db.fetch_one("SELECT model_dir FROM meta.model_runs WHERE id = %s", (p,))["model_dir"]
    path = Path(model_dir) / scoring.MODEL_FILE
    other = Path(db.fetch_one("SELECT model_dir FROM meta.model_runs WHERE id = %s", (q,))["model_dir"])
    saved = path.read_bytes()
    scoring.forget(dep["id"])
    path.write_bytes((other / scoring.MODEL_FILE).read_bytes())
    try:
        t = _test(lead, dep["id"], "changed.csv", _csv_of(setup["rows"].head(5).select("x", "z")))
    finally:
        path.write_bytes(saved)
    assert t["status"] == "failed" and "'p'" in t["error"] and "changed" in t["error"], t


def test_starting_a_test_is_refused_while_retired_archived_or_refreshing(setup):
    lead, x = setup["lead"], setup["x"]
    dep = _deploy(lead, x, run_ids=[setup["multi"]])
    content = _csv_of(setup["rows"].head(3).select("x", "z"))
    assert lead.post(f"/api/deployments/{dep['id']}/retire").status_code == 200
    assert "retired" in _post(lead, dep["id"], "r.csv", content, 409)["detail"]
    assert lead.post(f"/api/deployments/{dep['id']}/activate").status_code == 200
    assert lead.patch(f"/api/experiments/{x['id']}", json={"archived": True}).status_code == 200
    try:
        assert "archived" in _post(lead, dep["id"], "a.csv", content, 409)["detail"]
    finally:
        assert lead.patch(f"/api/experiments/{x['id']}", json={"archived": False}).status_code == 200
    db.execute("UPDATE meta.datasets SET status = 'refreshing' WHERE id = %s", (setup["ds"]["id"],))
    try:
        _post(lead, dep["id"], "f.csv", content, 409)
    finally:
        db.execute("UPDATE meta.datasets SET status = 'ready' WHERE id = %s", (setup["ds"]["id"],))
    assert lead.get(f"/api/deployments/{dep['id']}/tests").json() == []
    assert _post(lead, dep["id"], "t.xlsx", content, 415)["detail"]
    assert _post(lead, dep["id"], "t.csv", b"", 400)["detail"]
    assert lead.post("/api/deployments/999999999/tests", files={"file": ("t.csv", content)}).status_code == 404
    assert lead.get("/api/deployment-tests/999999999").status_code == 404


def test_a_viewer_reads_tests_but_cant_start_one(setup, ovr_dep, app_, login_as):  # noqa: F811
    viewer = login_as(TestClient(app_), "viewer", name=f"Viewer {uuid.uuid4().hex[:6]}")
    content = _csv_of(setup["rows"].head(3).select("x", "z"))
    t = _test(setup["lead"], ovr_dep["id"], "seen.csv", content)
    assert _post(viewer, ovr_dep["id"], "v.csv", content, 403)
    assert viewer.get(f"/api/deployments/{ovr_dep['id']}/tests").status_code == 200
    assert viewer.get(f"/api/deployment-tests/{t['id']}").json()["status"] == "done"
    assert viewer.get(f"/api/deployment-tests/{t['id']}/download?format=csv").status_code == 200


# ------------------------------------------------------------------------------------------- keeping and sweeping
def _folder(dep_id: int, test_id: int) -> Path:
    return get_settings().deployments_dir / str(dep_id) / "tests" / str(test_id)


def test_the_last_20_tests_are_kept_and_older_ones_go_with_their_files(setup):
    lead = setup["lead"]
    dep = _deploy(lead, setup["x"], run_ids=[setup["multi"]])
    old = []
    for k in range(20):
        tid = db.fetch_one(
            """INSERT INTO meta.deployment_tests (deployment_id, filename, status, rows, has_labels)
               VALUES (%s, %s, 'done', 0, false) RETURNING id""",
            (dep["id"], f"old{k}.csv"),
        )["id"]
        _folder(dep["id"], tid).mkdir(parents=True)
        (_folder(dep["id"], tid) / "results.parquet").write_bytes(b"x")
        old.append(tid)
    new = _test(lead, dep["id"], "new.csv", _csv_of(setup["rows"].head(3).select("x", "z")))
    kept = [t["id"] for t in lead.get(f"/api/deployments/{dep['id']}/tests").json()]
    assert len(kept) == 20 and kept[0] == new["id"] and old[0] not in kept and old[1] in kept
    assert not _folder(dep["id"], old[0]).exists() and _folder(dep["id"], old[1]).exists()
    assert _folder(dep["id"], new["id"]).exists()
    shutil.rmtree(get_settings().deployments_dir / str(dep["id"]), ignore_errors=True)


def test_a_test_whose_process_is_gone_is_failed_by_the_sweep(setup, ovr_dep):
    def row(owner):
        return db.fetch_one(
            """INSERT INTO meta.deployment_tests (deployment_id, filename, status, owner)
               VALUES (%s, 'x.csv', 'running', %s) RETURNING id""",
            (ovr_dep["id"], owner),
        )["id"]

    gone, mine = row(515_151), row(jobs.owner_id())
    try:
        jobs.sweep()
        got = db.fetch_one("SELECT status, error FROM meta.deployment_tests WHERE id = %s", (gone,))
        assert got == {"status": "failed", "error": jobs.INTERRUPTED}
        assert db.fetch_one("SELECT status FROM meta.deployment_tests WHERE id = %s", (mine,))["status"] == "running"
    finally:
        db.execute("DELETE FROM meta.deployment_tests WHERE id IN (%s, %s)", (gone, mine))
