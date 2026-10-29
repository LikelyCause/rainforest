"""Testing a deployment with an uploaded file: the job that scores it in chunks, the metrics a labelled file gets, the
results and their download, warnings, retention, how a file's labels are named, and what refuses a test. Needs RAINFOREST_TEST_DATABASE_URL."""

import io
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import average_precision_score

from app import db, deployments, jobs, scoring
from app.config import get_settings
from app.transform import prepare_targets

from .team_helpers import (  # noqa: F401  (fixtures)
    app_,
    experiment,
    export,
    fields,
    people,
    save,
    start,
    submit_and_finalize,
    upload,
    wait,
)


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
    given = ["abc", "", *map(str, df["v"].to_list()[2:])]  # a label that isn't a number is given, not missing
    m = _test(lead, dep["id"], "number.csv", _csv_of(df.with_columns(v=pl.Series(given))))["metrics"]
    got = np.array([r["scores"]["v"] for r in _predicted(lead, dep["id"], df)])
    err = (got - df["v"].to_numpy())[2:]
    assert m["labelled_rows"] == 199 and m["rows_without_label"] == 1 and m["rows_label_not_a_number"] == 1
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


def _trained_classes(tmp_path, values: list, task: dict | None = None) -> list[str]:
    """The class names training gives a target column holding `values`."""
    path = tmp_path / "y.parquet"
    pl.DataFrame({"y": values}).write_parquet(path)
    return prepare_targets(path, ["y"], {"y": task or {"task": "classes"}}, {}, 1)[2]["y"]


def _labels(values, classes, dtype, bins=None) -> list:
    """`values` (as a file sends them) named against `classes`, for a target of type `dtype` in the dataset."""
    from app.transform import target_labels

    out = target_labels(pl.Series("y", values), bins, classes, dtype=dtype)
    return [None if pd.isna(v) else v for v in out]


def test_a_files_labels_are_named_as_training_named_the_classes(tmp_path):
    # An integer target with a missing value reaches training as floats, so its classes are "1.0", "2.0" and "3.0".
    classes = _trained_classes(tmp_path, [1, 2, 3, None] * 3)
    assert classes == ["1.0", "2.0", "3.0"]
    got = _labels(["1", "2.0", " 3", "07", "7.5", "abc", "yes", "", None], classes, pl.Int64)
    assert got == ["1.0", "2.0", "3.0", "07", "7.5", "abc", "yes", None, None]
    assert _labels([1, 3, None], classes, pl.Int64) == ["1.0", "3.0", None]
    # Without the missing value they are "1", "2", "3", and a float label still finds them.
    classes = _trained_classes(tmp_path, [1, 2, 3] * 3)
    assert classes == ["1", "2", "3"] and _labels([1.0, 2.5], classes, pl.Int64) == ["1", "2.5"]
    # Text classes match only as text.
    assert _labels(["07", "7"], ["7", "a"], pl.String) == ["07", "7"]


def test_a_files_banded_and_grouped_labels_are_the_band_and_the_group(tmp_path):
    task = {"task": "bins", "bins": {"edges": [10, 20]}}
    classes = _trained_classes(tmp_path, [5.0, 15.0, 25.0] * 3, task)
    bins = {"edges": [10.0, 20.0], "labels": classes}
    assert _labels(["5", "10", "19.9", "30.5", "x", None], classes, pl.Float64, bins) == [
        classes[0], classes[1], classes[1], classes[2], "x", None,
    ]  # fmt: skip
    task = {"task": "bins", "bins": {"positive": ["a", "b"]}}
    classes = _trained_classes(tmp_path, ["a", "b", "c", "d"] * 3, task)
    assert classes == ["(rest)", "a / b"]
    assert _labels(["a", "b", "c", "zz", "", None], classes, pl.String, {"positive": ["a", "b"]}) == [
        "a / b", "a / b", "(rest)", "(rest)", None, None,
    ]  # fmt: skip
    # An integer target grouped on "1": a file's 1.0 is in the group too.
    assert _labels([1.0, 2.0], ["(rest)", "1"], pl.Int64, {"positive": ["1"]}) == ["1", "(rest)"]


def test_a_boolean_targets_labels_are_read_as_feature_values_are(tmp_path):
    classes = _trained_classes(tmp_path, [True, False] * 3)
    assert classes == ["False", "True"]
    # As text (a Parquet file's or a CSV column with other values), the way a Boolean feature reads it.
    got = _labels(["true", "1", "0", " FALSE", "True", "maybe", "", None], classes, pl.Boolean)
    assert got == ["True", "True", "False", "False", "True", "maybe", None, None]
    assert _labels([1, 0, 2], classes, pl.Boolean) == ["True", "False", "2"]
    assert _labels([True, None], classes, pl.Boolean) == ["True", None]


def test_a_boolean_target_grouped_on_the_uis_true_trains_on_both_groups(tmp_path):
    path = tmp_path / "y.parquet"
    pl.DataFrame({"y": [True, False, None] * 4}).write_parquet(path)
    task = {"task": "bins", "bins": {"positive": ["true"]}}
    _, y, classes, *_ = prepare_targets(path, ["y"], {"y": task}, {}, 1)
    assert classes["y"] == ["(rest)", "true"]
    codes = y["y"].tolist()
    assert codes[:3] == [1.0, 0.0, codes[2]] and np.isnan(codes[2])


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
    for tid in (gone, mine):
        _folder(ovr_dep["id"], tid).mkdir(parents=True)
    tests = deployments.tests_dir(ovr_dep["id"])
    dead_upload, live_upload = tests / ".incoming-515151-abc", tests / f".incoming-{jobs.owner_id()}-abc"
    for d in (dead_upload, live_upload):
        d.mkdir()
    try:
        jobs.sweep()
        got = db.fetch_one("SELECT status, error FROM meta.deployment_tests WHERE id = %s", (gone,))
        assert got == {"status": "failed", "error": jobs.INTERRUPTED}
        assert db.fetch_one("SELECT status FROM meta.deployment_tests WHERE id = %s", (mine,))["status"] == "running"
        # The failed test's half-written files and a dead process's upload go; a live process's stay.
        assert not _folder(ovr_dep["id"], gone).exists() and _folder(ovr_dep["id"], mine).exists()
        assert not dead_upload.exists() and live_upload.exists()
    finally:
        db.execute("DELETE FROM meta.deployment_tests WHERE id IN (%s, %s)", (gone, mine))
        for d in (_folder(ovr_dep["id"], mine), live_upload):
            shutil.rmtree(d, ignore_errors=True)


def test_a_member_who_cant_see_the_experiment_cant_see_its_tests(people):
    lead, owner, member = people["lead"], people["a"], people["c"]
    ds = upload(lead, _csv(300), targets="y")
    start(lead, ds, "tree", [owner])
    rid = lead.get(f"/api/datasets/{ds['id']}").json()["recipe_id"]
    x = owner.post(
        f"/api/datasets/{ds['id']}/experiments", json={"name": "mine", "source": f"assessment:{rid}:{owner.user['id']}"}
    ).json()
    assert x["private"], x
    run = _run(owner, export(owner, ds, x), order=["y"])
    dep = _deploy(owner, x, run_ids=[run["id"]])
    rows = pl.read_csv(io.StringIO(_csv(300))).head(3).drop("y")
    t = _test(owner, dep["id"], "private.csv", _csv_of(rows))
    assert t["status"] == "done", t["error"]
    paths = [
        f"/api/deployment-tests/{t['id']}",
        f"/api/deployment-tests/{t['id']}/download?format=csv",
        f"/api/deployment-tests/{t['id']}/download?format=parquet",
        f"/api/deployments/{dep['id']}/tests",
    ]
    assert [owner.get(p).status_code for p in paths] == [200] * 4
    assert [member.get(p).status_code for p in paths] == [404] * 4
    assert [lead.get(p).status_code for p in paths] == [404] * 4


# ------------------------------------------------------------------------------------------- typed targets
def _typed_csv(n=900, seed=11) -> str:
    """x, z features; five targets that follow x: y_int (codes 1-3, one row missing), y_float (classes 0.5, 1.5,
    2.5), y_band (a number cut into bands), y_group (a, b, c, predicted as a against the rest), y_bool (true, false)."""
    rng = np.random.default_rng(seed)
    x, z = rng.normal(size=n), rng.normal(size=n)
    s = x + 0.4 * rng.normal(size=n)
    k = np.digitize(s, [-0.4, 0.5])
    rows = []
    for i in range(n):
        y_int = "" if i == 0 else str(k[i] + 1)
        y_bool = "true" if k[i] else "false"
        rows.append(f"{x[i]:.3f},{z[i]:.3f},{y_int},{k[i] + 0.5},{10 * s[i]:.3f},{'abc'[k[i]]},{y_bool}\n")
    return "x,z,y_int,y_float,y_band,y_group,y_bool\n" + "".join(rows)


TYPED = {
    "y_int": {"target_task": "classes"},
    "y_float": {"target_task": "classes"},
    "y_band": {"target_task": "bins", "target_bins": {"edges": [-4, 5]}},
    "y_group": {"target_task": "bins", "target_bins": {"positive": ["a"]}},
    "y_bool": {"target_task": "classes"},
}


def _band(v: float) -> str:
    return "< -4" if v < -4 else "[-4, 5)" if v < 5 else "≥ 5"


# How training names each target's classes, written out independently of the code under test.
TRAINED_NAME = {
    "y_int": lambda v: f"{float(v)}",  # a missing value made the column float
    "y_float": lambda v: f"{float(v)}",
    "y_band": lambda v: _band(float(v)),
    "y_group": lambda v: "a" if v == "a" else "(rest)",
    "y_bool": str,  # pandas prints True as "True"
}


@pytest.fixture(scope="module")
def typed(people):
    lead = people["lead"]
    ds = upload(lead, _typed_csv(), targets=",".join(TYPED))
    start(lead, ds, "tree", [lead], targets=tuple(TYPED))
    f = fields(lead, ds)
    for name, decision in TYPED.items():
        r = save(lead, f[name]["id"], action="target", **decision)
        assert r.status_code == 200, r.text
    rid = submit_and_finalize(lead, ds)["recipe"]["id"]
    x = experiment(lead, ds, rid, "typed targets")
    e = export(lead, ds, x)
    path = db.fetch_one("SELECT parquet_path FROM meta.datasets WHERE id = %s", (ds["id"],))["parquet_path"]
    deps = {t: _deploy(lead, x, run_ids=[_run(lead, e, order=[t])["id"]]) for t in TYPED}
    return {"lead": lead, "deps": deps, "rows": pl.read_parquet(path)}


@pytest.mark.parametrize("target", list(TYPED))
def test_a_typed_targets_labels_count_as_training_named_them(typed, target):
    lead, dep = typed["lead"], typed["deps"][target]
    df = typed["rows"].slice(1, 300).select("x", "z", target)
    t = _test(lead, dep["id"], f"{target}.csv", _csv_of(df))
    assert t["status"] == "done", t["error"]
    m = t["metrics"]
    classes = {c["label"] for c in m["classes"]}
    names = [TRAINED_NAME[target](v) for v in df[target].to_list()]
    assert set(names) <= classes, (names[:5], classes)
    tops = [r["top"]["label"] for r in _predicted(lead, dep["id"], df)]
    want = np.mean([a == b for a, b in zip(tops, names)])
    assert m["rows_label_without_model"] == 0 and m["rows_without_label"] == 0
    assert m["top1_accuracy"] == pytest.approx(want) and want > 0.5


def test_a_boolean_targets_labels_sent_as_text_in_parquet_count_as_its_classes(typed):
    lead, dep = typed["lead"], typed["deps"]["y_bool"]
    df = typed["rows"].slice(1, 200).select("x", "z", "y_bool")
    text = [("1" if v else "0") if i % 2 else ("true" if v else "false") for i, v in enumerate(df["y_bool"])]
    buf = io.BytesIO()
    df.with_columns(y_bool=pl.Series(text)).write_parquet(buf)
    m = _test(lead, dep["id"], "bool.parquet", buf.getvalue())["metrics"]
    tops = [r["top"]["label"] for r in _predicted(lead, dep["id"], df)]
    want = np.mean([a == str(v) for a, v in zip(tops, df["y_bool"])])
    assert m["rows_label_without_model"] == 0 and m["top1_accuracy"] == pytest.approx(want) and want > 0.5


def test_a_typed_label_that_names_no_class_is_wrong_not_unlabelled(typed):
    lead, dep = typed["lead"], typed["deps"]["y_int"]
    df = typed["rows"].slice(1, 20).select("x", "z", "y_int")
    given = ["7.5", "abc", "yes", ""] + [str(v) for v in df["y_int"].to_list()[4:]]
    df = df.with_columns(y_int=pl.Series(given))
    m = _test(lead, dep["id"], "odd.csv", _csv_of(df))["metrics"]
    assert m["rows_without_label"] == 1 and m["labelled_rows"] == 19 and m["rows_label_without_model"] == 3
    tops = [r["top"]["label"] for r in _predicted(lead, dep["id"], df)]
    hits = sum(tops[i] == f"{float(given[i])}" for i in range(4, 20))
    assert m["top1_accuracy"] == pytest.approx(hits / 19)
