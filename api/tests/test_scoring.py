"""Scoring: raw records through an export's own preprocessing to each deployed run's scores, the equality check that
holds it to what training produced, and the shape of the output. The API tests need RAINFOREST_TEST_DATABASE_URL."""

import threading
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from app import scoring
from app.db import connection, fetch_one, jsonb

from .team_helpers import (  # noqa: F401  (fixtures)
    adjust,
    app_,
    experiment,
    fields,
    finalized_trees,
    people,
    start,
    submit_and_finalize,
    wait,
)
from .test_export_freeze import _dates, _export


def _csv(n: int = 900) -> str:
    """w and c numbers, site_code 1-7 (an integer-coded category), d dd/mm dates, batch text; y three classes that
    follow w / c and site_code."""
    rng = np.random.default_rng(8)
    dates = _dates(n, 200)
    rows = []
    for i in range(n):
        w, c, site = 150 + 10 * rng.normal(), 200 + 10 * rng.normal(), 1 + i % 7
        s = (w / c - 0.75) * 20 + (site <= 2) + 0.4 * rng.normal()
        y = "p" if s < -0.5 else "q" if s < 0.6 else "r"
        rows.append(f"{w:.3f},{c:.3f},{site},{dates[i]},{'pq'[i % 2]}{i % 3},{y}\n")
    return "w,c,site_code,d,batch,y\n" + "".join(rows)


DERIVED = [
    {"name": "wc", "kind": "formula", "spec": {"expr": "w / c"}},
    {"name": "band", "kind": "groups", "spec": {"source": "wc", "other": "high", "groups": [
        {"label": "low", "ranges": [[0, 0.75]]}]}},
    {"name": "hits", "kind": "count", "spec": {"fields": ["site_code"], "values": ["1", "2"]}},
]  # fmt: skip


def _run(client, e, **body) -> dict:
    r = client.post(
        f"/api/exports/{e['id']}/runs",
        json={"model": "lightgbm", "cv": 0, "cv_folds": 0, "params": {"n_estimators": 20}, **body},
    )
    assert r.status_code == 202, r.text
    run = wait(lambda: (x := client.get(f"/api/runs/{r.json()['id']}").json())["status"] in ("done", "failed") and x)
    assert run["status"] == "done", run.get("error")
    return run


@pytest.fixture(scope="module")
def setup(people):
    """An export whose features read w only through wc, and leave batch out; one-vs-rest runs of p, q and r, and one
    multiclass run of y."""
    lead = people["lead"]
    ds, _ = finalized_trees(lead, _csv())
    for d in DERIVED:
        r = lead.post(f"/api/datasets/{ds['id']}/derived", json=d)
        assert r.status_code == 201, r.text
    rid = lead.get(f"/api/datasets/{ds['id']}/baselines").json()[0]["latest_finalized_id"]
    start(lead, ds, "tree", [lead], start="baseline", source=rid)
    rid = submit_and_finalize(lead, ds)["recipe"]["id"]
    x = experiment(lead, ds, rid)
    for name in ("w", "batch"):
        adjust(lead, ds, x, name, action="drop")
    e = _export(lead, ds, x, exclude=[{"field": "batch", "values": ["zz"]}])
    ovr = [_run(lead, e, target="y", positive_class=c) for c in "pqr"]
    multi = _run(lead, e, order=["y"])
    return {
        "lead": lead,
        "ds": ds,
        "e": e,
        "ovr": [{"run_id": r["id"], "label": c, "kind": "binary", "positive_class": c} for r, c in zip(ovr, "pqr")],
        "multi": [{"run_id": multi["id"], "label": "y", "kind": "multiclass"}],
    }


def _scorer(setup, which="ovr") -> scoring.Scorer:
    with connection() as conn:
        return scoring.load(conn, {"export_id": setup["e"]["id"], "models": setup[which]})


def _records(setup, n=5) -> pl.DataFrame:
    """The dataset's first rows, as text, the way a CSV of new records arrives."""
    path = fetch_one("SELECT parquet_path FROM meta.datasets WHERE id = %s", (setup["ds"]["id"],))["parquet_path"]
    return pl.read_parquet(path, columns=["w", "c", "site_code", "d", "batch", "y"], n_rows=n).cast(pl.String)


@contextmanager
def _config(export_id: int, change):
    """The export's config changed by `change` inside the block."""
    with connection() as conn:
        before = conn.execute("SELECT config FROM meta.exports WHERE id = %s", (export_id,)).fetchone()["config"]
        conn.execute("UPDATE meta.exports SET config = %s WHERE id = %s", (jsonb(change(dict(before))), export_id))
    try:
        yield
    finally:
        with connection() as conn:
            conn.execute("UPDATE meta.exports SET config = %s WHERE id = %s", (jsonb(before), export_id))


def test_scoring_the_exports_test_rows_from_raw_values_reproduces_training(setup):
    with connection() as conn:
        assert scoring.equality_check(conn, setup["e"]["id"], setup["ovr"]) is None
        assert scoring.equality_check(conn, setup["e"]["id"], setup["multi"]) is None


def test_the_equality_check_names_the_model_and_row_that_differ(setup):
    def nudge(cfg):
        cfg["derived"] = [{**d, "spec": {"expr": "w / c * 1.01"}} if d["name"] == "wc" else d for d in cfg["derived"]]
        return cfg

    with _config(setup["e"]["id"], nudge), connection() as conn:
        msg = scoring.equality_check(conn, setup["e"]["id"], setup["ovr"])
    assert msg is not None and "row" in msg and any(f"y = {c}" in msg or repr(c) in msg for c in "pqr"), msg


def test_required_fields_are_the_features_sources_through_derived_fields(setup):
    s = _scorer(setup)
    # w is read only through wc; batch (dropped, but named by a row rule) and the target y aren't needed.
    assert set(s.required) == {"w", "c", "site_code", "d"}
    with pytest.raises(ValueError, match=r"Missing field: 'w'"):
        s.score(_records(setup).drop("w"))
    out = s.score(_records(setup).drop("batch", "y").with_columns(extra=pl.lit("ignored")))
    assert set(out.scores) == {"p", "q", "r"} and len(out.scores["p"]) == 5


def test_derived_fields_are_computed_from_the_frozen_spec_not_the_records(setup):
    s = _scorer(setup)
    recs = _records(setup)
    base = s.score(recs)
    # A wc column sent with the record is ignored: scoring computes it from w and c.
    sent = s.score(recs.with_columns(wc=pl.lit("999"), hits=pl.lit("9")))
    for c in "pqr":
        assert np.array_equal(base.scores[c], sent.scores[c])
    # Another w changes wc, and so the scores.
    moved = s.score(recs.with_columns(w=pl.lit("400")))
    assert not np.array_equal(base.scores["p"], moved.scores["p"])


def test_an_integer_coded_category_sent_as_7_0_or_07_scores_as_7(setup):
    s = _scorer(setup)
    recs = _records(setup, 1)
    seven = s.score(recs.with_columns(site_code=pl.lit("7")))
    for v in ("7.0", "07", " 7 "):
        out = s.score(recs.with_columns(site_code=pl.lit(v)))
        assert out.warnings == {}
        for c in "pqr":
            assert out.scores[c][0] == seven.scores[c][0], v
    as_number = s.score(recs.with_columns(site_code=pl.lit(7.0)))
    assert as_number.scores["p"][0] == seven.scores["p"][0]
    # 1 is a hits value: had "1.0" stayed text, the count would miss it.
    one = s.score(recs.with_columns(site_code=pl.lit("1")))
    assert s.score(recs.with_columns(site_code=pl.lit("1.0"))).scores["p"][0] == one.scores["p"][0]


def test_a_value_that_doesnt_convert_is_missing_and_counted(setup):
    s = _scorer(setup)
    recs = _records(setup, 4)
    bad = recs.with_columns(
        c=pl.Series(["abc", None, "", "201.5"]),
        site_code=pl.Series(["7.5", "x", "3", "3"]),
        d=pl.Series(["not a date", "01/02/2020", "31/12/2020", "2020-13-45"]),
    )
    out = s.score(bad)
    assert out.warnings == {"c": 1, "site_code": 2, "d": 2}  # null and blank are missing, not failures
    gone = recs.with_columns(
        c=pl.Series([None, None, None, "201.5"]),
        site_code=pl.Series([None, None, "3", "3"]),
        d=pl.Series([None, "01/02/2020", "31/12/2020", None]),
    )
    want = s.score(gone)
    assert want.warnings == {}
    for c in "pqr":
        assert np.array_equal(out.scores[c], want.scores[c])


def test_a_multiclass_run_scores_every_class_and_has_no_matches(setup):
    out = _scorer(setup, "multi").score(_records(setup))
    assert set(out.scores) == {"p", "q", "r"}
    total = out.scores["p"] + out.scores["q"] + out.scores["r"]
    assert np.allclose(total, 1.0)
    assert out.matches is None and out.none_matched is None
    for (label, score), k in zip(out.top, range(5)):
        assert score == max(out.scores[c][k] for c in "pqr") and out.scores[label][k] == score


def test_an_export_from_before_the_freeze_is_refused(setup):
    def strip(cfg):
        cfg.pop("derived")
        cfg.pop("input_schema")
        return cfg

    with _config(setup["e"]["id"], strip), pytest.raises(ValueError, match="Re-export to deploy"):
        _scorer(setup)


def test_a_feature_reading_an_unavailable_derived_field_is_refused_with_the_reason(setup):
    reason = "It reads 'gone', no longer in the dataset."

    def mark(cfg):
        cfg["derived"] = [{**d, "unavailable": reason} if d["name"] == "band" else d for d in cfg["derived"]]
        return cfg

    with _config(setup["e"]["id"], mark), pytest.raises(ValueError, match="band") as exc:
        _scorer(setup)
    assert reason in str(exc.value)


def test_an_unavailable_derived_field_only_a_row_rule_reads_doesnt_block(setup):
    def add(cfg):
        cfg["derived"] = [
            *cfg["derived"],
            {"name": "batch", "kind": "formula", "spec": {"expr": "gone + 1"}, "unavailable": "It reads 'gone'."},
        ]
        return cfg

    with _config(setup["e"]["id"], add):
        assert set(_scorer(setup).required) == {"w", "c", "site_code", "d"}


def test_a_scorer_scores_from_several_threads_at_once(setup):
    s = _scorer(setup)
    recs = _records(setup, 50)
    want = s.score(recs).scores
    bad = []

    def go():
        for _ in range(5):
            got = s.score(recs).scores
            if any(not np.array_equal(got[c], want[c]) for c in "pqr"):
                bad.append(got)

    threads = [threading.Thread(target=go) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not bad


def test_load_caches_deployments_by_id_up_to_the_setting(setup, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "deployment_cache", 2)
    scoring.clear()
    with connection() as conn:
        a = scoring.load(conn, {"id": -1, "export_id": setup["e"]["id"], "models": setup["ovr"]})
        assert scoring.load(conn, {"id": -1, "export_id": setup["e"]["id"], "models": setup["ovr"]}) is a
        scoring.load(conn, {"id": -2, "export_id": setup["e"]["id"], "models": setup["multi"]})
        scoring.load(conn, {"id": -3, "export_id": setup["e"]["id"], "models": setup["multi"]})
        assert scoring.load(conn, {"id": -1, "export_id": setup["e"]["id"], "models": setup["ovr"]}) is not a
    scoring.clear()


def _model(label, kind, labels, threshold=None):
    return SimpleNamespace(label=label, kind=kind, labels=tuple(labels), threshold=threshold)


def test_top_matches_and_none_matched_on_a_constructed_case():
    models = [_model("a", "binary", ["a"], 0.5), _model("b", "binary", ["b"], 0.3), _model("c", "binary", ["c"], 0.9)]
    outputs = [np.array([0.6, 0.1, 0.2]), np.array([0.7, 0.2, 0.25]), np.array([0.1, 0.05, 0.95])]
    out = scoring.summarise(models, outputs, {"w": 1})
    assert list(out.scores) == ["a", "b", "c"] and out.warnings == {"w": 1}
    assert out.top == [("b", 0.7), ("b", 0.2), ("c", 0.95)]
    assert out.matches == [["b", "a"], [], ["c"]]  # best first; at or above each model's own threshold
    assert out.none_matched == [False, True, False]

    at = scoring.summarise([_model("a", "binary", ["a"], 0.5)], [np.array([0.5])], {})
    assert at.matches == [["a"]] and at.none_matched == [False]


def test_a_number_run_has_scores_only():
    out = scoring.summarise([_model("price", "regression", ["price"])], [np.array([3.5, 7.25])], {})
    assert list(out.scores) == ["price"] and out.scores["price"].tolist() == [3.5, 7.25]
    assert out.top is None and out.matches is None and out.none_matched is None
