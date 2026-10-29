"""Scoring: raw records through an export's own preprocessing to each deployed run's scores, the equality check that
holds it to what training produced, and the shape of the output. The API tests need RAINFOREST_TEST_DATABASE_URL."""

import hashlib
import io
import json
import shutil
import threading
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
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


# =========================================================================== conversion of single values
def _in(name, dtype, fmt=None, comma=None, codes=False) -> scoring.Input:
    return scoring.Input(name, dtype, fmt, comma, codes)


def _conv(values, to, dtype=None):
    out, failed = scoring.convert(pl.Series(to.name, values, dtype=dtype), to)
    return out.to_list(), failed


def test_integer_codes_past_2_to_the_53_survive_exactly():
    codes = _in("x", pl.Int64, codes=True)
    big = [9007199254740993, 1234567890123456789]
    assert _conv(big, codes, pl.Int64) == (big, 0)
    assert _conv([str(v) for v in big], codes) == (big, 0)
    assert _conv(["7.0", "07", "+7", " 7 ", "7.5", "x"], codes) == ([7, 7, 7, 7, None, None], 2)
    assert _conv(["3000000000"], _in("x", pl.Int32, codes=True)) == ([None], 1)


def test_an_integer_typed_number_is_read_as_a_float():
    number = _in("age", pl.Int64)
    out, failed = scoring.convert(pl.Series("age", ["7.5", "8", "30.25", "abc"]), number)
    assert out.dtype == pl.Float64 and out.to_list() == [7.5, 8.0, 30.25, None] and failed == 1
    assert _conv([7.5, 8.0], number) == ([7.5, 8.0], 0)


def test_missing_is_the_upload_readers_null_tokens_and_nothing_else():
    text = _in("s", pl.String)
    got = _conv([" ", "NA", "null", "", None, "a", "N/A", "?"], text)
    assert got == ([" ", None, None, None, None, "a", None, None], 0)
    number = _in("f", pl.Float64)
    assert _conv(["NaN", "nan", "1.5", "abc", " "], number) == ([None, None, 1.5, None, None], 2)
    assert _conv([float("nan"), 2.0], number) == ([None, 2.0], 0)


def test_a_native_date_or_datetime_takes_iso_8601_text_only():
    from datetime import date, datetime, timezone

    when = _in("d", pl.Datetime("us"))
    # No guessing from the request's other values: day-first or month-first text is refused either way.
    assert _conv(["03/04/2020"], when) == ([None], 1)
    assert _conv(["13/04/2020", "03/04/2020"], when) == ([None, None], 2)
    got, failed = _conv(
        [
            "2020-04-03",
            "2020-04-03T10:00:00",
            "2020-04-03 10:00",
            "2020-04-03T10:00:00.5Z",
            "2020-04-03T10:00:00+02:00",
        ],
        when,
    )
    assert failed == 0 and got == [
        datetime(2020, 4, 3),
        datetime(2020, 4, 3, 10),
        datetime(2020, 4, 3, 10),
        datetime(2020, 4, 3, 10, 0, 0, 500000),
        datetime(2020, 4, 3, 8),
    ]
    day = _in("d", pl.Date)
    got = _conv(["2020-04-03", "2020-04-03T23:30:00-02:00", "2020/04/03"], day)
    assert got == ([date(2020, 4, 3), date(2020, 4, 4), None], 1)
    utc = _in("t", pl.Datetime("us", "UTC"))
    got, failed = _conv(["2020-04-03 10:00:00", "2020-04-03T10:00:00+02:00"], utc)
    assert failed == 0 and got == [
        datetime(2020, 4, 3, 10, tzinfo=timezone.utc),
        datetime(2020, 4, 3, 8, tzinfo=timezone.utc),
    ]
    assert _conv([datetime(2020, 4, 3, 10)], when) == ([datetime(2020, 4, 3, 10)], 0)
    assert _conv([20200403], when) == ([None], 1)


def test_a_boolean_takes_true_false_1_and_0():
    flag = _in("b", pl.Boolean)
    assert _conv(["yes", "TRUE", " false ", "0", None], flag) == ([None, True, False, False, None], 1)
    assert _conv([1, 0, 2], flag) == ([True, False, None], 1)
    assert _conv([True, None], flag) == ([True, None], 0)


def test_a_list_or_object_value_is_refused_naming_the_field():
    with pytest.raises(ValueError, match="'x'"):
        scoring.convert(pl.Series("x", [[1, 2]]), _in("x", pl.Int64))
    with pytest.raises(ValueError, match="'x'"):
        scoring.convert(pl.Series("x", [{"a": 1}]), _in("x", pl.String))


# =========================================================================== the equality check's rows
@contextmanager
def _swapped(table: str, row_id: int, column: str, value):
    """meta.<table>.<column> of one row set to `value` inside the block."""
    with connection() as conn:
        before = conn.execute(f"SELECT {column} FROM meta.{table} WHERE id = %s", (row_id,)).fetchone()[column]
        conn.execute(f"UPDATE meta.{table} SET {column} = %s WHERE id = %s", (value, row_id))
    try:
        yield before
    finally:
        with connection() as conn:
            conn.execute(f"UPDATE meta.{table} SET {column} = %s WHERE id = %s", (before, row_id))


def _nudge(cfg):
    cfg["derived"] = [{**d, "spec": {"expr": "w / c * 1.01"}} if d["name"] == "wc" else d for d in cfg["derived"]]
    return cfg


def _copy_export(setup, tmp_path, change) -> str:
    """The export's files copied, with model_ready.parquet changed by `change`."""
    src = fetch_one("SELECT output_dir FROM meta.exports WHERE id = %s", (setup["e"]["id"],))["output_dir"]
    dst = tmp_path / "export"
    shutil.copytree(src, dst)
    change(pl.read_parquet(dst / "model_ready.parquet")).write_parquet(dst / "model_ready.parquet")
    return str(dst)


def test_an_export_without_test_or_valid_rows_is_checked_on_a_sample_of_training_rows(setup, tmp_path):
    out = _copy_export(setup, tmp_path, lambda d: d.with_columns(pl.lit("train").alias("__split")))
    with _swapped("exports", setup["e"]["id"], "output_dir", out), connection() as conn:
        assert scoring.equality_check(conn, setup["e"]["id"], setup["ovr"]) is None
        with _config(setup["e"]["id"], _nudge):
            assert "row" in (scoring.equality_check(conn, setup["e"]["id"], setup["ovr"]) or "")


def test_the_equality_check_never_passes_having_compared_no_rows(setup, tmp_path):
    out = _copy_export(setup, tmp_path, lambda d: d.head(0))
    with _swapped("exports", setup["e"]["id"], "output_dir", out), connection() as conn:
        assert "no rows" in (scoring.equality_check(conn, setup["e"]["id"], setup["ovr"]) or "")


def test_the_refresh_hint_appears_only_when_the_dataset_was_refreshed_after_the_export(setup):
    with _config(setup["e"]["id"], _nudge), connection() as conn:
        msg = scoring.equality_check(conn, setup["e"]["id"], setup["multi"])
        assert msg and "refreshed" not in msg
        later = datetime.now(timezone.utc) + timedelta(days=1)
        with _swapped("datasets", setup["ds"]["id"], "profiled_at", later):
            assert "refreshed" in scoring.equality_check(conn, setup["e"]["id"], setup["multi"])


def test_a_one_vs_rest_run_without_an_f1_threshold_needs_one_given(setup, tmp_path):
    run_id = setup["ovr"][0]["run_id"]
    src = fetch_one("SELECT model_dir FROM meta.model_runs WHERE id = %s", (run_id,))["model_dir"]
    dst = tmp_path / "model"
    shutil.copytree(src, dst)
    report = json.loads((dst / "report.json").read_text())
    report.pop("f1_threshold")
    (dst / "report.json").write_text(json.dumps(report))
    with _swapped("model_runs", run_id, "model_dir", str(dst)), connection() as conn:
        with pytest.raises(ValueError, match="threshold"):
            scoring.load(conn, {"export_id": setup["e"]["id"], "models": setup["ovr"][:1]})
        given = [{**setup["ovr"][0], "threshold": 0.4}]
        assert scoring.load(conn, {"export_id": setup["e"]["id"], "models": given}).models[0].threshold == 0.4


# =========================================================================== native types and a number target
def _parquet(n: int = 800) -> bytes:
    """when (Datetime) and day (Date) native, flag Boolean, age an integer-typed number, x a float; price a number
    that follows them."""
    rng = np.random.default_rng(4)
    when = [datetime(2020, 1, 1) + timedelta(hours=int(h)) for h in rng.integers(0, 24 * 365, n)]
    day = [date(2021, 1, 1) + timedelta(days=int(d)) for d in rng.integers(0, 365, n)]
    flag = rng.random(n) < 0.4
    age = rng.integers(18, 91, n)
    x = rng.normal(size=n)
    month = np.array([w.month for w in when])
    price = 10 + 3 * x + 4 * flag + 0.05 * age + 0.3 * month + rng.normal(scale=0.3, size=n)
    buf = io.BytesIO()
    pl.DataFrame({"when": when, "day": day, "flag": flag, "age": age, "x": x, "price": price}).write_parquet(buf)
    return buf.getvalue()


AGE_GROUPS = {
    "name": "agegrp",
    "kind": "groups",
    "spec": {"source": "age", "other": "rest", "groups": [
        {"label": "thirty", "values": ["30"]}, {"label": "old", "ranges": [[60, 90]]}]},
}  # fmt: skip


@pytest.fixture(scope="module")
def typed(people):
    """A Parquet upload with native date, datetime and boolean columns and an integer-typed number read by a groups
    field; a number run of price."""
    lead = people["lead"]
    r = lead.post(
        "/api/datasets/upload",
        files={"file": ("typed.parquet", _parquet(), "application/octet-stream")},
        data={"targets": "price"},
    )
    assert r.status_code == 202, r.text
    ds = wait(lambda: (d := lead.get(f"/api/datasets/{r.json()['id']}").json())["status"] in ("ready", "failed") and d)
    assert ds["status"] == "ready", ds
    start(lead, ds, "tree", [lead], targets=("price",))
    rid = submit_and_finalize(lead, ds)["recipe"]["id"]
    assert lead.post(f"/api/datasets/{ds['id']}/derived", json=AGE_GROUPS).status_code == 201
    start(lead, ds, "tree", [lead], start="baseline", source=rid, targets=("price",))
    rid = submit_and_finalize(lead, ds)["recipe"]["id"]
    x = experiment(lead, ds, rid)
    adjust(lead, ds, x, "price", target_task="regression")
    e = _export(lead, ds, x)
    run = _run(lead, e, order=["price"])
    return {"ds": ds, "e": e, "models": [{"run_id": run["id"], "label": "price", "kind": "regression"}]}


def _typed_scorer(typed) -> scoring.Scorer:
    with connection() as conn:
        return scoring.load(conn, {"export_id": typed["e"]["id"], "models": typed["models"]})


def _typed_records(typed, n=4) -> pl.DataFrame:
    path = fetch_one("SELECT parquet_path FROM meta.datasets WHERE id = %s", (typed["ds"]["id"],))["parquet_path"]
    return pl.read_parquet(path, columns=["when", "day", "flag", "age", "x"], n_rows=n)


def test_the_typed_exports_inputs_keep_their_native_types(typed):
    schema = {i.name: i.dtype for i in _typed_scorer(typed).inputs}
    assert schema == {"when": pl.Datetime("us"), "day": pl.Date, "flag": pl.Boolean, "age": pl.Int64, "x": pl.Float64}


def test_a_number_run_scores_through_outputs_and_passes_the_equality_check(typed):
    s = _typed_scorer(typed)
    outputs, _ = s.outputs(_typed_records(typed))
    out = s.score(_typed_records(typed))
    assert list(out.scores) == ["price"] and out.scores["price"].shape == (4,)
    assert np.array_equal(outputs[0], out.scores["price"])
    assert out.top is None and out.matches is None and out.none_matched is None
    with connection() as conn:
        assert scoring.equality_check(conn, typed["e"]["id"], typed["models"]) is None


def test_native_dates_booleans_and_numbers_sent_as_text_score_as_the_native_values(typed):
    s = _typed_scorer(typed)
    recs = _typed_records(typed)
    native = s.score(recs).scores["price"]
    text = recs.with_columns(
        pl.col("when").dt.strftime("%Y-%m-%dT%H:%M:%S"),
        pl.col("day").dt.strftime("%Y-%m-%d"),
        pl.col("flag").cast(pl.String).str.to_uppercase(),
        pl.col("age").cast(pl.String),
        pl.col("x").cast(pl.String),
    )
    out = s.score(text)
    assert out.warnings == {} and np.array_equal(out.scores["price"], native)
    bad = text.with_columns(when=pl.lit("03/04/2020"), flag=pl.lit("yes"))
    assert s.score(bad).warnings == {"when": 4, "flag": 4}


def test_an_integer_typed_number_read_by_a_groups_field_matches_as_the_integer_it_was(typed):
    s = _typed_scorer(typed)
    recs = _typed_records(typed, 1)
    for v in (30, "30", 30.0, "30.0"):
        frame, warnings = s.frame(recs.with_columns(age=pl.lit(v)))
        assert frame["agegrp"].to_list() == ["thirty"] and warnings == {}, v
    frame, warnings = s.frame(recs.with_columns(age=pl.lit(30.5)))
    assert frame["agegrp"].to_list() == ["rest"] and frame["age"].to_list() == [30.5] and warnings == {}
    frame, _ = s.frame(recs.with_columns(age=pl.lit("75")))
    assert frame["agegrp"].to_list() == ["old"]


# =========================================================================== model files, features, time types
def _input_dtype(name: str, dtype: str):
    def change(cfg):
        cfg["input_schema"] = {**cfg["input_schema"], name: {**cfg["input_schema"][name], "dtype": dtype}}
        return cfg

    return change


@pytest.mark.parametrize("dtype", ["Time", "Duration(time_unit='us')"])
def test_a_field_stored_as_a_time_of_day_or_a_duration_is_refused_naming_it(setup, dtype):
    with _config(setup["e"]["id"], _input_dtype("c", dtype)), pytest.raises(ValueError, match="'c'") as exc:
        _scorer(setup)
    assert "deploy" in str(exc.value)


def test_the_equality_check_names_the_first_feature_that_differs_even_when_scores_agree(setup, tmp_path):
    src = Path(fetch_one("SELECT output_dir FROM meta.exports WHERE id = %s", (setup["e"]["id"],))["output_dir"])
    feats = json.loads((src / "manifest.json").read_text())["feature_names"]
    data = pl.read_parquet(src / "model_ready.parquet")
    name = next(f for f in feats if data[f].dtype == pl.Float32 and data[f].null_count() < len(data))

    def ulp(d: pl.DataFrame) -> pl.DataFrame:
        # One float32 step: no tree split moves, so every score agrees, but the features don't.
        return d.with_columns(pl.Series(name, np.nextafter(d[name].to_numpy(), np.float32(np.inf))))

    out = _copy_export(setup, tmp_path, ulp)
    with _swapped("exports", setup["e"]["id"], "output_dir", out), connection() as conn:
        msg = scoring.equality_check(conn, setup["e"]["id"], setup["multi"])
    assert msg and repr(name) in msg and "row" in msg, msg


def test_a_model_file_that_changed_since_it_was_recorded_is_refused_naming_the_model(setup):
    m = setup["ovr"][0]
    model_dir = Path(fetch_one("SELECT model_dir FROM meta.model_runs WHERE id = %s", (m["run_id"],))["model_dir"])
    digest = scoring.file_sha256(model_dir / scoring.MODEL_FILE)
    with connection() as conn:
        assert scoring.load(conn, {"export_id": setup["e"]["id"], "models": [{**m, "sha256": digest}]})
        with pytest.raises(scoring.ModelChanged, match="'p'"):
            scoring.load(conn, {"export_id": setup["e"]["id"], "models": [{**m, "sha256": "0" * 64}]})


def _model_path(m: dict) -> Path:
    model_dir = fetch_one("SELECT model_dir FROM meta.model_runs WHERE id = %s", (m["run_id"],))["model_dir"]
    return Path(model_dir) / scoring.MODEL_FILE


def test_the_model_loaded_is_the_bytes_whose_hash_was_checked(setup, monkeypatch):
    """A file swapped between the hash check and the unpickling is not what gets loaded."""
    p, q = setup["ovr"][0], setup["ovr"][1]
    p_file, q_bytes = _model_path(p), _model_path(q).read_bytes()
    saved = p_file.read_bytes()
    real = scoring.joblib.load

    def swap_then_load(*args, **kwargs):
        p_file.write_bytes(q_bytes)
        return real(*args, **kwargs)

    monkeypatch.setattr(scoring.joblib, "load", swap_then_load)
    try:
        with connection() as conn:
            got = scoring.load(
                conn, {"export_id": setup["e"]["id"], "models": [{**p, "sha256": hashlib.sha256(saved).hexdigest()}]}
            )
    finally:
        monkeypatch.undo()
        p_file.write_bytes(saved)
    want = _scorer(setup).outputs(_records(setup, 50))[0][0]
    assert np.array_equal(got.outputs(_records(setup, 50))[0][0], want)


def test_a_preprocessor_that_changed_since_it_was_recorded_is_refused_naming_the_export(setup):
    out = Path(fetch_one("SELECT output_dir FROM meta.exports WHERE id = %s", (setup["e"]["id"],))["output_dir"])
    with connection() as conn:
        digest = scoring.preprocessor_sha256(conn, setup["e"]["id"])
        assert digest == hashlib.sha256((out / "preprocessor.joblib").read_bytes()).hexdigest()
        spec = {"export_id": setup["e"]["id"], "models": setup["multi"]}
        assert scoring.load(conn, {**spec, "preprocessor_sha256": digest})
        with pytest.raises(scoring.ModelChanged, match=f"export {setup['e']['id']}"):
            scoring.load(conn, {**spec, "preprocessor_sha256": "0" * 64})
        with pytest.raises(scoring.ModelChanged, match=f"export {setup['e']['id']}"):
            scoring.equality_check(conn, setup["e"]["id"], setup["multi"], "0" * 64)


def test_iso_8601_text_follows_rfc_3339_case_and_needs_a_time_before_an_offset():
    got = scoring.iso_datetime(
        pl.Series(["2020-04-03t10:00:00z", "2020-04-03T10:00:00Z", "2020-04-03Z", "2020-04-03+02:00", "2020-04-03"])
    ).to_list()
    assert got == [datetime(2020, 4, 3, 10), datetime(2020, 4, 3, 10), None, None, datetime(2020, 4, 3)]


def test_an_integer_typed_number_with_a_decimal_comma_reads_as_its_integer_text():
    s = pl.Series("age", ["30,0", "30,5", "7"])
    out = scoring.coerce_numeric(s, True)
    assert scoring.integer_text(s, out, True).to_list() == ["30", "30.5", "7"]
