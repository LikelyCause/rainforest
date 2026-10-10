"""Wrong-data regressions: ingest fallback, type detection and coercion accounting, inf handling, number and date
formats, target / feature name collisions, rare classes and target eligibility.

The API tests at the end need an empty Postgres (RAINFOREST_TEST_DATABASE_URL), like test_explore.py.
"""
import datetime as dt
import json
import os
import time

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest

from app import ingest
from app.modeling.train import train
from app.profiling import coerce_numeric, detect_datetime_format, infer_type, parse_datetime, profile_column
from app.semantics import MAX_TARGET_LEVELS, bins_problem, default_task, suggest, validate_decision
from app.transform import FieldSpec, assign_splits, coercion_exprs, run_export, safe_feature_names

INF = float("inf")


def _profile(values, name="col", is_target=False):
    return profile_column(pl.Series(name, values), 0, is_target, None, None, 50)


def _infer(values, name="col"):
    s = pl.Series(name, values)
    nn = s.drop_nulls()
    return infer_type(s, name, int(nn.n_unique()), nn.len())


def _spec(name, t, family="tree", **kw):
    s = suggest(t, {}, False, family)
    return FieldSpec(name, t, s["imputation"], s.get("impute_value"), s["add_missing_indicator"], s["transform"],
                     s["scaling"], s["encoding"], s["params"], **kw)


def _export(tmp_path, df: pl.DataFrame, specs, targets, family="tree"):
    src = tmp_path / "raw.parquet"
    df.write_parquet(src)
    out = tmp_path / "export"
    manifest = run_export(src, specs, targets, out, {"seed": 1}, lambda *_: None, family=family)
    return manifest, out


# ------------------------------------------------------------------------------------------- #1 ingest
def test_csv_type_change_after_inference_window_still_gives_valid_parquet(tmp_path):
    src = tmp_path / "late.csv"
    src.write_text("a,b\n" + "".join(f"{i},{i}\n" for i in range(60_000)) + "3.5,abc\n")
    dest = tmp_path / "ds" / "raw.parquet"
    warnings = ingest.to_parquet(src, "csv", dest)
    df = pl.read_parquet(dest)
    assert df.height == 60_001 and df["a"][-1] == 3.5 and df["b"][-1] == "abc"
    assert any("full file" in w for w in warnings)
    assert [p.name for p in dest.parent.iterdir()] == ["raw.parquet"]


def test_rows_longer_than_the_header_are_truncated_with_a_warning(tmp_path):
    src = tmp_path / "ragged.csv"
    src.write_text("a,b,c\n1,2,3\n4,5\n6,7,8,9\n10,11,12\n")
    dest = tmp_path / "ds" / "raw.parquet"
    warnings = ingest.to_parquet(src, "csv", dest)
    df = pl.read_parquet(dest)
    assert df.rows() == [(1, 2, 3), (4, 5, None), (6, 7, 8), (10, 11, 12)]
    assert any("more values than the header" in w for w in warnings)


def test_well_formed_csv_gets_no_ragged_warning(tmp_path):
    src = tmp_path / "ok.csv"
    src.write_text("a,b\n1,2\n3,4\n")
    assert ingest.to_parquet(src, "csv", tmp_path / "ds" / "raw.parquet") == []


@pytest.mark.parametrize("sep,name", [(";", "semicolons"), ("\t", "tabs"), ("|", "pipes")])
def test_csv_separated_by_other_characters_reads_into_columns(tmp_path, sep, name):
    src = tmp_path / "data.csv"
    src.write_text(sep.join(["school", "age", "note"]) + "\n"
                   + "".join(sep.join(['"GP"', str(15 + i % 5), '"a, b"']) + "\n" for i in range(50)))
    dest = tmp_path / "ds" / "raw.parquet"
    warnings = ingest.to_parquet(src, "csv", dest)
    df = pl.read_parquet(dest)
    assert df.columns == ["school", "age", "note"] and df["age"][0] == 15 and df["note"][0] == "a, b"
    assert any(name in w for w in warnings)


def test_quoted_commas_dont_fool_the_separator_guess(tmp_path):
    src = tmp_path / "data.csv"
    src.write_text('name,city\n' + "".join(f'"Smith; J{i}","Paris; FR"\n' for i in range(20)))
    dest = tmp_path / "ds" / "raw.parquet"
    assert ingest.to_parquet(src, "csv", dest) == []
    assert pl.read_parquet(dest).columns == ["name", "city"]


def test_a_lone_question_mark_is_missing(tmp_path):
    src = tmp_path / "data.csv"
    src.write_text("race,weight,note\n" + "Caucasian,?,ok?\n?,[75-100),?\n" * 10)
    dest = tmp_path / "ds" / "raw.parquet"
    ingest.to_parquet(src, "csv", dest)
    df = pl.read_parquet(dest)
    assert df["race"].null_count() == 10 and df["weight"].null_count() == 10 and df["note"].null_count() == 10
    assert df["note"][0] == "ok?"


def test_failed_conversion_leaves_no_partial_file(tmp_path):
    src = tmp_path / "bad.parquet"
    src.write_bytes(b"not a parquet file")
    dest = tmp_path / "ds" / "raw.parquet"
    with pytest.raises(Exception):
        ingest.to_parquet(src, "parquet", dest)
    assert not dest.exists() and list(dest.parent.iterdir()) == []


# ------------------------------------------------------------------------------------------- #2 detection
def test_date_sorted_day_first_column_is_read_day_first():
    dates = [f"{d:02d}/{m:02d}/2024" for m in range(1, 13) for d in range(1, 29) for _ in range(500)]
    p = _profile(dates, "visit_date")
    assert p.inferred_type == "datetime" and p.stats["datetime_format"] == "%d/%m/%Y"
    assert p.summary["null_pct"] == 0
    assert p.stats["min"].startswith("2024-01-01") and p.stats["max"].startswith("2024-12-28")


def test_ambiguous_day_month_order_is_a_stated_assumption():
    t, _, reasons, hints = _infer(["01/02/2024", "03/04/2024", "05/06/2024", "07/08/2024"] * 100)
    assert t == "datetime" and hints["datetime_format"] == "%m/%d/%Y"
    assert any("ambiguous" in r for r in reasons)


def test_day_first_with_time_component():
    fmt, rate = detect_datetime_format(pl.Series(["13/01/2024 10:30", "02/03/2024 08:00"]))
    assert fmt == "%d/%m/%Y %H:%M" and rate == 1.0


def test_unparseable_numbers_count_as_missing_and_lower_confidence():
    vals = [f"{i}.5" for i in range(30_000)] + ["abc"] * 300
    p = _profile(vals, "amount")
    assert p.inferred_type == "numeric_continuous"
    assert p.summary["null_count"] == 300 and p.summary["null_pct"] > 0
    assert p.stats["unparseable_count"] == 300 and p.stats["unparseable_examples"] == ["abc"]
    assert p.confidence < 0.7 and any("don't parse" in r for r in p.reasons)


def test_type_is_decided_on_the_whole_column_not_its_head():
    vals = [f"{i}.5" for i in range(30_000)] + [f"word{i % 40}" for i in range(30_000)]
    assert _infer(vals)[0] == "categorical"


def test_unparseable_dates_count_as_missing():
    vals = [f"2024-01-{1 + i % 28:02d}" for i in range(30_000)] + ["soon"] * 200
    p = _profile(vals, "when")
    assert p.inferred_type == "datetime"
    assert p.stats["unparseable_count"] == 200 and p.summary["null_count"] == 200


# ------------------------------------------------------------------------------------------- #4 / #17 inf
def test_infinite_values_are_excluded_and_counted():
    p = _profile([i / 3 for i in range(1000)] + [INF] * 10, "x")
    assert p.stats["inf_count"] == 10 and p.inferred_type == "numeric_continuous"
    assert p.summary["mean"] == pytest.approx(166.5) and p.summary["std"] is not None
    assert p.summary["null_count"] == 10
    assert all(np.isfinite(p.charts["distribution"]["edges"]))
    assert any("infinite" in r for r in p.reasons)
    few = _profile([i / 3 for i in range(1000)] + [-INF], "x")
    assert few.summary["mean"] == pytest.approx(166.5) and few.stats["inf_count"] == 1
    text = _profile([f"{i / 3:.4f}" for i in range(1000)] + ["inf"] * 4, "x")
    assert text.stats["inf_count"] == 4 and text.summary["null_count"] == 4


def test_coerce_numeric_treats_infinity_as_missing():
    assert coerce_numeric(pl.Series([1.0, INF, -INF, None])).to_list() == [1.0, None, None, None]
    assert coerce_numeric(pl.Series(["2", "inf", "-Infinity"])).to_list() == [2.0, None, None]


@pytest.mark.parametrize("family", ["tree", "linear"])
def test_infinite_values_are_missing_in_exports(tmp_path, family):
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 400)
    x[:7] = INF
    x[7:9] = -INF
    df = pl.DataFrame({"x": x, "y": rng.choice(["no", "yes"], 400)})
    manifest, out = _export(tmp_path, df, [_spec("x", "numeric_continuous", family)], ["y"], family)
    X = pl.read_parquet(out / "model_ready.parquet")["x"].to_numpy()
    assert not np.isinf(X).any()
    assert manifest["inf_values"] == {"x": 9}
    if family == "tree":
        assert manifest["nan_features"] == {"x": 9}


# ------------------------------------------------------------------------------------------- #18 commas
def test_thousands_separators_need_valid_grouping():
    assert coerce_numeric(pl.Series(["1,234.5", "12,345", "-1,000,000", "7"])).to_list() == [
        1234.5, 12345.0, -1_000_000.0, 7.0]
    assert coerce_numeric(pl.Series(["1,5", "1,234", "12,34,5"]), decimal_comma=False).to_list() == [
        None, 1234.0, None]


def test_decimal_comma_column_is_parsed_as_decimals():
    assert coerce_numeric(pl.Series(["1,5", "2,25", "10", "1,500"])).to_list() == [1.5, 2.25, 10.0, 1.5]
    assert coerce_numeric(pl.Series(["1,5", "2.5"])).to_list() == [None, 2.5]  # a dot decimal rules it out
    p = _profile([f"{i},{i % 10}" for i in range(2000)], "weight")
    assert p.inferred_type == "numeric_continuous" and p.stats["decimal_comma"] is True
    assert p.summary["max_value"] == pytest.approx(1999.9)


def test_export_uses_the_profiled_number_format():
    s = pl.Series("w", ["1,500", "2,25"])
    assert coercion_exprs([_spec("w", "numeric_continuous", decimal_comma=True)])["w"](s).to_list() == [1.5, 2.25]
    assert coercion_exprs([_spec("w", "numeric_continuous")])["w"](s).to_list() == [1500.0, None]


# ------------------------------------------------------------------------------------------- #19 dates
def _days(n):
    start = dt.date(2019, 3, 1)
    return [start + dt.timedelta(days=i) for i in range(n)]


def test_yyyymmdd_integers_and_strings_are_dates():
    ints = [int(d.strftime("%Y%m%d")) for d in _days(2000)]
    for values in (ints, [str(v) for v in ints]):
        p = _profile(values, "event_date")
        assert p.inferred_type == "datetime" and p.stats["datetime_format"] == "%Y%m%d"
        assert p.stats["min"].startswith("2019-03-01")


def test_yyyymmdd_guard_for_id_like_names():
    ints = [int(d.strftime("%Y%m%d")) for d in _days(2000)]
    assert _infer(ints, "order_id")[0] == "identifier"


def test_numbers_retyped_to_datetime_are_parsed_by_range():
    def iso(values, fmt=None):
        return [v.isoformat()[:10] if v else None for v in parse_datetime(pl.Series(values), fmt).to_list()]

    assert iso([20240115, 20231231, None]) == ["2024-01-15", "2023-12-31", None]
    assert iso([20240115.0], "%Y%m%d") == ["2024-01-15"]
    assert iso([1_700_000_000, 1_600_000_000]) == ["2023-11-14", "2020-09-13"]
    assert iso([1_700_000_000_000]) == ["2023-11-14"]
    assert iso(["2024-01-05", "2024-02-06"]) == ["2024-01-05", "2024-02-06"]  # no stored format


# ------------------------------------------------------------------------------------------- #11 names
def test_a_stored_epoch_unit_reads_every_chunk_the_same_way():
    from app.profiling import column_datetime_format

    whole = pl.Series("t", [5_000_000_000, 1_600_000_000_000])  # ms; the first alone looks like seconds
    fmt = column_datetime_format(whole)
    assert fmt == "epoch_ms"
    first_chunk = parse_datetime(whole.head(1), fmt)
    assert first_chunk.dt.year().to_list() == [1970]  # 5e9 ms, not 5e9 s (year 2128)
    assert column_datetime_format(pl.Series("d", ["01/02/2023", "25/12/2023"])) == "%d/%m/%Y"


@pytest.mark.parametrize("scale,unit", [(1, "seconds"), (1000, "milliseconds")])
def test_unique_epoch_integers_are_dates_not_identifiers(scale, unit):
    start = int(dt.datetime(2021, 3, 1, tzinfo=dt.timezone.utc).timestamp())
    values = [(start + i * 3600) * scale for i in range(2000)]
    t, conf, reasons, _ = _infer(values, "created")
    assert t == "datetime" and unit in reasons[0]
    p = _profile(values, "created")
    assert p.inferred_type == "datetime" and p.stats["min"].startswith("2021-03-01")
    assert p.summary["null_count"] == 0


def test_epoch_like_values_under_a_key_name_stay_identifiers():
    values = list(range(1_600_000_000, 1_600_002_000))
    assert _infer(values, "event_id")[0] == "identifier"
    assert _infer([v % 7 for v in values], "bucket")[0] != "datetime"


def test_safe_feature_names_reserve_targets_and_internal_columns():
    assert safe_feature_names(["column_a", "x", "column_a"], reserved=["column_a", "__row_id"]) == [
        "column_a_2", "x", "column_a_3"]


def test_feature_named_like_a_target_never_replaces_the_target(tmp_path):
    rng = np.random.default_rng(3)
    n = 600
    label = rng.choice(["no", "yes"], n)
    df = pl.DataFrame({"column": rng.choice(["a", "b", "c"], n), "x": rng.normal(0, 1, n), "column_a": label})
    specs = [_spec("column", "categorical"), _spec("x", "numeric_continuous")]
    specs[0].encoding, specs[0].params = "onehot", {"max_categories": 30, "min_frequency": 0.001}
    manifest, out = _export(tmp_path, df, specs, ["column_a"])
    table = pq.read_table(out / "model_ready.parquet")
    assert len(set(table.schema.names)) == len(table.schema.names)
    assert "column_a" not in manifest["feature_names"]
    got = table.column("column_a").to_numpy()
    assert (got == (label == "yes").astype(int)).all()


# ------------------------------------------------------------------------------------------- #12 rare classes
def _targets(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    import pandas as pd

    return pd.DataFrame({"a": rng.choice(["n", "y"], n, p=[0.8, 0.2]), "b": rng.choice(["p", "q", "r"], n)})


def test_split_stays_stratified_when_one_combination_is_a_singleton():
    y = _targets()
    y.loc[0, "b"] = "rare"
    y.loc[1, "b"] = "rare"
    y.loc[1, "a"] = "y" if y.loc[0, "a"] == "n" else "n"  # two 'rare' rows, but each combination is a singleton
    codes, info = assign_splits(y, 0.15, 0.15, 0)
    assert info["stratified"] and info["stratified_on"] == ["a", "b"]
    train_share = (y["a"][codes == 0] == "y").mean()
    assert abs(train_share - (y["a"] == "y").mean()) < 0.01


def test_split_falls_back_to_one_target_when_combinations_cannot_be_stratified():
    import pandas as pd

    rng = np.random.default_rng(1)
    n = 60
    y = pd.DataFrame({"a": rng.choice(["n", "y"], n), "b": [f"c{i % 25}" for i in range(n)]})
    codes, info = assign_splits(y, 0.1, 0.1, 0)
    assert info["stratified_on"] == ["a"]
    assert set(codes) == {0, 1, 2}


def test_export_rejects_a_target_class_with_a_single_row(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.choice(["a", "b"], 300).astype(object)
    y[5] = "typo"
    df = pl.DataFrame({"x": rng.normal(0, 1, 300), "t": y})
    with pytest.raises(ValueError, match=r"'t'.*'typo'"):
        _export(tmp_path, df, [_spec("x", "numeric_continuous")], ["t"])


def _handmade_export(tmp_path, t2_train, t2_test, t1_train=None):
    rng = np.random.default_rng(0)
    n_tr, n_te = len(t2_train), len(t2_test)
    n = n_tr + n_te
    t2 = np.concatenate([t2_train, t2_test]).astype(np.int32)
    t1 = rng.integers(0, 2, n).astype(np.int32) if t1_train is None else np.concatenate(
        [t1_train, rng.integers(0, 2, n_te)]).astype(np.int32)
    feats = {f"f{k}": (rng.normal(0, 1, n) + (t2 == k % 3)).astype(np.float32) for k in range(4)}
    df = pl.DataFrame({"__row_id": np.arange(n), "__split": ["train"] * n_tr + ["test"] * n_te, **feats,
                       "t1": t1, "t2": t2})
    df.write_parquet(tmp_path / "model_ready.parquet")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "model_family": "tree", "contains_nan": False, "targets": ["t1", "t2"], "feature_names": list(feats),
        "split_column": "__split", "target_kinds": {"t1": "binary", "t2": "multiclass"},
        "target_classes": {"t1": ["n", "y"], "t2": ["a", "b", "c"]},
        "feature_source_field": {f: f for f in feats}, "files": {"data": "model_ready.parquet"},
    }))
    return tmp_path


def test_training_aligns_probabilities_when_a_class_is_missing_from_train(tmp_path):
    rng = np.random.default_rng(2)
    export = _handmade_export(tmp_path, rng.integers(0, 2, 400), rng.integers(0, 3, 150))
    report = train(export, "lightgbm", seed=0, params={"n_estimators": 20}, cv=3, out_dir=tmp_path / "m")
    ev = report["evaluation"]["test"]["t2"]
    assert ev["labels"] == ["a", "b", "c"] and len(ev["confusion_matrix"]) == 3
    support = {c["label"]: c["support"] for c in ev["per_class"]}
    assert support["c"] > 0 and support["c"] == sum(ev["confusion_matrix"][2])
    assert report["missing_from_train"] == {"t2": ["c"]}


def test_training_fails_early_when_a_target_has_one_class_in_train(tmp_path):
    rng = np.random.default_rng(2)
    export = _handmade_export(tmp_path, rng.integers(0, 3, 300), rng.integers(0, 3, 100), t1_train=np.zeros(300))
    with pytest.raises(ValueError, match=r"'t1'.*only one class"):
        train(export, "lightgbm", params={"n_estimators": 5}, cv=None, out_dir=tmp_path / "m")


# ------------------------------------------------------------------------------------------- #16 targets
@pytest.mark.parametrize("t,levels,task,ok", [
    ("binary", 2, None, True), ("categorical", 40, None, True), ("numeric_discrete", 7, None, True),
    ("categorical", MAX_TARGET_LEVELS + 1, None, False), ("numeric_continuous", 30, None, True),
    ("numeric_continuous", 5000, "regression", True), ("numeric_continuous", 30, "classes", False),
    ("numeric_discrete", 500, "classes", False), ("numeric_discrete", 500, "bins", True),
    ("categorical", 10, "regression", False), ("binary", 2, "bins", False),
    ("text", 10, None, False), ("datetime", 10, None, False), ("identifier", 10, None, False),
])
def test_target_eligibility_depends_on_type_and_task(t, levels, task, ok):
    errors = validate_decision({"action": "target", "semantic_type": t, "target_task": task,
                                "target_bins": {"k": 5} if task == "bins" else None}, n_unique=levels)
    assert (errors == []) is ok


@pytest.mark.parametrize("t,levels,task", [
    ("binary", 2, "classes"), ("categorical", 40, "classes"), ("numeric_discrete", 100, "classes"),
    ("numeric_discrete", 101, "regression"), ("numeric_continuous", 3, "regression"), ("text", 5, None),
])
def test_default_task(t, levels, task):
    assert default_task(t, levels) == task


@pytest.mark.parametrize("bins,ok", [
    ({"k": 5}, True), ({"k": 2}, True), ({"k": 20}, True), ({"k": 1}, False), ({"k": 21}, False),
    ({"k": 2.5}, False), ({"edges": [10]}, True), ({"edges": [10, 25.5, 80]}, True), ({"edges": []}, False),
    ({"edges": [3, 3]}, False), ({"edges": [5, 1]}, False), ({"edges": [1, float("inf")]}, False),
    ({"edges": list(range(20))}, False), ({"k": 5, "edges": [1]}, False), ({}, False), (None, False),
])
def test_bins_problem(bins, ok):
    assert (bins_problem(bins) is None) is ok


def test_upload_targets_keep_their_numeric_type():
    price = _profile([i / 3 for i in range(5000)], "price", is_target=True)
    assert price.is_target and price.inferred_type == "numeric_continuous"
    grade = _profile([i % 4 for i in range(5000)], "grade", is_target=True)
    assert grade.is_target and grade.inferred_type in ("numeric_discrete", "categorical")
    ids = _profile([f"id{i}" for i in range(5000)], "code", is_target=True)
    assert not ids.is_target and any("can't be a target" in r for r in ids.reasons)


def test_whole_number_target_that_looks_like_a_key_is_a_number():
    p = _profile(list(range(1000, 6000)), "price", is_target=True)
    assert p.is_target and p.inferred_type == "numeric_continuous"


# ------------------------------------------------------------------------------------------- API (needs DB)
DB_URL = os.environ.get("RAINFOREST_TEST_DATABASE_URL")


@pytest.fixture(scope="module")
def client(tmp_path_factory, login_as):
    if not DB_URL:
        pytest.skip("set RAINFOREST_TEST_DATABASE_URL to run the API tests")
    from fastapi.testclient import TestClient

    from app import db
    from app.config import get_settings

    mp = pytest.MonkeyPatch()
    mp.setenv("DATABASE_URL", DB_URL)
    mp.setenv("DATA_DIR", str(tmp_path_factory.mktemp("data")))
    get_settings.cache_clear()
    db.close_pool()
    from app.main import app

    with TestClient(app) as c:
        yield login_as(c)
    db.close_pool()
    mp.undo()
    get_settings.cache_clear()


def test_api_target_rules_follow_the_task(client):
    rng = np.random.default_rng(0)
    rows = "".join(f"{rng.normal():.6f},{'abc'[i % 3]},{i % 2},{rng.integers(0, 100)}\n" for i in range(800))
    ds = client.post("/api/datasets/upload", files={"file": ("t.csv", "cont,cat,y,score\n" + rows, "text/csv")},
                     data={"targets": "y"}).json()
    deadline = time.monotonic() + 120
    while (ds := client.get(f"/api/datasets/{ds['id']}").json())["status"] not in ("ready", "failed"):
        assert time.monotonic() < deadline
        time.sleep(0.3)
    assert ds["status"] == "ready", ds
    try:
        fields = {f["name"]: f for f in client.get(f"/api/datasets/{ds['id']}/fields").json()}
        assert client.patch(f"/api/datasets/{ds['id']}", json={"target_columns": ["y", "cont"]}).status_code == 200
        cont = fields["cont"]
        r = client.put(f"/api/fields/{cont['id']}/decision", json={
            **{k: cont[k] for k in ("semantic_type", "reviewed")}, "action": "target", "notes": None,
            "imputation": "none", "impute_value": None, "add_missing_indicator": False, "transform": "none",
            "scaling": "none", "encoding": "none", "params": {}, "target_task": "classes"})
        assert r.status_code == 422 and "continuous" in json.dumps(r.json())
        assert client.patch(f"/api/datasets/{ds['id']}", json={"target_columns": ["y", "cat"]}).status_code == 200
        assert client.get(f"/api/datasets/{ds['id']}").json()["target_columns"] == ["y", "cat"]
    finally:
        client.delete(f"/api/datasets/{ds['id']}")
