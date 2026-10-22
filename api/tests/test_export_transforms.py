"""Export preprocessing: step names, binary encoding, text with a tiny vocabulary, imputation order, feature names."""

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest

from app.semantics import suggest
from app.transform import (
    FieldSpec,
    TextFeatures,
    build_preprocessor,
    coercion_exprs,
    readable_feature_names,
    run_export,
    to_pandas_frame,
)


def _spec(name, t, family="tree", **kw):
    s = suggest(t, {}, False, family)
    base = dict(
        imputation=s["imputation"],
        impute_value=s.get("impute_value"),
        add_missing_indicator=s["add_missing_indicator"],
        transform=s["transform"],
        scaling=s["scaling"],
        encoding=s["encoding"],
        params=s["params"],
    )
    return FieldSpec(name, t, **{**base, **kw})


def _fit(df: pl.DataFrame, specs):
    frame = to_pandas_frame(df, specs, coercion_exprs(specs))
    pre = build_preprocessor(specs).fit(frame)
    return pre, pre.transform(frame), readable_feature_names(pre)


def _export(tmp_path, df, specs, targets, family="tree"):
    src = tmp_path / "raw.parquet"
    df.write_parquet(src)
    out = tmp_path / "export"
    return run_export(src, specs, targets, out, {"seed": 1}, lambda *_: None, family=family), out


@pytest.mark.parametrize("name", ["remainder", "verbose", "n_jobs", "transformers", "sparse_threshold"])
def test_fields_named_like_column_transformer_arguments_export(tmp_path, name):
    rng = np.random.default_rng(0)
    n = 400
    df = pl.DataFrame({name: rng.normal(0, 1, n), "y": rng.choice(["a", "b"], n)})
    manifest, out = _export(tmp_path, df, [_spec(name, "numeric_continuous")], ["y"])
    assert manifest["feature_names"] == [name]
    assert manifest["feature_source_field"] == {name: name}


def _binary_frame(n=300, missing=30, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.choice(["no", "yes"], n).astype(object)
    v[:missing] = None
    return pl.DataFrame({"smoker": list(v)}), v


def test_binary_with_a_missing_label_is_0_1_plus_an_indicator():
    df, v = _binary_frame()
    spec = _spec("smoker", "binary", imputation="constant", impute_value="__missing__")
    _, X, names = _fit(df, [spec])
    assert names == ["smoker_yes", "smoker__isna"]
    present = v != None  # noqa: E711
    assert set(np.unique(X[present, 0])) == {0.0, 1.0}
    assert (X[present, 0] == (v[present] == "yes")).all()
    assert (X[:, 1] == ~present).all() and (X[~present, 0] == 0).all()


def test_binary_most_frequent_is_one_0_1_column():
    df, v = _binary_frame()
    _, X, names = _fit(df, [_spec("smoker", "binary", imputation="most_frequent")])
    assert names == ["smoker_yes"] and set(np.unique(X)) == {0.0, 1.0}
    top = 1.0 if (v == "yes").sum() >= (v == "no").sum() else 0.0
    assert (X[:30, 0] == top).all()


def test_binary_filled_with_one_of_its_levels_needs_no_indicator():
    df, v = _binary_frame()
    _, X, names = _fit(df, [_spec("smoker", "binary", imputation="constant", impute_value="no")])
    assert names == ["smoker_yes"]
    assert (X[:30, 0] == 0).all() and set(np.unique(X)) == {0.0, 1.0}


def test_binary_export_is_complete_for_linear_models(tmp_path):
    df, _ = _binary_frame()
    df = df.with_columns(pl.Series("y", np.random.default_rng(1).choice(["a", "b"], df.height)))
    spec = _spec("smoker", "binary", family="linear", imputation="constant", impute_value="__missing__")
    manifest, _ = _export(tmp_path, df, [spec], ["y"], family="linear")
    assert not manifest["contains_nan"] and manifest["feature_names"] == ["smoker_yes", "smoker__isna"]


@pytest.mark.parametrize("docs", [["alpha"] * 50, ["alpha beta"] * 50, ["", "alpha"] * 25])
def test_text_with_a_one_term_vocabulary_exports(docs):
    tf = TextFeatures(min_df=1).fit(np.asarray(docs, dtype=object))
    out = tf.transform(np.asarray(docs, dtype=object))
    assert out.shape == (len(docs), tf.n_out_) and np.isfinite(out).all()
    assert len(tf.get_feature_names_out()) == tf.n_out_


def test_constant_fill_is_in_the_fields_own_units():
    vals = [1.0, 10.0, 100.0, None, 1000.0, None]
    df = pl.DataFrame({"x": vals})
    spec = _spec(
        "x",
        "numeric_continuous",
        family="linear",
        imputation="constant",
        impute_value="100",
        transform="log1p",
        scaling="none",
    )
    _, X, _ = _fit(df, [spec])
    assert X[3, 0] == pytest.approx(np.log1p(100.0)) == pytest.approx(X[2, 0])


@pytest.mark.parametrize("name", ["s", "sv", "n", "svd"])
def test_short_text_field_names_keep_their_prefix(name):
    rng = np.random.default_rng(0)
    words = np.array("red green blue cyan magenta yellow black white".split())
    docs = [" ".join(rng.choice(words, 6)) for _ in range(200)]
    df = pl.DataFrame({name: docs})
    _, _, names = _fit(df, [_spec(name, "text", params={"svd_components": 3, "max_features": 100, "min_df": 1})])
    assert names == [f"{name}__svd{i:02d}" for i in range(3)]


def test_feature_names_starting_with_the_field_name_are_kept():
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"colour": rng.choice(["red", "green", "blue"], 200)})
    _, _, names = _fit(df, [_spec("colour", "categorical", encoding="onehot")])
    assert sorted(names) == ["colour_blue", "colour_green", "colour_red"]


def test_export_reads_back(tmp_path):
    df, _ = _binary_frame()
    df = df.with_columns(pl.Series("y", np.random.default_rng(1).choice(["a", "b"], df.height)))
    _, out = _export(tmp_path, df, [_spec("smoker", "binary", imputation="most_frequent")], ["y"])
    assert pq.read_table(out / "model_ready.parquet").num_rows == df.height


# ------------------------------------------------------------------------------------------- numeric targets
def test_export_writes_regression_and_band_targets(tmp_path):
    n = 600
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"x": rng.normal(size=n), "price": rng.gamma(2.0, 50.0, size=n), "kind": ["a", "b"] * (n // 2)})
    src = tmp_path / "d.parquet"
    df.write_parquet(src)
    specs = [_spec("x", "numeric_continuous")]
    m = run_export(
        src,
        specs,
        ["price", "kind"],
        tmp_path / "out",
        {"seed": 1},
        lambda *_: None,
        target_tasks={"price": {"task": "bins", "bins": {"k": 4}}, "kind": {"task": "classes", "bins": None}},
    )
    assert m["target_tasks"] == {"price": "bins", "kind": "classes"}
    assert len(m["target_bins"]["price"]["edges"]) == 3 and m["target_classes"]["price"][0].startswith("< ")
    out = pl.read_parquet(tmp_path / "out" / "model_ready.parquet")
    train = out.filter(pl.col("__split") == "train")["price"].to_numpy()
    assert set(np.unique(train)) == {0, 1, 2, 3}
    # Edges come from training rows only: the training split's bands are near-equal.
    counts = np.bincount(train)
    assert counts.max() - counts.min() <= 0.05 * len(train) + 2

    m2 = run_export(
        src,
        specs,
        ["price"],
        tmp_path / "out2",
        {"seed": 1},
        lambda *_: None,
        target_tasks={"price": {"task": "regression", "bins": None}},
    )
    assert m2["target_kinds"]["price"] == "regression" and "price" not in m2["target_classes"]
    assert m2["target_stats"]["price"]["std"] > 0
    out2 = pl.read_parquet(tmp_path / "out2" / "model_ready.parquet")
    assert out2.schema["price"] == pl.Float64
    assert m2["split_info"]["stratified"] is True


def test_export_refuses_a_single_band(tmp_path):
    df = pl.DataFrame({"x": np.arange(300.0), "y": np.full(300, 7.0)})
    src = tmp_path / "d.parquet"
    df.write_parquet(src)
    with pytest.raises(ValueError, match="one band"):
        run_export(
            src,
            [_spec("x", "numeric_continuous")],
            ["y"],
            tmp_path / "o",
            {"seed": 1},
            lambda *_: None,
            target_tasks={"y": {"task": "bins", "bins": {"k": 3}}},
        )


def test_export_refuses_an_empty_band(tmp_path):
    df = pl.DataFrame({"x": np.arange(300.0), "y": np.r_[np.zeros(150), np.full(150, 100.0)]})
    df.write_parquet(tmp_path / "d.parquet")
    with pytest.raises(ValueError, match=r"\[10, 20\)"):
        run_export(
            tmp_path / "d.parquet",
            [_spec("x", "numeric_continuous")],
            ["y"],
            tmp_path / "o",
            {"seed": 1},
            lambda *_: None,
            target_tasks={"y": {"task": "bins", "bins": {"edges": [10, 20]}}},
        )


def test_numeric_target_stored_as_text_is_read_like_profiling(tmp_path):
    df = pl.DataFrame({"x": np.arange(300.0), "price": [f"{1000 + i:,}.5" for i in range(300)]})
    df.write_parquet(tmp_path / "d.parquet")
    m = run_export(
        tmp_path / "d.parquet",
        [_spec("x", "numeric_continuous")],
        ["price"],
        tmp_path / "o",
        {"seed": 1},
        lambda *_: None,
        target_tasks={"price": {"task": "regression", "bins": None}},
    )
    assert m["rows_written"] == 300 and m["target_stats"]["price"]["min"] >= 1000.5


def test_numeric_target_with_unreadable_values_is_refused(tmp_path):
    df = pl.DataFrame({"x": np.arange(300.0), "price": [str(i) for i in range(299)] + ["n/a?"]})
    df.write_parquet(tmp_path / "d.parquet")
    with pytest.raises(ValueError, match="don't read as numbers"):
        run_export(
            tmp_path / "d.parquet",
            [_spec("x", "numeric_continuous")],
            ["price"],
            tmp_path / "o",
            {"seed": 1},
            lambda *_: None,
            target_tasks={"price": {"task": "regression", "bins": None}},
        )


def test_prepare_export_keeps_regression_target_numeric(tmp_path):
    from app.analysis.prep import prepare_export

    n = 400
    df = pl.DataFrame({"x": np.arange(n, dtype=float), "price": np.arange(n) * 2.5 + 0.3})
    src = tmp_path / "d.parquet"
    df.write_parquet(src)
    m = run_export(
        src,
        [_spec("x", "numeric_continuous")],
        ["price"],
        tmp_path / "o",
        {"seed": 1},
        lambda *_: None,
        target_tasks={"price": {"task": "regression", "bins": None}},
    )
    p = prepare_export(tmp_path / "o", m, "price", 200)
    assert p.target_kind == "numeric" and np.any(p.y % 1 != 0)


def test_yeo_johnson_runs_after_imputation_and_before_scaling():
    from sklearn.preprocessing import PowerTransformer, StandardScaler

    rng = np.random.default_rng(0)
    x = rng.lognormal(0, 1, 300)
    x[::10] = np.nan
    spec = _spec("x", "numeric_continuous", family="linear", transform="yeo_johnson", scaling="standard")
    pre, out, names = _fit(pl.DataFrame({"x": x}), [spec])
    steps = pre.named_transformers_["x"].named_steps
    assert list(steps) == ["impute", "transform", "scale"]
    assert isinstance(steps["transform"], PowerTransformer) and steps["transform"].standardize is False
    filled = np.where(np.isnan(x), np.nanmedian(x), x).reshape(-1, 1)
    ref = StandardScaler().fit_transform(
        PowerTransformer(method="yeo-johnson", standardize=False).fit_transform(filled)
    )
    assert names == ["x"] and np.allclose(out, ref)


def test_a_linear_export_with_yeo_johnson(tmp_path):
    rng = np.random.default_rng(1)
    n = 400
    df = pl.DataFrame({"x": rng.lognormal(0, 1, n), "y": rng.choice(["a", "b"], n)})
    spec = _spec("x", "numeric_continuous", family="linear", transform="yeo_johnson")
    manifest, out = _export(tmp_path, df, [spec], ["y"], family="linear")
    assert manifest["feature_names"] == ["x"] and manifest["feature_source_field"] == {"x": "x"}
    col = pq.read_table(out / "model_ready.parquet", columns=["x"]).column("x").to_numpy()
    assert np.isfinite(col).all()


def test_an_unknown_transform_or_encoding_fails_loudly():
    with pytest.raises(ValueError, match="'acme.gone' isn't available"):
        build_preprocessor([_spec("x", "numeric_continuous", family="linear", transform="acme.gone")])
    with pytest.raises(ValueError, match="'acme.gone' isn't available"):
        build_preprocessor([_spec("c", "categorical", encoding="acme.gone")])
