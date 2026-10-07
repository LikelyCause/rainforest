"""End-to-end: profile -> suggested settings per family -> export -> ClassifierChain training (no database)."""
import json

import numpy as np
import polars as pl
import pytest

from app import profiling
from app.modeling.registry import MODELS, compatibility
from app.modeling.train import chain_predict_proba, load_export, train
from app.semantics import suggest
from app.transform import FieldSpec, run_export


@pytest.fixture(scope="module")
def profiles(sample_parquet):
    return profiling.profile_parquet(sample_parquet, ["column_a", "column_b"], 4000, 2000, 50)[1]


def _export(sample_parquet, profiles, family, out):
    specs, targets = [], []
    for p in profiles:
        s = suggest(p.inferred_type, p.summary, p.is_target, family)
        if s["action"] == "target":
            targets.append(p.name)
        elif s["action"] == "keep" and p.name != "leaky_outcome_flag":
            specs.append(FieldSpec(p.name, s["semantic_type"], s["imputation"], s.get("impute_value"),
                                   s["add_missing_indicator"], s["transform"], s["scaling"], s["encoding"],
                                   s["params"], p.stats.get("datetime_format")))
    run_export(sample_parquet, specs, targets, out, {"seed": 1, "chunk_rows": 1500}, lambda *_: None, family=family)
    return out


@pytest.fixture(scope="module")
def tree_export(sample_parquet, profiles, tmp_path_factory):
    return _export(sample_parquet, profiles, "tree", tmp_path_factory.mktemp("tree"))


@pytest.fixture(scope="module")
def linear_export(sample_parquet, profiles, tmp_path_factory):
    return _export(sample_parquet, profiles, "linear", tmp_path_factory.mktemp("linear"))


def test_export_artifacts(tree_export):
    manifest = json.loads((tree_export / "manifest.json").read_text())
    df = pl.read_parquet(tree_export / "model_ready.parquet")
    assert df.height == manifest["rows_written"] == 4000
    assert set(df["__split"].unique()) == {"train", "valid", "test"}
    assert len(manifest["feature_names"]) == len(set(manifest["feature_names"]))
    assert manifest["target_kinds"] == {"column_a": "binary", "column_b": "multiclass"}
    assert manifest["model_family"] == "tree"
    assert manifest["contains_nan"]  # tree family leaves missing values for the model
    assert set(manifest["feature_source_field"].values()) <= {f["name"] for f in manifest["fields"]}


def test_linear_export_is_complete_and_scaled(linear_export):
    manifest, df = load_export(linear_export)
    X = df.filter(pl.col("__split") == "train").select(manifest["feature_names"]).to_numpy()
    assert not manifest["contains_nan"] and not np.isnan(X).any()
    numeric = [n for n in manifest["feature_names"] if n.startswith("cont_")]
    assert abs(df.filter(pl.col("__split") == "train")[numeric[0]].mean()) < 0.1


@pytest.mark.parametrize("model", ["lightgbm", "hist_gradient_boosting", "random_forest"])
def test_tree_models_train_on_tree_export(tree_export, tmp_path, model):
    fast = {"lightgbm": {"n_estimators": 60}, "hist_gradient_boosting": {"max_iter": 60},
            "random_forest": {"n_estimators": 60}}[model]
    report = train(tree_export, model, seed=0, params=fast, cv=3, out_dir=tmp_path)
    m = report["metrics"]["test"]
    assert m["column_a"]["roc_auc"] > 0.75
    assert m["column_b"]["accuracy"] > 0.5  # 3 classes -> chance ~0.33-0.4
    assert report["chain_method"] == "predict"


def test_linear_model_needs_linear_export(tree_export, linear_export, tmp_path):
    with pytest.raises(ValueError, match="missing values"):
        train(tree_export, "logistic_regression", out_dir=tmp_path)
    report = train(linear_export, "logistic_regression", seed=0, cv=3, out_dir=tmp_path)
    assert report["metrics"]["test"]["column_a"]["roc_auc"] > 0.75


def test_compatibility_rules():
    assert compatibility(MODELS["lightgbm"], "linear", False) is None  # trees take any export
    assert compatibility(MODELS["mlp"], "linear", False)  # wrong family
    assert compatibility(MODELS["logistic_regression"], "linear", True)  # NaN present


def test_manual_chain_walk_matches_sklearn(tree_export, tmp_path):
    import joblib
    from sklearn import config_context

    train(tree_export, "lightgbm", seed=0, params={"n_estimators": 30}, cv=None, out_dir=tmp_path)
    chain = joblib.load(tmp_path / "classifier_chain.joblib")
    manifest, df = load_export(tree_export)
    X = df.filter(pl.col("__split") == "test").select(manifest["feature_names"]).to_numpy().astype(np.float32)
    with config_context(assume_finite=True):
        probas = chain_predict_proba(chain, X)
    assert np.array_equal(np.column_stack([p.argmax(1) for p in probas]), chain.predict(X).astype(int))
