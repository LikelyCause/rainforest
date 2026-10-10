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


@pytest.mark.parametrize("model", ["lightgbm", "xgboost", "hist_gradient_boosting", "random_forest"])
def test_tree_models_train_on_tree_export(tree_export, tmp_path, model):
    fast = {"lightgbm": {"n_estimators": 60}, "xgboost": {"n_estimators": 60},
            "hist_gradient_boosting": {"max_iter": 60}, "random_forest": {"n_estimators": 60}}[model]
    report = train(tree_export, model, seed=0, params=fast, cv=3, out_dir=tmp_path)
    m = report["metrics"]["test"]
    assert m["column_a"]["roc_auc"] > 0.75
    assert m["column_b"]["accuracy"] > 0.5  # 3 classes -> chance ~0.33-0.4
    assert report["chain_method"] == "predict"
    assert report["device"] == "cpu"
    assert report["params"] == {**MODELS[model].default_params, **fast}


def test_cuda_run_that_falls_back_to_cpu_fails(tree_export, tmp_path):
    """No GPU here, so XGBoost silently trains on the CPU; the run must not be recorded as a GPU run."""
    with pytest.raises(RuntimeError, match="GPU training was requested but XGBoost ran on cpu"):
        train(tree_export, "xgboost", seed=0, params={"n_estimators": 5}, cv=None, out_dir=tmp_path, device="cuda")


def test_linear_model_needs_linear_export(tree_export, linear_export, tmp_path):
    with pytest.raises(ValueError, match="missing values"):
        train(tree_export, "logistic_regression", out_dir=tmp_path)
    report = train(linear_export, "logistic_regression", seed=0, cv=3, out_dir=tmp_path)
    assert report["metrics"]["test"]["column_a"]["roc_auc"] > 0.75


@pytest.mark.parametrize("kind,method,auc", [("binary", "predict_proba", "roc_auc"),
                                             ("multiclass", "predict", "roc_auc_ovr_macro")])
def test_chain_for_each_target_kind(tmp_path, kind, method, auc):
    from app.sample_data import generate

    src = tmp_path / "raw.parquet"
    generate(n_rows=3000, n_fields=40, seed=3, target_kind=kind).write_parquet(src)
    profiles = profiling.profile_parquet(src, ["column_a", "column_b"], 3000, 3000, 50)[1]
    export = _export(src, profiles, "tree", tmp_path / "export")
    manifest = json.loads((export / "manifest.json").read_text())
    expected = "binary" if kind == "binary" else "multiclass"
    assert manifest["target_kinds"] == {"column_a": expected, "column_b": expected}
    report = train(export, "lightgbm", seed=0, params={"n_estimators": 40}, cv=3, out_dir=tmp_path / "m")
    assert report["chain_method"] == method
    for t in ("column_a", "column_b"):
        assert report["metrics"]["test"][t][auc] > 0.6


def test_sample_leak_clears_the_threshold_with_a_margin(profiles):
    from app.correlations import compute_associations
    from app.processing import LEAKAGE_THRESHOLD

    names = [p.name for p in profiles]
    leak, target = names.index("leaky_outcome_flag"), names.index("column_a")
    assocs = compute_associations([(i, p.corr_numeric) for i, p in enumerate(profiles) if p.corr_numeric is not None],
                                  [(i, p.corr_codes) for i, p in enumerate(profiles) if p.corr_codes is not None])
    (v,) = [a.value for a in assocs if {a.i, a.j} == {leak, target}]
    assert v >= LEAKAGE_THRESHOLD + 0.02


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


def test_evaluate_reports_precision_recall_f1_and_confusion():
    from app.modeling.train import evaluate

    y = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    proba = np.array([[.9, .1], [.8, .2], [.4, .6], [.3, .7], [.2, .8], [.6, .4], [.1, .9], [.7, .3]])
    m, d = evaluate(y, proba, ["no", "yes"])
    # predictions: 0 0 1 1 1 0 1 0 -> TP=3 FP=1 FN=1 TN=3
    assert d["confusion_matrix"] == [[3, 1], [1, 3]]
    assert m["precision"] == 0.75 and m["recall"] == 0.75 and m["specificity"] == 0.75
    assert abs(m["f1"] - 0.75) < 1e-12 and abs(m["f1_macro"] - 0.75) < 1e-12
    assert [c["label"] for c in d["per_class"]] == ["no", "yes"] and d["positive_label"] == "yes"
    assert sum(c["support"] for c in d["per_class"]) == 8
    (roc,) = d["roc"]  # binary: the positive class only
    assert roc["label"] == "yes" and abs(roc["auc"] - m["roc_auc"]) < 1e-12
    assert roc["fpr"][0] == 0 and roc["tpr"][0] == 0 and roc["fpr"][-1] == 1 and roc["tpr"][-1] == 1
    assert roc["fpr"] == sorted(roc["fpr"]) and roc["tpr"] == sorted(roc["tpr"])

    y3 = np.array([0, 1, 2, 2])
    p3 = np.array([[.8, .1, .1], [.2, .7, .1], [.1, .2, .7], [.5, .3, .2]])
    m3, d3 = evaluate(y3, p3, ["a", "b", "c"])
    assert d3["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [1, 0, 1]]
    assert "precision" not in m3 and {"precision_macro", "recall_weighted", "f1_macro"} <= m3.keys()
    assert all(isinstance(v, float) for v in m3.values())  # scalars only -> safe for v_model_metrics
    assert [c["label"] for c in d3["roc"]] == ["a", "b", "c"]  # one-vs-rest

    from app.modeling.train import ROC_MAX_POINTS, roc_curves
    rng = np.random.default_rng(0)
    yb = rng.integers(0, 2, 5000)
    pb = np.clip(yb * 0.3 + rng.random(5000) * 0.7, 0, 1)
    (big,) = roc_curves(yb, np.column_stack([1 - pb, pb]), ["n", "y"])
    assert len(big["fpr"]) <= ROC_MAX_POINTS and big["fpr"][-1] == 1 and big["tpr"][-1] == 1
    assert roc_curves(np.zeros(4, dtype=int), np.full((4, 2), 0.5), ["n", "y"]) == []  # no positives: no curve
