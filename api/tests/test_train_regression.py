import json

import numpy as np
import polars as pl
import pytest

from app.modeling.train import TargetChain, evaluate_regression, train
from app.transform import run_export

from .test_export_transforms import _spec


def test_regression_metrics_on_known_values():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m, d = evaluate_regression(y, np.array([1.0, 2.0, 3.0, 5.0]))
    assert m["mae"] == pytest.approx(0.25) and m["rmse"] == pytest.approx(0.5)
    assert m["r2"] == pytest.approx(1 - 1 / 5) and m["mape"] == pytest.approx(0.0625)
    assert d["kind"] == "regression" and len(d["residuals"]["counts"]) == 30 and len(d["scatter"]["actual"]) == 4


def test_mape_is_left_out_when_an_actual_is_zero():
    m, _ = evaluate_regression(np.array([0.0, 1.0, 2.0]), np.array([0.1, 1.0, 2.0]))
    assert "mape" not in m


def test_scatter_is_capped():
    y = np.arange(5000, dtype=float)
    _, d = evaluate_regression(y, y + 1)
    assert len(d["scatter"]["actual"]) == 1000


def test_target_chain_mixes_and_chains_out_of_fold():
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    a = (X[:, 0] > 0).astype(float)
    b = 3 * a + X[:, 1] + rng.normal(scale=0.1, size=300)
    chain = TargetChain(RandomForestClassifier(n_estimators=20, random_state=0),
                        RandomForestRegressor(n_estimators=20, random_state=0),
                        ["binary", "regression"], [0, 1], cv=3, handles_nan=True).fit(X, np.column_stack([a, b]))
    assert chain.estimators_[1].n_features_in_ == 4
    proba, values = chain.predict_all(X, [2, 0])
    assert proba.shape == (300, 2) and values.shape == (300,)
    assert np.corrcoef(values, b)[0, 1] > 0.9


def _export(tmp_path, tasks):
    n = 600
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)
    df = pl.DataFrame({"x": x, "z": rng.normal(size=n), "g1": (x > 0).astype(int),
                       "g3": 10 + 4 * x + rng.normal(scale=0.5, size=n)})
    src = tmp_path / "d.parquet"
    df.write_parquet(src)
    out = tmp_path / "export"
    run_export(src, [_spec("x", "numeric_continuous"), _spec("z", "numeric_continuous")], list(tasks), out,
               {"seed": 3}, lambda *_: None,
               target_tasks={t: {"task": k, "bins": {"k": 4} if k == "bins" else None} for t, k in tasks.items()})
    return out


def test_lightgbm_regression_run(tmp_path):
    report = train(_export(tmp_path, {"g3": "regression"}), "lightgbm", params={"n_estimators": 60}, cv=3,
                   out_dir=tmp_path / "m")
    m = report["metrics"]["valid"]["g3"]
    assert m["r2"] > 0.8 and {"rmse", "mae", "r2", "mape"} <= set(m)
    assert report["chain_kind"] == "target_chain" and report["target_tasks"] == {"g3": "regression"}
    assert report["evaluation"]["test"]["g3"]["kind"] == "regression"
    json.dumps(report)


def test_mixed_chain_classes_then_number(tmp_path):
    report = train(_export(tmp_path, {"g1": "classes", "g3": "regression"}), "lightgbm",
                   ["g1", "g3"], params={"n_estimators": 40}, cv=3, out_dir=tmp_path / "m")
    assert "roc_auc" in report["metrics"]["valid"]["g1"] and "r2" in report["metrics"]["valid"]["g3"]
    assert report["target_tasks"] == {"g1": "classes", "g3": "regression"}


def test_a_subset_run_reports_only_its_targets(tmp_path):
    report = train(_export(tmp_path, {"g1": "classes", "g3": "regression"}), "lightgbm",
                   ["g3"], params={"n_estimators": 20}, cv=None, out_dir=tmp_path / "m")
    assert report["target_tasks"] == {"g3": "regression"}


def test_all_class_runs_keep_classifier_chain(tmp_path):
    report = train(_export(tmp_path, {"g1": "classes", "g3": "bins"}), "lightgbm", params={"n_estimators": 20},
                   cv=None, out_dir=tmp_path / "m")
    assert report["chain_kind"] == "classifier_chain" and len(report["target_classes"]["g3"]) == 4


@pytest.mark.parametrize("model", ["random_forest", "hist_gradient_boosting", "xgboost"])
def test_other_models_predict_numbers(tmp_path, model):
    from app.modeling.registry import MODELS
    if not MODELS[model].available:
        pytest.skip(f"{model} not installed")
    params = {"n_estimators": 20} if model != "hist_gradient_boosting" else {"max_iter": 20}
    report = train(_export(tmp_path, {"g1": "classes", "g3": "regression"}), model, params=params, cv=3,
                   out_dir=tmp_path / "m")
    assert report["metrics"]["test"]["g3"]["r2"] > 0.5


def test_single_value_regression_target_is_refused(tmp_path):
    n = 100
    pl.DataFrame({"x": np.arange(n, dtype=float), "y": np.ones(n)}).write_parquet(tmp_path / "d.parquet")
    with pytest.raises(ValueError, match="single value"):
        run_export(tmp_path / "d.parquet", [_spec("x", "numeric_continuous")], ["y"], tmp_path / "e", {"seed": 1},
                   lambda *_: None, target_tasks={"y": {"task": "regression", "bins": None}})
