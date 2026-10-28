"""One-vs-rest runs: one binary model of one class of a class target against the rest, calibrated on the validation
split, with a fine threshold curve and the threshold rules a deployment picks from."""

import json

import joblib
import numpy as np
import polars as pl
import pytest
from sklearn.metrics import average_precision_score
from sklearn.multioutput import ClassifierChain

from app.modeling.mlflow_log import MODEL_FILE, model_file_name, run_params, run_tags
from app.modeling.registry import MODELS
from app.modeling.train import (
    balance_params,
    calibration_method,
    choose_threshold,
    load_export,
    predict_scores,
    train,
)
from app.transform import run_export

from .test_export_transforms import _spec

# Shares of V01..V04: V01 has well over 200 validation rows of each kind (isotonic), V04 far fewer (sigmoid).
SHARES = (0.45, 0.30, 0.20, 0.05)


@pytest.fixture(scope="module")
def ovr_export(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ovr")
    rng = np.random.default_rng(11)
    n = 6000
    x, z = rng.normal(size=n), rng.normal(size=n)
    score = x + 0.6 * rng.normal(size=n)
    cuts = np.quantile(score, np.cumsum(SHARES)[:-1])
    vehicle = np.array(["V01", "V02", "V03", "V04"])[np.searchsorted(cuts, score)]
    # Missing values reach the trees as NaN, through the calibrator too.
    z[rng.random(n) < 0.1] = np.nan
    df = pl.DataFrame({"x": x, "z": z, "vehicle": vehicle, "price": 10 + 4 * x + rng.normal(scale=0.5, size=n)})
    df.write_parquet(tmp / "d.parquet")
    out = tmp / "export"
    run_export(
        tmp / "d.parquet",
        [_spec("x", "numeric_continuous"), _spec("z", "numeric_continuous")],
        ["vehicle", "price"],
        out,
        {"seed": 3},
        lambda *_: None,
        target_tasks={"vehicle": {"task": "classes", "bins": None}, "price": {"task": "regression", "bins": None}},
    )
    return out


def _split(export_dir, split, cls):
    manifest, df = load_export(export_dir)
    part = df.filter(pl.col(manifest["split_column"]) == split)
    X = part.select(manifest["feature_names"]).to_numpy()
    y = part["vehicle"].to_numpy().astype(int) == manifest["target_classes"]["vehicle"].index(cls)
    return X, y


def _ovr(export_dir, out_dir, cls="V02", model="lightgbm", **kw):
    return train(
        export_dir,
        model,
        params={"n_estimators": 30},
        out_dir=out_dir,
        cv_folds=kw.pop("cv_folds", 0),
        target="vehicle",
        positive_class=cls,
        **kw,
    )


def test_a_one_vs_rest_run_is_a_binary_model_of_one_class(ovr_export, tmp_path):
    report = _ovr(ovr_export, tmp_path / "m")
    assert report["model_label"] == "LightGBM · vehicle = V02 vs rest"
    assert report["chain_kind"] == "one_vs_rest" and report["order"] == ["vehicle"]
    assert report["target"] == "vehicle" and report["positive_class"] == "V02"
    assert report["target_kinds"] == {"vehicle": "binary"}
    for split in ("valid", "test"):
        m = report["metrics"][split]
        assert set(m) == {"vehicle"}, "the export's other targets are left out"
        assert {"pr_auc", "roc_auc", "precision", "recall", "f1", "specificity", "log_loss"} <= set(m["vehicle"])
        ev = report["evaluation"][split]["vehicle"]
        assert ev["labels"] == ["not V02", "V02"] and ev["positive_label"] == "V02"
        assert len(ev["thresholds"]) == 19
        curve = report["threshold_curve"][split]
        assert len(curve) == 101 and curve[0]["threshold"] == 0.0 and curve[-1]["threshold"] == 1.0
        assert curve[37]["threshold"] == 0.37
        assert {"precision", "recall", "f1", "flagged"} <= set(curve[0])
        assert curve[0]["flagged"] == 1.0 and curve[0]["recall"] == 1.0
    pos = report["positives"]
    assert pos["train"] > 0 and pos["valid"] > 0 and pos["test"] > 0
    # The F1 choice is kept, so a batch's rows needn't read both curves.
    t, note = choose_threshold(report["threshold_curve"]["valid"], {"kind": "f1"})
    at = {s: next(p for p in report["threshold_curve"][s] if p["threshold"] == t)["f1"] for s in ("valid", "test")}
    assert report["f1_threshold"] == {"threshold": t, "valid_f1": at["valid"], "test_f1": at["test"], "note": note}
    cal = report["calibration"]
    assert cal["method"] in ("isotonic", "sigmoid")
    assert 0 <= cal["brier_after"] <= cal["brier_before"] + 1e-9 and cal["brier_before"] < 0.25
    json.dumps(report)


def test_predict_scores_are_the_saved_models_calibrated_probabilities(ovr_export, tmp_path):
    report = _ovr(ovr_export, tmp_path / "m")
    X, y = _split(ovr_export, "test", "V02")
    assert np.isnan(X).any()
    scores = predict_scores(tmp_path / "m", X)
    assert scores.shape == (len(X),) and scores.min() >= 0 and scores.max() <= 1
    model = joblib.load(tmp_path / "m" / MODEL_FILE)
    assert type(model).__name__ == "CalibratedClassifierCV"
    assert np.array_equal(scores, model.predict_proba(X.astype(np.float32))[:, 1])
    # Every metric is computed on the same calibrated scores.
    assert average_precision_score(y, scores) == pytest.approx(
        report["metrics"]["test"]["vehicle"]["pr_auc"], abs=1e-12
    )


def test_isotonic_needs_200_positives_and_200_negatives_in_validation(ovr_export, tmp_path):
    assert calibration_method(200, 200) == "isotonic"
    assert calibration_method(199, 5000) == "sigmoid" and calibration_method(5000, 199) == "sigmoid"
    common = _ovr(ovr_export, tmp_path / "a", "V01")
    rare = _ovr(ovr_export, tmp_path / "b", "V04")
    assert common["positives"]["valid"] >= 200 and common["calibration"]["method"] == "isotonic"
    assert rare["positives"]["valid"] < 200 and rare["calibration"]["method"] == "sigmoid"
    assert common["calibration"]["valid_positives"] == common["positives"]["valid"]
    # The calibrator learnt from the validation split's raw scores, not the training rows'.
    model = joblib.load(tmp_path / "a" / MODEL_FILE)
    X_va, _ = _split(ovr_export, "valid", "V01")
    raw = model.estimator.predict_proba(X_va.astype(np.float32))[:, 1]
    iso = model.calibrated_classifiers_[0].calibrators[0]
    assert (iso.X_min_, iso.X_max_) == (raw.min(), raw.max())


def test_balance_weights_lightgbm_by_negatives_over_positives_of_the_training_rows(ovr_export, tmp_path):
    plain = _ovr(ovr_export, tmp_path / "plain", "V04")
    balanced = _ovr(ovr_export, tmp_path / "bal", "V04", balance=True)
    pos = balanced["positives"]["train"]
    ratio = (balanced["n_train"] - pos) / pos

    def fitted(d):
        return joblib.load(d / MODEL_FILE).estimator.estimator.get_params().get("scale_pos_weight", 1.0)

    assert fitted(tmp_path / "plain") == 1.0 and plain["balance"] is False
    assert fitted(tmp_path / "bal") == pytest.approx(ratio) and ratio > 10
    assert balanced["balance"] is True and balanced["params"]["scale_pos_weight"] == pytest.approx(ratio)
    assert balanced["class_weighting"] == {"scale_pos_weight": pytest.approx(ratio), "note": None}


def test_balance_by_model():
    y = np.array([1] * 10 + [0] * 30)
    assert balance_params(MODELS["lightgbm"], y) == ({"scale_pos_weight": 3.0}, None)
    assert balance_params(MODELS["xgboost"], y) == ({"scale_pos_weight": 3.0}, None)
    for key in ("random_forest", "hist_gradient_boosting", "logistic_regression"):
        assert balance_params(MODELS[key], y) == ({"class_weight": "balanced"}, None)
    params, note = balance_params(MODELS["mlp"], y)
    assert params == {} and "MLP" in note


def test_a_forest_balances_with_class_weight(ovr_export, tmp_path):
    report = _ovr(ovr_export, tmp_path / "m", "V04", model="random_forest", balance=True)
    base = joblib.load(tmp_path / "m" / MODEL_FILE).estimator.estimator
    assert base.get_params()["class_weight"] == "balanced" and report["params"]["class_weight"] == "balanced"


def test_balance_with_a_weighting_setting_is_refused(ovr_export, tmp_path):
    with pytest.raises(ValueError, match="balance"):
        train(
            ovr_export,
            "lightgbm",
            params={"n_estimators": 10, "is_unbalance": True},
            out_dir=tmp_path / "m",
            target="vehicle",
            positive_class="V02",
            balance=True,
        )


def test_balance_with_weighting_settings_at_their_defaults_is_accepted(ovr_export, tmp_path):
    # A client that sends every setting sends the weighting ones at their defaults; only a real choice clashes.
    report = train(
        ovr_export,
        "lightgbm",
        params={"n_estimators": 10, "is_unbalance": False, "scale_pos_weight": 1.0, "class_weight": None},
        out_dir=tmp_path / "m",
        target="vehicle",
        positive_class="V04",
        balance=True,
    )
    assert report["params"]["scale_pos_weight"] > 10


def test_weighting_clash_names_only_settings_moved_off_their_defaults():
    from app.modeling.registry import weighting_clash

    lgbm = MODELS["lightgbm"]
    assert weighting_clash(lgbm, {"is_unbalance": False, "scale_pos_weight": 1, "class_weight": None}) == []
    assert weighting_clash(lgbm, {"is_unbalance": True, "scale_pos_weight": 1.0}) == ["is_unbalance"]
    assert weighting_clash(lgbm, {"scale_pos_weight": 3.0, "class_weight": "balanced"}) == [
        "class_weight",
        "scale_pos_weight",
    ]
    assert weighting_clash(MODELS["random_forest"], {"class_weight": None}) == []
    assert weighting_clash(MODELS["random_forest"], {"class_weight": "balanced"}) == ["class_weight"]


def test_cross_validation_on_a_one_vs_rest_run(ovr_export, tmp_path):
    report = _ovr(ovr_export, tmp_path / "m", cv_folds=3)
    assert report["cross_validation"]["folds"] == 3 and report["cross_validation"]["stratified_on"] == "vehicle"
    assert {"pr_auc", "roc_auc", "f1"} <= set(report["metrics"]["cv"]["vehicle"])


def test_unknown_classes_and_number_targets_are_refused(ovr_export, tmp_path):
    with pytest.raises(ValueError, match="V99"):
        _ovr(ovr_export, tmp_path / "a", "V99")
    with pytest.raises(ValueError, match="number"):
        train(ovr_export, "lightgbm", out_dir=tmp_path / "b", target="price", positive_class="10")


def test_runs_without_a_positive_class_are_unchanged(ovr_export, tmp_path):
    report = train(ovr_export, "lightgbm", ["vehicle"], params={"n_estimators": 10}, cv=None, out_dir=tmp_path / "m")
    assert report["chain_kind"] == "classifier_chain" and report["model_label"] == "LightGBM"
    assert not {"threshold_curve", "calibration", "positive_class", "target", "balance"} & set(report)
    assert isinstance(joblib.load(tmp_path / "m" / MODEL_FILE), ClassifierChain)
    X, _ = _split(ovr_export, "test", "V01")
    assert predict_scores(tmp_path / "m", X).shape == (len(X), 4), "a single-target run scores every class"


CURVE = [
    {"threshold": round(t, 2), "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "flagged": 0}
    for t, p, r in [
        (0.0, 0.1, 1.0),
        (0.1, 0.4, 0.9),
        (0.2, 0.6, 0.8),
        (0.3, 0.7, 0.5),
        (0.4, 0.82, 0.3),
        (0.5, 0.9, 0.1),
        (0.6, 0.0, 0.0),
    ]
]


def test_choose_threshold_rules():
    assert choose_threshold(CURVE, {"kind": "f1"}) == (0.2, None)
    assert choose_threshold(CURVE, {"kind": "precision", "min": 0.8}) == (0.4, None)
    assert choose_threshold(CURVE, {"kind": "precision", "min": 0.6}) == (0.2, None)
    t, note = choose_threshold(CURVE, {"kind": "precision", "min": 0.95})
    assert t == 0.2 and "0.95" in note and "F1" in note
    assert choose_threshold(CURVE, {"kind": "fixed", "value": 0.33}) == (0.33, None)
    with pytest.raises(ValueError):
        choose_threshold(CURVE, {"kind": "fixed", "value": 1.5})
    with pytest.raises(ValueError):
        choose_threshold(CURVE, {"kind": "median"})


def test_choose_threshold_on_an_empty_curve_says_so():
    for rule in ({"kind": "f1"}, {"kind": "precision", "min": 0.8}):
        with pytest.raises(ValueError, match="no threshold curve"):
            choose_threshold([], rule)


def test_a_curve_where_nothing_scores_falls_back_to_one_half():
    flat = [{**p, "precision": 0.0, "recall": 0.0, "f1": 0.0} for p in CURVE]
    t, note = choose_threshold(flat, {"kind": "f1"})
    assert t == 0.5 and note


def test_mlflow_records_the_positive_class_and_the_batch():
    report = {"model": "lightgbm", "target": "vehicle", "positive_class": "V02", "balance": True}
    params = run_params(report)
    assert params["positive_class"] == "V02" and params["target"] == "vehicle" and params["balance"] == "True"
    assert "positive_class" not in run_params({"model": "lightgbm"})
    run = {"dataset_id": 1, "export_id": 2, "id": 3, "batch_id": 9, "batch_kind": "one_vs_rest"}
    tags = run_tags(run, "d", report, None)
    assert tags["rainforest.positive_class"] == "V02"
    assert tags["rainforest.batch_id"] == "9" and tags["rainforest.batch_kind"] == "one_vs_rest"
    plain = run_tags({"dataset_id": 1, "export_id": 2, "id": 3}, "d", {"model": "lightgbm"}, None)
    assert not {"rainforest.positive_class", "rainforest.batch_id", "rainforest.batch_kind"} & set(plain)
    assert model_file_name({"chain_kind": "one_vs_rest"}) == "calibrated_classifier.joblib"


def test_scores_of_a_single_number_target_run_are_its_predicted_numbers(ovr_export, tmp_path):
    report = train(ovr_export, "lightgbm", ["price"], params={"n_estimators": 30}, cv=None, out_dir=tmp_path / "m")
    assert report["chain_kind"] == "target_chain"
    manifest, df = load_export(ovr_export)
    test = df.filter(pl.col(manifest["split_column"]) == "test")
    X, y = test.select(manifest["feature_names"]).to_numpy(), test["price"].to_numpy()
    values = predict_scores(tmp_path / "m", X)
    assert values.shape == (len(X),) and values.dtype.kind == "f"
    # The numbers the run was scored on: the same rows give the report's test RMSE.
    rmse = float(np.sqrt(np.mean((values - y) ** 2)))
    assert rmse == pytest.approx(report["metrics"]["test"]["price"]["rmse"], rel=1e-9)


def test_run_requests_default_to_no_positive_class():
    from app.schemas import RunRequest

    r = RunRequest()
    assert r.positive_class is None and r.target is None and r.balance is False


from .team_helpers import app_, people  # noqa: E402,F401  (fixtures)


def _two_target_csv(n=900, seed=4) -> str:
    rng = np.random.default_rng(seed)
    x, z = rng.normal(size=n), rng.normal(size=n)
    s = x + 0.5 * rng.normal(size=n)
    y = np.where(s < -0.4, "p", np.where(s < 0.5, "q", "r"))
    v = 3 * x + rng.normal(scale=0.3, size=n)
    return "x,z,y,v\n" + "".join(f"{a:.3f},{b:.3f},{c},{d:.3f}\n" for a, b, c, d in zip(x, z, y, v))


def test_one_vs_rest_runs_through_the_api(people):
    from .team_helpers import experiment, export, start, submit_and_finalize, upload, wait

    lead = people["lead"]
    ds = upload(lead, _two_target_csv(), targets="y,v")
    start(lead, ds, "tree", [lead], targets=("y", "v"))
    rid = submit_and_finalize(lead, ds)["recipe"]["id"]
    e = export(lead, ds, experiment(lead, ds, rid, "ovr"))
    assert e["summary"]["target_kinds"] == {"y": "multiclass", "v": "regression"}, e["summary"]["target_kinds"]

    def post(**body):
        return lead.post(
            f"/api/exports/{e['id']}/runs",
            json={"model": "lightgbm", "cv": 0, "cv_folds": 0, "params": {"n_estimators": 20}, **body},
        )

    refused = {
        "unknown class": post(target="y", positive_class="zzz"),
        "number target": post(target="v", positive_class="1"),
        "target needed": post(positive_class="q"),
        "unknown target": post(target="w", positive_class="q"),
        "balance alone": post(balance=True),
        "target alone": post(target="y"),
        "another order": post(target="y", positive_class="q", order=["v"]),
        "weights twice": post(target="y", positive_class="q", balance=True, params={"is_unbalance": True}),
    }
    for why, r in refused.items():
        assert r.status_code == 422, (why, r.status_code, r.text)
    assert "zzz" in refused["unknown class"].text and "number" in refused["number target"].text

    r = post(target="y", positive_class="q", balance=True)
    assert r.status_code == 202, r.text
    assert r.json()["report"]["model_label"] == "LightGBM · y = q vs rest"
    run = wait(lambda: (g := lead.get(f"/api/runs/{r.json()['id']}").json())["status"] in ("done", "failed") and g)
    assert run["status"] == "done", run.get("error")
    assert run["config"]["positive_class"] == "q" and run["config"]["target"] == "y" and run["config"]["balance"]
    assert run["report"]["model_label"] == "LightGBM · y = q vs rest"
    assert run["report"]["calibration"]["method"] == "sigmoid"
    assert len(run["report"]["threshold_curve"]["valid"]) == 101
    assert "pr_auc" in run["metrics"]["valid"]["y"] and "v" not in run["metrics"]["valid"]

    plain = post()
    assert plain.status_code == 202
    assert not {"positive_class", "target", "balance"} & set(plain.json()["config"])
    done = wait(lambda: (g := lead.get(f"/api/runs/{plain.json()['id']}").json())["status"] in ("done", "failed") and g)
    assert done["status"] == "done" and done["report"]["chain_kind"] == "target_chain", done.get("error")
