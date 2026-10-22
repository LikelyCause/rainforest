"""Cross-validation inside a run: folds of the training rows, scored like validation and test, summarized as mean ±
spread per metric."""

import numpy as np
import pytest

from app.modeling.crossval import folds, summarize
from app.modeling.train import train

from .test_train_regression import _export


def test_folds_hold_out_every_row_once():
    out, note, strat = folds(None, 103, 5, seed=1, strat_name=None)
    held = np.concatenate([te for _, te in out])
    assert len(out) == 5 and note is None and not strat and sorted(held.tolist()) == list(range(103))
    for tr, te in out:
        assert not set(tr) & set(te) and len(tr) + len(te) == 103


def test_stratified_folds_keep_class_shares():
    y = np.array([1] * 30 + [0] * 70)
    out, note, strat = folds(y, 100, 5, seed=1, strat_name="y")
    assert note is None and strat
    for _, te in out:
        assert abs(y[te].mean() - 0.3) <= 0.05


def test_a_rare_class_gives_unstratified_folds_with_a_note():
    y = np.array([2] * 3 + [1] * 40 + [0] * 57)
    out, note, strat = folds(y, 100, 5, seed=1, strat_name="grade")
    assert len(out) == 5 and not strat and "rarest class of grade has 3 training rows" in note


def test_fewer_rows_than_folds_lowers_k():
    out, note, _ = folds(None, 3, 5, seed=1, strat_name=None)
    assert len(out) == 3 and "3 folds" in note


def test_summarize_means_and_spreads_finite_values():
    s = summarize([{"y": {"auc": 0.8, "f1": 0.5}}, {"y": {"auc": 0.9, "f1": float("nan")}}, {"y": {"auc": 1.0}}])
    assert s["y"]["auc"]["mean"] == pytest.approx(0.9) and s["y"]["auc"]["std"] == pytest.approx(
        np.std([0.8, 0.9, 1.0])
    )
    assert s["y"]["auc"]["folds"] == [0.8, 0.9, 1.0]
    assert s["y"]["f1"]["mean"] == 0.5 and s["y"]["f1"]["folds"] == [0.5, None, None]
    assert summarize([{"y": {"auc": None}}]) == {}


def test_a_run_reports_cross_validation_for_a_number(tmp_path):
    report = train(
        _export(tmp_path, {"g3": "regression"}),
        "lightgbm",
        params={"n_estimators": 30},
        cv=None,
        out_dir=tmp_path / "m",
        cv_folds=3,
    )
    cv = report["cross_validation"]
    assert cv["folds"] == 3 and cv["stratified_on"] is None
    assert len(cv["targets"]["g3"]["r2"]["folds"]) == 3
    assert abs(cv["targets"]["g3"]["r2"]["mean"] - report["metrics"]["valid"]["g3"]["r2"]) < 0.1
    assert report["metrics"]["cv"]["g3"]["r2"] == cv["targets"]["g3"]["r2"]["mean"]
    assert {"valid", "test"} <= set(report["metrics"])


def test_a_run_reports_cross_validation_for_classes(tmp_path):
    report = train(
        _export(tmp_path, {"g1": "classes"}),
        "lightgbm",
        params={"n_estimators": 30},
        cv=None,
        out_dir=tmp_path / "m",
        cv_folds=4,
    )
    assert report["cross_validation"]["folds"] == 4 and report["cross_validation"]["stratified_on"] == "g1"
    assert {"roc_auc", "f1_macro", "accuracy"} <= set(report["metrics"]["cv"]["g1"])


def test_cross_validation_off_reports_none(tmp_path):
    report = train(
        _export(tmp_path, {"g1": "classes"}), "lightgbm", params={"n_estimators": 10}, cv=None, out_dir=tmp_path / "m"
    )
    assert "cross_validation" not in report and "cv" not in report["metrics"]


def test_a_fold_without_a_class_skips_cross_validation_not_the_run(tmp_path, monkeypatch):
    from app.modeling import crossval

    def minority_held_out(y, n, k, seed, name, groups=None):
        rest = np.flatnonzero(y == 0)
        return [(rest[: len(rest) // 2], np.flatnonzero(y == 1))] * k, None, False

    monkeypatch.setattr(crossval, "folds", minority_held_out)
    report = train(
        _export(tmp_path, {"g1": "classes"}),
        "random_forest",
        params={"n_estimators": 10},
        cv=None,
        out_dir=tmp_path / "m",
        cv_folds=5,
    )
    assert report["cross_validation"]["folds"] == 0 and "skipped" in report["cross_validation"]["note"]
    assert "cv" not in report["metrics"] and "test" in report["metrics"]


def test_runs_take_cv_folds_from_0_or_2_to_20_default_5():
    from pydantic import ValidationError

    from app.schemas import RunRequest

    assert (
        RunRequest().cv_folds == 5 and RunRequest(cv_folds=0).cv_folds == 0 and RunRequest(cv_folds=20).cv_folds == 20
    )
    for bad in (1, 21, -1):
        with pytest.raises(ValidationError):
            RunRequest(cv_folds=bad)


def test_mlflow_logs_cross_validation_means_spreads_and_folds():
    from app.modeling.mlflow_log import run_metrics, run_params

    report = {
        "metrics": {"cv": {"y": {"roc_auc": 0.9}}},
        "cv_folds": 5,
        "cross_validation": {"folds": 5, "targets": {"y": {"roc_auc": {"mean": 0.9, "std": 0.02, "folds": []}}}},
    }
    m = run_metrics(report)
    assert m["cv.y.roc_auc"] == 0.9 and m["cv_std.y.roc_auc"] == 0.02
    assert run_params(report)["cv_folds"] == "5" and run_params({"model": "x"})["cv_folds"] == "0"


def _rare_chain_export(tmp_path):
    import polars as pl

    from app.transform import run_export

    from .test_export_transforms import _spec

    rng = np.random.default_rng(5)
    n = 800
    x, z = rng.normal(size=n), rng.normal(size=n)
    b = np.zeros(n, dtype=int)
    b[[3, 5, 7, 9, 11, 13]] = 1
    pl.DataFrame({"x": x, "z": z, "b": b, "c": (x > 0).astype(int), "g": 2 * x + rng.normal(size=n)}).write_parquet(
        tmp_path / "d.parquet"
    )
    out = tmp_path / "export"
    run_export(
        tmp_path / "d.parquet",
        [_spec("x", "numeric_continuous"), _spec("z", "numeric_continuous")],
        ["b", "c", "g"],
        out,
        {"seed": 3},
        lambda *_: None,
        target_tasks={
            "b": {"task": "classes", "bins": None},
            "c": {"task": "classes", "bins": None},
            "g": {"task": "regression", "bins": None},
        },
    )
    return out


def test_a_fold_that_fails_skips_cross_validation_not_the_run(tmp_path, monkeypatch):
    from app.modeling import crossval

    def one_rare_row_in_training(y, n, k, seed, name, groups=None):
        rare, common = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        rows = np.concatenate([rare[:1], common])
        return [(rows, rare[1:])] * k, None, True

    monkeypatch.setattr(crossval, "folds", one_rare_row_in_training)
    report = train(
        _rare_chain_export(tmp_path),
        "lightgbm",
        ["b", "c"],
        params={"n_estimators": 10},
        cv=5,
        out_dir=tmp_path / "m",
        cv_folds=5,
    )
    cv = report["cross_validation"]
    assert cv["folds"] == 0 and "fold 1 of 5" in cv["note"] and "cv" not in report["metrics"]
    assert "test" in report["metrics"]


def test_a_fold_without_a_class_is_found_before_any_fold_is_fitted(tmp_path, monkeypatch):
    from app.modeling import crossval

    real = crossval.folds

    def last_fold_loses_a_class(y, n, k, seed, name, groups=None):
        out, note, strat = real(y, n, k, seed, name)
        ones = np.flatnonzero(y == 1)
        return out[:-1] + [(np.setdiff1d(np.arange(n), ones), ones)], note, strat

    monkeypatch.setattr(crossval, "folds", last_fold_loses_a_class)
    said = []
    report = train(
        _export(tmp_path, {"g1": "classes"}),
        "lightgbm",
        params={"n_estimators": 10},
        cv=None,
        out_dir=tmp_path / "m",
        cv_folds=4,
        progress=said.append,
    )
    assert report["cross_validation"]["folds"] == 0 and "fold 4" in report["cross_validation"]["note"]
    assert not [m for m in said if m.startswith("Cross-validating")]


def test_a_mixed_chain_cross_validates_every_target(tmp_path):
    report = train(
        _rare_chain_export(tmp_path),
        "lightgbm",
        ["c", "g"],
        params={"n_estimators": 20},
        cv=3,
        out_dir=tmp_path / "m",
        cv_folds=3,
    )
    cv = report["cross_validation"]
    assert cv["folds"] == 3 and cv["stratified_on"] == "c"
    assert "roc_auc" in cv["targets"]["c"] and "r2" in cv["targets"]["g"]


def test_a_class_missing_from_a_folds_training_rows_still_scores(tmp_path, monkeypatch):
    import polars as pl

    from app.modeling import crossval
    from app.transform import run_export

    from .test_export_transforms import _spec

    rng = np.random.default_rng(4)
    n = 600
    x = rng.normal(size=n)
    m = (x > 0).astype(int)
    m[:12] = 2
    pl.DataFrame({"x": x, "m": m}).write_parquet(tmp_path / "d.parquet")
    run_export(
        tmp_path / "d.parquet",
        [_spec("x", "numeric_continuous")],
        ["m"],
        tmp_path / "e",
        {"seed": 3},
        lambda *_: None,
        target_tasks={"m": {"task": "classes", "bins": None}},
    )

    def third_class_held_out(y, n, k, seed, name, groups=None):
        rare = np.flatnonzero(y == 2)
        rest = np.setdiff1d(np.arange(n), rare)
        cut = len(rest) * 4 // 5
        return [(rest[:cut], np.concatenate([rest[cut:], rare]))] * k, None, False

    monkeypatch.setattr(crossval, "folds", third_class_held_out)
    report = train(
        tmp_path / "e", "random_forest", params={"n_estimators": 10}, cv=None, out_dir=tmp_path / "m", cv_folds=2
    )
    cv = report["cross_validation"]
    assert cv["folds"] == 2 and np.isfinite(cv["targets"]["m"]["log_loss"]["mean"])


def test_notes_count_rows_and_keep_both_reasons():
    _, note, _ = folds(np.array([1, 0, 0]), 3, 5, seed=1, strat_name="y")
    assert "3 folds: there are only 3 training rows" in note and "rarest class of y has 1 training row" in note


from .team_helpers import app_, people  # noqa: E402,F401  (fixtures)


def test_a_run_started_through_the_api_stores_and_reports_cross_validation(people):
    from .team_helpers import experiment, export, finalized_trees, wait

    lead = people["lead"]
    ds, rid = finalized_trees(lead)
    e = export(lead, ds, experiment(lead, ds, rid, "cv"))
    assert lead.post(f"/api/exports/{e['id']}/runs", json={"model": "lightgbm", "cv_folds": 1}).status_code == 422
    r = lead.post(
        f"/api/exports/{e['id']}/runs", json={"model": "lightgbm", "cv_folds": 2, "params": {"n_estimators": 10}}
    )
    assert r.status_code == 202, r.text
    run = wait(lambda: (x := lead.get(f"/api/runs/{r.json()['id']}").json())["status"] in ("done", "failed") and x)
    assert run["status"] == "done", run.get("error")
    assert run["config"]["cv_folds"] == 2 and run["report"]["cross_validation"]["folds"] == 2 and "cv" in run["metrics"]
