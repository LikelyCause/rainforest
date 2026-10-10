"""MLflow logging of model runs: against a local file store (always), Postgres-backed run handling (with
RAINFOREST_TEST_DATABASE_URL) and a real tracking server (with MLFLOW_TEST_URI)."""
import json
import os
import time
from datetime import datetime, timezone

import pytest
from mlflow import MlflowClient

from app import db
from app.config import get_settings
from app.modeling import mlflow_backfill, mlflow_log

from .test_pipeline import profiles, tree_export  # noqa: F401  (module fixtures)

DB_URL = os.environ.get("RAINFOREST_TEST_DATABASE_URL")
UNREACHABLE = "http://127.0.0.1:9"

REPORT = {
    "model": "lightgbm", "model_label": "LightGBM", "export_family": "tree", "order": ["column_b", "column_a"],
    "chain_method": "predict", "cv": 5, "seed": 42, "device": "cpu", "n_train": 100, "n_features": 7,
    "fit_seconds": 1.5, "params": {"n_estimators": 300, "learning_rate": 0.05, "class_weight": None},
    "metrics": {"valid": {"column_a": {"roc_auc": 0.81, "f1": 0.5, "log_loss": float("nan")},
                          "price (usd)": {"accuracy": 0.7}},
                "test": {"column_a": {"roc_auc": 0.8}}},
}


@pytest.fixture
def file_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("GIT_SHA", "abc1234")
    get_settings.cache_clear()
    yield MlflowClient(tracking_uri=uri)
    get_settings.cache_clear()


def _model_dir(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / mlflow_log.MODEL_FILE).write_bytes(b"model bytes")
    (d / mlflow_log.REPORT_FILE).write_text(json.dumps({"model": "lightgbm"}))
    return d


def _run_row(run_id=7, dataset_id=3):
    return {"id": run_id, "dataset_id": dataset_id, "export_id": 5, "model": "lightgbm",
            "created_at": datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc),
            "finished_at": datetime(2026, 10, 1, 12, 5, tzinfo=timezone.utc)}


def test_logs_params_metrics_tags_and_artifacts(file_store, tmp_path):
    exp_id, run_id = mlflow_log.log_run(_run_row(), "Churn", REPORT, _model_dir(tmp_path), user="ana")
    assert file_store.get_experiment(exp_id).name == "rainforest/3-Churn"
    run = file_store.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.info.start_time == int(_run_row()["created_at"].timestamp() * 1000)
    assert run.info.end_time == int(_run_row()["finished_at"].timestamp() * 1000)
    p = run.data.params
    assert p["model"] == "lightgbm" and p["order"] == "column_b,column_a" and p["cv"] == "5"
    assert p["device"] == "cpu" and p["seed"] == "42" and p["chain_method"] == "predict"
    assert p["estimator.n_estimators"] == "300" and p["estimator.learning_rate"] == "0.05"
    assert "estimator.class_weight" in p
    m = run.data.metrics
    assert m["valid.column_a.roc_auc"] == 0.81 and m["test.column_a.roc_auc"] == 0.8
    assert m["valid.price _usd_.accuracy"] == 0.7  # characters MLflow refuses in keys become _
    assert "valid.column_a.log_loss" not in m  # not finite
    assert m["fit_seconds"] == 1.5
    t = run.data.tags
    assert t["rainforest.dataset_id"] == "3" and t["rainforest.dataset_name"] == "Churn"
    assert t["rainforest.export_id"] == "5" and t["rainforest.run_id"] == "7"
    assert t["rainforest.model_family"] == "tree" and t["rainforest.user"] == "ana"
    assert t["rainforest.git_sha"] == "abc1234" and t["mlflow.source.git.commit"] == "abc1234"
    assert t["mlflow.runName"] == "run 7 · LightGBM"
    assert {a.path for a in file_store.list_artifacts(run_id)} == {"model", "report.json"}
    assert [a.path for a in file_store.list_artifacts(run_id, "model")] == ["model/classifier_chain.joblib"]


def test_one_experiment_per_dataset_even_after_a_rename(file_store, tmp_path):
    d = _model_dir(tmp_path)
    first, _ = mlflow_log.log_run(_run_row(1), "Churn", REPORT, d)
    again, _ = mlflow_log.log_run(_run_row(2), "Churn v2", REPORT, d)
    other, last = mlflow_log.log_run(_run_row(3, dataset_id=4), "Churn", REPORT, d)
    assert first == again != other
    assert file_store.get_experiment(first).tags["mlflow.experimentKind"] == "custom_model_development"
    assert "rainforest.user" not in file_store.get_run(last).data.tags


def test_adopts_an_existing_experiment_with_the_same_name(file_store, tmp_path):
    made = file_store.create_experiment("rainforest/9-Loans")
    exp_id, _ = mlflow_log.log_run(_run_row(dataset_id=9), "Loans", REPORT, _model_dir(tmp_path))
    assert exp_id == made
    assert file_store.get_experiment(made).tags["rainforest.dataset_id"] == "9"


def test_a_failure_part_way_removes_the_half_logged_run(file_store, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("artifact upload failed")

    monkeypatch.setattr(MlflowClient, "log_artifact", boom)
    with pytest.raises(ConnectionError):
        mlflow_log.log_run(_run_row(), "Churn", REPORT, _model_dir(tmp_path))
    exp = file_store.get_experiment_by_name("rainforest/3-Churn")
    assert file_store.search_runs([exp.experiment_id]) == []


def test_unreachable_server_fails_fast(monkeypatch, tmp_path):
    t0 = time.monotonic()
    with pytest.raises(Exception) as err:
        mlflow_log.log_run(_run_row(), "Churn", REPORT, _model_dir(tmp_path), tracking_uri=UNREACHABLE)
    assert time.monotonic() - t0 < 30
    assert "127.0.0.1" in mlflow_log.error_message(err.value)


# ---------------------------------------------------------------------------------- with Postgres
@pytest.fixture(scope="module")
def database(tmp_path_factory):
    if not DB_URL:
        pytest.skip("set RAINFOREST_TEST_DATABASE_URL to run the database-backed MLflow tests")
    mp = pytest.MonkeyPatch()
    mp.setenv("DATABASE_URL", DB_URL)
    mp.setenv("DATA_DIR", str(tmp_path_factory.mktemp("data")))
    get_settings.cache_clear()
    db.close_pool()
    db.bootstrap()
    yield
    db.close_pool()
    mp.undo()
    get_settings.cache_clear()


def _train_run(export_dir) -> int:
    from app.routers import runs

    ds = db.fetch_one("INSERT INTO meta.datasets (name, file_format, status) VALUES ('Churn', 'generated', 'ready') "
                      "RETURNING id")["id"]
    exp = db.fetch_one("INSERT INTO meta.exports (dataset_id, status, output_dir) VALUES (%s, 'done', %s) RETURNING id",
                       (ds, str(export_dir)))["id"]
    user = db.fetch_one("INSERT INTO meta.users (name, email) VALUES ('Ana', 'ana@example.com') RETURNING id")["id"]
    run_id = db.fetch_one("INSERT INTO meta.model_runs (export_id, dataset_id, model, family, created_by) "
                          "VALUES (%s, %s, 'lightgbm', 'tree', %s) RETURNING id", (exp, ds, user))["id"]
    req = {"model": "lightgbm", "order": None, "seed": 0, "cv": 2, "params": {"n_estimators": 20}, "device": "cpu"}
    runs._run(run_id, str(export_dir), req)
    return run_id


def test_training_run_is_logged_and_linked(database, file_store, tree_export):  # noqa: F811
    from app.routers import runs

    run_id = _train_run(tree_export)
    run = runs.get_run(run_id)
    assert run["status"] == "done", run["error"]
    assert run["mlflow_run_id"] and "mlflow_error" not in run["report"]
    assert run["mlflow_url"] == f"/mlflow/#/experiments/{run['mlflow_experiment_id']}/runs/{run['mlflow_run_id']}"
    listed = next(r for r in runs.list_runs(run["dataset_id"]) if r["id"] == run_id)
    assert listed["mlflow_url"] == run["mlflow_url"]
    logged = file_store.get_run(run["mlflow_run_id"])
    assert logged.data.tags["rainforest.run_id"] == str(run_id)
    assert logged.data.tags["rainforest.user"] == "Ana <ana@example.com>"
    assert logged.data.params["estimator.n_estimators"] == "20" and logged.data.params["cv"] == "2"
    assert logged.data.metrics["test.column_a.roc_auc"] == run["metrics"]["test"]["column_a"]["roc_auc"]
    assert {a.path for a in file_store.list_artifacts(run["mlflow_run_id"], "model")} == {
        "model/classifier_chain.joblib"}


def test_unreachable_mlflow_still_completes_the_run_then_backfill_logs_it(database, file_store, tree_export,  # noqa: F811
                                                                         monkeypatch):
    from app.routers import runs

    monkeypatch.setenv("MLFLOW_TRACKING_URI", UNREACHABLE)
    get_settings.cache_clear()
    run_id = _train_run(tree_export)
    run = runs.get_run(run_id)
    assert run["status"] == "done" and run["mlflow_run_id"] is None and run["mlflow_url"] is None
    assert "127.0.0.1:9" in run["report"]["mlflow_error"]

    monkeypatch.setenv("MLFLOW_TRACKING_URI", file_store.tracking_uri)
    get_settings.cache_clear()
    assert run_id in {r["id"] for r in mlflow_backfill.pending()}
    logged, failed = mlflow_backfill.backfill([run_id], say=lambda _: None)
    assert (logged, failed) == (1, 0)
    run = runs.get_run(run_id)
    assert run["mlflow_run_id"] and "mlflow_error" not in run["report"]
    assert file_store.get_run(run["mlflow_run_id"]).data.metrics["fit_seconds"] == run["report"]["fit_seconds"]
    assert mlflow_backfill.pending([run_id]) == []


def test_logging_off_without_a_tracking_uri(database, tree_export, monkeypatch):  # noqa: F811
    from app.routers import runs

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
    get_settings.cache_clear()
    run = runs.get_run(_train_run(tree_export))
    assert run["status"] == "done" and run["mlflow_run_id"] is None and "mlflow_error" not in run["report"]
    get_settings.cache_clear()


# ------------------------------------------------------------------------------ real tracking server
@pytest.mark.skipif(not os.environ.get("MLFLOW_TEST_URI"), reason="set MLFLOW_TEST_URI to test a real server")
def test_real_tracking_server(tmp_path):
    uri = os.environ["MLFLOW_TEST_URI"]
    name = f"ci-{int(time.time() * 1000)}"
    exp_id, run_id = mlflow_log.log_run(_run_row(dataset_id=int(time.time())), name, REPORT, _model_dir(tmp_path),
                                        tracking_uri=uri)
    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(run_id)
    assert run.info.experiment_id == exp_id and run.info.status == "FINISHED"
    assert run.data.metrics["valid.column_a.roc_auc"] == 0.81
    assert {a.path for a in client.list_artifacts(run_id)} == {"model", "report.json"}
    local = client.download_artifacts(run_id, "model/classifier_chain.joblib", str(tmp_path / "dl"))
    assert open(local, "rb").read() == b"model bytes"
