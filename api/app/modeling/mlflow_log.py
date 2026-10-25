"""Log finished model runs to the MLflow tracking server: one experiment per dataset, one MLflow run per model run.

Logging is best effort. Training never waits long on it or fails because of it: any error comes back as a message
that the caller stores as `mlflow_error` in the run report.
"""

from __future__ import annotations

import logging
import math
import os
import re
from datetime import datetime
from pathlib import Path

from ..config import get_settings

log = logging.getLogger(__name__)

# Read by the MLflow client on every request; the defaults (120 s, 7 retries with backoff) would hold a job for minutes.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "15")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

# The fitted chain's name on disk, for either kind of chain (runs already stored keep it). What it's called when it
# leaves the app, as a download or an MLflow artifact, says which kind it is: `model_file_name`.
MODEL_FILE = "classifier_chain.joblib"
REPORT_FILE = "report.json"
_KEY_UNSAFE = re.compile(r"[^\w./\- :]")
_PARAM_MAX = 6000


def model_file_name(report: dict | None) -> str:
    """`target_chain.joblib` when any target is a number, else `classifier_chain.joblib`."""
    report = report or {}
    kind = report.get("chain_kind")
    if kind not in ("classifier_chain", "target_chain"):  # a report from before chain_kind: read its target kinds
        numeric = "regression" in (report.get("target_kinds") or {}).values()
        kind = "target_chain" if numeric else "classifier_chain"
    return f"{kind}.joblib"


def enabled() -> bool:
    return bool(get_settings().mlflow_tracking_uri)


def experiment_prefix() -> str:
    return get_settings().mlflow_experiment_prefix.strip().strip("/") or "rainforest"


def experiment_name(dataset_id: int, dataset_name: str) -> str:
    return f"{experiment_prefix()}/{dataset_id}-{dataset_name}"


def browse_base() -> str:
    """The MLflow address browsers use, without a trailing slash."""
    return get_settings().mlflow_browse_url.strip().rstrip("/") or "/mlflow"


def mlflow_url(experiment_id: str | None, run_id: str | None) -> str | None:
    return f"{browse_base()}/#/experiments/{experiment_id}/runs/{run_id}" if experiment_id and run_id else None


def _key(*parts: str) -> str:
    return _KEY_UNSAFE.sub("_", ".".join(parts))


def _ms(ts: datetime | None) -> int | None:
    return int(ts.timestamp() * 1000) if ts else None


def run_params(report: dict) -> dict[str, str]:
    out = {
        "model": report.get("model"),
        "order": ",".join(report.get("order") or []),
        "cv": report.get("cv") or 0,
        "cv_folds": report.get("cv_folds") or 0,
        "device": report.get("device", "cpu"),
        "chain_method": report.get("chain_method"),
        "export_family": report.get("export_family"),
        "n_train": report.get("n_train"),
        "n_features": report.get("n_features"),
    }
    if "seed" in report:
        out["seed"] = report["seed"]
    out = {k: v for k, v in out.items() if v is not None}
    for k, v in (report.get("params") or {}).items():
        out[_key("estimator", k)] = v  # None included: it is a setting too (e.g. max_depth=None)
    for t, task in (report.get("target_tasks") or {}).items():
        out[_key("target", t, "task")] = task
    return {_key(k): str(v)[:_PARAM_MAX] for k, v in out.items()}


def run_metrics(report: dict) -> dict[str, float]:
    """`<split>.<target>.<metric>` for every scalar metric, plus the fit time."""
    out: dict[str, float] = {}
    for split, by_target in (report.get("metrics") or {}).items():
        for target, values in by_target.items():
            for name, v in values.items():
                if isinstance(v, (int, float)) and math.isfinite(v):
                    out[_key(split, target, name)] = float(v)
    for target, values in ((report.get("cross_validation") or {}).get("targets") or {}).items():
        for name, s in values.items():
            if isinstance(s.get("std"), (int, float)) and math.isfinite(s["std"]):
                out[_key("cv_std", target, name)] = float(s["std"])
    if isinstance(report.get("fit_seconds"), (int, float)):
        out["fit_seconds"] = float(report["fit_seconds"])
    return out


def run_tags(run: dict, dataset_name: str, report: dict, user: str | None) -> dict[str, str]:
    tags = {
        "rainforest.dataset_id": run["dataset_id"],
        "rainforest.dataset_name": dataset_name,
        "rainforest.export_id": run["export_id"],
        "rainforest.recipe_id": run.get("recipe_id"),
        "rainforest.recipe_version": run.get("recipe_version"),
        "rainforest.experiment_id": run.get("experiment_id"),
        "rainforest.experiment_name": run.get("experiment_name"),
        "rainforest.run_id": run["id"],
        "rainforest.model": report.get("model") or run.get("model"),
        "rainforest.model_family": report.get("export_family"),
        "rainforest.user": user,
    }
    if baseline := run.get("baseline"):
        tags["rainforest.baseline"] = baseline.get("label")
        tags["rainforest.baseline_family"] = baseline.get("family")
    if (adj := run.get("adjustments")) is not None:
        tags["rainforest.adjusted"] = adj.get("count", 0)
        tags["rainforest.targets_changed"] = "true" if adj.get("targets_changed") else "false"
    if sha := get_settings().git_sha:
        tags["rainforest.git_sha"] = sha
        tags["mlflow.source.git.commit"] = sha
    return {k: str(v) for k, v in tags.items() if v is not None}


def _adopt(client, exp, tags: dict[str, str]) -> str:
    if exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)
    for k, v in tags.items():
        client.set_experiment_tag(exp.experiment_id, k, v)
    return exp.experiment_id


def _experiment(client, dataset_id: int, dataset_name: str) -> str:
    from mlflow.exceptions import MlflowException

    prefix = experiment_prefix()
    found = client.search_experiments(
        filter_string=f"tags.`rainforest.dataset_id` = '{dataset_id}' AND tags.`rainforest.prefix` = '{prefix}'",
        max_results=1,
    )
    if found:
        return found[0].experiment_id
    name = experiment_name(dataset_id, dataset_name)
    # The kind makes the MLflow UI open these as model training, not GenAI, experiments.
    tags = {
        "rainforest.dataset_id": str(dataset_id),
        "rainforest.prefix": prefix,
        "mlflow.experimentKind": "custom_model_development",
    }
    # An experiment made before the prefix tag existed, by this name.
    exp = client.get_experiment_by_name(name)
    if exp is not None:
        return _adopt(client, exp, tags)
    try:
        return client.create_experiment(name, tags=tags)
    except MlflowException:
        # Another worker created it first, or one by this name was deleted in the MLflow UI.
        exp = client.get_experiment_by_name(name)
        if exp is None:
            raise
        return _adopt(client, exp, tags)


def log_run(
    run: dict,
    dataset_name: str,
    report: dict,
    model_dir: Path | None,
    user: str | None = None,
    tracking_uri: str | None = None,
) -> tuple[str, str]:
    """Log one finished model run; returns (experiment id, MLflow run id). Raises on any failure, after deleting
    a half-logged MLflow run so a retry (or the backfill) doesn't leave a duplicate behind.

    `run` is the meta.model_runs row (id, dataset_id, export_id, model, created_at, finished_at), with the export's
    recipe_id and recipe_version and the experiment_id and experiment_name when known; `report` is the training
    report including `metrics`.
    """
    from mlflow import MlflowClient
    from mlflow.entities import Metric, Param

    client = MlflowClient(tracking_uri=tracking_uri or get_settings().mlflow_tracking_uri)
    experiment_id = _experiment(client, run["dataset_id"], dataset_name)
    created = client.create_run(
        experiment_id,
        start_time=_ms(run.get("created_at")),
        tags=run_tags(run, dataset_name, report, user),
        run_name=f"run {run['id']} · {report.get('model_label') or report.get('model') or run.get('model')}",
    )
    mlflow_run_id = created.info.run_id
    try:
        step_ts = _ms(run.get("finished_at")) or _ms(datetime.now())
        prep = {"prep.adjusted_fields": run["adjustments"].get("count", 0)} if run.get("adjustments") else {}
        cfg = run.get("export_config") or {}
        prep |= {
            f"prep.{k}": cfg[c]
            for k, c in (("valid_size", "valid_size"), ("test_size", "test_size"), ("split_seed", "seed"))
            if cfg.get(c) is not None
        }
        params = [Param(k, str(v)) for k, v in {**run_params(report), **prep}.items()]
        metrics = [Metric(k, v, step_ts, 0) for k, v in run_metrics(report).items()]
        for i in range(0, max(len(params), len(metrics)), 100):
            client.log_batch(mlflow_run_id, metrics=metrics[i : i + 100], params=params[i : i + 100])
        if model_dir is not None:
            if (Path(model_dir) / REPORT_FILE).exists():
                client.log_artifact(mlflow_run_id, str(Path(model_dir) / REPORT_FILE))
            if (Path(model_dir) / MODEL_FILE).exists():
                _log_model_file(client, mlflow_run_id, Path(model_dir) / MODEL_FILE, model_file_name(report))
        manifest = Path(run["export_dir"]) / "manifest.json" if run.get("export_dir") else None
        if manifest is not None and manifest.exists():
            # Every field's preparation (type, imputation, transform, scaling, encoding), the split and the targets.
            client.log_artifact(mlflow_run_id, str(manifest), "preparation")
        if run.get("adjustments"):
            # How the experiment changed its baseline, field by field: the preparation behind these results.
            client.log_dict(mlflow_run_id, {"baseline": run.get("baseline"), **run["adjustments"]}, "adjustments.json")
        client.set_terminated(mlflow_run_id, "FINISHED", end_time=_ms(run.get("finished_at")))
    except BaseException:
        try:
            client.delete_run(mlflow_run_id)
        except Exception:  # noqa: BLE001
            pass
        raise
    return experiment_id, mlflow_run_id


def _log_model_file(client, mlflow_run_id: str, path: Path, name: str) -> None:
    """Log the chain under `model/<name>`; MLflow names an artifact after its file, so a target chain goes through a
    temporary link (or copy) carrying its own name."""
    if path.name == name:
        client.log_artifact(mlflow_run_id, str(path), "model")
        return
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        named = Path(tmp) / name
        try:
            os.link(path, named)
        except OSError:
            shutil.copyfile(path, named)
        client.log_artifact(mlflow_run_id, str(named), "model")


def error_message(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {' '.join(str(exc).split())}"[:500]


def log_model_run(run_id: int, report: dict, user: str | None = None) -> dict[str, str | None]:
    """Log a meta.model_runs row, tagged with the user who started it unless `user` is given. Returns what to store: the MLflow ids, or `mlflow_error` for the report.
    Never raises."""
    from ..db import fetch_one

    try:
        run = fetch_one(
            """SELECT r.id, r.dataset_id, r.export_id, r.model, r.created_at, r.finished_at, d.name AS dataset_name,
                      u.name || coalesce(' <' || u.email || '>', '') AS user_label,
                      e.recipe_id, rc.version AS recipe_version, r.experiment_id, x.name AS experiment_name,
                      e.config -> 'baseline' AS baseline, e.config -> 'adjustments' AS adjustments,
                      e.config AS export_config, e.output_dir AS export_dir
               FROM meta.model_runs r JOIN meta.datasets d ON d.id = r.dataset_id
               LEFT JOIN meta.exports e ON e.id = r.export_id
               LEFT JOIN meta.recipes rc ON rc.id = e.recipe_id
               LEFT JOIN meta.experiments x ON x.id = r.experiment_id
               LEFT JOIN meta.users u ON u.id = r.created_by WHERE r.id = %s""",
            (run_id,),
        )
        if run is None:
            raise LookupError(f"model run {run_id} not found")
        if run["finished_at"] is None:
            run["finished_at"] = datetime.now().astimezone()
        model_dir = Path(report["model_dir"]) if report.get("model_dir") else None
        experiment_id, mlflow_run_id = log_run(
            run, run["dataset_name"], report, model_dir, user or run.pop("user_label")
        )
        return {"mlflow_experiment_id": experiment_id, "mlflow_run_id": mlflow_run_id, "mlflow_error": None}
    except Exception as exc:  # noqa: BLE001
        log.warning("MLflow logging for model run %s failed: %s", run_id, exc)
        return {"mlflow_experiment_id": None, "mlflow_run_id": None, "mlflow_error": error_message(exc)}
