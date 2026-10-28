"""Train target chains on finished exports and compare the runs."""

from __future__ import annotations

import json
import logging
import time
import traceback
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .. import experiments, gpu, jobs
from ..auth import CurrentUser
from ..db import connection, execute, fetch_all, fetch_one, jsonb
from ..modeling import mlflow_log
from ..modeling import params as model_params
from ..modeling.registry import (
    MODELS,
    PLANNED,
    catalog,
    compatibility,
    dry_fit_problem,
    label_for,
    one_vs_rest_target,
    order_problem,
    ovr_label,
    param_problems,
    regression_problem,
    weighting_clash,
)
from ..schemas import RunRequest
from ._lock import refuse_while_refreshing
from .exports import visible_export

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["models"])

RUN_COLS = """id, export_id, dataset_id, model, family, status, status_message, error, config, metrics,
              report, created_at, updated_at, finished_at, mlflow_experiment_id,
              mlflow_run_id, created_by, experiment_id, note, chosen_at,
              (SELECT name FROM meta.users u WHERE u.id = chosen_by) AS chosen_by_name"""


def _with_url(row: dict | None) -> dict | None:
    """The run row with its MLflow link, built from `MLFLOW_BROWSE_URL`."""
    if row is None:
        return None
    return {**row, "mlflow_url": mlflow_log.mlflow_url(row["mlflow_experiment_id"], row["mlflow_run_id"])}


@router.get("/models")
def list_models() -> list[dict]:
    return catalog()


@router.get("/system")
def system() -> dict:
    """Training hardware (`gpu` is null when no NVIDIA GPU is visible to the API container), and whether runs are
    logged to MLflow and where browsers open it."""
    enabled = mlflow_log.enabled()
    browse_url = mlflow_log.browse_base() if enabled else None
    return {"gpu": gpu.detect(), "mlflow": {"enabled": enabled, "browseUrl": browse_url}}


def _export_manifest(export_id: int, user: dict) -> tuple[dict, dict]:
    exp = visible_export(export_id, user)
    if exp["status"] != "done":
        raise HTTPException(409, "Export is not finished")
    try:
        manifest = json.loads((Path(exp["output_dir"]) / "manifest.json").read_text())
    except (OSError, ValueError) as exc:
        raise HTTPException(410, "Export files were removed") from exc
    return exp, manifest


@router.get("/exports/{export_id}/models")
def models_for_export(export_id: int, user: CurrentUser) -> list[dict]:
    """Every registered model, with whether (and why not) it can train on this export."""
    exp, manifest = _export_manifest(export_id, user)
    out = []
    for m in catalog():
        if m["planned"]:
            out.append({**m, "compatible": False, "reason": "planned"})
            continue
        spec = MODELS[m["key"]]
        reason = None if spec.available else "not installed"
        reason = reason or compatibility(spec, exp["model_family"], bool(manifest.get("contains_nan")))
        out.append({**m, "compatible": reason is None, "reason": reason})
    return out


@router.post("/exports/{export_id}/runs", status_code=202)
def start_run(export_id: int, req: RunRequest, user: CurrentUser) -> dict:
    exp, manifest = _export_manifest(export_id, user)
    if req.experiment_id is not None and req.experiment_id != exp["experiment_id"]:
        raise HTTPException(422, f"Export #{export_id} belongs to experiment {exp['experiment_id']}; its runs do too")
    if req.model in PLANNED:
        raise HTTPException(422, f"{PLANNED[req.model]['label']} is planned and can't be trained yet.")
    spec = MODELS.get(req.model)
    if spec is None:
        raise HTTPException(422, f"Unknown model {req.model!r}")
    if not spec.available:
        raise HTTPException(422, f"{spec.label} is not installed")
    reason = compatibility(spec, exp["model_family"], bool(manifest.get("contains_nan")))
    if reason:
        raise HTTPException(422, reason)
    if req.device not in spec.devices:
        raise HTTPException(422, f"{spec.label} can't train on {req.device!r} (supports: {', '.join(spec.devices)})")
    if req.device == "cuda" and not gpu.gpu_available():
        info = gpu.detect()
        raise HTTPException(409, f"No usable GPU: {info['reason']}" if info else "No GPU is available to the API")
    if problems := param_problems(spec, req.params):
        raise HTTPException(422, {"message": "Invalid model parameters", "errors": problems})
    one_vs_rest = req.positive_class is not None
    if one_vs_rest:
        req.target, problem = one_vs_rest_target(manifest, req.target, req.positive_class)
        if problem:
            raise HTTPException(422, problem)
        if req.order is not None and req.order != [req.target]:
            raise HTTPException(422, f"A one-vs-rest run trains only {req.target}; leave the order out")
        if req.balance and (clash := weighting_clash(spec, req.params)):
            raise HTTPException(422, f"Choose balance or set {', '.join(clash)}, not both")
        targets, kinds = [req.target], ["binary"]
    else:
        if req.target is not None or req.balance:
            raise HTTPException(422, "target and balance apply to a one-vs-rest run; set positive_class too")
        if req.order is not None and (problem := order_problem(req.order, manifest["targets"])):
            raise HTTPException(422, problem)
        targets = req.order or manifest["targets"]
        kinds = [(manifest.get("target_kinds") or {}).get(t) for t in targets]
    if problem := regression_problem(spec, kinds):
        raise HTTPException(422, problem)
    if model_params.settings_for(spec.key):
        req.params = model_params.normalize(spec.key, req.params)
        if problems := model_params.setting_problems(spec.key, req.params):
            raise HTTPException(422, {"message": "Invalid model settings", "errors": problems})
        stats = manifest.get("target_stats") or {}
        for t, kind in zip(targets, kinds):
            if kind == "regression" and (
                p := model_params.objective_problem(req.params, (stats.get(t) or {}).get("min"))
            ):
                raise HTTPException(422, p)
        if problem := dry_fit_problem(spec, req.params, kinds):
            raise HTTPException(422, problem)
    # A run that isn't one-vs-rest stores the config it always has.
    stored = {"experiment_id"} if one_vs_rest else {"experiment_id", "positive_class", "target", "balance"}
    label = ovr_label(spec, req.target, req.positive_class) if one_vs_rest else label_for(spec, kinds)
    with connection() as conn:
        # Recorded under the row lock, so a refresh starting now sees the run and refuses.
        refuse_while_refreshing(conn, exp["dataset_id"])
        experiments.require_writable(experiments.get_visible(conn, exp["experiment_id"], user), user)
        # Named for its targets from the start ("Ridge regression" for numbers), so a queued or failed run reads right.
        run = conn.execute(
            f"""INSERT INTO meta.model_runs (export_id, dataset_id, model, family, status, status_message, config,
                                             owner, created_by, experiment_id, report)
                VALUES (%s, %s, %s, %s, 'pending', 'Queued', %s, %s, %s, %s, %s) RETURNING {RUN_COLS}""",
            (
                export_id,
                exp["dataset_id"],
                spec.key,
                spec.family,
                jsonb(req.model_dump(exclude=stored)),
                jobs.owner_id(),
                user["id"],
                exp["experiment_id"],
                jsonb({"model_label": label}),
            ),
        ).fetchone()
    jobs.submit(_run, run["id"], exp["output_dir"], req.model_dump())
    return _with_url(run)


@router.get("/datasets/{dataset_id}/runs")
def list_runs(dataset_id: int, user: CurrentUser, experiment_id: int | None = None) -> list[dict]:
    rows = fetch_all(
        f"""SELECT {RUN_COLS} FROM meta.model_runs
             WHERE dataset_id = %s AND experiment_id IN ({experiments.visible_ids(user["id"])})
               AND (%s::bigint IS NULL OR experiment_id = %s) ORDER BY created_at DESC""",
        (dataset_id, experiment_id, experiment_id),
    )
    return [_with_url(r) for r in rows]


def run_row(run_id: int, user_id: int | None = None) -> dict | None:
    """One run with its MLflow link; with `user_id`, only if that user may see its experiment."""
    visible = "" if user_id is None else f" AND experiment_id IN ({experiments.visible_ids(user_id)})"
    row = fetch_one(
        f"""SELECT *, (SELECT name FROM meta.users u WHERE u.id = chosen_by) AS chosen_by_name
                          FROM meta.model_runs WHERE id = %s{visible}""",
        (run_id,),
    )
    return _with_url(row)


@router.get("/runs/{run_id}")
def get_run(run_id: int, user: CurrentUser) -> dict:
    run = run_row(run_id, user["id"])
    if run is None:
        raise HTTPException(404, "Run not found")
    return run


class RunPatch(BaseModel):
    note: str | None = Field(None, max_length=4000)
    chosen: bool | None = Field(
        None,
        description="Lead only: mark this run as the experiment's chosen model (one per "
        "experiment; marking another moves it)",
    )


@router.patch("/runs/{run_id}")
def update_run(run_id: int, body: RunPatch, user: CurrentUser) -> dict:
    """The run's note (its author or a lead), or whether it is the experiment's chosen model (a lead)."""
    run = get_run(run_id, user)
    fields = body.model_fields_set
    if "note" in fields and user["role"] != "lead" and run["created_by"] != user["id"]:
        raise HTTPException(403, "Only the run's author or a lead can change its note")
    if "chosen" in fields:
        if user["role"] != "lead":
            raise HTTPException(403, "A lead chooses the team's model")
        if body.chosen and run["status"] != "done":
            raise HTTPException(409, "Only a finished run can be chosen")
    with connection() as conn:
        refuse_while_refreshing(conn, run["dataset_id"])
        if "note" in fields:
            conn.execute(
                "UPDATE meta.model_runs SET note = %s WHERE id = %s", ((body.note or "").strip() or None, run_id)
            )
        if "chosen" in fields:
            conn.execute(
                "UPDATE meta.model_runs SET chosen_at = NULL, chosen_by = NULL WHERE experiment_id = %s "
                "AND chosen_at IS NOT NULL",
                (run["experiment_id"],),
            )
            if body.chosen:
                conn.execute(
                    "UPDATE meta.model_runs SET chosen_at = now(), chosen_by = %s WHERE id = %s", (user["id"], run_id)
                )
    return get_run(run_id, user)


@router.get("/runs/{run_id}/files/{kind}")
def download(run_id: int, kind: str, user: CurrentUser) -> FileResponse:
    run = get_run(run_id, user)
    # The file on disk keeps one name for either kind of chain; the download says which kind it is.
    files = {
        "model": (mlflow_log.MODEL_FILE, mlflow_log.model_file_name(run["report"]), "application/octet-stream"),
        "report": (mlflow_log.REPORT_FILE, mlflow_log.REPORT_FILE, "application/json"),
    }
    if kind not in files:
        raise HTTPException(404, f"Unknown file {kind!r}")
    if run["status"] != "done":
        raise HTTPException(409, "Run is not finished")
    stored, name, media = files[kind]
    path = Path(run["model_dir"]) / stored
    if not path.exists():
        raise HTTPException(410, "Model files were removed")
    return FileResponse(path, media_type=media, filename=f"run{run_id}_{run['model']}_{name}")


def _finish(conn, run_id: int, report: dict, logged: dict) -> None:
    conn.execute(
        """UPDATE meta.model_runs SET status = 'done', error = NULL, status_message = 'Done', metrics = %s,
                  report = %s, mlflow_experiment_id = %s, mlflow_run_id = %s,
                  model_dir = %s, finished_at = now(), updated_at = now() WHERE id = %s""",
        (
            jsonb(report["metrics"]),
            jsonb({k: v for k, v in report.items() if k != "metrics"}),
            logged.get("mlflow_experiment_id"),
            logged.get("mlflow_run_id"),
            report["model_dir"],
            run_id,
        ),
    )


def _run(run_id: int, export_dir: str, req: dict) -> None:
    from ..config import get_settings
    from ..modeling.train import train

    def progress(msg: str) -> None:
        execute(
            "UPDATE meta.model_runs SET status = 'running', error = NULL, status_message = %s, updated_at = now() "
            "WHERE id = %s",
            (msg, run_id),
        )

    try:
        progress("Loading export")
        out_dir = get_settings().models_dir / f"run{run_id}_{req['model']}_{int(time.time())}"
        params = req.get("params") or {}
        if "n_jobs" in MODELS[req["model"]].default_params:
            params = {"n_jobs": jobs.cpu_threads(), **params}
        report = train(
            Path(export_dir),
            req["model"],
            req.get("order"),
            req["seed"],
            params,
            req["cv"] or None,
            out_dir,
            progress,
            device=req.get("device", "cpu"),
            cv_folds=req.get("cv_folds", 0),
            target=req.get("target"),
            positive_class=req.get("positive_class"),
            balance=req.get("balance", False),
        )
        if mlflow_log.enabled():
            with connection() as conn:
                # MLflow has no access control, so a private experiment's run is logged once review opens.
                if experiments.run_is_private(conn, run_id):
                    _finish(conn, run_id, {**report, "mlflow_pending": True}, {})
                    return
            progress("Logging to MLflow")
            logged = mlflow_log.log_model_run(run_id, report)
            if logged["mlflow_error"]:
                report["mlflow_error"] = logged["mlflow_error"]
        else:
            logged = {}
        with connection() as conn:
            _finish(conn, run_id, report, logged)
    except Exception as exc:  # noqa: BLE001
        log.exception("model run %s failed", run_id)
        execute(
            """UPDATE meta.model_runs SET status = 'failed', status_message = 'Failed', error = %s,
                      finished_at = now(), updated_at = now() WHERE id = %s""",
            (f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc(limit=4)}", run_id),
        )
