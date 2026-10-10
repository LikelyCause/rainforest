"""Train ClassifierChain models on finished exports and compare the runs."""
from __future__ import annotations

import json
import logging
import time
import traceback
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .. import gpu, jobs
from ..auth import CurrentUser
from ..db import execute, fetch_all, fetch_one, jsonb
from ..modeling import mlflow_log
from ..modeling.registry import MODELS, catalog, compatibility, param_problems
from ..schemas import RunRequest

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["models"])

MLFLOW_URL = """CASE WHEN mlflow_run_id IS NOT NULL
                   THEN '/mlflow/#/experiments/' || mlflow_experiment_id || '/runs/' || mlflow_run_id END AS mlflow_url"""
RUN_COLS = f"""id, export_id, dataset_id, model, family, status, status_message, error, config, metrics,
              report - 'top_features' AS report, created_at, updated_at, finished_at, mlflow_experiment_id,
              mlflow_run_id, {MLFLOW_URL}, created_by"""


@router.get("/models")
def list_models() -> list[dict]:
    return catalog()


@router.get("/system")
def system() -> dict:
    """Training hardware. `gpu` is null when no NVIDIA GPU is visible to the API container."""
    return {"gpu": gpu.detect()}


def _export_manifest(export_id: int) -> tuple[dict, dict]:
    exp = fetch_one("SELECT * FROM meta.exports WHERE id = %s", (export_id,))
    if exp is None:
        raise HTTPException(404, "Export not found")
    if exp["status"] != "done":
        raise HTTPException(409, "Export is not finished")
    manifest = json.loads((Path(exp["output_dir"]) / "manifest.json").read_text())
    return exp, manifest


@router.get("/exports/{export_id}/models")
def models_for_export(export_id: int) -> list[dict]:
    """Every registered model, with whether (and why not) it can train on this export."""
    exp, manifest = _export_manifest(export_id)
    out = []
    for m in catalog():
        spec = MODELS[m["key"]]
        reason = None if spec.available else "not installed"
        reason = reason or compatibility(spec, exp["model_family"], bool(manifest.get("contains_nan")))
        out.append({**m, "compatible": reason is None, "reason": reason})
    return out


@router.post("/exports/{export_id}/runs", status_code=202)
def start_run(export_id: int, req: RunRequest, user: CurrentUser) -> dict:
    exp, manifest = _export_manifest(export_id)
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
    if req.order and sorted(req.order) != sorted(manifest["targets"]):
        raise HTTPException(422, f"order must be a permutation of {manifest['targets']}")
    run = fetch_one(
        f"""INSERT INTO meta.model_runs (export_id, dataset_id, model, family, status, status_message, config, owner,
                                         created_by)
            VALUES (%s, %s, %s, %s, 'pending', 'Queued', %s, %s, %s) RETURNING {RUN_COLS}""",
        (export_id, exp["dataset_id"], spec.key, spec.family, jsonb(req.model_dump()), jobs.owner_id(), user["id"]),
    )
    jobs.submit(_run, run["id"], exp["output_dir"], req.model_dump())
    return run


@router.get("/datasets/{dataset_id}/runs")
def list_runs(dataset_id: int) -> list[dict]:
    return fetch_all(
        f"SELECT {RUN_COLS} FROM meta.model_runs WHERE dataset_id = %s ORDER BY created_at DESC", (dataset_id,)
    )


@router.get("/runs/{run_id}")
def get_run(run_id: int) -> dict:
    run = fetch_one(f"SELECT *, {MLFLOW_URL} FROM meta.model_runs WHERE id = %s", (run_id,))
    if run is None:
        raise HTTPException(404, "Run not found")
    return run


@router.get("/runs/{run_id}/files/{kind}")
def download(run_id: int, kind: str) -> FileResponse:
    run = get_run(run_id)
    files = {"model": ("classifier_chain.joblib", "application/octet-stream"), "report": ("report.json", "application/json")}
    if kind not in files or run["status"] != "done":
        raise HTTPException(404, "File not available")
    name, media = files[kind]
    path = Path(run["model_dir"]) / name
    if not path.exists():
        raise HTTPException(410, "Model files were removed")
    return FileResponse(path, media_type=media, filename=f"run{run_id}_{run['model']}_{name}")


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
        report = train(Path(export_dir), req["model"], req.get("order"), req["seed"], params,
                       req["cv"] or None, out_dir, progress, device=req.get("device", "cpu"))
        logged: dict = {}
        if mlflow_log.enabled():
            progress("Logging to MLflow")
            logged = mlflow_log.log_model_run(run_id, report)
            if logged["mlflow_error"]:
                report["mlflow_error"] = logged["mlflow_error"]
        execute(
            """UPDATE meta.model_runs SET status = 'done', error = NULL, status_message = 'Done', metrics = %s,
                      report = %s, mlflow_experiment_id = %s, mlflow_run_id = %s,
                      model_dir = %s, finished_at = now(), updated_at = now() WHERE id = %s""",
            (jsonb(report["metrics"]), jsonb({k: v for k, v in report.items() if k != "metrics"}),
             logged.get("mlflow_experiment_id"), logged.get("mlflow_run_id"), report["model_dir"], run_id),
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("model run %s failed", run_id)
        execute(
            """UPDATE meta.model_runs SET status = 'failed', status_message = 'Failed', error = %s,
                      finished_at = now(), updated_at = now() WHERE id = %s""",
            (f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc(limit=4)}", run_id),
        )
