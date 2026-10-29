"""Deployments: deploy finished runs of one export, retire and reactivate them, manage their API keys, and score
records through `POST /api/deployments/{id}/predict` with a session or a key."""

from __future__ import annotations

import json

import polars as pl
from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from .. import deployments, experiments, scoring
from ..auth import PREDICT_MAX_BODY, CurrentUser
from ..config import get_settings
from ..db import connection, fetch_all, json_safe, jsonb
from ..schemas import DeploymentKeyRequest, DeploymentRequest
from ._lock import refuse_while_refreshing

router = APIRouter(prefix="/api", tags=["deployments"])

COLS = """d.id, d.dataset_id, d.experiment_id, x.name AS experiment_name, d.export_id, d.batch_id,
          (SELECT name FROM meta.run_batches b WHERE b.id = d.batch_id) AS batch_name, d.name, d.models, d.status,
          d.created_by, (SELECT name FROM meta.users u WHERE u.id = d.created_by) AS created_by_name, d.created_at,
          d.retired_at, (SELECT count(*) FROM meta.deployment_keys k
                          WHERE k.deployment_id = d.id AND k.revoked_at IS NULL) AS keys"""
FROM = "meta.deployments d JOIN meta.experiments x ON x.id = d.experiment_id"
KEY_COLS = """k.id, k.deployment_id, k.name, k.prefix, k.created_by,
              (SELECT name FROM meta.users u WHERE u.id = k.created_by) AS created_by_name, k.created_at,
              k.last_used_at, k.revoked_at"""
RETIRED = "This deployment is retired; reactivate it to score with it."
FILES_GONE = "The deployment's export or model files were removed"


def _can_manage(row: dict, user: dict) -> bool:
    return user["role"] == "lead" or row["created_by"] == user["id"]


def _row(conn, deployment_id: int, user: dict | None, lock: bool = False) -> dict:
    """The deployment, or 404 when it doesn't exist or `user` can't see its experiment."""
    visible = "" if user is None else f" AND d.experiment_id IN ({experiments.visible_ids(user['id'])})"
    # FOR UPDATE OF d: the row's own lock, not one on the joined experiment.
    row = conn.execute(
        f"SELECT {COLS} FROM {FROM} WHERE d.id = %s{visible}{' FOR UPDATE OF d' if lock else ''}", (deployment_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(404, "Deployment not found")
    return row


def _shown(row: dict, user: dict) -> dict:
    return {**row, "can_manage": _can_manage(row, user)}


def _managed(conn, deployment_id: int, user: dict) -> dict:
    """Retiring, reactivating and keys: the deployment's creator or a lead."""
    row = _row(conn, deployment_id, user, lock=True)
    if not _can_manage(row, user):
        raise HTTPException(403, "Only the deployment's creator or a lead can change it.")
    return row


def _chosen_runs(conn, x: dict, req: DeploymentRequest) -> list[dict]:
    """The runs `req` names: its run ids, or a one-vs-rest batch's finished runs (some of them, with run ids)."""
    if req.batch_id is not None:
        batch = conn.execute(
            "SELECT id, kind, status FROM meta.run_batches WHERE id = %s AND experiment_id = %s",
            (req.batch_id, x["id"]),
        ).fetchone()
        if batch is None:
            raise HTTPException(404, "Batch not found in this experiment")
        if req.run_ids is None:
            if batch["kind"] == "search":
                raise HTTPException(422, "A search batch is deployed by choosing its rows: give their run_ids")
            if batch["status"] == "running":
                raise HTTPException(409, "The batch is still running; deploy it once it ends, or choose its runs")
            return conn.execute(
                """SELECT id, export_id, status, report FROM meta.model_runs
                    WHERE batch_id = %s AND status = 'done' ORDER BY id""",
                (batch["id"],),
            ).fetchall()
    if not req.run_ids:
        raise HTTPException(422, "Choose the runs to deploy, or a one-vs-rest batch")
    ids = list(dict.fromkeys(req.run_ids))
    rows = {
        r["id"]: r
        for r in conn.execute(
            """SELECT id, export_id, status, report, batch_id FROM meta.model_runs
                WHERE id = ANY(%s) AND experiment_id = %s""",
            (ids, x["id"]),
        ).fetchall()
    }
    if missing := [i for i in ids if i not in rows]:
        raise HTTPException(422, f"Not runs of this experiment: {', '.join(map(str, missing))}")
    if req.batch_id is not None and (outside := [i for i in ids if rows[i]["batch_id"] != req.batch_id]):
        raise HTTPException(422, f"Not runs of batch {req.batch_id}: {', '.join(map(str, outside))}")
    return [rows[i] for i in ids]


@router.post("/experiments/{experiment_id}/deployments", status_code=201)
def create_deployment(experiment_id: int, req: DeploymentRequest, user: CurrentUser) -> dict:
    """Check what may share the deployment, resolve its thresholds, and score the export's test rows through it
    (`scoring.equality_check`) before anything is stored."""
    with connection() as conn:
        x = experiments.get_visible(conn, experiment_id, user)
        refuse_while_refreshing(conn, x["dataset_id"])
        experiments.require_writable(x, user)
        runs = _chosen_runs(conn, x, req)
        try:
            export_id, models = deployments.compose(runs, req.threshold_rule, req.thresholds)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        batch = conn.execute("SELECT name FROM meta.run_batches WHERE id = %s", (req.batch_id,)).fetchone()
    # Outside a transaction: the check reads model files and scores rows, which takes a while.
    with connection() as conn:
        try:
            problem = scoring.equality_check(conn, export_id, models)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(410, FILES_GONE) from exc
    if problem:
        raise HTTPException(422, f"Scoring through the deployment doesn't reproduce training: {problem}")
    name = (req.name or "").strip() or (batch["name"] if batch else models[0]["model_label"] or "Deployment")
    with connection() as conn:
        refuse_while_refreshing(conn, x["dataset_id"])
        experiments.require_writable(experiments.get_visible(conn, experiment_id, user), user)
        new_id = conn.execute(
            """INSERT INTO meta.deployments (dataset_id, experiment_id, export_id, batch_id, name, models, created_by)
               VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (x["dataset_id"], experiment_id, export_id, req.batch_id, name, jsonb(models), user["id"]),
        ).fetchone()["id"]
    return get_deployment(new_id, user)


@router.get("/datasets/{dataset_id}/deployments")
def list_deployments(dataset_id: int, user: CurrentUser) -> list[dict]:
    rows = fetch_all(
        f"""SELECT {COLS} FROM {FROM} WHERE d.dataset_id = %s
               AND d.experiment_id IN ({experiments.visible_ids(user["id"])}) ORDER BY d.id DESC""",
        (dataset_id,),
    )
    return [_shown(r, user) for r in rows]


@router.get("/deployments/{deployment_id}")
def get_deployment(deployment_id: int, user: CurrentUser) -> dict:
    """The deployment with `fields` (what a record must carry) and `example` (two of the dataset's rows of them), for
    the page's `curl` example; both null, with `problem`, when the export can't be read."""
    with connection() as conn:
        row = _row(conn, deployment_id, user)
        path = conn.execute("SELECT parquet_path FROM meta.datasets WHERE id = %s", (row["dataset_id"],)).fetchone()
        try:
            inputs = scoring.inputs(conn, row["export_id"])
        except (ValueError, OSError) as exc:
            return {**_shown(row, user), "fields": None, "example": None, "problem": str(exc) or FILES_GONE}
    fields = [{"name": i.name, "dtype": str(i.dtype)} for i in inputs]
    try:
        example = pl.read_parquet(path["parquet_path"], columns=[i.name for i in inputs], n_rows=2).to_dicts()
    except (OSError, pl.exceptions.PolarsError, TypeError):
        example = None
    return {**_shown(row, user), "fields": fields, "example": json_safe(example), "problem": None}


@router.post("/deployments/{deployment_id}/retire")
def retire(deployment_id: int, user: CurrentUser) -> dict:
    """Stop answering (`410`); the record, keys and tests stay. Never held back by a refresh: stopping access
    shouldn't wait."""
    with connection() as conn:
        row = _managed(conn, deployment_id, user)
        if row["status"] == "retired":
            raise HTTPException(409, "The deployment is already retired")
        conn.execute("UPDATE meta.deployments SET status = 'retired', retired_at = now() WHERE id = %s", (row["id"],))
    scoring.forget(deployment_id)
    return get_deployment(deployment_id, user)


@router.post("/deployments/{deployment_id}/activate")
def activate(deployment_id: int, user: CurrentUser) -> dict:
    with connection() as conn:
        row = _managed(conn, deployment_id, user)
        refuse_while_refreshing(conn, row["dataset_id"])
        if row["status"] == "active":
            raise HTTPException(409, "The deployment is already active")
        conn.execute("UPDATE meta.deployments SET status = 'active', retired_at = NULL WHERE id = %s", (row["id"],))
    scoring.forget(deployment_id)
    return get_deployment(deployment_id, user)


# ------------------------------------------------------------------------------------------- keys
@router.post("/deployments/{deployment_id}/keys", status_code=201)
def create_key(deployment_id: int, body: DeploymentKeyRequest, user: CurrentUser) -> dict:
    """A new key, returned once: only its SHA-256 and first 8 characters are kept."""
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "A key needs a name")
    key, prefix, digest = deployments.new_key()
    with connection() as conn:
        row = _managed(conn, deployment_id, user)
        refuse_while_refreshing(conn, row["dataset_id"])
        new_id = conn.execute(
            """INSERT INTO meta.deployment_keys (deployment_id, name, prefix, hash, created_by)
               VALUES (%s, %s, %s, %s, %s) RETURNING id""",
            (deployment_id, name, prefix, digest, user["id"]),
        ).fetchone()["id"]
        shown = conn.execute(f"SELECT {KEY_COLS} FROM meta.deployment_keys k WHERE k.id = %s", (new_id,)).fetchone()
    return {**shown, "key": key}


@router.get("/deployments/{deployment_id}/keys")
def list_keys(deployment_id: int, user: CurrentUser) -> list[dict]:
    with connection() as conn:
        _row(conn, deployment_id, user)
        return conn.execute(
            f"SELECT {KEY_COLS} FROM meta.deployment_keys k WHERE k.deployment_id = %s ORDER BY k.id", (deployment_id,)
        ).fetchall()


@router.delete("/deployments/{deployment_id}/keys/{key_id}")
def revoke_key(deployment_id: int, key_id: int, user: CurrentUser) -> dict:
    """Revoke a key; the next call with it is `401`. Never held back by a refresh."""
    with connection() as conn:
        _managed(conn, deployment_id, user)
        row = conn.execute(
            f"""UPDATE meta.deployment_keys k SET revoked_at = coalesce(revoked_at, now())
                 WHERE k.id = %s AND k.deployment_id = %s RETURNING {KEY_COLS}""",
            (key_id, deployment_id),
        ).fetchone()
    if row is None:
        raise HTTPException(404, "Key not found")
    return row


# ------------------------------------------------------------------------------------------- predict
async def _body(request: Request) -> bytes:
    """The body, refused once it passes the limit: a chunked request carries no length to check up front."""
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > PREDICT_MAX_BODY:
            raise HTTPException(413, "A predict request can be 5 MB at most")
        chunks.append(chunk)
    return b"".join(chunks)


def _refuse_constant(name: str):
    raise ValueError(f"{name} isn't a JSON number")


def _records(body: bytes) -> list[dict]:
    try:
        payload = json.loads(body, parse_constant=_refuse_constant)
    except ValueError as exc:
        raise HTTPException(422, f"The body isn't valid JSON: {exc}") from exc
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise HTTPException(422, 'Send {"records": [{"<field>": value, ...}, ...]}')
    if bad := next((i for i, r in enumerate(records, 1) if not isinstance(r, dict)), None):
        raise HTTPException(422, f"Record {bad} isn't an object of field values")
    most = get_settings().predict_max_records
    if len(records) > most:
        raise HTTPException(
            413, f"At most {most:,} records per request (PREDICT_MAX_RECORDS); this one has {len(records):,}"
        )
    return records


def _predict(deployment_id: int, request: Request, records: list[dict]) -> dict:
    user = getattr(request.state, "user", None)
    with connection() as conn:
        # A key was matched to this deployment before the route ran; a session sees only what its user may.
        row = _row(conn, deployment_id, user)
        if row["status"] != "active":
            raise HTTPException(410, RETIRED)
        try:
            scorer = scoring.load(conn, {"id": row["id"], "export_id": row["export_id"], "models": row["models"]})
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(410, FILES_GONE) from exc
    out = {"deployment": deployment_id, "models": deployments.labels(scorer), "results": [], "warnings": {}}
    if not records:
        return out
    try:
        scored = scorer.score(deployments.records_frame(records, scorer.required))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {**out, "results": deployments.results(scored, len(records)), "warnings": scored.warnings}


@router.post("/deployments/{deployment_id}/predict")
async def predict(deployment_id: int, request: Request) -> dict:
    """Score `{"records": [...]}` with the deployment's models, in record order. A session (not a Viewer's) or
    `Authorization: Bearer <key>` of this deployment; the middleware has checked which."""
    records = _records(await _body(request))
    return await run_in_threadpool(_predict, deployment_id, request, records)
