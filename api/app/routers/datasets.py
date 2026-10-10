from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

import polars as pl

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile

from .. import jobs
from ..config import get_settings
from ..correlations import target_associations
from ..db import connection, execute, fetch_all, fetch_one
from ..modeling.registry import MODELS
from ..processing import _Status, delete_dataset_files, process_dataset, upsert_correlations
from ..profiling import corr_vector, sample_indices
from ..sample_data import generate
from ..schemas import DatasetPatch, SampleRequest
from ..semantics import FAMILIES, NUMERIC
from .fields import _sync_targets, has_screen_fields, screen_pairs, screen_targets, set_targets

log = logging.getLogger(__name__)
CORRELATED_TYPES = NUMERIC | {"datetime", "categorical", "binary"}
router = APIRouter(prefix="/api/datasets", tags=["datasets"])

DATASET_COLS = """id, name, source_filename, file_format, target_columns, model_family, row_count, column_count, status,
                  progress, status_message, error, warnings, created_at, updated_at, profiled_at, raw_table"""


def _targets(raw: str | None) -> list[str]:
    # Targets are picked from the real columns after upload; none is a valid starting point.
    if raw is None:
        return []
    return list(dict.fromkeys(t.strip() for t in raw.split(",") if t.strip()))


def _create(name: str, filename: str | None, fmt: str, targets: list[str], upload_path: Path | None = None) -> dict:
    return fetch_one(
        f"""INSERT INTO meta.datasets (name, source_filename, file_format, target_columns, status, status_message,
                                       owner, upload_path)
            VALUES (%s, %s, %s, %s, 'pending', 'Queued', %s, %s) RETURNING {DATASET_COLS}""",
        (name, filename, fmt, targets, jobs.owner_id(), None if upload_path is None else str(upload_path)),
    )


# Metric shown for a run in the dataset list, in order of preference.
HEADLINE_METRICS = ("roc_auc", "roc_auc_ovr_macro", "accuracy")


@router.get("")
def list_datasets() -> list[dict]:
    rows = fetch_all(
        f"""SELECT {", ".join("d." + c.strip() for c in DATASET_COLS.split(","))},
                   coalesce(c.field_count, 0) AS field_count, coalesce(c.reviewed_count, 0) AS reviewed_count,
                   coalesce(c.kept_count, 0) AS kept_count, coalesce(c.dropped_count, 0) AS dropped_count,
                   coalesce(c.target_count, 0) AS target_count,
                   (SELECT count(*) FROM meta.exports x WHERE x.dataset_id = d.id) AS export_count,
                   (SELECT count(*) FROM meta.model_runs x WHERE x.dataset_id = d.id) AS run_count,
                   e.id AS e_id, e.model_family AS e_model_family, e.status AS e_status,
                   e.created_at AS e_created_at, e.n_features AS e_n_features,
                   r.id AS r_id, r.model AS r_model, r.status AS r_status, r.report -> 'order' AS r_order,
                   r.report ->> 'model_label' AS r_label, r.metrics -> 'test' AS r_test
              FROM meta.datasets d
              LEFT JOIN (SELECT dataset_id, count(*) AS field_count,
                                count(*) FILTER (WHERE reviewed) AS reviewed_count,
                                count(*) FILTER (WHERE action = 'keep') AS kept_count,
                                count(*) FILTER (WHERE action = 'drop') AS dropped_count,
                                count(*) FILTER (WHERE action = 'target') AS target_count
                           FROM meta.field_decisions GROUP BY dataset_id) c ON c.dataset_id = d.id
              LEFT JOIN LATERAL (SELECT * FROM meta.exports x WHERE x.dataset_id = d.id
                                  ORDER BY x.created_at DESC, x.id DESC LIMIT 1) e ON true
              LEFT JOIN LATERAL (SELECT * FROM meta.model_runs x WHERE x.dataset_id = d.id
                                  ORDER BY x.created_at DESC, x.id DESC LIMIT 1) r ON true
             ORDER BY d.created_at DESC"""
    )
    return [_with_latest(row) for row in rows]


def _with_latest(row: dict) -> dict:
    out = {k: v for k, v in row.items() if not k.startswith(("e_", "r_"))}
    out["last_export"] = None if row["e_id"] is None else {
        "id": row["e_id"], "model_family": row["e_model_family"], "status": row["e_status"],
        "created_at": row["e_created_at"], "n_features": row["e_n_features"],
    }
    out["last_run"] = None if row["r_id"] is None else {
        "id": row["r_id"], "model": row["r_model"],
        "model_label": row["r_label"] or (MODELS[row["r_model"]].label if row["r_model"] in MODELS else row["r_model"]),
        "status": row["r_status"],
        **headline_metric(row["r_status"], row["r_order"], row["r_test"]),
    }
    return out


def headline_metric(status: str, order: list[str] | None, test: dict | None) -> dict:
    """Test-split metric for the first target in chain order, so one number can stand for a run."""
    if status == "done" and order and test:
        by_metric = test.get(order[0]) or {}
        for m in HEADLINE_METRICS:
            if isinstance(by_metric.get(m), (int, float)):
                return {"metric": m, "value": float(by_metric[m])}
    return {"metric": None, "value": None}


@router.post("/upload", status_code=202)
def upload(
    file: UploadFile = File(...),
    name: str | None = Form(None),
    targets: str | None = Form(None, description="Comma-separated target columns"),
) -> dict:
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()
    fmt = {".csv": "csv", ".txt": "csv", ".tsv": "csv", ".parquet": "parquet", ".pq": "parquet"}.get(suffix)
    if fmt is None:
        raise HTTPException(415, "Only .csv and .parquet files are supported")
    if suffix == ".tsv":
        raise HTTPException(415, "TSV is not supported yet; convert to CSV or Parquet")
    s = get_settings()
    s.uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = s.uploads_dir / f"{uuid.uuid4().hex}{suffix}"
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out, length=8 * 1024 * 1024)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    if dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "Uploaded file is empty")
    try:
        ds = _create(name or Path(filename).stem, filename, fmt, _targets(targets), dest)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    jobs.submit(process_dataset, ds["id"], str(dest), fmt)
    return ds


@router.get("/sample/limits")
def sample_limits() -> dict:
    return {"max_cells": get_settings().sample_max_cells}


@router.post("/sample", status_code=202)
def create_sample(req: SampleRequest) -> dict:
    cap = get_settings().sample_max_cells
    if req.rows * req.fields > cap:
        raise HTTPException(
            422, f"rows × fields must be at most {cap:,} (asked for {req.rows:,} × {req.fields} = "
                 f"{req.rows * req.fields:,}). With {req.fields} fields, use up to {cap // req.fields:,} rows.")
    ds = _create(
        req.name or f"sample-{req.rows // 1000}k-{req.fields}f-{req.target_kind}",
        None, "generated", ["column_a", "column_b"],
    )
    jobs.submit(_generate_and_process, ds["id"], req.model_dump())
    return ds


def _generate_and_process(dataset_id: int, req: dict) -> None:
    s = get_settings()
    out = s.datasets_dir / str(dataset_id) / "raw.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        execute(
            "UPDATE meta.datasets SET status='ingesting', error=NULL, status_message='Generating sample data', "
            "updated_at=now() WHERE id=%s",
            (dataset_id,),
        )
        generate(req["rows"], req["fields"], req["seed"], req["target_kind"]).write_parquet(
            out, row_group_size=100_000)  # as ingest writes uploads
    except Exception as exc:  # noqa: BLE001
        execute(
            "UPDATE meta.datasets SET status='failed', error=%s, updated_at=now() WHERE id=%s",
            (f"sample generation failed: {exc}", dataset_id),
        )
        return
    process_dataset(dataset_id, None, "parquet")


@router.get("/{dataset_id}")
def get_dataset(dataset_id: int) -> dict:
    ds = fetch_one(f"SELECT {DATASET_COLS} FROM meta.datasets WHERE id = %s", (dataset_id,))
    if ds is None:
        raise HTTPException(404, "Dataset not found")
    return ds


@router.patch("/{dataset_id}")
def update_dataset(dataset_id: int, body: DatasetPatch) -> dict:
    """Rename a dataset, switch the model family its review settings are shown for, or choose its targets."""
    ds = get_dataset(dataset_id)
    if body.model_family is not None and body.model_family not in FAMILIES:
        raise HTTPException(422, f"Unknown model family {body.model_family!r}")
    if body.target_columns is not None and ds["status"] != "ready":
        raise HTTPException(409, "Targets can be changed once the dataset is ready")
    uncorrelated: list[dict] = []
    screen: list[int] = []
    with connection() as conn:
        conn.execute(
            """UPDATE meta.datasets SET model_family = coalesce(%s, model_family), name = coalesce(%s, name),
                      updated_at = now() WHERE id = %s""",
            (body.model_family, body.name, dataset_id),
        )
        if body.target_columns is not None:
            uncorrelated, added = set_targets(conn, dataset_id, list(dict.fromkeys(body.target_columns)))
            screen = added if added and has_screen_fields(conn, dataset_id) else []
            conn.execute(
                """UPDATE meta.datasets SET warnings = array(
                       SELECT w FROM unnest(warnings) w WHERE w NOT LIKE 'Target column(s) not found%%')
                    WHERE id = %s""",
                (dataset_id,),
            )
            if uncorrelated:
                conn.execute(
                    """UPDATE meta.datasets SET status = 'correlating', progress = 0, error = NULL, owner = %s,
                              status_message = 'Queued: associations for new target(s)' WHERE id = %s""",
                    (jobs.owner_id(), dataset_id),
                )
    if uncorrelated:
        jobs.submit(correlate_targets, dataset_id, [r["id"] for r in uncorrelated], screen)
    elif screen:
        jobs.submit(screen_targets, dataset_id, screen)
    return get_dataset(dataset_id)


def correlate_targets(dataset_id: int, target_ids: list[int], screen_ids: list[int] | None = None) -> None:
    """Associations between new targets and every other field, for targets profiled as types that never got
    an association sample (text, identifier). Treated as class labels, as profiling treats targets.
    `screen_ids` (new targets) also get the uncapped leak screen."""
    s = get_settings()
    st = _Status(dataset_id)
    try:
        st.set("correlating", 0.0, "Computing associations for new target(s)", force=True)
        ds = fetch_one("SELECT parquet_path, row_count FROM meta.datasets WHERE id = %s", (dataset_id,))
        fields = fetch_all(
            """SELECT id, name, inferred_type, stats ->> 'datetime_format' AS datetime_format
                 FROM meta.fields WHERE dataset_id = %s ORDER BY ordinal""",
            (dataset_id,),
        )
        path = Path(ds["parquet_path"])
        corr_idx = sample_indices(ds["row_count"], s.corr_sample_rows, seed=29)
        targets, numeric, categorical = [], [], []
        for k, f in enumerate(fields):
            is_new = f["id"] in target_ids
            if not is_new and f["inferred_type"] not in CORRELATED_TYPES:
                continue
            col = pl.read_parquet(path, columns=[f["name"]]).to_series()
            t = "categorical" if is_new else f["inferred_type"]
            num, codes = corr_vector(col, t, f["datetime_format"], corr_idx, s.corr_max_levels)
            if is_new:
                targets.append((k, codes))
            elif num is not None:
                numeric.append((k, num))
            elif codes is not None:
                categorical.append((k, codes))
            st.set("correlating", 0.8 * (k + 1) / len(fields), f"Reading {f['name']} ({k + 1}/{len(fields)})")
        assocs = list(target_associations(targets, numeric, categorical))
        screened = screen_pairs(dataset_id, screen_ids) if screen_ids else []
        with connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO meta.field_correlations
                              (dataset_id, field_a, field_b, method, value, strength, n_obs)
                       VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (field_a, field_b) DO NOTHING""",
                    [(dataset_id, *sorted((fields[a.i]["id"], fields[a.j]["id"])), a.method, a.value, abs(a.value),
                      a.n_obs) for a in assocs],
                )
            upsert_correlations(conn, dataset_id, screened)
            _sync_targets(conn, dataset_id)
            conn.execute(
                """UPDATE meta.datasets SET warnings = array(
                       SELECT w FROM unnest(warnings) w WHERE w NOT LIKE 'Associations for the new target(s)%%')
                    WHERE id = %s""",
                (dataset_id,),
            )
        st.set("ready", 1.0, "Ready", force=True)
    except Exception as exc:  # noqa: BLE001
        log.exception("target associations for dataset %s failed", dataset_id)
        # The dataset itself is intact; only the new target's associations are missing.
        execute(
            """UPDATE meta.datasets SET status = 'ready', progress = 1, status_message = 'Ready',
                      warnings = array_append(warnings, %s), updated_at = now() WHERE id = %s""",
            (f"Associations for the new target(s) could not be computed: {exc}", dataset_id),
        )


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int) -> dict:
    """Delete a dataset with its fields, exports and model runs, including their files on disk."""
    s = get_settings()
    with connection() as conn:
        ds = conn.execute("SELECT status FROM meta.datasets WHERE id = %s FOR UPDATE", (dataset_id,)).fetchone()
        if ds is None:
            raise HTTPException(404, "Dataset not found")
        if ds["status"] in ("pending", "ingesting", "profiling", "correlating"):
            raise HTTPException(409, "Dataset is still processing")
        busy = conn.execute(
            """SELECT EXISTS (SELECT 1 FROM meta.exports
                               WHERE dataset_id = %(ds)s AND status IN ('pending', 'running'))
                   OR EXISTS (SELECT 1 FROM meta.model_runs
                               WHERE dataset_id = %(ds)s AND status IN ('pending', 'running')) AS busy""",
            {"ds": dataset_id},
        ).fetchone()["busy"]
        if busy:
            raise HTTPException(409, "An export or model run for this dataset is still running")
        exports = [r["id"] for r in conn.execute(
            "SELECT id FROM meta.exports WHERE dataset_id = %s", (dataset_id,)).fetchall()]
        runs = conn.execute("SELECT id, model_dir FROM meta.model_runs WHERE dataset_id = %s", (dataset_id,)).fetchall()
        n_fields = conn.execute("SELECT count(*) AS n FROM meta.fields WHERE dataset_id = %s",
                                (dataset_id,)).fetchone()["n"]
        conn.execute("DELETE FROM meta.datasets WHERE id = %s", (dataset_id,))
    delete_dataset_files(dataset_id)
    for e in exports:
        shutil.rmtree(s.exports_dir / str(e), ignore_errors=True)
    for path in _run_dirs(s.models_dir, runs):
        shutil.rmtree(path, ignore_errors=True)
    return {"fields": n_fields, "exports": len(exports), "runs": len(runs)}


def _run_dirs(models_dir: Path, runs: list[dict]) -> set[Path]:
    """Every directory a run may have written, including failed runs that never recorded model_dir."""
    root = models_dir.resolve()
    out: set[Path] = set()
    for r in runs:
        out.update(p.resolve() for p in models_dir.glob(f"run{r['id']}_*") if p.is_dir())
        if r["model_dir"]:
            p = Path(r["model_dir"]).resolve()
            if p.parent == root:
                out.add(p)
    return out


@router.get("/{dataset_id}/summary")
def summary(dataset_id: int) -> dict:
    get_dataset(dataset_id)
    with connection() as conn:
        by_type = conn.execute(
            """SELECT inferred_type AS type, count(*) AS n FROM meta.fields WHERE dataset_id = %s
               GROUP BY 1 ORDER BY 2 DESC""",
            (dataset_id,),
        ).fetchall()
        by_action = conn.execute(
            """SELECT action, count(*) AS n, count(*) FILTER (WHERE reviewed) AS reviewed
                 FROM meta.field_decisions WHERE dataset_id = %s GROUP BY 1""",
            (dataset_id,),
        ).fetchall()
        top_target = conn.execute(
            """SELECT field_id, field_name, other_field_name AS target, method, value, strength
                 FROM meta.v_correlations
                WHERE dataset_id = %s AND other_is_target AND field_id NOT IN
                      (SELECT id FROM meta.fields WHERE dataset_id = %s AND is_target)
                ORDER BY strength DESC LIMIT 15""",
            (dataset_id, dataset_id),
        ).fetchall()
    total = sum(r["n"] for r in by_action)
    reviewed = sum(r["reviewed"] for r in by_action)
    return {
        "by_type": by_type,
        "by_action": by_action,
        "total_fields": total,
        "reviewed": reviewed,
        "top_target_associations": top_target,
    }


@router.get("/{dataset_id}/fields")
def list_fields(dataset_id: int) -> list[dict]:
    """Lightweight rows for the review sidebar."""
    return fetch_all(
        """SELECT f.id, f.ordinal, f.name, f.inferred_type, f.inference_confidence, f.is_target, f.null_pct,
                  f.n_unique, f.max_target_assoc, f.leakage_flag AS leakage,
                  d.action, d.semantic_type, d.reviewed, d.source
             FROM meta.fields f JOIN meta.field_decisions d ON d.field_id = f.id
            WHERE f.dataset_id = %s ORDER BY f.ordinal""",
        (dataset_id,),
    )
