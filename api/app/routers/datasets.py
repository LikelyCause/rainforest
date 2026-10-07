from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile

from .. import jobs
from ..config import get_settings
from ..db import connection, execute, fetch_all, fetch_one
from ..processing import delete_dataset_files, process_dataset
from ..sample_data import generate
from ..schemas import DatasetPatch, SampleRequest
from ..semantics import FAMILIES

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

DATASET_COLS = """id, name, source_filename, file_format, target_columns, model_family, row_count, column_count, status,
                  progress, status_message, error, warnings, created_at, updated_at, profiled_at, raw_table"""


def _targets(raw: str | None) -> list[str]:
    if raw is None or not raw.strip():
        return get_settings().target_list
    return [t.strip() for t in raw.split(",") if t.strip()]


def _create(name: str, filename: str | None, fmt: str, targets: list[str]) -> dict:
    return fetch_one(
        f"""INSERT INTO meta.datasets (name, source_filename, file_format, target_columns, status, status_message)
            VALUES (%s, %s, %s, %s, 'pending', 'Queued') RETURNING {DATASET_COLS}""",
        (name, filename, fmt, targets),
    )


@router.get("")
def list_datasets() -> list[dict]:
    return fetch_all(f"SELECT {DATASET_COLS} FROM meta.datasets ORDER BY created_at DESC")


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
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out, length=8 * 1024 * 1024)
    if dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "Uploaded file is empty")
    ds = _create(name or Path(filename).stem, filename, fmt, _targets(targets))
    jobs.submit(process_dataset, ds["id"], str(dest), fmt)
    return ds


@router.post("/sample", status_code=202)
def create_sample(req: SampleRequest) -> dict:
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
            "UPDATE meta.datasets SET status='ingesting', status_message='Generating sample data', "
            "updated_at=now() WHERE id=%s",
            (dataset_id,),
        )
        generate(req["rows"], req["fields"], req["seed"], req["target_kind"]).write_parquet(out)
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
    """Rename a dataset or switch the model family its review settings are shown for."""
    get_dataset(dataset_id)
    if body.model_family is not None and body.model_family not in FAMILIES:
        raise HTTPException(422, f"Unknown model family {body.model_family!r}")
    return fetch_one(
        f"""UPDATE meta.datasets SET model_family = coalesce(%s, model_family), name = coalesce(%s, name),
                   updated_at = now() WHERE id = %s RETURNING {DATASET_COLS}""",
        (body.model_family, body.name, dataset_id),
    )


@router.delete("/{dataset_id}", status_code=204, response_class=Response)
def delete_dataset(dataset_id: int) -> Response:
    ds = get_dataset(dataset_id)
    if ds["status"] in ("pending", "ingesting", "profiling", "correlating"):
        raise HTTPException(409, "Dataset is still processing")
    delete_dataset_files(dataset_id)
    for e in fetch_all("SELECT id FROM meta.exports WHERE dataset_id = %s", (dataset_id,)):
        shutil.rmtree(get_settings().exports_dir / str(e["id"]), ignore_errors=True)
    execute("DELETE FROM meta.datasets WHERE id = %s", (dataset_id,))
    return Response(status_code=204)


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
