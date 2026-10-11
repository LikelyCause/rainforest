"""Derived fields (see derived.py): preview, add, list and remove. They change the dataset's columns, so they're added
or removed while its current baseline is being set up or is finalized, never mid-assessment; a new baseline version
then assesses them like any other field."""
from __future__ import annotations

import os
from typing import Any, Literal

import polars as pl
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import derived, recipes
from ..auth import CurrentUser
from ..config import get_settings
from ..db import connection
from ..ingest import sanitize_identifiers
from ..processing import insert_field
from ..profiling import profile_column, sample_indices
from .datasets import queue_target_work

router = APIRouter(prefix="/api", tags=["derived"])


class DerivedIn(BaseModel):
    name: str = Field("", max_length=100)
    kind: Literal["formula", "groups", "count"]
    spec: dict[str, Any]


def _dataset(conn, dataset_id: int) -> dict:
    ds = conn.execute("SELECT id, status, parquet_path FROM meta.datasets WHERE id = %s", (dataset_id,)).fetchone()
    if ds is None:
        raise HTTPException(404, "Dataset not found")
    if ds["status"] != "ready" or not ds["parquet_path"]:
        raise HTTPException(409, "Add fields once the dataset is ready")
    return ds


def _phase_ok(conn, dataset_id: int) -> None:
    cur = recipes.current(conn, dataset_id, lock="update")
    if cur["phase"] != "finalized" and not recipes.is_setup(conn, cur):
        raise HTTPException(409, f"{recipes.label(cur)} is being worked on; add or remove fields once it is finalized "
                                 "(then start a new version to assess them) or before its first baseline is set up.")


@router.post("/datasets/{dataset_id}/derived/preview")
def preview(dataset_id: int, body: DerivedIn, _: CurrentUser) -> dict:
    with connection() as conn:
        ds = _dataset(conn, dataset_id)
    columns = list(pl.read_parquet_schema(ds["parquet_path"]))
    reads = derived.check_spec(body.kind, body.spec, columns)
    df = pl.read_parquet(ds["parquet_path"], columns=reads)
    s = derived.compute(df, body.kind, body.spec)
    return {"type": "number" if s.dtype.is_numeric() else "text", "missing": int(s.null_count()), "rows": s.len(),
            "sample": s.head(8).to_list(), "distinct": int(s.n_unique())}


@router.get("/datasets/{dataset_id}/derived")
def list_derived(dataset_id: int, _: CurrentUser) -> list[dict]:
    with connection() as conn:
        return conn.execute("""SELECT id, name, derived ->> 'kind' AS kind, derived -> 'spec' AS spec FROM meta.fields
                                WHERE dataset_id = %s AND derived IS NOT NULL ORDER BY ordinal""",
                            (dataset_id,)).fetchall()


@router.post("/datasets/{dataset_id}/derived", status_code=201)
def add_derived(dataset_id: int, body: DerivedIn, _: CurrentUser) -> dict:
    s = get_settings()
    with connection() as conn:
        recipes.lock_dataset(conn, dataset_id)
        ds = _dataset(conn, dataset_id)
        _phase_ok(conn, dataset_id)
        existing = [r["name"] for r in conn.execute("SELECT name FROM meta.fields WHERE dataset_id = %s",
                                                    (dataset_id,)).fetchall()]
        name = derived.check_name(body.name, existing)
        df = pl.read_parquet(ds["parquet_path"])
        derived.check_spec(body.kind, body.spec, df.columns)
        col = derived.compute(df, body.kind, body.spec).alias(name)
        if col.null_count() == col.len():
            raise HTTPException(422, "Every row comes out missing; check the fields and values")
        tmp = f"{ds['parquet_path']}.tmp"
        df.with_columns(col).write_parquet(tmp)
        n = df.height
        del df
        ordinal = conn.execute("SELECT coalesce(max(ordinal), -1) + 1 AS o FROM meta.fields WHERE dataset_id = %s",
                               (dataset_id,)).fetchone()["o"]
        p = profile_column(col, ordinal, False, sample_indices(n, s.profile_sample_rows, 17),
                           sample_indices(n, s.corr_sample_rows, 29), s.corr_max_levels)
        db_col = sanitize_identifiers([*existing, name])[-1]
        fid = insert_field(conn, dataset_id, p, db_col, {"kind": body.kind, "spec": body.spec})
        os.replace(tmp, ds["parquet_path"])  # last: a failure above leaves the data as it was
        conn.execute("UPDATE meta.datasets SET column_count = column_count + 1, updated_at = now() WHERE id = %s",
                     (dataset_id,))
        targets = [r["id"] for r in conn.execute("SELECT id FROM meta.fields WHERE dataset_id = %s AND is_target",
                                                 (dataset_id,)).fetchall()]
        work = queue_target_work(conn, dataset_id, [], targets)
        row = conn.execute("SELECT id, name, inferred_type, n_unique, null_pct FROM meta.fields WHERE id = %s",
                           (fid,)).fetchone()
    work()
    return row


@router.delete("/datasets/{dataset_id}/derived/{field_id}")
def remove_derived(dataset_id: int, field_id: int, _: CurrentUser) -> dict:
    with connection() as conn:
        recipes.lock_dataset(conn, dataset_id)
        ds = _dataset(conn, dataset_id)
        _phase_ok(conn, dataset_id)
        f = conn.execute("SELECT name, derived FROM meta.fields WHERE id = %s AND dataset_id = %s",
                         (field_id, dataset_id)).fetchone()
        if f is None or f["derived"] is None:
            raise HTTPException(404, "No derived field with that id in this dataset")
        conn.execute("DELETE FROM meta.fields WHERE id = %s", (field_id,))
        df = pl.read_parquet(ds["parquet_path"])
        if f["name"] in df.columns:
            df.drop(f["name"]).write_parquet(f"{ds['parquet_path']}.tmp")
            os.replace(f"{ds['parquet_path']}.tmp", ds["parquet_path"])
        conn.execute("UPDATE meta.datasets SET column_count = column_count - 1, updated_at = now() WHERE id = %s",
                     (dataset_id,))
    return {"removed": f["name"]}
