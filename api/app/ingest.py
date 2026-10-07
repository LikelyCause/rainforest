"""File ingest: CSV / Parquet -> normalised Parquet (analytics copy) -> Postgres raw table."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable

import polars as pl
import pyarrow.parquet as pq
from psycopg import sql

from .db import connection

log = logging.getLogger(__name__)

NULL_TOKENS = ["", "NA", "N/A", "n/a", "NaN", "nan", "null", "NULL", "None", "#N/A"]
ROW_ID = "__row_id"


def sanitize_identifiers(names: list[str]) -> list[str]:
    """Map arbitrary column names to unique, lower-case Postgres identifiers (<= 59 chars)."""
    out: list[str] = []
    seen: set[str] = {ROW_ID}
    for i, raw in enumerate(names):
        base = re.sub(r"[^0-9a-zA-Z]+", "_", raw.strip()).strip("_").lower() or f"col_{i}"
        if base[0].isdigit():
            base = f"c_{base}"
        base = base[:55]
        cand, k = base, 2
        while cand in seen:
            cand = f"{base}_{k}"
            k += 1
        seen.add(cand)
        out.append(cand)
    return out


def _normalize(lf: pl.LazyFrame) -> tuple[pl.LazyFrame, list[str]]:
    """Make column names usable and dtypes flat / Postgres friendly."""
    warnings: list[str] = []
    schema = lf.collect_schema()
    renames: dict[str, str] = {}
    used: set[str] = set()
    for i, name in enumerate(schema.names()):
        new = name.strip().lstrip("﻿") or f"unnamed_{i}"
        while new in used:
            new = f"{new}_dup"
        used.add(new)
        if new != name:
            renames[name] = new
    exprs: list[pl.Expr] = []
    for name, dt in schema.items():
        col = pl.col(name)
        base = dt.base_type()
        if base in (pl.Categorical, pl.Enum, pl.Null, pl.Time, pl.Object):
            col = col.cast(pl.String)
        elif base == pl.Decimal:
            col = col.cast(pl.Float64)
        elif base == pl.Duration:
            col = col.dt.total_seconds().cast(pl.Float64)
        elif base in (pl.List, pl.Array, pl.Struct, pl.Binary):
            warnings.append(f"Dropped column {name!r}: nested/binary type {dt} is not supported")
            continue
        exprs.append(col.alias(renames.get(name, name)))
    return lf.select(exprs), warnings


def to_parquet(src: Path, fmt: str, dest: Path) -> list[str]:
    """Convert an upload into the canonical Parquet copy. Streams; memory stays bounded."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        lf = pl.scan_parquet(src)
        lf, warnings = _normalize(lf)
        lf.sink_parquet(dest, row_group_size=100_000)
        return warnings

    def scan(infer_rows: int | None) -> pl.LazyFrame:
        return pl.scan_csv(
            src,
            infer_schema_length=infer_rows,
            null_values=NULL_TOKENS,
            try_parse_dates=False,
            encoding="utf8-lossy",
            truncate_ragged_lines=True,
        )

    try:
        lf, warnings = _normalize(scan(50_000))
        lf.sink_parquet(dest, row_group_size=100_000)
    except pl.exceptions.ComputeError as exc:
        # A value late in the file broke the inferred schema; re-infer over the whole file.
        log.warning("CSV schema inference on head failed (%s); rescanning full file", exc)
        lf, warnings = _normalize(scan(None))
        lf.sink_parquet(dest, row_group_size=100_000)
        warnings.append("CSV schema had to be inferred from the full file (mixed types late in the file).")
    return warnings


def pg_type(dt: pl.DataType) -> str:
    if dt == pl.Boolean:
        return "BOOLEAN"
    if dt == pl.UInt64:
        return "NUMERIC(20,0)"
    if dt.is_integer():
        return "BIGINT"
    if dt.is_float():
        return "DOUBLE PRECISION"
    if dt == pl.Date:
        return "DATE"
    if isinstance(dt, pl.Datetime):
        return "TIMESTAMPTZ" if dt.time_zone else "TIMESTAMP"
    return "TEXT"


def load_raw_table(
    dataset_id: int,
    parquet_path: Path,
    db_columns: list[str],
    batch_rows: int,
    progress: Callable[[int, int], None] | None = None,
) -> str:
    """COPY the Parquet data into raw.ds_<id>. Returns the qualified table name."""
    table = f"ds_{dataset_id}"
    schema = pl.read_parquet_schema(parquet_path)
    src_names = list(schema.keys())
    col_defs = [sql.SQL("{} BIGINT").format(sql.Identifier(ROW_ID))] + [
        sql.SQL("{} {}").format(sql.Identifier(c), sql.SQL(pg_type(dt)))
        for c, dt in zip(db_columns, schema.values())
    ]
    ident = sql.Identifier("raw", table)
    pf = pq.ParquetFile(parquet_path)
    total = pf.metadata.num_rows
    with connection() as conn:
        conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(ident))
        conn.execute(sql.SQL("CREATE TABLE {} ({})").format(ident, sql.SQL(", ").join(col_defs)))
        copy_sql = sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT csv, NULL '\\N')").format(
            ident, sql.SQL(", ").join(sql.Identifier(c) for c in [ROW_ID, *db_columns])
        )
        offset = 0
        with conn.cursor() as cur, cur.copy(copy_sql) as copy:
            for batch in pf.iter_batches(batch_size=batch_rows, columns=src_names):
                df = pl.from_arrow(batch)
                assert isinstance(df, pl.DataFrame)
                df = df.with_row_index(ROW_ID, offset=offset)
                copy.write(df.write_csv(include_header=False, null_value="\\N"))
                offset += df.height
                if progress:
                    progress(offset, total)
        conn.execute(
            sql.SQL("ALTER TABLE {} ADD PRIMARY KEY ({})").format(ident, sql.Identifier(ROW_ID))
        )
        conn.execute(sql.SQL("ANALYZE {}").format(ident))
    return f"raw.{table}"
