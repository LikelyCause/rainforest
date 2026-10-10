"""File ingest: CSV / Parquet -> normalised Parquet (analytics copy) -> Postgres raw table."""
from __future__ import annotations

import csv
import io
import logging
import os
import re
from pathlib import Path
from typing import Callable

import polars as pl
import pyarrow.parquet as pq
from psycopg import sql

from .db import connection

log = logging.getLogger(__name__)

# A whole cell holding one of these is missing. "?" is how many public datasets (UCI's among them) mark blanks.
NULL_TOKENS = ["", "NA", "N/A", "n/a", "NaN", "nan", "null", "NULL", "None", "#N/A", "?"]
SEPARATORS = {",": "commas", ";": "semicolons", "\t": "tabs", "|": "pipes"}
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
    """Convert an upload into the canonical Parquet copy. Streams; memory stays bounded.

    Writes to a temporary file and renames it into place only once it reads back, so a failed or retried
    sink never leaves a partial file at `dest`.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        warnings = _sink(src, fmt, tmp)
        pl.read_parquet_schema(tmp)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return warnings


def _sink(src: Path, fmt: str, out: Path) -> list[str]:
    if fmt == "parquet":
        lf, warnings = _normalize(pl.scan_parquet(src))
        lf.sink_parquet(out, row_group_size=100_000)
        return warnings

    sep = guess_separator(src)
    full, ragged = False, False
    while True:
        try:
            lf, warnings = _normalize(pl.scan_csv(
                src,
                separator=sep,
                infer_schema_length=None if full else 50_000,
                null_values=NULL_TOKENS,
                try_parse_dates=False,
                encoding="utf8-lossy",
                truncate_ragged_lines=ragged,
            ))
            lf.sink_parquet(out, row_group_size=100_000)
            break
        except pl.exceptions.ComputeError as exc:
            out.unlink(missing_ok=True)
            if not ragged and "more fields" in str(exc):
                ragged = True
            elif not full:
                # A value late in the file broke the inferred schema; re-infer over the whole file.
                log.warning("CSV schema inference on head failed (%s); rescanning full file", exc)
                full = True
            else:
                raise
    if sep != ",":
        warnings.append(f"Read as a file whose columns are separated by {SEPARATORS[sep]}.")
    if ragged:
        warnings.append("Some rows have more values than the header; the extra values were dropped.")
    if full:
        warnings.append("CSV schema had to be inferred from the full file (mixed types late in the file).")
    return warnings


def guess_separator(src: Path, sample_bytes: int = 256_000) -> str:
    """The column separator of a delimited text file: the candidate that splits the header into the most columns and
    every sampled line into the same number (quotes respected), else a comma."""
    with open(src, "rb") as fh:
        head = fh.read(sample_bytes).decode("utf-8", errors="replace")
    lines = head.splitlines()[:50]
    if len(head) == sample_bytes and len(lines) > 1:
        lines = lines[:-1]  # the last line may be cut off
    best, best_cols = ",", 1
    for sep in SEPARATORS:
        try:
            counts = {len(row) for row in csv.reader(io.StringIO("\n".join(lines)), delimiter=sep) if row}
        except csv.Error:
            continue
        if len(counts) == 1 and (cols := counts.pop()) > best_cols:
            best, best_cols = sep, cols
    return best


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
) -> tuple[str, list[str]]:
    """COPY the Parquet data into raw.ds_<id>. Returns the qualified table name and any warnings.

    Strings are quoted, so only an unquoted \\N (a real null) reads as NULL; text can't hold NUL bytes in
    Postgres, so they are removed from the raw table (the Parquet copy keeps them).
    """
    table = f"ds_{dataset_id}"
    schema = pl.read_parquet_schema(parquet_path)
    src_names = list(schema.keys())
    text_cols = [c for c, dt in schema.items() if dt == pl.String]
    nul_counts: dict[str, int] = {}
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
                for c in text_cols:
                    if n := int(df[c].str.contains("\x00", literal=True).sum()):
                        nul_counts[c] = nul_counts.get(c, 0) + n
                        df = df.with_columns(pl.col(c).str.replace_all("\x00", "", literal=True))
                df = df.with_row_index(ROW_ID, offset=offset)
                copy.write(df.write_csv(include_header=False, null_value="\\N", quote_style="non_numeric"))
                offset += df.height
                if progress:
                    progress(offset, total)
        conn.execute(
            sql.SQL("ALTER TABLE {} ADD PRIMARY KEY ({})").format(ident, sql.Identifier(ROW_ID))
        )
        conn.execute(sql.SQL("ANALYZE {}").format(ident))
    warnings = [f"Removed NUL characters from {n:,} value(s) of {c!r} in raw.{table} (Postgres text can't hold "
                f"them; profiling and exports read the file as uploaded)." for c, n in nul_counts.items()]
    return f"raw.{table}", warnings
