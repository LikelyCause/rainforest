from __future__ import annotations

import datetime as dt
import decimal
import re
import time
import uuid

import psycopg
from fastapi import APIRouter, HTTPException
from psycopg import sql

from ..config import get_settings
from ..db import fetch_all, readonly_conninfo
from ..schemas import FieldFilter, FieldQuery, SqlRequest

router = APIRouter(prefix="/api", tags=["query"])

# Columns of meta.v_fields that the structured filter may touch, with their kind.
QUERYABLE: dict[str, str] = {
    "name": "text", "ordinal": "num", "physical_dtype": "text", "inferred_type": "text",
    "inference_confidence": "num", "is_target": "bool", "row_count": "num", "null_count": "num",
    "null_pct": "num", "n_unique": "num", "unique_pct": "num", "mean": "num", "std": "num",
    "min_value": "num", "max_value": "num", "median": "num", "skewness": "num", "kurtosis": "num",
    "zeros_pct": "num", "outlier_pct": "num", "top_value": "text", "top_value_pct": "num",
    "avg_length": "num", "max_target_assoc": "num", "action": "text", "semantic_type": "text",
    "imputation": "text", "transform": "text", "scaling": "text", "encoding": "text",
    "add_missing_indicator": "bool", "reviewed": "bool", "leakage_flag": "bool", "model_family": "text",
    "settings_source": "text", "decision_source": "text", "notes": "text", "suggested_action": "text",
}

SQL_EXAMPLES = [
    {"title": "All text fields",
     "sql": "SELECT name, null_pct, avg_length FROM meta.v_fields\nWHERE dataset_id = :dataset AND inferred_type = 'text'"},
    {"title": "Fields with > 20% missing",
     "sql": "SELECT name, inferred_type, round(null_pct::numeric, 3) AS null_pct\nFROM meta.v_fields\n"
            "WHERE dataset_id = :dataset AND null_pct > 0.2\nORDER BY null_pct DESC"},
    {"title": "Most target-associated fields",
     "sql": "SELECT field_name, other_field_name AS target, method, round(value::numeric, 3) AS value\n"
            "FROM meta.v_correlations\nWHERE dataset_id = :dataset AND other_is_target\nORDER BY strength DESC LIMIT 25"},
    {"title": "Highly skewed numerics",
     "sql": "SELECT name, skewness, transform, scaling FROM meta.v_fields\n"
            "WHERE dataset_id = :dataset AND abs(skewness) > 2 ORDER BY abs(skewness) DESC"},
    {"title": "Redundant pairs (|assoc| > 0.8)",
     "sql": "SELECT field_name, other_field_name, method, round(value::numeric, 3) AS value\n"
            "FROM meta.v_correlations\nWHERE dataset_id = :dataset AND strength > 0.8 AND field_id < other_field_id\n"
            "ORDER BY strength DESC"},
    {"title": "Decisions I changed from the suggestion",
     "sql": "SELECT name, action, semantic_type, imputation, scaling, encoding, decided_at\n"
            "FROM meta.v_fields\nWHERE dataset_id = :dataset AND decision_source = 'user'\nORDER BY decided_at DESC"},
    {"title": "Compare model runs (test split)",
     "sql": "SELECT model, run_id, target, metric, round(value::numeric, 4) AS value\nFROM meta.v_model_metrics\n"
            "WHERE dataset_id = :dataset AND split = 'test' AND metric IN ('roc_auc', 'roc_auc_ovr_macro', 'accuracy')\n"
            "ORDER BY target, metric, value DESC"},
    {"title": "Settings across model families",
     "sql": "SELECT name, family, imputation, transform, scaling, encoding, source\nFROM meta.v_field_settings\n"
            "WHERE dataset_id = :dataset AND action = 'keep'\nORDER BY name, family LIMIT 300"},
    {"title": "Raw data peek",
     "sql": "SELECT * FROM raw.ds_:dataset LIMIT 100"},
]


@router.post("/datasets/{dataset_id}/fields/query")
def query_fields(dataset_id: int, q: FieldQuery) -> list[dict]:
    if q.order_by not in QUERYABLE:
        raise HTTPException(422, f"Cannot order by {q.order_by!r}")
    query = sql.SQL("SELECT * FROM meta.v_fields WHERE {} ORDER BY {} {} NULLS LAST LIMIT {}").format(
        fields_where(dataset_id, q.filters),
        sql.Identifier(q.order_by),
        sql.SQL("DESC" if q.descending else "ASC"),
        sql.Literal(q.limit),
    )
    return fetch_all(query)


def fields_where(dataset_id: int, filters: list[FieldFilter]) -> sql.Composable:
    """WHERE clause over meta.v_fields for a structured filter; shared by the query and bulk endpoints."""
    clauses = [sql.SQL("dataset_id = {}").format(sql.Literal(dataset_id))]
    for f in filters:
        kind = QUERYABLE.get(f.column)
        if kind is None:
            raise HTTPException(422, f"Unknown column {f.column!r}")
        col = sql.Identifier(f.column)
        if f.op == "is_null":
            clauses.append(sql.SQL("{} IS NULL").format(col))
        elif f.op == "not_null":
            clauses.append(sql.SQL("{} IS NOT NULL").format(col))
        elif f.op == "contains":
            clauses.append(sql.SQL("{}::text ILIKE {}").format(col, sql.Literal(f"%{f.value}%")))
        elif f.op == "in":
            values = f.value if isinstance(f.value, list) else [v.strip() for v in str(f.value).split(",")]
            typed = [_coerce(v, kind, f.column) for v in values]
            if kind == "text":
                clauses.append(sql.SQL("{} = ANY({}::text[])").format(col, sql.Literal(typed)))
            else:
                clauses.append(sql.SQL("{} = ANY({})").format(col, sql.Literal(typed)))
        else:
            value = _coerce(f.value, kind, f.column)
            clauses.append(sql.SQL("{} {} {}").format(col, sql.SQL(f.op), sql.Literal(value)))
    return sql.SQL(" AND ").join(clauses)


_TRUE = ("1", "true", "yes", "t", "y")
_FALSE = ("0", "false", "no", "f", "n")


def _coerce(value, kind: str, column: str):
    try:
        if kind == "num":
            return float(value)
        if kind == "bool":
            v = str(value).strip().lower()
            if v in _TRUE or v in _FALSE:
                return v in _TRUE
            raise ValueError(value)
        return str(value)
    except (TypeError, ValueError):
        raise HTTPException(422, f"{column} expects a {kind} value") from None


@router.get("/sql/examples")
def sql_examples() -> list[dict]:
    return SQL_EXAMPLES


@router.post("/sql")
def run_sql(req: SqlRequest) -> dict:
    """Run one statement as the SELECT-only role, inside a READ ONLY transaction."""
    s = get_settings()
    statement = req.sql.strip().rstrip(";").strip()
    if ";" in _strip_literals(statement):
        raise HTTPException(400, {"message": "Run one statement at a time (no ';' separators)."})
    started = time.perf_counter()
    try:
        with psycopg.connect(readonly_conninfo(), connect_timeout=5, autocommit=False) as conn:
            conn.execute("SET TRANSACTION READ ONLY")
            conn.execute(sql.SQL("SET LOCAL statement_timeout = {}").format(sql.Literal(f"{s.sql_timeout_ms}ms")))
            cur = conn.execute(statement, prepare=False)  # type: ignore[arg-type]
            if cur.description is None:
                return {"columns": [], "rows": [], "row_count": 0, "truncated": False,
                        "elapsed_ms": _ms(started)}
            columns = [d.name for d in cur.description]
            rows = cur.fetchmany(s.sql_max_rows + 1)
            conn.rollback()
    except psycopg.Error as exc:
        msg = exc.diag.message_primary if getattr(exc, "diag", None) and exc.diag.message_primary else str(exc)
        raise HTTPException(400, {"message": msg, "detail": getattr(exc.diag, "message_detail", None)}) from None
    truncated = len(rows) > s.sql_max_rows
    rows = rows[: s.sql_max_rows]
    return {
        "columns": columns,
        "rows": [[_cell(v) for v in r] for r in rows],
        "row_count": len(rows),
        "truncated": truncated,
        "elapsed_ms": _ms(started),
    }


def _strip_literals(q: str) -> str:
    """Remove quoted strings / identifiers and comments so ';' inside them is ignored."""
    q = re.sub(r"--[^\n]*", " ", q)
    q = re.sub(r"/\*.*?\*/", " ", q, flags=re.S)
    return re.sub(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"", "''", q)


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _cell(v):
    if isinstance(v, float) and v != v:  # NaN
        return None
    if isinstance(v, (dt.datetime, dt.date, dt.time)):
        return v.isoformat()
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (uuid.UUID, dt.timedelta, memoryview, bytes)):
        return str(v)
    return v
