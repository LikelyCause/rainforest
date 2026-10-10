"""Background job for one meta.analyses row, plus the cache bookkeeping around it."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from .. import jobs
from ..db import execute, fetch_all, fetch_one, jsonb
from . import compute
from .prep import AnalysisError, NotEnough, prepare_export, prepare_raw

log = logging.getLogger(__name__)

ROW_COLS = """id, dataset_id, kind, params, status, progress, status_message, error, result, created_at, finished_at,
              data_version"""
KEEP_OLD = 3


class Superseded(Exception):
    pass


def params_key(params: dict) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def data_version(conn, dataset_id: int, data: str) -> str:
    """Raw mode: a hash of every field's decision, so any write path (save, bulk, undo, targets) invalidates."""
    if data.startswith("export:"):
        return data
    rows = conn.execute(
        """SELECT field_id, action, semantic_type FROM meta.field_decisions
            WHERE dataset_id = %s ORDER BY field_id""",
        (dataset_id,),
    ).fetchall()
    text = "\n".join(f"{r['field_id']}|{r['action']}|{r['semantic_type']}" for r in rows)
    return "raw:" + hashlib.md5(text.encode()).hexdigest()


def public(row: dict, current_version: str | None) -> dict:
    out = {k: v for k, v in row.items() if k != "data_version"}
    out["stale"] = row["status"] == "done" and current_version is not None and row["data_version"] != current_version
    return out


def find_or_create(conn, dataset_id: int, kind: str, params: dict, recompute: bool) -> tuple[dict, bool]:
    """(row, created). Call inside a transaction holding the dataset row lock."""
    key = params_key(params)
    version = data_version(conn, dataset_id, params["data"])
    if not recompute:
        for status_clause, same_version in (("status = 'done'", True), ("status IN ('pending', 'running')", True),
                                            ("status = 'done'", False)):
            row = conn.execute(
                f"""SELECT {ROW_COLS} FROM meta.analyses
                     WHERE dataset_id = %s AND kind = %s AND params_key = %s
                       AND data_version {'=' if same_version else '<>'} %s AND {status_clause}
                     ORDER BY created_at DESC, id DESC LIMIT 1""",
                (dataset_id, kind, key, version),
            ).fetchone()
            if row is not None:
                return public(row, version), False
    conn.execute(
        """UPDATE meta.analyses SET status = 'failed', error = 'Superseded', status_message = 'Superseded',
                  finished_at = now(), updated_at = now()
            WHERE dataset_id = %s AND kind = %s AND status IN ('pending', 'running')""",
        (dataset_id, kind),
    )
    row = conn.execute(
        f"""INSERT INTO meta.analyses (dataset_id, kind, params, params_key, data_version, status_message, owner)
            VALUES (%s, %s, %s, %s, %s, 'Queued', %s) RETURNING {ROW_COLS}""",
        (dataset_id, kind, jsonb(params), key, version, jobs.owner_id()),
    ).fetchone()
    _prune(conn, dataset_id, kind, version)
    return public(row, version), True


def _prune(conn, dataset_id: int, kind: str, version: str) -> None:
    """Keep the current result per params and the newest few others (stale or failed)."""
    conn.execute(
        """WITH ranked AS (
               SELECT id, status, data_version, created_at,
                      row_number() OVER (PARTITION BY params_key, data_version, status
                                         ORDER BY created_at DESC, id DESC) AS rk
                 FROM meta.analyses
                WHERE dataset_id = %(ds)s AND kind = %(kind)s AND status IN ('done', 'failed')),
           others AS (
               SELECT id, row_number() OVER (ORDER BY created_at DESC, id DESC) AS rn FROM ranked
                WHERE NOT (status = 'done' AND rk = 1 AND (data_version = %(ver)s OR data_version LIKE 'export:%%')))
           DELETE FROM meta.analyses WHERE id IN (SELECT id FROM others WHERE rn > %(keep)s)""",
        {"ds": dataset_id, "kind": kind, "ver": version, "keep": KEEP_OLD},
    )


def _progress(analysis_id: int):
    def check(frac: float, msg: str) -> None:
        row = fetch_one(
            """UPDATE meta.analyses SET status = 'running', progress = %s, status_message = %s, updated_at = now()
                WHERE id = %s AND status IN ('pending', 'running') RETURNING id""",
            (round(frac, 4), msg, analysis_id),
        )
        if row is None:
            raise Superseded

    return check


def _prepare(row: dict, ds: dict):
    p = row["params"]
    if p["data"].startswith("export:"):
        exp = fetch_one("SELECT output_dir, status FROM meta.exports WHERE id = %s AND dataset_id = %s",
                        (int(p["data"].split(":", 1)[1]), row["dataset_id"]))
        if exp is None or exp["status"] != "done" or not exp["output_dir"]:
            raise AnalysisError("The export is missing or not finished.")
        out_dir = Path(exp["output_dir"])
        try:
            manifest = json.loads((out_dir / "manifest.json").read_text())
        except (OSError, ValueError) as exc:
            raise AnalysisError("The export's files were removed.") from exc
        return prepare_export(out_dir, manifest, p["target"], p["sample"])
    fields = fetch_all(
        """SELECT f.id, f.name, f.stats ->> 'datetime_format' AS datetime_format, d.action, d.semantic_type
             FROM meta.fields f JOIN meta.field_decisions d ON d.field_id = f.id
            WHERE f.dataset_id = %s ORDER BY f.ordinal""",
        (row["dataset_id"],),
    )
    return prepare_raw(Path(ds["parquet_path"]), fields, p["kept"], p["target"], p["sample"])


def run_analysis(analysis_id: int) -> None:
    row = fetch_one("SELECT * FROM meta.analyses WHERE id = %s", (analysis_id,))
    if row is None or row["status"] != "pending":
        return
    check = _progress(analysis_id)
    try:
        check(0.0, "Reading a sample of rows")
        ds = fetch_one("SELECT parquet_path FROM meta.datasets WHERE id = %s", (row["dataset_id"],))
        prepared = _prepare(row, ds)
        check(0.1, f"Sampled {prepared.n:,} rows × {len(prepared.columns)} fields")
        p = row["params"]
        if row["kind"] == "map":
            result = compute.run_map(prepared, p["method"], p.get("perplexity", 30), p.get("n_neighbors", 15), check)
        elif row["kind"] == "tests":
            result = compute.run_tests(prepared, check)
        else:
            result = compute.run_importance(prepared, check)
        execute(
            """UPDATE meta.analyses SET status = 'done', progress = 1, status_message = 'Done', result = %s,
                      finished_at = now(), updated_at = now()
                WHERE id = %s AND status = 'running'""",
            (jsonb(result), analysis_id),
        )
    except Superseded:
        return
    except NotEnough as exc:
        _fail(analysis_id, str(exc), {"reason": exc.reason})
    except AnalysisError as exc:
        _fail(analysis_id, str(exc), None)
    except Exception as exc:  # noqa: BLE001
        log.exception("analysis %s failed", analysis_id)
        _fail(analysis_id, f"{type(exc).__name__}: {exc}", None)


def _fail(analysis_id: int, error: str, result: dict | None) -> None:
    execute(
        """UPDATE meta.analyses SET status = 'failed', status_message = 'Failed', error = %s, result = %s,
                  finished_at = now(), updated_at = now()
            WHERE id = %s AND status IN ('pending', 'running')""",
        (error, None if result is None else jsonb(result), analysis_id),
    )
