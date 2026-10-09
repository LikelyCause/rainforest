"""Dataset processing job: ingest -> profile -> correlate -> seed decisions."""
from __future__ import annotations

import json
import logging
import shutil
import time
import traceback
from pathlib import Path

import polars as pl
from psycopg import sql

from . import ingest, profiling
from .config import get_settings
from .correlations import compute_associations
from .db import connection, execute, fetch_one, jsonb
from .semantics import suggest_all

log = logging.getLogger(__name__)

LEAKAGE_THRESHOLD = 0.95

SUMMARY_COLS = [
    "row_count", "null_count", "null_pct", "n_unique", "unique_pct", "mean", "std", "min_value",
    "max_value", "median", "skewness", "kurtosis", "zeros_pct", "outlier_pct", "top_value",
    "top_value_pct", "avg_length",
]


class _Status:
    """Throttled status writer so tight loops don't hammer the database."""

    def __init__(self, dataset_id: int):
        self.dataset_id = dataset_id
        self._last = 0.0

    def set(self, status: str, progress: float, message: str, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last < 1.0:
            return
        self._last = now
        execute(
            """UPDATE meta.datasets SET status = %s, progress = %s, status_message = %s, updated_at = now()
               WHERE id = %s""",
            (status, round(progress, 4), message, self.dataset_id),
        )


def process_dataset(dataset_id: int, upload_path: str | None, fmt: str) -> None:
    s = get_settings()
    st = _Status(dataset_id)
    ds = fetch_one("SELECT * FROM meta.datasets WHERE id = %s", (dataset_id,))
    if ds is None:
        return
    targets: list[str] = list(ds["target_columns"] or [])
    parquet_path = s.datasets_dir / str(dataset_id) / "raw.parquet"
    try:
        # ---- 1. ingest ------------------------------------------------------------
        st.set("ingesting", 0.01, "Converting to Parquet", force=True)
        warnings: list[str] = []
        if upload_path is not None:
            warnings = ingest.to_parquet(Path(upload_path), fmt, parquet_path)
        schema = pl.read_parquet_schema(parquet_path)
        if upload_path is not None:
            Path(upload_path).unlink(missing_ok=True)
        names = list(schema.keys())
        db_cols = ingest.sanitize_identifiers(names)
        missing_targets = [t for t in targets if t not in schema]
        if missing_targets:
            warnings.append(f"Target column(s) not found in file: {', '.join(missing_targets)}")

        st.set("ingesting", 0.05, "Loading raw table into Postgres", force=True)
        raw_table = ingest.load_raw_table(
            dataset_id,
            parquet_path,
            db_cols,
            s.copy_batch_rows,
            progress=lambda done, total: st.set(
                "ingesting", 0.05 + 0.30 * done / max(total, 1), f"Loading raw table: {done:,} / {total:,} rows"
            ),
        )
        execute(
            """UPDATE meta.datasets SET parquet_path = %s, raw_table = %s, column_count = %s,
                      warnings = %s, updated_at = now() WHERE id = %s""",
            (str(parquet_path), raw_table, len(names), warnings, dataset_id),
        )

        # ---- 2. profile -----------------------------------------------------------
        n_rows, profiles = profiling.profile_parquet(
            parquet_path,
            targets,
            s.profile_sample_rows,
            s.corr_sample_rows,
            s.corr_max_levels,
            progress=lambda i, n, name: st.set(
                "profiling", 0.35 + 0.40 * i / max(n, 1), f"Profiling {name} ({i + 1}/{n})"
            ),
        )
        field_ids = _persist_profiles(dataset_id, profiles, db_cols)
        execute("UPDATE meta.datasets SET row_count = %s WHERE id = %s", (n_rows, dataset_id))
        rejected = [p.name for p in profiles if p.name in targets and not p.is_target]
        if rejected:
            warnings.append(f"Not used as target(s), too many distinct values: {', '.join(rejected)}")
            execute("UPDATE meta.datasets SET target_columns = %s, warnings = %s WHERE id = %s",
                    ([t for t in targets if t not in rejected], warnings, dataset_id))

        # ---- 3. correlations ------------------------------------------------------
        st.set("correlating", 0.75, "Computing pairwise associations", force=True)
        numeric = [(i, p.corr_numeric) for i, p in enumerate(profiles) if p.corr_numeric is not None]
        categorical = [(i, p.corr_codes) for i, p in enumerate(profiles) if p.corr_codes is not None]
        assocs = compute_associations(
            numeric,
            categorical,
            progress=lambda f: st.set("correlating", 0.75 + 0.2 * f, f"Computing associations ({f:.0%})"),
        )
        _persist_correlations(dataset_id, field_ids, assocs)
        with connection() as conn:
            refresh_target_state(conn, dataset_id)

        st.set("ready", 1.0, "Ready", force=True)
        execute("UPDATE meta.datasets SET profiled_at = now() WHERE id = %s", (dataset_id,))
        log.info("dataset %s processed: %s rows x %s fields", dataset_id, n_rows, len(profiles))
    except Exception as exc:  # noqa: BLE001
        log.exception("processing dataset %s failed", dataset_id)
        execute(
            """UPDATE meta.datasets SET status = 'failed', error = %s, status_message = 'Failed',
                      updated_at = now() WHERE id = %s""",
            (f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc(limit=4)}", dataset_id),
        )
    finally:
        if upload_path is not None:
            Path(upload_path).unlink(missing_ok=True)


def _persist_profiles(dataset_id: int, profiles: list[profiling.FieldProfile], db_cols: list[str]) -> list[int]:
    cols = [
        "dataset_id", "ordinal", "name", "db_column", "physical_dtype", "inferred_type",
        "inference_confidence", "inference_reasons", "is_target", *SUMMARY_COLS, "stats", "charts", "suggestion",
    ]
    insert = sql.SQL("INSERT INTO meta.fields ({}) VALUES ({}) RETURNING id").format(
        sql.SQL(", ").join(map(sql.Identifier, cols)), sql.SQL(", ").join(sql.Placeholder() * len(cols))
    )
    ids: list[int] = []
    with connection() as conn:
        conn.execute("DELETE FROM meta.fields WHERE dataset_id = %s", (dataset_id,))
        for p, db_col in zip(profiles, db_cols):
            by_family = suggest_all(p.inferred_type, p.summary, p.is_target)
            row = [
                dataset_id, p.ordinal, p.name, db_col, p.physical_dtype, p.inferred_type, p.confidence,
                p.reasons, p.is_target, *[p.summary.get(c) for c in SUMMARY_COLS],
                jsonb(p.stats), jsonb(p.charts), jsonb(by_family),
            ]
            fid = conn.execute(insert, row).fetchone()["id"]
            ids.append(fid)
            seed_decision(conn, dataset_id, fid, by_family)
    return ids


def seed_decision(conn, dataset_id: int, field_id: int, by_family: dict[str, dict]) -> None:
    """Shared decision from any family's suggestion (they agree on action/type) + settings per family."""
    first = next(iter(by_family.values()))
    conn.execute(
        """INSERT INTO meta.field_decisions (field_id, dataset_id, action, semantic_type, source, reviewed)
           VALUES (%s, %s, %s, %s, 'suggested', false)
           ON CONFLICT (field_id) DO NOTHING""",
        (field_id, dataset_id, first["action"], first["semantic_type"]),
    )
    for family, sugg in by_family.items():
        upsert_settings(conn, dataset_id, field_id, family, sugg, "suggested")


def upsert_settings(conn, dataset_id: int, field_id: int, family: str, d: dict, source: str) -> dict:
    return conn.execute(
        """INSERT INTO meta.field_settings
              (field_id, family, dataset_id, imputation, impute_value, add_missing_indicator, transform,
               scaling, encoding, params, source, updated_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
           ON CONFLICT (field_id, family) DO UPDATE SET
              imputation = EXCLUDED.imputation, impute_value = EXCLUDED.impute_value,
              add_missing_indicator = EXCLUDED.add_missing_indicator, transform = EXCLUDED.transform,
              scaling = EXCLUDED.scaling, encoding = EXCLUDED.encoding, params = EXCLUDED.params,
              source = EXCLUDED.source, updated_at = now()
           RETURNING *""",
        (field_id, family, dataset_id, d["imputation"], d.get("impute_value"), d.get("add_missing_indicator", False),
         d["transform"], d["scaling"], d["encoding"], jsonb(d.get("params") or {}), source),
    ).fetchone()


def _persist_correlations(dataset_id: int, field_ids: list[int], assocs) -> None:
    with connection() as conn:
        conn.execute("DELETE FROM meta.field_correlations WHERE dataset_id = %s", (dataset_id,))
        with conn.cursor() as cur, cur.copy(
            "COPY meta.field_correlations (dataset_id, field_a, field_b, method, value, strength, n_obs) FROM STDIN"
        ) as copy:
            for a in assocs:
                fa, fb = field_ids[a.i], field_ids[a.j]
                if fa > fb:
                    fa, fb = fb, fa
                copy.write_row((dataset_id, fa, fb, a.method, a.value, abs(a.value), a.n_obs))


def target_suggestion(t: str, stats: dict, is_target: bool, max_target_assoc: float | None) -> tuple[dict, bool]:
    """Per-family suggestion and leakage flag; fields almost perfectly associated with a target are probably leaks."""
    by_family = suggest_all(t, stats, is_target)
    leak = not is_target and max_target_assoc is not None and max_target_assoc >= LEAKAGE_THRESHOLD
    if leak:
        reason = f"association {max_target_assoc:.2f} with a target — likely target leakage"
        by_family = {
            fam: {**sugg, "action": "drop", "reasons": [reason, *sugg.get("reasons", [])]}
            for fam, sugg in by_family.items()
        }
    return json.loads(json.dumps(by_family)), leak


def refresh_target_state(conn, dataset_id: int) -> None:
    """Re-derive everything that depends on which fields are targets: max_target_assoc (non-targets only),
    suggestions, leakage flags, and the action of decisions nobody has touched (source = 'suggested').

    Call after fields.is_target changes or new correlations land. Rows only change when their value does,
    so updated_at stays meaningful for "changed since export".
    """
    conn.execute(
        """UPDATE meta.fields f SET max_target_assoc = s.m
             FROM (SELECT f2.id, CASE WHEN f2.is_target THEN NULL ELSE m.m END AS m
                     FROM meta.fields f2
                     LEFT JOIN (SELECT c.fid, max(c.strength) AS m
                                  FROM (SELECT field_a AS fid, field_b AS other, strength FROM meta.field_correlations
                                         WHERE dataset_id = %(ds)s
                                        UNION ALL
                                        SELECT field_b, field_a, strength FROM meta.field_correlations
                                         WHERE dataset_id = %(ds)s) c
                                  JOIN meta.fields t ON t.id = c.other AND t.is_target
                                 GROUP BY c.fid) m ON m.fid = f2.id
                    WHERE f2.dataset_id = %(ds)s) s
            WHERE f.id = s.id AND f.max_target_assoc IS DISTINCT FROM s.m""",
        {"ds": dataset_id},
    )
    rows = conn.execute(
        f"""SELECT id, inferred_type, is_target, max_target_assoc, leakage_flag, suggestion,
                   {", ".join(SUMMARY_COLS)}
              FROM meta.fields WHERE dataset_id = %s""",
        (dataset_id,),
    ).fetchall()
    for r in rows:
        by_family, leak = target_suggestion(
            r["inferred_type"], {c: r[c] for c in SUMMARY_COLS}, r["is_target"], r["max_target_assoc"]
        )
        if by_family != r["suggestion"] or leak != r["leakage_flag"]:
            conn.execute(
                "UPDATE meta.fields SET suggestion = %s, leakage_flag = %s WHERE id = %s",
                (jsonb(by_family), leak, r["id"]),
            )
    conn.execute(
        """UPDATE meta.field_decisions d SET action = s.action, updated_at = now()
             FROM (SELECT f.id, coalesce(f.suggestion -> ds.model_family ->> 'action', 'keep') AS action
                     FROM meta.fields f JOIN meta.datasets ds ON ds.id = f.dataset_id
                    WHERE f.dataset_id = %s) s
            WHERE d.field_id = s.id AND d.source = 'suggested' AND d.action IS DISTINCT FROM s.action""",
        (dataset_id,),
    )


def delete_dataset_files(dataset_id: int) -> None:
    s = get_settings()
    shutil.rmtree(s.datasets_dir / str(dataset_id), ignore_errors=True)
    execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier("raw", f"ds_{dataset_id}")))

