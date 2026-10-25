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
from .correlations import Assoc, compute_associations, screen_associations
from .db import connection, execute, fetch_all, fetch_one, json_safe, jsonb
from .semantics import suggest_all

log = logging.getLogger(__name__)

LEAKAGE_THRESHOLD = 0.95

SUMMARY_COLS = [
    "row_count",
    "null_count",
    "null_pct",
    "n_unique",
    "unique_pct",
    "mean",
    "std",
    "min_value",
    "max_value",
    "median",
    "skewness",
    "kurtosis",
    "zeros_pct",
    "outlier_pct",
    "top_value",
    "top_value_pct",
    "avg_length",
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
            """UPDATE meta.datasets SET status = %s, progress = %s, status_message = %s, error = NULL,
                      updated_at = now()
               WHERE id = %s""",
            (status, round(progress, 4), message, self.dataset_id),
        )


def process_dataset(dataset_id: int, upload_path: str | None, fmt: str) -> None:
    s = get_settings()
    st = _Status(dataset_id)
    try:
        ds = fetch_one("SELECT * FROM meta.datasets WHERE id = %s", (dataset_id,))
        if ds is None:  # deleted while queued
            return
        targets: list[str] = list(ds["target_columns"] or [])
        parquet_path = s.datasets_dir / str(dataset_id) / "raw.parquet"
        # ---- 1. ingest ------------------------------------------------------------
        st.set("ingesting", 0.01, "Converting to Parquet", force=True)
        warnings: list[str] = list(ds["warnings"] or [])  # a pull's notes come before processing's
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
        raw_table, load_warnings = ingest.load_raw_table(
            dataset_id,
            parquet_path,
            db_cols,
            s.copy_batch_rows,
            progress=lambda done, total: st.set(
                "ingesting", 0.05 + 0.30 * done / max(total, 1), f"Loading raw table: {done:,} / {total:,} rows"
            ),
        )
        warnings += load_warnings
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
            warnings.append(not_used_as_targets(rejected))
            execute(
                "UPDATE meta.datasets SET target_columns = %s, warnings = %s WHERE id = %s",
                ([t for t in targets if t not in rejected], warnings, dataset_id),
            )

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
        screened = screen_associations(
            [(i, p.screen_codes) for i, p in enumerate(profiles) if p.is_target],
            [(i, p.screen_codes) for i, p in enumerate(profiles) if p.screen_codes is not None and not p.is_target],
        )
        with connection() as conn:
            upsert_correlations(conn, dataset_id, [(field_ids[a.i], field_ids[a.j], a) for a in screened])
            refresh_target_state(conn, dataset_id)
        _apply_pending_recipe(dataset_id)

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


def _apply_pending_recipe(dataset_id: int) -> None:
    """A dataset uploaded to start from a recipe gets it before it is ready, so nobody assesses it first. A recipe
    that can no longer be used leaves a warning; the dataset then starts from the suggestions."""
    from fastapi import HTTPException

    from . import recipes, reuse
    from .routers.datasets import queue_target_work

    try:
        with connection() as conn:
            recipe = recipes.current(conn, dataset_id, lock="update")
            pending = conn.execute(
                "SELECT 1 FROM meta.recipes WHERE id = %s AND parent_id IS NOT NULL AND reuse IS NULL", (recipe["id"],)
            ).fetchone()
            if pending is None:
                return
            source = reuse.check_source(conn, recipe["parent_id"])
            _, added, uncorrelated = reuse.apply(conn, recipe, source, recipe["created_by"])
            start = queue_target_work(conn, dataset_id, uncorrelated, added, inline=True)
        start()
    except HTTPException as exc:
        execute(
            """UPDATE meta.recipes SET parent_id = NULL
                    WHERE id = (SELECT min(id) FROM meta.recipes WHERE dataset_id = %s)""",
            (dataset_id,),
        )
        execute(
            "UPDATE meta.datasets SET warnings = array_append(warnings, %s) WHERE id = %s",
            (f"Could not start from the recipe: {exc.detail}", dataset_id),
        )


def insert_field(conn, dataset_id: int, p: profiling.FieldProfile, db_col: str, derived: dict | None = None) -> int:
    """A profiled column as a field, with its suggestions seeded into the working decisions."""
    cols = [
        "dataset_id",
        "ordinal",
        "name",
        "db_column",
        "physical_dtype",
        "inferred_type",
        "inference_confidence",
        "inference_reasons",
        "is_target",
        *SUMMARY_COLS,
        "stats",
        "charts",
        "suggestion",
        "derived",
    ]
    insert = sql.SQL("INSERT INTO meta.fields ({}) VALUES ({}) RETURNING id").format(
        sql.SQL(", ").join(map(sql.Identifier, cols)), sql.SQL(", ").join(sql.Placeholder() * len(cols))
    )
    by_family = suggest_all(p.inferred_type, p.summary, p.is_target)
    row = [
        dataset_id,
        p.ordinal,
        p.name,
        db_col,
        p.physical_dtype,
        p.inferred_type,
        p.confidence,
        p.reasons,
        p.is_target,
        *[json_safe(p.summary.get(c)) for c in SUMMARY_COLS],
        jsonb(p.stats),
        jsonb(p.charts),
        jsonb(by_family),
        jsonb(derived) if derived else None,
    ]
    fid = conn.execute(insert, row).fetchone()["id"]
    seed_decision(conn, dataset_id, fid, by_family)
    return fid


def update_field(conn, field_id: int, p: profiling.FieldProfile, db_col: str) -> None:
    """A field's new profile after a refresh, keeping its id, name and whatever anyone decided about it. A decision
    or settings nobody chose (still the suggestion) follow the new suggestion, as a new field's would, except a
    target's type, which is part of its role, and a reviewed field's settings; actions and target state are
    re-derived by refresh_target_state afterwards."""
    by_family = suggest_all(p.inferred_type, p.summary, p.is_target)
    values = {
        "ordinal": p.ordinal,
        "db_column": db_col,
        "physical_dtype": p.physical_dtype,
        "inferred_type": p.inferred_type,
        "inference_confidence": p.confidence,
        "inference_reasons": p.reasons,
        **{c: json_safe(p.summary.get(c)) for c in SUMMARY_COLS},
        "stats": jsonb(p.stats),
        "charts": jsonb(p.charts),
        "suggestion": jsonb(by_family),
    }
    conn.execute(
        sql.SQL("UPDATE meta.fields SET {} WHERE id = %s").format(
            sql.SQL(", ").join(sql.SQL("{} = %s").format(sql.Identifier(c)) for c in values)
        ),
        [*values.values(), field_id],
    )
    first = next(iter(by_family.values()))
    d = conn.execute(
        "SELECT dataset_id, action, semantic_type, source, reviewed FROM meta.field_decisions WHERE field_id = %s",
        (field_id,),
    ).fetchone()
    # A reviewed field's settings were looked at with its decision, so they stay as they were.
    if d is None or d["reviewed"]:
        return
    t = d["semantic_type"]
    if d["source"] == "suggested" and d["action"] != "target" and t != first["semantic_type"]:
        t = first["semantic_type"]
        conn.execute(
            "UPDATE meta.field_decisions SET semantic_type = %s, updated_at = now() WHERE field_id = %s", (t, field_id)
        )
    if t != first["semantic_type"]:
        by_family = suggest_all(t, p.summary, p.is_target)
    from .assessment import settings_key  # assessment imports this module

    current = {
        r["family"]: r
        for r in conn.execute("SELECT * FROM meta.field_settings WHERE field_id = %s", (field_id,)).fetchall()
    }
    for family, sugg in by_family.items():
        row = current.get(family)
        # Rewritten only when the suggestion moved, so updated_at keeps meaning "changed".
        if row is None or (row["source"] != "user" and settings_key(row) != settings_key(sugg)):
            upsert_settings(conn, d["dataset_id"], field_id, family, sugg, "suggested")


def not_used_as_targets(names: list[str]) -> str:
    """The warning for named targets profiling turned down."""
    return (
        f"Not used as target(s): {', '.join(names)} (text, identifiers and dates can't be targets, nor labels with "
        "more than 100 values)"
    )


def _persist_profiles(dataset_id: int, profiles: list[profiling.FieldProfile], db_cols: list[str]) -> list[int]:
    with connection() as conn:
        conn.execute("DELETE FROM meta.fields WHERE dataset_id = %s", (dataset_id,))
        return [insert_field(conn, dataset_id, p, db_col) for p, db_col in zip(profiles, db_cols)]


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
        (
            field_id,
            family,
            dataset_id,
            d["imputation"],
            d.get("impute_value"),
            d.get("add_missing_indicator", False),
            d["transform"],
            d["scaling"],
            d["encoding"],
            jsonb(d.get("params") or {}),
            source,
        ),
    ).fetchone()


def _persist_correlations(dataset_id: int, field_ids: list[int], assocs) -> None:
    with connection() as conn:
        replace_correlations(conn, dataset_id, field_ids, assocs)


def replace_correlations(conn, dataset_id: int, field_ids: list[int], assocs) -> None:
    """The dataset's associations, all of them, from `compute_associations` output indexed like `field_ids`."""
    conn.execute("DELETE FROM meta.field_correlations WHERE dataset_id = %s", (dataset_id,))
    with (
        conn.cursor() as cur,
        cur.copy(
            "COPY meta.field_correlations (dataset_id, field_a, field_b, method, value, strength, n_obs) FROM STDIN"
        ) as copy,
    ):
        for a in assocs:
            fa, fb = field_ids[a.i], field_ids[a.j]
            if fa > fb:
                fa, fb = fb, fa
            copy.write_row((dataset_id, fa, fb, a.method, a.value, abs(a.value), a.n_obs))


def upsert_correlations(conn, dataset_id: int, pairs: list[tuple[int, int, Assoc]]) -> None:
    """Write (field id, field id, association) rows, replacing an existing value for the same pair."""
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO meta.field_correlations (dataset_id, field_a, field_b, method, value, strength, n_obs)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (field_a, field_b) DO UPDATE SET method = EXCLUDED.method, value = EXCLUDED.value,
                  strength = EXCLUDED.strength, n_obs = EXCLUDED.n_obs""",
            [(dataset_id, *sorted((fa, fb)), a.method, a.value, abs(a.value), a.n_obs) for fa, fb, a in pairs],
        )


def target_suggestion(t: str, stats: dict, is_target: bool, max_target_assoc: float | None) -> tuple[dict, bool]:
    """Per-family suggestion and leakage flag; fields almost perfectly associated with a target are probably leaks."""
    by_family = suggest_all(t, stats, is_target)
    leak = not is_target and max_target_assoc is not None and max_target_assoc >= LEAKAGE_THRESHOLD
    if leak:
        reason = (
            f"Drop: likely target leakage; its association with a target is {max_target_assoc:.2f}, too close to "
            "the target itself"
        )
        by_family = {
            fam: {**sugg, "action": "drop", "reasons": [reason, *sugg.get("reasons", [])]}
            for fam, sugg in by_family.items()
        }
    return json.loads(json.dumps(by_family)), leak


def refresh_target_state(conn, dataset_id: int) -> None:
    """Re-derive everything that depends on which fields are targets: max_target_assoc (non-targets only),
    suggestions, leakage flags, and the action of decisions nobody has touched (still the suggestion, not yet
    reviewed).

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
            WHERE d.field_id = s.id AND d.source = 'suggested' AND NOT d.reviewed
              AND d.action IS DISTINCT FROM s.action""",
        (dataset_id,),
    )
    # Assessors start from the suggestions too; the ones they haven't touched keep following them.
    conn.execute(
        """UPDATE meta.assessments a SET action = s.action, updated_at = now()
             FROM (SELECT f.id, coalesce(f.suggestion -> ds.model_family ->> 'action', 'keep') AS action
                     FROM meta.fields f JOIN meta.datasets ds ON ds.id = f.dataset_id
                    WHERE f.dataset_id = %(ds)s) s
            WHERE a.field_id = s.id AND a.source = 'suggested' AND NOT a.reviewed
              AND a.action IS DISTINCT FROM s.action
              AND a.recipe_id IN (SELECT id FROM meta.recipes WHERE dataset_id = %(ds)s AND phase = 'assessing')""",
        {"ds": dataset_id},
    )


def sweep_uploads(min_age_s: float = 600) -> list[Path]:
    """Remove uploads that no queued or ingesting dataset is waiting for (a crash mid-ingest leaves them behind).

    Files younger than `min_age_s` are left alone: an upload is streamed to disk before its row exists.
    """
    uploads = get_settings().uploads_dir
    if not uploads.is_dir():
        return []
    needed = {
        r["upload_path"]
        for r in fetch_all(
            "SELECT upload_path FROM meta.datasets WHERE status IN ('pending', 'ingesting') AND upload_path IS NOT NULL"
        )
    }
    cutoff = time.time() - min_age_s
    removed = []
    for p in uploads.iterdir():
        try:
            if p.is_file() and str(p) not in needed and p.stat().st_mtime < cutoff:
                p.unlink()
                removed.append(p)
        except OSError:
            continue
    if removed:
        log.info("removed %d orphaned upload(s)", len(removed))
    return removed


def delete_dataset_files(dataset_id: int) -> None:
    s = get_settings()
    shutil.rmtree(s.datasets_dir / str(dataset_id), ignore_errors=True)
    for table in (f"ds_{dataset_id}", f"ds_{dataset_id}_new"):  # _new: a refresh interrupted mid-load
        execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier("raw", table)))
