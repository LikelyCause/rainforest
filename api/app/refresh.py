"""Refreshing a connected dataset: its collections pulled again and swapped in under the same fields.

Everything slow happens beside the live data: the pull into `raw.parquet.new`, derived fields recomputed into it, the
load into `raw.ds_<id>_new`, profiling and associations. One short transaction then swaps file and table in and
reconciles the fields by name, so a field that persists keeps its id and with it its decisions, history and settings.
Each model family's latest finalized baseline is compared with the new fields (`reuse.match`) and the result stored on
the pull as its drift summary."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import polars as pl
from fastapi import HTTPException
from psycopg import sql
from pymongo.errors import PyMongoError

from . import derived, ingest, profiling, recipes, reuse, sources
from .analysis.runner import mark_stale
from .config import get_settings
from .correlations import compute_associations, screen_associations
from .db import connection, execute, fetch_all, fetch_one, jsonb
from .processing import (
    _Status,
    insert_field,
    not_used_as_targets,
    refresh_target_state,
    replace_correlations,
    update_field,
    upsert_correlations,
)
from .routers.fields import _merged_many, _write_history
from .sources.pull import names, pull_into, warnings_for

log = logging.getLogger(__name__)

NEW = "raw.parquet.new"


def run_refresh(dataset_id: int, pull_id: int) -> None:
    """The refresh job. Whoever queued it recorded the pull as running and set the dataset `refreshing`. A failure
    before the swap commits leaves the old data, fields and warnings as they were and the dataset `ready`."""
    src = None
    # Everything is inside the try: a failure that skipped the handler would leave the dataset locked.
    try:
        ds = fetch_one("SELECT * FROM meta.datasets WHERE id = %s", (dataset_id,))
        if ds is None:  # deleted while queued; the pull row went with it
            return
        src = fetch_one("SELECT * FROM meta.dataset_sources WHERE dataset_id = %s", (dataset_id,))
        if src is None:
            raise ValueError("The dataset has no source to pull from")
        _refresh(ds, src, pull_id)
    except Exception as exc:  # noqa: BLE001
        log.exception("refreshing dataset %s failed", dataset_id)
        # Before the dataset is ready again, so a refresh started next can't lose its own _new file or table to it.
        try:
            _discard_new(dataset_id)
        except Exception:  # noqa: BLE001 - the next refresh clears them too; the failure must still be recorded
            log.exception("could not remove the new data of dataset %s", dataset_id)
        execute(
            "UPDATE meta.dataset_pulls SET status = 'failed', error = %s, finished_at = now() WHERE id = %s",
            (_error(src, exc), pull_id),
        )
        execute(
            """UPDATE meta.datasets SET status = 'ready', progress = 1, status_message = 'Ready', error = NULL,
                      updated_at = now() WHERE id = %s""",
            (dataset_id,),
        )


def _refresh(ds: dict, src: dict, pull_id: int) -> None:
    s = get_settings()
    dataset_id = ds["id"]
    st = _Status(dataset_id)
    base = s.datasets_dir / str(dataset_id)
    live, new = base / "raw.parquet", base / NEW
    _discard_new(dataset_id)  # whatever an interrupted refresh left behind

    pulled = pull_into(dataset_id, src, new, status="refreshing")
    formulas = fetch_all(
        "SELECT id, name, derived FROM meta.fields WHERE dataset_id = %s AND derived IS NOT NULL ORDER BY ordinal",
        (dataset_id,),
    )
    # Before the load, so the new raw table has the derived columns too.
    dropped, dropped_warnings = _add_derived(new, formulas)
    columns = list(pl.read_parquet_schema(new))
    db_cols = ingest.sanitize_identifiers(columns)
    st.set("refreshing", 0.3, "Loading raw table into Postgres", force=True)
    _, load_warnings = ingest.load_raw_table(
        dataset_id,
        new,
        db_cols,
        s.copy_batch_rows,
        progress=lambda done, total: st.set(
            "refreshing", 0.3 + 0.2 * done / max(total, 1), f"Loading raw table: {done:,} / {total:,} rows"
        ),
        table_suffix="_new",
    )
    n_rows, profiles = profiling.profile_parquet(
        new,
        [t for t in ds["target_columns"] or [] if t in columns],
        s.profile_sample_rows,
        s.corr_sample_rows,
        s.corr_max_levels,
        progress=lambda i, n, name: st.set("refreshing", 0.5 + 0.3 * i / max(n, 1), f"Profiling {name} ({i + 1}/{n})"),
    )
    st.set("refreshing", 0.8, "Computing pairwise associations", force=True)
    assocs = compute_associations(
        [(i, p.corr_numeric) for i, p in enumerate(profiles) if p.corr_numeric is not None],
        [(i, p.corr_codes) for i, p in enumerate(profiles) if p.corr_codes is not None],
        progress=lambda f: st.set("refreshing", 0.8 + 0.15 * f, f"Computing associations ({f:.0%})"),
    )
    screened = screen_associations(
        [(i, p.screen_codes) for i, p in enumerate(profiles) if p.is_target],
        [(i, p.screen_codes) for i, p in enumerate(profiles) if p.screen_codes is not None and not p.is_target],
    )
    st.set("refreshing", 0.95, "Swapping in the new data", force=True)

    # No status writes from here on: they take the dataset row this transaction holds.
    with connection() as conn:
        recipes.lock_dataset(conn, dataset_id)
        _swap_tables(conn, dataset_id)
        row = conn.execute(
            "SELECT model_family, target_columns FROM meta.datasets WHERE id = %s", (dataset_id,)
        ).fetchone()
        family, targets = row["model_family"], list(row["target_columns"] or [])
        rejected = [p.name for p in profiles if p.name in targets and not p.is_target]
        rejected_ids, before = _untarget(conn, dataset_id, family, rejected)
        field_ids, vanished = _reconcile(conn, dataset_id, profiles, db_cols, dropped)
        replace_correlations(conn, dataset_id, field_ids, assocs)
        upsert_correlations(conn, dataset_id, [(field_ids[a.i], field_ids[a.j], a) for a in screened])
        refresh_target_state(conn, dataset_id)
        if rejected_ids:
            _write_history(conn, dataset_id, family, None, before, _merged_many(conn, rejected_ids, family), None)
        mark_stale(conn, dataset_id)
        # The refresh reports anew: none of the old warnings (the last pull's notes, processing's, a recipe's) are
        # kept, since they described data that is gone.
        warnings = warnings_for(pulled["flatten_notes"], pulled["removed_duplicates"], src["dedupe"]) + load_warnings
        if others := [n for n in vanished if n not in targets]:
            warnings.append(f"Fields no longer in the data: {names(others)}")
        warnings += [f"Target column no longer in the data: {t}" for t in targets if t not in columns]
        if rejected:
            warnings.append(not_used_as_targets(rejected))
        warnings += dropped_warnings
        fields = conn.execute(
            "SELECT name, inferred_type, null_pct, n_unique FROM meta.fields WHERE dataset_id = %s ORDER BY ordinal",
            (dataset_id,),
        ).fetchall()
        drift = drift_by_family(conn, dataset_id, fields)
        conn.execute(
            """UPDATE meta.dataset_pulls SET status = 'done', rows = %s, per_source = %s, flatten_notes = %s,
                      drift = %s, error = NULL, finished_at = now() WHERE id = %s""",
            (
                n_rows,
                jsonb(pulled["per_source"]),
                jsonb(pulled["flatten_notes"]),
                jsonb(drift) if drift else None,
                pull_id,
            ),
        )
        conn.execute(
            """UPDATE meta.datasets SET status = 'ready', progress = 1, status_message = 'Ready', error = NULL,
                      warnings = %s, target_columns = %s, row_count = %s, column_count = %s, parquet_path = %s,
                      raw_table = %s, profiled_at = now(), updated_at = now()
                WHERE id = %s""",
            (
                warnings,
                [t for t in targets if t in columns and t not in rejected],
                n_rows,
                len(columns),
                str(live),
                f"raw.ds_{dataset_id}",
                dataset_id,
            ),
        )
        # Last: anything above failing rolls the database back with the old file still in place.
        os.replace(new, live)
    log.info("dataset %s refreshed: %s rows x %s fields", dataset_id, n_rows, len(profiles))


def _add_derived(path: Path, formulas: list[dict]) -> tuple[list[int], list[str]]:
    """Recompute the derived fields onto the pulled data at `path`, in the order they were added (one may read
    another). Returns the ids of those that can't be, and why, as warnings."""
    if not formulas:
        return [], []
    df = pl.read_parquet(path)
    pulled = set(df.columns)
    dropped, warnings = [], []
    for f in formulas:
        kind, spec = f["derived"]["kind"], f["derived"]["spec"]
        try:
            if f["name"] in pulled:
                raise ValueError("the data now has a column of that name")
            derived.check_spec(kind, spec, df.columns)
            df = df.with_columns(derived.compute(df, kind, spec).alias(f["name"]))
        except (HTTPException, ValueError, pl.exceptions.PolarsError) as exc:
            reason = exc.detail if isinstance(exc, HTTPException) else str(exc)
            dropped.append(f["id"])
            warnings.append(f"Derived field removed: {f['name']} — {reason}")
    tmp = path.with_name(path.name + ".tmp")
    try:
        df.write_parquet(tmp, row_group_size=100_000)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return dropped, warnings


def _swap_tables(conn, dataset_id: int) -> None:
    """`raw.ds_<id>_new` becomes `raw.ds_<id>`; readers see one table or the other, never neither."""
    live, old = f"ds_{dataset_id}", f"ds_{dataset_id}_old"
    conn.execute(
        sql.SQL("ALTER TABLE IF EXISTS {} RENAME TO {}").format(sql.Identifier("raw", live), sql.Identifier(old))
    )
    conn.execute(
        sql.SQL("ALTER TABLE {} RENAME TO {}").format(sql.Identifier("raw", f"{live}_new"), sql.Identifier(live))
    )
    conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier("raw", old)))
    # The key was named after the _new table; the next refresh's _new table would otherwise get a numbered name.
    pkey = conn.execute(
        "SELECT conname FROM pg_constraint WHERE conrelid = %s::regclass AND contype = 'p'", (f"raw.{live}",)
    ).fetchone()
    if pkey is not None and pkey["conname"] != f"{live}_pkey":
        conn.execute(
            sql.SQL("ALTER TABLE {} RENAME CONSTRAINT {} TO {}").format(
                sql.Identifier("raw", live), sql.Identifier(pkey["conname"]), sql.Identifier(f"{live}_pkey")
            )
        )


def _untarget(conn, dataset_id: int, family: str, rejected: list[str]) -> tuple[list[int], dict[int, dict]]:
    """Targets profiling turns down on the new data (labels with more than 100 values, say) stop being targets, as at
    upload: features again, unreviewed, so they follow their suggestions. Returns their ids and decisions before."""
    if not rejected:
        return [], {}
    ids = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM meta.fields WHERE dataset_id = %s AND name = ANY(%s)", (dataset_id, rejected)
        ).fetchall()
    ]
    before = _merged_many(conn, ids, family)
    conn.execute(
        """UPDATE meta.field_decisions SET action = 'keep', source = 'suggested', reviewed = false, updated_at = now()
            WHERE field_id = ANY(%s) AND action = 'target'""",
        (ids,),
    )
    conn.execute("UPDATE meta.fields SET is_target = false WHERE id = ANY(%s)", (ids,))
    return ids, before


def _reconcile(
    conn, dataset_id: int, profiles: list[profiling.FieldProfile], db_cols: list[str], dropped: list[int]
) -> tuple[list[int], list[str]]:
    """Fields by name: a persisting one is updated in place, a new one inserted with seeded suggestions, a vanished
    one deleted (its decisions, history, settings and correlations go with it), and so is a derived field that
    couldn't be recomputed. Returns the field ids in profile order and the names of the vanished fields."""
    conn.execute("DELETE FROM meta.fields WHERE id = ANY(%s)", (dropped,))
    existing = {
        r["name"]: r["id"]
        for r in conn.execute("SELECT id, name FROM meta.fields WHERE dataset_id = %s", (dataset_id,)).fetchall()
    }
    present = {p.name for p in profiles}
    vanished = [n for n in existing if n not in present]
    conn.execute("DELETE FROM meta.fields WHERE id = ANY(%s)", ([existing[n] for n in vanished],))
    # (dataset_id, ordinal) is unique and not deferrable, so the survivors move out of the way before renumbering.
    conn.execute("UPDATE meta.fields SET ordinal = -1 - ordinal WHERE dataset_id = %s", (dataset_id,))
    ids = []
    for p, db_col in zip(profiles, db_cols):
        if p.name in existing:
            update_field(conn, existing[p.name], p, db_col)
            ids.append(existing[p.name])
        else:
            ids.append(insert_field(conn, dataset_id, p, db_col))
    return ids, vanished


def drift_by_family(conn, dataset_id: int, fields: list[dict]) -> dict[str, dict]:
    """`fields` (name, inferred_type, null_pct, n_unique) against each model family's latest finalized baseline:
    carried, reopened and missing, as `reuse.match` reports them. Empty when no baseline is finalized yet."""
    rows = conn.execute(
        """SELECT DISTINCT ON (family) family, snapshot FROM meta.recipes
            WHERE dataset_id = %s AND phase = 'finalized' AND snapshot IS NOT NULL
            ORDER BY family, finalized_at DESC, id DESC""",
        (dataset_id,),
    ).fetchall()
    return {r["family"]: reuse.match(r["snapshot"], fields) for r in rows}


def _discard_new(dataset_id: int) -> None:
    base = get_settings().datasets_dir / str(dataset_id)
    for name in (NEW, f"{NEW}.tmp"):
        (base / name).unlink(missing_ok=True)
    shutil.rmtree(base / "pull", ignore_errors=True)
    execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier("raw", f"ds_{dataset_id}_new")))


def _error(src: dict | None, exc: Exception) -> str:
    if isinstance(exc, PyMongoError):  # pull_into redacts the pull's own; this keeps any other path redacted
        try:
            source = sources.registry().get(src["connection"]) if src else None
        except Exception:  # noqa: BLE001 - the registry failing too must not keep the dataset locked
            source = None
        return source.redact(str(exc)) if source else "The MongoDB driver failed"
    return str(exc) or type(exc).__name__
