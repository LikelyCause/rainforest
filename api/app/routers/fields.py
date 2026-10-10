from __future__ import annotations

import logging
import uuid
from pathlib import Path

import polars as pl
from fastapi import APIRouter, HTTPException, Query
from psycopg import sql

from .. import jobs, recipes
from ..assessment import Source, lock_decisions, source_for, write_decision, write_settings
from ..assessment import restore as restore_assessment
from ..auth import CurrentUser
from ..config import get_settings
from ..correlations import Assoc, screen_associations
from ..db import connection, execute, fetch_all, fetch_one, jsonb
from ..processing import SUMMARY_COLS, refresh_target_state, upsert_correlations, upsert_settings
from ..profiling import column_datetime_format, needs_screen, sample_indices, screen_codes
from ..schemas import BulkDecisionIn, DecisionIn
from ..semantics import (FAMILIES, NON_FEATURE_TYPES, SETTING_KEYS, settings_of, suggest, target_problem,
                         validate_decision)
from .query import fields_where

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["fields"])

SHARED_COLS = ["action", "semantic_type", "notes", "reviewed"]
STAT_KEYS = ["null_pct", "skewness", "outlier_pct", "min_value", "n_unique"]
# Profiling builds no association sample for these, so as targets they need one computed on demand.
UNCORRELATED_TYPES = {"text", "identifier"}


def _field(field_id: int) -> dict:
    f = fetch_one(
        "SELECT f.*, d.model_family AS active_family FROM meta.fields f "
        "JOIN meta.datasets d ON d.id = f.dataset_id WHERE f.id = %s",
        (field_id,),
    )
    if f is None:
        raise HTTPException(404, "Field not found")
    return f


def _family(src: Source, family: str | None) -> str:
    """The model family a request reads or writes settings for: always its baseline's."""
    if family is not None and family != src.family:
        raise HTTPException(422, f"{recipes.label(src.recipe)} prepares data for "
                                 f"{FAMILIES[src.family]['label'].lower()}; it has no {family!r} settings")
    return src.family


def view_targets(conn, src: Source) -> list[str]:
    """The targets a non-team source decides on, in chain order (an experiment's own order first)."""
    names = [r["name"] for r in conn.execute(
        sql.SQL("""SELECT f.name FROM meta.fields f JOIN {} d ON d.field_id = f.id
                    WHERE d.action = 'target' ORDER BY f.ordinal""").format(src.decisions())).fetchall()]
    order = (src.experiment or {}).get("target_order") or []
    return [n for n in order if n in names] + [n for n in names if n not in order]


def _merged_sql(src: Source | None) -> sql.Composable:
    """Shared decision joined to one family's settings, from `src` (the team's decision when None)."""
    return sql.SQL(
        """SELECT d.field_id, d.dataset_id, d.action, d.semantic_type, d.notes, d.reviewed, d.source,
                  d.updated_at, s.family, s.imputation, s.impute_value, s.add_missing_indicator, s.transform,
                  s.scaling, s.encoding, s.params, s.source AS settings_source
             FROM {} d
             JOIN {} s ON s.field_id = d.field_id AND s.family = %s"""
    ).format(src.decisions() if src else sql.SQL("meta.field_decisions"),
             src.settings() if src else sql.SQL("meta.field_settings"))


def _merged(conn, field_id: int, family: str, src: Source | None = None) -> dict:
    """Shared decision + this family's settings, as one flat object."""
    return conn.execute(_merged_sql(src) + sql.SQL(" WHERE d.field_id = %s"), (family, field_id)).fetchone()


def _merged_many(conn, field_ids: list[int], family: str, src: Source | None = None) -> dict[int, dict]:
    rows = conn.execute(_merged_sql(src) + sql.SQL(" WHERE d.field_id = ANY(%s)"), (family, field_ids)).fetchall()
    return {r["field_id"]: r for r in rows}


@router.get("/fields/{field_id}")
def get_field(field_id: int, user: CurrentUser, family: str | None = None, view: str | None = None) -> dict:
    """A field with the decision the viewer works on (the source `view` names, else the baseline in progress); in
    an experiment, also the baseline's own decision for it."""
    f = _field(field_id)
    baseline = None
    with connection() as conn:
        src = source_for(conn, f["dataset_id"], user, view=view)
        fam = _family(src, family)
        decision = _merged(conn, field_id, fam, src)
        locked = None if src.kind in ("experiment", "baseline") else recipes.locked_problem(conn, src.recipe, [field_id])
        if src.team:
            targets = conn.execute(
                """SELECT other_field_id AS field_id, other_field_name AS name, method, value, strength
                     FROM meta.v_correlations WHERE field_id = %s AND other_is_target ORDER BY other_field_name""",
                (field_id,),
            ).fetchall()
        else:
            targets = conn.execute(
                """SELECT other_field_id AS field_id, other_field_name AS name, method, value, strength
                     FROM meta.v_correlations WHERE field_id = %s AND other_field_name = ANY(%s)
                    ORDER BY other_field_name""",
                (field_id, view_targets(conn, src)),
            ).fetchall()
        if src.kind == "experiment":
            entry = recipes.snapshot_of(conn, src.recipe)["fields"].get(f["name"])
            if entry is not None:
                baseline = {**entry["decision"], "settings": entry["settings"].get(fam)}
    suggestions = f.pop("suggestion") or {}
    f["suggestion"] = suggestions.get(fam, {})
    out = {"field": f, "family": fam, "decision": decision, "target_associations": targets, "locked": locked}
    if src.kind == "experiment":
        out["baseline"] = baseline
    return out


@router.get("/fields/{field_id}/correlations")
def correlations(field_id: int, limit: int = Query(20, ge=1, le=200), exclude_targets: bool = False) -> list[dict]:
    _field(field_id)
    return fetch_all(
        f"""SELECT other_field_id AS field_id, other_field_name AS name, other_type AS type,
                   other_is_target AS is_target, method, value, strength, n_obs
              FROM meta.v_correlations
             WHERE field_id = %s {"AND NOT other_is_target" if exclude_targets else ""}
             ORDER BY strength DESC LIMIT %s""",
        (field_id, limit),
    )


@router.get("/fields/{field_id}/history")
def history(field_id: int) -> list[dict]:
    return fetch_all(
        """SELECT h.changed_at, h.family, h.previous, h.current, h.batch_id, h.user_id, u.name AS user_name
             FROM meta.decision_history h LEFT JOIN meta.users u ON u.id = h.user_id
            WHERE h.field_id = %s ORDER BY h.changed_at DESC LIMIT 50""",
        (field_id,),
    )


@router.put("/fields/{field_id}/decision")
def save_decision(field_id: int, body: DecisionIn, user: CurrentUser, view: str | None = None) -> dict:
    f = _field(field_id)
    d = body.model_dump(exclude={"family"})
    screen_after = False
    with connection() as conn:
        src = source_for(conn, f["dataset_id"], user, lock="share", view=view)
        fam = _family(src, body.family)
        errors = validate_decision(d, fam, n_unique=f["n_unique"])
        if errors:
            # A type change brings the new type's suggested settings when the ones sent were for the old type.
            was = _merged(conn, field_id, fam, src)
            if was is not None and was["semantic_type"] != d["semantic_type"]:
                sugg = suggest(d["semantic_type"], {k: f[k] for k in STAT_KEYS}, False, fam)
                d = {**d, **settings_of(sugg)}
                errors = validate_decision(d, fam, n_unique=f["n_unique"])
        if errors:
            raise HTTPException(422, {"message": "Invalid decision", "errors": errors})
        src.require_write()
        if (locked := recipes.locked_problem(conn, src.recipe, [field_id])) is not None:
            raise HTTPException(409, locked)
        lock_decisions(conn, src, [field_id])
        prev = _merged(conn, field_id, fam, src)
        if prev is None:
            raise HTTPException(409, "This field has no decision to change")
        # Provenance follows real changes: confirming or annotating a suggestion leaves it a suggestion.
        decided = prev["action"] != d["action"] or prev["semantic_type"] != d["semantic_type"]
        write_decision(conn, src, field_id, d, decided)
        settings = settings_of(d)
        write_settings(conn, src, f["dataset_id"], field_id, fam, settings,
                       prev["settings_source"] if _same_settings(prev, settings) else "user")
        if prev["semantic_type"] != d["semantic_type"] and src.kind != "experiment":
            # Other families' settings were chosen for the old type; re-derive them for the new one.
            stats = {k: f[k] for k in STAT_KEYS}
            for other in FAMILIES:
                if other != fam:
                    write_settings(conn, src, f["dataset_id"], field_id, other,
                                   suggest(d["semantic_type"], stats, False, other), "suggested")
        row = _merged(conn, field_id, fam, src)
        if src.kind == "experiment" and (prev["action"] == "target") != (d["action"] == "target"):
            screen_after = _experiment_target_changed(conn, src, f, d["action"] == "target")
        if src.kind == "experiment" and d["semantic_type"] == "datetime" and not (f["stats"] or {}).get("datetime_format"):
            # The column's format is a fact about the data, not anyone's choice, so an experiment may store it.
            _remember_datetime_format(conn, f)
        if not src.team:
            # Only the team decision has history and target state: an assessor's choices stay theirs until review,
            # and an experiment's stay in the experiment.
            if screen_after:
                conn.commit()
                jobs.submit(target_associations_job, f["dataset_id"], [field_id])
            return row
        if d["semantic_type"] == "datetime" and not (f["stats"] or {}).get("datetime_format"):
            _remember_datetime_format(conn, f)
        conn.execute(
            """INSERT INTO meta.decision_history (field_id, dataset_id, family, previous, current, user_id)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (field_id, f["dataset_id"], fam, jsonb(_jsonable(prev)), jsonb(_jsonable(row)), user["id"]),
        )
        screen = False
        if (prev["action"] == "target") != (d["action"] == "target"):
            _sync_targets(conn, f["dataset_id"], user_id=user["id"])
            screen = d["action"] == "target" and has_screen_fields(conn, f["dataset_id"])
    if screen:
        jobs.submit(screen_targets, f["dataset_id"], [field_id])
    return row


def _remember_datetime_format(conn, f: dict) -> None:
    """Detect a retyped field's date format on the whole column once, so export chunks, Explore and
    associations all read it the same way instead of each guessing from their own slice."""
    ds = conn.execute("SELECT parquet_path FROM meta.datasets WHERE id = %s", (f["dataset_id"],)).fetchone()
    if not ds or not ds["parquet_path"]:
        return
    fmt = column_datetime_format(pl.read_parquet(ds["parquet_path"], columns=[f["name"]]).to_series())
    if fmt:
        conn.execute(
            """UPDATE meta.fields
                  SET stats = coalesce(stats, '{}'::jsonb) || jsonb_build_object('datetime_format', %s::text)
                WHERE id = %s""",
            (fmt, f["id"]),
        )


@router.post("/datasets/{dataset_id}/decisions/bulk")
def bulk_decisions(dataset_id: int, body: BulkDecisionIn, user: CurrentUser, view: str | None = None) -> dict:
    _active_family(dataset_id)
    with connection() as conn:
        src = source_for(conn, dataset_id, user, lock="share", view=view)
        src.require_write()
        fam = src.family
        if body.filters is not None:
            target = sql.SQL("d.field_id IN (SELECT field_id FROM {} WHERE {})").format(
                src.fields_view(), fields_where(dataset_id, body.filters))
        else:
            target = sql.SQL("d.field_id = ANY({})").format(sql.Literal(body.field_ids))
        # Fields carried over into this version are locked; bulk edits pass over them.
        if (open_ := recipes.open_field_ids(conn, src.recipe)) is not None:
            target = sql.SQL("{} AND d.field_id = ANY({})").format(target, sql.Literal(open_))
        # Resolve and lock in one statement so the set can't drift between preview and write. An assessor's rows
        # are only theirs to change, so they are locked right after.
        rows = conn.execute(
            sql.SQL("""SELECT d.field_id, f.name, d.action, d.semantic_type, d.reviewed, d.notes, f.suggestion
                         FROM {} d JOIN meta.fields f ON f.id = d.field_id
                        WHERE d.dataset_id = {} AND {} AND NOT d.field_id = ANY({})
                        ORDER BY f.ordinal {}""").format(
                src.decisions(), sql.Literal(dataset_id), target, sql.Literal(body.excluded_ids),
                sql.SQL("FOR UPDATE OF d" if src.team else "")),
        ).fetchall()
        ids = [r["field_id"] for r in rows]
        if not src.team:
            lock_decisions(conn, src, ids)
        settings: dict[int, dict[str, dict]] = {}
        for srow in conn.execute(sql.SQL("SELECT * FROM {} s WHERE field_id = ANY(%s)").format(src.settings()),
                                 (ids,)).fetchall():
            settings.setdefault(srow["field_id"], {})[srow["family"]] = srow
        if src.kind in ("experiment", "baseline"):
            # An experiment's copy holds its baseline's model type only; other types' suggestions don't apply.
            rows = [{**r, "suggestion": {src.family: (r["suggestion"] or {}).get(src.family)}
                     if (r["suggestion"] or {}).get(src.family) else {}} for r in rows]
        planned = plan_bulk(rows, settings, body)
        result = summarize_bulk(len(rows), planned)
        if body.dry_run or not planned:
            conn.rollback()
            return {"batch_id": None, "dry_run": body.dry_run, **result}

        batch_id = uuid.uuid4()
        changed_ids = [c["field_id"] for c in planned]
        snapshot = _snapshot(conn, src, changed_ids)
        prev = _merged_many(conn, changed_ids, fam, src)
        notes = {r["field_id"]: r["notes"] for r in rows}
        for c in planned:
            decided = any(c["after"][k] != c["before"][k] for k in ("action", "semantic_type"))
            write_decision(conn, src, c["field_id"], {**c["after"], "notes": notes[c["field_id"]]}, decided)
            for family, sugg in c["settings"].items():
                write_settings(conn, src, dataset_id, c["field_id"], family, sugg, "suggested")
        summary = jsonb({**{k: v for k, v in result.items() if k != "changes"},
                         "request": body.model_dump(exclude={"dry_run"})})
        if src.personal:
            conn.execute(
                """INSERT INTO meta.assessment_batches (id, recipe_id, user_id, summary, snapshot)
                   VALUES (%s, %s, %s, %s, %s)""",
                (batch_id, src.recipe["id"], src.user_id, summary, jsonb(snapshot)),
            )
            return {"batch_id": str(batch_id), "dry_run": False, **result}
        if src.kind == "experiment":
            conn.execute(
                """INSERT INTO meta.experiment_batches (id, experiment_id, user_id, summary, snapshot)
                   VALUES (%s, %s, %s, %s, %s)""",
                (batch_id, src.experiment["id"], user["id"], summary, jsonb(snapshot)),
            )
            return {"batch_id": str(batch_id), "dry_run": False, **result}
        _write_history(conn, dataset_id, fam, batch_id, prev, _merged_many(conn, changed_ids, fam), user["id"])
        conn.execute(
            "INSERT INTO meta.bulk_batches (id, dataset_id, summary, snapshot, user_id) VALUES (%s, %s, %s, %s, %s)",
            (batch_id, dataset_id, summary, jsonb(snapshot), user["id"]),
        )
        _sync_targets(conn, dataset_id, user_id=user["id"])
    return {"batch_id": str(batch_id), "dry_run": False, **result}


@router.post("/datasets/{dataset_id}/decisions/bulk/{batch_id}/undo")
def undo_bulk(dataset_id: int, batch_id: str, user: CurrentUser, view: str | None = None) -> dict:
    try:
        bid = uuid.UUID(batch_id)
    except ValueError:
        raise HTTPException(404, "Bulk batch not found") from None
    _active_family(dataset_id)
    with connection() as conn:
        src = source_for(conn, dataset_id, user, lock="share", view=view)
        fam = src.family
        batches = "assessment_batches" if src.personal else "experiment_batches"
        if src.personal:
            batch = conn.execute(
                """SELECT snapshot, undone_at FROM meta.assessment_batches
                    WHERE id = %s AND recipe_id = %s AND user_id = %s FOR UPDATE""",
                (bid, src.recipe["id"], src.user_id),
            ).fetchone()
        elif src.kind == "experiment":
            batch = conn.execute(
                """SELECT snapshot, undone_at FROM meta.experiment_batches
                    WHERE id = %s AND experiment_id = %s FOR UPDATE""",
                (bid, src.experiment["id"]),
            ).fetchone()
        else:
            batch = conn.execute(
                "SELECT snapshot, undone_at FROM meta.bulk_batches WHERE id = %s AND dataset_id = %s FOR UPDATE",
                (bid, dataset_id),
            ).fetchone()
        if batch is None:
            raise HTTPException(404, "Bulk batch not found")
        src.require_write()
        if batch["undone_at"] is not None:
            raise HTTPException(409, "This bulk change was already undone")
        snapshot: dict = batch["snapshot"]
        if not src.team:
            ids = sorted(int(k) for k in snapshot)
            lock_decisions(conn, src, ids)
            for fid in ids:
                restore_assessment(conn, src, fid, snapshot[str(fid)])
            conn.execute(sql.SQL("UPDATE meta.{} SET undone_at = now() WHERE id = %s").format(sql.Identifier(batches)),
                         (bid,))
            return {"restored": len(ids), "batch_id": None}
        # Fields deleted since (dataset reprocessed) can't be restored; skip them.
        ids = [r["field_id"] for r in conn.execute(
            """SELECT field_id FROM meta.field_decisions WHERE dataset_id = %s AND field_id = ANY(%s)
                ORDER BY field_id FOR UPDATE""",
            (dataset_id, [int(k) for k in snapshot]),
        ).fetchall()]
        if (locked := recipes.locked_problem(conn, src.recipe, ids)) is not None:
            raise HTTPException(409, locked)
        prev = _merged_many(conn, ids, fam)
        for fid in ids:
            snap = snapshot[str(fid)]
            conn.execute(
                """UPDATE meta.field_decisions d
                      SET action = r.action, semantic_type = r.semantic_type, notes = r.notes,
                          reviewed = r.reviewed, source = r.source, updated_at = r.updated_at
                     FROM jsonb_populate_record(NULL::meta.field_decisions, %s) r
                    WHERE d.field_id = r.field_id""",
                (jsonb(snap["decision"]),),
            )
            families = [s["family"] for s in snap["settings"]]
            conn.execute("DELETE FROM meta.field_settings WHERE field_id = %s AND NOT family = ANY(%s)",
                         (fid, families))
            conn.execute(
                """INSERT INTO meta.field_settings
                   SELECT * FROM jsonb_populate_recordset(NULL::meta.field_settings, %s)
                   ON CONFLICT (field_id, family) DO UPDATE SET
                      imputation = EXCLUDED.imputation, impute_value = EXCLUDED.impute_value,
                      add_missing_indicator = EXCLUDED.add_missing_indicator, transform = EXCLUDED.transform,
                      scaling = EXCLUDED.scaling, encoding = EXCLUDED.encoding, params = EXCLUDED.params,
                      source = EXCLUDED.source, updated_at = EXCLUDED.updated_at""",
                (jsonb(snap["settings"]),),
            )
        # The undo gets its own batch id so its history rows are distinguishable from the original batch.
        undo_id = uuid.uuid4()
        _write_history(conn, dataset_id, fam, undo_id, prev, _merged_many(conn, ids, fam), user["id"])
        conn.execute("UPDATE meta.bulk_batches SET undone_at = now() WHERE id = %s", (bid,))
        _sync_targets(conn, dataset_id, user_id=user["id"])
    return {"restored": len(ids), "batch_id": str(undo_id)}


def plan_bulk(rows: list[dict], settings: dict[int, dict[str, dict]], body: BulkDecisionIn) -> list[dict]:
    """Per-field before/after for a bulk request; only fields that would actually change are returned.

    `settings` maps field_id -> family -> current field_settings row. Each result carries the families
    whose settings an accepted suggestion would rewrite.
    """
    out = []
    for r in rows:
        before = {k: r[k] for k in ("action", "semantic_type", "reviewed")}
        after = dict(before)
        new_settings: dict[str, dict] = {}
        by_family: dict = r.get("suggestion") or {}
        if body.accept_suggestion and by_family:
            first = next(iter(by_family.values()))
            # Bulk never moves a field into or out of the target role; that stays a deliberate single edit.
            if before["action"] != "target" and first["action"] != "target":
                after["action"] = first["action"]
            after["semantic_type"] = first["semantic_type"]
            after["reviewed"] = True
            current = settings.get(r["field_id"], {})
            retyped = after["semantic_type"] != before["semantic_type"]
            # Settings you customised stay, unless the type changes under them and they no longer apply.
            new_settings = {f: sg for f, sg in by_family.items()
                            if not _same_settings(current.get(f), sg)
                            and (retyped or (current.get(f) or {}).get("source") != "user")}
        if (body.action is not None and after["action"] != "target"
                and not (body.action == "keep" and after["semantic_type"] in NON_FEATURE_TYPES)):
            after["action"] = body.action
        if body.reviewed is not None:
            after["reviewed"] = body.reviewed
        if after != before or new_settings:
            out.append({"field_id": r["field_id"], "name": r["name"], "before": before, "after": after,
                        "settings": new_settings})
    return out


def summarize_bulk(matched: int, planned: list[dict]) -> dict:
    transitions: dict[str, int] = {}
    for c in planned:
        a, b = c["before"]["action"], c["after"]["action"]
        if a != b:
            transitions[f"{a}→{b}"] = transitions.get(f"{a}→{b}", 0) + 1
    return {
        "matched": matched,
        "changed": len(planned),
        "unchanged": matched - len(planned),
        "reviewed_changed": sum(1 for c in planned if c["before"]["reviewed"]),
        "transitions": transitions,
        "changes": [{k: c[k] for k in ("field_id", "name", "before", "after")} for c in planned],
    }


def _same_settings(row: dict | None, sugg: dict) -> bool:
    if row is None:
        return False
    norm = {
        "impute_value": lambda v: None if v is None else str(v),
        "add_missing_indicator": bool,
        "params": lambda v: v or {},
    }
    return all(norm.get(k, lambda v: v)(row.get(k)) == norm.get(k, lambda v: v)(sugg.get(k)) for k in SETTING_KEYS)


def _active_family(dataset_id: int) -> str:
    ds = fetch_one("SELECT model_family FROM meta.datasets WHERE id = %s", (dataset_id,))
    if ds is None:
        raise HTTPException(404, "Dataset not found")
    return ds["model_family"]


def _snapshot(conn, src: Source, field_ids: list[int]) -> dict:
    rows = conn.execute(
        sql.SQL("""SELECT d.field_id, to_jsonb(d) AS decision,
                          coalesce((SELECT jsonb_agg(to_jsonb(s) ORDER BY s.family) FROM {} s
                                     WHERE s.field_id = d.field_id), '[]') AS settings
                     FROM {} d WHERE d.field_id = ANY(%s)""").format(src.settings(), src.decisions()),
        (field_ids,),
    ).fetchall()
    return {str(r["field_id"]): {"decision": r["decision"], "settings": r["settings"]} for r in rows}


def _write_history(conn, dataset_id: int, family: str, batch_id: uuid.UUID | None, prev: dict, cur: dict,
                   user_id: int | None) -> None:
    """`user_id` is None only for changes a background job makes on its own (knock-on target state)."""
    with conn.cursor() as c:
        c.executemany(
            """INSERT INTO meta.decision_history (field_id, dataset_id, family, previous, current, batch_id, user_id)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            [(fid, dataset_id, family, jsonb(_jsonable(prev.get(fid))), jsonb(_jsonable(row)), batch_id, user_id)
             for fid, row in cur.items()],
        )


def _sync_targets(conn, dataset_id: int, before: dict[int, dict] | None = None,
                  order: list[str] | None = None, user_id: int | None = None) -> None:
    """Make is_target, target_columns and every target-derived value follow the decisions.

    Knock-on decision changes (leak drops appearing or clearing) get history rows; pass `before` (taken
    before the caller's own edits) to record those edits too. `order` fixes the target order, otherwise
    existing targets keep their place and new ones follow in column order.
    """
    ds = conn.execute("SELECT model_family, target_columns FROM meta.datasets WHERE id = %s",
                      (dataset_id,)).fetchone()
    fam = ds["model_family"]
    ids = [r["field_id"] for r in conn.execute(
        "SELECT field_id FROM meta.field_decisions WHERE dataset_id = %s", (dataset_id,)).fetchall()]
    if before is None:
        before = _merged_many(conn, ids, fam)
    conn.execute(
        """UPDATE meta.fields f SET is_target = (d.action = 'target')
             FROM meta.field_decisions d
            WHERE d.field_id = f.id AND f.dataset_id = %s AND f.is_target IS DISTINCT FROM (d.action = 'target')""",
        (dataset_id,),
    )
    names = [r["name"] for r in conn.execute(
        "SELECT name FROM meta.fields WHERE dataset_id = %s AND is_target ORDER BY ordinal", (dataset_id,)
    ).fetchall()]
    conn.execute(
        "UPDATE meta.datasets SET target_columns = %s, updated_at = now() WHERE id = %s",
        (target_order(order if order is not None else list(ds["target_columns"] or []), names), dataset_id),
    )
    refresh_target_state(conn, dataset_id)
    after = _merged_many(conn, ids, fam)
    changed = {fid: row for fid, row in after.items() if _decision_key(row) != _decision_key(before.get(fid))}
    if changed:
        _write_history(conn, dataset_id, fam, None, before, changed, user_id)


def target_order(preferred: list[str], targets: list[str]) -> list[str]:
    return [t for t in preferred if t in targets] + [t for t in targets if t not in preferred]


def _decision_key(row: dict | None) -> dict | None:
    return None if row is None else {k: v for k, v in row.items() if k != "updated_at"}


def has_screen_fields(conn, dataset_id: int) -> bool:
    """Whether any non-target field needs the uncapped leak screen against a newly chosen target."""
    return conn.execute(
        """SELECT EXISTS (SELECT 1 FROM meta.fields WHERE dataset_id = %s AND NOT is_target
                            AND (inferred_type = 'text'
                                 OR (inferred_type IN ('categorical', 'binary') AND n_unique > %s))) AS e""",
        (dataset_id, get_settings().corr_max_levels),
    ).fetchone()["e"]


def screen_pairs(dataset_id: int, target_ids: list[int]) -> list[tuple[int, int, Assoc]]:
    """Uncapped leak-screen associations between `target_ids` and every field needing the screen."""
    s = get_settings()
    ds = fetch_one("SELECT parquet_path, row_count FROM meta.datasets WHERE id = %s", (dataset_id,))
    fields = fetch_all(
        "SELECT id, name, inferred_type, n_unique, is_target FROM meta.fields WHERE dataset_id = %s ORDER BY ordinal",
        (dataset_id,),
    )
    path = Path(ds["parquet_path"])
    corr_idx = sample_indices(ds["row_count"], s.corr_sample_rows, seed=29)

    def codes(f: dict):
        return f["id"], screen_codes(pl.read_parquet(path, columns=[f["name"]]).to_series(), corr_idx)

    targets = [codes(f) for f in fields if f["id"] in target_ids]
    others = [codes(f) for f in fields if not f["is_target"]
              and needs_screen(f["inferred_type"], f["n_unique"] or 0, False, s.corr_max_levels)]
    return [(a.i, a.j, a) for a in screen_associations(targets, others)]


def _experiment_target_changed(conn, src: Source, f: dict, became: bool) -> bool:
    """Keep an experiment's target order in step with its copy; whether the new target needs the leak screen."""
    x = conn.execute("SELECT id, target_order FROM meta.experiments WHERE id = %s", (src.experiment["id"],)).fetchone()
    order = [n for n in (x.get("target_order") or []) if n != f["name"]] + ([f["name"]] if became else [])
    conn.execute("UPDATE meta.experiments SET target_order = %s WHERE id = %s", (order, x["id"]))
    return became and has_screen_fields(conn, f["dataset_id"])


def target_associations_job(dataset_id: int, target_ids: list[int]) -> None:
    """Background job: the leak screen for fields an experiment made targets. Associations are shared, so this only
    adds them; the baseline's own target state is left alone."""
    try:
        pairs = screen_pairs(dataset_id, target_ids)
        with connection() as conn:
            upsert_correlations(conn, dataset_id, pairs)
    except Exception:  # noqa: BLE001
        log.exception("leak screen for experiment targets of dataset %s failed", dataset_id)


def screen_targets(dataset_id: int, target_ids: list[int]) -> None:
    """Background job: screen newly chosen targets against text and high-cardinality fields, then re-derive
    leakage flags. The dataset stays usable meanwhile; a failure only leaves a warning."""
    try:
        pairs = screen_pairs(dataset_id, target_ids)
        with connection() as conn:
            upsert_correlations(conn, dataset_id, pairs)
            _sync_targets(conn, dataset_id)
    except Exception as exc:  # noqa: BLE001
        log.exception("leak screen for dataset %s failed", dataset_id)
        execute("UPDATE meta.datasets SET warnings = array_append(warnings, %s) WHERE id = %s",
                (f"The leak screen for the new target(s) could not run: {exc}", dataset_id))


def set_targets(conn, dataset_id: int, names: list[str], user_id: int) -> tuple[list[dict], list[int]]:
    """Make exactly `names` the targets (in that order). Returns the target fields whose associations were
    never computed because their profiled type has no association sample (text, identifier), and the ids of
    fields that just became targets."""
    fam = conn.execute("SELECT model_family FROM meta.datasets WHERE id = %s", (dataset_id,)).fetchone()["model_family"]
    fields = conn.execute(
        f"""SELECT f.id, f.name, f.inferred_type, d.action, d.semantic_type, s.family, s.source AS settings_source,
                   {", ".join("f." + c for c in SUMMARY_COLS)}
              FROM meta.fields f
              JOIN meta.field_decisions d ON d.field_id = f.id
              LEFT JOIN meta.field_settings s ON s.field_id = f.id
             WHERE f.dataset_id = %s ORDER BY f.ordinal
               FOR UPDATE OF d""",
        (dataset_id,),
    ).fetchall()
    by_name: dict[str, dict] = {}
    suggested_families: dict[int, list[str]] = {}
    for r in fields:
        by_name.setdefault(r["name"], r)
        if r["settings_source"] == "suggested":
            suggested_families.setdefault(r["id"], []).append(r["family"])
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise HTTPException(422, {"message": "Unknown column(s)", "errors": unknown})
    ineligible = [f"{n}: {p}" for n in names
                  if (p := target_problem(by_name[n]["semantic_type"], by_name[n]["n_unique"]))]
    if ineligible:
        raise HTTPException(422, {"message": "Not a valid target", "errors": ineligible})
    before = _merged_many(conn, [r["id"] for r in by_name.values()], fam)
    added = [by_name[n]["id"] for n in names if by_name[n]["action"] != "target"]
    removed = [r for n, r in by_name.items() if r["action"] == "target" and n not in names]
    conn.execute(
        "UPDATE meta.field_decisions SET action = 'target', updated_at = now() WHERE field_id = ANY(%s)",
        (added,),
    )
    for r in removed:
        # The action itself is re-derived from the field's (now non-target) suggestion by refresh_target_state;
        # it was reviewed as a target, so as a feature it is unreviewed.
        conn.execute(
            "UPDATE meta.field_decisions SET action = 'keep', source = 'suggested', reviewed = false, "
            "updated_at = now() WHERE field_id = %s",
            (r["id"],),
        )
        stats = {c: r[c] for c in SUMMARY_COLS}
        for family in suggested_families.get(r["id"], []):
            upsert_settings(conn, dataset_id, r["id"], family,
                            suggest(r["semantic_type"], stats, False, family), "suggested")
    _sync_targets(conn, dataset_id, before=before, order=names, user_id=user_id)
    candidates = [by_name[n] for n in names if by_name[n]["inferred_type"] in UNCORRELATED_TYPES]
    if not candidates:
        return [], added
    ids = [r["id"] for r in candidates]
    has_corr = {r["fid"] for r in conn.execute(
        """SELECT field_a AS fid FROM meta.field_correlations WHERE field_a = ANY(%(ids)s)
           UNION SELECT field_b FROM meta.field_correlations WHERE field_b = ANY(%(ids)s)""",
        {"ids": ids},
    ).fetchall()}
    return [r for r in candidates if r["id"] not in has_corr], added


def _jsonable(row: dict | None) -> dict | None:
    if row is None:
        return None
    return {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in row.items()}

