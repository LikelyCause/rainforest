from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Query
from psycopg import sql

from ..db import connection, fetch_all, fetch_one, jsonb
from ..processing import SUMMARY_COLS, refresh_target_state, upsert_settings
from ..schemas import BulkDecisionIn, DecisionIn
from ..semantics import FAMILIES, NON_FEATURE_TYPES, SETTING_KEYS, settings_of, suggest, validate_decision
from .query import fields_where

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


def _family(f: dict, family: str | None) -> str:
    fam = family or f["active_family"]
    if fam not in FAMILIES:
        raise HTTPException(422, f"Unknown model family {fam!r}")
    return fam


_MERGED_SQL = """SELECT d.field_id, d.dataset_id, d.action, d.semantic_type, d.notes, d.reviewed, d.source,
                  d.updated_at, s.family, s.imputation, s.impute_value, s.add_missing_indicator, s.transform,
                  s.scaling, s.encoding, s.params, s.source AS settings_source
             FROM meta.field_decisions d
             JOIN meta.field_settings s ON s.field_id = d.field_id AND s.family = %s"""


def _merged(conn, field_id: int, family: str) -> dict:
    """Shared decision + this family's settings, as one flat object."""
    return conn.execute(_MERGED_SQL + " WHERE d.field_id = %s", (family, field_id)).fetchone()


def _merged_many(conn, field_ids: list[int], family: str) -> dict[int, dict]:
    rows = conn.execute(_MERGED_SQL + " WHERE d.field_id = ANY(%s)", (family, field_ids)).fetchall()
    return {r["field_id"]: r for r in rows}


@router.get("/fields/{field_id}")
def get_field(field_id: int, family: str | None = None) -> dict:
    f = _field(field_id)
    fam = _family(f, family)
    with connection() as conn:
        decision = _merged(conn, field_id, fam)
    targets = fetch_all(
        """SELECT other_field_id AS field_id, other_field_name AS name, method, value, strength
             FROM meta.v_correlations WHERE field_id = %s AND other_is_target ORDER BY other_field_name""",
        (field_id,),
    )
    suggestions = f.pop("suggestion") or {}
    f["suggestion"] = suggestions.get(fam, {})
    return {"field": f, "family": fam, "decision": decision, "target_associations": targets}


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
        "SELECT changed_at, family, previous, current, batch_id FROM meta.decision_history WHERE field_id = %s "
        "ORDER BY changed_at DESC LIMIT 50",
        (field_id,),
    )


@router.put("/fields/{field_id}/decision")
def save_decision(field_id: int, body: DecisionIn) -> dict:
    f = _field(field_id)
    fam = _family(f, body.family)
    d = body.model_dump(exclude={"family"})
    errors = validate_decision(d, fam)
    if errors:
        raise HTTPException(422, {"message": "Invalid decision", "errors": errors})
    with connection() as conn:
        conn.execute("SELECT 1 FROM meta.field_decisions WHERE field_id = %s FOR UPDATE", (field_id,))
        prev = _merged(conn, field_id, fam)
        conn.execute(
            f"""UPDATE meta.field_decisions
                   SET {", ".join(f"{c} = %({c})s" for c in SHARED_COLS)}, source = 'user', updated_at = now()
                 WHERE field_id = %(field_id)s""",
            {**d, "field_id": field_id},
        )
        upsert_settings(conn, f["dataset_id"], field_id, fam, settings_of(d), "user")
        if prev["semantic_type"] != d["semantic_type"]:
            # Other families' settings were chosen for the old type; re-derive them for the new one.
            stats = {k: f[k] for k in STAT_KEYS}
            for other in FAMILIES:
                if other != fam:
                    upsert_settings(conn, f["dataset_id"], field_id, other,
                                    suggest(d["semantic_type"], stats, False, other), "suggested")
        row = _merged(conn, field_id, fam)
        conn.execute(
            """INSERT INTO meta.decision_history (field_id, dataset_id, family, previous, current)
               VALUES (%s, %s, %s, %s, %s)""",
            (field_id, f["dataset_id"], fam, jsonb(_jsonable(prev)), jsonb(_jsonable(row))),
        )
        if (prev["action"] == "target") != (d["action"] == "target"):
            _sync_targets(conn, f["dataset_id"])
    return row


@router.post("/datasets/{dataset_id}/decisions/bulk")
def bulk_decisions(dataset_id: int, body: BulkDecisionIn) -> dict:
    fam = _active_family(dataset_id)
    with connection() as conn:
        if body.filters is not None:
            target = sql.SQL("d.field_id IN (SELECT field_id FROM meta.v_fields WHERE {})").format(
                fields_where(dataset_id, body.filters))
        else:
            target = sql.SQL("d.field_id = ANY({})").format(sql.Literal(body.field_ids))
        # Resolve and lock in one statement so the set can't drift between preview and write.
        rows = conn.execute(
            sql.SQL("""SELECT d.field_id, f.name, d.action, d.semantic_type, d.reviewed, f.suggestion
                         FROM meta.field_decisions d JOIN meta.fields f ON f.id = d.field_id
                        WHERE d.dataset_id = {} AND {} AND NOT d.field_id = ANY({})
                        ORDER BY f.ordinal FOR UPDATE OF d""").format(
                sql.Literal(dataset_id), target, sql.Literal(body.excluded_ids)),
        ).fetchall()
        ids = [r["field_id"] for r in rows]
        settings: dict[int, dict[str, dict]] = {}
        for srow in conn.execute("SELECT * FROM meta.field_settings WHERE field_id = ANY(%s)", (ids,)).fetchall():
            settings.setdefault(srow["field_id"], {})[srow["family"]] = srow
        planned = plan_bulk(rows, settings, body)
        result = summarize_bulk(len(rows), planned)
        if body.dry_run or not planned:
            conn.rollback()
            return {"batch_id": None, "dry_run": body.dry_run, **result}

        batch_id = uuid.uuid4()
        changed_ids = [c["field_id"] for c in planned]
        snapshot = _snapshot(conn, changed_ids)
        prev = _merged_many(conn, changed_ids, fam)
        for c in planned:
            conn.execute(
                """UPDATE meta.field_decisions SET action = %s, semantic_type = %s, reviewed = %s,
                          source = 'user', updated_at = now() WHERE field_id = %s""",
                (c["after"]["action"], c["after"]["semantic_type"], c["after"]["reviewed"], c["field_id"]),
            )
            for family, sugg in c["settings"].items():
                upsert_settings(conn, dataset_id, c["field_id"], family, sugg, "suggested")
        _write_history(conn, dataset_id, fam, batch_id, prev, _merged_many(conn, changed_ids, fam))
        conn.execute(
            "INSERT INTO meta.bulk_batches (id, dataset_id, summary, snapshot) VALUES (%s, %s, %s, %s)",
            (batch_id, dataset_id,
             jsonb({**{k: v for k, v in result.items() if k != "changes"},
                    "request": body.model_dump(exclude={"dry_run"})}),
             jsonb(snapshot)),
        )
        _sync_targets(conn, dataset_id)
    return {"batch_id": str(batch_id), "dry_run": False, **result}


@router.post("/datasets/{dataset_id}/decisions/bulk/{batch_id}/undo")
def undo_bulk(dataset_id: int, batch_id: str) -> dict:
    try:
        bid = uuid.UUID(batch_id)
    except ValueError:
        raise HTTPException(404, "Bulk batch not found") from None
    fam = _active_family(dataset_id)
    with connection() as conn:
        batch = conn.execute(
            "SELECT snapshot, undone_at FROM meta.bulk_batches WHERE id = %s AND dataset_id = %s FOR UPDATE",
            (bid, dataset_id),
        ).fetchone()
        if batch is None:
            raise HTTPException(404, "Bulk batch not found")
        if batch["undone_at"] is not None:
            raise HTTPException(409, "This bulk change was already undone")
        snapshot: dict = batch["snapshot"]
        # Fields deleted since (dataset reprocessed) can't be restored; skip them.
        ids = [r["field_id"] for r in conn.execute(
            """SELECT field_id FROM meta.field_decisions WHERE dataset_id = %s AND field_id = ANY(%s)
                ORDER BY field_id FOR UPDATE""",
            (dataset_id, [int(k) for k in snapshot]),
        ).fetchall()]
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
        _write_history(conn, dataset_id, fam, undo_id, prev, _merged_many(conn, ids, fam))
        conn.execute("UPDATE meta.bulk_batches SET undone_at = now() WHERE id = %s", (bid,))
        _sync_targets(conn, dataset_id)
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
            new_settings = {f: sg for f, sg in by_family.items() if not _same_settings(current.get(f), sg)}
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


def _snapshot(conn, field_ids: list[int]) -> dict:
    rows = conn.execute(
        """SELECT d.field_id, to_jsonb(d) AS decision,
                  coalesce((SELECT jsonb_agg(to_jsonb(s) ORDER BY s.family) FROM meta.field_settings s
                             WHERE s.field_id = d.field_id), '[]') AS settings
             FROM meta.field_decisions d WHERE d.field_id = ANY(%s)""",
        (field_ids,),
    ).fetchall()
    return {str(r["field_id"]): {"decision": r["decision"], "settings": r["settings"]} for r in rows}


def _write_history(conn, dataset_id: int, family: str, batch_id: uuid.UUID | None, prev: dict, cur: dict) -> None:
    with conn.cursor() as c:
        c.executemany(
            """INSERT INTO meta.decision_history (field_id, dataset_id, family, previous, current, batch_id)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            [(fid, dataset_id, family, jsonb(_jsonable(prev.get(fid))), jsonb(_jsonable(row)), batch_id)
             for fid, row in cur.items()],
        )


def _sync_targets(conn, dataset_id: int, before: dict[int, dict] | None = None,
                  order: list[str] | None = None) -> None:
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
        _write_history(conn, dataset_id, fam, None, before, changed)


def target_order(preferred: list[str], targets: list[str]) -> list[str]:
    return [t for t in preferred if t in targets] + [t for t in targets if t not in preferred]


def _decision_key(row: dict | None) -> dict | None:
    return None if row is None else {k: v for k, v in row.items() if k != "updated_at"}


def set_targets(conn, dataset_id: int, names: list[str]) -> list[dict]:
    """Make exactly `names` the targets (in that order). Returns the target fields whose associations were
    never computed because their profiled type has no association sample (text, identifier)."""
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
    before = _merged_many(conn, [r["id"] for r in by_name.values()], fam)
    added = [by_name[n]["id"] for n in names if by_name[n]["action"] != "target"]
    removed = [r for n, r in by_name.items() if r["action"] == "target" and n not in names]
    conn.execute(
        "UPDATE meta.field_decisions SET action = 'target', updated_at = now() WHERE field_id = ANY(%s)",
        (added,),
    )
    for r in removed:
        # The action itself is re-derived from the field's (now non-target) suggestion by refresh_target_state.
        conn.execute(
            "UPDATE meta.field_decisions SET action = 'keep', source = 'suggested', updated_at = now() "
            "WHERE field_id = %s",
            (r["id"],),
        )
        stats = {c: r[c] for c in SUMMARY_COLS}
        for family in suggested_families.get(r["id"], []):
            upsert_settings(conn, dataset_id, r["id"], family,
                            suggest(r["semantic_type"], stats, False, family), "suggested")
    _sync_targets(conn, dataset_id, before=before, order=names)
    candidates = [by_name[n] for n in names if by_name[n]["inferred_type"] in UNCORRELATED_TYPES]
    if not candidates:
        return []
    ids = [r["id"] for r in candidates]
    has_corr = {r["fid"] for r in conn.execute(
        """SELECT field_a AS fid FROM meta.field_correlations WHERE field_a = ANY(%(ids)s)
           UNION SELECT field_b FROM meta.field_correlations WHERE field_b = ANY(%(ids)s)""",
        {"ids": ids},
    ).fetchall()}
    return [r for r in candidates if r["id"] not in has_corr]


def _jsonable(row: dict | None) -> dict | None:
    if row is None:
        return None
    return {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in row.items()}

