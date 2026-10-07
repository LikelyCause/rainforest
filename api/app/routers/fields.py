from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..db import connection, fetch_all, fetch_one, jsonb
from ..processing import upsert_settings
from ..schemas import BulkDecisionIn, DecisionIn
from ..semantics import FAMILIES, settings_of, suggest, validate_decision

router = APIRouter(prefix="/api", tags=["fields"])

SHARED_COLS = ["action", "semantic_type", "notes", "reviewed"]
STAT_KEYS = ["null_pct", "skewness", "outlier_pct", "min_value", "n_unique"]


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


def _merged(conn, field_id: int, family: str) -> dict:
    """Shared decision + this family's settings, as one flat object."""
    return conn.execute(
        """SELECT d.field_id, d.dataset_id, d.action, d.semantic_type, d.notes, d.reviewed, d.source,
                  d.updated_at, s.family, s.imputation, s.impute_value, s.add_missing_indicator, s.transform,
                  s.scaling, s.encoding, s.params, s.source AS settings_source
             FROM meta.field_decisions d
             JOIN meta.field_settings s ON s.field_id = d.field_id AND s.family = %s
            WHERE d.field_id = %s""",
        (family, field_id),
    ).fetchone()


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
        "SELECT changed_at, family, previous, current FROM meta.decision_history WHERE field_id = %s "
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
    with connection() as conn:
        n = 0
        if body.accept_suggestion:
            rows = conn.execute(
                "SELECT id, suggestion FROM meta.fields WHERE dataset_id = %s AND id = ANY(%s)",
                (dataset_id, body.field_ids),
            ).fetchall()
            for r in rows:
                by_family: dict = r["suggestion"]
                first = next(iter(by_family.values()))
                conn.execute(
                    """UPDATE meta.field_decisions SET action = %s, semantic_type = %s, reviewed = true,
                              source = 'user', updated_at = now() WHERE field_id = %s""",
                    (first["action"], first["semantic_type"], r["id"]),
                )
                for fam, sugg in by_family.items():
                    upsert_settings(conn, dataset_id, r["id"], fam, sugg, "suggested")
                n += 1
        if body.action is not None or body.reviewed is not None:
            cur = conn.execute(
                """UPDATE meta.field_decisions
                      SET action = CASE WHEN action = 'target' THEN action ELSE coalesce(%s::text, action) END,
                          reviewed = coalesce(%s::boolean, reviewed), source = 'user', updated_at = now()
                    WHERE dataset_id = %s AND field_id = ANY(%s)
                      AND (%s::text IS NULL OR %s::text = 'drop'
                           OR semantic_type NOT IN ('identifier', 'constant'))""",
                (body.action, body.reviewed, dataset_id, body.field_ids, body.action, body.action),
            )
            n = max(n, cur.rowcount)
        _sync_targets(conn, dataset_id)
    return {"updated": n}


def _sync_targets(conn, dataset_id: int) -> None:
    """Keep fields.is_target / datasets.target_columns / max_target_assoc consistent with decisions."""
    conn.execute(
        """UPDATE meta.fields f SET is_target = (d.action = 'target')
             FROM meta.field_decisions d WHERE d.field_id = f.id AND f.dataset_id = %s""",
        (dataset_id,),
    )
    conn.execute(
        """UPDATE meta.datasets SET target_columns = coalesce(
               (SELECT array_agg(name ORDER BY ordinal) FROM meta.fields WHERE dataset_id = %s AND is_target),
               '{}'), updated_at = now()
            WHERE id = %s""",
        (dataset_id, dataset_id),
    )
    conn.execute(
        """UPDATE meta.fields f SET max_target_assoc = s.m
             FROM (SELECT f2.id, (SELECT max(c.strength) FROM meta.v_correlations c
                                   WHERE c.field_id = f2.id AND c.other_is_target) AS m
                     FROM meta.fields f2 WHERE f2.dataset_id = %s) s
            WHERE f.id = s.id""",
        (dataset_id,),
    )


def _jsonable(row: dict | None) -> dict | None:
    if row is None:
        return None
    return {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in row.items()}

