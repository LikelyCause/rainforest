"""Recipes: the versioned, finalizable set of field decisions of a dataset.

A dataset has one row per version in meta.recipes and at most one version that isn't finalized. The version being
worked moves `assessing → review → finalized`; finalizing freezes a snapshot of the team decision.
"""
from __future__ import annotations

import hashlib

from fastapi import HTTPException
from psycopg import sql

from .db import jsonb
from .semantics import SETTING_KEYS

PHASES = ("assessing", "review", "finalized")
RECIPE_COLS = """id, dataset_id, version, phase, scope, parent_id, created_by, created_at, finalized_by, finalized_at,
                 schema_fingerprint"""
# Joins a dataset row `d` to its current (latest) version as `rc`.
CURRENT_JOIN = """LEFT JOIN LATERAL (SELECT id, version, phase, scope FROM meta.recipes
                                       WHERE dataset_id = d.id ORDER BY version DESC LIMIT 1) rc ON true"""


def create_first(conn, dataset_id: int, user_id: int) -> dict:
    """Version 1 of a new dataset, open for assessment."""
    return conn.execute(
        f"""INSERT INTO meta.recipes (dataset_id, version, phase, created_by)
            VALUES (%s, 1, 'assessing', %s) RETURNING {RECIPE_COLS}""",
        (dataset_id, user_id),
    ).fetchone()


def start_version(conn, dataset_id: int, user_id: int | None, phase: str = "assessing", scope: str = "all") -> dict:
    """The next version after a finalized one, with the finalized one as its parent."""
    parent = current(conn, dataset_id, lock="update")
    if parent["phase"] != "finalized":
        raise HTTPException(409, f"{label(parent)} isn't finalized yet")
    return conn.execute(
        f"""INSERT INTO meta.recipes (dataset_id, version, phase, scope, parent_id, created_by)
            VALUES (%s, %s, %s, %s, %s, %s) RETURNING {RECIPE_COLS}""",
        (dataset_id, parent["version"] + 1, phase, scope, parent["id"], user_id),
    ).fetchone()


def current(conn, dataset_id: int, lock: str | None = None) -> dict:
    """The dataset's latest version. `lock` ("update" or "share") locks the row: phase changes take "update",
    decision writes "share", so a write never lands in a phase it wasn't checked against."""
    row = conn.execute(
        f"""SELECT {RECIPE_COLS} FROM meta.recipes WHERE dataset_id = %s
             ORDER BY version DESC LIMIT 1{f' FOR {lock.upper()}' if lock else ''}""",
        (dataset_id,),
    ).fetchone()
    if row is None:
        if conn.execute("SELECT 1 FROM meta.datasets WHERE id = %s", (dataset_id,)).fetchone() is None:
            raise HTTPException(404, "Dataset not found")
        raise HTTPException(409, "This dataset has no recipe")
    return row


def get(conn, recipe_id: int) -> dict | None:
    return conn.execute(f"SELECT {RECIPE_COLS} FROM meta.recipes WHERE id = %s", (recipe_id,)).fetchone()


def latest_finalized(conn, dataset_id: int) -> dict | None:
    return conn.execute(
        f"""SELECT {RECIPE_COLS} FROM meta.recipes WHERE dataset_id = %s AND phase = 'finalized'
             ORDER BY version DESC LIMIT 1""",
        (dataset_id,),
    ).fetchone()


def open_field_ids(conn, recipe: dict) -> list[int] | None:
    """The fields this version works on: None for every field, else the reopened ones (the rest were carried over
    and are locked)."""
    if recipe["scope"] == "all":
        return None
    return [r["field_id"] for r in conn.execute(
        "SELECT field_id FROM meta.recipe_reopened WHERE recipe_id = %s ORDER BY field_id", (recipe["id"],)
    ).fetchall()]


def locked_problem(conn, recipe: dict, field_ids: list[int]) -> str | None:
    """Why some of `field_ids` can't change in this version (they were carried over), or None."""
    open_ = open_field_ids(conn, recipe)
    if open_ is None or recipe["phase"] == "finalized":
        return None
    locked = conn.execute(
        "SELECT name FROM meta.fields WHERE id = ANY(%s) AND NOT id = ANY(%s) ORDER BY ordinal", (field_ids, open_)
    ).fetchall()
    if not locked:
        return None
    one = len(locked) == 1
    names = ", ".join(r["name"] for r in locked[:8]) + ("…" if len(locked) > 8 else "")
    return (f"{names}: carried over into {label(recipe).lower()} and locked. To change "
            f"{'it' if one else 'them'}, start a new version that reopens {'it' if one else 'them'}.")


def snapshot_of(conn, recipe: dict) -> dict:
    """A finalized version's stored snapshot; for the version being worked, the team decision as it stands."""
    if recipe["phase"] == "finalized":
        row = conn.execute("SELECT snapshot FROM meta.recipes WHERE id = %s", (recipe["id"],)).fetchone()
        if row and row["snapshot"] is not None:
            return row["snapshot"]
    return snapshot(conn, recipe["dataset_id"])


def label(recipe: dict) -> str:
    return f"Recipe v{recipe['version']}"


def fingerprint(columns: list[tuple[str, str]]) -> str:
    """Identifies a dataset's columns (name and physical type, in order), for matching a recipe to new data."""
    return hashlib.md5("\n".join(f"{n}\t{t}" for n, t in columns).encode()).hexdigest()


def schema_fingerprint(conn, dataset_id: int) -> str:
    rows = conn.execute(
        "SELECT name, physical_dtype FROM meta.fields WHERE dataset_id = %s ORDER BY ordinal", (dataset_id,)
    ).fetchall()
    return fingerprint([(r["name"], r["physical_dtype"]) for r in rows])


def snapshot(conn, dataset_id: int, src=None) -> dict:
    """The team decision as a recipe stores it: per field name, the decision, settings for every model family, the
    inferred type and the stats a later upload is checked against for drift; plus the target order. With `src` (an
    assessment.Source), the same from that source's decisions instead."""
    dec = src.decisions() if src is not None else sql.SQL("meta.field_decisions")
    sets = src.settings() if src is not None else sql.SQL("meta.field_settings")
    fields = conn.execute(
        sql.SQL("""SELECT f.id, f.name, f.inferred_type, f.null_pct, f.n_unique, d.action, d.semantic_type, d.notes
                     FROM meta.fields f JOIN {} d ON d.field_id = f.id
                    WHERE f.dataset_id = %s ORDER BY f.ordinal""").format(dec),
        (dataset_id,),
    ).fetchall()
    settings: dict[int, dict[str, dict]] = {}
    for s in conn.execute(
        sql.SQL("SELECT * FROM {} s WHERE s.dataset_id = %s ORDER BY field_id, family").format(sets), (dataset_id,)
    ).fetchall():
        settings.setdefault(s["field_id"], {})[s["family"]] = {k: s[k] for k in SETTING_KEYS}
    ds = conn.execute("SELECT target_columns FROM meta.datasets WHERE id = %s", (dataset_id,)).fetchone()
    return {
        "fields": {
            f["name"]: {
                "decision": {k: f[k] for k in ("action", "semantic_type", "notes")},
                "settings": settings.get(f["id"], {}),
                "inferred_type": f["inferred_type"],
                "null_pct": f["null_pct"],
                "n_unique": f["n_unique"],
            }
            for f in fields
        },
        "target_order": list(ds["target_columns"] or []),
    }


def finalize(conn, recipe: dict, user_id: int) -> dict:
    return conn.execute(
        f"""UPDATE meta.recipes SET phase = 'finalized', finalized_by = %s, finalized_at = now(), snapshot = %s,
                   schema_fingerprint = %s
             WHERE id = %s RETURNING {RECIPE_COLS}""",
        (user_id, jsonb(snapshot(conn, recipe["dataset_id"])), schema_fingerprint(conn, recipe["dataset_id"]),
         recipe["id"]),
    ).fetchone()


def backfill_snapshots(conn) -> int:
    """Snapshot finalized versions that have none (datasets that existed before recipes)."""
    rows = conn.execute(
        "SELECT id, dataset_id FROM meta.recipes WHERE phase = 'finalized' AND snapshot IS NULL"
    ).fetchall()
    for r in rows:
        conn.execute(
            "UPDATE meta.recipes SET snapshot = %s, schema_fingerprint = %s WHERE id = %s",
            (jsonb(snapshot(conn, r["dataset_id"])), schema_fingerprint(conn, r["dataset_id"]), r["id"]),
        )
    return len(rows)


def snapshot_targets(snap: dict) -> list[str]:
    """The snapshot's targets in chain order."""
    targets = [n for n, f in snap["fields"].items() if f["decision"]["action"] == "target"]
    order = [t for t in snap.get("target_order") or [] if t in targets]
    return order + [t for t in targets if t not in order]


def spec_rows(conn, dataset_id: int, snap: dict, family: str) -> list[dict]:
    """Each field of the dataset the snapshot covers, in column order: its decision and `family`'s settings from the
    snapshot, plus what the data says (inferred type, leakage flag, suggestion, stored date format). Exports and their
    preview read this, never the live team decision, so a version always exports the same way."""
    out = []
    for f in conn.execute(
        """SELECT name, inferred_type, leakage_flag AS leakage, suggestion, n_unique,
                  stats ->> 'datetime_format' AS datetime_format,
                  coalesce((stats ->> 'decimal_comma')::boolean, false) AS decimal_comma
             FROM meta.fields WHERE dataset_id = %s ORDER BY ordinal""",
        (dataset_id,),
    ).fetchall():
        s = snap["fields"].get(f["name"])
        if s is None:
            continue
        settings = s["settings"].get(family) or (f["suggestion"] or {}).get(family) or {}
        out.append({**f, "action": s["decision"]["action"], "semantic_type": s["decision"]["semantic_type"],
                    **{k: settings.get(k) for k in SETTING_KEYS}})
    return out


def _norm(key: str, v):
    if key == "impute_value":
        return None if v is None else str(v)
    if key == "add_missing_indicator":
        return bool(v)
    if key == "params":
        return v or {}
    return v


def diff(before: dict, after: dict, family: str | None = None) -> dict:
    """What changed between two snapshots: per field, each part of the decision and each family's setting that
    differs (only `family`'s when given), fields added or removed, and the targets."""
    b_fields, a_fields = before["fields"], after["fields"]
    out = []
    for name in [*a_fields, *(n for n in b_fields if n not in a_fields)]:
        b, a = b_fields.get(name), a_fields.get(name)
        if b is None or a is None:
            out.append({"name": name, "change": "added" if b is None else "removed", "changes": []})
            continue
        changes = [{"key": k, "family": None, "before": b["decision"].get(k), "after": a["decision"].get(k)}
                   for k in ("action", "semantic_type", "notes") if b["decision"].get(k) != a["decision"].get(k)]
        for fam in [family] if family else sorted(set(b["settings"]) | set(a["settings"])):
            bs, as_ = b["settings"].get(fam) or {}, a["settings"].get(fam) or {}
            changes += [{"key": k, "family": fam, "before": bs.get(k), "after": as_.get(k)}
                        for k in SETTING_KEYS if _norm(k, bs.get(k)) != _norm(k, as_.get(k))]
        if changes:
            out.append({"name": name, "change": "changed", "changes": changes})
    tb, ta = snapshot_targets(before), snapshot_targets(after)
    return {"fields": out, "targets": None if tb == ta else {"before": tb, "after": ta}}
