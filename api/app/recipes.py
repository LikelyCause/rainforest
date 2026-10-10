"""Recipes: the versioned, finalizable set of field decisions of a dataset.

A dataset has one row per version in meta.recipes and at most one version that isn't finalized. The version being
worked moves `assessing → review → finalized`; finalizing freezes a snapshot of the team decision.
"""
from __future__ import annotations

import hashlib

from fastapi import HTTPException

from .db import jsonb
from .semantics import SETTING_KEYS

PHASES = ("assessing", "review", "finalized")
RECIPE_COLS = """id, dataset_id, version, phase, parent_id, created_by, created_at, finalized_by, finalized_at,
                 schema_fingerprint"""
# Joins a dataset row `d` to its current (latest) version as `rc`.
CURRENT_JOIN = """LEFT JOIN LATERAL (SELECT id, version, phase FROM meta.recipes
                                       WHERE dataset_id = d.id ORDER BY version DESC LIMIT 1) rc ON true"""


def create_first(conn, dataset_id: int, user_id: int) -> dict:
    """Version 1 of a new dataset, open for assessment."""
    return conn.execute(
        f"""INSERT INTO meta.recipes (dataset_id, version, phase, created_by)
            VALUES (%s, 1, 'assessing', %s) RETURNING {RECIPE_COLS}""",
        (dataset_id, user_id),
    ).fetchone()


def start_version(conn, dataset_id: int, user_id: int | None, phase: str = "assessing") -> dict:
    """The next version after a finalized one, with the finalized one as its parent."""
    parent = current(conn, dataset_id, lock="update")
    if parent["phase"] != "finalized":
        raise HTTPException(409, f"{label(parent)} isn't finalized yet")
    return conn.execute(
        f"""INSERT INTO meta.recipes (dataset_id, version, phase, parent_id, created_by)
            VALUES (%s, %s, %s, %s, %s) RETURNING {RECIPE_COLS}""",
        (dataset_id, parent["version"] + 1, phase, parent["id"], user_id),
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


def snapshot(conn, dataset_id: int) -> dict:
    """The team decision as a recipe stores it: per field name, the decision, settings for every model family, the
    inferred type and the stats a later upload is checked against for drift; plus the target order."""
    fields = conn.execute(
        """SELECT f.id, f.name, f.inferred_type, f.null_pct, f.n_unique, d.action, d.semantic_type, d.notes
             FROM meta.fields f JOIN meta.field_decisions d ON d.field_id = f.id
            WHERE f.dataset_id = %s ORDER BY f.ordinal""",
        (dataset_id,),
    ).fetchall()
    settings: dict[int, dict[str, dict]] = {}
    for s in conn.execute(
        "SELECT * FROM meta.field_settings WHERE dataset_id = %s ORDER BY field_id, family", (dataset_id,)
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
