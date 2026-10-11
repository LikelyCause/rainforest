"""Baselines: the versioned, finalizable field decisions of a dataset for one model family (a row of meta.recipes).

A dataset has up to one baseline line per family, each with its own versions, and at most one version in progress
(`assessing` or `review`) across all of them. The working tables (meta.field_decisions / meta.field_settings) hold
the version in progress: starting any baseline or version first loads them from a snapshot (`load_working`), so they
never carry another baseline's choices. Finalizing freezes a snapshot of the team decision for the baseline's family.
"""
from __future__ import annotations

import hashlib

from fastapi import HTTPException
from psycopg import sql

from .db import jsonb
from .semantics import SETTING_KEYS, suggest, validate_decision

DECISION_KEYS = ("action", "semantic_type", "notes", "target_task", "target_bins")
FAMILY_LABEL = {"tree": "Trees", "linear": "Linear", "neural": "Neural"}
NAME_MAX = 60
RECIPE_COLS = """id, dataset_id, family, version, phase, scope, parent_id, created_by, created_at, finalized_by,
                 finalized_at, schema_fingerprint, meta.baseline_name(dataset_id, family) AS name"""
# Joins a dataset row `d` to its current (latest started) baseline version as `rc`.
CURRENT_JOIN = """LEFT JOIN LATERAL (SELECT id, family, version, phase, scope,
                                            meta.baseline_name(dataset_id, family) AS name FROM meta.recipes
                                       WHERE dataset_id = d.id ORDER BY id DESC LIMIT 1) rc ON true"""


def create_first(conn, dataset_id: int, user_id: int, family: str = "tree") -> dict:
    """Version 1 of a new dataset's first baseline, open for assessment and not set up yet (`is_setup`)."""
    return conn.execute(
        f"""INSERT INTO meta.recipes (dataset_id, family, version, phase, created_by)
            VALUES (%s, %s, 1, 'assessing', %s) RETURNING {RECIPE_COLS}""",
        (dataset_id, family, user_id),
    ).fetchone()


def next_version(conn, dataset_id: int, family: str) -> int:
    return conn.execute(
        "SELECT coalesce(max(version), 0) + 1 AS v FROM meta.recipes WHERE dataset_id = %s AND family = %s",
        (dataset_id, family),
    ).fetchone()["v"]


def require_none_in_progress(cur: dict) -> None:
    if cur["phase"] != "finalized":
        raise HTTPException(409, f"{label(cur)} is still {'in team review' if cur['phase'] == 'review' else 'being assessed'}"
                                 "; finalize it before starting another baseline or version.")


def start_version(conn, dataset_id: int, user_id: int | None, family: str, phase: str = "assessing",
                  scope: str = "all") -> dict:
    """The next version of `family`'s baseline, from its latest finalized version, whose snapshot becomes the
    working copy (every field carried over as reviewed). Returns the new row with `_target_work`, the
    (uncorrelated, added) pair from setting its targets."""
    lock_dataset(conn, dataset_id)
    require_none_in_progress(current(conn, dataset_id, lock="update"))
    parent = latest_finalized(conn, dataset_id, family)
    if parent is None:
        raise HTTPException(409, f"There's no finalized {FAMILY_LABEL[family]} baseline to start a new version from.")
    row = conn.execute(
        f"""INSERT INTO meta.recipes (dataset_id, family, version, phase, scope, parent_id, created_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING {RECIPE_COLS}""",
        (dataset_id, family, next_version(conn, dataset_id, family), phase, scope, parent["id"], user_id),
    ).fetchone()
    snap = snapshot_of(conn, parent)
    work = load_working(conn, dataset_id, snap, family, snapshot_targets(snap), user_id, reviewed=True)
    return {**row, "_target_work": work}


def lock_dataset(conn, dataset_id: int) -> None:
    """Serialise starting baselines and versions on one dataset, so two leads starting at once get a clear 409
    rather than racing to insert."""
    conn.execute("SELECT 1 FROM meta.datasets WHERE id = %s FOR UPDATE", (dataset_id,))


def is_setup(conn, recipe: dict) -> bool:
    """A dataset's first baseline before anyone works on it: its family, start and targets can still change."""
    if recipe["version"] != 1 or recipe["phase"] != "assessing" or recipe["parent_id"] is not None:
        return False
    return conn.execute(
        """SELECT reuse IS NULL AND NOT EXISTS (SELECT 1 FROM meta.dataset_assessors WHERE recipe_id = %(r)s)
                  AND NOT EXISTS (SELECT 1 FROM meta.assessments WHERE recipe_id = %(r)s)
                  AND NOT EXISTS (SELECT 1 FROM meta.recipes o WHERE o.dataset_id = %(ds)s AND o.id <> %(r)s) AS ok
             FROM meta.recipes WHERE id = %(r)s""",
        {"r": recipe["id"], "ds": recipe["dataset_id"]},
    ).fetchone()["ok"]


def load_working(conn, dataset_id: int, snap: dict | None, family: str, targets: list[str], user_id: int | None,
                 reviewed: bool = False) -> tuple[list[dict], list[int]]:
    """Make the working tables a baseline's starting point: `snap`'s decisions and settings (its own family's, or
    when populating from another family, the source's where they are valid for `family`), else the suggestions.
    Settings that aren't valid for the decided type take its suggestion. Sets the targets and the dataset's family;
    returns what `fields.set_targets` reports for the caller to queue."""
    from .assessment import settings_key
    from .processing import upsert_settings
    from .routers.fields import STAT_KEYS, set_targets

    for f in conn.execute("SELECT * FROM meta.fields WHERE dataset_id = %s ORDER BY ordinal", (dataset_id,)).fetchall():
        stats = {k: f[k] for k in STAT_KEYS}
        entry = (snap or {}).get("fields", {}).get(f["name"])
        base = (f["suggestion"] or {}).get(family) or suggest(f["inferred_type"], stats, False, family)
        dec = entry["decision"] if entry else {"action": base["action"], "semantic_type": base["semantic_type"],
                                                "notes": None}
        t = dec["semantic_type"]
        sugg = base if base["semantic_type"] == t else suggest(t, stats, False, family)
        action = "keep" if dec["action"] == "target" else dec["action"]  # the target role comes from set_targets
        # A decision from a snapshot is the team's, not a suggestion, so re-deriving suggestions for the new
        # targets never overwrites it.
        as_suggested = entry is None
        conn.execute(
            """UPDATE meta.field_decisions SET action = %s, semantic_type = %s, notes = %s, reviewed = %s, source = %s,
                      updated_at = now() WHERE field_id = %s""",
            (action, t, dec.get("notes"), reviewed, "suggested" if as_suggested else "user", f["id"]),
        )
        settings = (entry or {}).get("settings") or {}
        chosen = settings.get(family) or next(iter(settings.values()), None)
        if chosen is not None and validate_decision({**chosen, "action": "keep", "semantic_type": t}, family,
                                                    n_unique=f["n_unique"]):
            chosen = None
        chosen = {k: (chosen or sugg).get(k) for k in SETTING_KEYS}
        upsert_settings(conn, dataset_id, f["id"], family, chosen,
                        "suggested" if settings_key(chosen) == settings_key(sugg) else "user")
    conn.execute("UPDATE meta.datasets SET model_family = %s, updated_at = now() WHERE id = %s", (family, dataset_id))
    tasks = None if snap is None else {t: {"target_task": v["task"], "target_bins": v["bins"]}
                                       for t, v in snapshot_target_tasks(snap).items()}
    return set_targets(conn, dataset_id, list(targets), user_id, tasks)


def current(conn, dataset_id: int, lock: str | None = None) -> dict:
    """The dataset's latest started version. `lock` ("update" or "share") locks the row: phase changes take
    "update", decision writes "share", so a write never lands in a phase it wasn't checked against."""
    row = conn.execute(
        f"""SELECT {RECIPE_COLS} FROM meta.recipes WHERE dataset_id = %s
             ORDER BY id DESC LIMIT 1{f' FOR {lock.upper()}' if lock else ''}""",
        (dataset_id,),
    ).fetchone()
    if row is None:
        if conn.execute("SELECT 1 FROM meta.datasets WHERE id = %s", (dataset_id,)).fetchone() is None:
            raise HTTPException(404, "Dataset not found")
        raise HTTPException(409, "This dataset has no recipe")
    return row


def get(conn, recipe_id: int) -> dict | None:
    return conn.execute(f"SELECT {RECIPE_COLS} FROM meta.recipes WHERE id = %s", (recipe_id,)).fetchone()


def latest_finalized(conn, dataset_id: int, family: str | None = None) -> dict | None:
    """The most recently finalized version (of `family` when given)."""
    return conn.execute(
        f"""SELECT {RECIPE_COLS} FROM meta.recipes WHERE dataset_id = %s AND phase = 'finalized'
              AND (%s::text IS NULL OR family = %s)
             ORDER BY finalized_at DESC, id DESC LIMIT 1""",
        (dataset_id, family, family),
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
    return (f"{names}: carried over into {label(recipe, lower=True)} and locked. To change "
            f"{'it' if one else 'them'}, start a new version that reopens {'it' if one else 'them'}.")


def snapshot_of(conn, recipe: dict) -> dict:
    """A finalized version's stored snapshot; for the version being worked, the team decision as it stands."""
    if recipe["phase"] == "finalized":
        row = conn.execute("SELECT snapshot FROM meta.recipes WHERE id = %s", (recipe["id"],)).fetchone()
        if row and row["snapshot"] is not None:
            return row["snapshot"]
    return snapshot(conn, recipe["dataset_id"], family=recipe["family"])


def label(recipe: dict, lower: bool = False) -> str:
    """ "Dropout risk v2 (Trees)", or "Trees baseline v2" unnamed ("trees baseline v2" with `lower`, for mid-sentence;
    a name keeps its case). `name` is required, so a query that forgets it fails instead of labelling the baseline by
    its model type."""
    name, fam = recipe["name"], FAMILY_LABEL[recipe["family"]]
    if name:
        return f"{name} v{recipe['version']} ({fam})"
    text = f"{fam} baseline v{recipe['version']}"
    return text.lower() if lower else text


def short_label(family: str, version: int, name: str | None) -> str:
    return f"{name or FAMILY_LABEL[family]} v{version}"


def set_name(conn, dataset_id: int, family: str, name: str | None) -> None:
    if name is None:
        conn.execute("DELETE FROM meta.baseline_names WHERE dataset_id = %s AND family = %s", (dataset_id, family))
        return
    # A model type can lose its versions' family (reuse switches a new dataset's placeholder); its name goes with it.
    conn.execute("""DELETE FROM meta.baseline_names n WHERE n.dataset_id = %s AND NOT EXISTS (
                      SELECT 1 FROM meta.recipes r WHERE r.dataset_id = n.dataset_id AND r.family = n.family)""",
                 (dataset_id,))
    taken = conn.execute("SELECT family, name FROM meta.baseline_names WHERE dataset_id = %s AND family <> %s "
                         "AND lower(name) = lower(%s)", (dataset_id, family, name)).fetchone()
    if taken:
        raise HTTPException(422, f"This dataset's {FAMILY_LABEL[taken['family']]} baseline is already called "
                                 f"{taken['name']!r}")
    conn.execute("""INSERT INTO meta.baseline_names (dataset_id, family, name) VALUES (%s, %s, %s)
                    ON CONFLICT (dataset_id, family) DO UPDATE SET name = excluded.name""", (dataset_id, family, name))


def clean_name(raw: str | None) -> str | None:
    name = (raw or "").strip()
    if len(name) > NAME_MAX:
        raise HTTPException(422, f"A baseline name is at most {NAME_MAX} characters")
    return name or None


def fingerprint(columns: list[tuple[str, str]]) -> str:
    """Identifies a dataset's columns (name and physical type, in order), for matching a recipe to new data."""
    return hashlib.md5("\n".join(f"{n}\t{t}" for n, t in columns).encode()).hexdigest()


def schema_fingerprint(conn, dataset_id: int) -> str:
    rows = conn.execute(
        "SELECT name, physical_dtype FROM meta.fields WHERE dataset_id = %s ORDER BY ordinal", (dataset_id,)
    ).fetchall()
    return fingerprint([(r["name"], r["physical_dtype"]) for r in rows])


def snapshot(conn, dataset_id: int, src=None, family: str | None = None, targets: list[str] | None = None) -> dict:
    """The team decision as a baseline stores it: per field name, the decision, the settings (`family`'s only, when
    given), the inferred type and the stats a later upload is checked against for drift; plus the target order
    (`targets` when given, else the dataset's). With `src` (an assessment.Source), the same from its decisions."""
    dec = src.decisions() if src is not None else sql.SQL("meta.field_decisions")
    sets = src.settings() if src is not None else sql.SQL("meta.field_settings")
    fields = conn.execute(
        sql.SQL("""SELECT f.id, f.name, f.inferred_type, f.null_pct, f.n_unique, d.action, d.semantic_type, d.notes,
                          d.target_task, d.target_bins
                     FROM meta.fields f JOIN {} d ON d.field_id = f.id
                    WHERE f.dataset_id = %s ORDER BY f.ordinal""").format(dec),
        (dataset_id,),
    ).fetchall()
    settings: dict[int, dict[str, dict]] = {}
    for s in conn.execute(
        sql.SQL("""SELECT * FROM {} s WHERE s.dataset_id = %s AND (%s::text IS NULL OR s.family = %s)
                    ORDER BY field_id, family""").format(sets), (dataset_id, family, family)
    ).fetchall():
        settings.setdefault(s["field_id"], {})[s["family"]] = {k: s[k] for k in SETTING_KEYS}
    ds = conn.execute("SELECT target_columns FROM meta.datasets WHERE id = %s", (dataset_id,)).fetchone()
    return {
        "fields": {
            f["name"]: {
                "decision": {k: f[k] for k in DECISION_KEYS},
                "settings": settings.get(f["id"], {}),
                "inferred_type": f["inferred_type"],
                "null_pct": f["null_pct"],
                "n_unique": f["n_unique"],
            }
            for f in fields
        },
        "target_order": list(targets if targets is not None else ds["target_columns"] or []),
    }


def finalize(conn, recipe: dict, user_id: int) -> dict:
    return conn.execute(
        f"""UPDATE meta.recipes SET phase = 'finalized', finalized_by = %s, finalized_at = now(), snapshot = %s,
                   schema_fingerprint = %s
             WHERE id = %s RETURNING {RECIPE_COLS}""",
        (user_id, jsonb(snapshot(conn, recipe["dataset_id"], family=recipe["family"])),
         schema_fingerprint(conn, recipe["dataset_id"]),
         recipe["id"]),
    ).fetchone()


def backfill_snapshots(conn) -> int:
    """Snapshot finalized versions that have none (datasets that existed before recipes)."""
    rows = conn.execute(
        "SELECT id, dataset_id, family FROM meta.recipes WHERE phase = 'finalized' AND snapshot IS NULL"
    ).fetchall()
    for r in rows:
        conn.execute(
            "UPDATE meta.recipes SET snapshot = %s, schema_fingerprint = %s WHERE id = %s",
            (jsonb(snapshot(conn, r["dataset_id"], family=r["family"])), schema_fingerprint(conn, r["dataset_id"]),
             r["id"]),
        )
    return len(rows)


def snapshot_targets(snap: dict) -> list[str]:
    """The snapshot's targets in chain order."""
    targets = [n for n, f in snap["fields"].items() if f["decision"]["action"] == "target"]
    order = [t for t in snap.get("target_order") or [] if t in targets]
    return order + [t for t in targets if t not in order]


def snapshot_target_tasks(snap: dict) -> dict[str, dict]:
    """How each of the snapshot's targets is predicted, in chain order."""
    return {t: {"task": snap["fields"][t]["decision"].get("target_task") or "classes",
                "bins": snap["fields"][t]["decision"].get("target_bins")} for t in snapshot_targets(snap)}


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
                    "target_task": s["decision"].get("target_task"), "target_bins": s["decision"].get("target_bins"),
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


def diff(before: dict, after: dict, family: str | None = None, decisions_only: bool = False) -> dict:
    """What changed between two snapshots: per field, each part of the decision and each family's setting that
    differs (only `family`'s when given, none with `decisions_only`), fields added or removed, and the targets."""
    b_fields, a_fields = before["fields"], after["fields"]
    out = []
    for name in [*a_fields, *(n for n in b_fields if n not in a_fields)]:
        b, a = b_fields.get(name), a_fields.get(name)
        if b is None or a is None:
            out.append({"name": name, "change": "added" if b is None else "removed", "changes": []})
            continue
        changes = [{"key": k, "family": None, "before": b["decision"].get(k), "after": a["decision"].get(k)}
                   for k in ("action", "semantic_type", "target_task", "target_bins", "notes")
                   if b["decision"].get(k) != a["decision"].get(k)]
        for fam in [] if decisions_only else [family] if family else sorted(set(b["settings"]) | set(a["settings"])):
            bs, as_ = b["settings"].get(fam) or {}, a["settings"].get(fam) or {}
            changes += [{"key": k, "family": fam, "before": bs.get(k), "after": as_.get(k)}
                        for k in SETTING_KEYS if _norm(k, bs.get(k)) != _norm(k, as_.get(k))]
        if changes:
            out.append({"name": name, "change": "changed", "changes": changes})
    tb, ta = snapshot_targets(before), snapshot_targets(after)
    return {"fields": out, "targets": None if tb == ta else {"before": tb, "after": ta}}
