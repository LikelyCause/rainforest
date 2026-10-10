"""Reuse a finalized recipe on new data.

Fields match by name. A matched field whose inferred type is unchanged and whose stats haven't drifted is carried over
from the recipe and locked; every other field of the new data is reopened for assessment, and recipe fields the new
data lacks are listed. Targets carry over when they are still valid targets.
"""
from __future__ import annotations

from fastapi import HTTPException

from . import recipes
from .db import jsonb
from .semantics import target_problem, validate_decision

# A matched field has drifted when its share of missing values moves by more than this (absolute, 0-1) ...
NULL_PCT_DRIFT = 0.10
# ... or its number of distinct values changes by more than this factor either way.
UNIQUE_RATIO = 2.0
_EPS = 1e-9


def drift(old: dict, new: dict) -> dict | None:
    """The drifted stats of a matched field, as {stat: [recipe value, new value]}, or None when within bounds."""
    out = {}
    a, b = old.get("null_pct") or 0.0, new.get("null_pct") or 0.0
    if abs(b - a) > NULL_PCT_DRIFT + _EPS:
        out["null_pct"] = [a, b]
    u, v = old.get("n_unique") or 0, new.get("n_unique") or 0
    if (u == 0) != (v == 0) or (u and not (1 / UNIQUE_RATIO - _EPS <= v / u <= UNIQUE_RATIO + _EPS)):
        out["n_unique"] = [u, v]
    return out or None


def match(snap: dict, fields: list[dict]) -> dict:
    """How a recipe snapshot maps onto new fields (each with name, inferred_type, null_pct, n_unique), before targets
    and settings are checked: carried names, reopened [{name, reason, ...}] and recipe fields missing from the data."""
    recipe_fields = snap["fields"]
    carried, reopened = [], []
    for f in fields:
        old = recipe_fields.get(f["name"])
        if old is None:
            reopened.append({"name": f["name"], "reason": "new"})
        elif old["inferred_type"] != f["inferred_type"]:
            reopened.append({"name": f["name"], "reason": "type_changed", "before": old["inferred_type"],
                             "after": f["inferred_type"]})
        elif (d := drift(old, f)) is not None:
            reopened.append({"name": f["name"], "reason": "drift", "drift": d})
        else:
            carried.append(f["name"])
    names = {f["name"] for f in fields}
    return {"carried": carried, "reopened": reopened, "missing": [n for n in recipe_fields if n not in names]}


def _settings_problem(entry: dict, family: str, n_unique: int | None) -> list[str]:
    """Why a field's settings for the baseline's family can't be carried as they are."""
    s = entry["settings"].get(family)
    if s is None:
        return [f"no {family} settings"]
    return validate_decision({**entry["decision"], **s}, family, n_unique=n_unique)


def apply(conn, recipe: dict, source: dict, user_id: int) -> tuple[dict, list[int], list[dict]]:
    """Start `recipe` (a dataset's first version, nothing assessed yet) from the finalized `source` recipe. Writes the
    carried fields into the team decision, sets the targets, records the reopened fields and the summary, and moves
    the version to assessing, or to review when nothing needs assessing (a lead then confirms by finalizing).

    Returns the summary, plus what `fields.set_targets` reports (new targets, targets lacking associations) for the
    caller to start the matching jobs. Call holding the recipe row lock."""
    from .routers.fields import _merged_many, _write_history, set_targets

    dataset_id = recipe["dataset_id"]
    snap = conn.execute("SELECT snapshot FROM meta.recipes WHERE id = %s", (source["id"],)).fetchone()["snapshot"]
    fields = conn.execute(
        "SELECT id, name, inferred_type, null_pct, n_unique FROM meta.fields WHERE dataset_id = %s ORDER BY ordinal",
        (dataset_id,),
    ).fetchall()
    by_name = {f["name"]: f for f in fields}
    m = match(snap, fields)
    carried = []
    for name in m["carried"]:
        entry = snap["fields"][name]
        if entry["decision"]["action"] == "target" and (p := target_problem(entry["decision"]["semantic_type"],
                                                                            by_name[name]["n_unique"],
                                                                            entry["decision"].get("target_task"))):
            m["reopened"].append({"name": name, "reason": "target", "problem": p})
        elif errors := _settings_problem(entry, source["family"], by_name[name]["n_unique"]):
            m["reopened"].append({"name": name, "reason": "settings", "problem": "; ".join(errors[:3])})
        else:
            carried.append(name)

    # The new dataset's baseline is for the source's model family.
    fam = source["family"]
    conn.execute("UPDATE meta.recipes SET family = %s WHERE id = %s", (fam, recipe["id"]))
    conn.execute("UPDATE meta.datasets SET model_family = %s, updated_at = now() WHERE id = %s", (fam, dataset_id))
    ids = [f["id"] for f in fields]
    before = _merged_many(conn, ids, fam)
    for name in carried:
        entry, fid = snap["fields"][name], by_name[name]["id"]
        dec = entry["decision"]
        # The target role goes through set_targets below, so a carried target starts as the plain feature it was.
        conn.execute(
            """UPDATE meta.field_decisions
                  SET action = CASE WHEN %(action)s = 'target' THEN action ELSE %(action)s END,
                      semantic_type = %(type)s, notes = %(notes)s, reviewed = true,
                      source = CASE WHEN action = %(action)s AND semantic_type = %(type)s THEN source ELSE 'user' END,
                      updated_at = now()
                WHERE field_id = %(fid)s""",
            {"action": dec["action"], "type": dec["semantic_type"], "notes": dec.get("notes"), "fid": fid},
        )
        for family, s in entry["settings"].items():
            if family != fam:
                continue
            conn.execute(
                """UPDATE meta.field_settings
                      SET imputation = %s, impute_value = %s, add_missing_indicator = %s, transform = %s, scaling = %s,
                          encoding = %s, params = %s, source = 'user', updated_at = now()
                    WHERE field_id = %s AND family = %s""",
                (s["imputation"], s.get("impute_value"), bool(s.get("add_missing_indicator")), s["transform"],
                 s["scaling"], s["encoding"], jsonb(s.get("params") or {}), fid, family),
            )
    changed = {fid: row for fid, row in _merged_many(conn, ids, fam).items()
               if {k: v for k, v in row.items() if k != "updated_at"}
               != {k: v for k, v in (before.get(fid) or {}).items() if k != "updated_at"}}
    _write_history(conn, dataset_id, fam, None, before, changed, user_id)

    tasks = recipes.snapshot_target_tasks(snap)
    targets, not_carried = [], []
    current = {r["name"]: r for r in conn.execute(
        """SELECT f.name, f.n_unique, d.semantic_type FROM meta.fields f JOIN meta.field_decisions d
             ON d.field_id = f.id WHERE f.dataset_id = %s""", (dataset_id,)).fetchall()}
    for name in recipes.snapshot_targets(snap):
        if name not in current:
            continue
        if p := target_problem(current[name]["semantic_type"], current[name]["n_unique"], tasks[name]["task"]):
            not_carried.append({"name": name, "problem": p})
        else:
            targets.append(name)
    uncorrelated, added = set_targets(conn, dataset_id, targets, user_id,
                                      {t: {"target_task": tasks[t]["task"], "target_bins": tasks[t]["bins"]}
                                       for t in targets})

    reopened_ids = [by_name[r["name"]]["id"] for r in m["reopened"]]
    conn.execute(
        """INSERT INTO meta.recipe_reopened (recipe_id, field_id) SELECT %s, unnest(%s::bigint[])
           ON CONFLICT DO NOTHING""",
        (recipe["id"], reopened_ids),
    )
    conn.execute(
        "UPDATE meta.field_decisions SET reviewed = false WHERE field_id = ANY(%s) AND reviewed", (reopened_ids,)
    )
    ds = conn.execute("SELECT name FROM meta.datasets WHERE id = %s", (source["dataset_id"],)).fetchone()
    order = {f["name"]: i for i, f in enumerate(fields)}
    summary = {
        "source": {"recipe_id": source["id"], "dataset_id": source["dataset_id"], "family": source["family"],
                   "dataset_name": ds["name"] if ds else None, "version": source["version"]},
        "carried": carried,
        "reopened": sorted(m["reopened"], key=lambda r: order[r["name"]]),
        "missing": m["missing"],
        "targets": targets,
        "targets_not_carried": not_carried,
    }
    conn.execute(
        """UPDATE meta.recipes SET parent_id = %s, scope = 'reopened', reuse = %s, phase = %s WHERE id = %s""",
        (source["id"], jsonb(summary), "assessing" if reopened_ids else "review", recipe["id"]),
    )
    return summary, added, uncorrelated


def check_source(conn, recipe_id: int) -> dict:
    """A recipe that can be reused: it exists and is finalized."""
    source = recipes.get(conn, recipe_id)
    if source is None:
        raise HTTPException(404, "Recipe not found")
    if source["phase"] != "finalized":
        raise HTTPException(409, f"{recipes.label(source)} of that dataset isn't finalized; only finalized recipes "
                                 "can be reused.")
    return source


def check_target(conn, recipe: dict) -> None:
    """A dataset can start from a recipe while its first version hasn't been assessed or reused yet."""
    row = conn.execute(
        """SELECT reuse IS NOT NULL AS reused,
                  EXISTS (SELECT 1 FROM meta.dataset_assessors WHERE recipe_id = %(r)s)
                  OR EXISTS (SELECT 1 FROM meta.assessments WHERE recipe_id = %(r)s) AS busy
             FROM meta.recipes WHERE id = %(r)s""",
        {"r": recipe["id"]},
    ).fetchone()
    if recipe["version"] != 1 or recipe["phase"] != "assessing" or row["reused"]:
        raise HTTPException(409, "Only a new dataset can start from a recipe: its first version, still assessing, "
                                 "with no recipe applied yet.")
    if row["busy"]:
        raise HTTPException(409, "Assessment has started on this dataset; it can't start from a recipe any more.")
