"""Experiments: where exports and model runs live, and who may see them.

An experiment is built from a source:
- `recipe:<recipe_id>`, a finalized recipe version (a team experiment), visible to everyone;
- `assessment:<recipe_id>:<user_id>`, one assessor's own assessment of a version (personal). It is private to that
  assessor while the version is `assessing`, and visible to the team from `review` onwards (phases only move
  forward, so once visible it stays visible).

Every read of experiments, exports and runs goes through `visible_ids` / `get_visible`. The SQL console's role gets
the same rule from row-level security (migration 010). MLflow has no access control of its own, so runs of a
private experiment are not logged to it until the version reaches review (`flush_deferred`).
"""
from __future__ import annotations

import logging

from fastapi import HTTPException

from . import recipes
from .db import fetch_all, jsonb
from .semantics import SETTING_KEYS

log = logging.getLogger(__name__)

COLS = """x.id, x.dataset_id, x.name, x.owner, x.recipe_id, x.assessor_id, x.source, x.notes, x.created_at,
          x.archived_at, x.is_default, x.target_order, xr.family AS recipe_family, xr.version AS recipe_version,
          xr.phase AS recipe_phase, meta.baseline_name(xr.dataset_id, xr.family) AS recipe_name,
          (x.assessor_id IS NOT NULL AND xr.phase = 'assessing') AS private,
          (SELECT name FROM meta.users u WHERE u.id = coalesce(x.owner, x.assessor_id)) AS owner_name"""
FROM = "meta.experiments x JOIN meta.recipes xr ON xr.id = x.recipe_id"
# Metric that stands for a run when comparing experiments, in order of preference (validation split).
HEADLINE_METRICS = ("roc_auc", "roc_auc_ovr_macro", "r2", "accuracy")


def visible_ids(user_id: int) -> str:
    """SQL for the ids of the experiments `user_id` may see, to use as `experiment_id IN (...)`."""
    return (f"SELECT x.id FROM {FROM} WHERE x.assessor_id IS NULL OR x.assessor_id = {int(user_id)} "
            "OR xr.phase <> 'assessing'")


def source_key(recipe_id: int, assessor_id: int | None) -> str:
    return f"recipe:{recipe_id}" if assessor_id is None else f"assessment:{recipe_id}:{assessor_id}"


def parse_source(source: str) -> tuple[int, int | None]:
    """(recipe id, assessor id or None) from a source key; 422 for anything else."""
    parts = source.split(":")
    try:
        if parts[0] == "recipe" and len(parts) == 2:
            return int(parts[1]), None
        if parts[0] == "assessment" and len(parts) == 3:
            return int(parts[1]), int(parts[2])
    except ValueError:
        pass
    raise HTTPException(422, "source must be recipe:<recipe_id> or assessment:<recipe_id>:<user_id>")


def get(conn, experiment_id: int) -> dict | None:
    return conn.execute(f"SELECT {COLS} FROM {FROM} WHERE x.id = %s", (experiment_id,)).fetchone()


def get_visible(conn, experiment_id: int, user: dict) -> dict:
    """The experiment, or 404 when it doesn't exist or `user` may not see it (the same answer, so a private
    experiment's existence doesn't show)."""
    x = get(conn, experiment_id)
    if x is None or (x["private"] and x["assessor_id"] != user["id"]):
        raise HTTPException(404, "Experiment not found")
    return x


def require_writable(x: dict, user: dict) -> None:
    """Adding exports or runs: anyone on a team experiment, only its assessor on a personal one; never archived."""
    if x["archived_at"] is not None:
        raise HTTPException(409, f"Experiment {x['name']!r} is archived; unarchive it to add to it.")
    if x["assessor_id"] is not None and x["assessor_id"] != user["id"]:
        raise HTTPException(403, "Only the assessor whose assessment it is can add to a personal experiment.")


def adjust_denied(x: dict, user: dict) -> tuple[int, str] | None:
    """Why `user` can't adjust this experiment's copy of its baseline, or None: only its owner can, on a team
    experiment they made (not the default one for a baseline's exports), while it isn't archived."""
    if x["assessor_id"] is not None:
        return 409, "A personal experiment follows your assessment; change your assessment instead."
    if x["is_default"] or x["owner"] != user["id"]:
        return 403, ("Only the experiment's owner adjusts it. Start your own experiment from this baseline to try "
                     "changes.")
    if x["archived_at"] is not None:
        return 409, f"Experiment {x['name']!r} is archived; unarchive it to adjust it."
    return None


def can_adjust(x: dict, user: dict) -> bool:
    return adjust_denied(x, user) is None


def require_editable(x: dict, user: dict) -> None:
    """Renaming, notes, archiving: the owner (the assessor, for a personal one) or a lead."""
    if user["role"] != "lead" and user["id"] not in (x["owner"], x["assessor_id"]):
        raise HTTPException(403, "Only its owner or a lead can change this experiment.")


def create(conn, dataset_id: int, name: str, source: str, user: dict, notes: str | None = None) -> dict:
    recipe_id, assessor_id = parse_source(source)
    recipe = recipes.get(conn, recipe_id)
    if recipe is None or recipe["dataset_id"] != dataset_id:
        raise HTTPException(404, "Recipe not found")
    if assessor_id is None:
        if recipe["phase"] != "finalized":
            raise HTTPException(409, f"{recipes.label(recipe)} isn't finalized; a team experiment needs a finalized "
                                     "version. Experiment on your own assessment instead.")
    else:
        if assessor_id != user["id"]:
            raise HTTPException(403, "A personal experiment can only be built from your own assessment.")
        if conn.execute("SELECT 1 FROM meta.assessments WHERE recipe_id = %s AND user_id = %s LIMIT 1",
                        (recipe_id, assessor_id)).fetchone() is None:
            raise HTTPException(403, f"You don't assess {recipes.label(recipe, lower=True)}.")
    row = conn.execute(
        """INSERT INTO meta.experiments (dataset_id, name, owner, recipe_id, assessor_id, source, notes)
           VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
        (dataset_id, name.strip(), user["id"], recipe_id, assessor_id, source_key(recipe_id, assessor_id), notes),
    ).fetchone()
    x = get(conn, row["id"])
    seed_copy(conn, x)
    return get(conn, row["id"])


def team_for(conn, recipe: dict) -> dict:
    """The default team experiment of a finalized version, created on first use."""
    conn.execute("SELECT pg_advisory_xact_lock(7311207, %s)", (recipe["id"],))
    row = conn.execute(
        f"SELECT {COLS} FROM {FROM} WHERE x.recipe_id = %s AND x.is_default ORDER BY x.id LIMIT 1",
        (recipe["id"],),
    ).fetchone()
    if row is not None:
        return row
    made = conn.execute(
        """INSERT INTO meta.experiments (dataset_id, name, recipe_id, source, is_default)
           VALUES (%s, %s, %s, %s, true) RETURNING id""",
        (recipe["dataset_id"], team_name(recipe), recipe["id"], source_key(recipe["id"], None)),
    ).fetchone()
    seed_copy(conn, get(conn, made["id"]))
    return get(conn, made["id"])


def seed_copy(conn, x: dict) -> None:
    """Give a team experiment its own copy of its (finalized) baseline: every field's decision and the settings for
    the baseline's family, and the baseline's target order. Rows it already has stay."""
    recipe = recipes.get(conn, x["recipe_id"])
    if x["assessor_id"] is not None or recipe is None or recipe["phase"] != "finalized":
        return
    snap = recipes.snapshot_of(conn, recipe)
    fam = recipe["family"]
    fields = conn.execute("SELECT id, name, suggestion FROM meta.fields WHERE dataset_id = %s",
                          (x["dataset_id"],)).fetchall()
    decisions, settings = [], []
    for f in fields:
        entry = snap["fields"].get(f["name"])
        if entry is None:
            continue
        d = entry["decision"]
        decisions.append((x["id"], f["id"], d["action"], d["semantic_type"], d.get("notes"), d.get("target_task"),
                          jsonb(d["target_bins"]) if d.get("target_bins") else None))
        s = entry["settings"].get(fam) or (f["suggestion"] or {}).get(fam) or {}
        settings.append((x["id"], f["id"], fam, s.get("imputation") or "none", s.get("impute_value"),
                         bool(s.get("add_missing_indicator")), s.get("transform") or "none",
                         s.get("scaling") or "none", s.get("encoding") or "none", jsonb(s.get("params") or {})))
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO meta.experiment_decisions (experiment_id, field_id, action, semantic_type, notes, target_task,
                                                     target_bins)
               VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""", decisions)
        cur.executemany(
            """INSERT INTO meta.experiment_settings (experiment_id, field_id, family, imputation, impute_value,
                          add_missing_indicator, transform, scaling, encoding, params)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""", settings)
    conn.execute("UPDATE meta.experiments SET target_order = %s WHERE id = %s AND target_order IS NULL",
                 (recipes.snapshot_targets(snap), x["id"]))


def snapshot_for(conn, x: dict) -> dict:
    """What an export of this experiment is built from, shaped like a baseline snapshot: a team experiment's copy
    of its baseline (with its adjustments), a default one's baseline itself, a personal one's assessment."""
    from .assessment import Source, personal_snapshot

    recipe = recipes.get(conn, x["recipe_id"])
    if x["assessor_id"] is not None:
        return personal_snapshot(conn, recipe, x["assessor_id"])
    if x["is_default"] or recipe["phase"] != "finalized":
        return recipes.snapshot_of(conn, recipe)
    return recipes.snapshot(conn, x["dataset_id"], Source(recipe, experiment=x), family=recipe["family"],
                            targets=x["target_order"] or [])


NO_ADJUSTMENTS = {"fields": [], "targets": None, "count": 0, "targets_changed": False}


def adjustments(conn, x: dict, snapshots: dict[int, dict] | None = None) -> dict:
    """How a team experiment differs from its baseline: per field, what changed (notes aside); the targets when
    they differ. `snapshots` caches baseline snapshots by recipe id across calls."""
    if x["assessor_id"] is not None or x["is_default"]:
        return dict(NO_ADJUSTMENTS)
    snapshots = {} if snapshots is None else snapshots
    if x["recipe_id"] not in snapshots:
        recipe = recipes.get(conn, x["recipe_id"])
        snapshots[x["recipe_id"]] = (recipe, recipes.snapshot_of(conn, recipe) if recipe["phase"] == "finalized" else None)
    recipe, base = snapshots[x["recipe_id"]]
    if base is None:
        return dict(NO_ADJUSTMENTS)
    d = recipes.diff(base, snapshot_for(conn, x), recipe["family"])
    changed = []
    for f in d["fields"]:
        kept = [c for c in f["changes"] if c["key"] != "notes"]
        if kept or f["change"] != "changed":
            changed.append({**f, "changes": kept})
    return {"fields": changed, "targets": d["targets"], "count": len(changed),
            "targets_changed": d["targets"] is not None}


def revert(conn, x: dict, field_ids: list[int] | None) -> None:
    """Put fields of a team experiment's copy (every field when None) back to its baseline."""
    if field_ids is None:
        conn.execute("DELETE FROM meta.experiment_decisions WHERE experiment_id = %s", (x["id"],))
        conn.execute("DELETE FROM meta.experiment_settings WHERE experiment_id = %s", (x["id"],))
        conn.execute("UPDATE meta.experiments SET target_order = NULL WHERE id = %s", (x["id"],))
        seed_copy(conn, {**x, "target_order": None})
        return
    conn.execute("DELETE FROM meta.experiment_decisions WHERE experiment_id = %s AND field_id = ANY(%s)",
                 (x["id"], field_ids))
    conn.execute("DELETE FROM meta.experiment_settings WHERE experiment_id = %s AND field_id = ANY(%s)",
                 (x["id"], field_ids))
    seed_copy(conn, x)
    # The order keeps the experiment's own targets as they were, then any the revert brought back.
    targets = [r["name"] for r in conn.execute(
        """SELECT f.name FROM meta.experiment_decisions e JOIN meta.fields f ON f.id = e.field_id
            WHERE e.experiment_id = %s AND e.action = 'target' ORDER BY f.ordinal""", (x["id"],)).fetchall()]
    base = recipes.snapshot_targets(recipes.snapshot_of(conn, recipes.get(conn, x["recipe_id"])))
    order = [t for t in (x["target_order"] or []) if t in targets]
    order += [t for t in base if t in targets and t not in order] + [t for t in targets if t not in order]
    conn.execute("UPDATE meta.experiments SET target_order = %s WHERE id = %s", (order, x["id"]))


def backfill_copies(conn) -> int:
    """Copies for team experiments on finalized baselines that have none (made before experiments had them)."""
    rows = conn.execute(
        f"""SELECT {COLS} FROM {FROM}
             WHERE x.assessor_id IS NULL AND xr.phase = 'finalized'
               AND NOT EXISTS (SELECT 1 FROM meta.experiment_decisions e WHERE e.experiment_id = x.id)"""
    ).fetchall()
    for x in rows:
        seed_copy(conn, x)
    return len(rows)


def team_name(recipe: dict) -> str:
    return f"Team · {recipes.short_label(recipe['family'], recipe['version'], recipe['name'])}"


def list_for(conn, dataset_id: int, user: dict) -> list[dict]:
    rows = conn.execute(
        f"""SELECT {COLS},
                   (SELECT count(*) FROM meta.exports e WHERE e.experiment_id = x.id) AS export_count,
                   (SELECT count(*) FROM meta.model_runs r WHERE r.experiment_id = x.id) AS run_count,
                   (SELECT max(r.mlflow_experiment_id) FROM meta.model_runs r WHERE r.experiment_id = x.id)
                       AS mlflow_experiment_id
              FROM {FROM}
             WHERE x.dataset_id = %s AND x.id IN ({visible_ids(user['id'])})
             ORDER BY x.archived_at IS NOT NULL, x.created_at DESC, x.id DESC""",
        (dataset_id,),
    ).fetchall()
    best = best_by_experiment(conn, [r["id"] for r in rows])
    out, snapshots = [], {}
    for r in rows:
        adj = adjustments(conn, r, snapshots)
        out.append({**r, "best": best.get(r["id"]), "adjusted_count": adj["count"],
                    "targets_changed": adj["targets_changed"], "can_adjust": can_adjust(r, user)})
    return out


def best_by_experiment(conn, ids: list[int]) -> dict[int, dict]:
    """Per experiment, its best finished run on the validation split: the first target (in chain order) scored with
    the most preferred metric any of its runs has."""
    out: dict[int, dict] = {}
    runs = conn.execute(
        """SELECT id, experiment_id, report -> 'order' AS run_order, metrics -> 'valid' AS valid
             FROM meta.model_runs WHERE experiment_id = ANY(%s) AND status = 'done'""",
        (ids,),
    ).fetchall()
    by_exp: dict[int, list[dict]] = {}
    for r in runs:
        order, valid = r["run_order"] or [], r["valid"] or {}
        scores = valid.get(order[0]) if order else None
        if isinstance(scores, dict):
            by_exp.setdefault(r["experiment_id"], []).append({"run_id": r["id"], "target": order[0], **scores})
    for xid, scored in by_exp.items():
        for metric in HEADLINE_METRICS:
            have = [s for s in scored if isinstance(s.get(metric), (int, float))]
            if have:
                top = max(have, key=lambda s: s[metric])
                out[xid] = {"metric": metric, "value": float(top[metric]), "run_id": top["run_id"],
                            "target": top["target"]}
                break
    return out


# ------------------------------------------------------------------------------------------- MLflow
def run_is_private(conn, run_id: int) -> bool:
    """Whether a run's experiment is private now. Locks the version's row (FOR SHARE) so the answer holds until the
    caller commits: opening review takes it FOR UPDATE, so a run either sees review open or is flushed by it."""
    row = conn.execute(
        """SELECT x.assessor_id, xr.phase FROM meta.model_runs r JOIN meta.experiments x ON x.id = r.experiment_id
             JOIN meta.recipes xr ON xr.id = x.recipe_id WHERE r.id = %s FOR SHARE OF xr""",
        (run_id,),
    ).fetchone()
    return row is not None and row["assessor_id"] is not None and row["phase"] == "assessing"


DEFERRED = ("SELECT r.id FROM meta.model_runs r JOIN meta.experiments x ON x.id = r.experiment_id "
            "WHERE r.status = 'done' AND r.mlflow_run_id IS NULL AND r.report ? 'mlflow_pending' "
            "AND (%(recipe)s::bigint IS NULL OR x.recipe_id = %(recipe)s)")


def flush_deferred(recipe_id: int | None = None) -> int:
    """Log the runs that waited for their experiment to stop being private (all such runs, or one version's).
    Runs when review opens and at startup; mlflow_backfill skips runs still private, and records a failure on the
    run as for any other logging failure."""
    from .modeling import mlflow_backfill, mlflow_log

    if not mlflow_log.enabled():
        return 0
    ids = [r["id"] for r in fetch_all(DEFERRED, {"recipe": recipe_id})]
    if not ids:
        return 0
    logged, _ = mlflow_backfill.backfill(ids, say=lambda m: log.info("deferred MLflow logging: %s", m))
    return logged


def startup_flush() -> None:
    try:
        flush_deferred()
    except Exception:  # noqa: BLE001
        log.exception("flushing deferred MLflow runs failed")
