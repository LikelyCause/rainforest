"""Starting baselines: one per model family, from scratch, from another baseline of the dataset, or (for a new
dataset) from another dataset's baseline, with its targets and assessors chosen up front."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import recipes, reuse
from ..assessment import seed
from ..auth import CurrentUser, LeadUser
from ..db import connection
from ..schemas import StartBaselineIn
from .datasets import queue_target_work
from .team import _team_state

router = APIRouter(prefix="/api", tags=["baselines"])
FAMILY_ORDER = tuple(recipes.FAMILY_LABEL)


@router.get("/datasets/{dataset_id}/baselines")
def list_baselines(dataset_id: int, _: CurrentUser) -> list[dict]:
    """The dataset's baselines, one per family that has one (Trees, Linear, Neural), each with its versions newest
    first and its latest finalized version's id."""
    with connection() as conn:
        recipes.current(conn, dataset_id)
        rows = conn.execute(
            """SELECT r.id, r.family, r.version, r.phase, r.scope, r.parent_id, p.family AS parent_family,
                      p.version AS parent_version, p.dataset_id AS parent_dataset_id, r.created_at, r.finalized_at,
                      cu.name AS created_by_name, fu.name AS finalized_by_name
                 FROM meta.recipes r
                 LEFT JOIN meta.recipes p ON p.id = r.parent_id
                 LEFT JOIN meta.users cu ON cu.id = r.created_by
                 LEFT JOIN meta.users fu ON fu.id = r.finalized_by
                WHERE r.dataset_id = %s ORDER BY r.version DESC""",
            (dataset_id,),
        ).fetchall()
    out = []
    for fam in FAMILY_ORDER:
        versions = [{**r, "label": recipes.label(r)} for r in rows if r["family"] == fam]
        if versions:
            fin = [v for v in versions if v["phase"] == "finalized"]
            latest = max(fin, key=lambda v: (v["finalized_at"], v["id"]))["id"] if fin else None
            out.append({"family": fam, "label": recipes.FAMILY_LABEL[fam], "versions": versions,
                        "latest_finalized_id": latest})
    return out


@router.post("/datasets/{dataset_id}/baselines")
def start_baseline(dataset_id: int, body: StartBaselineIn, user: LeadUser) -> dict:
    """Start a baseline for `family` with its assessors (the lead may be one). A new dataset's first baseline is set
    up in place while nobody has started on it; otherwise the baseline in progress must be finalized first."""
    if body.family not in recipes.FAMILY_LABEL:
        raise HTTPException(422, f"Unknown model family {body.family!r}")
    assessors = list(dict.fromkeys(body.assessors))
    if not assessors:
        raise HTTPException(422, "Choose at least one assessor")
    if body.start != "recipe" and not body.targets:
        raise HTTPException(422, "Choose at least one target")
    if body.start != "scratch" and body.source_recipe_id is None:
        raise HTTPException(422, "Choose the baseline to start from")
    with connection() as conn:
        ds = conn.execute("SELECT status FROM meta.datasets WHERE id = %s", (dataset_id,)).fetchone()
        if ds is None:
            raise HTTPException(404, "Dataset not found")
        if ds["status"] != "ready":
            raise HTTPException(409, "Start a baseline once the dataset is ready")
        users = {r["id"]: r for r in conn.execute("SELECT id, active FROM meta.users WHERE id = ANY(%s)",
                                                   (assessors,)).fetchall()}
        if bad := [str(u) for u in assessors if u not in users or not users[u]["active"]]:
            raise HTTPException(422, {"message": "Unknown or deactivated assessor(s)", "errors": bad})
        cur = recipes.current(conn, dataset_id, lock="update")
        setup = recipes.is_setup(conn, cur)
        if body.start == "recipe":
            if not setup:
                raise HTTPException(409, "Only a new dataset's first baseline can start from another dataset's.")
            source = reuse.check_source(conn, body.source_recipe_id)
            if source["dataset_id"] == dataset_id:
                raise HTTPException(422, "Start from this dataset's own baseline with start=\"baseline\"")
            reuse.check_target(conn, cur)
            _, added, uncorrelated = reuse.apply(conn, cur, source, user["id"])
            recipe = recipes.get(conn, cur["id"])
        else:
            snap = None
            if body.start == "baseline":
                src = recipes.get(conn, body.source_recipe_id)
                if src is None or src["dataset_id"] != dataset_id or src["phase"] != "finalized":
                    raise HTTPException(404, "No finalized baseline of this dataset with that id")
                snap = recipes.snapshot_of(conn, src)
            if setup:
                conn.execute("UPDATE meta.recipes SET family = %s WHERE id = %s", (body.family, cur["id"]))
                recipe = recipes.get(conn, cur["id"])
            else:
                recipes.require_none_in_progress(cur)
                recipe = conn.execute(
                    f"""INSERT INTO meta.recipes (dataset_id, family, version, phase, scope, parent_id, created_by)
                        VALUES (%s, %s, %s, 'assessing', 'all', %s, %s) RETURNING {recipes.RECIPE_COLS}""",
                    (dataset_id, body.family, recipes.next_version(conn, dataset_id, body.family),
                     body.source_recipe_id, user["id"]),
                ).fetchone()
            uncorrelated, added = recipes.load_working(conn, dataset_id, snap, body.family, body.targets, user["id"])
        if recipe["phase"] == "assessing":
            for uid in assessors:
                conn.execute("INSERT INTO meta.dataset_assessors (recipe_id, user_id, assigned_by) VALUES (%s, %s, %s)",
                             (recipe["id"], uid, user["id"]))
                seed(conn, recipe, uid)
        work = queue_target_work(conn, dataset_id, uncorrelated, added)
        state = _team_state(conn, dataset_id, user)
    work()
    return state
