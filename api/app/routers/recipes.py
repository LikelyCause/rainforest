"""Recipe versions: their history and differences, starting the next version, and reusing a recipe on new data."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from .. import recipes, reuse
from ..auth import CurrentUser, LeadUser
from ..db import connection
from ..schemas import ApplyRecipeIn, NewVersionIn
from .datasets import queue_target_work
from .team import _team_state

router = APIRouter(prefix="/api", tags=["recipes"])


@router.get("/recipes")
def finalized_recipes(_: CurrentUser) -> list[dict]:
    """Every finalized recipe version of every dataset, newest first: what "Start from recipe" offers."""
    with connection() as conn:
        rows = conn.execute(
            """SELECT r.id, r.dataset_id, d.name AS dataset_name, r.family, r.version,
                      meta.baseline_name(r.dataset_id, r.family) AS name,
                      (SELECT count(*) FROM jsonb_object_keys(r.snapshot -> 'fields')) AS field_count,
                      r.finalized_by, u.name AS finalized_by_name, r.finalized_at
                 FROM meta.recipes r JOIN meta.datasets d ON d.id = r.dataset_id
                 LEFT JOIN meta.users u ON u.id = r.finalized_by
                WHERE r.phase = 'finalized' AND r.snapshot IS NOT NULL
                ORDER BY r.finalized_at DESC, r.id DESC"""
        ).fetchall()
    return [{**r, "label": recipes.label(r)} for r in rows]


@router.get("/datasets/{dataset_id}/recipes")
def versions(dataset_id: int, _: CurrentUser) -> list[dict]:
    """The dataset's baseline versions, by family then newest first, with who started and finalized each, how many
    fields it reopened (null when every field was open), what it was built from, and how a reused one matched."""
    with connection() as conn:
        recipes.current(conn, dataset_id)
        rows = conn.execute(
            """SELECT r.id, r.dataset_id, r.family, r.version, r.phase, r.scope, r.parent_id, r.created_by,
                      cu.name AS created_by_name, r.created_at, r.finalized_by, fu.name AS finalized_by_name,
                      r.finalized_at, r.reuse,
                      CASE WHEN r.scope = 'reopened'
                           THEN (SELECT count(*) FROM meta.recipe_reopened o WHERE o.recipe_id = r.id) END AS reopened,
                      p.dataset_id AS parent_dataset_id, p.family AS parent_family, p.version AS parent_version,
                      meta.baseline_name(r.dataset_id, r.family) AS name,
                      meta.baseline_name(p.dataset_id, p.family) AS parent_name,
                      pd.name AS parent_dataset_name
                 FROM meta.recipes r
                 LEFT JOIN meta.users cu ON cu.id = r.created_by
                 LEFT JOIN meta.users fu ON fu.id = r.finalized_by
                 LEFT JOIN meta.recipes p ON p.id = r.parent_id
                 LEFT JOIN meta.datasets pd ON pd.id = p.dataset_id
                WHERE r.dataset_id = %s
                ORDER BY array_position(ARRAY['tree', 'linear', 'neural'], r.family), r.version DESC""",
            (dataset_id,),
        ).fetchall()
    return [{**r, "label": recipes.label(r)} for r in rows]


def _recipe_of(conn, dataset_id: int, recipe_id: int) -> dict:
    row = recipes.get(conn, recipe_id)
    if row is None or row["dataset_id"] != dataset_id:
        raise HTTPException(404, "Baseline version not found")
    return row


@router.get("/datasets/{dataset_id}/recipes/diff")
def diff(dataset_id: int, _: CurrentUser, a: int = Query(..., ge=1), b: int = Query(..., ge=1)) -> dict:
    """Fields whose team decision or settings differ between baseline versions `a` and `b` (recipe ids). Versions of
    different families compare their decisions only. A version still being worked is compared as its team decision
    stands; assessments are never part of it."""
    with connection() as conn:
        ra, rb = _recipe_of(conn, dataset_id, a), _recipe_of(conn, dataset_id, b)
        out = recipes.diff(
            recipes.snapshot_of(conn, ra),
            recipes.snapshot_of(conn, rb),
            ra["family"] if ra["family"] == rb["family"] else None,
            decisions_only=ra["family"] != rb["family"],
        )
    keys = ("id", "family", "version", "phase", "name")
    return {
        "a": {**{k: ra[k] for k in keys}, "label": recipes.label(ra)},
        "b": {**{k: rb[k] for k in keys}, "label": recipes.label(rb)},
        **out,
    }


@router.post("/datasets/{dataset_id}/recipes")
def new_version(dataset_id: int, body: NewVersionIn, user: LeadUser) -> dict:
    """Start the next version from the latest, finalized one. Reopened fields go through blind assessment and review
    again; the rest are carried over and locked. With nothing reopened, the lead edits the team decision directly
    (the version opens in review) and finalizes."""
    reopen = list(dict.fromkeys(body.reopen))
    with connection() as conn:
        status = conn.execute("SELECT status FROM meta.datasets WHERE id = %s", (dataset_id,)).fetchone()
        if status is None:
            raise HTTPException(404, "Dataset not found")
        if status["status"] != "ready":
            raise HTTPException(409, "The dataset is still processing")
        known = {
            r["id"]
            for r in conn.execute(
                "SELECT id FROM meta.fields WHERE dataset_id = %s AND id = ANY(%s)", (dataset_id, reopen)
            ).fetchall()
        }
        if unknown := [i for i in reopen if i not in known]:
            raise HTTPException(422, {"message": "Unknown field(s)", "errors": [str(i) for i in unknown]})
        family = body.family or recipes.current(conn, dataset_id)["family"]
        if family not in recipes.FAMILY_LABEL:
            raise HTTPException(422, f"Unknown model family {family!r}")
        recipe = recipes.start_version(
            conn, dataset_id, user["id"], family, "assessing" if reopen else "review", "reopened" if reopen else "all"
        )
        uncorrelated, added = recipe.pop("_target_work")
        if reopen:
            reopen += [
                r["field_id"]
                for r in conn.execute(
                    "SELECT field_id FROM meta.field_decisions WHERE dataset_id = %s AND NOT reviewed AND field_id <> ALL(%s)",
                    (dataset_id, reopen),
                ).fetchall()
            ]
            conn.execute(
                "INSERT INTO meta.recipe_reopened (recipe_id, field_id) SELECT %s, unnest(%s::bigint[])",
                (recipe["id"], reopen),
            )
            # The queue of what is left to look at: reopened fields are to do again, carried ones are settled.
            conn.execute(
                """UPDATE meta.field_decisions SET reviewed = (field_id <> ALL(%s))
                    WHERE dataset_id = %s AND reviewed IS DISTINCT FROM (field_id <> ALL(%s))""",
                (reopen, dataset_id, reopen),
            )
        start = queue_target_work(conn, dataset_id, uncorrelated, added)
        state = _team_state(conn, dataset_id, user)
    start()
    return state


@router.post("/datasets/{dataset_id}/recipes/apply")
def apply_recipe(dataset_id: int, body: ApplyRecipeIn, user: LeadUser) -> dict:
    """Start a new dataset from a finalized recipe of any dataset (see reuse.py). The team state's recipe carries the
    reuse summary."""
    with connection() as conn:
        status = conn.execute("SELECT status FROM meta.datasets WHERE id = %s", (dataset_id,)).fetchone()
        if status is None:
            raise HTTPException(404, "Dataset not found")
        if status["status"] != "ready":
            raise HTTPException(409, "Start from a recipe once the dataset is ready")
        recipe = recipes.current(conn, dataset_id, lock="update")
        reuse.check_target(conn, recipe)
        source = reuse.check_source(conn, body.recipe_id)
        _, added, uncorrelated = reuse.apply(conn, recipe, source, user["id"])
        start = queue_target_work(conn, dataset_id, uncorrelated, added)
        state = _team_state(conn, dataset_id, user)
    start()
    return state
