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
        return conn.execute(
            """SELECT r.id, r.dataset_id, d.name AS dataset_name, r.version,
                      (SELECT count(*) FROM jsonb_object_keys(r.snapshot -> 'fields')) AS field_count,
                      r.finalized_by, u.name AS finalized_by_name, r.finalized_at
                 FROM meta.recipes r JOIN meta.datasets d ON d.id = r.dataset_id
                 LEFT JOIN meta.users u ON u.id = r.finalized_by
                WHERE r.phase = 'finalized' AND r.snapshot IS NOT NULL
                ORDER BY r.finalized_at DESC, r.id DESC"""
        ).fetchall()


@router.get("/datasets/{dataset_id}/recipes")
def versions(dataset_id: int, _: CurrentUser) -> list[dict]:
    """The dataset's recipe versions, newest first, with who started and finalized each, how many fields it reopened
    (null when every field was open), what it was built from, and how a reused recipe matched."""
    with connection() as conn:
        recipes.current(conn, dataset_id)
        return conn.execute(
            """SELECT r.id, r.dataset_id, r.version, r.phase, r.scope, r.parent_id, r.created_by,
                      cu.name AS created_by_name, r.created_at, r.finalized_by, fu.name AS finalized_by_name,
                      r.finalized_at, r.reuse,
                      CASE WHEN r.scope = 'reopened'
                           THEN (SELECT count(*) FROM meta.recipe_reopened o WHERE o.recipe_id = r.id) END AS reopened,
                      p.dataset_id AS parent_dataset_id, p.version AS parent_version, pd.name AS parent_dataset_name
                 FROM meta.recipes r
                 LEFT JOIN meta.users cu ON cu.id = r.created_by
                 LEFT JOIN meta.users fu ON fu.id = r.finalized_by
                 LEFT JOIN meta.recipes p ON p.id = r.parent_id
                 LEFT JOIN meta.datasets pd ON pd.id = p.dataset_id
                WHERE r.dataset_id = %s ORDER BY r.version DESC""",
            (dataset_id,),
        ).fetchall()


def _version(conn, dataset_id: int, version: int) -> dict:
    row = conn.execute(f"SELECT {recipes.RECIPE_COLS} FROM meta.recipes WHERE dataset_id = %s AND version = %s",
                       (dataset_id, version)).fetchone()
    if row is None:
        raise HTTPException(404, f"Recipe v{version} not found")
    return row


@router.get("/datasets/{dataset_id}/recipes/diff")
def diff(dataset_id: int, _: CurrentUser, a: int = Query(..., ge=1), b: int = Query(..., ge=1),
         family: str | None = None) -> dict:
    """Fields whose team decision or settings differ between versions `a` and `b`. A version still being worked is
    compared as its team decision stands; assessments are never part of it."""
    with connection() as conn:
        ra, rb = _version(conn, dataset_id, a), _version(conn, dataset_id, b)
        out = recipes.diff(recipes.snapshot_of(conn, ra), recipes.snapshot_of(conn, rb), family)
    return {"a": {k: ra[k] for k in ("id", "version", "phase")}, "b": {k: rb[k] for k in ("id", "version", "phase")},
            **out}


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
        known = {r["id"] for r in conn.execute("SELECT id FROM meta.fields WHERE dataset_id = %s AND id = ANY(%s)",
                                                (dataset_id, reopen)).fetchall()}
        if unknown := [i for i in reopen if i not in known]:
            raise HTTPException(422, {"message": "Unknown field(s)", "errors": [str(i) for i in unknown]})
        recipe = recipes.start_version(conn, dataset_id, user["id"], "assessing" if reopen else "review",
                                       "reopened" if reopen else "all")
        if reopen:
            conn.execute("INSERT INTO meta.recipe_reopened (recipe_id, field_id) SELECT %s, unnest(%s::bigint[])",
                         (recipe["id"], reopen))
            # The queue of what is left to look at: reopened fields are to do again, carried ones are settled.
            conn.execute(
                """UPDATE meta.field_decisions SET reviewed = (field_id <> ALL(%s))
                    WHERE dataset_id = %s AND reviewed IS DISTINCT FROM (field_id <> ALL(%s))""",
                (reopen, dataset_id, reopen),
            )
        return _team_state(conn, dataset_id, user)


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
