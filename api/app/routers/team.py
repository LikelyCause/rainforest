"""The team process on a dataset's current recipe version: assessors, submitting, team review and finalizing."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import experiments, jobs, recipes
from ..assessment import (Source, adopt, assessments_by_field, assessor_row, core_key, disagreements, progress, seed,
                          settings_key, write_decision, write_settings)
from ..auth import CurrentUser, LeadUser
from ..db import connection
from ..schemas import AssessorIn, CommentIn, FinalizeIn
from ..semantics import suggest
from .fields import (STAT_KEYS, _active_family, _field, _merged_many, _remember_datetime_format, _sync_targets,
                     _write_history, has_screen_fields, screen_targets)

router = APIRouter(prefix="/api", tags=["team"])


def _assessor_ids(conn, recipe_id: int) -> list[int]:
    return [r["user_id"] for r in conn.execute(
        "SELECT user_id FROM meta.dataset_assessors WHERE recipe_id = %s ORDER BY user_id", (recipe_id,)
    ).fetchall()]


def _by_field(conn, recipe: dict) -> dict[int, list[dict]]:
    """The assessors' choices on the fields this version works on (carried-over fields aren't assessed)."""
    by_field = assessments_by_field(conn, recipe["id"], _assessor_ids(conn, recipe["id"]))
    open_ = recipes.open_field_ids(conn, recipe)
    if open_ is None:
        return by_field
    keep = set(open_)
    return {fid: rows for fid, rows in by_field.items() if fid in keep}


def _require_phase(recipe: dict, *phases: str, doing: str) -> None:
    if recipe["phase"] not in phases:
        raise HTTPException(409, f"{recipes.label(recipe)} is {recipe['phase']}; {doing} isn't possible now.")


def _open_count(conn, recipe: dict) -> int:
    open_ = recipes.open_field_ids(conn, recipe)
    if open_ is not None:
        return len(open_)
    return conn.execute("SELECT count(*) AS n FROM meta.fields WHERE dataset_id = %s",
                        (recipe["dataset_id"],)).fetchone()["n"]


def _review_summary(conn, recipe: dict) -> dict:
    by_field = _by_field(conn, recipe)
    dis = sorted(disagreements(by_field))
    total = _open_count(conn, recipe)
    open_ = conn.execute(
        "SELECT count(*) AS n FROM meta.field_decisions WHERE field_id = ANY(%s) AND NOT reviewed", (dis,)
    ).fetchone()["n"]
    return {"fields": total, "adopted": total - len(dis) if by_field else 0, "disagreements": len(dis),
            "open": open_}


def _team_state(conn, dataset_id: int, user: dict) -> dict:
    recipe = recipes.current(conn, dataset_id)
    assessors = progress(conn, recipe)
    mine = next((a for a in assessors if a["user_id"] == user["id"]), None)
    names = {r["id"]: r["name"] for r in conn.execute(
        "SELECT id, name FROM meta.users WHERE id = ANY(%s)", ([recipe["created_by"], recipe["finalized_by"]],)
    ).fetchall()}
    open_ = recipes.open_field_ids(conn, recipe)
    reuse = conn.execute("SELECT reuse FROM meta.recipes WHERE id = %s", (recipe["id"],)).fetchone()["reuse"]
    return {
        "recipe": {**recipe, "label": recipes.label(recipe), "setup": recipes.is_setup(conn, recipe),
                   "created_by_name": names.get(recipe["created_by"]),
                   "finalized_by_name": names.get(recipe["finalized_by"]),
                   "reopened": None if open_ is None else len(open_), "reuse": reuse},
        "assessors": assessors,
        "me": None if mine is None else {"submitted": mine["submitted_at"] is not None,
                                         "reviewed": mine["reviewed"], "total": mine["total"]},
        "review": None if recipe["phase"] == "assessing" else _review_summary(conn, recipe),
    }


@router.get("/datasets/{dataset_id}/team")
def team_state(dataset_id: int, user: CurrentUser) -> dict:
    """The current recipe version, who assesses it and how far along they are (never their choices), the viewer's
    own assessment, and once review is open, how many fields were adopted and how many are still open."""
    with connection() as conn:
        return _team_state(conn, dataset_id, user)


@router.post("/datasets/{dataset_id}/assessors")
def assign(dataset_id: int, body: AssessorIn, user: LeadUser) -> dict:
    with connection() as conn:
        recipe = recipes.current(conn, dataset_id, lock="update")
        _require_phase(recipe, "assessing", doing="assigning assessors")
        ds = conn.execute("SELECT status FROM meta.datasets WHERE id = %s", (dataset_id,)).fetchone()
        if ds["status"] != "ready":
            raise HTTPException(409, "Assign assessors once the dataset is ready")
        who = conn.execute("SELECT id, active FROM meta.users WHERE id = %s", (body.user_id,)).fetchone()
        if who is None:
            raise HTTPException(404, "User not found")
        if not who["active"]:
            raise HTTPException(409, "That user is deactivated")
        conn.execute(
            """INSERT INTO meta.dataset_assessors (recipe_id, user_id, assigned_by) VALUES (%s, %s, %s)
               ON CONFLICT DO NOTHING""",
            (recipe["id"], body.user_id, user["id"]),
        )
        seed(conn, recipe, body.user_id)
        return _team_state(conn, dataset_id, user)


@router.delete("/datasets/{dataset_id}/assessors/{user_id}")
def unassign(dataset_id: int, user_id: int, user: LeadUser) -> dict:
    """Take someone off the assessment. Their choices are kept (and come back if they are assigned again) but don't
    count. If everyone left has submitted, review opens."""
    screen: list[int] = []
    with connection() as conn:
        recipe = recipes.current(conn, dataset_id, lock="update")
        _require_phase(recipe, "assessing", doing="changing assessors")
        gone = conn.execute("DELETE FROM meta.dataset_assessors WHERE recipe_id = %s AND user_id = %s RETURNING 1",
                            (recipe["id"], user_id)).fetchone()
        if gone is None:
            raise HTTPException(404, "That user isn't assigned")
        screen = maybe_open_review(conn, recipe, user["id"])
        state = _team_state(conn, dataset_id, user)
    _after_phase_change(dataset_id, recipe, state, screen)
    return state


def _after_phase_change(dataset_id: int, recipe: dict, state: dict, screen: list[int]) -> None:
    """Once committed: leak-screen new targets, and when review has opened, log the personal experiments' runs that
    waited for it (MLflow can't keep them private)."""
    if screen:
        jobs.submit(screen_targets, dataset_id, screen)
    if state["recipe"]["id"] == recipe["id"] and state["recipe"]["phase"] != "assessing":
        jobs.submit(experiments.flush_deferred, recipe["id"])


@router.post("/datasets/{dataset_id}/assessment/submit")
def submit(dataset_id: int, user: CurrentUser) -> dict:
    screen: list[int] = []
    with connection() as conn:
        recipe = recipes.current(conn, dataset_id, lock="update")
        _require_phase(recipe, "assessing", doing="submitting")
        mine = assessor_row(conn, recipe["id"], user["id"])
        if mine is None:
            raise HTTPException(403, "You aren't assigned to assess this dataset")
        if mine["submitted_at"] is None:
            open_ = recipes.open_field_ids(conn, recipe)
            left = conn.execute(
                """SELECT count(*) AS n FROM meta.fields f
                    WHERE f.dataset_id = %(ds)s AND (%(all)s OR f.id = ANY(%(open)s)) AND NOT EXISTS (
                          SELECT 1 FROM meta.assessments a WHERE a.recipe_id = %(r)s AND a.user_id = %(u)s
                             AND a.field_id = f.id AND a.reviewed)""",
                {"ds": dataset_id, "r": recipe["id"], "u": user["id"], "all": open_ is None, "open": open_ or []},
            ).fetchone()["n"]
            if left:
                raise HTTPException(409, f"Review every field before submitting ({left} left).")
            conn.execute("UPDATE meta.dataset_assessors SET submitted_at = now() WHERE recipe_id = %s AND user_id = %s",
                         (recipe["id"], user["id"]))
            screen = maybe_open_review(conn, recipe, user["id"])
        state = _team_state(conn, dataset_id, user)
    _after_phase_change(dataset_id, recipe, state, screen)
    return state


@router.post("/datasets/{dataset_id}/assessment/unsubmit")
def unsubmit(dataset_id: int, user: CurrentUser) -> dict:
    with connection() as conn:
        recipe = recipes.current(conn, dataset_id, lock="update")
        if recipe["phase"] != "assessing":
            raise HTTPException(409, "Review has opened; assessments can't change any more.")
        if assessor_row(conn, recipe["id"], user["id"]) is None:
            raise HTTPException(403, "You aren't assigned to assess this dataset")
        conn.execute("UPDATE meta.dataset_assessors SET submitted_at = NULL WHERE recipe_id = %s AND user_id = %s",
                     (recipe["id"], user["id"]))
        return _team_state(conn, dataset_id, user)


def maybe_open_review(conn, recipe: dict, actor_id: int) -> list[int]:
    """Open review if there are assessors and all of them have submitted. Returns new targets to leak-screen."""
    rows = conn.execute("SELECT submitted_at FROM meta.dataset_assessors WHERE recipe_id = %s",
                        (recipe["id"],)).fetchall()
    if not rows or any(r["submitted_at"] is None for r in rows):
        return []
    return open_review(conn, recipe, actor_id)


def open_review(conn, recipe: dict, actor_id: int) -> list[int]:
    """Move to review: adopt every field the assessors agree on into the team decision (assessment.adopt) and mark
    the rest unreviewed, which is the review queue. Call holding the recipe row lock."""
    dataset_id = recipe["dataset_id"]
    conn.execute("UPDATE meta.recipes SET phase = 'review' WHERE id = %s", (recipe["id"],))
    team = Source({**recipe, "phase": "review"})
    fam = _active_family(dataset_id)
    by_field = _by_field(conn, recipe)
    open_ = recipes.open_field_ids(conn, recipe)
    keep = None if open_ is None else set(open_)
    fields = [f for f in conn.execute("SELECT * FROM meta.fields WHERE dataset_id = %s ORDER BY ordinal",
                                      (dataset_id,)).fetchall()
              if keep is None or f["id"] in keep]
    before = _merged_many(conn, [f["id"] for f in fields], fam)
    for f in fields:
        stats = {k: f[k] for k in STAT_KEYS}

        def suggestion(t: str, family: str, f=f, stats=stats) -> dict:
            s = (f["suggestion"] or {}).get(family)
            return s if s and s["semantic_type"] == t else suggest(t, stats, f["is_target"], family)

        rows = by_field.get(f["id"], [])
        plan = adopt(rows, suggestion)
        if plan is None:
            conn.execute("UPDATE meta.field_decisions SET reviewed = false, updated_at = now() WHERE field_id = %s",
                         (f["id"],))
            continue
        prev = before[f["id"]]
        notes = "\n".join(f"{r['user_name']}: {r['notes']}" for r in rows if r["notes"]) or None
        decided = core_key(prev) != core_key(plan)
        write_decision(conn, team, f["id"], {"action": plan["action"], "semantic_type": plan["semantic_type"],
                                             "target_task": plan["target_task"], "target_bins": plan["target_bins"],
                                             "notes": notes, "reviewed": True}, decided)
        for family, s in plan["settings"].items():
            same = settings_key(s) == settings_key(suggestion(plan["semantic_type"], family))
            write_settings(conn, team, dataset_id, f["id"], family, s, "suggested" if same else "user")
        if plan["semantic_type"] == "datetime" and not (f["stats"] or {}).get("datetime_format"):
            _remember_datetime_format(conn, f)
    after = _merged_many(conn, [f["id"] for f in fields], fam)
    changed = {fid: row for fid, row in after.items()
               if {k: v for k, v in row.items() if k != "updated_at"}
               != {k: v for k, v in before[fid].items() if k != "updated_at"}}
    _write_history(conn, dataset_id, fam, None, before, changed, actor_id)
    targets_before = {fid for fid, r in before.items() if r["action"] == "target"}
    targets_after = {fid for fid, r in after.items() if r["action"] == "target"}
    if targets_before != targets_after:
        _sync_targets(conn, dataset_id, user_id=actor_id)
        added = sorted(targets_after - targets_before)
        return added if added and has_screen_fields(conn, dataset_id) else []
    return []


@router.get("/datasets/{dataset_id}/review")
def review_queue(dataset_id: int, user: CurrentUser) -> dict:
    """The fields the assessors disagreed on, each with every assessor's choice and notes and the team decision."""
    with connection() as conn:
        recipe = recipes.current(conn, dataset_id)
        _require_phase(recipe, "review", "finalized", doing="team review")
        by_field = _by_field(conn, recipe)
        dis = sorted(disagreements(by_field))
        rows = conn.execute(
            """SELECT f.id, f.name, f.ordinal, f.inferred_type, d.action, d.semantic_type, d.target_task, d.target_bins,
                      d.reviewed,
                      (SELECT count(*) FROM meta.review_comments c
                        WHERE c.recipe_id = %s AND c.field_id = f.id) AS comments
                 FROM meta.fields f JOIN meta.field_decisions d ON d.field_id = f.id
                WHERE f.id = ANY(%s) ORDER BY f.ordinal""",
            (recipe["id"], dis),
        ).fetchall()
        queue = [{
            "field_id": r["id"], "name": r["name"], "inferred_type": r["inferred_type"],
            "team": {k: r[k] for k in ("action", "semantic_type", "target_task", "target_bins")},
            "resolved": r["reviewed"], "comments": r["comments"],
            "assessments": [{k: a[k] for k in ("user_id", "user_name", "action", "semantic_type", "target_task",
                                               "target_bins", "notes")}
                            for a in by_field[r["id"]]],
        } for r in rows]
        return {"recipe": recipe, "summary": _review_summary(conn, recipe), "queue": queue}


@router.get("/fields/{field_id}/review")
def field_review(field_id: int, user: CurrentUser) -> dict:
    """One field in review: every assessor's decision, notes and settings for every family, and the comments."""
    f = _field(field_id)
    with connection() as conn:
        recipe = recipes.current(conn, f["dataset_id"])
        _require_phase(recipe, "review", "finalized", doing="team review")
        rows = _by_field(conn, recipe).get(field_id, [])
        return {
            "disagreement": field_id in disagreements({field_id: rows}),
            "assessments": [{k: a[k] for k in ("user_id", "user_name", "action", "semantic_type", "target_task",
                                               "target_bins", "notes", "settings")}
                            for a in rows],
            "comments": _comments(conn, recipe["id"], field_id),
        }


def _comments(conn, recipe_id: int, field_id: int) -> list[dict]:
    return conn.execute(
        """SELECT c.id, c.user_id, u.name AS user_name, c.body, c.created_at
             FROM meta.review_comments c JOIN meta.users u ON u.id = c.user_id
            WHERE c.recipe_id = %s AND c.field_id = %s ORDER BY c.created_at, c.id""",
        (recipe_id, field_id),
    ).fetchall()


@router.post("/fields/{field_id}/comments", status_code=201)
def add_comment(field_id: int, body: CommentIn, user: CurrentUser) -> list[dict]:
    f = _field(field_id)
    with connection() as conn:
        recipe = recipes.current(conn, f["dataset_id"], lock="share")
        _require_phase(recipe, "review", doing="commenting")
        conn.execute("INSERT INTO meta.review_comments (recipe_id, field_id, user_id, body) VALUES (%s, %s, %s, %s)",
                     (recipe["id"], field_id, user["id"], body.body.strip()))
        return _comments(conn, recipe["id"], field_id)


@router.post("/datasets/{dataset_id}/finalize")
def finalize(dataset_id: int, body: FinalizeIn, user: LeadUser) -> dict:
    """Freeze the team decision as this recipe version's snapshot. Disagreements nobody settled need
    `accept_open`, which keeps their current team decision."""
    with connection() as conn:
        recipe = recipes.current(conn, dataset_id, lock="update")
        _require_phase(recipe, "review", doing="finalizing")
        status = conn.execute("SELECT status FROM meta.datasets WHERE id = %s", (dataset_id,)).fetchone()["status"]
        if status != "ready":
            raise HTTPException(409, "The dataset is still processing")
        open_ = _review_summary(conn, recipe)["open"]
        if open_ and not body.accept_open:
            raise HTTPException(409, {"message": f"{open_} disagreement(s) are still open. Settle them, or confirm "
                                                 "to keep their current team decision.", "open": open_})
        recipes.finalize(conn, recipe, user["id"])
        return _team_state(conn, dataset_id, user)
