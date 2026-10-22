"""Experiments: create, list, open, rename, annotate and archive. Visibility rules live in app/experiments.py."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import experiments
from ..auth import CurrentUser
from ..db import connection
from ..schemas import ExperimentIn, ExperimentPatch, RevertIn
from .exports import list_exports
from .runs import list_runs

router = APIRouter(prefix="/api", tags=["experiments"])


@router.get("/datasets/{dataset_id}/experiments")
def list_experiments(dataset_id: int, user: CurrentUser) -> list[dict]:
    """The dataset's experiments the viewer may see, archived last, with export and run counts and the best run."""
    with connection() as conn:
        if conn.execute("SELECT 1 FROM meta.datasets WHERE id = %s", (dataset_id,)).fetchone() is None:
            raise HTTPException(404, "Dataset not found")
        return experiments.list_for(conn, dataset_id, user)


@router.post("/datasets/{dataset_id}/experiments", status_code=201)
def create_experiment(dataset_id: int, body: ExperimentIn, user: CurrentUser) -> dict:
    with connection() as conn:
        if conn.execute("SELECT 1 FROM meta.datasets WHERE id = %s", (dataset_id,)).fetchone() is None:
            raise HTTPException(404, "Dataset not found")
        return experiments.create(conn, dataset_id, body.name, body.source, user, body.notes)


@router.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: int, user: CurrentUser) -> dict:
    """One experiment with its exports and runs, newest first."""
    with connection() as conn:
        x = experiments.get_visible(conn, experiment_id, user)
        best = experiments.best_by_experiment(conn, [x["id"]]).get(x["id"])
        adjustments = experiments.adjustments(conn, x)
    return {
        **x,
        "best": best,
        "adjustments": adjustments,
        "can_adjust": experiments.can_adjust(x, user),
        "exports": list_exports(x["dataset_id"], user, x["id"]),
        "runs": list_runs(x["dataset_id"], user, x["id"]),
    }


@router.post("/experiments/{experiment_id}/adjustments/revert")
def revert_adjustments(experiment_id: int, body: RevertIn, user: CurrentUser) -> dict:
    """Put fields of a team experiment (all of them when `field_ids` is null) back to its baseline: its owner."""
    with connection() as conn:
        x = experiments.get_visible(conn, experiment_id, user)
        x = conn.execute(
            f"SELECT {experiments.COLS} FROM {experiments.FROM} WHERE x.id = %s FOR UPDATE OF x", (x["id"],)
        ).fetchone()
        if (denied := experiments.adjust_denied(x, user)) is not None:
            raise HTTPException(*denied)
        experiments.revert(conn, x, body.field_ids)
        x = experiments.get(conn, x["id"])
        return {**x, "adjustments": experiments.adjustments(conn, x), "can_adjust": True}


@router.patch("/experiments/{experiment_id}")
def patch_experiment(experiment_id: int, body: ExperimentPatch, user: CurrentUser) -> dict:
    """Rename, change the notes, archive or unarchive: the owner or a lead."""
    with connection() as conn:
        x = experiments.get_visible(conn, experiment_id, user)
        experiments.require_editable(x, user)
        sets = body.model_dump(exclude_unset=True)
        if "name" in sets and sets["name"] is not None:
            conn.execute("UPDATE meta.experiments SET name = %s WHERE id = %s", (sets["name"].strip(), x["id"]))
        if "notes" in sets:
            conn.execute("UPDATE meta.experiments SET notes = %s WHERE id = %s", (sets["notes"] or None, x["id"]))
        if sets.get("archived") is not None:
            conn.execute(
                "UPDATE meta.experiments SET archived_at = CASE WHEN %s THEN coalesce(archived_at, now()) END "
                "WHERE id = %s",
                (sets["archived"], x["id"]),
            )
        return experiments.get(conn, x["id"])
