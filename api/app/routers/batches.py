"""Batches of runs started together: one model per class of a target, or a search over settings."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import polars as pl
from fastapi import APIRouter, HTTPException

from .. import batches, experiments
from ..auth import CurrentUser
from ..config import get_settings
from ..db import connection, fetch_all, fetch_one, jsonb
from ..modeling import params as model_params
from ..modeling.registry import MODELS, class_target
from ..schemas import BatchRequest, RunRequest, Sampler
from ._lock import refuse_while_refreshing
from .runs import Checked, check_model, check_params, export_manifest, insert_run

router = APIRouter(prefix="/api", tags=["batches"])

BATCH_COLS = """b.id, b.experiment_id, b.export_id, e.dataset_id, b.kind, b.name, b.spec, b.skipped, b.status,
                b.created_by, (SELECT name FROM meta.users u WHERE u.id = b.created_by) AS created_by_name,
                b.created_at, b.finished_at"""
BATCH_FROM = "meta.run_batches b JOIN meta.exports e ON e.id = b.export_id"
STATUSES = ("pending", "running", "done", "failed", "cancelled")
# What a row needs of a run's report; the rest (curves, confusion matrices) would make a 500-run batch megabytes.
SUMMARY_REPORT = """jsonb_build_object('model_label', report -> 'model_label', 'retried', report -> 'retried',
                        'fit_seconds', report -> 'fit_seconds', 'positives', report -> 'positives',
                        'threshold_curve', report -> 'threshold_curve',
                        'cross_validation', jsonb_build_object('targets', report -> 'cross_validation' -> 'targets'))"""


def _text(detail: Any) -> str:
    if isinstance(detail, dict):
        return "; ".join(detail.get("errors") or [str(detail.get("message", detail))])
    return str(detail)


def _class_counts(exp: dict, manifest: dict, target: str) -> batches.ClassCounts:
    """Each class's rows of `target` in each split of the export."""
    split_col = manifest["split_column"]
    labels = manifest["target_classes"][target]
    try:
        df = pl.read_parquet(Path(exp["output_dir"]) / manifest["files"]["data"], columns=[split_col, target])
    except OSError as exc:
        raise HTTPException(410, "Export files were removed") from exc
    valid_rows = int((df[split_col] == "valid").sum())
    counts = {c: {"train": 0, "valid": 0, "test": 0, "valid_rows": valid_rows} for c in labels}
    for split, code, n in df.group_by([split_col, target]).len().iter_rows():
        if split in ("train", "valid", "test") and code is not None and 0 <= code < len(labels):
            counts[labels[code]][split] = n
    return counts


def _search_classes(req: BatchRequest, counts: batches.ClassCounts, min_pos: int) -> list[str]:
    if req.search_classes is None:
        chosen = batches.search_classes_default(counts, min_pos)
        if not chosen:
            raise HTTPException(422, "No class of this target has enough rows to tune on")
        return chosen
    if not req.search_classes:
        raise HTTPException(422, "Choose at least one class to tune on")
    try:
        trained, skipped = batches.plan_classes(counts, req.search_classes, min_pos)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if refused := [s for s in skipped if s["reason"] != "not selected"]:
        raise HTTPException(422, "; ".join(f"{s['class']}: {s['reason']}" for s in refused))
    return trained


def _cap(n: int) -> None:
    most = get_settings().batch_max_runs
    if n > most:
        raise HTTPException(422, f"This batch would start {n:,} runs; at most {most:,} may run in one (BATCH_MAX_RUNS)")


def _default_name(req: BatchRequest, label: str, target: str | None, classes: list) -> str:
    if req.kind == "one_vs_rest":
        return f"{label} · one model per class of {target}"
    name = f"{label} · search over {', '.join(req.space or {})}"
    return f"{name} on {len(classes)} classes of {target}" if target else name


@router.post("/exports/{export_id}/batches", status_code=202)
def start_batch(export_id: int, req: BatchRequest, user: CurrentUser) -> dict:
    """Plan the batch, check every run it would start with a single run's checks, then insert them all, waiting, and
    submit the first. Nothing is inserted when any check fails."""
    exp, manifest = export_manifest(export_id, user)
    with connection() as conn:
        # Checked now, before the planning work, and again where the rows are inserted.
        refuse_while_refreshing(conn, exp["dataset_id"])
        experiments.require_writable(experiments.get_visible(conn, exp["experiment_id"], user), user)
    settings = get_settings()
    base = {"model": req.model, "device": req.device, "seed": req.seed, "cv": req.cv, "cv_folds": req.cv_folds}
    one_vs_rest = req.kind == "one_vs_rest" or req.target is not None or req.search_classes is not None
    target, skipped, redraws = None, [], 0
    classes: list[str | None] = [None]
    if one_vs_rest:
        target, problem = class_target(manifest, req.target)
        if problem:
            raise HTTPException(422, problem)
        counts = _class_counts(exp, manifest, target)
        if req.kind == "one_vs_rest":
            try:
                trained, skipped = batches.plan_classes(counts, req.classes, settings.ovr_min_positives)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            if not trained:
                reasons = "; ".join(f"{s['class']}: {s['reason']}" for s in skipped)
                raise HTTPException(422, f"No class of {target} can be trained ({reasons})")
            classes = list(trained)
        else:
            classes = list(_search_classes(req, counts, settings.ovr_min_positives))

    dry_fits: dict = {}

    def request(cls: str | None, combo: dict) -> RunRequest:
        # balance without a class is refused by the run checks, as for a single run.
        extra = {"positive_class": cls, "target": target} if cls is not None else {}
        return RunRequest(**base, params={**req.params, **combo}, balance=req.balance, **extra)

    first = check_model(exp, manifest, request(classes[0], {}))
    spec_key = first.spec.key

    def problem_of(combo: dict) -> str | None:
        r = request(classes[0], combo)
        try:
            check_params(check_model(exp, manifest, r), manifest, r, dry_fits)
        except HTTPException as exc:
            if exc.status_code != 422:
                raise
            return _text(exc.detail)
        return None

    ranked_target = target or manifest["targets"][0]
    ranking = None
    if req.kind == "search":
        kind = "binary" if one_vs_rest else (manifest.get("target_kinds") or {}).get(ranked_target)
        ranking = req.ranking or batches.default_ranking(kind)
        if problem := batches.ranking_problem(ranking, kind):
            raise HTTPException(422, problem)

    combos: list[dict] = [{}]
    sampler = None
    if req.kind == "search":
        space = req.space or {}
        if (catalogue := model_params.settings_for(spec_key)) is not None:
            names = {s.name for s in catalogue if s.group != "run"}
            if unknown := [k for k in space if k not in names]:
                raise HTTPException(422, f"Not settings of {MODELS[spec_key].label}: {', '.join(unknown)}")
        sampler = req.sampler or Sampler()
        try:
            if sampler.kind == "grid":
                _cap(batches.grid_size(space) * len(classes))
                combos = batches.grid(space)
                for combo in combos:
                    if problem := problem_of(combo):
                        raise HTTPException(422, f"The settings {combo} are refused: {problem}")
            else:
                _cap((sampler.n or 0) * len(classes))
                combos, redraws = batches.random_samples(space, sampler.n or 0, sampler.seed, problem_of)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    _cap(len(classes) * len(combos))

    planned: list[tuple[Checked, RunRequest]] = []
    for combo, cls in itertools.product(combos, classes):
        r = request(cls, combo)
        checked = check_model(exp, manifest, r)
        check_params(checked, manifest, r, dry_fits)
        planned.append((checked, r))

    spec = {
        "model": spec_key,
        "device": req.device,
        "params": req.params,
        "seed": req.seed,
        "cv": req.cv,
        "cv_folds": req.cv_folds,
        "target": target,
        "balance": req.balance,
        **({"classes": classes} if req.kind == "one_vs_rest" else {}),
        **(
            {
                "space": req.space,
                "sampler": sampler.model_dump() if sampler else None,
                "ranking": ranking,
                "ranked_target": ranked_target,
                "search_classes": classes if one_vs_rest else None,
            }
            if req.kind == "search"
            else {}
        ),
    }
    name = (req.name or "").strip() or _default_name(req, first.spec.label, target, classes)
    with connection() as conn:
        refuse_while_refreshing(conn, exp["dataset_id"])
        experiments.require_writable(experiments.get_visible(conn, exp["experiment_id"], user), user)
        batch_id = conn.execute(
            """INSERT INTO meta.run_batches (experiment_id, export_id, kind, name, spec, skipped, created_by)
               VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (exp["experiment_id"], export_id, req.kind, name, jsonb(spec), jsonb(skipped), user["id"]),
        ).fetchone()["id"]
        run_ids = [insert_run(conn, exp, c, r, user["id"], None, batch_id)["id"] for c, r in planned]
    batches.feed(batch_id)
    return {
        "batch": _with_counts([_batch_row(batch_id)])[0],
        "run_ids": run_ids,
        "skipped": skipped,
        "redraws": redraws,
    }


def _batch_row(batch_id: int, user: dict | None = None) -> dict | None:
    visible = "" if user is None else f" AND b.experiment_id IN ({experiments.visible_ids(user['id'])})"
    return fetch_one(f"SELECT {BATCH_COLS} FROM {BATCH_FROM} WHERE b.id = %s{visible}", (batch_id,))


def _with_counts(rows: list[dict]) -> list[dict]:
    """Each batch with its runs' count by status."""
    counts = {r["id"]: dict.fromkeys(STATUSES, 0) for r in rows}
    for c in fetch_all(
        "SELECT batch_id, status, count(*) AS n FROM meta.model_runs WHERE batch_id = ANY(%s) GROUP BY 1, 2",
        (list(counts),),
    ):
        counts[c["batch_id"]][c["status"]] = c["n"]
    return [{**r, "counts": counts[r["id"]]} for r in rows]


def _visible_batch(batch_id: int, user: dict) -> dict:
    row = _batch_row(batch_id, user)
    if row is None:
        raise HTTPException(404, "Batch not found")
    return row


@router.get("/experiments/{experiment_id}/batches")
def list_batches(experiment_id: int, user: CurrentUser) -> list[dict]:
    with connection() as conn:
        experiments.get_visible(conn, experiment_id, user)
    rows = fetch_all(
        f"SELECT {BATCH_COLS} FROM {BATCH_FROM} WHERE b.experiment_id = %s ORDER BY b.id DESC", (experiment_id,)
    )
    return _with_counts(rows)


@router.get("/batches/{batch_id}")
def get_batch(batch_id: int, user: CurrentUser) -> dict:
    """The batch, its runs' counts by status, a row per run, and for a search its leaderboard."""
    batch = _with_counts([_visible_batch(batch_id, user)])[0]
    runs = fetch_all(
        f"""SELECT id, status, status_message, error, config, metrics, {SUMMARY_REPORT} AS report, finished_at
              FROM meta.model_runs WHERE batch_id = %s ORDER BY id""",
        (batch_id,),
    )
    spec = {**batch["spec"], "kind": batch["kind"]}
    rows = [batches.run_summary(r, spec, get_settings().ovr_weak_pr_auc) for r in runs]
    out = {**batch, "rows": rows}
    if batch["kind"] == "search":
        out["leaderboard"] = batches.leaderboard(rows, spec["ranking"])
    return out


@router.post("/batches/{batch_id}/cancel")
def cancel_batch(batch_id: int, user: CurrentUser) -> dict:
    """Cancel the batch's waiting runs; a run already training finishes."""
    batch = _visible_batch(batch_id, user)
    with connection() as conn:
        refuse_while_refreshing(conn, batch["dataset_id"])
        experiments.require_writable(experiments.get_visible(conn, batch["experiment_id"], user), user)
        status = conn.execute("SELECT status FROM meta.run_batches WHERE id = %s FOR UPDATE", (batch_id,)).fetchone()
        if status["status"] != "running":
            raise HTTPException(409, f"The batch is already {status['status']}")
        conn.execute(
            """UPDATE meta.model_runs SET status = 'cancelled', status_message = 'Cancelled', finished_at = now(),
                      updated_at = now() WHERE batch_id = %s AND status = 'pending'""",
            (batch_id,),
        )
        conn.execute("UPDATE meta.run_batches SET status = 'cancelled', finished_at = now() WHERE id = %s", (batch_id,))
    return get_batch(batch_id, user)
