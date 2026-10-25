from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Callable

import polars as pl
import pyarrow.parquet as pq
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from psycopg import sql

from .. import experiments, jobs, recipes, reuse
from ..assessment import source_for
from ..auth import CurrentUser, LeadUser
from ..config import get_settings
from ..correlations import target_associations
from ..db import connection, execute, fetch_all, fetch_one, jsonb
from ..modeling.registry import MODELS
from ..processing import _Status, delete_dataset_files, process_dataset, upsert_correlations
from ..profiling import corr_vector, sample_indices
from ..sample_data import generate
from ..sample_library import LIBRARY
from ..sample_library import path_of as library_path
from ..schemas import ConnectRequest, DatasetPatch, SampleRequest
from ..semantics import NUMERIC
from ..sources import SourceError
from ..sources.pull import RANK, run_connect
from .fields import _sync_targets, has_screen_fields, screen_pairs, screen_targets, set_targets
from .sources import _source, collection_specs

log = logging.getLogger(__name__)
CORRELATED_TYPES = NUMERIC | {"datetime", "categorical", "binary"}
router = APIRouter(prefix="/api/datasets", tags=["datasets"])

DATASET_COLS = """id, name, source_filename, file_format, target_columns, model_family, row_count, column_count, status,
                  progress, status_message, error, warnings, created_at, updated_at, profiled_at, raw_table, created_by"""
# A dataset `d` with its current recipe version `rc` (recipes.CURRENT_JOIN).
SELECT_DATASET = (
    ", ".join("d." + c.strip() for c in DATASET_COLS.split(","))
    + ", rc.id AS recipe_id, rc.version AS recipe_version, rc.phase"
    # A connected dataset's source and newest pull, as JSON for `_with_source` to shape; both null for an upload.
    + """, (SELECT to_jsonb(s) FROM meta.dataset_sources s WHERE s.dataset_id = d.id) AS source,
       (SELECT jsonb_build_object('id', p.id, 'status', p.status, 'started_at', p.started_at,
                                  'finished_at', p.finished_at, 'rows', p.rows, 'has_drift', p.drift IS NOT NULL)
          FROM meta.dataset_pulls p WHERE p.dataset_id = d.id ORDER BY p.id DESC LIMIT 1) AS last_pull"""
)


def _targets(raw: str | None) -> list[str]:
    # Targets are picked from the real columns after upload; none is a valid starting point.
    if raw is None:
        return []
    return list(dict.fromkeys(t.strip() for t in raw.split(",") if t.strip()))


def _with_source(row: dict) -> dict:
    """`source` as the API shows it: the stored row without who made it, and each collection's filter and projection as
    the JSON text the dialog edits."""
    src = row.get("source")
    if src is not None:

        def text(doc: dict | None) -> str | None:
            return None if doc is None else json.dumps(doc)

        row["source"] = {
            "kind": src["kind"],
            "connection": src["connection"],
            "database": src["database"],
            "collections": [
                {
                    "name": c["name"],
                    "label": c.get("label") or c["name"],
                    "filter": text(c.get("filter")),
                    "projection": text(c.get("projection")),
                }
                for c in src["collections"]
            ],
            "add_source": src["add_source"],
            "dedupe": src["dedupe"],
            "match_on": src["match_on"],
        }
    return row


def _create(
    name: str,
    filename: str | None,
    fmt: str,
    targets: list[str],
    user_id: int,
    upload_path: Path | None = None,
    recipe_id: int | None = None,
) -> dict:
    with connection() as conn:
        ds = conn.execute(
            """INSERT INTO meta.datasets (name, source_filename, file_format, target_columns, status, status_message,
                                          owner, upload_path, created_by)
               VALUES (%s, %s, %s, %s, 'pending', 'Queued', %s, %s, %s) RETURNING id""",
            (name, filename, fmt, targets, jobs.owner_id(), None if upload_path is None else str(upload_path), user_id),
        ).fetchone()
        first = recipes.create_first(conn, ds["id"], user_id)
        if recipe_id is not None:
            # Processing applies it (reuse.apply) before the dataset is ready.
            conn.execute("UPDATE meta.recipes SET parent_id = %s WHERE id = %s", (recipe_id, first["id"]))
    return get_dataset(ds["id"])


# Metric shown for a run in the dataset list, in order of preference.


@router.get("")
def list_datasets(user: CurrentUser) -> list[dict]:
    visible = experiments.visible_ids(user["id"])
    rows = fetch_all(
        f"""SELECT {SELECT_DATASET}, (a.user_id IS NOT NULL) AS assigned, a.submitted_at AS my_submitted_at,
                   (SELECT count(*) FROM meta.assessments x
                     WHERE x.recipe_id = rc.id AND x.user_id = a.user_id AND x.reviewed
                       AND (rc.scope = 'all' OR x.field_id IN (SELECT field_id FROM meta.recipe_reopened o
                                                                WHERE o.recipe_id = rc.id))) AS my_reviewed,
                   CASE WHEN rc.scope = 'reopened'
                        THEN (SELECT count(*) FROM meta.recipe_reopened o WHERE o.recipe_id = rc.id) END AS reopened,
                   (SELECT count(*) FROM meta.dataset_assessors o WHERE o.recipe_id = rc.id) AS my_assessors_total,
                   (SELECT count(*) FROM meta.dataset_assessors o
                     WHERE o.recipe_id = rc.id AND o.submitted_at IS NOT NULL) AS my_assessors_submitted,
                   coalesce(c.field_count, 0) AS field_count, coalesce(c.reviewed_count, 0) AS reviewed_count,
                   coalesce(c.kept_count, 0) AS kept_count, coalesce(c.dropped_count, 0) AS dropped_count,
                   coalesce(c.target_count, 0) AS target_count,
                   (SELECT count(*) FROM meta.exports x WHERE x.dataset_id = d.id
                       AND x.experiment_id IN ({visible})) AS export_count,
                   (SELECT count(*) FROM meta.model_runs x WHERE x.dataset_id = d.id
                       AND x.experiment_id IN ({visible})) AS run_count,
                   e.id AS e_id, e.model_family AS e_model_family, e.status AS e_status,
                   e.created_at AS e_created_at, e.n_features AS e_n_features,
                   r.id AS r_id, r.model AS r_model, r.status AS r_status, r.report -> 'order' AS r_order,
                   r.report ->> 'model_label' AS r_label, r.metrics -> 'test' AS r_test
              FROM meta.datasets d
              {recipes.CURRENT_JOIN}
              LEFT JOIN meta.dataset_assessors a ON a.recipe_id = rc.id AND a.user_id = %(user)s
              LEFT JOIN (SELECT dataset_id, count(*) AS field_count,
                                count(*) FILTER (WHERE reviewed) AS reviewed_count,
                                count(*) FILTER (WHERE action = 'keep') AS kept_count,
                                count(*) FILTER (WHERE action = 'drop') AS dropped_count,
                                count(*) FILTER (WHERE action = 'target') AS target_count
                           FROM meta.field_decisions GROUP BY dataset_id) c ON c.dataset_id = d.id
              LEFT JOIN LATERAL (SELECT * FROM meta.exports x WHERE x.dataset_id = d.id
                                    AND x.experiment_id IN ({visible})
                                  ORDER BY x.created_at DESC, x.id DESC LIMIT 1) e ON true
              LEFT JOIN LATERAL (SELECT * FROM meta.model_runs x WHERE x.dataset_id = d.id
                                    AND x.experiment_id IN ({visible})
                                  ORDER BY x.created_at DESC, x.id DESC LIMIT 1) r ON true
             ORDER BY d.created_at DESC""",
        {"user": user["id"]},
    )
    best = best_by_dataset([r["id"] for r in rows], visible)
    out = with_baselines([{**_with_latest(row), "best_run": best.get(row["id"])} for row in rows])
    with connection() as conn:
        for row in out:
            row.update(_team_glance(conn, row, user))
    return out


TASK_KIND = {"classes": "classes", "regression": "number", "bins": "bands"}


def _team_glance(conn, row: dict, user: dict) -> dict:
    """For the datasets list: how each target is predicted as the viewer may read it (`target_kinds`, with
    `target_kinds_suggested` when that reading is the suggestion nobody has decided yet: a version being assessed,
    read by someone who isn't assessing it) and, in team review, how many disagreements are still open."""
    from .team import _review_summary

    out = {"target_kinds": {}, "target_kinds_suggested": False, "review_open": None}
    if row["recipe_id"] is None or row["status"] != "ready":
        return out
    recipe = recipes.get(conn, row["recipe_id"])
    if row["phase"] == "review":
        out["review_open"] = _review_summary(conn, recipe)["open"]
    if row["target_columns"]:
        src = source_for(conn, row["id"], user, recipe)
        kinds = conn.execute(
            sql.SQL("""SELECT f.name, d.target_task FROM meta.fields f JOIN {} d ON d.field_id = f.id
                        WHERE f.dataset_id = %s AND f.name = ANY(%s) AND d.action = 'target'""").format(
                src.decisions()
            ),
            (row["id"], row["target_columns"]),
        ).fetchall()
        out["target_kinds"] = {k["name"]: TASK_KIND.get(k["target_task"] or "classes") for k in kinds}
        out["target_kinds_suggested"] = src.kind == "suggested" or (src.team and recipe["phase"] == "assessing")
    return out


def best_by_dataset(ids: list[int], visible: str) -> dict[int, dict]:
    """Per dataset, the run that stands for it: the chosen model (the most recently chosen, when several experiments
    have one), else the best finished run on the validation split by the same rule as `experiments.best_by_experiment`.
    Only runs in experiments the viewer may see."""
    if not ids:
        return {}
    runs = fetch_all(
        f"""SELECT r.id, r.dataset_id, r.experiment_id, r.model, r.report ->> 'model_label' AS label,
                   r.report -> 'order' AS run_order, r.metrics -> 'valid' AS valid, r.chosen_at,
                   x.name AS experiment_name
              FROM meta.model_runs r JOIN meta.experiments x ON x.id = r.experiment_id
             WHERE r.dataset_id = ANY(%s) AND r.status = 'done' AND r.experiment_id IN ({visible})""",
        (ids,),
    )
    by_ds: dict[int, list[dict]] = {}
    for r in runs:
        order, valid = r["run_order"] or [], r["valid"] or {}
        scores = valid.get(order[0]) if order else None
        by_ds.setdefault(r["dataset_id"], []).append({**r, "scores": scores if isinstance(scores, dict) else {}})

    def entry(r: dict, chosen: bool) -> dict:
        metric = next((m for m in experiments.HEADLINE_METRICS if isinstance(r["scores"].get(m), (int, float))), None)
        return {
            "run_id": r["id"],
            "experiment_id": r["experiment_id"],
            "experiment_name": r["experiment_name"],
            "model": r["model"],
            "model_label": r["label"] or (MODELS[r["model"]].label if r["model"] in MODELS else r["model"]),
            "chosen": chosen,
            "metric": metric,
            "value": None if metric is None else float(r["scores"][metric]),
        }

    out: dict[int, dict] = {}
    for ds_id, scored in by_ds.items():
        chosen = [r for r in scored if r["chosen_at"] is not None]
        if chosen:
            out[ds_id] = entry(max(chosen, key=lambda r: r["chosen_at"]), True)
            continue
        for metric in experiments.HEADLINE_METRICS:
            have = [r for r in scored if isinstance(r["scores"].get(metric), (int, float))]
            if have:
                out[ds_id] = entry(max(have, key=lambda r: r["scores"][metric]), False)
                break
    return out


def _with_latest(row: dict) -> dict:
    out = {k: v for k, v in row.items() if not k.startswith(("e_", "r_", "my_")) and k != "assigned"}
    _with_source(out)
    # Who assesses the current version and how many have submitted (counts only; the team sees these anyway).
    out["assessors"] = (
        None
        if not row["my_assessors_total"]
        else {"total": row["my_assessors_total"], "submitted": row["my_assessors_submitted"]}
    )
    # The viewer's own assessment of the current version, when they are assigned to it.
    out["my_assessment"] = (
        None
        if not row["assigned"]
        else {"reviewed": row["my_reviewed"], "submitted": row["my_submitted_at"] is not None}
    )
    out["last_export"] = (
        None
        if row["e_id"] is None
        else {
            "id": row["e_id"],
            "model_family": row["e_model_family"],
            "status": row["e_status"],
            "created_at": row["e_created_at"],
            "n_features": row["e_n_features"],
        }
    )
    out["last_run"] = (
        None
        if row["r_id"] is None
        else {
            "id": row["r_id"],
            "model": row["r_model"],
            "model_label": row["r_label"]
            or (MODELS[row["r_model"]].label if row["r_model"] in MODELS else row["r_model"]),
            "status": row["r_status"],
            **headline_metric(row["r_status"], row["r_order"], row["r_test"]),
        }
    )
    return out


def headline_metric(status: str, order: list[str] | None, test: dict | None) -> dict:
    """Test-split metric for the first target in chain order, so one number can stand for a run."""
    if status == "done" and order and test:
        by_metric = test.get(order[0]) or {}
        for m in experiments.HEADLINE_METRICS:
            if isinstance(by_metric.get(m), (int, float)):
                return {"metric": m, "value": float(by_metric[m])}
    return {"metric": None, "value": None}


@router.post("/upload", status_code=202)
def upload(
    user: CurrentUser,
    file: UploadFile = File(...),
    name: str | None = Form(None),
    targets: str | None = Form(None, description="Comma-separated target columns"),
    recipe_id: int | None = Form(None, description="Start from this finalized recipe once processed (leads only)"),
) -> dict:
    if recipe_id is not None:
        if user["role"] != "lead":
            raise HTTPException(403, "Only a lead can start a dataset from a recipe.")
        with connection() as conn:
            reuse.check_source(conn, recipe_id)
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()
    fmt = {".csv": "csv", ".txt": "csv", ".tsv": "csv", ".parquet": "parquet", ".pq": "parquet"}.get(suffix)
    if fmt is None:
        raise HTTPException(415, "Only .csv, .tsv, .txt and .parquet files are supported")
    s = get_settings()
    s.uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = s.uploads_dir / f"{uuid.uuid4().hex}{suffix}"
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out, length=8 * 1024 * 1024)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    if dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "Uploaded file is empty")
    try:
        ds = _create(name or Path(filename).stem, filename, fmt, _targets(targets), user["id"], dest, recipe_id)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    jobs.submit(process_dataset, ds["id"], str(dest), fmt)
    return ds


@router.post("/connect", status_code=202)
def connect(body: ConnectRequest, user: CurrentUser) -> dict:
    """Create a dataset from collections of a configured MongoDB connection. Everything that can be checked without
    a pull is checked before a dataset exists: a refused request leaves nothing behind."""
    src = body.source
    source = _source(src.connection)
    labels = [c.label.strip() if c.label and c.label.strip() else c.name for c in src.collections]
    for label in labels:
        if label == RANK:
            raise HTTPException(422, f"Collection label is reserved: {label}")
        if labels.count(label) > 1:
            raise HTTPException(422, f"Collection labels must be unique: {label}")
    specs = collection_specs(src.collections)
    # A label is how the pull files its chunks and what the `source` column holds.
    for spec, label in zip(specs, labels):
        spec.label = label
    several = len(specs) > 1
    if several and src.add_source is False:
        raise HTTPException(422, "Several collections need the source column to tell their rows apart")
    add_source = several if src.add_source is None else src.add_source
    dedupe = several if src.dedupe is None else src.dedupe
    if body.recipe_id is not None:
        if user["role"] != "lead":
            raise HTTPException(403, "Only a lead can start a dataset from a recipe.")
        with connection() as conn:
            reuse.check_source(conn, body.recipe_id)
    try:
        estimate = source.preview(src.database, specs)
    except SourceError as exc:
        raise HTTPException(502, str(exc)) from None
    max_cells = get_settings().source_max_cells
    if estimate.estimated_cells > max_cells:
        raise HTTPException(422, {"message": "Too large", "estimated": estimate.estimated_cells, "max": max_cells})

    def stored(text: str | None) -> dict | None:
        # Already validated as a document; kept as the extended JSON the client sent, which `specs_from` re-parses.
        return json.loads(text) if text and text.strip() else None

    collections = [
        {"name": c.name, "label": label, "filter": stored(c.filter), "projection": stored(c.projection)}
        for c, label in zip(src.collections, labels)
    ]
    match_on = list(dict.fromkeys(m.strip() for m in src.match_on or [] if m.strip())) or None
    targets = list(dict.fromkeys(t.strip() for t in body.targets if t.strip()))
    ds = _create(body.name, None, "mongodb", targets, user["id"], None, body.recipe_id)
    try:
        execute(
            """INSERT INTO meta.dataset_sources
                   (dataset_id, kind, connection, database, collections, add_source, dedupe, match_on, created_by)
               VALUES (%s, 'mongodb', %s, %s, %s, %s, %s, %s, %s)""",
            (ds["id"], src.connection, src.database, jsonb(collections), add_source, dedupe, match_on, user["id"]),
        )
    except Exception:
        execute("DELETE FROM meta.datasets WHERE id = %s", (ds["id"],))  # a dataset with no source never pulls
        raise
    jobs.submit(run_connect, ds["id"])
    return get_dataset(ds["id"])


@router.get("/sample/limits")
def sample_limits() -> dict:
    return {"max_cells": get_settings().sample_max_cells}


@router.get("/library")
def library(_: CurrentUser) -> list[dict]:
    """The real datasets that ship with the app, with their size, suggested targets, source and citation."""
    out = []
    for key, entry in LIBRARY.items():
        meta = pq.ParquetFile(library_path(key)).metadata
        out.append({"key": key, **entry, "rows": meta.num_rows, "columns": meta.num_columns})
    return out


@router.post("/library/{key}", status_code=202)
def add_from_library(key: str, user: CurrentUser) -> dict:
    """Add a library dataset: a copy of its file goes through the same processing as an upload, with its suggested
    targets set."""
    entry = LIBRARY.get(key)
    if entry is None:
        raise HTTPException(404, f"No library dataset {key!r}")
    s = get_settings()
    s.uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = s.uploads_dir / f"{uuid.uuid4().hex}.parquet"
    shutil.copyfile(library_path(key), dest)
    try:
        ds = _create(entry["name"], f"{key}.parquet", "parquet", list(entry["targets"]), user["id"], dest)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    jobs.submit(process_dataset, ds["id"], str(dest), "parquet")
    return ds


@router.post("/sample", status_code=202)
def create_sample(req: SampleRequest, user: CurrentUser) -> dict:
    cap = get_settings().sample_max_cells
    width = req.fields + int(req.numeric_target)
    if req.rows * width > cap:
        raise HTTPException(
            422,
            f"rows × fields must be at most {cap:,} (asked for {req.rows:,} × {width} = "
            f"{req.rows * width:,}). With {width} fields, use up to {cap // width:,} rows.",
        )
    ds = _create(
        req.name
        or f"sample-{req.rows // 1000}k-{req.fields}f-{req.target_kind}" + ("-num" if req.numeric_target else ""),
        None,
        "generated",
        ["column_a", "column_b", *(["column_c"] if req.numeric_target else [])],
        user["id"],
    )
    jobs.submit(_generate_and_process, ds["id"], req.model_dump())
    return ds


def _generate_and_process(dataset_id: int, req: dict) -> None:
    s = get_settings()
    out = s.datasets_dir / str(dataset_id) / "raw.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        execute(
            "UPDATE meta.datasets SET status='ingesting', error=NULL, status_message='Generating sample data', "
            "updated_at=now() WHERE id=%s",
            (dataset_id,),
        )
        df = generate(req["rows"], req["fields"], req["seed"], req["target_kind"], req.get("numeric_target", False))
        df.write_parquet(out, row_group_size=100_000)  # as ingest writes uploads
    except Exception as exc:  # noqa: BLE001
        execute(
            "UPDATE meta.datasets SET status='failed', error=%s, updated_at=now() WHERE id=%s",
            (f"sample generation failed: {exc}", dataset_id),
        )
        return
    process_dataset(dataset_id, None, "parquet")


@router.get("/{dataset_id}")
def get_dataset(dataset_id: int) -> dict:
    ds = fetch_one(
        f"SELECT {SELECT_DATASET} FROM meta.datasets d {recipes.CURRENT_JOIN} WHERE d.id = %s", (dataset_id,)
    )
    if ds is None:
        raise HTTPException(404, "Dataset not found")
    return with_baselines([_with_source(ds)])[0]


@router.get("/{dataset_id}/pulls")
def list_pulls(dataset_id: int, user: CurrentUser) -> list[dict]:
    if fetch_one("SELECT 1 FROM meta.datasets WHERE id = %s", (dataset_id,)) is None:
        raise HTTPException(404, "Dataset not found")
    return fetch_all(
        """SELECT p.id, p.status, p.started_at, p.finished_at, p.rows, p.per_source, p.flatten_notes, p.drift,
                  p.error, u.name AS created_by_name
             FROM meta.dataset_pulls p LEFT JOIN meta.users u ON u.id = p.created_by
            WHERE p.dataset_id = %s ORDER BY p.id DESC""",
        (dataset_id,),
    )


def with_baselines(rows: list[dict]) -> list[dict]:
    """Each dataset with `baseline` (its current version: id, family, version, phase, label, and `setup` while it is a
    new dataset's first baseline nobody has started on) and `baselines` (per family that has one: the latest version
    and the latest finalized version's id), Trees first."""
    ids = [r["id"] for r in rows]
    if not ids:
        return rows
    with connection() as conn:
        latest = conn.execute(
            """SELECT DISTINCT ON (r.dataset_id, r.family) r.dataset_id, r.family, r.id, r.version, r.phase,
                      meta.baseline_name(r.dataset_id, r.family) AS name,
                      (SELECT f.id FROM meta.recipes f WHERE f.dataset_id = r.dataset_id AND f.family = r.family
                          AND f.phase = 'finalized' ORDER BY f.finalized_at DESC, f.id DESC LIMIT 1)
                          AS latest_finalized_id
                 FROM meta.recipes r WHERE r.dataset_id = ANY(%s)
                ORDER BY r.dataset_id, r.family, r.id DESC""",
            (ids,),
        ).fetchall()
        current = {
            r["dataset_id"]: r
            for r in conn.execute(
                """SELECT DISTINCT ON (r.dataset_id) r.dataset_id, r.id, r.family, r.version, r.phase,
                      meta.baseline_name(r.dataset_id, r.family) AS name,
                      (r.version = 1 AND r.phase = 'assessing' AND r.parent_id IS NULL AND r.reuse IS NULL
                       AND NOT EXISTS (SELECT 1 FROM meta.dataset_assessors a WHERE a.recipe_id = r.id)
                       AND NOT EXISTS (SELECT 1 FROM meta.assessments a WHERE a.recipe_id = r.id)
                       AND NOT EXISTS (SELECT 1 FROM meta.recipes o WHERE o.dataset_id = r.dataset_id
                                         AND o.id <> r.id)) AS setup
                 FROM meta.recipes r WHERE r.dataset_id = ANY(%s) ORDER BY r.dataset_id, r.id DESC""",
                (ids,),
            ).fetchall()
        }
    order = list(recipes.FAMILY_LABEL)
    by_ds: dict[int, list[dict]] = {}
    for r in sorted(latest, key=lambda r: order.index(r["family"])):
        by_ds.setdefault(r["dataset_id"], []).append(
            {k: r[k] for k in ("family", "id", "version", "phase", "latest_finalized_id", "name")}
            | {"label": recipes.label(r)}
        )
    out = []
    for row in rows:
        cur = current.get(row["id"])
        out.append(
            {
                **row,
                "baselines": by_ds.get(row["id"], []),
                "baseline": None
                if cur is None
                else {
                    **{k: cur[k] for k in ("id", "family", "version", "phase", "setup", "name")},
                    "label": recipes.label(cur),
                },
            }
        )
    return out


@router.patch("/{dataset_id}")
def update_dataset(dataset_id: int, body: DatasetPatch, user: CurrentUser) -> dict:
    """Rename a dataset or choose its targets. Its model family is its current baseline's."""
    ds = get_dataset(dataset_id)
    if body.model_family is not None and body.model_family != ds["model_family"]:
        raise HTTPException(
            422, "The model type comes from the baseline. Start a baseline for another type on the Baselines page."
        )
    if body.target_columns is not None and ds["status"] != "ready":
        raise HTTPException(409, "Targets can be changed once the dataset is ready")
    start = _nothing
    with connection() as conn:
        if body.target_columns is not None:
            _check_targets_editable(conn, dataset_id, user, list(dict.fromkeys(body.target_columns)))
        conn.execute(
            "UPDATE meta.datasets SET name = coalesce(%s, name), updated_at = now() WHERE id = %s",
            (body.name, dataset_id),
        )
        if body.target_columns is not None:
            uncorrelated, added = set_targets(conn, dataset_id, list(dict.fromkeys(body.target_columns)), user["id"])
            start = queue_target_work(conn, dataset_id, uncorrelated, added)
    start()
    return get_dataset(dataset_id)


def _nothing() -> None:
    pass


def queue_target_work(
    conn, dataset_id: int, uncorrelated: list[dict], added: list[int], inline: bool = False
) -> Callable[[], None]:
    """After targets changed (fields.set_targets): the work still to do, to call once the transaction commits.
    Targets with no associations yet get them computed (the dataset shows as correlating meanwhile), new targets get
    the leak screen. `inline` runs it in the calling job instead of queueing a job."""
    screen = added if added and has_screen_fields(conn, dataset_id) else []
    conn.execute(
        """UPDATE meta.datasets SET warnings = array(
               SELECT w FROM unnest(warnings) w WHERE w NOT LIKE 'Target column(s) not found%%')
            WHERE id = %s""",
        (dataset_id,),
    )
    run = (lambda fn, *args: fn(*args)) if inline else jobs.submit
    if uncorrelated:
        if not inline:
            conn.execute(
                """UPDATE meta.datasets SET status = 'correlating', progress = 0, error = NULL, owner = %s,
                          status_message = 'Queued: associations for new target(s)' WHERE id = %s""",
                (jobs.owner_id(), dataset_id),
            )
        ids = [r["id"] for r in uncorrelated]
        return lambda: run(correlate_targets, dataset_id, ids, screen)
    if screen:
        return lambda: run(screen_targets, dataset_id, screen)
    return _nothing


def _check_targets_editable(conn, dataset_id: int, user: dict, names: list[str]) -> None:
    """Targets are part of the team decision. They are set before anyone assesses (every assessor starts from
    them); after that only a lead changes them, in review. A finalized version is frozen, and fields carried over
    into a later version keep their role."""
    recipe = recipes.current(conn, dataset_id, lock="update")
    if recipe["phase"] == "assessing":
        if conn.execute(
            "SELECT 1 FROM meta.dataset_assessors WHERE recipe_id = %s LIMIT 1", (recipe["id"],)
        ).fetchone():
            raise HTTPException(
                409,
                "Targets are fixed once assessors are assigned. A lead can change them in "
                "review, or unassign everyone first.",
            )
    else:
        source_for(conn, dataset_id, user, recipe).require_write()
    changed = [
        r["id"]
        for r in conn.execute(
            """SELECT f.id FROM meta.fields f JOIN meta.field_decisions d ON d.field_id = f.id
            WHERE f.dataset_id = %s AND (d.action = 'target') <> (f.name = ANY(%s))""",
            (dataset_id, names),
        ).fetchall()
    ]
    if (locked := recipes.locked_problem(conn, recipe, changed)) is not None:
        raise HTTPException(409, locked)


def correlate_targets(dataset_id: int, target_ids: list[int], screen_ids: list[int] | None = None) -> None:
    """Associations between new targets and every other field, for targets profiled as types that never got
    an association sample (text, identifier). Treated as class labels, as profiling treats targets.
    `screen_ids` (new targets) also get the uncapped leak screen."""
    s = get_settings()
    st = _Status(dataset_id)
    try:
        st.set("correlating", 0.0, "Computing associations for new target(s)", force=True)
        ds = fetch_one("SELECT parquet_path, row_count FROM meta.datasets WHERE id = %s", (dataset_id,))
        fields = fetch_all(
            """SELECT id, name, inferred_type, stats ->> 'datetime_format' AS datetime_format
                 FROM meta.fields WHERE dataset_id = %s ORDER BY ordinal""",
            (dataset_id,),
        )
        path = Path(ds["parquet_path"])
        corr_idx = sample_indices(ds["row_count"], s.corr_sample_rows, seed=29)
        targets, numeric, categorical = [], [], []
        for k, f in enumerate(fields):
            is_new = f["id"] in target_ids
            if not is_new and f["inferred_type"] not in CORRELATED_TYPES:
                continue
            col = pl.read_parquet(path, columns=[f["name"]]).to_series()
            t = "categorical" if is_new else f["inferred_type"]
            num, codes = corr_vector(col, t, f["datetime_format"], corr_idx, s.corr_max_levels)
            if is_new:
                targets.append((k, codes))
            elif num is not None:
                numeric.append((k, num))
            elif codes is not None:
                categorical.append((k, codes))
            st.set("correlating", 0.8 * (k + 1) / len(fields), f"Reading {f['name']} ({k + 1}/{len(fields)})")
        assocs = list(target_associations(targets, numeric, categorical))
        screened = screen_pairs(dataset_id, screen_ids) if screen_ids else []
        with connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO meta.field_correlations
                              (dataset_id, field_a, field_b, method, value, strength, n_obs)
                       VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (field_a, field_b) DO NOTHING""",
                    [
                        (
                            dataset_id,
                            *sorted((fields[a.i]["id"], fields[a.j]["id"])),
                            a.method,
                            a.value,
                            abs(a.value),
                            a.n_obs,
                        )
                        for a in assocs
                    ],
                )
            upsert_correlations(conn, dataset_id, screened)
            _sync_targets(conn, dataset_id)
            conn.execute(
                """UPDATE meta.datasets SET warnings = array(
                       SELECT w FROM unnest(warnings) w WHERE w NOT LIKE 'Associations for the new target(s)%%')
                    WHERE id = %s""",
                (dataset_id,),
            )
        st.set("ready", 1.0, "Ready", force=True)
    except Exception as exc:  # noqa: BLE001
        log.exception("target associations for dataset %s failed", dataset_id)
        # The dataset itself is intact; only the new target's associations are missing.
        execute(
            """UPDATE meta.datasets SET status = 'ready', progress = 1, status_message = 'Ready',
                      warnings = array_append(warnings, %s), updated_at = now() WHERE id = %s""",
            (f"Associations for the new target(s) could not be computed: {exc}", dataset_id),
        )


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int, _: LeadUser) -> dict:
    """Delete a dataset with its fields, exports and model runs, including their files on disk."""
    s = get_settings()
    with connection() as conn:
        ds = conn.execute("SELECT status FROM meta.datasets WHERE id = %s FOR UPDATE", (dataset_id,)).fetchone()
        if ds is None:
            raise HTTPException(404, "Dataset not found")
        if ds["status"] in ("pending", "ingesting", "profiling", "correlating"):
            raise HTTPException(409, "Dataset is still processing")
        busy = conn.execute(
            """SELECT EXISTS (SELECT 1 FROM meta.exports
                               WHERE dataset_id = %(ds)s AND status IN ('pending', 'running'))
                   OR EXISTS (SELECT 1 FROM meta.model_runs
                               WHERE dataset_id = %(ds)s AND status IN ('pending', 'running')) AS busy""",
            {"ds": dataset_id},
        ).fetchone()["busy"]
        if busy:
            raise HTTPException(409, "An export or model run for this dataset is still running")
        exports = [
            r["id"] for r in conn.execute("SELECT id FROM meta.exports WHERE dataset_id = %s", (dataset_id,)).fetchall()
        ]
        runs = conn.execute("SELECT id, model_dir FROM meta.model_runs WHERE dataset_id = %s", (dataset_id,)).fetchall()
        n_fields = conn.execute(
            "SELECT count(*) AS n FROM meta.fields WHERE dataset_id = %s", (dataset_id,)
        ).fetchone()["n"]
        conn.execute("DELETE FROM meta.datasets WHERE id = %s", (dataset_id,))
    delete_dataset_files(dataset_id)
    for e in exports:
        shutil.rmtree(s.exports_dir / str(e), ignore_errors=True)
    for path in _run_dirs(s.models_dir, runs):
        shutil.rmtree(path, ignore_errors=True)
    return {"fields": n_fields, "exports": len(exports), "runs": len(runs)}


def _run_dirs(models_dir: Path, runs: list[dict]) -> set[Path]:
    """Every directory a run may have written, including failed runs that never recorded model_dir."""
    root = models_dir.resolve()
    out: set[Path] = set()
    for r in runs:
        out.update(p.resolve() for p in models_dir.glob(f"run{r['id']}_*") if p.is_dir())
        if r["model_dir"]:
            p = Path(r["model_dir"]).resolve()
            if p.parent == root:
                out.add(p)
    return out


@router.get("/{dataset_id}/fields")
def list_fields(dataset_id: int, user: CurrentUser, view: str | None = None) -> list[dict]:
    """Lightweight rows for the review sidebar, with the decisions the viewer works on (their own while assessing,
    or the source `view` names). Outside the team decision, targets and leak flags follow that source's targets."""
    from ..leakage import leak_names
    from .fields import view_targets

    with connection() as conn:
        src = source_for(conn, dataset_id, user, view=view)
        # Carried-over fields of a later version are locked; a finalized version is frozen whole anyway.
        open_ = None if src.recipe["phase"] == "finalized" else recipes.open_field_ids(conn, src.recipe)
        rows = conn.execute(
            sql.SQL("""SELECT f.id, f.ordinal, f.name, f.inferred_type, f.inference_confidence, f.is_target,
                              f.null_pct, f.n_unique, f.max_target_assoc, f.leakage_flag AS leakage,
                              d.action, d.semantic_type, d.reviewed, d.source,
                              NOT (%s OR f.id = ANY(%s)) AS locked
                         FROM meta.fields f JOIN {} d ON d.field_id = f.id
                        WHERE f.dataset_id = %s ORDER BY f.ordinal""").format(src.decisions()),
            (open_ is None, open_ or [], dataset_id),
        ).fetchall()
        if src.kind in ("experiment", "baseline", "suggested"):
            leaks = leak_names(conn, dataset_id, view_targets(conn, src))
            rows = [{**r, "is_target": r["action"] == "target", "leakage": r["name"] in leaks} for r in rows]
        return rows
