from __future__ import annotations

import json
import logging
import time
import traceback
from pathlib import Path

import polars as pl
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

import numpy as np

from .. import bands, experiments, jobs, recipes
from ..auth import CurrentUser
from ..config import get_settings
from ..db import connection, execute, fetch_all, fetch_one, jsonb
from ..leakage import leak_names
from ..schemas import ExportRequest
from ..semantics import FAMILIES, NON_FEATURE_TYPES, validate_decision
from ..profiling import coerce_numeric
from ..transform import FieldSpec, check_target_classes, run_export
from .fields import _same_settings

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["exports"])

FILES = {
    "data": ("model_ready.parquet", "application/vnd.apache.parquet"),
    "preprocessor": ("preprocessor.joblib", "application/octet-stream"),
    "manifest": ("manifest.json", "application/json"),
}


EXPORT_COLS = """e.id, e.dataset_id, e.status, e.progress, e.status_message, e.error, e.model_family, e.config,
                 e.output_dir, e.n_rows, e.n_features, e.summary, e.created_at, e.updated_at, e.finished_at, e.owner,
                 e.created_by, e.recipe_id, e.experiment_id, r.family AS recipe_family, r.version AS recipe_version,
                 meta.baseline_name(r.dataset_id, r.family) AS recipe_name, x.name AS experiment_name"""
EXPORT_SELECT = (f"SELECT {EXPORT_COLS} FROM meta.exports e LEFT JOIN meta.recipes r ON r.id = e.recipe_id "
                 "LEFT JOIN meta.experiments x ON x.id = e.experiment_id")
PHASE_WAIT = {"assessing": "is being assessed", "review": "is in team review"}


def source_problem(recipe: dict) -> str | None:
    """Why a team export can't be built from this recipe version yet."""
    if recipe["phase"] == "finalized":
        return None
    return (f"{recipes.label(recipe)} {PHASE_WAIT[recipe['phase']]}. Team exports open once a lead finalizes it.")


def export_source(conn, dataset_id: int, recipe_id: int | None = None) -> dict:
    """The recipe version a team export is built from: `recipe_id`, else the dataset's latest finalized version
    (with none yet, the current one, which says why it can't export). Only finalized versions export, from their
    snapshot. Personal exports come from an experiment instead (`_plan`)."""
    recipe = _pick_recipe(conn, dataset_id, recipe_id)
    if (problem := source_problem(recipe)) is not None:
        raise HTTPException(409, problem)
    return recipe


def _pick_recipe(conn, dataset_id: int, recipe_id: int | None) -> dict:
    if recipe_id is not None:
        recipe = recipes.get(conn, recipe_id)
        if recipe is None or recipe["dataset_id"] != dataset_id:
            raise HTTPException(404, "Recipe not found")
        return recipe
    family = conn.execute("SELECT model_family FROM meta.datasets WHERE id = %s", (dataset_id,)).fetchone()
    return (recipes.latest_finalized(conn, dataset_id, family and family["model_family"])
            or recipes.latest_finalized(conn, dataset_id) or recipes.current(conn, dataset_id))


def _experiment(conn, dataset_id: int, experiment_id: int, user: dict, recipe_id: int | None) -> dict:
    x = experiments.get_visible(conn, experiment_id, user)
    if x["dataset_id"] != dataset_id:
        raise HTTPException(404, "Experiment not found")
    if recipe_id is not None and recipe_id != x["recipe_id"]:
        raise HTTPException(422, f"Experiment {x['name']!r} is built from recipe v{x['recipe_version']}")
    return x


def _plan(conn, dataset_id: int, user: dict, recipe_id: int | None,
          experiment_id: int | None) -> tuple[dict, dict, dict | None]:
    """(recipe, snapshot, experiment) an export would be built from: the experiment's source when given (a
    personal one reads its assessor's assessment as it stands now), else a recipe version."""
    if experiment_id is None:
        recipe = _pick_recipe(conn, dataset_id, recipe_id)
        return recipe, recipes.snapshot_of(conn, recipe), None
    x = _experiment(conn, dataset_id, experiment_id, user, recipe_id)
    return recipes.get(conn, x["recipe_id"]), experiments.snapshot_for(conn, x), x


def _unreviewed(conn, recipe: dict, x: dict | None) -> int:
    if x is not None and x["assessor_id"] is not None:
        open_ = recipes.open_field_ids(conn, recipe)
        return conn.execute(
            """SELECT count(*) AS n FROM meta.assessments WHERE recipe_id = %s AND user_id = %s AND NOT reviewed
                  AND (%s OR field_id = ANY(%s))""",
            (recipe["id"], x["assessor_id"], open_ is None, open_ or []),
        ).fetchone()["n"]
    if recipe["phase"] == "finalized":
        return 0
    return conn.execute("SELECT count(*) AS n FROM meta.field_decisions WHERE dataset_id = %s AND NOT reviewed",
                        (recipe["dataset_id"],)).fetchone()["n"]


@router.get("/datasets/{dataset_id}/export-preview")
def preview(dataset_id: int, user: CurrentUser, family: str | None = None, recipe_id: int | None = None,
            experiment_id: int | None = None) -> dict:
    """What an export would contain, plus anything worth fixing first: of an experiment's source when given (with a
    team experiment's adjustments), else of a baseline version (default: the latest finalized of the current
    baseline's model type, else of any; with none yet, the team decision as it stands, blocked). The model family is
    the baseline's."""
    if fetch_one("SELECT 1 FROM meta.datasets WHERE id = %s", (dataset_id,)) is None:
        raise HTTPException(404, "Dataset not found")
    with connection() as conn:
        recipe, snap, x = _plan(conn, dataset_id, user, recipe_id, experiment_id)
        return _preview_of(conn, dataset_id, user, recipe, snap, x, family)


def _preview_of(conn, dataset_id: int, user: dict, recipe: dict, snap: dict, x: dict | None,
                family: str | None) -> dict:
    fam = recipe["family"]
    if family is not None and family != fam:
        raise HTTPException(422, f"{recipes.label(recipe)} prepares data for {FAMILIES[fam]['label'].lower()}, "
                                 f"not {family!r}")
    ds = conn.execute("SELECT parquet_path FROM meta.datasets WHERE id = %s", (dataset_id,)).fetchone()
    rows = recipes.spec_rows(conn, dataset_id, snap, fam)
    targets = recipes.snapshot_targets(snap)
    tasks = target_tasks(conn, dataset_id, snap)
    leaking = leak_names(conn, dataset_id, targets)
    unchecked = [r["name"] for r in conn.execute(
        """SELECT f.name FROM meta.fields f WHERE f.dataset_id = %s AND f.name = ANY(%s)
              AND NOT EXISTS (SELECT 1 FROM meta.field_correlations c WHERE f.id IN (c.field_a, c.field_b))""",
        (dataset_id, targets)).fetchall()]
    base_targets = None
    if x is not None and x["assessor_id"] is None and recipe["phase"] == "finalized":
        base_targets = recipes.snapshot_targets(recipes.snapshot_of(conn, recipe))
    unreviewed = _unreviewed(conn, recipe, x)
    experiment_id = None if x is None else x["id"]
    last = conn.execute(
        f"""SELECT e.id, e.created_at, e.status, e.recipe_id, e.snapshot FROM meta.exports e
             WHERE e.dataset_id = %s AND e.model_family = %s AND e.status = 'done'
               AND e.experiment_id IN ({experiments.visible_ids(user['id'])})
               AND (%s::bigint IS NULL OR e.experiment_id = %s)
             ORDER BY e.created_at DESC, e.id DESC LIMIT 1""",
        (dataset_id, fam, experiment_id, experiment_id),
    ).fetchone()
    changed = None
    if last is not None and last["recipe_id"] is not None:
        # Fields whose decision or settings for this family differ from what the last export was built from.
        before = last["snapshot"] or recipes.snapshot_of(conn, recipes.get(conn, last["recipe_id"]))
        changed = len(recipes.diff(before, snap, fam)["fields"])
    if not rows:
        raise HTTPException(404, "Dataset has no profiled fields")
    keep = [r for r in rows if r["action"] == "keep" and r["semantic_type"] not in NON_FEATURE_TYPES]
    warnings: list[str] = []
    if unreviewed:
        warnings.append(f"{unreviewed} field(s) not reviewed yet — their suggested settings will be used.")
    if base_targets is not None and base_targets != targets:
        warnings.append(f"Targets differ from {recipes.label(recipe)}: {', '.join(base_targets) or 'none'} → "
                        f"{', '.join(targets) or 'none'}. Results aren't comparable with experiments on the "
                        "baseline's targets.")
    if unchecked:
        warnings.append(f"No leak check has run for target(s) {', '.join(unchecked)}: no associations were "
                        "computed for them.")
    leaks = [r["name"] for r in keep if r["name"] in leaking]
    if leaks:
        warnings.append(f"Kept field(s) flagged as possible target leakage: {', '.join(leaks)}")
    invalid = [r["name"] for r in keep if validate_decision({**r, "action": "keep"}, fam)]
    by_type: dict[str, int] = {}
    for r in keep:
        by_type[r["semantic_type"]] = by_type.get(r["semantic_type"], 0) + 1
    blocking = [m for m, bad in [("No target selected", not targets), ("No features kept", not keep)] if bad]
    personal = x is not None and x["assessor_id"] is not None
    if not personal and (problem := source_problem(recipe)) is not None:
        blocking.insert(0, problem)
    class_targets = [t for t in targets if tasks[t]["task"] == "classes"]
    if lone := _lone_class_problem(ds["parquet_path"], class_targets):
        blocking.append(lone)
    blocking += _numeric_target_problems(ds["parquet_path"], tasks)
    if invalid:
        blocking.append(f"{len(invalid)} field(s) have settings that aren't valid for {FAMILIES[fam]['label'].lower()}: "
                        f"{', '.join(invalid[:8])}{'…' if len(invalid) > 8 else ''}")
    return {
        "family": fam,
        "recipe": {**{k: recipe[k] for k in ("id", "family", "version", "phase", "name")}, "label": recipes.label(recipe)},
        "source": "assessment" if personal else "recipe",
        "experiment_id": experiment_id,
        "kept": len(keep),
        "dropped": sum(1 for r in rows if r["action"] == "drop"),
        "targets": targets,
        "target_tasks": {t: v["task"] for t, v in tasks.items()},
        "kept_by_type": by_type,
        "customized": sum(1 for r in keep if not _same_settings(r, (r["suggestion"] or {}).get(fam) or {})),
        "unreviewed": unreviewed,
        "warnings": warnings,
        "blocking": blocking,
        "last_export": None if last is None else {k: last[k] for k in ("id", "created_at", "status", "recipe_id")},
        "changed_since_export": changed,
    }


def _numeric_target_problems(parquet_path: str | None, tasks: dict[str, dict]) -> list[str]:
    """The export's own checks on numbers and bands, run on the whole column now rather than after queueing (bands
    with equal-sized counts are cut on the training rows then, so only chosen cut points are checked here)."""
    numeric = [t for t, v in tasks.items() if v["task"] != "classes"]
    if not numeric or not parquet_path or not Path(parquet_path).exists():
        return []
    out = []
    raw = pl.read_parquet(parquet_path, columns=numeric)
    for t in numeric:
        num = coerce_numeric(raw[t], tasks[t].get("decimal_comma"))
        given = raw[t].fill_nan(None) if raw[t].dtype.is_float() else raw[t]
        if lost := int(given.is_not_null().sum() - num.is_not_null().sum()):
            out.append(f"Target {t!r} has {lost:,} value(s) that don't read as numbers (or are infinite); fix them or "
                       "predict it as classes.")
            continue
        vals = num.to_numpy()
        if np.unique(vals[~np.isnan(vals)]).size < 2:
            out.append(f"Target {t!r} has a single value; there is nothing to predict.")
        elif tasks[t]["task"] == "bins" and "edges" in (tasks[t]["bins"] or {}):
            if thin := bands.thin_bands(vals, [float(e) for e in tasks[t]["bins"]["edges"]]):
                out.append(f"Target {t!r}: band(s) {', '.join(thin)} have fewer than 2 rows; move or remove cut "
                           "points, or use fewer bands.")
    return out


def target_tasks(conn, dataset_id: int, snap: dict) -> dict[str, dict]:
    """How each target of a snapshot is predicted, with how its column writes decimals (for reading it as numbers)."""
    tasks = recipes.snapshot_target_tasks(snap)
    comma = {r["name"]: r["decimal_comma"] for r in conn.execute(
        """SELECT name, coalesce((stats ->> 'decimal_comma')::boolean, false) AS decimal_comma FROM meta.fields
            WHERE dataset_id = %s AND name = ANY(%s)""", (dataset_id, list(tasks))).fetchall()}
    return {t: {**v, "decimal_comma": comma.get(t, False)} for t, v in tasks.items()}


def _lone_class_problem(parquet_path: str | None, targets: list[str]) -> str | None:
    """The export's own single-row-class check, run on the target columns now rather than after queueing."""
    if not targets or not parquet_path or not Path(parquet_path).exists():
        return None
    try:
        check_target_classes(pl.read_parquet(parquet_path, columns=targets).to_pandas())
    except ValueError as exc:
        return str(exc)
    return None


@router.post("/datasets/{dataset_id}/exports", status_code=202)
def start_export(dataset_id: int, req: ExportRequest, user: CurrentUser) -> dict:
    """A team export of a finalized version (into the experiment given, else the version's default Team
    experiment), or a personal export of the viewer's own assessment (into their personal experiment). What it is
    built from is frozen now: the version's snapshot, or the experiment's (with its adjustments)."""
    ds = fetch_one("SELECT status FROM meta.datasets WHERE id = %s", (dataset_id,))
    if ds is None:
        raise HTTPException(404, "Dataset not found")
    if ds["status"] != "ready":
        raise HTTPException(409, "Dataset is not ready")
    if req.valid_size + req.test_size >= 0.9:
        raise HTTPException(422, "valid_size + test_size must be < 0.9")
    with connection() as conn:
        if req.experiment_id is None:
            if req.source == "assessment":
                raise HTTPException(422, "An export of your own assessment belongs to a personal experiment: "
                                         "give its experiment_id.")
            recipe = export_source(conn, dataset_id, req.recipe_id)
            x = experiments.team_for(conn, recipe)
            snap = recipes.snapshot_of(conn, recipe)
        else:
            x = _experiment(conn, dataset_id, req.experiment_id, user, req.recipe_id)
            experiments.require_writable(x, user)
            # Holds off adjustments until the export is recorded, so it is exactly what was checked.
            x = conn.execute(f"SELECT {experiments.COLS} FROM {experiments.FROM} WHERE x.id = %s FOR UPDATE OF x",
                             (x["id"],)).fetchone()
            experiments.require_writable(x, user)
            personal = x["assessor_id"] is not None
            if req.source is not None and (req.source == "assessment") != personal:
                raise HTTPException(422, f"Experiment {x['name']!r} is built from "
                                         f"{'an assessment' if personal else 'a baseline'}, not {req.source!r}")
            recipe = recipes.get(conn, x["recipe_id"])
            if not personal and (problem := source_problem(recipe)) is not None:
                raise HTTPException(409, problem)
            snap = experiments.snapshot_for(conn, x)
        pv = _preview_of(conn, dataset_id, user, recipe, snap, x, req.model_family)
        if pv["blocking"]:
            raise HTTPException(422, {"message": "Cannot export", "errors": pv["blocking"]})
        personal = x["assessor_id"] is not None
        adj = experiments.adjustments(conn, x)
        config = {**req.model_dump(exclude={"model_family", "experiment_id"}),
                  "source": "assessment" if personal else "recipe", "recipe_id": recipe["id"],
                  "recipe_version": recipe["version"], "experiment_id": x["id"],
                  "baseline": {"recipe_id": recipe["id"], "family": recipe["family"], "version": recipe["version"],
                               "name": recipe["name"], "label": recipes.label(recipe)},
                  "adjustments": {"count": adj["count"], "fields": [f["name"] for f in adj["fields"]],
                                  "targets_changed": adj["targets_changed"], "targets": adj["targets"],
                                  "detail": adj["fields"]}}
        if personal:
            config["assessor_id"] = x["assessor_id"]
        exp = conn.execute(
            """INSERT INTO meta.exports (dataset_id, status, status_message, model_family, config, owner, created_by,
                                         recipe_id, experiment_id, snapshot)
               VALUES (%s, 'pending', 'Queued', %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (dataset_id, pv["family"], jsonb(config), jobs.owner_id(), user["id"], recipe["id"], x["id"], jsonb(snap)),
        ).fetchone()
        out = conn.execute(f"{EXPORT_SELECT} WHERE e.id = %s", (exp["id"],)).fetchone()
    jobs.submit(_run, exp["id"])
    return out


@router.get("/datasets/{dataset_id}/exports")
def list_exports(dataset_id: int, user: CurrentUser, experiment_id: int | None = None) -> list[dict]:
    rows = fetch_all(
        f"""{EXPORT_SELECT} WHERE e.dataset_id = %s AND e.experiment_id IN ({experiments.visible_ids(user['id'])})
              AND (%s::bigint IS NULL OR e.experiment_id = %s) ORDER BY e.created_at DESC""",
        (dataset_id, experiment_id, experiment_id),
    )
    return [_backfill_features_by_type(e) for e in rows]


def visible_export(export_id: int, user: dict) -> dict:
    """The export, or 404 when it doesn't exist or belongs to an experiment `user` may not see."""
    exp = fetch_one(f"{EXPORT_SELECT} WHERE e.id = %s AND e.experiment_id IN ({experiments.visible_ids(user['id'])})",
                    (export_id,))
    if exp is None:
        raise HTTPException(404, "Export not found")
    return exp


@router.get("/exports/{export_id}")
def get_export(export_id: int, user: CurrentUser) -> dict:
    return _backfill_features_by_type(visible_export(export_id, user))


def features_by_type(manifest: dict) -> dict[str, dict[str, int]]:
    """Input fields and the output features they expand into, per semantic type."""
    type_of = {f["name"]: f["semantic_type"] for f in manifest["fields"]}
    out: dict[str, dict[str, int]] = {}
    for t in type_of.values():
        out.setdefault(t, {"fields": 0, "features": 0})["fields"] += 1
    for source in manifest["feature_source_field"].values():
        t = type_of.get(source)
        if t is not None:
            out[t]["features"] += 1
    return out


def _backfill_features_by_type(exp: dict) -> dict:
    """Exports made before features_by_type existed get it from their saved manifest, once."""
    summary = exp.get("summary")
    if exp["status"] != "done" or not isinstance(summary, dict) or "features_by_type" in summary:
        return exp
    try:
        manifest = json.loads((Path(exp["output_dir"]) / "manifest.json").read_text())
        by_type = features_by_type(manifest)
    except (OSError, ValueError, KeyError, TypeError):
        return exp
    execute(
        "UPDATE meta.exports SET summary = summary || %s WHERE id = %s",
        (jsonb({"features_by_type": by_type}), exp["id"]),
    )
    return {**exp, "summary": {**summary, "features_by_type": by_type}}


@router.get("/exports/{export_id}/files/{kind}")
def download(export_id: int, kind: str, user: CurrentUser) -> FileResponse:
    exp = visible_export(export_id, user)
    if kind not in FILES:
        raise HTTPException(404, f"Unknown file {kind!r}")
    if exp["status"] != "done":
        raise HTTPException(409, "Export is not finished")
    fname, media = FILES[kind]
    path = Path(exp["output_dir"]) / fname
    if not path.exists():
        raise HTTPException(410, "Export files were removed")
    return FileResponse(path, media_type=media, filename=f"export{export_id}_{fname}")


def export_spec(conn, dataset_id: int, snap: dict, family: str) -> tuple[list[FieldSpec], list[str], dict[str, dict]]:
    """The kept fields' specs, the targets (in chain order) and how each target is predicted, of a snapshot."""
    specs = [
        FieldSpec(
            name=r["name"], semantic_type=r["semantic_type"], imputation=r["imputation"],
            impute_value=r["impute_value"], add_missing_indicator=bool(r["add_missing_indicator"]),
            transform=r["transform"], scaling=r["scaling"], encoding=r["encoding"],
            params=r["params"] or {}, datetime_format=r["datetime_format"], decimal_comma=r["decimal_comma"],
        )
        for r in recipes.spec_rows(conn, dataset_id, snap, family)
        if r["action"] == "keep"
    ]
    return specs, recipes.snapshot_targets(snap), target_tasks(conn, dataset_id, snap)


def _run(export_id: int) -> None:
    exp = fetch_one("SELECT * FROM meta.exports WHERE id = %s", (export_id,))
    ds = fetch_one("SELECT * FROM meta.datasets WHERE id = %s", (exp["dataset_id"],))
    out_dir = get_settings().exports_dir / str(export_id)
    last = [0.0]

    def progress(frac: float, msg: str) -> None:
        now = time.monotonic()
        if now - last[0] < 1.0 and frac < 1.0:
            return
        last[0] = now
        execute(
            "UPDATE meta.exports SET status='running', error=NULL, progress=%s, status_message=%s, updated_at=now() "
            "WHERE id=%s",
            (round(frac, 4), msg, export_id),
        )

    try:
        progress(0.0, "Starting")
        fam = exp["model_family"]
        with connection() as conn:
            snap = exp["snapshot"] or recipes.snapshot_of(conn, recipes.get(conn, exp["recipe_id"]))
            specs, targets, tasks = export_spec(conn, ds["id"], snap, fam)
        manifest = run_export(Path(ds["parquet_path"]), specs, targets, out_dir, exp["config"], progress, family=fam,
                              target_tasks=tasks)
        summary = {
            "split_counts": manifest["split_counts"],
            "split_info": manifest["split_info"],
            "targets": manifest["targets"],
            "target_classes": manifest["target_classes"],
            "target_kinds": manifest["target_kinds"],
            "target_tasks": manifest["target_tasks"],
            "fit_sample_rows": manifest["fit_sample_rows"],
            "model_family": fam,
            "contains_nan": manifest["contains_nan"],
            "features_by_type": features_by_type(manifest),
            "train_command": f"docker compose exec api python -m app.modeling.train --export-id {export_id}"
                             + ("" if fam == "tree" else f" --model {'logistic_regression' if fam == 'linear' else 'mlp'}"),
        }
        execute(
            """UPDATE meta.exports SET status='done', error=NULL, progress=1, status_message='Done',
                      output_dir=%s,
                      n_rows=%s, n_features=%s, summary=%s, finished_at=now(), updated_at=now() WHERE id=%s""",
            (str(out_dir), manifest["rows_written"], manifest["n_features"], jsonb(summary), export_id),
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("export %s failed", export_id)
        execute(
            """UPDATE meta.exports SET status='failed', error=%s, status_message='Failed', finished_at=now(),
                      updated_at=now() WHERE id=%s""",
            (f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc(limit=4)}", export_id),
        )
