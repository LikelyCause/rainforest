from __future__ import annotations

import json
import logging
import time
import traceback
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .. import jobs
from ..config import get_settings
from ..db import execute, fetch_all, fetch_one, jsonb
from ..schemas import ExportRequest
from ..semantics import FAMILIES, NON_FEATURE_TYPES, validate_decision
from ..transform import FieldSpec, run_export

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["exports"])

FILES = {
    "data": ("model_ready.parquet", "application/vnd.apache.parquet"),
    "preprocessor": ("preprocessor.joblib", "application/octet-stream"),
    "manifest": ("manifest.json", "application/json"),
}


@router.get("/datasets/{dataset_id}/export-preview")
def preview(dataset_id: int, family: str | None = None) -> dict:
    """What an export for a model family would contain right now, plus anything worth fixing first."""
    ds = fetch_one("SELECT model_family FROM meta.datasets WHERE id = %s", (dataset_id,))
    if ds is None:
        raise HTTPException(404, "Dataset not found")
    fam = family or ds["model_family"]
    if fam not in FAMILIES:
        raise HTTPException(422, f"Unknown model family {fam!r}")
    rows = fetch_all(
        """SELECT f.name, f.inferred_type, f.leakage_flag AS leakage, d.action, d.semantic_type, d.reviewed,
                  s.imputation, s.impute_value, s.add_missing_indicator, s.transform, s.scaling, s.encoding,
                  s.params, s.source AS settings_source
             FROM meta.fields f
             JOIN meta.field_decisions d ON d.field_id = f.id
             JOIN meta.field_settings s ON s.field_id = f.id AND s.family = %s
            WHERE f.dataset_id = %s ORDER BY f.ordinal""",
        (fam, dataset_id),
    )
    if not rows:
        raise HTTPException(404, "Dataset has no profiled fields")
    keep = [r for r in rows if r["action"] == "keep" and r["semantic_type"] not in NON_FEATURE_TYPES]
    targets = [r["name"] for r in rows if r["action"] == "target"]
    warnings: list[str] = []
    unreviewed = [r["name"] for r in rows if not r["reviewed"]]
    if unreviewed:
        warnings.append(f"{len(unreviewed)} field(s) not reviewed yet — their suggested settings will be used.")
    leaks = [r["name"] for r in keep if r["leakage"]]
    if leaks:
        warnings.append(f"Kept field(s) flagged as possible target leakage: {', '.join(leaks)}")
    if len(targets) != 2:
        warnings.append(f"{len(targets)} target(s) selected; the ClassifierChain expects two (column_a, column_b).")
    invalid = [r["name"] for r in keep if validate_decision({**r, "action": "keep"}, fam)]
    by_type: dict[str, int] = {}
    for r in keep:
        by_type[r["semantic_type"]] = by_type.get(r["semantic_type"], 0) + 1
    last = fetch_one(
        """SELECT e.id, e.created_at, e.status,
                  (SELECT count(*) FROM meta.fields f
                     JOIN meta.field_decisions d ON d.field_id = f.id
                     LEFT JOIN meta.field_settings s ON s.field_id = f.id AND s.family = e.model_family
                    WHERE f.dataset_id = e.dataset_id
                      AND (d.updated_at > e.created_at OR s.updated_at > e.created_at)) AS changed
             FROM meta.exports e
            WHERE e.dataset_id = %s AND e.model_family = %s AND e.status = 'done'
            ORDER BY e.created_at DESC, e.id DESC LIMIT 1""",
        (dataset_id, fam),
    )
    blocking = [m for m, bad in [("No target selected", not targets), ("No features kept", not keep)] if bad]
    if invalid:
        blocking.append(f"{len(invalid)} field(s) have settings that aren't valid for {FAMILIES[fam]['label'].lower()}: "
                        f"{', '.join(invalid[:8])}{'…' if len(invalid) > 8 else ''}")
    return {
        "family": fam,
        "kept": len(keep),
        "dropped": sum(1 for r in rows if r["action"] == "drop"),
        "targets": targets,
        "kept_by_type": by_type,
        "customized": sum(1 for r in keep if r["settings_source"] == "user"),
        "unreviewed": len(unreviewed),
        "warnings": warnings,
        "blocking": blocking,
        "last_export": None if last is None else {k: last[k] for k in ("id", "created_at", "status")},
        "changed_since_export": None if last is None else last["changed"],
    }


@router.post("/datasets/{dataset_id}/exports", status_code=202)
def start_export(dataset_id: int, req: ExportRequest) -> dict:
    ds = fetch_one("SELECT status FROM meta.datasets WHERE id = %s", (dataset_id,))
    if ds is None:
        raise HTTPException(404, "Dataset not found")
    if ds["status"] != "ready":
        raise HTTPException(409, "Dataset is not ready")
    pv = preview(dataset_id, req.model_family)
    if pv["blocking"]:
        raise HTTPException(422, {"message": "Cannot export", "errors": pv["blocking"]})
    if req.valid_size + req.test_size >= 0.9:
        raise HTTPException(422, "valid_size + test_size must be < 0.9")
    exp = fetch_one(
        """INSERT INTO meta.exports (dataset_id, status, status_message, model_family, config)
           VALUES (%s, 'pending', 'Queued', %s, %s) RETURNING *""",
        (dataset_id, pv["family"], jsonb(req.model_dump(exclude={"model_family"}))),
    )
    jobs.submit(_run, exp["id"])
    return exp


@router.get("/datasets/{dataset_id}/exports")
def list_exports(dataset_id: int) -> list[dict]:
    rows = fetch_all(
        "SELECT * FROM meta.exports WHERE dataset_id = %s ORDER BY created_at DESC", (dataset_id,)
    )
    return [_backfill_features_by_type(e) for e in rows]


@router.get("/exports/{export_id}")
def get_export(export_id: int) -> dict:
    exp = fetch_one("SELECT * FROM meta.exports WHERE id = %s", (export_id,))
    if exp is None:
        raise HTTPException(404, "Export not found")
    return _backfill_features_by_type(exp)


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
def download(export_id: int, kind: str) -> FileResponse:
    exp = get_export(export_id)
    if kind not in FILES:
        raise HTTPException(404, f"Unknown file {kind!r}")
    if exp["status"] != "done":
        raise HTTPException(409, "Export is not finished")
    fname, media = FILES[kind]
    path = Path(exp["output_dir"]) / fname
    if not path.exists():
        raise HTTPException(410, "Export files were removed")
    return FileResponse(path, media_type=media, filename=f"export{export_id}_{fname}")


def _run(export_id: int) -> None:
    exp = get_export(export_id)
    ds = fetch_one("SELECT * FROM meta.datasets WHERE id = %s", (exp["dataset_id"],))
    out_dir = get_settings().exports_dir / str(export_id)
    last = [0.0]

    def progress(frac: float, msg: str) -> None:
        now = time.monotonic()
        if now - last[0] < 1.0 and frac < 1.0:
            return
        last[0] = now
        execute(
            "UPDATE meta.exports SET status='running', progress=%s, status_message=%s, updated_at=now() WHERE id=%s",
            (round(frac, 4), msg, export_id),
        )

    try:
        progress(0.0, "Starting")
        fam = exp["model_family"]
        rows = fetch_all(
            """SELECT f.name, f.stats ->> 'datetime_format' AS datetime_format, d.action, d.semantic_type, s.*
                 FROM meta.fields f
                 JOIN meta.field_decisions d ON d.field_id = f.id
                 JOIN meta.field_settings s ON s.field_id = f.id AND s.family = %s
                WHERE f.dataset_id = %s ORDER BY f.ordinal""",
            (fam, ds["id"]),
        )
        specs = [
            FieldSpec(
                name=r["name"], semantic_type=r["semantic_type"], imputation=r["imputation"],
                impute_value=r["impute_value"], add_missing_indicator=r["add_missing_indicator"],
                transform=r["transform"], scaling=r["scaling"], encoding=r["encoding"],
                params=r["params"] or {}, datetime_format=r["datetime_format"],
            )
            for r in rows
            if r["action"] == "keep"
        ]
        target_rows = [r["name"] for r in rows if r["action"] == "target"]
        order = list(ds["target_columns"] or [])
        targets = [t for t in order if t in target_rows] + [t for t in target_rows if t not in order]
        manifest = run_export(Path(ds["parquet_path"]), specs, targets, out_dir, exp["config"], progress, family=fam)
        summary = {
            "split_counts": manifest["split_counts"],
            "split_info": manifest["split_info"],
            "targets": manifest["targets"],
            "target_classes": manifest["target_classes"],
            "target_kinds": manifest["target_kinds"],
            "fit_sample_rows": manifest["fit_sample_rows"],
            "model_family": fam,
            "contains_nan": manifest["contains_nan"],
            "features_by_type": features_by_type(manifest),
            "train_command": f"docker compose exec api python -m app.modeling.train --export-id {export_id}"
                             + ("" if fam == "tree" else f" --model {'logistic_regression' if fam == 'linear' else 'mlp'}"),
        }
        execute(
            """UPDATE meta.exports SET status='done', progress=1, status_message='Done', output_dir=%s,
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
