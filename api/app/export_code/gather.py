"""Read what an export's script is written from: the export's frozen decisions and settings, where its data came
from, the derived fields it uses, and what the app read and wrote, which the script checks itself against."""
from __future__ import annotations

import dataclasses
import json
import re
import tempfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import sklearn

from .. import derived as derived_fields
from .. import recipes
from ..config import get_settings
from ..db import jsonb
from ..ingest import _sink
from ..sample_library import LIBRARY, path_of
from ..semantics import NON_FEATURE_TYPES
from ..transform import FieldSpec
from .render import ExportInfo
from .script import column_summary, output_summary

RULE_KEYS = {"dedupe": False, "exclude": [], "first_per": None, "group_by": None}
# How the Datasets page names a sample it generates (routers/datasets.create_sample), for samples made before the
# settings were recorded; it always passes seed 42.
SAMPLE_NAME = re.compile(r"^sample-(\d+)k-(\d+)f-(mixed|binary|multiclass)(-num)?$")


class CodeUnavailable(Exception):
    """Why this export can't be written as a script."""


def info_for(conn, export_id: int) -> ExportInfo:
    from ..routers.exports import export_spec, target_tasks  # the export's own readers (and that module imports us)

    e = conn.execute(
        """SELECT e.*, r.family AS recipe_family, r.version AS recipe_version,
                  meta.baseline_name(r.dataset_id, r.family) AS recipe_name, x.name AS experiment_name,
                  u.name AS created_by_name
             FROM meta.exports e LEFT JOIN meta.recipes r ON r.id = e.recipe_id
             LEFT JOIN meta.experiments x ON x.id = e.experiment_id LEFT JOIN meta.users u ON u.id = e.created_by
            WHERE e.id = %s""", (export_id,)).fetchone()
    if e is None:
        raise CodeUnavailable("Export not found")
    if e["status"] != "done":
        raise CodeUnavailable("The export isn't finished.")
    ds = conn.execute("""SELECT id, name, source_filename, file_format, parquet_path, source_spec FROM meta.datasets
                          WHERE id = %s""", (e["dataset_id"],)).fetchone()
    raw = Path(ds["parquet_path"] or "")
    if not ds["parquet_path"] or not raw.exists():
        raise CodeUnavailable("The dataset's data is missing.")
    out_dir = Path(e["output_dir"]) if e["output_dir"] else None
    manifest = _json(out_dir / "manifest.json") if out_dir else None
    config, summary = e["config"] or {}, e["summary"] or {}
    snap = e["snapshot"] or recipes.snapshot_of(conn, recipes.get(conn, e["recipe_id"]))
    family = e["model_family"]
    warnings: list[str] = []

    if manifest is not None:
        known = {f.name for f in dataclasses.fields(FieldSpec)}
        specs = [FieldSpec(**{k: v for k, v in f.items() if k in known}) for f in manifest["fields"]]
        names = list(manifest["targets"])
    else:
        specs = [s for s in export_spec(conn, ds["id"], snap, family)[0] if s.semantic_type not in NON_FEATURE_TYPES]
        names = recipes.snapshot_targets(snap)
        warnings.append("The export's files were removed from the app, so this script can't check its result "
                        "against them.")
    tasks = target_tasks(conn, ds["id"], snap)
    if set(tasks) != set(names):
        raise CodeUnavailable("The export's targets no longer match its snapshot.")

    fields_rows = conn.execute("""SELECT name, derived FROM meta.fields WHERE dataset_id = %s ORDER BY ordinal""",
                               (ds["id"],)).fetchall()
    now_derived = [{"name": r["name"], **r["derived"]} for r in fields_rows if r["derived"] is not None]
    derived_list = config["derived"] if "derived" in config else now_derived
    columns = list(pl.read_parquet_schema(raw))
    current_derived = {d["name"] for d in now_derived}
    originals = [c for c in columns if c not in current_derived]
    rule_fields = [r["field"] for r in config.get("exclude") or []] + \
        [v for v in ((config.get("first_per") or {}).get("key"), (config.get("first_per") or {}).get("order"),
                     config.get("group_by")) if v]
    uses = [s.name for s in specs] + names + rule_fields
    used_derived = _needed(derived_list, originals, uses)

    expected_input = {"rows": _rows(raw), "columns": {
        **_input_summary(str(raw), raw.stat().st_mtime_ns, tuple(originals)),
        **{d["name"]: None for d in used_derived},
        **_input_summary(str(raw), raw.stat().st_mtime_ns, tuple(d["name"] for d in used_derived if d["name"] in columns)),
    }}
    source = _source(conn, ds, expected_input, originals)

    model_ready = out_dir / "model_ready.parquet" if out_dir else None
    expected_output: dict[str, Any] = {}
    if manifest is not None and model_ready is not None and model_ready.exists():
        expected_output = _output_summary(str(model_ready), model_ready.stat().st_mtime_ns,
                                          tuple(manifest["feature_names"]))
    versions = (manifest or {}).get("versions") or {}
    now = {"sklearn": sklearn.__version__, "polars": pl.__version__, "numpy": np.__version__}
    if changed := [f"{k} {versions[k]} (now {now[k]})" for k in now if versions.get(k) and versions[k] != now[k]]:
        warnings.append(f"The export was made with {', '.join(changed)}; this script uses the versions the app runs "
                        "now, so its result may differ slightly from the export.")
    if undated := [s.name for s in specs if s.semantic_type == "datetime" and not s.datetime_format
                   and _dtype(raw, s.name) == pl.String]:
        warnings.append(f"The app recorded no date format for {', '.join(undated)}, so each batch of rows works one "
                        "out; the script follows the app's batches, but if they differ the values can too.")

    adjustments = config.get("adjustments") or {}
    tchange = adjustments.get("targets")
    return ExportInfo(
        export_id=e["id"], dataset_name=ds["name"], family=family, specs=specs,
        targets={t: tasks[t] for t in names},
        row_rules={k: v if config.get(k) is None else config[k] for k, v in RULE_KEYS.items()},
        split={"valid_size": config.get("valid_size", 0.15), "test_size": config.get("test_size", 0.15),
               "seed": config.get("seed", 42)},
        fit_sample_rows=(manifest or {}).get("fit_sample_rows") or summary.get("fit_sample_rows"),
        chunk_rows=int((manifest or {}).get("chunk_rows") or config.get("chunk_rows") or 200_000),
        source=source,
        derived=[{"name": d["name"], "kind": d["kind"], "spec": d["spec"]} for d in used_derived],
        notes={n: f["decision"].get("notes") for n, f in snap["fields"].items() if f["decision"].get("notes")},
        adjusted={f["name"]: _change_text(f) for f in adjustments.get("detail") or []},
        targets_changed=(f"{', '.join(tchange['before']) or 'none'} → {', '.join(tchange['after']) or 'none'}"
                         if tchange else None),
        left_out=_left_out(snap),
        expected_input=expected_input,
        expected_output=expected_output,
        baseline=recipes.label({"name": e["recipe_name"], "family": e["recipe_family"], "version": e["recipe_version"]})
        if e["recipe_family"] else None,
        experiment=e["experiment_name"],
        adjustments=int(adjustments.get("count") or 0),
        created_at=_when(e["created_at"]),
        created_by=e["created_by_name"],
        split_counts=(manifest or {}).get("split_counts") or summary.get("split_counts"),
        n_features=e["n_features"],
        made_at=_when(datetime.now(timezone.utc)),
        app_version=get_settings().git_sha[:7] or None,
        warnings=warnings,
    )


# ---- the data -------------------------------------------------------------------------------------------------------
def _source(conn, ds: dict, expected: dict, originals: list[str]) -> dict[str, Any]:
    """How the script gets the data: the uploaded file, a library dataset rebuilt from its public source, or the
    sample generator run again."""
    spec = ds["source_spec"] or {}
    if spec.get("kind") == "library" or (ds["file_format"] == "parquet" and _is_library(ds, expected, originals)):
        key = spec.get("key") or Path(ds["source_filename"]).stem
        return {"kind": "library", "key": key, "file": f"{key}.parquet", "source_url": LIBRARY[key]["source_url"]}
    if spec.get("kind") == "generated":
        return {"kind": "generated", "settings": spec["settings"]}
    if ds["file_format"] == "generated":
        settings = _inferred_sample(ds, expected, originals)
        conn.execute("UPDATE meta.datasets SET source_spec = %s WHERE id = %s",
                     (jsonb({"kind": "generated", "settings": settings}), ds["id"]))
        return {"kind": "generated", "settings": settings}
    return {"kind": "upload", "format": ds["file_format"], "file": ds["source_filename"]}


def _is_library(ds: dict, expected: dict, originals: list[str]) -> bool:
    """A library dataset added before the source was recorded: its file name is a library key and its data is the
    library file's, read the app's way."""
    key = Path(ds["source_filename"] or "").stem
    if key not in LIBRARY or not path_of(key).exists():
        return False
    rows, summary = _library_summary(key)
    return rows == expected["rows"] and summary == {c: expected["columns"].get(c) for c in originals}


@lru_cache(maxsize=8)
def _library_summary(key: str) -> tuple[int, dict[str, dict]]:
    """A library file as the app ingests it, summarised per column."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "raw.parquet"
        _sink(path_of(key), "parquet", out)
        df = pl.read_parquet(out)
    return df.height, {c: column_summary(df[c]) for c in df.columns}


def _inferred_sample(ds: dict, expected: dict, originals: list[str]) -> dict[str, Any]:
    """The settings of a sample generated before they were recorded, from its default name, confirmed by making it
    again."""
    from ..sample_data import generate

    m = SAMPLE_NAME.match(ds["name"] or "")
    if m:
        settings = {"rows": expected["rows"], "fields": int(m.group(2)), "seed": 42, "target_kind": m.group(3),
                    "numeric_target": bool(m.group(4))}
        if int(m.group(1)) == expected["rows"] // 1000 and _same(generate(
                settings["rows"], settings["fields"], 42, settings["target_kind"], settings["numeric_target"]),
                expected, originals):
            return settings
    raise CodeUnavailable("This sample was generated before Rainforest recorded how it made samples, and its settings "
                          "can't be worked out. Generate the sample again to export it as code.")


def _same(df: pl.DataFrame, expected: dict, originals: list[str]) -> bool:
    if df.height != expected["rows"] or list(df.columns) != originals:
        return False
    return all(column_summary(df[c]) == expected["columns"][c] for c in originals)


def _needed(derived_list: list[dict], originals: list[str], uses: list[str]) -> list[dict]:
    """The derived fields the export reads, with the derived fields they read in turn, in the order they were
    added."""
    by_name = {d["name"]: d for d in derived_list}
    columns = originals + [d["name"] for d in derived_list]
    need: set[str] = set()
    stack = [n for n in uses if n not in originals]
    while stack:
        n = stack.pop()
        if n in need or n in originals:
            continue
        if n not in by_name:
            raise CodeUnavailable(f"The export uses {n!r}, which is no longer in the dataset.")
        need.add(n)
        try:
            stack += derived_fields.check_spec(by_name[n]["kind"], by_name[n]["spec"], columns)
        except Exception as exc:  # noqa: BLE001 - the parser answers with an HTTP error
            raise CodeUnavailable(f"The derived field {n!r} can't be read: {getattr(exc, 'detail', exc)}") from exc
    return [d for d in derived_list if d["name"] in need]


@lru_cache(maxsize=32)
def _input_summary(path: str, mtime_ns: int, columns: tuple[str, ...]) -> dict[str, dict]:
    """Per column of the dataset's Parquet copy, what the script checks the data it reads against. One column at a
    time, so a wide file never sits in memory whole."""
    return {c: column_summary(pl.read_parquet(path, columns=[c])[c]) for c in columns}


@lru_cache(maxsize=32)
def _output_summary(path: str, mtime_ns: int, features: tuple[str, ...]) -> dict[str, Any]:
    names = pq.read_schema(path).names
    summary = {"rows": pq.ParquetFile(path).metadata.num_rows, "columns": {}}
    for c in names:
        summary["columns"].update(output_summary(pl.read_parquet(path, columns=[c]), list(features))["columns"])
    return summary


def _rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def _dtype(path: Path, name: str):
    return pl.read_parquet_schema(path).get(name)


def _json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


# ---- what people wrote ----------------------------------------------------------------------------------------------
def _left_out(snap: dict) -> list[tuple[str, str, str]]:
    out = []
    for name, f in snap["fields"].items():
        d = f["decision"]
        if d["action"] == "drop":
            out.append((name, d["semantic_type"], "dropped"))
        elif d["action"] == "keep" and d["semantic_type"] in NON_FEATURE_TYPES:
            out.append((name, d["semantic_type"], "kept, but identifiers and constants are never features"))
    return out


def _change_text(f: dict) -> str:
    if f.get("change") != "changed":
        return f"{f.get('change')} in this experiment"
    return "; ".join(f"{c['key']} {_short(c['before'])} → {_short(c['after'])}" for c in f.get("changes") or [])


def _short(v: Any) -> str:
    if v is None:
        return "none"
    text = v if isinstance(v, str) else json.dumps(v)
    return text if len(text) <= 60 else text[:57] + "…"


def _when(t: datetime | None) -> str | None:
    if t is None:
        return None
    return t.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
