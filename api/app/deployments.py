"""Deployments: a frozen set of finished runs of one export, what may share one, their thresholds, the API keys that
may call a deployment's predict endpoint, turning predict's JSON records into the frame `scoring` reads, and the job
that tests a deployment with an uploaded file."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.compute as pc
import pyarrow.parquet as pq
from psycopg import sql
from sklearn.metrics import average_precision_score

from . import ingest, scoring
from .config import get_settings
from .db import connection, jsonb
from .modeling.train import choose_threshold
from .profiling import coerce_numeric
from .transform import target_labels

log = logging.getLogger(__name__)

KEY_PREFIX = "rf_"
KEY_LEN = len(KEY_PREFIX) + 32
SHOWN = 8  # the characters kept in clear, so a list can tell keys apart
F1 = {"kind": "f1"}
INT64 = 2**63
WORDS = {2: "two", 3: "three"}


# =========================================================================== keys
def new_key() -> tuple[str, str, str]:
    """A key, its shown prefix and its SHA-256: 24 random bytes are 32 URL-safe characters."""
    key = KEY_PREFIX + secrets.token_urlsafe(24)
    return key, key[:SHOWN], digest(key)


def digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def key_for(deployment_id: int, key: str) -> int | None:
    """The id of `deployment_id`'s unrevoked key `key`, or None. Using it records `last_used_at`, at most once a
    minute, so a busy script doesn't write on every call."""
    if not key.startswith(KEY_PREFIX) or len(key) != KEY_LEN:
        return None
    want = digest(key)
    with connection() as conn:
        rows = conn.execute(
            """SELECT id, hash FROM meta.deployment_keys
                WHERE deployment_id = %s AND prefix = %s AND revoked_at IS NULL""",
            (deployment_id, key[:SHOWN]),
        ).fetchall()
        hit = next((r["id"] for r in rows if hmac.compare_digest(r["hash"], want)), None)
        if hit is not None:
            conn.execute(
                """UPDATE meta.deployment_keys SET last_used_at = now()
                    WHERE id = %s AND (last_used_at IS NULL OR last_used_at < now() - interval '1 minute')""",
                (hit,),
            )
    return hit


# =========================================================================== composition
def kind_of(report: dict) -> tuple[str, str]:
    """(binary, multiclass or regression; the target) from a run's report, or ValueError for a chain."""
    order = report.get("order") or []
    if report.get("chain_kind") == "one_vs_rest":
        return "binary", report["target"]
    if len(order) != 1:
        raise ValueError(f"chains several targets ({', '.join(order)})")
    return ("regression" if report["target_kinds"][order[0]] == "regression" else "multiclass"), order[0]


def _runs(ids: list[int]) -> str:
    return ", ".join(map(str, ids))


def compose(runs: list[dict], rule: dict | None, per_run: dict[int, dict] | None) -> tuple[int, list[dict]]:
    """The export and the models of a deployment of `runs` (rows with id, export_id, status and report), each binary
    model with its threshold resolved. A deployment is either one-vs-rest runs of one export and one target, one run
    per class, or a single run with one target; anything else is a ValueError naming the conflict."""
    if not runs:
        raise ValueError("Choose the runs to deploy, or a one-vs-rest batch")
    if unfinished := [r["id"] for r in runs if r["status"] != "done" or not r["report"]]:
        raise ValueError(f"Only finished runs can be deployed; run {_runs(unfinished)} hasn't finished")
    exports = sorted({r["export_id"] for r in runs})
    if len(exports) > 1:
        raise ValueError(
            f"A deployment's runs share one export, so they share its preprocessing; these come from exports "
            f"{_runs(exports)}"
        )
    kinds: dict[int, tuple[str, str]] = {}
    for r in runs:
        try:
            kinds[r["id"]] = kind_of(r["report"])
        except ValueError as exc:
            raise ValueError(f"Run {r['id']} {exc}; deploy a run with one target") from None
    binary = [r for r in runs if kinds[r["id"]][0] == "binary"]
    single = [r for r in runs if kinds[r["id"]][0] != "binary"]
    if binary and single:
        others = "; ".join(f"run {r['id']} is {kinds[r['id']][0]}" for r in single)
        raise ValueError(
            f"A deployment is one-vs-rest runs or a single run, not both: run {_runs([r['id'] for r in binary])} "
            f"{'is' if len(binary) == 1 else 'are'} one-vs-rest; {others}"
        )
    if len(single) > 1:
        raise ValueError(
            f"A multiclass or number run is deployed on its own, one run per deployment; got runs "
            f"{_runs([r['id'] for r in single])}"
        )
    targets = sorted({kinds[r["id"]][1] for r in binary})
    if len(targets) > 1:
        raise ValueError(f"One-vs-rest runs of one target share a deployment; these are of {', '.join(targets)}")
    by_class: dict[str, list[int]] = defaultdict(list)
    for r in binary:
        by_class[r["report"]["positive_class"]].append(r["id"])
    if dup := next(((c, ids) for c, ids in by_class.items() if len(ids) > 1), None):
        c, ids = dup
        raise ValueError(f"There are {WORDS.get(len(ids), len(ids))} runs for {c!r} (runs {_runs(ids)}); deploy one")
    per_run = per_run or {}
    if stray := sorted(set(per_run) - {r["id"] for r in runs}):
        raise ValueError(f"Thresholds were given for run {_runs(stray)}, which isn't being deployed")
    if no_threshold := sorted(set(per_run) & {r["id"] for r in single}):
        raise ValueError(f"Run {_runs(no_threshold)} isn't a one-vs-rest model, so it has no threshold")
    if rule is not None and not binary:
        raise ValueError("threshold_rule sets one-vs-rest models' thresholds, and this deployment has none")
    return exports[0], [_model(r, kinds[r["id"]], per_run.get(r["id"]) or rule or F1) for r in runs]


def normal_rule(rule: dict) -> dict:
    """The rule as stored: its kind and the one number that kind reads, nothing else that came with it."""
    kind = rule["kind"]
    if kind == "precision":
        return {"kind": kind, "min": float(rule["min"])}
    if kind == "fixed":
        return {"kind": kind, "value": float(rule["value"])}
    return {"kind": kind}


def with_file_hashes(models: list[dict], runs: list[dict]) -> list[dict]:
    """Each model with its file's SHA-256, which scoring checks before every load. A run without a model folder is
    left to scoring, which refuses it."""
    dirs = {r["id"]: r.get("model_dir") for r in runs}
    return [
        {**m, "sha256": scoring.file_sha256(Path(dirs[m["run_id"]]) / scoring.MODEL_FILE)} if dirs[m["run_id"]] else m
        for m in models
    ]


def _model(run: dict, kind: tuple[str, str], rule: dict) -> dict:
    report = run["report"]
    kind_name, target = kind
    out: dict[str, Any] = {
        "run_id": run["id"],
        "model_label": report.get("model_label"),
        "kind": kind_name,
        "target": target,
        "threshold": None,
        "threshold_rule": None,
    }
    if kind_name != "binary":
        return {**out, "label": target}
    curve = (report.get("threshold_curve") or {}).get("valid") or []
    try:
        threshold, note = choose_threshold(curve, rule)
    except ValueError as exc:
        raise ValueError(f"Run {run['id']}: {exc}") from None
    cls = report["positive_class"]
    return {
        **out,
        "label": cls,
        "positive_class": cls,
        "threshold": threshold,
        "threshold_rule": normal_rule(rule),
        **({"threshold_note": note} if note else {}),
    }


# =========================================================================== predict
def records_frame(records: list[dict], names: list[str]) -> pl.DataFrame:
    """The `names` columns of JSON records. Each column's type comes from all of its values, never the first record's:
    numbers of one kind stay numbers, and anything mixed is text, which scoring converts to the field's type."""
    cols = {}
    for name in names:
        values = []
        for i, rec in enumerate(records, 1):
            if name not in rec:
                if not any(name in r for r in records):
                    raise ValueError(f"Missing field: {name!r}")
                raise ValueError(f"Record {i} has no {name!r}; send null for a missing value")
            v = rec[name]
            if isinstance(v, (list, dict)):
                raise ValueError(f"Field {name!r} takes a single value, not a list or an object (record {i})")
            values.append(v)
        cols[name] = _column(name, values)
    return pl.DataFrame(cols)


def _column(name: str, values: list) -> pl.Series:
    kinds = {type(v) for v in values if v is not None}
    if kinds == {bool}:
        return pl.Series(name, values, dtype=pl.Boolean)
    if kinds == {int} and all(-INT64 <= v < INT64 for v in values if v is not None):
        return pl.Series(name, values, dtype=pl.Int64)
    if kinds == {float}:
        return pl.Series(name, values, dtype=pl.Float64)
    # Text keeps an integer exact past 2^53 and "7" as "7" rather than "7.0" for a field the dataset holds as text.
    return pl.Series(name, [None if v is None else _text(v) for v in values], dtype=pl.String)


def _text(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return repr(v) if isinstance(v, float) else str(v)


def labels(scorer: scoring.Scorer) -> list[str]:
    return [label for m in scorer.models for label in m.labels]


def results(out: scoring.ScoreResult, n: int) -> list[dict]:
    """One result per record: its scores, and (as they apply to the deployment) its top class, the classes that
    matched and whether none did."""
    scores = {k: [_number(x) for x in np.asarray(v, dtype=float)] for k, v in out.scores.items()}
    return [
        {
            "scores": {k: v[i] for k, v in scores.items()},
            "top": None if out.top is None else {"label": out.top[i][0], "score": _number(out.top[i][1])},
            "matches": None if out.matches is None else out.matches[i],
            "none_matched": None if out.none_matched is None else bool(out.none_matched[i]),
        }
        for i in range(n)
    ]


def _number(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


# =========================================================================== removal
def remove_for_dataset(conn, dataset_id: int) -> list[int]:
    """Delete the dataset's deployments (their keys and tests go with them) ahead of the dataset, whose exports they
    hold on to. `discard` them once the transaction commits."""
    rows = conn.execute("DELETE FROM meta.deployments WHERE dataset_id = %s RETURNING id", (dataset_id,)).fetchall()
    return [r["id"] for r in rows]


def discard(ids: list[int]) -> None:
    """Drop the deployments' loaded models and their files."""
    for i in ids:
        scoring.forget(i)
        shutil.rmtree(get_settings().deployments_dir / str(i), ignore_errors=True)


# =========================================================================== tests
RETIRED = "This deployment is retired; reactivate it to score with it."
FILES_GONE = "The deployment's export or model files were removed or unreadable"
TEST_CHUNK = 50_000
KEEP_TESTS = 20
SHOWN_RESULTS = 500
CONFUSION_CLASSES = 20
UPLOAD = "upload"  # + the uploaded file's extension
INCOMING = ".incoming-"
INPUT = "input.parquet"
RESULTS = "results.parquet"
MATCH_SEP = "; "


class _Gone(Exception):
    """The test's row was deleted while it ran (its deployment or dataset was deleted)."""


def scoring_spec(row: dict) -> dict:
    """What `scoring.load` reads of a deployment row."""
    return {k: row[k] for k in ("id", "export_id", "models", "preprocessor_sha256")}


def tests_dir(deployment_id: int) -> Path:
    return get_settings().deployments_dir / str(deployment_id) / "tests"


def test_dir(deployment_id: int, test_id: int) -> Path:
    return tests_dir(deployment_id) / str(test_id)


def prune_tests(conn, deployment_id: int) -> list[int]:
    """Delete the finished tests older than the deployment's newest KEEP_TESTS; `discard_tests` their files once the
    transaction commits. A test still running is left to its job."""
    rows = conn.execute(
        """DELETE FROM meta.deployment_tests WHERE deployment_id = %s AND status IN ('done', 'failed')
              AND id NOT IN (SELECT id FROM meta.deployment_tests WHERE deployment_id = %s ORDER BY id DESC LIMIT %s)
           RETURNING id""",
        (deployment_id, deployment_id, KEEP_TESTS),
    ).fetchall()
    return [r["id"] for r in rows]


def discard_tests(deployment_id: int, ids: list[int]) -> None:
    for i in ids:
        shutil.rmtree(test_dir(deployment_id, i), ignore_errors=True)


def incoming_dir(deployment_id: int, owner: int) -> Path:
    """Where an upload lands before its test exists; named by the process taking it, so a restart can tell it was
    abandoned."""
    return tests_dir(deployment_id) / f"{INCOMING}{owner}-{secrets.token_hex(8)}"


def clear_uploads(abandoned) -> None:
    """Remove the incoming uploads whose process `abandoned(owner)` says is gone."""
    for folder in get_settings().deployments_dir.glob(f"*/tests/{INCOMING}*"):
        owner = folder.name[len(INCOMING) :].split("-")[0]
        if not owner.isdigit() or abandoned(int(owner)):
            shutil.rmtree(folder, ignore_errors=True)


def result_columns(models: list[dict], labels: list[str], id_field: str | None, label_field: str | None) -> dict:
    """The results file's column for each part of a result. The file's identifier and label keep their field's name;
    a generated column that would clash with one gets a leading underscore."""
    taken = {c for c in (id_field, label_field) if c}

    def free(name: str) -> str:
        while name in taken:
            name = "_" + name
        taken.add(name)
        return name

    kinds = {m["kind"] for m in models}
    top, ovr = bool(kinds - {"regression"}), "binary" in kinds
    return {
        "row": free("row"),
        "id": id_field,
        "label": label_field,
        "top": free("top") if top else None,
        "top_score": free("top_score") if top else None,
        "matches": free("matches") if ovr else None,
        "none_matched": free("none_matched") if ovr else None,
        "scores": {label: free(f"score_{label}") for label in labels},
    }


def test_columns(test: dict, models: list[dict]) -> dict:
    """`result_columns` of a finished test row."""
    label_field = models[0]["target"] if test["has_labels"] else None
    return result_columns(models, test["labels"] or [], test["id_field"], label_field)


def shown_results(path: Path, cols: dict, n: int = SHOWN_RESULTS) -> list[dict]:
    """The first `n` results as predict gives them, with each record's row number, identifier and label."""
    df = pl.read_parquet(path, n_rows=n)

    def col(role: str) -> list:
        return df[cols[role]].to_list() if cols[role] else [None] * df.height

    rows, ids, given, top, top_score = col("row"), col("id"), col("label"), col("top"), col("top_score")
    matches, none = col("matches"), col("none_matched")
    scores = {label: df[c].to_list() for label, c in cols["scores"].items()}
    return [
        {
            "row": rows[i],
            "id": ids[i],
            "label": given[i],
            "top": None if top[i] is None else {"label": top[i], "score": top_score[i]},
            "matches": matches[i],
            "none_matched": none[i],
            "scores": {label: v[i] for label, v in scores.items()},
        }
        for i in range(df.height)
    ]


def csv_chunks(path: Path, cols: dict) -> Iterator[bytes]:
    """The results file as CSV, a chunk at a time, `matches` joined with "; "."""

    def text(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col(cols["matches"]).list.join(MATCH_SEP)) if cols["matches"] else df

    first = True
    for batch in pq.ParquetFile(path).iter_batches(batch_size=TEST_CHUNK):
        yield text(pl.from_arrow(batch)).write_csv(include_header=first).encode()
        first = False
    if first:
        yield text(pl.read_parquet(path, n_rows=0)).write_csv().encode()


def _set(test_id: int, finished: bool = False, **cols: Any) -> bool:
    """Update the test's row; False when it no longer exists."""
    query = sql.SQL("UPDATE meta.deployment_tests SET {}, updated_at = now(){} WHERE id = %s RETURNING id").format(
        sql.SQL(", ").join(sql.SQL("{} = %s").format(sql.Identifier(k)) for k in cols),
        sql.SQL(", finished_at = now()" if finished else ""),
    )
    with connection() as conn:
        return conn.execute(query, [*cols.values(), test_id]).fetchone() is not None


def run_test(test_id: int) -> None:
    """The job behind a test: score the uploaded file through its deployment as predict does, TEST_CHUNK rows at a
    time, into results.parquet, with metrics when the file carries the deployment's target."""
    with connection() as conn:
        t = conn.execute(
            """SELECT t.id, t.deployment_id, t.filename, d.dataset_id, d.export_id, d.models, d.preprocessor_sha256,
                      d.status FROM meta.deployment_tests t JOIN meta.deployments d ON d.id = t.deployment_id
                WHERE t.id = %s""",
            (test_id,),
        ).fetchone()
    if t is None:
        return
    folder = test_dir(t["deployment_id"], test_id)
    try:
        if not _set(test_id, status="running", progress=0, status_message="Reading the file"):
            raise _Gone
        result = _score_test(t, folder)
        (folder / INPUT).unlink(missing_ok=True)
        if not _set(test_id, finished=True, status="done", progress=1, status_message="Done", error=None, **result):
            raise _Gone
    except _Gone:
        shutil.rmtree(folder, ignore_errors=True)
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, ValueError):  # what the test can't use, said as it is: ModelChanged among them
            error = str(exc)
        else:
            log.exception("deployment test %s failed", test_id)
            error = f"{type(exc).__name__}: {exc}"
        shutil.rmtree(folder, ignore_errors=True)
        _set(test_id, finished=True, status="failed", status_message="Failed", error=error)


def _score_test(t: dict, folder: Path) -> dict:
    """Score the test's file into `folder`/results.parquet; the columns its row gets."""
    suffix = Path(t["filename"]).suffix.lower()
    upload, data = folder / f"{UPLOAD}{suffix}", folder / INPUT
    if t["status"] != "active":
        raise ValueError(RETIRED)
    try:
        notes = ingest.to_parquet(upload, ingest.UPLOAD_FORMATS[suffix], data)
    except pl.exceptions.PolarsError as exc:
        raise ValueError(f"The file can't be read: {exc}") from exc
    upload.unlink()
    names = pq.read_schema(data).names
    target, kind = t["models"][0]["target"], t["models"][0]["kind"]
    label_field = target if target in names else None
    with connection() as conn:
        try:
            scorer = scoring.load(conn, scoring_spec(t))
        except OSError as exc:
            log.warning("deployment %s can't load its files: %s", t["deployment_id"], exc)
            raise ValueError(FILES_GONE) from exc
        scorer.require(names)
        id_field = _identifier(conn, t["dataset_id"], names)
        reading = _label_reading(conn, t["export_id"], target, kind) if label_field else None
    scored_by = labels(scorer)
    cols = result_columns(t["models"], scored_by, id_field, label_field)
    source = pq.ParquetFile(data)
    total = source.metadata.num_rows
    read = list(dict.fromkeys([*scorer.required, *(c for c in (id_field, label_field) if c)]))
    failed: dict[str, int] = defaultdict(int)
    given: dict[str, int] = defaultdict(int)
    tmp = folder / f"{RESULTS}.tmp"
    writer, done, unreadable = None, 0, 0
    try:
        for batch in source.iter_batches(batch_size=TEST_CHUNK, columns=read):
            df = pl.from_arrow(batch)
            out = scorer.score(df.select(scorer.required))
            for name, n in out.warnings.items():
                failed[name] += n
            for name in scorer.required:
                given[name] += int(scoring.missing_as_null(df[name]).is_not_null().sum())
            chunk, bad = _results_frame(df, out, cols, done, reading)
            unreadable += bad
            table = chunk.to_arrow()
            writer = writer or pq.ParquetWriter(tmp, table.schema)
            writer.write_table(table)
            done += df.height
            if not _set(t["id"], progress=round(done / total, 4), status_message=f"Scored {done:,} of {total:,} rows"):
                raise _Gone
        if writer is None:
            empty = pl.read_parquet(data, columns=read, n_rows=0)
            pq.write_table(_results_frame(empty, None, cols, 0, reading)[0].to_arrow(), tmp)
    finally:
        if writer is not None:
            writer.close()
    os.replace(tmp, folder / RESULTS)
    warnings = {
        n: {"failed": failed[n], "values": given[n], "flagged": failed[n] > given[n] / 2}
        for n in scorer.required
        if failed[n]
    }
    metrics = test_metrics(folder / RESULTS, cols, scorer.models, unreadable) if label_field else None
    return {
        "rows": done,
        "has_labels": label_field is not None,
        "metrics": None if metrics is None else jsonb(metrics),
        "warnings": jsonb(warnings),
        "notes": notes,
        "id_field": id_field,
        "labels": scored_by,
    }


def _identifier(conn, dataset_id: int, names: list[str]) -> str | None:
    """The first of the dataset's identifier fields (as decided, else as inferred) that the file carries."""
    rows = conn.execute(
        """SELECT f.name FROM meta.fields f LEFT JOIN meta.field_decisions d ON d.field_id = f.id
            WHERE f.dataset_id = %s AND coalesce(d.semantic_type, f.inferred_type) = 'identifier' ORDER BY f.ordinal""",
        (dataset_id,),
    ).fetchall()
    return next((r["name"] for r in rows if r["name"] in names), None)


def _label_reading(conn, export_id: int, target: str, kind: str) -> dict:
    """How the export's training read its target, so a file's labels are read the same way: its classes, bands or
    group, its type in the dataset, and how its column writes decimals."""
    exp = conn.execute("SELECT dataset_id, output_dir, config FROM meta.exports WHERE id = %s", (export_id,)).fetchone()
    manifest = json.loads((Path(exp["output_dir"]) / "manifest.json").read_text())
    bins = (manifest.get("target_bins") or {}).get(target)
    comma = (bins or {}).get("decimal_comma")
    if comma is None:  # what training read the decimals by, for a manifest that doesn't record it
        row = conn.execute(
            """SELECT coalesce((stats ->> 'decimal_comma')::boolean, false) AS c FROM meta.fields
                WHERE dataset_id = %s AND name = %s""",
            (exp["dataset_id"], target),
        ).fetchone()
        comma = bool(row and row["c"])
    schema = ((exp["config"] or {}).get("input_schema") or {}).get(target)
    return {
        "number": kind == "regression",
        "bins": bins,
        "classes": (manifest.get("target_classes") or {}).get(target) or [],
        "dtype": scoring.parse_dtype(schema["dtype"]) if schema else None,
        "decimal_comma": comma,
    }


def _label_values(s: pl.Series, how: dict) -> tuple[pl.Series, int]:
    """A file's labels as training named its classes (`target_labels`), or a number target's as numbers; and how many
    given labels of a number target aren't numbers. A class label that names no class keeps its own text."""
    if how["number"]:
        given = scoring.missing_as_null(s)
        out = coerce_numeric(given, how["decimal_comma"])
        return out.alias(s.name), int(given.is_not_null().sum() - out.is_not_null().sum())
    named = target_labels(s, how["bins"], how["classes"], how["dtype"], how["decimal_comma"])
    return pl.Series(s.name, [None if pd.isna(v) else v for v in named], dtype=pl.String), 0


def _results_frame(
    df: pl.DataFrame, out: scoring.ScoreResult | None, cols: dict, start: int, reading: dict | None
) -> tuple[pl.DataFrame, int]:
    """One chunk of results, and how many of its number labels aren't numbers; with `out` None, none (the columns of
    a file with no rows)."""
    data: dict[str, pl.Series] = {cols["row"]: pl.Series(np.arange(start + 1, start + df.height + 1), dtype=pl.Int64)}
    unreadable = 0
    if cols["id"]:
        data[cols["id"]] = df[cols["id"]]
    if cols["label"]:
        data[cols["label"]], unreadable = _label_values(df[cols["label"]], reading)
    if cols["top"]:
        top = (out and out.top) or []
        data[cols["top"]] = pl.Series([label for label, _ in top], dtype=pl.String)
        data[cols["top_score"]] = pl.Series([score for _, score in top], dtype=pl.Float64)
    if cols["matches"]:
        data[cols["matches"]] = pl.Series((out and out.matches) or [], dtype=pl.List(pl.String))
        data[cols["none_matched"]] = pl.Series((out and out.none_matched) or [], dtype=pl.Boolean)
    for label, c in cols["scores"].items():
        data[c] = pl.Series([] if out is None else np.asarray(out.scores[label], dtype=float), dtype=pl.Float64)
    return pl.DataFrame({k: v.alias(k) for k, v in data.items()}), unreadable


# --------------------------------------------------------------------------- metrics
def _codes(path: Path, col: str) -> tuple[np.ndarray, list[str]]:
    """A text column as codes into its distinct values (-1 where missing), read dictionary-encoded so the column's
    text is never held whole."""
    table = pq.read_table(path, columns=[col], read_dictionary=[col]).unify_dictionaries()
    if not table.num_rows:
        return np.empty(0, dtype=np.int64), []
    arr = table.column(0).combine_chunks()
    return pc.fill_null(arr.indices, -1).to_numpy().astype(np.int64), arr.dictionary.to_pylist()


def test_metrics(path: Path, cols: dict, models: tuple[scoring.Model, ...], unreadable: int = 0) -> dict:
    """What a labelled file says of the deployment, over the rows with a label. A row whose label has no model counts
    as wrong in top-1 accuracy and has no row of its own; the nothing-matched share is over all rows. Labels and tops
    are compared as codes: the labels the models score first, then the file's other labels."""

    def read(c: str) -> np.ndarray:
        return pl.read_parquet(path, columns=[c]).to_series().to_numpy()

    if models[0].kind == "regression":
        y = read(cols["label"]).astype(float)
        missing = int(np.isnan(y).sum()) - unreadable
        return {
            "labelled_rows": len(y) - missing,
            "rows_without_label": missing,
            "rows_label_not_a_number": unreadable,
            "kind": "number",
            **_errors(y, read(cols["scores"][models[0].labels[0]])),
        }
    known = [c for m in models for c in m.labels]
    lab_idx, lab_names = _codes(path, cols["label"])
    top_idx, top_names = _codes(path, cols["top"])
    vocab = known + [v for v in lab_names if v not in set(known)]
    code = {v: k for k, v in enumerate(vocab)}
    has = lab_idx >= 0
    lab = np.array([code[v] for v in lab_names], dtype=np.int64)[lab_idx[has]]
    top = np.array([code[v] for v in top_names], dtype=np.int64)[top_idx[has]]
    classes = []
    for m in models:
        for c in m.labels:
            score = read(cols["scores"][c])[has]
            pred = score >= m.threshold if m.kind == "binary" else top == code[c]
            classes.append(_class_metrics(c, lab == code[c], score, pred, m.threshold))
    none = read(cols["none_matched"]) if cols["none_matched"] else None
    return {
        "labelled_rows": int(has.sum()),
        "rows_without_label": int((~has).sum()),
        "kind": "classes",
        "top1_accuracy": float((top == lab).mean()) if lab.size else None,
        "rows_label_without_model": int((lab >= len(known)).sum()),
        "nothing_matched_share": float(none.mean()) if none is not None and none.size else None,
        "classes": classes,
        "confusion": _confusion(lab, top, vocab),
    }


def _class_metrics(label: str, y: np.ndarray, score: np.ndarray, pred: np.ndarray, threshold: float | None) -> dict:
    tp, positives, predicted = int((y & pred).sum()), int(y.sum()), int(pred.sum())
    precision = tp / predicted if predicted else None
    recall = tp / positives if positives else None
    f1 = None
    if precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "label": label,
        "support": positives,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": float(average_precision_score(y, score)) if positives else None,
    }


def _confusion(lab: np.ndarray, top: np.ndarray, vocab: list[str]) -> dict:
    """Counts of `top` for each of the CONFUSION_CLASSES most frequent labels (codes into `vocab`): one column per
    listed label, and a last one for a top outside the list."""
    counts = np.bincount(lab, minlength=len(vocab))
    listed = sorted((k for k in range(len(vocab)) if counts[k]), key=lambda k: (-counts[k], vocab[k]))
    listed = listed[:CONFUSION_CLASSES]
    at = np.full(len(vocab), -1)
    at[listed] = np.arange(len(listed))
    grid = np.zeros((len(listed), len(listed) + 1), dtype=int)
    rows = at[lab]
    cols = np.where(at[top] >= 0, at[top], len(listed))
    keep = rows >= 0
    np.add.at(grid, (rows[keep], cols[keep]), 1)
    return {"labels": [vocab[k] for k in listed], "counts": grid.tolist()}


def _errors(y: np.ndarray, p: np.ndarray) -> dict:
    p = p.astype(float)
    ok = np.isfinite(y) & np.isfinite(p)
    err, y = p[ok] - y[ok], y[ok]
    if not err.size:
        return {"mae": None, "rmse": None, "r2": None}
    sst = float(((y - y.mean()) ** 2).sum())
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "r2": 1 - float((err**2).sum()) / sst if sst > 0 else None,
    }
