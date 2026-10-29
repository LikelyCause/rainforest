"""Deployments: a frozen set of finished runs of one export, what may share one, their thresholds, the API keys that
may call a deployment's predict endpoint, and turning predict's JSON records into the frame `scoring` reads."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from . import scoring
from .config import get_settings
from .db import connection
from .modeling.train import choose_threshold

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
