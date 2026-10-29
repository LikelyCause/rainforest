"""Batches: which classes a one-vs-rest batch trains, the combinations a search tries, the feeder that submits a
batch's runs one at a time, and the rows of its results.

A batch's runs are inserted `pending` with no owner. The feeder gives the next one an owner and submits it only when
none of the batch's runs is submitted or running, so other jobs queue between them instead of behind the whole batch.
"""

import itertools
import json
import math
import random
import statistics
from collections.abc import Callable
from typing import Any

WAITING = "Waiting in its batch"
# Fewer test positives than this and a class's test metrics say little, so its row is flagged weak.
WEAK_TEST_POSITIVES = 30

LOWER_IS_BETTER = frozenset({"log_loss", "rmse", "mae", "mape"})
NUMBER_METRICS = frozenset({"rmse", "mae", "r2", "mape", "band_accuracy", "band_f1_macro"})
CLASS_METRICS = frozenset(
    {
        "accuracy",
        "balanced_accuracy",
        "log_loss",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "pr_auc_macro",
        "roc_auc_ovr_macro",
        "precision",
        "recall",
        "f1",
        "specificity",
        "roc_auc",
        "pr_auc",
    }
)

# Per class, the positives in each split: {"train": n, "valid": n, "test": n}; "train" and "valid" are required.
# "valid_rows" (all rows of the validation split, not just this class's) is optional; 0 means nothing to calibrate on.
ClassCounts = dict[str, dict[str, int]]


def plan_classes(class_counts: ClassCounts, wanted: list[str] | None, min_pos: int) -> tuple[list[str], list[dict]]:
    """The classes to train and, for every class not trained, why: not ticked, too few training positives, or
    nothing to calibrate and threshold on in the validation split (such a class would train but could not be
    calibrated, so it is skipped here rather than failing its run). One reason per class, "not selected" first."""
    if wanted is not None and (unknown := [c for c in wanted if c not in class_counts]):
        raise ValueError(f"Not classes of this target: {', '.join(map(str, unknown))}")
    for cls, n in class_counts.items():
        for key in ("train", "valid"):
            if key not in n:
                raise ValueError(f"{cls}: no {key} count")
    trained: list[str] = []
    skipped: list[dict] = []
    for cls, n in class_counts.items():
        if wanted is not None and cls not in wanted:
            reason = "not selected"
        elif n["train"] < min_pos:
            reason = f"fewer than {min_pos} positives in training"
        elif n.get("valid_rows") == 0:
            reason = "no validation rows"
        elif n["valid"] == 0:
            reason = "no validation positives"
        else:
            trained.append(cls)
            continue
        skipped.append({"class": cls, "reason": reason})
    return trained, skipped


def search_classes_default(class_counts: ClassCounts, min_pos: int) -> list[str]:
    """The classes a one-vs-rest search is tuned on: the most frequent, the median and the rarest of those that
    train, distinct, so settings are judged on a common, a typical and a hard class."""
    trained, _ = plan_classes(class_counts, None, min_pos)
    by_frequency = sorted(trained, key=lambda c: (-class_counts[c]["train"], c))
    if not by_frequency:
        return []
    picks = [by_frequency[0], by_frequency[(len(by_frequency) - 1) // 2], by_frequency[-1]]
    return list(dict.fromkeys(picks))


def _is_range(spec: Any) -> bool:
    return isinstance(spec, dict)


def grid_size(space: dict[str, list]) -> int:
    """How many combinations `grid` would give, without building them."""
    for name, values in space.items():
        if _is_range(values):
            raise ValueError(f"{name}: a grid takes lists of values; use random sampling for a range")
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"{name}: give a list of values")
        if not values:
            raise ValueError(f"{name}: no values to try")
    return math.prod(len(v) for v in space.values())


def grid(space: dict[str, list]) -> list[dict]:
    """Every combination of the lists, in the space's order."""
    grid_size(space)
    return [dict(zip(space, combo)) for combo in itertools.product(*space.values())]


def _finite(b: Any) -> bool:
    try:
        return isinstance(b, (int, float)) and not isinstance(b, bool) and math.isfinite(b)
    except OverflowError:  # an int too large for a float
        return False


def _check_range(name: str, spec: dict) -> None:
    if not (_finite(spec.get("low")) and _finite(spec.get("high"))):
        raise ValueError(f"{name}: a range needs a finite numeric low and high")
    if spec["low"] > spec["high"]:
        raise ValueError(f"{name}: the range's low is above its high")
    if spec.get("log") and spec["low"] <= 0:
        raise ValueError(f"{name}: a log range needs a low above 0")
    if spec.get("integer") and math.ceil(spec["low"]) > math.floor(spec["high"]):
        raise ValueError(f"{name}: the range holds no whole number")


def _draw(rng: random.Random, name: str, spec: list | dict) -> Any:
    if not _is_range(spec):
        if not isinstance(spec, (list, tuple)) or not spec:
            raise ValueError(f"{name}: give a non-empty list of values or a range")
        return rng.choice(spec)
    low, high = spec["low"], spec["high"]
    if spec.get("integer"):
        lo, hi = math.ceil(low), math.floor(high)
        if not spec.get("log"):
            return rng.randint(lo, hi)
        # Spread over half a step beyond each end integer, so rounding gives the ends the mass of the others.
        return min(max(round(math.exp(rng.uniform(math.log(lo - 0.5), math.log(hi + 0.5)))), lo), hi)
    if spec.get("log"):
        return min(max(math.exp(rng.uniform(math.log(low), math.log(high))), low), high)
    return rng.uniform(low, high)


def random_samples(
    space: dict[str, dict | list],
    n: int,
    seed: int,
    valid: Callable[[dict], str | None],
    max_redraws: int | None = None,
) -> tuple[list[dict], int]:
    """`n` distinct combinations drawn from lists and ranges, and how many draws were thrown away. `valid` returns
    a reason to reject a combination (asked once per combination); rejected and repeated draws are redrawn until
    `max_redraws` (default 10 n)."""
    for name, spec in space.items():
        if _is_range(spec):
            _check_range(name, spec)
        elif not isinstance(spec, (list, tuple)) or not spec:
            raise ValueError(f"{name}: give a non-empty list of values or a range")
    limit = 10 * n if max_redraws is None else max_redraws
    rng = random.Random(seed)
    samples: list[dict] = []
    drawn: dict[str, str | None] = {}  # combination -> why it was rejected, or None when it was kept
    redraws = 0
    while len(samples) < n:
        sample = {name: _draw(rng, name, spec) for name, spec in space.items()}
        key = repr(sorted(sample.items()))
        if key in drawn:
            reason = drawn[key] or "already drawn"
        else:
            reason = drawn[key] = valid(sample)
            if reason is None:
                samples.append(sample)
                continue
        redraws += 1
        if redraws > limit:
            raise ValueError(
                f"Could not draw {n} distinct valid combinations within {limit} redraws "
                f"({len(samples)} found); the last was rejected: {reason}"
            )
    return samples, redraws


# ---- ranking ------------------------------------------------------------------------------------------------------

_BINARY_ONLY = frozenset({"precision", "recall", "f1", "specificity", "roc_auc", "pr_auc"})
_MULTICLASS_ONLY = frozenset({"pr_auc_macro", "roc_auc_ovr_macro"})


def default_ranking(kind: str | None) -> str:
    """The validation metric a search ranks on: PR-AUC for a binary (or one-vs-rest) target, macro-F1 for several
    classes, R² for a number."""
    return {"multiclass": "f1_macro", "regression": "r2"}.get(kind or "", "pr_auc")


def ranking_problem(metric: str, kind: str | None) -> str | None:
    """Why `metric` can't rank runs of a target of this kind, or None."""
    if kind == "regression":
        allowed = NUMBER_METRICS
    else:
        allowed = CLASS_METRICS - (_BINARY_ONLY if kind == "multiclass" else _MULTICLASS_ONLY)
    if metric not in allowed:
        return f"{metric!r} isn't a metric of this target (choose from {', '.join(sorted(allowed))})"
    return None


def _number(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def run_summary(run: dict, spec: dict, weak_pr_auc: float) -> dict:
    """One row of a batch's results: the run's state and, once it has them, the metrics its batch kind compares."""
    from .modeling.train import choose_threshold

    config, report, metrics = run.get("config") or {}, run.get("report") or {}, run.get("metrics") or {}
    target = config.get("target") or spec.get("ranked_target")
    row: dict[str, Any] = {
        "run_id": run["id"],
        "status": run["status"],
        "status_message": run.get("status_message"),
        "error": run.get("error"),
        "label": report.get("model_label"),
        "positive_class": config.get("positive_class"),
        "retried": bool(report.get("retried")),
        "fit_seconds": report.get("fit_seconds"),
        "finished_at": run.get("finished_at"),
    }

    def metric(split: str, name: str) -> float | None:
        return _number(((metrics.get(split) or {}).get(target) or {}).get(name))

    plan = config.get("batch_plan") or {}
    if spec["kind"] == "search":
        ranking = spec["ranking"]
        cv = (((report.get("cross_validation") or {}).get("targets") or {}).get(target) or {}).get(ranking) or {}
        params = config.get("params") or {}
        settings = plan.get("settings") or {k: params.get(k) for k in spec["space"]}
        row.update(settings=settings, value=metric("valid", ranking), cv_std=cv.get("std"))
    if row["positive_class"] is not None:
        if stored := report.get("f1_threshold"):
            threshold, note = stored["threshold"], stored.get("note")
            f1 = {split: _number(stored.get(f"{split}_f1")) for split in ("valid", "test")}
        else:  # a run trained before the choice was stored
            curve = report.get("threshold_curve") or {}
            threshold, note = choose_threshold(curve["valid"], {"kind": "f1"}) if curve.get("valid") else (None, None)
            points = {s: next((p for p in curve.get(s) or [] if p["threshold"] == threshold), None) for s in curve}
            f1 = {split: _number((points.get(split) or {}).get("f1")) for split in ("valid", "test")}
        positives = report.get("positives") or plan.get("positives")
        row.update(
            positives=positives,
            threshold=threshold,
            threshold_note=note,
            calibration=report.get("calibration"),
            **{
                split: {"pr_auc": metric(split, "pr_auc"), "roc_auc": metric(split, "roc_auc"), "f1": f1[split]}
                for split in ("valid", "test")
            },
        )
        row["weak"] = None
        if run["status"] == "done":
            valid_pr_auc = row["valid"]["pr_auc"]
            row["weak"] = (valid_pr_auc is not None and valid_pr_auc < weak_pr_auc) or (
                (positives or {}).get("test", 0) < WEAK_TEST_POSITIVES
            )
    return row


def leaderboard(rows: list[dict], ranking: str) -> list[dict]:
    """A search's combinations, best first by the ranking metric on validation: for a one-vs-rest search, the mean
    over its classes. A combination is ranked only once every one of its runs has the metric, so a partial mean never
    passes for the whole."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(json.dumps(r["settings"], sort_keys=True, default=str), []).append(r)
    entries = []
    for runs in groups.values():
        values = [r["value"] for r in runs]
        stds = [r["cv_std"] for r in runs if _number(r["cv_std"]) is not None]
        entries.append(
            {
                "settings": runs[0]["settings"],
                "run_ids": [r["run_id"] for r in runs],
                "value": statistics.fmean(values) if all(v is not None for v in values) else None,
                "classes": {r["positive_class"]: r["value"] for r in runs if r["positive_class"] is not None},
                "cv_std": statistics.fmean(stds) if len(stds) == len(runs) else None,
                "fit_seconds": sum(r["fit_seconds"] or 0 for r in runs),
            }
        )
    sign = 1 if ranking in LOWER_IS_BETTER else -1
    entries.sort(key=lambda e: (e["value"] is None, sign * (e["value"] or 0)))
    for i, e in enumerate(entries):
        e["rank"] = i + 1 if e["value"] is not None else None
    return entries


# ---- feeding ------------------------------------------------------------------------------------------------------


def running_batch(conn, *, dataset_id: int | None = None, experiment_id: int | None = None) -> dict | None:
    """A batch with runs still to finish on the dataset or in the experiment (the oldest), or None. Its waiting runs hold off
    whatever would pull the data or the experiment from under them, so the refusal can say which batch to cancel."""
    return conn.execute(
        """SELECT b.id, b.name FROM meta.run_batches b JOIN meta.exports e ON e.id = b.export_id
            WHERE b.status = 'running' AND (%(ds)s::bigint IS NULL OR e.dataset_id = %(ds)s)
              AND EXISTS (SELECT 1 FROM meta.model_runs r WHERE r.batch_id = b.id AND r.status IN ('pending', 'running'))
              AND (%(x)s::bigint IS NULL OR b.experiment_id = %(x)s)
            ORDER BY b.id LIMIT 1""",
        {"ds": dataset_id, "x": experiment_id},
    ).fetchone()


def feed(batch_id: int) -> None:
    """Submit the batch's next waiting run unless one of its runs is already submitted or running; with none left,
    the batch is done. Called when a batch starts, when one of its runs ends, and after a restart; the batch row's
    lock keeps two callers from both submitting."""
    from . import jobs
    from .db import connection
    from .routers.runs import _run

    owner = jobs.owner_id()
    with connection() as conn:
        batch = conn.execute(
            """SELECT b.status, e.output_dir FROM meta.run_batches b JOIN meta.exports e ON e.id = b.export_id
                WHERE b.id = %s FOR UPDATE OF b""",
            (batch_id,),
        ).fetchone()
        if batch is None or batch["status"] != "running":
            return
        if conn.execute(
            """SELECT 1 FROM meta.model_runs WHERE batch_id = %s
                  AND (status = 'running' OR (status = 'pending' AND owner IS NOT NULL)) LIMIT 1""",
            (batch_id,),
        ).fetchone():
            return
        run = conn.execute(
            """UPDATE meta.model_runs SET owner = %s, status_message = 'Queued', updated_at = now()
                WHERE id = (SELECT id FROM meta.model_runs WHERE batch_id = %s AND status = 'pending' AND owner IS NULL
                             ORDER BY id LIMIT 1)
            RETURNING id, config""",
            (owner, batch_id),
        ).fetchone()
        if run is None:
            conn.execute("UPDATE meta.run_batches SET status = 'done', finished_at = now() WHERE id = %s", (batch_id,))
            return
    jobs.submit(_run, run["id"], batch["output_dir"], run["config"])


def resume_all() -> None:
    """After a restart: every running batch queues the run the restart interrupted once more (marked `retried`; a
    second interruption leaves it failed), then carries on with its waiting runs."""
    from . import jobs
    from .db import connection

    with connection() as conn:
        conn.execute(
            """UPDATE meta.model_runs r
                  SET status = 'pending', status_message = %s, error = NULL, owner = NULL, finished_at = NULL,
                      updated_at = now(), report = coalesce(r.report, '{}') || '{"retried": true}'
                 FROM meta.run_batches b
                WHERE b.id = r.batch_id AND b.status = 'running' AND r.status = 'failed' AND r.error = %s
                  AND r.report -> 'retried' IS NULL""",
            (WAITING, jobs.INTERRUPTED),
        )
        running = [
            b["id"] for b in conn.execute("SELECT id FROM meta.run_batches WHERE status = 'running' ORDER BY id")
        ]
    for batch_id in running:
        feed(batch_id)
