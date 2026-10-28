"""Batches: which classes a one-vs-rest batch trains, and the combinations a search tries."""

import itertools
import math
import random
from collections.abc import Callable
from typing import Any

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


def grid(space: dict[str, list]) -> list[dict]:
    """Every combination of the lists, in the space's order."""
    for name, values in space.items():
        if _is_range(values):
            raise ValueError(f"{name}: a grid takes lists of values; use random sampling for a range")
        if not values:
            raise ValueError(f"{name}: no values to try")
    return [dict(zip(space, combo)) for combo in itertools.product(*space.values())]


def _check_range(name: str, spec: dict) -> None:
    bounds = (spec.get("low"), spec.get("high"))
    if not all(isinstance(b, (int, float)) and not isinstance(b, bool) and math.isfinite(b) for b in bounds):
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
