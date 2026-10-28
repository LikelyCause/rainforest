"""Batches: which classes a one-vs-rest batch trains, and the combinations a search tries."""

import itertools
import math
import random
from collections.abc import Callable
from typing import Any

# Per class, the positives in each split: {"train": n, "valid": n, "test": n}. "valid_rows" (all rows of the
# validation split, not just this class's) is optional; 0 means there is nothing to calibrate on.
ClassCounts = dict[str, dict[str, int]]


def plan_classes(class_counts: ClassCounts, wanted: list[str] | None, min_pos: int) -> tuple[list[str], list[dict]]:
    """The classes to train and, for each one asked for but not trained, why. A class that trains but cannot be
    calibrated and thresholded on the validation split is skipped here rather than failing its run."""
    if wanted is not None and (unknown := [c for c in wanted if c not in class_counts]):
        raise ValueError(f"Not classes of this target: {', '.join(map(str, unknown))}")
    trained: list[str] = []
    skipped: list[dict] = []
    for cls, n in class_counts.items():
        if wanted is not None and cls not in wanted:
            continue
        if n.get("train", 0) < min_pos:
            reason = f"fewer than {min_pos} positives in training"
        elif n.get("valid_rows") == 0:
            reason = "no validation rows"
        elif n.get("valid", 0) == 0:
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
    if not isinstance(spec.get("low"), (int, float)) or not isinstance(spec.get("high"), (int, float)):
        raise ValueError(f"{name}: a range needs a numeric low and high")
    if spec["low"] > spec["high"]:
        raise ValueError(f"{name}: the range's low is above its high")
    if spec.get("log") and spec["low"] <= 0:
        raise ValueError(f"{name}: a log range needs a low above 0")
    if spec.get("integer") and math.ceil(spec["low"]) > math.floor(spec["high"]):
        raise ValueError(f"{name}: the range holds no whole number")


def _draw(rng: random.Random, name: str, spec: list | dict) -> Any:
    if not _is_range(spec):
        if not spec:
            raise ValueError(f"{name}: no values to try")
        return rng.choice(spec)
    low, high = spec["low"], spec["high"]
    if spec.get("log"):
        value = math.exp(rng.uniform(math.log(low), math.log(high)))
        value = min(max(value, low), high)
    elif spec.get("integer"):
        return rng.randint(math.ceil(low), math.floor(high))
    else:
        value = rng.uniform(low, high)
    if spec.get("integer"):
        return min(max(round(value), math.ceil(low)), math.floor(high))
    return value


def random_samples(
    space: dict[str, dict | list],
    n: int,
    seed: int,
    valid: Callable[[dict], str | None],
    max_redraws: int | None = None,
) -> tuple[list[dict], int]:
    """`n` distinct combinations drawn from lists and ranges, and how many draws were thrown away. `valid` returns
    a reason to reject a combination; rejected and repeated draws are redrawn until `max_redraws` (default 10 n)."""
    for name, spec in space.items():
        if _is_range(spec):
            _check_range(name, spec)
    limit = 10 * n if max_redraws is None else max_redraws
    rng = random.Random(seed)
    samples: list[dict] = []
    seen: set[str] = set()
    redraws = 0
    while len(samples) < n:
        sample = {name: _draw(rng, name, spec) for name, spec in space.items()}
        key = repr(sorted(sample.items()))
        reason = "already drawn" if key in seen else valid(sample)
        if reason is None:
            seen.add(key)
            samples.append(sample)
            continue
        redraws += 1
        if redraws > limit:
            raise ValueError(
                f"Could not draw {n} distinct valid combinations within {limit} redraws "
                f"({len(samples)} found); the last was rejected: {reason}"
            )
    return samples, redraws
