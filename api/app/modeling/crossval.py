"""Cross-validation of a run on its training rows: which rows each fold holds out, and the mean ± spread of each
metric across folds. The run's model is still the one fit on every training row; these scores only say how stable
it is."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def folds(y_strat: np.ndarray | None, n: int, k: int, seed: int,
          strat_name: str | None) -> tuple[list[tuple[np.ndarray, np.ndarray]], str | None, bool]:
    """(train rows, held-out rows) for each of `k` folds of `n` rows, stratified on `y_strat` when every class has
    at least `k` rows. Returns the folds, a note when they differ from what was asked, and whether they're
    stratified."""
    note = None
    if n < k:
        k, note = n, f"{n} folds: there are only {n} training rows"
    if k < 2:
        return [], note, False
    idx = np.arange(n)
    if y_strat is not None:
        counts = np.bincount(y_strat.astype(int))
        rarest = int(counts[counts > 0].min())
        if rarest >= k:
            return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(idx, y_strat)), note, True
        unstratified = (f"Folds aren't stratified: the rarest class of {strat_name} has {rarest} training "
                        f"row{'' if rarest == 1 else 's'}")
        note = f"{note}. {unstratified}" if note else unstratified
    return list(KFold(k, shuffle=True, random_state=seed).split(idx)), note, False


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def summarize(per_fold: list[dict[str, dict[str, Any]]]) -> dict[str, dict[str, dict[str, Any]]]:
    """Per target and metric: the mean and standard deviation over the folds where it is a finite number, and each
    fold's value (None where it wasn't defined, e.g. ROC AUC on a fold with one class)."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    targets = dict.fromkeys(t for f in per_fold for t in f)
    for t in targets:
        names = dict.fromkeys(m for f in per_fold for m in f.get(t, {}))
        for m in names:
            values = [f.get(t, {}).get(m) for f in per_fold]
            values = [float(v) if _finite(v) else None for v in values]
            got = [v for v in values if v is not None]
            if got:
                out.setdefault(t, {})[m] = {"mean": float(np.mean(got)), "std": float(np.std(got)), "folds": values}
    return out
