"""Pairwise association measures across mixed column types.

* numeric x numeric        -> Spearman rho (pairwise-complete; masked matmuls, exact re-ranking where needed)
* categorical x categorical -> bias-corrected Cramér's V (nulls form their own level)
* numeric x categorical    -> correlation ratio eta

All inputs come from the same seeded row sample built during profiling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
from scipy.stats import rankdata

MIN_OBS = 30
# First-pass Spearman values at or above this are recomputed exactly when the pair's missingness differs.
REFINE_AT = 0.3


@dataclass
class Assoc:
    i: int  # index into the caller's field list
    j: int
    method: str
    value: float
    n_obs: int


def spearman_matrix(X: np.ndarray, refine_at: float = REFINE_AT) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise-complete Spearman correlation for an (n, p) matrix with NaNs.

    A first pass ranks each column over its own non-missing values and takes Pearson per pair over the rows
    where both are present (sufficient statistics, one matmul). That is exact only when both columns are
    missing on the same rows; otherwise a column ranked over more rows than the pair's has gaps in its
    ranks, which biases rho (towards 0 for a monotone relation). Such pairs whose first-pass |rho| reaches
    `refine_at` are recomputed on ranks taken within the pair's own rows. Returns (rho, n_obs).
    """
    rho, n = _spearman_fast(X)
    nn = np.diag(n)
    biased = (n < nn[:, None]) | (n < nn[None, :])
    with np.errstate(invalid="ignore"):
        todo = np.argwhere(np.triu(biased & (np.abs(rho) >= refine_at) & (n >= MIN_OBS), k=1))
    if len(todo):
        rankers: dict[int, tuple[np.ndarray, np.ndarray, bool]] = {}
        present = ~np.isnan(X)
        for a, b in todo:
            for k in (a, b):
                if k not in rankers:
                    rankers[k] = _ranker(X[:, k])
            mask = present[:, a] & present[:, b]
            r = _pearson(_ranks_within(*rankers[a], mask), _ranks_within(*rankers[b], mask))
            rho[a, b] = rho[b, a] = r
    return rho, n


def _ranker(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Sort order and tie-group starts, computed once per column and reused for every pair."""
    order = np.argsort(x, kind="stable")
    xs = x[order]
    starts = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1]])
    return order, starts, len(starts) < len(x) - int(np.isnan(x).sum())


def _ranks_within(order: np.ndarray, starts: np.ndarray, ties: bool, mask: np.ndarray) -> np.ndarray:
    """Average ranks of the rows in `mask`, ranked among those rows only (as rankdata on x[mask])."""
    m = mask[order]
    if ties:
        cnt = np.add.reduceat(m.astype(np.int64), starts)
        avg = np.cumsum(cnt) - cnt + (cnt + 1) / 2
        sorted_ranks = np.repeat(avg, np.diff(np.r_[starts, len(m)]))
    else:
        sorted_ranks = np.cumsum(m).astype(np.float64)
    out = np.empty(len(m))
    out[order] = sorted_ranks
    return out[mask]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float(np.clip((a * b).sum() / den, -1, 1)) if den > 0 else float("nan")


def _spearman_fast(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R = np.column_stack([rankdata(X[:, k], nan_policy="omit") for k in range(X.shape[1])])
    M = (~np.isnan(R)).astype(np.float64)
    Z = np.where(M > 0, R, 0.0)
    n = M.T @ M
    sx = Z.T @ M  # sum of x_i over rows where x_j is present
    sxx = (Z * Z).T @ M
    sxy = Z.T @ Z
    num = n * sxy - sx * sx.T
    var_i = n * sxx - sx**2
    den = np.sqrt(np.clip(var_i, 0, None) * np.clip(var_i.T, 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where((den > 0) & (n >= MIN_OBS), num / den, np.nan)
    return np.clip(rho, -1, 1), n.astype(np.int64)


def cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    ka, kb = int(a.max()) + 1, int(b.max()) + 1
    table = np.bincount(a.astype(np.int64) * kb + b, minlength=ka * kb).reshape(ka, kb)
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    r, k = table.shape
    n = table.sum()
    if r < 2 or k < 2 or n < MIN_OBS:
        return float("nan")
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    chi2 = ((table - expected) ** 2 / expected).sum()
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else float("nan")


def cramers_v_sparse(a: np.ndarray, b: np.ndarray) -> float:
    """cramers_v over only the non-empty cells, so a field with ~n levels costs O(n), not O(levels x classes)."""
    n = a.size
    if n < MIN_OBS:
        return float("nan")
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    r, k = int(ai.max()) + 1, int(bi.max()) + 1
    if r < 2 or k < 2:
        return float("nan")
    cells, counts = np.unique(ai.astype(np.int64) * k + bi, return_counts=True)
    rows, cols = np.bincount(ai, minlength=r), np.bincount(bi, minlength=k)
    # Σ (O - E)² / E = Σ O² / E - n, and only non-empty cells have O > 0.
    chi2 = float((counts.astype(np.float64) ** 2 / (rows[cells // k] * cols[cells % k] / n)).sum() - n)
    phi2c = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else float("nan")


def screen_associations(targets: list[tuple[int, np.ndarray]],
                        fields: list[tuple[int, np.ndarray]]) -> Iterator[Assoc]:
    """Uncapped Cramér's V between targets and fields whose capped codes would hide a leak (more levels than
    CORR_MAX_LEVELS, or free text compared by exact value). Pairs as (field, target)."""
    for it, t in targets:
        for ifield, codes in fields:
            v = cramers_v_sparse(codes, t)
            if np.isfinite(v):
                yield Assoc(ifield, it, "cramers_v", v, int(codes.size))


def correlation_ratio(x: np.ndarray, codes: np.ndarray) -> tuple[float, int]:
    mask = ~np.isnan(x)
    x, c = x[mask], codes[mask]
    n = x.size
    if n < MIN_OBS:
        return float("nan"), n
    cnt = np.bincount(c)
    sums = np.bincount(c, weights=x)
    present = cnt > 0
    if present.sum() < 2:
        return float("nan"), n
    means = sums[present] / cnt[present]
    mu = x.mean()
    ss_between = float((cnt[present] * (means - mu) ** 2).sum())
    ss_total = float(((x - mu) ** 2).sum())
    if ss_total <= 0:
        return float("nan"), n
    return float(np.sqrt(ss_between / ss_total)), n


def compute_associations(
    numeric: list[tuple[int, np.ndarray]],
    categorical: list[tuple[int, np.ndarray]],
    progress: Callable[[float], None] | None = None,
) -> Iterator[Assoc]:
    """Yield an Assoc for every pair with a defined value.

    `numeric` / `categorical` are (field_index, sample_vector) lists sharing the same rows.
    """
    total = (
        len(numeric) * (len(numeric) - 1) // 2
        + len(categorical) * (len(categorical) - 1) // 2
        + len(numeric) * len(categorical)
    ) or 1
    done = 0

    def tick(k: int) -> None:
        nonlocal done
        done += k
        if progress and done % 500 < k:
            progress(done / total)

    if len(numeric) >= 2:
        X = np.column_stack([v for _, v in numeric])
        rho, nobs = spearman_matrix(X)
        for a in range(len(numeric)):
            for b in range(a + 1, len(numeric)):
                if np.isfinite(rho[a, b]):
                    yield Assoc(numeric[a][0], numeric[b][0], "spearman", float(rho[a, b]), int(nobs[a, b]))
            tick(len(numeric) - a - 1)

    for a in range(len(categorical)):
        ia, ca = categorical[a]
        for b in range(a + 1, len(categorical)):
            ib, cb = categorical[b]
            v = cramers_v(ca, cb)
            if np.isfinite(v):
                yield Assoc(ia, ib, "cramers_v", v, int(ca.size))
        tick(len(categorical) - a - 1)

    for ia, x in numeric:
        for ib, c in categorical:
            v, nobs_ = correlation_ratio(x, c)
            if np.isfinite(v):
                yield Assoc(ia, ib, "correlation_ratio", v, nobs_)
        tick(len(categorical))
    if progress:
        progress(1.0)


def target_associations(
    targets: list[tuple[int, np.ndarray]],
    numeric: list[tuple[int, np.ndarray]],
    categorical: list[tuple[int, np.ndarray]],
) -> Iterator[Assoc]:
    """Pairs between newly coded targets and every other field (and each other), on the same row sample.

    `targets` hold class codes; `numeric` / `categorical` must not include them.
    """
    for k, (it, codes) in enumerate(targets):
        for ib, x in numeric:
            v, nobs = correlation_ratio(x, codes)
            if np.isfinite(v):
                yield Assoc(ib, it, "correlation_ratio", v, nobs)
        for ib, c in [*categorical, *targets[k + 1:]]:
            v = cramers_v(codes, c)
            if np.isfinite(v):
                yield Assoc(it, ib, "cramers_v", v, int(codes.size))
