"""Cutting a numeric target into bands: equal-count edges from training values, or the cut points someone chose."""

from __future__ import annotations

import numpy as np


def quantile_edges(values: np.ndarray, k: int) -> list[float]:
    """Equal-count cut points, rounded to 4 significant digits. Repeated quantiles collapse; a cut at the minimum would
    leave the lowest band empty, so it moves up to the next value (90 zeros and 1..10 cut into `< 1` and `≥ 1`)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return []
    lo = v.min()
    above = v[v > lo]
    edges = {
        _readable(float(e), lo, v) if e > lo else (float(above.min()) if above.size else None)
        for e in np.quantile(v, np.arange(1, k) / k)
    }
    return sorted(e for e in edges if e is not None)


def _readable(e: float, lo: float, v: np.ndarray) -> float:
    """A cut point rounded to as few significant digits (4 or more) as move almost no rows across it (at most 0.5%):
    rounding across a cluster of equal values would move them all, and could empty a band."""
    below = v < e
    for digits in range(4, 13):
        r = float(f"{e:.{digits}g}")
        if r > lo and np.count_nonzero((v < r) != below) <= 0.005 * v.size:
            return r
    return e


def edges_for(bins: dict, train_values: np.ndarray) -> list[float]:
    """The cut points an export uses: the chosen ones, or equal-count ones from the training values."""
    edges = [float(e) for e in bins["edges"]] if "edges" in bins else quantile_edges(train_values, int(bins["k"]))
    if not edges:
        raise ValueError("these bands leave one band only: the training values are all the same")
    v = np.asarray(train_values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size and np.unique(np.searchsorted(edges, v, side="right")).size < 2:
        raise ValueError("these bands leave one band only in the training rows; choose other cut points or fewer bands")
    return edges


def _fmt(v: float) -> str:
    return f"{v:.12g}"


def band_labels(edges: list[float]) -> list[str]:
    return (
        [f"< {_fmt(edges[0])}"]
        + [f"[{_fmt(a)}, {_fmt(b)})" for a, b in zip(edges, edges[1:])]
        + [f"≥ {_fmt(edges[-1])}"]
    )


def assign(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """Band index per value (float, so a missing value stays NaN)."""
    v = np.asarray(values, dtype=float)
    out = np.searchsorted(np.asarray(edges, dtype=float), v, side="right").astype(float)
    out[np.isnan(v)] = np.nan
    return out


def thin_bands(values: np.ndarray, edges: list[float]) -> list[str]:
    """Labels of the bands with fewer than 2 of `values`: every band needs rows to train on and to score."""
    idx = assign(values, edges)
    counts = np.bincount(idx[~np.isnan(idx)].astype(int), minlength=len(edges) + 1)
    return [label for label, n in zip(band_labels(edges), counts) if n < 2]
