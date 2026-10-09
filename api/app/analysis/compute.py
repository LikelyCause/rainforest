"""The numeric work behind each analysis kind. Pure functions over a `Prepared` sample.

`check(fraction, message)` reports progress and raises when the job has been superseded.
"""
from __future__ import annotations

import math
import os
import warnings
from typing import Callable

import numpy as np
from joblib import parallel_config
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from .prep import (
    OTHER, SEED, AnalysisError, Column, NotEnough, Prepared, code_matrix, design_matrix, display_values,
)

Check = Callable[[float, str], None]
MIN_CLASS_ROWS = 5
MIN_ROWS = 10
DETAIL_LEVELS = 10
SCATTER_POINTS = 500
HOVER_FIELDS = 5
PCA_COMPONENTS = 20
LOADING_COMPONENTS = 5
TOP_LOADINGS = 20
N_JOBS = max(1, (os.cpu_count() or 2) - 1)


def _noop(frac: float, msg: str) -> None:
    pass


def _num(v: float | None, digits: int = 6) -> float | None:
    if v is None:
        return None
    v = float(v)
    return round(v, digits) if math.isfinite(v) else None


def _sig(v: float | None) -> float | None:
    """Keep p-values like 1e-300 instead of rounding them to zero."""
    if v is None or not math.isfinite(float(v)):
        return None
    return float(f"{float(v):.6g}")


def _need_rows(p: Prepared, minimum: int = MIN_ROWS) -> None:
    if p.n < minimum:
        raise NotEnough("too_few_rows", f"Not enough rows: {p.n} in the sample, at least {minimum} needed.")


def _need_fields(p: Prepared, minimum: int) -> None:
    if len(p.columns) < minimum:
        raise NotEnough("too_few_fields",
                        f"Not enough usable fields: {len(p.columns)}, at least {minimum} needed.")


def _target_classes_present(p: Prepared) -> None:
    if p.target_kind == "categorical" and np.unique(p.y).size < 2:
        raise NotEnough("single_class", f"Not enough classes: {p.target} has a single class in the sample.")


# ------------------------------------------------------------------------------------- map
def run_map(p: Prepared, method: str = "tsne", perplexity: int = 30, n_neighbors: int = 15,
            check: Check = _noop) -> dict:
    _need_fields(p, 2)
    _need_rows(p)
    if method == "tsne" and perplexity >= p.n:
        raise NotEnough("too_few_rows", f"Not enough rows: t-SNE with perplexity {perplexity} needs more than "
                                        f"{perplexity} rows; the sample has {p.n}.")
    if method == "umap" and n_neighbors >= p.n:
        raise NotEnough("too_few_rows", f"Not enough rows: UMAP with {n_neighbors} neighbours needs more than "
                                        f"{n_neighbors} rows; the sample has {p.n}.")
    check(0.1, "Encoding features")
    X, _, owners = design_matrix(p.columns)
    pca = PCA(n_components=min(50, *X.shape), random_state=SEED).fit(X)
    hover = _hover_columns(p.columns, pca, owners)
    if method == "umap":
        try:
            import umap
        except Exception as exc:  # noqa: BLE001 - numba/llvmlite can fail at import, not just be missing
            raise AnalysisError(f"UMAP is not available on this server ({type(exc).__name__}: {exc})") from exc
        check(0.2, f"Running UMAP on {p.n:,} rows")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # a fixed seed turns off UMAP's parallelism, and it says so every run
            emb = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=SEED).fit_transform(X)
    else:
        from sklearn.manifold import TSNE

        Z = pca.transform(X) if X.shape[1] > 50 else X
        check(0.2, f"Running t-SNE on {p.n:,} rows")
        emb = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
                   random_state=SEED, n_jobs=N_JOBS).fit_transform(Z)
    check(0.95, "Preparing points")
    if p.target_kind == "categorical":
        colour = [int(v) for v in p.y]
    elif p.target_kind == "numeric":
        colour = [_num(v) for v in p.y]
    else:
        colour = [None] * p.n
    values = [display_values(c) for c in hover]
    return {
        "method": method,
        "n": p.n,
        "target": p.target,
        "target_kind": p.target_kind,
        "classes": p.classes if p.target_kind == "categorical" else [],
        "points": [[round(float(x), 4), round(float(y), 4), c] for (x, y), c in zip(emb, colour)],
        "hover_fields": [c.name for c in hover],
        "hover_values": [list(row) for row in zip(*values)] if values else [[] for _ in range(p.n)],
        "skipped": p.skipped,
    }


def _hover_columns(columns: list[Column], pca: PCA, owners: list[int]) -> list[Column]:
    """The fields that spread the rows out most: largest weight in the first two principal components."""
    k = min(2, pca.n_components_)
    w = (pca.components_[:k] ** 2 * pca.explained_variance_ratio_[:k, None]).sum(axis=0)
    by_column = np.bincount(owners, weights=w, minlength=len(columns))
    return [columns[i] for i in np.argsort(-by_column, kind="stable")[:HOVER_FIELDS]]


# ----------------------------------------------------------------------------------- tests
def run_tests(p: Prepared, check: Check = _noop) -> dict:
    if p.target is None:
        raise NotEnough("no_target", "Not enough to run: feature tests need a target. Choose one first.")
    classes, excluded, y, rows = p.classes, [], p.y, np.arange(p.n)
    if p.target_kind == "categorical":
        counts = np.bincount(p.y, minlength=len(p.classes))
        ok = counts >= MIN_CLASS_ROWS
        excluded = [{"name": c, "n": int(n)} for c, n, good in zip(p.classes, counts, ok) if not good]
        if ok.sum() < 2:
            raise NotEnough("single_class", f"Not enough classes: {p.target} needs at least two classes with "
                                            f"{MIN_CLASS_ROWS}+ rows in the sample.")
        rows = np.flatnonzero(ok[p.y])
        remap = np.cumsum(ok) - 1
        y = remap[p.y[rows]]
        classes = [c for c, good in zip(p.classes, ok) if good]
    n = int(rows.size)
    if n < MIN_ROWS:
        raise NotEnough("too_few_rows", f"Not enough rows: {n} in the sample, at least {MIN_ROWS} needed.")
    _need_fields(p, 1)

    out, skipped = [], list(p.skipped)
    for k, col in enumerate(p.columns):
        if k % 10 == 0:
            check(0.1 + 0.85 * k / len(p.columns), f"Testing {col.name} ({k + 1}/{len(p.columns)})")
        x = col.values[rows]
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            if p.target_kind == "categorical":
                row = (_numeric_vs_classes(x, y, classes) if col.kind == "numeric"
                       else _levels_vs_classes(x, col.levels, y, classes))
            else:
                row = (_numeric_vs_numeric(x, y) if col.kind == "numeric"
                       else _levels_vs_numeric(x, col.levels, y))
        if isinstance(row, str):
            skipped.append({"name": col.name, "reason": row})
            continue
        out.append({"field_id": col.field_id, "name": col.name, "type": col.type, **row})

    finite = [i for i, r in enumerate(out) if r["p"] is not None]
    if finite:
        adj = stats.false_discovery_control([out[i]["p"] for i in finite], method="bh")
        for i, a in zip(finite, adj):
            out[i]["p_adj"] = _sig(a)
    out.sort(key=lambda r: (r["effect"] is None, -(r["effect"] or 0.0)))
    return {
        "target": p.target,
        "target_kind": p.target_kind,
        "classes": classes if p.target_kind == "categorical" else [],
        "excluded_classes": excluded,
        "n": n,
        "rows": out,
        "skipped": skipped,
    }


def _row(test: str, statistic, p, effect, effect_name: str, n: int, secondary: dict | None, detail: dict) -> dict:
    return {"test": test, "statistic": _num(statistic), "p": _sig(p), "p_adj": None, "effect": _num(effect),
            "effect_name": effect_name, "n": int(n), "secondary": secondary, "detail": detail}


def _eta_squared(values: np.ndarray, groups: list[np.ndarray]) -> float | None:
    mu = values.mean()
    total = float(((values - mu) ** 2).sum())
    if total <= 0:
        return None
    return float(sum(g.size * (g.mean() - mu) ** 2 for g in groups if g.size)) / total


def _anova(values: np.ndarray, groups: list[np.ndarray]) -> tuple | str:
    usable = [g for g in groups if g.size >= 2]
    if len(usable) < 2:
        return "fewer than two groups with data"
    eta2 = _eta_squared(values, usable)
    if eta2 is None:
        return "constant in sample"
    f, p = stats.f_oneway(*usable)
    if not math.isfinite(float(p)) and math.isinf(float(f)):
        p = 0.0  # zero within-group spread: the groups are perfectly separated
    try:
        h, hp = stats.kruskal(*usable)
        secondary = {"test": "kruskal", "statistic": _num(h), "p": _sig(hp)}
    except ValueError:
        secondary = None
    return f, p, eta2, sum(g.size for g in usable), secondary


def _box(label: str, g: np.ndarray) -> dict:
    if g.size == 0:
        return {"class": label, "n": 0, "min": None, "q1": None, "median": None, "q3": None, "max": None}
    q1, med, q3 = np.percentile(g, [25, 50, 75])
    iqr = q3 - q1
    inside = g[(g >= q1 - 1.5 * iqr) & (g <= q3 + 1.5 * iqr)]
    return {"class": label, "n": int(g.size), "min": _num(inside.min()), "q1": _num(q1), "median": _num(med),
            "q3": _num(q3), "max": _num(inside.max())}


def _numeric_vs_classes(x: np.ndarray, y: np.ndarray, classes: list[str]) -> dict | str:
    m = ~np.isnan(x)
    groups = [x[m & (y == c)] for c in range(len(classes))]
    res = _anova(x[m], groups)
    if isinstance(res, str):
        return res
    f, p, eta2, n, secondary = res
    detail = {"kind": "box", "groups": [_box(c, g) for c, g in zip(classes, groups)]}
    return _row("anova", f, p, eta2, "η²", n, secondary, detail)


def _top_levels(codes: np.ndarray, levels: list[str]) -> tuple[np.ndarray, list[str]]:
    """Fold everything past the 10 most frequent levels into "Other" for display."""
    if len(levels) <= DETAIL_LEVELS + 1 and OTHER not in levels:
        return codes, list(levels)
    freq = np.bincount(codes, minlength=len(levels))
    if OTHER in levels:
        freq[levels.index(OTHER)] = -1  # the sample's own "Other" bucket always folds into ours
    top = [i for i in np.argsort(-freq, kind="stable") if freq[i] >= 0][:DETAIL_LEVELS]
    remap = np.full(len(levels), len(top))
    remap[top] = np.arange(len(top))
    return remap[codes], [levels[i] for i in top] + [OTHER]


def _levels_vs_classes(x: np.ndarray, levels: list[str], y: np.ndarray, classes: list[str]) -> dict | str:
    table = np.zeros((len(levels), len(classes)), dtype=np.int64)
    np.add.at(table, (x, y), 1)
    t = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    if min(t.shape) < 2:
        return "a single level in sample"
    chi2, p, _, _ = stats.chi2_contingency(t)
    v = math.sqrt(chi2 / (t.sum() * (min(t.shape) - 1)))
    codes, shown = _top_levels(x, levels)
    counts = np.zeros((len(classes), len(shown)), dtype=np.int64)
    np.add.at(counts, (y, codes), 1)
    detail = {"kind": "levels", "levels": shown, "counts": counts.tolist()}
    return _row("chi2", chi2, p, min(v, 1.0), "Cramér's V", int(t.sum()), None, detail)


def _numeric_vs_numeric(x: np.ndarray, y: np.ndarray) -> dict | str:
    m = ~np.isnan(x) & ~np.isnan(y)
    a, b = x[m], y[m]
    if a.size < 3:
        return "fewer than three rows with both values"
    if a.min() == a.max() or b.min() == b.max():
        return "constant in sample"
    rho, p = stats.spearmanr(a, b)
    r, rp = stats.pearsonr(a, b)
    pick = np.random.default_rng(SEED).choice(a.size, size=min(SCATTER_POINTS, a.size), replace=False)
    detail = {"kind": "scatter", "points": [[_num(a[i]), _num(b[i])] for i in np.sort(pick)]}
    return _row("spearman", rho, p, abs(float(rho)), "|ρ|", int(a.size),
                {"test": "pearson", "statistic": _num(r), "p": _sig(rp)}, detail)


def _levels_vs_numeric(x: np.ndarray, levels: list[str], y: np.ndarray) -> dict | str:
    m = ~np.isnan(y)
    xs, ys = x[m], y[m]
    groups = [ys[xs == k] for k in range(len(levels))]
    res = _anova(ys, groups)
    if isinstance(res, str):
        return res
    f, p, eta2, n, secondary = res
    codes, shown = _top_levels(xs, levels)
    detail = {"kind": "box", "groups": [_box(lv, ys[codes == k]) for k, lv in enumerate(shown)]}
    return _row("anova", f, p, eta2, "η²", n, secondary, detail)


# ------------------------------------------------------------------------------ importance
def run_importance(p: Prepared, check: Check = _noop) -> dict:
    _need_fields(p, 1)
    _need_rows(p)
    mi: list[dict] = []
    if p.target is not None:
        _target_classes_present(p)
        check(0.1, "Mutual information with the target")
        X, discrete = code_matrix(p.columns)
        # Threads, not loky processes: spawning workers from inside the API costs more than it saves here.
        with parallel_config(backend="threading", n_jobs=N_JOBS):
            if p.target_kind == "categorical":
                scores = mutual_info_classif(X, p.y, discrete_features=discrete, random_state=SEED, n_jobs=N_JOBS)
            else:
                m = ~np.isnan(p.y)
                scores = mutual_info_regression(X[m], p.y[m], discrete_features=discrete, random_state=SEED,
                                                n_jobs=N_JOBS)
        mi = sorted(
            ({"field_id": c.field_id, "name": c.name, "value": _num(s)} for c, s in zip(p.columns, scores)),
            key=lambda r: -(r["value"] or 0.0),
        )
    check(0.7, "Principal components")
    X, names, _ = design_matrix(p.columns)
    if X.shape[1] < 2:
        raise NotEnough("too_few_fields", "Not enough usable features for PCA: at least 2 needed.")
    pca = PCA(n_components=min(PCA_COMPONENTS, *X.shape), random_state=SEED).fit(X)
    # Rounded down so the shares never add up to more than 1.
    explained = [math.floor(float(v) * 1e6) / 1e6 for v in pca.explained_variance_ratio_]
    components = []
    for i, comp in enumerate(pca.components_[:LOADING_COMPONENTS]):
        top = np.argsort(-np.abs(comp), kind="stable")[:TOP_LOADINGS]
        components.append({"index": i + 1, "loadings": [{"feature": names[j], "value": _num(comp[j])} for j in top]})
    return {
        "target": p.target,
        "target_kind": p.target_kind,
        "n": p.n,
        "mi": mi,
        "pca": {
            "n_features": int(X.shape[1]),
            "explained": explained,
            "cumulative": [min(1.0, _num(v)) for v in np.cumsum(explained)],
            "components": components,
        },
        "skipped": p.skipped,
    }
