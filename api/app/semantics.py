"""Semantic field types, model families, the preprocessing valid for each, and default suggestions.

Single source of truth for "what can I do with a field of type X when training a model of family F".
The API serves `option_catalog()` to the UI so dropdowns never drift from validation.

Rainforest is tree-first: `tree` is the default family. Tree ensembles split on value *order*, so
scaling and monotonic transforms (log, clipping) cannot change a single split — they are not offered
for the tree family at all, and missing values are left for the model to route natively.
"""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

SEMANTIC_TYPES = [
    "numeric_continuous",
    "numeric_discrete",
    "categorical",
    "binary",
    "text",
    "datetime",
    "identifier",
    "constant",
]
NUMERIC = {"numeric_continuous", "numeric_discrete"}

TYPE_LABELS = {
    "numeric_continuous": "Numeric (continuous)",
    "numeric_discrete": "Numeric (discrete)",
    "categorical": "Categorical",
    "binary": "Binary",
    "text": "Free text",
    "datetime": "Date / time",
    "identifier": "Identifier",
    "constant": "Constant",
}

# ------------------------------------------------------------------------------ families
DEFAULT_FAMILY = "tree"
FAMILIES: dict[str, dict[str, Any]] = {
    "tree": {
        "label": "Tree ensembles",
        "examples": "LightGBM, XGBoost, histogram GBM, random forest",
        "handles_nan": True,
        "scale_invariant": True,
        "needs_onehot": False,
    },
    "linear": {
        "label": "Linear models",
        "examples": "logistic or ridge regression",
        "handles_nan": False,
        "scale_invariant": False,
        "needs_onehot": True,
    },
    "neural": {
        "label": "Neural / distance-based",
        "examples": "MLP",
        "handles_nan": False,
        "scale_invariant": False,
        "needs_onehot": True,
    },
}

# Why a dimension is locked for a family (shown in the UI instead of a dropdown).
FAMILY_NOTES: dict[str, dict[str, str]] = {
    "tree": {
        "scaling": "Not used for tree models — they split on value order, so rescaling changes nothing.",
        "transform": "Not used for tree models — log or clipping preserves order, so splits are identical.",
    },
    "linear": {
        "imputation": "Linear models can't take missing values; choose how to fill them.",
    },
    "neural": {
        "imputation": "Neural networks can't take missing values; choose how to fill them.",
    },
}

# option name -> human label, per decision dimension
IMPUTATIONS = {
    "none": "None (leave missing — the model routes NaN natively)",
    "mean": "Mean",
    "median": "Median",
    "most_frequent": "Most frequent",
    "constant": "Constant value",
}
TRANSFORMS = {"none": "None", "log1p": "Signed log1p", "clip": "Winsorize (clip to p1 / p99)"}
SCALINGS = {
    "none": "None",
    "standard": "Standard (z-score)",
    "robust": "Robust (median / IQR)",
    "minmax": "Min-max [0, 1]",
    "quantile_normal": "Quantile → normal",
}
ENCODINGS = {
    "none": "None",
    "onehot": "One-hot (rare levels grouped)",
    "ordinal": "Ordinal codes",
    "frequency": "Frequency encoding",
    "binary": "Binary 0/1",
    "tfidf_svd": "TF-IDF + SVD components",
    "text_stats": "Length / word-count features",
    "datetime_parts": "Date parts + epoch days",
}
DIMS = ("imputation", "transform", "scaling", "encoding")
SETTING_KEYS = ("imputation", "impute_value", "add_missing_indicator", "transform", "scaling", "encoding", "params")

# Everything a type could ever use, before family rules are applied.
_BASE_ALLOWED: dict[str, dict[str, list[str]]] = {
    "numeric_continuous": {
        "imputation": ["none", "mean", "median", "constant"],
        "transform": ["none", "log1p", "clip"],
        "scaling": ["none", "standard", "robust", "minmax", "quantile_normal"],
        "encoding": ["none"],
    },
    "numeric_discrete": {
        "imputation": ["none", "mean", "median", "most_frequent", "constant"],
        "transform": ["none", "log1p", "clip"],
        "scaling": ["none", "standard", "robust", "minmax", "quantile_normal"],
        "encoding": ["none"],
    },
    "categorical": {
        "imputation": ["constant", "most_frequent"],
        "transform": ["none"],
        "scaling": ["none"],
        "encoding": ["ordinal", "onehot", "frequency"],
    },
    "binary": {"imputation": ["constant", "most_frequent"], "transform": ["none"], "scaling": ["none"], "encoding": ["binary"]},
    "text": {"imputation": ["constant"], "transform": ["none"], "scaling": ["none"], "encoding": ["tfidf_svd", "text_stats"]},
    "datetime": {
        "imputation": ["none", "median"],
        "transform": ["none"],
        "scaling": ["none", "standard", "minmax"],
        "encoding": ["datetime_parts"],
    },
    "identifier": {"imputation": ["none"], "transform": ["none"], "scaling": ["none"], "encoding": ["none"]},
    "constant": {"imputation": ["none"], "transform": ["none"], "scaling": ["none"], "encoding": ["none"]},
}

NON_FEATURE_TYPES = {"identifier", "constant"}

# A target is predicted as classes (binary, categorical, or a numeric field with few values) or, when numeric, as a
# number or as bands of numbers.
TARGET_TYPES = ("binary", "categorical", "numeric_discrete", "numeric_continuous")
CLASS_TYPES = ("binary", "categorical", "numeric_discrete")
MAX_TARGET_LEVELS = 100
TARGET_TASKS = ("classes", "regression", "bins")
DEFAULT_BIN_COUNT = 5
MAX_BIN_COUNT = 20


def default_task(t: str, n_unique: int | None) -> str | None:
    """How a target of type `t` is predicted unless someone chooses otherwise; None when it can't be a target.
    Mirrors meta.default_target_task (migration 014)."""
    if t in ("binary", "categorical"):
        return "classes"
    if t == "numeric_continuous":
        return "regression"
    if t == "numeric_discrete":
        return "classes" if (n_unique or 0) <= MAX_TARGET_LEVELS else "regression"
    return None


def target_problem(t: str, n_unique: int | None, task: str | None = None) -> str | None:
    """Why a field typed `t` with `n_unique` distinct values can't be a target predicted as `task` (its default when
    None), or None if it can."""
    if t not in TARGET_TYPES:
        return (f"a {TYPE_LABELS.get(t, t).lower()} field can't be a target; targets must be binary, categorical or "
                f"numeric (change the field's type first if its values are class labels)")
    task = task or default_task(t, n_unique)
    if task not in TARGET_TASKS:
        return f"unknown target task {task!r} (expected one of {', '.join(TARGET_TASKS)})"
    if task == "classes":
        if t not in CLASS_TYPES:
            return "a continuous field is predicted as a number or as bands, not as classes"
        if n_unique is not None and n_unique > MAX_TARGET_LEVELS:
            return (f"{n_unique:,} distinct values; predicting classes allows at most {MAX_TARGET_LEVELS}"
                    + ("" if t in ("binary", "categorical") else " (predict it as a number or as bands instead)"))
        return None
    if task == "bins" and t in ("binary", "categorical"):
        return None  # one level (or several) against the rest: bins = {"positive": [...]}
    if t not in NUMERIC:
        return f"only a numeric field can be predicted as {'a number' if task == 'regression' else 'bands'}"
    return None


def grouped_target(spec: dict | None) -> list[str] | None:
    """The positive classes of a class target predicted one-vs-rest, or None."""
    bins = (spec or {}).get("bins") or {}
    return list(bins["positive"]) if (spec or {}).get("task") == "bins" and "positive" in bins else None


# Sorts before any class name, so the chosen classes stay the positive class (encoded 1) of a binary target.
GROUP_REST = "(rest)"


def bins_problem(bins: dict | None) -> str | None:
    """What's wrong with a bands setting: exactly one of `k` (2-20 equal-count bands) or `edges` (1-19 ascending cut
    points)."""
    if isinstance(bins, dict) and set(bins) == {"positive"}:
        pos = bins["positive"]
        if not isinstance(pos, list) or not pos or len(pos) > MAX_TARGET_LEVELS or not all(isinstance(v, str) and v for v in pos):
            return "choose at least one class to predict against the rest"
        return None
    if not isinstance(bins, dict) or len(bins) != 1 or not ({"k", "edges"} & set(bins)):
        return "bands need either a count (k) or cut points (edges)"
    if "k" in bins:
        k = bins["k"]
        if isinstance(k, bool) or not isinstance(k, int) or not 2 <= k <= MAX_BIN_COUNT:
            return f"the band count must be a whole number from 2 to {MAX_BIN_COUNT}"
        return None
    edges = bins["edges"]
    if (not isinstance(edges, list) or not 1 <= len(edges) <= MAX_BIN_COUNT - 1
            or not all(isinstance(e, (int, float)) and not isinstance(e, bool) and math.isfinite(e) for e in edges)):
        return f"cut points must be 1 to {MAX_BIN_COUNT - 1} finite numbers"
    if any(b <= a for a, b in zip(edges, edges[1:])):
        return "cut points must be in ascending order, each different"
    return None


DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "onehot": {"max_categories": 30, "min_frequency": 0.01},
    "ordinal": {"min_frequency": 0.001},
    "tfidf_svd": {"svd_components": 32, "max_features": 20000, "min_df": 3},
}
# encoding -> param -> (type, min, max); min_frequency is a share of rows.
PARAM_RULES: dict[str, dict[str, tuple[type, float, float]]] = {
    "onehot": {"max_categories": (int, 2, 1000), "min_frequency": (float, 0, 0.99)},
    "ordinal": {"min_frequency": (float, 0, 0.99)},
    "tfidf_svd": {"svd_components": (int, 1, 256), "max_features": (int, 10, 1_000_000), "min_df": (int, 1, 10_000)},
}


def check_family(family: str) -> str:
    if family not in FAMILIES:
        raise ValueError(f"unknown model family {family!r} (expected one of {', '.join(FAMILIES)})")
    return family


def allowed(t: str, family: str = DEFAULT_FAMILY) -> dict[str, list[str]]:
    """Options valid for a type under a model family."""
    fam = FAMILIES[check_family(family)]
    a = deepcopy(_BASE_ALLOWED[t])
    if fam["scale_invariant"]:
        a["transform"] = ["none"]
        a["scaling"] = ["none"]
    if not fam["handles_nan"] and t in NUMERIC | {"datetime"}:
        a["imputation"] = [o for o in a["imputation"] if o != "none"]
    if fam["needs_onehot"] and t == "categorical":
        # Integer codes impose a fake order that linear / neural models take literally.
        a["encoding"] = ["onehot", "frequency"]
    return a


def option_catalog() -> dict[str, Any]:
    return {
        "types": [{"value": t, "label": TYPE_LABELS[t]} for t in SEMANTIC_TYPES],
        "families": [{"value": k, **v} for k, v in FAMILIES.items()],
        "default_family": DEFAULT_FAMILY,
        "family_notes": FAMILY_NOTES,
        "labels": {"imputation": IMPUTATIONS, "transform": TRANSFORMS, "scaling": SCALINGS, "encoding": ENCODINGS},
        "allowed": {f: {t: allowed(t, f) for t in SEMANTIC_TYPES} for f in FAMILIES},
        "default_params": DEFAULT_PARAMS,
        "param_rules": {enc: {k: {"integer": kind is int, "min": lo, "max": hi} for k, (kind, lo, hi) in rules.items()}
                        for enc, rules in PARAM_RULES.items()},
        "non_feature_types": sorted(NON_FEATURE_TYPES),
        "target_rule": {"types": list(TARGET_TYPES), "numeric_types": sorted(NUMERIC), "max_levels": MAX_TARGET_LEVELS,
                        "tasks": list(TARGET_TASKS), "default_k": DEFAULT_BIN_COUNT},
    }


def validate_decision(d: dict[str, Any], family: str = DEFAULT_FAMILY, n_unique: int | None = None) -> list[str]:
    """Return a list of problems; empty means the decision is coherent for this family.

    `n_unique` (the field's distinct values) is needed to check the level limit on targets.
    """
    errors: list[str] = []
    t = d.get("semantic_type")
    if t not in _BASE_ALLOWED:
        return [f"unknown semantic_type {t!r}"]
    if family not in FAMILIES:
        return [f"unknown model family {family!r}"]
    if d.get("action") == "keep" and t in NON_FEATURE_TYPES:
        errors.append(f"a field typed '{t}' cannot be kept as a feature; drop it or change its type")
    if d.get("action") == "target":
        if problem := target_problem(t, n_unique, d.get("target_task")):
            errors.append(problem)
        elif (d.get("target_task") or default_task(t, n_unique)) == "bins" and d.get("target_bins") is not None:
            grouped = isinstance(d.get("target_bins"), dict) and "positive" in d["target_bins"]
            if problem := bins_problem(d.get("target_bins")):
                errors.append(problem)
            elif grouped != (t in ("binary", "categorical")):
                errors.append("a class field is predicted against the rest by choosing its positive classes; a number "
                              "is cut into bands" if grouped else "choose the class(es) to predict against the rest")
    if d.get("action") in ("drop", "target"):
        return errors
    a = allowed(t, family)
    for dim in DIMS:
        v = d.get(dim)
        if v not in a[dim]:
            errors.append(f"{dim}={v!r} is not valid for {t} with {family} models (allowed: {', '.join(a[dim])})")
    if d.get("imputation") == "constant" and t in NUMERIC:
        try:
            fill = float(d.get("impute_value") or "")
        except ValueError:
            errors.append("numeric constant imputation needs a numeric impute_value")
        else:
            if not math.isfinite(fill):
                errors.append(f"impute_value must be a finite number (got {d.get('impute_value')!r})")
    errors += param_errors(d.get("encoding"), d.get("params") or {})
    return errors


def param_errors(encoding: str | None, params: dict[str, Any]) -> list[str]:
    """Type and range problems in the encoder parameters the export reads for `encoding`.

    Keys the encoding doesn't use are ignored: they can be left over from a previous encoding choice.
    """
    errors = []
    for key, (kind, lo, hi) in PARAM_RULES.get(encoding or "", {}).items():
        if key not in params or params[key] is None:
            continue
        v = params[key]
        ok = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        if ok and kind is int:
            ok = float(v).is_integer()
        if not ok or not lo <= v <= hi:
            what = "an integer" if kind is int else "a number"
            errors.append(f"params.{key} must be {what} from {lo} to {hi} for {encoding} (got {v!r})")
    return errors


def defaults_for_type(t: str, family: str = DEFAULT_FAMILY) -> dict[str, Any]:
    """Neutral, valid settings for a type — used when the user changes a field's type."""
    a = allowed(t, family)
    pick = lambda dim, pref: pref if pref in a[dim] else a[dim][0]  # noqa: E731
    encoding = a["encoding"][0]
    return {
        "imputation": pick("imputation", "none" if FAMILIES[family]["handles_nan"] else "median"),
        "impute_value": "__missing__" if t in ("categorical", "binary") else ("" if t == "text" else None),
        "add_missing_indicator": False,
        "transform": "none",
        "scaling": "none",
        "encoding": encoding,
        "params": dict(DEFAULT_PARAMS.get(encoding, {})),
    }


def suggest(t: str, stats: dict[str, Any], is_target: bool, family: str = DEFAULT_FAMILY) -> dict[str, Any]:
    """Heuristic starting point for a field under a model family, with human-readable reasons.

    `action` / `semantic_type` never depend on the family; the preprocessing settings do.
    """
    fam = FAMILIES[check_family(family)]
    out: dict[str, Any] = {"action": "keep", "semantic_type": t, **defaults_for_type(t, family)}
    reasons: list[str] = []
    if is_target:
        out["action"] = "target"
        out["reasons"] = ["Target: marked as a target column, so it is predicted rather than used as a feature"]
        return out

    null_pct = stats.get("null_pct") or 0.0
    skew = stats.get("skewness") or 0.0
    outliers = stats.get("outlier_pct") or 0.0
    n_levels = stats.get("n_unique") or 0

    if t in NON_FEATURE_TYPES:
        out["action"] = "drop"
        reasons.append("Drop: an identifier (a key or ID) carries no signal that carries over to new rows" if t == "identifier"
                       else "Drop: every row has the same value, so it can't help predict anything")
    elif null_pct >= 0.95:
        out["action"] = "drop"
        reasons.append(f"Drop: {null_pct:.0%} of values are missing")

    if t in NUMERIC:
        if fam["handles_nan"]:
            if null_pct > 0:
                reasons.append(f"Imputation: none; {null_pct:.1%} of values are missing and tree models learn where missing "
                               "values belong")
            if abs(skew) > 2 or outliers > 0.03:
                reasons.append("Transform: none; skew and outliers don't affect tree models, which only use the order of values")
        else:
            if null_pct > 0:
                out["imputation"] = "median"
                if null_pct >= 0.01:
                    out["add_missing_indicator"] = True
                    reasons.append(f"Imputation: median, plus a column flagging missing rows; {null_pct:.1%} of values are "
                                   "missing")
                else:
                    reasons.append(f"Imputation: median; only {null_pct:.2%} of values are missing")
            if abs(skew) > 1:
                out["transform"] = "log1p"
                reasons.append(f"Transform: signed log1p, to even out a skewed distribution (skewness {skew:.1f})")
            elif outliers > 0.03:
                out["transform"] = "clip"
                reasons.append(f"Transform: winsorize (clip extremes); {outliers:.1%} of values are outliers by the IQR rule")
            out["scaling"] = "quantile_normal" if family == "neural" and abs(skew) > 1 else "standard"
            reasons.append(f"Scaling: {SCALINGS[out['scaling']].lower()}; {fam['label'].lower()} fit better when features "
                           "share a scale")
        if t == "numeric_discrete" and n_levels <= 12:
            reasons.append(f"Type: only {n_levels} distinct values; consider treating it as categorical")
    elif t == "categorical":
        if fam["needs_onehot"]:
            out["encoding"] = "onehot" if n_levels <= 200 else "frequency"
            if out["encoding"] == "onehot":
                reasons.append(f"Encoding: one-hot; {n_levels} levels, the 30 most common get their own column and "
                                   "the rest share one")
            else:
                reasons.append(f"Encoding: frequency; {n_levels:,} levels are too many for one-hot")
        elif n_levels <= 8:
            out["encoding"] = "onehot"
            reasons.append(f"Encoding: one-hot; only {n_levels} levels")
        elif n_levels <= 1000:
            out["encoding"] = "ordinal"
            reasons.append(f"Encoding: ordinal codes; {n_levels} levels, which trees split well without adding columns")
        else:
            out["encoding"] = "frequency"
            reasons.append(f"Encoding: frequency; {n_levels:,} levels are too many for one-hot")
        out["params"] = dict(DEFAULT_PARAMS.get(out["encoding"], {}))
        if null_pct > 0:
            reasons.append("Imputation: missing values become their own level, '__missing__'")
    elif t == "binary":
        out["imputation"] = "most_frequent" if null_pct < 0.05 else "constant"
    elif t == "text":
        out["encoding"] = "tfidf_svd"
        out["params"] = dict(DEFAULT_PARAMS["tfidf_svd"])
        reasons.append("Encoding: TF-IDF reduced to 32 SVD components, which turns free text into numbers")
    elif t == "datetime":
        if not fam["handles_nan"]:
            out["imputation"] = "median"
            out["scaling"] = "standard"
        reasons.append("Encoding: split into year, month, weekday, hour and days since 1970")

    out["reasons"] = reasons
    return out


def suggest_all(t: str, stats: dict[str, Any], is_target: bool) -> dict[str, dict[str, Any]]:
    return {f: suggest(t, stats, is_target, f) for f in FAMILIES}


def settings_of(d: dict[str, Any]) -> dict[str, Any]:
    return {k: d.get(k) for k in SETTING_KEYS}
