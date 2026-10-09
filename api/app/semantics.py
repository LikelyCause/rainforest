"""Semantic field types, model families, the preprocessing valid for each, and default suggestions.

Single source of truth for "what can I do with a field of type X when training a model of family F".
The API serves `option_catalog()` to the UI so dropdowns never drift from validation.

Rainforest is tree-first: `tree` is the default family. Tree ensembles split on value *order*, so
scaling and monotonic transforms (log, clipping) cannot change a single split — they are not offered
for the tree family at all, and missing values are left for the model to route natively.
"""
from __future__ import annotations

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
        "examples": "logistic regression",
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

# Targets are class labels for a ClassifierChain; a continuous or free-form field would become thousands of classes.
TARGET_TYPES = ("binary", "categorical", "numeric_discrete")
MAX_TARGET_LEVELS = 100


def target_problem(t: str, n_unique: int | None) -> str | None:
    """Why a field typed `t` with `n_unique` distinct values can't be a target, or None if it can."""
    if t not in TARGET_TYPES:
        return (f"a {TYPE_LABELS.get(t, t).lower()} field can't be a target; targets must be binary, categorical "
                f"or discrete numeric (change the field's type first if its values are class labels)")
    if n_unique is not None and n_unique > MAX_TARGET_LEVELS:
        return f"{n_unique:,} distinct values; a target can have at most {MAX_TARGET_LEVELS} classes"
    return None

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "onehot": {"max_categories": 30, "min_frequency": 0.01},
    "ordinal": {"min_frequency": 0.001},
    "tfidf_svd": {"svd_components": 32, "max_features": 20000, "min_df": 3},
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
        "non_feature_types": sorted(NON_FEATURE_TYPES),
        "target_rule": {"types": list(TARGET_TYPES), "max_levels": MAX_TARGET_LEVELS},
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
    if d.get("action") == "target" and (problem := target_problem(t, n_unique)):
        errors.append(problem)
    if d.get("action") in ("drop", "target"):
        return errors
    a = allowed(t, family)
    for dim in DIMS:
        v = d.get(dim)
        if v not in a[dim]:
            errors.append(f"{dim}={v!r} is not valid for {t} with {family} models (allowed: {', '.join(a[dim])})")
    if d.get("imputation") == "constant" and t in NUMERIC:
        try:
            float(d.get("impute_value") or "")
        except ValueError:
            errors.append("numeric constant imputation needs a numeric impute_value")
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
        out["reasons"] = ["configured as a target column"]
        return out

    null_pct = stats.get("null_pct") or 0.0
    skew = stats.get("skewness") or 0.0
    outliers = stats.get("outlier_pct") or 0.0
    n_levels = stats.get("n_unique") or 0

    if t in NON_FEATURE_TYPES:
        out["action"] = "drop"
        reasons.append("identifiers carry no generalisable signal" if t == "identifier" else "no variance")
    elif null_pct >= 0.95:
        out["action"] = "drop"
        reasons.append(f"{null_pct:.0%} missing")

    if t in NUMERIC:
        if fam["handles_nan"]:
            if null_pct > 0:
                reasons.append(f"{null_pct:.1%} missing → left as NaN; the trees learn which side missing values go")
            if abs(skew) > 2 or outliers > 0.03:
                reasons.append("skew / outliers need no treatment — tree splits only depend on order")
        else:
            if null_pct > 0:
                out["imputation"] = "median"
                if null_pct >= 0.01:
                    out["add_missing_indicator"] = True
                    reasons.append(f"{null_pct:.1%} missing → median + missing indicator")
                else:
                    reasons.append("few missing values → median")
            if abs(skew) > 1:
                out["transform"] = "log1p"
                reasons.append(f"skewness {skew:.1f} → signed log1p")
            elif outliers > 0.03:
                out["transform"] = "clip"
                reasons.append(f"{outliers:.1%} IQR outliers → winsorize")
            out["scaling"] = "quantile_normal" if family == "neural" and abs(skew) > 1 else "standard"
            reasons.append(f"{SCALINGS[out['scaling']].lower()} scaling for gradient-based fitting")
        if t == "numeric_discrete" and n_levels <= 12:
            reasons.append("few distinct levels — consider treating as categorical")
    elif t == "categorical":
        if fam["needs_onehot"]:
            out["encoding"] = "onehot" if n_levels <= 200 else "frequency"
            if out["encoding"] == "onehot":
                reasons.append(f"{n_levels} levels → one-hot (top 30 kept, rest grouped)")
            else:
                reasons.append(f"{n_levels:,} levels → frequency encoding")
        elif n_levels <= 8:
            out["encoding"] = "onehot"
            reasons.append(f"{n_levels} levels → one-hot")
        elif n_levels <= 1000:
            out["encoding"] = "ordinal"
            reasons.append(f"{n_levels} levels → ordinal codes (trees split these well, no width blow-up)")
        else:
            out["encoding"] = "frequency"
            reasons.append(f"{n_levels:,} levels → frequency encoding")
        out["params"] = dict(DEFAULT_PARAMS.get(out["encoding"], {}))
        if null_pct > 0:
            reasons.append("missing values become their own '__missing__' level")
    elif t == "binary":
        out["imputation"] = "most_frequent" if null_pct < 0.05 else "constant"
    elif t == "text":
        out["encoding"] = "tfidf_svd"
        out["params"] = dict(DEFAULT_PARAMS["tfidf_svd"])
        reasons.append("free text → TF-IDF reduced to 32 SVD components")
    elif t == "datetime":
        if not fam["handles_nan"]:
            out["imputation"] = "median"
            out["scaling"] = "standard"
        reasons.append("expanded to year / month / weekday / hour / epoch-days")

    out["reasons"] = reasons
    return out


def suggest_all(t: str, stats: dict[str, Any], is_target: bool) -> dict[str, dict[str, Any]]:
    return {f: suggest(t, stats, is_target, f) for f in FAMILIES}


def settings_of(d: dict[str, Any]) -> dict[str, Any]:
    return {k: d.get(k) for k in SETTING_KEYS}
