"""Each model's settings as data: what the Train menu shows, what a run may set, and which links of the chain a value
reaches (`applies`). A model with no entry here keeps free-form `params`, checked by the registry as before."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

GROUPS = ("run", "trees", "learning", "sampling", "regularization", "classes", "numbers", "advanced")


@dataclass(frozen=True)
class Setting:
    name: str
    label: str
    help: str
    group: str
    kind: str  # choice | int | float | bool | int_or_none
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    log: bool = False
    options: tuple[dict, ...] | None = None
    applies: str = "both"
    when: dict[str, tuple] | None = None
    none_value: int | None = None
    none_label: str | None = None

    def to_json(self) -> dict:
        d = asdict(self)
        d["options"] = list(self.options) if self.options else None
        d["when"] = {k: list(v) for k, v in self.when.items()} if self.when else None
        return d


def _opt(value, label, help=None) -> dict:
    return {"value": value, "label": label, **({"help": help} if help else {})}


RUN = [
    Setting("seed", "Seed", "Fixes the random choices, so the same settings train the same model.", "run", "int", 42,
            0, 2**31 - 1),
    Setting("cv", "Chain folds", "Folds that produce the out-of-fold predictions a chain passes down; off passes "
            "the true values instead.", "run", "choice", 5,
            options=(_opt(0, "Off"), *(_opt(k, str(k)) for k in range(2, 11)))),
]

DART = {"boosting_type": ("dart",)}
GOSS = {"data_sample_strategy": ("goss",)}
LIGHTGBM = RUN + [
    Setting("boosting_type", "Boosting", "How trees are added.", "trees", "choice", "gbdt", options=(
        _opt("gbdt", "Gradient boosting", "Each tree corrects the previous ones. The usual choice."),
        _opt("dart", "DART", "Drops random trees while boosting, which can reduce overfitting; slower."),
        _opt("rf", "Random forest", "Independent trees on bagged rows; needs subsample < 1 and a bagging frequency."))),
    Setting("n_estimators", "Trees", "Boosting rounds. More fits more closely, and slower.", "trees", "int", 600, 1,
            10_000, 50),
    Setting("num_leaves", "Leaves per tree", "The main complexity control. More leaves fit finer patterns and "
            "overfit sooner.", "trees", "int", 63, 2, 4096),
    Setting("max_depth", "Max depth", "Caps how deep a tree grows; leaves per tree usually matters more.", "trees",
            "int_or_none", -1, 1, 64, none_value=-1, none_label="No limit"),
    Setting("min_child_samples", "Min rows per leaf", "Larger values make smoother, more conservative trees.",
            "trees", "int", 40, 1, 10_000),
    Setting("learning_rate", "Learning rate", "How much each tree counts. Lower needs more trees and usually "
            "generalizes better.", "learning", "float", 0.04, 0.001, 1.0, log=True),
    Setting("subsample", "Row sample", "Share of rows each tree sees (bagging).", "sampling", "float", 0.8, 0.1,
            1.0, 0.05),
    Setting("subsample_freq", "Bagging every", "Redraw the row sample every k trees; 0 turns bagging off.",
            "sampling", "int", 1, 0, 100),
    Setting("colsample_bytree", "Feature sample", "Share of features each tree sees.", "sampling", "float", 0.7,
            0.1, 1.0, 0.05),
    Setting("reg_alpha", "L1 penalty", "Pushes leaf values to zero; helps with many weak features.",
            "regularization", "float", 0.0, 0.0, 100.0, 0.1),
    Setting("reg_lambda", "L2 penalty", "Shrinks leaf values; steadies noisy targets.", "regularization", "float",
            1.0, 0.0, 100.0, 0.1),
    Setting("min_split_gain", "Min split gain", "A split must improve the loss by at least this much.",
            "regularization", "float", 0.0, 0.0, 100.0, 0.01),
    Setting("class_weight", "Class weights", "Balanced weights rare classes up, trading precision for recall.",
            "classes", "choice", None, options=(_opt(None, "None"), _opt("balanced", "Balanced")), applies="classes"),
    Setting("objective", "Loss", "What a number target is fitted to minimise.", "numbers", "choice", "regression",
            options=(
                _opt("regression", "Squared error", "The usual choice; large errors count most."),
                _opt("regression_l1", "Absolute error", "Robust to outliers; predicts the median."),
                _opt("huber", "Huber", "Squared for small errors, absolute for large ones."),
                _opt("fair", "Fair", "Like Huber, smoother."),
                _opt("quantile", "Quantile", "Predicts a quantile (alpha) instead of the mean."),
                _opt("mape", "Percentage error", "Relative errors; targets near 0 dominate."),
                _opt("poisson", "Poisson", "Counts; needs targets ≥ 0."),
                _opt("gamma", "Gamma", "Positive, skewed amounts; needs targets > 0."),
                _opt("tweedie", "Tweedie", "Amounts with many zeros; needs targets ≥ 0.")), applies="numbers"),
    Setting("alpha", "Alpha", "Huber's switch point, or the quantile to predict (between 0 and 1).", "numbers",
            "float", 0.9, 0.001, 1000.0, 0.05, applies="numbers", when={"objective": ("huber", "quantile")}),
    Setting("max_bin", "Max bins", "Value buckets per feature; more is finer and slower.", "advanced", "int", 255,
            2, 65_535),
    Setting("min_child_weight", "Min leaf weight", "Minimum summed hessian in a leaf.", "advanced", "float", 0.001,
            0.0, 100.0, 0.001),
    Setting("subsample_for_bin", "Rows for binning", "Rows sampled to choose the buckets.", "advanced", "int",
            200_000, 1_000, 10_000_000, 10_000),
    Setting("extra_trees", "Extremely randomized", "Pick split thresholds at random; faster and less overfit.",
            "advanced", "bool", False),
    Setting("path_smooth", "Path smoothing", "Shrinks leaves towards their parents; helps small leaves.",
            "advanced", "float", 0.0, 0.0, 1000.0, 0.1),
    Setting("max_delta_step", "Max leaf output", "Caps each leaf's value; 0 means no cap.", "advanced", "float",
            0.0, 0.0, 100.0, 0.1),
    Setting("feature_fraction_bynode", "Feature sample per split", "Share of features tried at each split.",
            "advanced", "float", 1.0, 0.1, 1.0, 0.05),
    Setting("linear_tree", "Linear leaves", "Fit a linear model in each leaf instead of a constant.", "advanced",
            "bool", False),
    Setting("data_sample_strategy", "Row sampling", "Bagging samples rows at random; GOSS keeps the rows with the "
            "largest errors.", "advanced", "choice", "bagging", options=(_opt("bagging", "Bagging"),
                                                                         _opt("goss", "GOSS"))),
    Setting("top_rate", "GOSS top share", "Share of large-error rows GOSS keeps.", "advanced", "float", 0.2, 0.01,
            1.0, 0.05, when=GOSS),
    Setting("other_rate", "GOSS other share", "Share of the remaining rows GOSS samples.", "advanced", "float", 0.1,
            0.01, 1.0, 0.05, when=GOSS),
    Setting("drop_rate", "DART drop rate", "Share of trees dropped each round.", "advanced", "float", 0.1, 0.0, 1.0,
            0.05, when=DART),
    Setting("max_drop", "DART max drops", "Most trees dropped in one round.", "advanced", "int", 50, 1, 1000,
            when=DART),
    Setting("skip_drop", "DART skip chance", "Chance of skipping the drop in a round.", "advanced", "float", 0.5,
            0.0, 1.0, 0.05, when=DART),
    Setting("uniform_drop", "DART uniform drop", "Drop trees with equal chance instead of by weight.", "advanced",
            "bool", False, when=DART),
    Setting("xgboost_dart_mode", "XGBoost DART mode", "Use XGBoost's DART normalization.", "advanced", "bool",
            False, when=DART),
    Setting("is_unbalance", "Unbalanced (binary)", "Reweight a binary target's classes; don't combine with the "
            "positive-class weight.", "advanced", "bool", False, applies="classes"),
    Setting("scale_pos_weight", "Positive-class weight", "Weight of the positive class of a binary target.",
            "advanced", "float", 1.0, 0.01, 1000.0, 0.1, applies="classes"),
]

CATALOG: dict[str, list[Setting]] = {"lightgbm": LIGHTGBM}
NON_NEGATIVE = {"poisson": "≥ 0", "tweedie": "≥ 0", "gamma": "> 0"}


def settings_for(model: str) -> list[Setting] | None:
    return CATALOG.get(model)


def _value_problem(s: Setting, v: Any) -> str | None:
    if s.kind == "choice":
        return None if v in [o["value"] for o in s.options or ()] else (
            f"{s.name} must be one of {', '.join(repr(o['value']) for o in s.options or ())}")
    if s.kind == "bool":
        return None if isinstance(v, bool) else f"{s.name} must be true or false"
    if s.kind == "int_or_none" and v == s.none_value:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return f"{s.name} must be a number"
    if s.kind in ("int", "int_or_none") and not float(v).is_integer():
        return f"{s.name} must be a whole number"
    if (s.min is not None and v < s.min) or (s.max is not None and v > s.max):
        return f"{s.name} must be from {s.min:g} to {s.max:g}" + (f" (or {s.none_label})" if s.none_label else "")
    return None


def setting_problems(model: str, params: dict) -> list[str]:
    """What's wrong with a run's `params` for `model`: unknown names, values outside their rules, settings sent while
    the setting they depend on hides them, and LightGBM's quantile alpha outside (0, 1)."""
    catalog = {s.name: s for s in settings_for(model) or [] if s.group != "run"}
    out = []
    for name, v in params.items():
        s = catalog.get(name)
        if s is None:
            out.append(f"{name} isn't a setting of this model")
            continue
        if problem := _value_problem(s, v):
            out.append(problem)
            continue
        for other, allowed in (s.when or {}).items():
            current = params.get(other, catalog[other].default)
            if current not in allowed:
                out.append(f"{name} applies only when {other} is {' or '.join(map(str, allowed))}")
    if params.get("objective") == "quantile" and not 0 < params.get("alpha", 0.9) < 1:
        out.append("alpha must be between 0 and 1 for the quantile loss")
    return out


def applies_to(model: str, kind: str | None = None) -> set[str]:
    """Names of `model`'s settings that apply to one kind of link only (`kind`: classes or numbers; None for both)."""
    return {s.name for s in settings_for(model) or [] if s.applies != "both" and (kind is None or s.applies == kind)}


def route(model: str, params: dict) -> tuple[dict, dict]:
    """Split a run's params into the classifier's and the regressor's by each setting's `applies`."""
    applies = {s.name: s.applies for s in settings_for(model) or []}
    clf = {k: v for k, v in params.items() if applies.get(k, "both") in ("both", "classes")}
    reg = {k: v for k, v in params.items() if applies.get(k, "both") in ("both", "numbers")}
    return clf, reg


def objective_problem(params: dict, target_min: float | None) -> str | None:
    """A loss that needs non-negative (or positive) numbers, against a number target's training minimum."""
    rule = NON_NEGATIVE.get(params.get("objective", ""))
    if rule is None or target_min is None:
        return None
    if (rule == "> 0" and target_min <= 0) or (rule == "≥ 0" and target_min < 0):
        return f"The {params['objective']} loss needs targets {rule}; this target's training minimum is {target_min:g}"
    return None
