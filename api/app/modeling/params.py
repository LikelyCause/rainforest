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
BAGGING = {"data_sample_strategy": ("bagging",)}
LIGHTGBM = RUN + [
    Setting("boosting_type", "Boosting", "How trees are added.", "trees", "choice", "gbdt", options=(
        _opt("gbdt", "Gradient boosting", "Each tree corrects the previous ones. The usual choice."),
        _opt("dart", "DART", "Drops random trees while boosting, which can reduce overfitting; slower."),
        _opt("rf", "Random forest", "Independent trees on sampled rows or features; needs a row sample below 1 with "
             "bagging, or a feature sample below 1."))),
    Setting("n_estimators", "Trees", "Boosting rounds. More fits more closely, and slower.", "trees", "int", 600, 1,
            10_000, 50),
    Setting("num_leaves", "Leaves per tree", "The main complexity control. More leaves fit finer patterns and "
            "overfit sooner.", "trees", "int", 63, 2, 4096),
    Setting("max_depth", "Max depth", "Caps how deep a tree grows; leaves per tree usually matters more.", "trees",
            "int_or_none", -1, 1, 64, none_value=-1, none_label="No limit"),
    Setting("min_child_samples", "Min rows per leaf", "Larger values make smoother, more conservative trees.",
            "trees", "int", 40, 1, 10_000),
    Setting("learning_rate", "Learning rate", "How much each tree counts. Lower needs more trees and usually "
            "generalizes better.", "learning", "float", 0.04, 0.001, 1.0, 0.005),
    Setting("subsample", "Row sample", "Share of rows each tree sees (bagging).", "sampling", "float", 0.8, 0.1,
            1.0, 0.05, when=BAGGING),
    Setting("subsample_freq", "Bagging every", "Redraw the row sample every k trees; 0 turns bagging off.",
            "sampling", "int", 1, 0, 100, when=BAGGING),
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
            "largest errors; with GOSS the bagging settings don't apply.", "advanced", "choice", "bagging",
            options=(_opt("bagging", "Bagging"), _opt("goss", "GOSS"))),
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

XGB_DART = {"booster": ("dart",)}
XGBOOST = RUN + [
    Setting("booster", "Boosting", "How trees are added.", "trees", "choice", "gbtree", options=(
        _opt("gbtree", "Gradient boosting", "Each tree corrects the previous ones. The usual choice."),
        _opt("dart", "DART", "Drops random trees while boosting, which can reduce overfitting; slower."))),
    Setting("n_estimators", "Trees", "Boosting rounds. More fits more closely, and slower.", "trees", "int", 600, 1,
            10_000, 50),
    Setting("max_depth", "Max depth", "The main complexity control. Deeper trees fit finer patterns and overfit "
            "sooner.", "trees", "int_or_none", 6, 1, 64, none_value=0, none_label="No limit"),
    Setting("grow_policy", "Growth", "Depth-wise splits a tree level by level; loss-guided splits whichever leaf "
            "helps most, as LightGBM does.", "trees", "choice", "depthwise",
            options=(_opt("depthwise", "Depth-wise"), _opt("lossguide", "Loss-guided"))),
    Setting("max_leaves", "Max leaves", "Caps the leaves per tree; mostly used with loss-guided growth.", "trees",
            "int_or_none", 0, 2, 4096, none_value=0, none_label="No limit"),
    Setting("min_child_weight", "Min leaf weight", "A leaf needs at least this much weight (its row count, for "
            "squared error). Larger values make smoother, more conservative trees.", "trees", "float", 1.0, 0.0,
            1000.0, 0.5),
    Setting("learning_rate", "Learning rate", "How much each tree counts. Lower needs more trees and usually "
            "generalizes better.", "learning", "float", 0.05, 0.001, 1.0, 0.005),
    Setting("subsample", "Row sample", "Share of rows each tree sees.", "sampling", "float", 0.8, 0.1, 1.0, 0.05),
    Setting("colsample_bytree", "Feature sample", "Share of features each tree sees.", "sampling", "float", 0.7,
            0.1, 1.0, 0.05),
    Setting("reg_alpha", "L1 penalty", "Pushes leaf values to zero; helps with many weak features.",
            "regularization", "float", 0.0, 0.0, 100.0, 0.1),
    Setting("reg_lambda", "L2 penalty", "Shrinks leaf values; steadies noisy targets.", "regularization", "float",
            1.0, 0.0, 100.0, 0.1),
    Setting("gamma", "Min split loss", "A split must reduce the loss by at least this much.", "regularization",
            "float", 0.0, 0.0, 100.0, 0.01),
    Setting("scale_pos_weight", "Positive-class weight", "Weight of a binary target's positive class; above 1 "
            "trades precision for recall. Multiclass targets ignore it.", "classes", "float", 1.0, 0.01, 1000.0, 0.1,
            applies="classes"),
    Setting("objective", "Loss", "What a number target is fitted to minimise.", "numbers", "choice",
            "reg:squarederror", options=(
                _opt("reg:squarederror", "Squared error", "The usual choice; large errors count most."),
                _opt("reg:absoluteerror", "Absolute error", "Robust to outliers; predicts the median."),
                _opt("reg:pseudohubererror", "Pseudo-Huber", "Squared for small errors, absolute for large ones."),
                _opt("reg:quantileerror", "Quantile", "Predicts a quantile instead of the mean."),
                _opt("reg:squaredlogerror", "Squared log error", "Relative errors; needs targets > -1."),
                _opt("count:poisson", "Poisson", "Counts; needs targets ≥ 0."),
                _opt("reg:gamma", "Gamma", "Positive, skewed amounts; needs targets > 0."),
                _opt("reg:tweedie", "Tweedie", "Amounts with many zeros; needs targets ≥ 0.")), applies="numbers"),
    Setting("huber_slope", "Huber slope", "Where Pseudo-Huber turns from squared to absolute error.", "numbers",
            "float", 1.0, 0.001, 1000.0, 0.1, applies="numbers", when={"objective": ("reg:pseudohubererror",)}),
    Setting("quantile_alpha", "Quantile", "The quantile to predict; 0.5 is the median.", "numbers", "float", 0.5,
            0.01, 0.99, 0.05, applies="numbers", when={"objective": ("reg:quantileerror",)}),
    Setting("tweedie_variance_power", "Tweedie power", "Near 1 behaves like Poisson, near 2 like gamma.", "numbers",
            "float", 1.5, 1.0, 1.99, 0.05, applies="numbers", when={"objective": ("reg:tweedie",)}),
    Setting("max_bin", "Max bins", "Value buckets per feature; more is finer and slower.", "advanced", "int", 256,
            2, 65_535),
    Setting("max_delta_step", "Max leaf step", "Caps each leaf's update; 0 means no cap. 1 to 10 can steady a very "
            "unbalanced binary target.", "advanced", "float", 0.0, 0.0, 100.0, 0.1),
    Setting("colsample_bylevel", "Feature sample per level", "Share of the tree's features tried at each depth.",
            "advanced", "float", 1.0, 0.1, 1.0, 0.05),
    Setting("colsample_bynode", "Feature sample per split", "Share of the level's features tried at each split.",
            "advanced", "float", 1.0, 0.1, 1.0, 0.05),
    Setting("sampling_method", "Row sampling", "Uniform draws rows at random; gradient-based favours rows with "
            "large errors, so a small row sample loses less.", "advanced", "choice", "uniform",
            options=(_opt("uniform", "Uniform"), _opt("gradient_based", "Gradient-based"))),
    Setting("num_parallel_tree", "Trees per round", "Grow several trees each round; with row or feature sampling "
            "below 1, each round is a small random forest.", "advanced", "int", 1, 1, 100),
    Setting("multi_strategy", "Trees for multiclass", "One tree per class each round, or one tree whose leaves "
            "hold a value for every class.", "advanced", "choice", "one_output_per_tree",
            options=(_opt("one_output_per_tree", "One per class"), _opt("multi_output_tree", "One for all classes")),
            applies="classes"),
    Setting("rate_drop", "DART drop rate", "Share of trees dropped each round.", "advanced", "float", 0.1, 0.0, 1.0,
            0.05, when=XGB_DART),
    Setting("skip_drop", "DART skip chance", "Chance of skipping the drop in a round.", "advanced", "float", 0.0,
            0.0, 1.0, 0.05, when=XGB_DART),
    Setting("one_drop", "DART drop at least one", "Always drop at least one tree.", "advanced", "bool", False,
            when=XGB_DART),
    Setting("sample_type", "DART selection", "Drop trees with equal chance, or by weight.", "advanced", "choice",
            "uniform", options=(_opt("uniform", "Equal chance"), _opt("weighted", "By weight")), when=XGB_DART),
    Setting("normalize_type", "DART normalization", "How new and dropped trees are weighted afterwards.",
            "advanced", "choice", "tree", options=(_opt("tree", "Per tree"), _opt("forest", "Per forest")),
            when=XGB_DART),
]

CATALOG: dict[str, list[Setting]] = {"lightgbm": LIGHTGBM, "xgboost": XGBOOST}
# The lowest training target a loss accepts: (bound, whether the bound itself is allowed).
TARGET_FLOOR = {"poisson": (0, True), "tweedie": (0, True), "gamma": (0, False), "count:poisson": (0, True),
                "reg:tweedie": (0, True), "reg:gamma": (0, False), "reg:squaredlogerror": (-1, False)}


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
    alpha = params.get("alpha", 0.9)
    if params.get("objective") == "quantile" and isinstance(alpha, (int, float)) and not 0 < alpha < 1:
        out.append("alpha must be between 0 and 1 for the quantile loss")
    return out


def normalize(model: str, params: dict) -> dict:
    """A run's params with whole numbers for whole-number settings (JSON clients may send 600.0, which LightGBM refuses
    mid-run)."""
    kinds = {s.name: s.kind for s in settings_for(model) or []}
    return {k: int(v) if kinds.get(k) in ("int", "int_or_none") and isinstance(v, float) and v.is_integer() else v
            for k, v in params.items()}


def effective(model: str, params: dict) -> dict:
    """What a model is built with. A setting that another one switches on gets its shown default even when unchanged
    (XGBoost's quantile loss has no alpha of its own), and GOSS samples rows itself, so bagging is switched off."""
    catalog = {s.name: s for s in settings_for(model) or []}
    out = dict(params)
    for s in catalog.values():
        if s.when and s.name not in out and all(out.get(k, catalog[k].default) in v for k, v in s.when.items()):
            out[s.name] = s.default
    if model == "lightgbm" and out.get("data_sample_strategy") == "goss":
        out.update(subsample=1.0, subsample_freq=0)
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
    """A loss that needs targets above a floor (Poisson ≥ 0, gamma > 0, ...), against a number target's training
    minimum."""
    floor = TARGET_FLOOR.get(params.get("objective", ""))
    if floor is None or target_min is None:
        return None
    bound, allowed = floor
    if target_min < bound or (target_min == bound and not allowed):
        rule = f"{'≥' if allowed else '>'} {bound:g}"
        return f"The {params['objective']} loss needs targets {rule}; this target's training minimum is {target_min:g}"
    return None
