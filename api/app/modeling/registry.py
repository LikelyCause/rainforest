"""Model registry: every model that can sit in a target chain, as a classifier and as a regressor.

Adding a model = adding one `ModelSpec` below. A spec says which family it belongs to (and so
which exports it can train on), whether it routes NaN natively, how to build it, and how to read
feature importances back out. Optional libraries (XGBoost) register only when importable.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..semantics import FAMILIES
from . import params as model_params


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str
    handles_nan: bool
    build: Callable[[dict[str, Any], int], Any]
    default_params: dict[str, Any] = field(default_factory=dict)
    importance: str = "feature_importances_"  # "booster_gain" | "feature_importances_" | "coef_" | "none"
    requires: str | None = None  # optional import name
    description: str = ""
    cuda_params: dict[str, Any] | None = None  # set => the model can train on a CUDA GPU
    build_regressor: Callable[[dict[str, Any], int], Any] | None = None  # for targets predicted as numbers
    regressor_defaults: dict[str, Any] = field(default_factory=dict)

    @property
    def devices(self) -> list[str]:
        return ["cpu", "cuda"] if self.cuda_params is not None else ["cpu"]

    @property
    def available(self) -> bool:
        return self.requires is None or importlib.util.find_spec(self.requires) is not None


def _lightgbm(params: dict[str, Any], seed: int):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(**params, random_state=seed)


def _xgboost(params: dict[str, Any], seed: int):
    from .xgb_compact import CompactXGBClassifier

    return CompactXGBClassifier(**params, random_state=seed)


def _hgb(params: dict[str, Any], seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(**params, random_state=seed)


def _random_forest(params: dict[str, Any], seed: int):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(**params, random_state=seed)


def _logreg(params: dict[str, Any], seed: int):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(**params, random_state=seed)


def _mlp(params: dict[str, Any], seed: int):
    from sklearn.neural_network import MLPClassifier

    return MLPClassifier(**params, random_state=seed)


def _lightgbm_reg(params: dict[str, Any], seed: int):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(**params, random_state=seed)


def _xgboost_reg(params: dict[str, Any], seed: int):
    from xgboost import XGBRegressor

    return XGBRegressor(**params, random_state=seed)


def _hgb_reg(params: dict[str, Any], seed: int):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(**params, random_state=seed)


def _random_forest_reg(params: dict[str, Any], seed: int):
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(**params, random_state=seed)


def _ridge(params: dict[str, Any], seed: int):
    from sklearn.linear_model import Ridge

    return Ridge(**params, random_state=seed)


def _mlp_reg(params: dict[str, Any], seed: int):
    from sklearn.neural_network import MLPRegressor

    return MLPRegressor(**params, random_state=seed)


MODELS: dict[str, ModelSpec] = {
    m.key: m
    for m in [
        ModelSpec(
            "lightgbm", "LightGBM", "tree", True, _lightgbm,
            {"n_estimators": 600, "learning_rate": 0.04, "num_leaves": 63, "min_child_samples": 40,
             "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.7, "reg_lambda": 1.0,
             "n_jobs": -1, "verbose": -1},
            importance="booster_gain",
            description="Leaf-wise gradient boosting. The default.",
            build_regressor=_lightgbm_reg,
        ),
        ModelSpec(
            "xgboost", "XGBoost", "tree", True, _xgboost,
            {"n_estimators": 600, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
             "colsample_bytree": 0.7, "tree_method": "hist", "n_jobs": -1, "verbosity": 0},
            requires="xgboost",
            description="Depth-wise gradient boosting; trains on an NVIDIA GPU in the GPU image.",
            cuda_params={"device": "cuda", "tree_method": "hist"},
            build_regressor=_xgboost_reg,
        ),
        ModelSpec(
            "hist_gradient_boosting", "Histogram GBM (sklearn)", "tree", True, _hgb,
            {"max_iter": 400, "learning_rate": 0.06, "max_leaf_nodes": 63, "l2_regularization": 1.0,
             "early_stopping": False},
            importance="none",
            description="scikit-learn's LightGBM-style booster; no extra dependency.",
            build_regressor=_hgb_reg,
        ),
        ModelSpec(
            "random_forest", "Random forest", "tree", True, _random_forest,
            {"n_estimators": 400, "min_samples_leaf": 3, "max_features": "sqrt", "n_jobs": -1},
            description="Bagged trees; a robust, low-tuning baseline.",
            build_regressor=_random_forest_reg,
        ),
        ModelSpec(
            "logistic_regression", "Logistic regression", "linear", False, _logreg,
            {"max_iter": 3000, "C": 1.0},
            importance="coef_",
            description="Linear baseline; needs a linear-family export (scaled, imputed, one-hot). Numbers use ridge "
                        "regression.",
            build_regressor=_ridge, regressor_defaults={"alpha": 1.0},
        ),
        ModelSpec(
            "mlp", "MLP", "neural", False, _mlp,
            {"hidden_layer_sizes": (128, 64), "alpha": 1e-4, "early_stopping": True, "max_iter": 200},
            importance="none",
            description="Small feed-forward network; needs a neural-family export.",
            build_regressor=_mlp_reg,
        ),
    ]
}
DEFAULT_MODEL = "lightgbm"


def get_model(key: str) -> ModelSpec:
    spec = MODELS.get(key)
    if spec is None:
        raise ValueError(f"unknown model {key!r} (choose from {', '.join(MODELS)})")
    if not spec.available:
        raise ValueError(f"{spec.label} is not installed in this image ({spec.description})")
    return spec


def device_params(spec: ModelSpec, device: str) -> dict[str, Any]:
    """Parameters that put a model on `device`. CPU adds nothing, so the CPU path is unchanged."""
    if device == "cpu":
        return {}
    if device == "cuda" and spec.cuda_params is not None:
        return dict(spec.cuda_params)
    raise ValueError(f"{spec.label} can't train on {device!r} (supports: {', '.join(spec.devices)})")


def _lightgbm_native_params() -> set[str]:
    """LightGBM also takes its native parameter names and aliases (min_data_in_leaf, lambda_l2, ...)."""
    try:
        from lightgbm.basic import _ConfigAliases

        return {a for names in _ConfigAliases._get_all_param_aliases().values() for a in names}
    except Exception:  # noqa: BLE001 - private API; without it only the sklearn names are accepted
        return set()


def param_problems(spec: ModelSpec, params: dict[str, Any]) -> list[str]:
    """Why `params` can't configure `spec`'s estimator: names it doesn't have, or values its own checks reject."""
    if "random_state" in params:
        return ["random_state is set from the run's seed; pass seed instead"]
    est = spec.build({}, 0)
    known = set(est.get_params())
    unknown = sorted(set(params) - known - (_lightgbm_native_params() if spec.key == "lightgbm" else set()))
    if unknown:
        return [f"{spec.label} has no parameter {k!r}" for k in unknown]
    try:
        est.set_params(**{k: v for k, v in params.items() if k in known})
        if hasattr(est, "_parameter_constraints"):  # sklearn's own type and range checks
            est._validate_params()
    except (ValueError, TypeError) as exc:
        return [str(exc)]
    return []


# Accepted by some regressors too (LGBMRegressor takes class_weight), but they describe classes.
CLASSIFIER_ONLY = {"class_weight", "is_unbalance", "unbalance", "unbalanced_sets", "scale_pos_weight", "objective",
                   "num_class", "num_classes", "multi_class", "C", "penalty", "l1_ratio"}


def _classifier_only(spec: ModelSpec) -> set[str]:
    """CLASSIFIER_ONLY plus LightGBM's aliases of those names (application, app, objective_type, loss, ...)."""
    if spec.key != "lightgbm":
        return CLASSIFIER_ONLY
    try:
        from lightgbm.basic import _ConfigAliases

        return CLASSIFIER_ONLY | {a for k in CLASSIFIER_ONLY for a in _ConfigAliases.get(k)}
    except Exception:  # noqa: BLE001 - private API, as in _lightgbm_native_params
        return CLASSIFIER_ONLY


def regressor_params(spec: ModelSpec, params: dict[str, Any]) -> dict[str, Any]:
    """The run's parameters the regressor accepts, minus the ones that describe classes."""
    if spec.build_regressor is None:
        return {}
    known = set(spec.build_regressor({}, 0).get_params())
    if spec.key == "lightgbm":
        known |= _lightgbm_native_params()
    if model_params.settings_for(spec.key):
        # A catalogued model says which settings reach a regressor; the rest (class weights, ...) stay with classifiers.
        catalogued = {s.name for s in model_params.settings_for(spec.key) or []}
        routed = model_params.route(spec.key, params)[1]
        return {k: v for k, v in routed.items()
                if k in known and (k in catalogued or k not in _classifier_only(spec))}
    return {k: v for k, v in params.items() if k in known and k not in _classifier_only(spec)}


def classifier_params(spec: ModelSpec, params: dict[str, Any]) -> dict[str, Any]:
    """The run's parameters a classifier gets: a catalogued model's settings for number targets stay out."""
    if not model_params.settings_for(spec.key):
        return params
    return {k: v for k, v in params.items() if k not in model_params.applies_to(spec.key, "numbers")}


def dry_fit_problem(spec: ModelSpec, params: dict[str, Any], kinds: list[str]) -> str | None:
    """Fit the run's settings on 64 synthetic rows, so a combination the library refuses fails when the run is asked
    for, not minutes into it. Rounds are capped so a large tree count stays quick."""
    merged = {**spec.default_params, **params}
    if "n_estimators" in merged:
        merged["n_estimators"] = min(int(merged["n_estimators"]), 5)
    if "n_jobs" in merged:
        merged["n_jobs"] = 1
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 4))
    classes = [k for k in kinds if k != "regression"]
    try:
        if classes:
            y = (X[:, 0] > 0).astype(int) + (0 if all(k == "binary" for k in classes) else (X[:, 1] > 0))
            spec.build(classifier_params(spec, merged), 0).fit(X, y)
        if "regression" in kinds and spec.build_regressor is not None:
            spec.build_regressor({**spec.regressor_defaults, **regressor_params(spec, merged)}, 0).fit(X, np.abs(X[:, 0]) + 1)
    except Exception as exc:  # noqa: BLE001 - the library's own message is the useful part
        text = str(exc)
        plain = next((msg for needle, msg in LIBRARY_RULES.items() if needle in text), text)
        return f"{spec.label} refused these settings: {plain}"
    return None


# LightGBM's fatal checks, in words.
LIBRARY_RULES = {
    "rf.hpp": "Random forest needs bagging (a row sample below 1 with a bagging frequency above 0) or a feature sample "
              "below 1",
    "goss.hpp": "the GOSS top and other shares must add up to at most 1",
    "is_unbalance and scale_pos_weight": "set Unbalanced or a positive-class weight, not both",
}


def regression_problem(spec: ModelSpec, kinds: list[str]) -> str | None:
    if "regression" in kinds and spec.build_regressor is None:
        return f"{spec.label} can't predict a number yet; choose another model or predict the target as bands"
    return None


def compatibility(spec: ModelSpec, export_family: str, contains_nan: bool) -> str | None:
    """None if the model can train on an export, else the reason it can't."""
    if contains_nan and not spec.handles_nan:
        return f"{spec.label} can't handle missing values and this export contains NaN"
    if spec.family != export_family and spec.family != "tree":
        return (f"{spec.label} needs a {FAMILIES[spec.family]['label'].lower()} export "
                f"(this one was prepared for {FAMILIES[export_family]['label'].lower()})")
    return None


# Models on the roadmap: listed so people know they're coming, never trainable.
PLANNED = {
    "linear_svm": {"label": "Linear SVM", "family": "linear", "description": "Linear support vector machine."},
    "knn": {"label": "k-nearest neighbours", "family": "neural", "description": "Distance-based classifier."},
    "cnn": {"label": "Convolutional network", "family": "neural", "description": "Convolutional neural network."},
    "rnn": {"label": "Recurrent network", "family": "neural", "description": "Recurrent neural network."},
}


def catalog() -> list[dict[str, Any]]:
    return [
        {"key": m.key, "label": m.label, "family": m.family, "handles_nan": m.handles_nan,
         "available": m.available, "description": m.description, "devices": m.devices, "planned": False,
         "settings": [s.to_json() for s in model_params.settings_for(m.key) or []] or None}
        for m in MODELS.values()
    ] + [
        {"key": key, **p, "handles_nan": False, "available": False, "devices": [], "planned": True, "settings": None}
        for key, p in PLANNED.items()
    ]


def importances(spec: ModelSpec, est: Any, names: list[str]) -> np.ndarray | None:
    try:
        if spec.importance == "booster_gain":
            return np.asarray(est.booster_.feature_importance(importance_type="gain"), dtype=float)
        if spec.importance == "feature_importances_":
            return np.asarray(est.feature_importances_, dtype=float)
        if spec.importance == "coef_":
            return np.abs(np.atleast_2d(np.asarray(est.coef_, dtype=float))).mean(axis=0)
    except (AttributeError, ValueError):
        return None
    return None


def order_problem(order: list[str], targets: list[str]) -> str | None:
    """What's wrong with a run's `order`: the export targets to train, once each, in chain order. One target trains a
    single classifier; a subset leaves the other targets out of the run."""
    if not order:
        return "Choose at least one target"
    if len(set(order)) != len(order):
        return "Each target can appear only once in the order"
    if unknown := [t for t in order if t not in targets]:
        return f"Not targets of this export: {', '.join(unknown)} (its targets: {', '.join(targets)})"
    return None
