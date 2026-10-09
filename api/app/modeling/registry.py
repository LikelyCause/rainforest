"""Model registry: every classifier that can sit inside the ClassifierChain.

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
    from xgboost import XGBClassifier

    return XGBClassifier(**params, random_state=seed)


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
        ),
        ModelSpec(
            "xgboost", "XGBoost", "tree", True, _xgboost,
            {"n_estimators": 600, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
             "colsample_bytree": 0.7, "tree_method": "hist", "n_jobs": -1, "verbosity": 0},
            requires="xgboost",
            description="Depth-wise gradient boosting; trains on an NVIDIA GPU in the GPU image.",
            cuda_params={"device": "cuda", "tree_method": "hist"},
        ),
        ModelSpec(
            "hist_gradient_boosting", "Histogram GBM (sklearn)", "tree", True, _hgb,
            {"max_iter": 400, "learning_rate": 0.06, "max_leaf_nodes": 63, "l2_regularization": 1.0,
             "early_stopping": False},
            importance="none",
            description="scikit-learn's LightGBM-style booster; no extra dependency.",
        ),
        ModelSpec(
            "random_forest", "Random forest", "tree", True, _random_forest,
            {"n_estimators": 400, "min_samples_leaf": 3, "max_features": "sqrt", "n_jobs": -1},
            description="Bagged trees; a robust, low-tuning baseline.",
        ),
        ModelSpec(
            "logistic_regression", "Logistic regression", "linear", False, _logreg,
            {"max_iter": 3000, "C": 1.0},
            importance="coef_",
            description="Linear baseline; needs a linear-family export (scaled, imputed, one-hot).",
        ),
        ModelSpec(
            "mlp", "MLP", "neural", False, _mlp,
            {"hidden_layer_sizes": (128, 64), "alpha": 1e-4, "early_stopping": True, "max_iter": 200},
            importance="none",
            description="Small feed-forward network; needs a neural-family export.",
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


def compatibility(spec: ModelSpec, export_family: str, contains_nan: bool) -> str | None:
    """None if the model can train on an export, else the reason it can't."""
    if contains_nan and not spec.handles_nan:
        return f"{spec.label} can't handle missing values and this export contains NaN"
    if spec.family != export_family and spec.family != "tree":
        return (f"{spec.label} needs a {FAMILIES[spec.family]['label'].lower()} export "
                f"(this one was prepared for {FAMILIES[export_family]['label'].lower()})")
    return None


def catalog() -> list[dict[str, Any]]:
    return [
        {"key": m.key, "label": m.label, "family": m.family, "handles_nan": m.handles_nan,
         "available": m.available, "description": m.description, "devices": m.devices}
        for m in MODELS.values()
    ]


def importances(spec: ModelSpec, est: Any, names: list[str]) -> np.ndarray | None:
    try:
        if spec.importance == "booster_gain":
            return np.asarray(est.booster_.feature_importance(importance_type="gain"), dtype=float)
        if spec.importance == "feature_importances_":
            return np.asarray(est.feature_importances_, dtype=float)
        if spec.importance == "coef_":
            return np.abs(np.asarray(est.coef_, dtype=float)).mean(axis=0)
    except (AttributeError, ValueError):
        return None
    return None
