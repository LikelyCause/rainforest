"""The transformers Rainforest registers itself, and the existing encoders' parameters as settings for the decision
panel's form (semantics.PARAM_RULES and DEFAULT_PARAMS stay their checks and defaults; a test keeps them equal)."""

from __future__ import annotations

from typing import Any

from ..modeling.params import Setting
from .registry import BuildContext, TransformSpec


def _yeo_johnson(params: dict[str, Any], ctx: BuildContext):
    from sklearn.preprocessing import PowerTransformer

    # Scaling is the field's own setting, so the power transform doesn't standardize too.
    return PowerTransformer(method="yeo-johnson", standardize=False)


YEO_JOHNSON = TransformSpec(
    key="yeo_johnson",
    dimension="transform",
    types=("numeric_continuous", "numeric_discrete"),
    families=("linear", "neural"),
    label="Yeo-Johnson (power transform)",
    help="Makes skewed numbers closer to normal. Unlike a log, it handles zero and negative values.",
    build=_yeo_johnson,
)

BUILTINS = [YEO_JOHNSON]


def _param(name: str, label: str, help: str, kind: str, default: Any, lo: float, hi: float, step: float) -> Setting:
    return Setting(name, label, help, "field", kind, default, lo, hi, step)


ENCODING_SETTINGS: dict[str, list[Setting]] = {
    "onehot": [
        _param(
            "max_categories",
            "Max categories",
            "Columns at most, counting the one the rarest levels share.",
            "int",
            30,
            2,
            1000,
            1,
        ),
        _param(
            "min_frequency",
            "Min frequency",
            "Levels rarer than this share of rows go to the shared column.",
            "float",
            0.01,
            0,
            0.99,
            0.005,
        ),
    ],
    "ordinal": [
        _param(
            "min_frequency",
            "Min frequency",
            "Levels rarer than this share of rows share one code.",
            "float",
            0.001,
            0,
            0.99,
            0.001,
        ),
    ],
    "tfidf_svd": [
        _param("svd_components", "SVD components", "Columns the text becomes.", "int", 32, 1, 256, 1),
        _param(
            "max_features",
            "Max vocabulary",
            "The most frequent words kept before SVD.",
            "int",
            20000,
            10,
            1_000_000,
            1000,
        ),
        _param("min_df", "Min rows", "Words in fewer rows than this are left out.", "int", 3, 1, 10_000, 1),
    ],
}
