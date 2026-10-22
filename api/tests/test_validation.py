"""Input validation and JSON safety: decision settings, fill values, run parameters, cv, non-finite metrics."""

import math

import numpy as np
import pytest

from app.db import json_safe
from app.modeling.train import evaluate


def test_json_safe_turns_non_finite_numbers_into_null_and_drops_nul():
    value = {"roc_auc": float("nan"), "x": [float("inf"), 1.5, np.float32(2.5)], "s": "a\x00b", "n": np.int64(3)}
    assert json_safe(value) == {"roc_auc": None, "x": [None, 1.5, 2.5], "s": "ab", "n": 3}


def test_evaluate_on_a_single_class_split_has_no_auc_and_stays_finite():
    y = np.zeros(20, dtype=int)
    proba = np.column_stack([np.full(20, 0.8), np.full(20, 0.2)])
    m, _ = evaluate(y, proba, ["no", "yes"])
    assert "roc_auc" not in m
    assert all(math.isfinite(v) for v in json_safe(m).values() if v is not None)


@pytest.mark.parametrize("fill", ["nan", "inf", "-inf", "NaN", "1e999"])
def test_numeric_fill_values_must_be_finite(fill):
    from app.semantics import validate_decision

    d = {
        "action": "keep",
        "semantic_type": "numeric_continuous",
        "imputation": "constant",
        "impute_value": fill,
        "transform": "none",
        "scaling": "standard",
        "encoding": "none",
        "params": {},
    }
    assert any("finite" in e for e in validate_decision(d, "linear"))


def _keep(t, encoding, params, family="tree"):
    from app.semantics import suggest, validate_decision

    s = suggest(t, {}, False, family)
    return validate_decision({**s, "action": "keep", "encoding": encoding, "params": params}, family)


@pytest.mark.parametrize(
    "t,encoding,params",
    [
        ("categorical", "onehot", {"max_categories": 0}),
        ("categorical", "onehot", {"max_categories": 2.5}),
        ("categorical", "onehot", {"min_frequency": 1.5}),
        ("categorical", "onehot", {"min_frequency": "lots"}),
        ("categorical", "ordinal", {"min_frequency": -0.1}),
        ("text", "tfidf_svd", {"svd_components": 0}),
        ("text", "tfidf_svd", {"max_features": True}),
        ("text", "tfidf_svd", {"min_df": float("nan")}),
    ],
)
def test_encoder_params_out_of_range_are_rejected(t, encoding, params):
    assert any(f"params.{next(iter(params))}" in e for e in _keep(t, encoding, params))


def test_valid_and_leftover_encoder_params_pass():
    assert _keep("categorical", "onehot", {"max_categories": 12, "min_frequency": 0.02}) == []
    assert _keep("text", "tfidf_svd", {"svd_components": 8, "max_features": 500, "min_df": 2}) == []
    # Left over from a previous encoding choice; frequency encoding reads none of them.
    assert _keep("categorical", "frequency", {"max_categories": 0}) == []


def test_option_catalog_publishes_param_ranges():
    from app.semantics import option_catalog

    rules = option_catalog()["param_rules"]
    assert rules["onehot"]["max_categories"] == {"integer": True, "min": 2, "max": 1000}


def test_run_params_are_checked_against_the_estimator():
    from app.modeling.registry import MODELS, param_problems

    assert param_problems(MODELS["lightgbm"], {"n_estimators": 10, "min_data_in_leaf": 5}) == []
    assert "no parameter 'n_estimaters'" in param_problems(MODELS["lightgbm"], {"n_estimaters": 10})[0]
    assert param_problems(MODELS["random_forest"], {"n_estimators": -3})
    assert param_problems(MODELS["logistic_regression"], {"random_state": 1})
    assert param_problems(MODELS["mlp"], {"hidden_layer_sizes": [32, 16]}) == []


def test_training_refuses_unknown_params_and_one_fold(tmp_path):
    from app.modeling.train import train

    with pytest.raises(ValueError, match="no parameter"):
        train(tmp_path, "lightgbm", params={"n_estimaters": 5}, out_dir=tmp_path)
    with pytest.raises(ValueError, match="cv must be"):
        train(tmp_path, "lightgbm", cv=1, out_dir=tmp_path)


def test_run_request_refuses_one_fold():
    from pydantic import ValidationError

    from app.schemas import RunRequest

    assert RunRequest(cv=0).cv == 0 and RunRequest(cv=2).cv == 2
    with pytest.raises(ValidationError, match="cv must be"):
        RunRequest(cv=1)
