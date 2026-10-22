import numpy as np
import pytest

from app.modeling.registry import MODELS, ModelSpec, importances, regression_problem, regressor_params


@pytest.mark.parametrize("key", [k for k, m in MODELS.items() if m.available])
def test_every_model_builds_a_regressor_that_fits(key):
    spec = MODELS[key]
    est = spec.build_regressor({**spec.regressor_defaults, **regressor_params(spec, spec.default_params)}, 0)
    X = np.random.default_rng(0).normal(size=(80, 3))
    est.fit(X, X[:, 0] * 2 + 1)
    assert est.predict(X).shape == (80,)


def test_classifier_only_parameters_are_left_out():
    assert "C" not in regressor_params(MODELS["logistic_regression"], {"C": 2.0, "max_iter": 50})
    reg = regressor_params(
        MODELS["lightgbm"],
        {
            "class_weight": "balanced",
            "is_unbalance": True,
            "objective": "huber",
            "application": "binary",
            "min_data_in_leaf": 5,
        },
    )
    assert not reg.keys() & {"class_weight", "is_unbalance", "application"}
    assert reg["objective"] == "huber" and reg["min_data_in_leaf"] == 5
    if MODELS["xgboost"].available:
        assert regressor_params(MODELS["xgboost"], {"scale_pos_weight": 3, "max_depth": 4}) == {"max_depth": 4}


def test_a_model_without_a_regressor_refuses_a_number():
    spec = ModelSpec("x", "X", "tree", True, lambda p, s: None)
    assert "number" in regression_problem(spec, ["binary", "regression"])
    assert regression_problem(spec, ["binary"]) is None


def test_ridge_importances_are_per_feature():
    est = MODELS["logistic_regression"].build_regressor({}, 0).fit(np.eye(4), [1.0, 2.0, 3.0, 4.0])
    assert importances(MODELS["logistic_regression"], est, list("abcd")).shape == (4,)


def test_xgboost_classifier_takes_labels_with_gaps():
    pytest.importorskip("xgboost")
    from sklearn.multioutput import ClassifierChain

    est = MODELS["xgboost"].build({"n_estimators": 5}, 0)
    X = np.random.default_rng(0).normal(size=(200, 3))
    y = np.r_[np.zeros(100), [1], np.full(99, 2)].astype(int)
    est.fit(X[y != 1], y[y != 1])
    assert est.classes_.tolist() == [0, 2] and set(est.predict(X)) <= {0, 2}
    assert est.predict_proba(X).shape == (200, 2)
    chain = ClassifierChain(MODELS["xgboost"].build({"n_estimators": 5}, 0), cv=5, chain_method="predict")
    chain.fit(X, np.column_stack([y, (X[:, 0] > 0).astype(int)]))
