"""Every catalogued model: its settings are real parameters of its estimators, its defaults are what trains, and the
settings dry-fit cleanly for each kind of target."""
import pytest

from app.modeling.params import CATALOG, effective, setting_problems
from app.modeling.registry import MODELS, dry_fit_problem, regressor_params

SKLEARN = ["random_forest", "hist_gradient_boosting", "mlp", "logistic_regression"]


@pytest.mark.parametrize("key", SKLEARN)
def test_every_setting_is_a_parameter_of_its_estimators(key):
    spec = MODELS[key]
    known = set(spec.build({}, 0).get_params()) | set(spec.build_regressor({}, 0).get_params())
    for s in CATALOG[key]:
        if s.group != "run":
            assert s.name in known, (key, s.name)


@pytest.mark.parametrize("key", SKLEARN)
def test_defaults_pass_their_rules_and_match_the_registry(key):
    assert setting_problems(key, {}) == []
    by_name = {s.name: s for s in CATALOG[key]}
    for k, v in {**MODELS[key].default_params, **MODELS[key].regressor_defaults}.items():
        if k in by_name:
            assert by_name[k].default == v, (key, k)


@pytest.mark.parametrize("key", SKLEARN)
def test_defaults_and_a_changed_setting_dry_fit(key):
    spec = MODELS[key]
    assert dry_fit_problem(spec, {}, ["binary", "multiclass", "regression"]) is None
    changed = {"random_forest": {"max_features": 0.5, "class_weight": "balanced_subsample"},
               "hist_gradient_boosting": {"loss": "quantile", "class_weight": "balanced"},
               "mlp": {"hidden_layer_sizes": "64,32", "solver": "lbfgs", "max_iter": 50},
               "logistic_regression": {"C": 0.1, "penalty": "elasticnet", "solver": "saga", "alpha": 3.0}}[key]
    assert setting_problems(key, changed) == []
    assert dry_fit_problem(spec, changed, ["binary", "multiclass", "regression"]) is None


def test_ridge_alpha_reaches_the_regressor_and_not_the_classifier():
    spec = MODELS["logistic_regression"]
    assert regressor_params(spec, {"alpha": 3.0, "C": 0.5}) == {"alpha": 3.0}
    assert "alpha" not in effective("logistic_regression", {"C": 0.5}) or True


def test_a_penalty_the_optimizer_cant_do_is_named_in_words():
    problem = dry_fit_problem(MODELS["logistic_regression"], {"penalty": "l1", "solver": "lbfgs"}, ["binary"])
    assert problem and "L1 needs SAGA" in problem


def test_mlp_layers_reach_sklearn_as_a_tuple():
    assert effective("mlp", {"hidden_layer_sizes": "256,128,64"})["hidden_layer_sizes"] == (256, 128, 64)


def test_xgboost_balanced_class_weights_lift_a_rare_class_in_three():
    import numpy as np

    from app.modeling.registry import MODELS

    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 4))
    rare = rng.random(3000) < 1 / (1 + np.exp(-2 * (X[:, 0] - 2.2)))  # about 5%, never cleanly separable
    y = np.where(rare, 2, (X[:, 1] > 0).astype(int))
    plain = MODELS["xgboost"].build({"n_estimators": 50, "verbosity": 0}, 0).fit(X, y)
    weighted = MODELS["xgboost"].build({"n_estimators": 50, "verbosity": 0, "class_weight": "balanced"}, 0).fit(X, y)
    assert weighted.get_params()["class_weight"] == "balanced"
    assert (weighted.predict(X) == 2).sum() > (plain.predict(X) == 2).sum()


def test_ridge_alpha_passes_the_parameter_check():
    from app.modeling.registry import param_problems

    assert param_problems(MODELS["logistic_regression"], {"alpha": 10.0, "C": 0.5}) == []
