import lightgbm
import pytest

from app.modeling.params import CATALOG, objective_problem, route, setting_problems, settings_for
from app.modeling.registry import MODELS, _lightgbm_native_params

LGB = settings_for("lightgbm")


def test_every_lightgbm_setting_is_a_real_parameter():
    known = set(lightgbm.LGBMClassifier().get_params()) | set(lightgbm.LGBMRegressor().get_params())
    known |= _lightgbm_native_params()
    for s in LGB:
        if s.group != "run":
            assert s.name in known, s.name


def test_defaults_pass_their_own_rules_and_match_the_registry():
    defaults = {s.name: s.default for s in LGB if s.group != "run"}
    assert setting_problems("lightgbm", {}) == []
    for k, v in MODELS["lightgbm"].default_params.items():
        if k in defaults:
            assert defaults[k] == v, k
    for s in LGB:
        if s.group != "run" and s.when is None:
            assert setting_problems("lightgbm", {s.name: s.default}) == [], s.name


def test_left_out_parameters_are_absent():
    names = {s.name for s in LGB}
    assert not names & {"n_jobs", "verbose", "random_state", "importance_type"}


@pytest.mark.parametrize("params,needle", [
    ({"num_leaves": 1}, "num_leaves"), ({"num_leaves": 2.5}, "whole number"), ({"learning_rate": 5}, "learning_rate"),
    ({"boosting_type": "nope"}, "boosting_type"), ({"extra_trees": "yes"}, "extra_trees"), ({"bogus": 1}, "bogus"),
    ({"subsample": 0}, "subsample"),
])
def test_bad_values_are_named(params, needle):
    problems = setting_problems("lightgbm", params)
    assert problems and needle in problems[0]


def test_when_hidden_setting_is_refused():
    assert setting_problems("lightgbm", {"boosting_type": "dart", "drop_rate": 0.2}) == []
    assert "dart" in setting_problems("lightgbm", {"drop_rate": 0.2})[0]


def test_int_or_none_uses_its_none_value():
    depth = next(s for s in LGB if s.name == "max_depth")
    assert depth.kind == "int_or_none" and depth.none_value == -1 and depth.default == -1
    assert setting_problems("lightgbm", {"max_depth": -1}) == [] and setting_problems("lightgbm", {"max_depth": 0})


def test_routing_by_applies():
    clf, reg = route("lightgbm", {"num_leaves": 127, "class_weight": "balanced", "objective": "huber", "alpha": 2.0})
    assert clf == {"num_leaves": 127, "class_weight": "balanced"}
    assert reg == {"num_leaves": 127, "objective": "huber", "alpha": 2.0}


def test_objectives_that_need_non_negative_numbers():
    assert objective_problem({"objective": "poisson"}, -1.0) and objective_problem({"objective": "tweedie"}, -0.1)
    assert objective_problem({"objective": "gamma"}, 0.0)
    assert objective_problem({"objective": "poisson"}, 0.0) is None and objective_problem({}, -5.0) is None


def test_quantile_alpha_must_be_below_one():
    assert setting_problems("lightgbm", {"objective": "quantile", "alpha": 1.5})
    assert setting_problems("lightgbm", {"objective": "huber", "alpha": 1.5}) == []


def test_catalog_serializes():
    import json
    json.dumps([s.to_json() for s in LGB])
    assert settings_for("mlp") is None and set(CATALOG) == {"lightgbm"}


import time  # noqa: E402

from app.modeling.registry import catalog, dry_fit_problem, regressor_params  # noqa: E402


def test_models_list_carries_settings():
    rows = {m["key"]: m for m in catalog()}
    assert rows["lightgbm"]["settings"][0]["name"] == "seed" and rows["mlp"]["settings"] is None
    assert rows["knn"]["settings"] is None


def test_dry_fit_catches_lightgbm_rules():
    spec = MODELS["lightgbm"]
    rf = dry_fit_problem(spec, {"boosting_type": "rf", "subsample": 1.0, "colsample_bytree": 1.0}, ["binary"])
    assert rf and "Random forest needs" in rf
    assert "at most 1" in dry_fit_problem(spec, {"data_sample_strategy": "goss", "top_rate": 0.8, "other_rate": 0.5},
                                          ["binary"])
    assert "not both" in dry_fit_problem(spec, {"is_unbalance": True, "scale_pos_weight": 2.0}, ["binary"])
    assert dry_fit_problem(spec, {"num_leaves": 31}, ["binary", "regression"]) is None
    assert dry_fit_problem(spec, {"objective": "huber", "alpha": 2.0}, ["multiclass", "regression"]) is None


def test_dry_fit_is_quick_with_many_trees():
    t = time.monotonic()
    assert dry_fit_problem(MODELS["lightgbm"], {"n_estimators": 10_000}, ["multiclass", "regression"]) is None
    assert time.monotonic() - t < 3


def test_regressor_gets_routed_settings():
    assert regressor_params(MODELS["lightgbm"], {"class_weight": "balanced", "objective": "huber"}) == {"objective": "huber"}


def test_goss_works_with_the_default_bagging_settings():
    from app.modeling.params import effective
    assert dry_fit_problem(MODELS["lightgbm"], {"data_sample_strategy": "goss"}, ["binary", "regression"]) is None
    assert effective("lightgbm", {"data_sample_strategy": "goss"})["subsample_freq"] == 0
    assert "bagging" in setting_problems("lightgbm", {"data_sample_strategy": "goss", "subsample": 0.5})[0]


def test_whole_numbers_sent_as_floats_become_ints():
    from app.modeling.params import normalize
    out = normalize("lightgbm", {"n_estimators": 600.0, "max_depth": -1.0, "learning_rate": 0.1})
    assert out == {"n_estimators": 600, "max_depth": -1, "learning_rate": 0.1}
    assert type(out["n_estimators"]) is int


def test_dry_fit_tries_a_binary_link_in_a_mixed_chain():
    assert dry_fit_problem(MODELS["lightgbm"], {"is_unbalance": True, "scale_pos_weight": 2.0}, ["binary", "multiclass"])
