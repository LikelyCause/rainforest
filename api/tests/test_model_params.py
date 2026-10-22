import lightgbm
import pytest

from app.modeling.params import (
    CATALOG,
    Setting,
    objective_problem,
    route,
    setting_problems,
    settings_for,
    value_problem,
)
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


@pytest.mark.parametrize(
    "params,needle",
    [
        ({"num_leaves": 1}, "num_leaves"),
        ({"num_leaves": 2.5}, "whole number"),
        ({"learning_rate": 5}, "learning_rate"),
        ({"boosting_type": "nope"}, "boosting_type"),
        ({"extra_trees": "yes"}, "extra_trees"),
        ({"bogus": 1}, "bogus"),
        ({"subsample": 0}, "subsample"),
    ],
)
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

    for key in CATALOG:
        json.dumps([s.to_json() for s in CATALOG[key]])
    assert set(CATALOG) == set(MODELS)


import time  # noqa: E402

from app.modeling.registry import catalog, dry_fit_problem, regressor_params  # noqa: E402


def test_models_list_carries_settings():
    rows = {m["key"]: m for m in catalog()}
    assert rows["lightgbm"]["settings"][0]["name"] == "seed" and rows["mlp"]["settings"][0]["name"] == "seed"
    assert rows["knn"]["settings"] is None


def test_dry_fit_catches_lightgbm_rules():
    spec = MODELS["lightgbm"]
    rf = dry_fit_problem(spec, {"boosting_type": "rf", "subsample": 1.0, "colsample_bytree": 1.0}, ["binary"])
    assert rf and "Random forest needs" in rf
    assert "at most 1" in dry_fit_problem(
        spec, {"data_sample_strategy": "goss", "top_rate": 0.8, "other_rate": 0.5}, ["binary"]
    )
    assert "not both" in dry_fit_problem(spec, {"is_unbalance": True, "scale_pos_weight": 2.0}, ["binary"])
    assert dry_fit_problem(spec, {"num_leaves": 31}, ["binary", "regression"]) is None
    assert dry_fit_problem(spec, {"objective": "huber", "alpha": 2.0}, ["multiclass", "regression"]) is None


def test_dry_fit_is_quick_with_many_trees():
    t = time.monotonic()
    assert dry_fit_problem(MODELS["lightgbm"], {"n_estimators": 10_000}, ["multiclass", "regression"]) is None
    assert time.monotonic() - t < 3


def test_regressor_gets_routed_settings():
    reg = regressor_params(MODELS["lightgbm"], {"class_weight": "balanced", "objective": "huber"})
    assert "class_weight" not in reg and reg["objective"] == "huber" and reg["alpha"] == 0.9


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
    assert dry_fit_problem(
        MODELS["lightgbm"], {"is_unbalance": True, "scale_pos_weight": 2.0}, ["binary", "multiclass"]
    )


XGB = settings_for("xgboost")
xgboost = pytest.importorskip("xgboost")


def _active(settings, s):
    """Params that switch on `s`'s dependencies, so the library sees the setting in use."""
    return {k: allowed[0] for k, allowed in (s.when or {}).items()}


def test_every_xgboost_setting_is_one_xgboost_uses():
    import warnings

    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 3))
    for s in XGB:
        if s.group == "run":
            continue
        est = xgboost.XGBClassifier if s.applies == "classes" else xgboost.XGBRegressor
        y = (X[:, 0] > 0).astype(int) if s.applies == "classes" else np.abs(X[:, 0]) + 1
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            est(**{"n_estimators": 2, "verbosity": 1, **_active(XGB, s), s.name: s.default}).fit(X, y)
        assert not [w for w in caught if "not used" in str(w.message)], s.name


def test_xgboost_defaults_are_what_trains():
    """A setting's shown default is the library's own unless the registry sets it, or the setting is filled in when
    it becomes active."""
    import json

    import numpy as np

    X = np.random.default_rng(0).normal(size=(64, 3))
    config = json.loads(
        xgboost.XGBRegressor(n_estimators=1, verbosity=0).fit(X, np.abs(X[:, 0]) + 1).get_booster().save_config()
    )

    def flat(d, out):
        for k, v in d.items():
            flat(v, out) if isinstance(v, dict) else out.setdefault(k, v)
        return out

    values = flat(config, {})
    values["booster"] = config["learner"]["gradient_booster"]["name"]
    values["objective"] = config["learner"]["objective"]["name"]
    registry = MODELS["xgboost"].default_params
    for s in XGB:
        if s.group == "run" or s.when or s.name in registry or s.name in ("n_estimators", "class_weight"):
            continue  # class_weight is the wrapper's (row weights at fit), not a booster parameter
        assert s.name in values, s.name
        shown = str(int(s.default)) if isinstance(s.default, bool) else str(s.default)
        assert values[s.name] == shown or float(values[s.name]) == float(s.default), (s.name, values[s.name])
    for k, v in registry.items():
        if s := next((x for x in XGB if x.name == k), None):
            assert s.default == v, k


def test_xgboost_left_out_parameters_are_absent():
    names = {s.name for s in XGB}
    assert not names & {
        "n_jobs",
        "verbosity",
        "random_state",
        "device",
        "tree_method",
        "base_score",
        "missing",
        "importance_type",
        "eval_metric",
        "early_stopping_rounds",
        "enable_categorical",
        "max_cat_to_onehot",
        "monotone_constraints",
        "interaction_constraints",
        "updater",
    }
    booster = next(s for s in XGB if s.name == "booster")
    assert [o["value"] for o in booster.options] == ["gbtree", "dart"]


def test_xgboost_routing_by_applies():
    clf, reg = route(
        "xgboost", {"max_depth": 4, "scale_pos_weight": 3.0, "objective": "reg:pseudohubererror", "huber_slope": 2.0}
    )
    assert clf == {"max_depth": 4, "scale_pos_weight": 3.0}
    assert reg == {"max_depth": 4, "objective": "reg:pseudohubererror", "huber_slope": 2.0}


def test_xgboost_objectives_that_need_a_minimum():
    assert objective_problem({"objective": "count:poisson"}, -1.0) and objective_problem({"objective": "reg:gamma"}, 0)
    assert objective_problem({"objective": "reg:tweedie"}, -0.1)
    assert "> -1" in objective_problem({"objective": "reg:squaredlogerror"}, -1.0)
    assert objective_problem({"objective": "reg:squaredlogerror"}, -0.5) is None


def test_active_dependent_settings_are_filled_in():
    from app.modeling.params import effective

    assert effective("xgboost", {"objective": "reg:quantileerror"})["quantile_alpha"] == 0.5
    assert "quantile_alpha" not in effective("xgboost", {"objective": "reg:squarederror"})
    assert effective("xgboost", {"booster": "dart"})["rate_drop"] == 0.1
    assert effective("xgboost", {"booster": "dart", "rate_drop": 0.3})["rate_drop"] == 0.3


def test_xgboost_settings_pass_the_parameter_check():
    from app.modeling.registry import param_problems

    assert (
        param_problems(
            MODELS["xgboost"],
            {"objective": "reg:tweedie", "tweedie_variance_power": 1.2, "booster": "dart", "rate_drop": 0.2},
        )
        == []
    )
    assert param_problems(MODELS["xgboost"], {"not_a_setting": 1})


def test_xgboost_regressor_gets_its_loss_settings():
    reg = regressor_params(
        MODELS["xgboost"], {"objective": "reg:pseudohubererror", "huber_slope": 2.0, "scale_pos_weight": 3.0}
    )
    assert reg == {"objective": "reg:pseudohubererror", "huber_slope": 2.0}


def test_xgboost_dry_fit():
    spec = MODELS["xgboost"]
    assert dry_fit_problem(spec, {"objective": "reg:quantileerror"}, ["binary", "regression"]) is None
    assert (
        dry_fit_problem(spec, {"booster": "dart", "multi_strategy": "multi_output_tree"}, ["multiclass", "regression"])
        is None
    )
    assert dry_fit_problem(spec, {"max_depth": 0, "grow_policy": "lossguide", "max_leaves": 0}, ["binary"]) is None


def test_xgboost_bad_values_are_named():
    assert (
        "tweedie_variance_power"
        in setting_problems("xgboost", {"objective": "reg:tweedie", "tweedie_variance_power": 2.0})[0]
    )
    assert "objective" in setting_problems("xgboost", {"huber_slope": 2.0})[0]
    assert setting_problems("xgboost", {"booster": "gblinear"})


def test_float_or_none_and_target_kinds():
    smooth = Setting("smooth", "Smoothing", "", "field", "float_or_none", None, 0, 1000, none_label="Auto")
    assert value_problem(smooth, None) is None and value_problem(smooth, 12.5) is None
    assert value_problem(smooth, 1001) == "smooth must be from 0 to 1000 (or Auto)"
    assert value_problem(smooth, "x") == "smooth must be a number"
    target = Setting("target", "Target", "", "field", "target", None, none_label="First target")
    assert value_problem(target, None) is None and value_problem(target, "readmitted") is None
    assert value_problem(target, "") == value_problem(target, 3) == "target must be a target's name"
    assert smooth.to_json()["none_value"] is None and target.to_json()["kind"] == "target"
