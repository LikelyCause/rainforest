import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import transforms
from app.modeling.params import Setting
from app.semantics import DEFAULT_PARAMS, PARAM_RULES, allowed, option_catalog, suggest, validate_decision
from app.transforms import registry
from app.transforms.builtin import ENCODING_SETTINGS, TARGET, YEO_JOHNSON


def _spec(key="t", **kw):
    base = dict(
        dimension="transform",
        types=("numeric_continuous",),
        families=("linear",),
        label=key.capitalize(),
        help="",
        build=lambda p, c: None,
    )
    return registry.TransformSpec(key=key, **{**base, **kw})


@pytest.fixture
def scratch_registry(monkeypatch):
    monkeypatch.setattr(registry, "_SPECS", dict(registry._SPECS))


def test_yeo_johnson_is_registered_for_linear_and_neural_numbers():
    spec = transforms.get("yeo_johnson")
    assert spec is YEO_JOHNSON and spec.dimension == "transform" and spec.available
    assert set(spec.types) == {"numeric_continuous", "numeric_discrete"}
    assert set(spec.families) == {"linear", "neural"}
    assert [s.key for s in transforms.specs("transform")] == ["yeo_johnson"]
    assert [s.key for s in transforms.specs("encoding")] == ["target"]


def test_yeo_johnson_keeps_missing_values_and_names_its_column():
    from sklearn.preprocessing import PowerTransformer

    X = np.array([[1.0], [2.0], [np.nan], [10.0], [100.0], [-3.0]])
    step = YEO_JOHNSON.build({}, registry.BuildContext())
    out = step.fit(X).transform(X)
    ref = PowerTransformer(method="yeo-johnson", standardize=False).fit(X).transform(X)
    assert np.isnan(out[2, 0]) and np.allclose(np.delete(out, 2), np.delete(ref, 2))
    assert list(step.get_feature_names_out(["age"])) == ["age"]


def test_register_refuses_bad_specs(scratch_registry):
    with pytest.raises(ValueError, match="twice"):
        registry.register(_spec("yeo_johnson"))
    with pytest.raises(ValueError, match="dimension"):
        registry.register(_spec("x1", dimension="scaling"))
    with pytest.raises(ValueError, match="lower-case"):
        registry.register(_spec("Bad-Key"))
    with pytest.raises(ValueError, match="setting name"):
        registry.register(_spec("x2", settings=(Setting("k", "K", "", "field", "int", 1, 0, 9),) * 2))


def test_unavailable_specs_are_registered_but_not_listed(scratch_registry):
    registry.register(_spec("needs_lib", requires="no_such_module_rf"))
    assert registry.get("needs_lib") is not None and not registry.get("needs_lib").available
    assert "needs_lib" not in [s.key for s in registry.specs()]


def test_params_for_and_param_problems():
    spec = _spec(settings=(Setting("k", "K", "", "field", "int", 5, 2, 10, 1),))
    assert registry.params_for(spec, {"k": 7, "max_categories": 3}) == {"k": 7}
    assert registry.params_for(spec, None) == {"k": 5}
    assert registry.param_problems(spec, {"k": 11}) == ["params.k must be from 2 to 10"]
    assert registry.param_problems(spec, {"k": 2.5}) == ["params.k must be a whole number"]
    assert registry.param_problems(spec, {"k": 3, "left_over": "x"}) == []


def test_encoder_settings_match_the_checks_and_defaults():
    assert set(ENCODING_SETTINGS) == set(PARAM_RULES)
    for enc, settings in ENCODING_SETTINGS.items():
        assert {s.name: s.default for s in settings} == DEFAULT_PARAMS[enc], enc
        assert {s.name: (int if s.kind == "int" else float, s.min, s.max) for s in settings} == PARAM_RULES[enc], enc


def test_the_registry_does_not_import_semantics():
    code = "import sys, app.transforms; print('app.semantics' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert out.stdout.strip() == "False"


SKEWED = {"skewness": 3.0, "null_pct": 0.1}


def _numeric(family, **kw):
    return {**suggest("numeric_continuous", SKEWED, False, family), **kw}


def test_yeo_johnson_is_offered_to_linear_and_neural_only():
    for fam in ("linear", "neural"):
        for t in ("numeric_continuous", "numeric_discrete"):
            assert "yeo_johnson" in allowed(t, fam)["transform"], (fam, t)
        assert validate_decision(_numeric(fam, transform="yeo_johnson"), fam) == []
    assert allowed("numeric_continuous", "tree")["transform"] == ["none"]
    assert "yeo_johnson" not in allowed("categorical", "linear")["transform"]
    errors = validate_decision(_numeric("tree", transform="yeo_johnson"), "tree")
    assert errors == ["transform='yeo_johnson' is not valid for numeric_continuous with tree models (allowed: none)"]


def test_an_unavailable_key_is_named_not_replaced(scratch_registry):
    registry.register(_spec("gone", requires="no_such_module_rf"))
    for key in ("gone", "acme.iqr_clip"):
        assert validate_decision(_numeric("linear", transform=key), "linear") == [
            f"transform={key!r} is unavailable: it isn't installed on this server"
        ]


def test_spec_params_and_missing_values_are_checked(scratch_registry):
    registry.register(
        _spec(
            "strict",
            families=("tree", "linear"),
            handles_nan=False,
            settings=(Setting("k", "K", "", "field", "int", 5, 2, 10, 1),),
        )
    )
    tree = _numeric("tree", transform="strict")
    assert tree["imputation"] == "none"
    assert validate_decision(tree, "tree") == ["Strict needs missing values filled; choose an imputation"]
    assert validate_decision({**tree, "imputation": "median"}, "tree") == []
    assert validate_decision({**tree, "imputation": "median", "params": {"k": 50}}, "tree") == [
        "params.k must be from 2 to 10"
    ]


def test_the_catalog_publishes_registry_options():
    cat = option_catalog()
    assert cat["labels"]["transform"]["yeo_johnson"] == "Yeo-Johnson (power transform)"
    assert cat["labels"]["transform"]["log1p"] == "Signed log1p"
    assert [t["key"] for t in cat["transforms"]] == ["yeo_johnson", "target"]
    assert cat["settings"]["yeo_johnson"] == []
    assert [s["name"] for s in cat["settings"]["tfidf_svd"]] == ["svd_components", "max_features", "min_df"]
    assert cat["param_rules"]["onehot"]["max_categories"] == {"integer": True, "min": 2, "max": 1000}
    assert "yeo_johnson" in cat["allowed"]["neural"]["numeric_discrete"]["transform"]
    json.dumps(cat)


def test_target_encoding_is_registered_for_categoricals_in_every_family():
    spec = transforms.get("target")
    assert spec is TARGET and spec.dimension == "encoding" and spec.needs_target and spec.handles_nan
    assert spec.types == ("categorical",) and set(spec.families) == {"tree", "linear", "neural"}
    assert spec.defaults() == {"target": None, "smooth": None, "cv": 5}
    assert [s.kind for s in spec.settings] == ["target", "float_or_none", "int"]
    assert registry.param_problems(spec, {"cv": 11, "smooth": -1}) == [
        "params.cv must be from 2 to 10",
        "params.smooth must be from 0 to 1000 (or Auto)",
    ]


def test_the_encoder_is_built_from_the_context():
    ctx = registry.BuildContext(seed=7, target=registry.TargetInfo("y", "multiclass", ("a", "b", "c")))
    enc = TARGET.build({"target": None, "smooth": None, "cv": 3}, ctx)
    assert (enc.target_type, enc.smooth, enc.cv, enc.shuffle, enc.random_state) == ("multiclass", "auto", 3, True, 7)
    assert TARGET.build({"target": None, "smooth": 12.5, "cv": 5}, ctx).smooth == 12.5


def test_pick_target_feeds_one_column():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"c": rng.choice([f"L{i}" for i in range(40)], 600).astype(object)})
    y = pd.DataFrame({"first": rng.integers(0, 3, 600).astype(float), "second": rng.normal(0, 1, 600)})

    def ctx(kind, classes=()):
        return registry.BuildContext(target=registry.TargetInfo("x", kind, classes))

    classes = ("lo", "mid", "hi")
    multi = registry.PickTarget(
        TARGET.build(TARGET.defaults(), ctx("multiclass", classes)), "first", "multiclass", classes
    )
    out = multi.fit_transform(X, y)
    assert out.shape == (600, 3) and list(multi.get_feature_names_out(["c"])) == ["c_hi", "c_lo", "c_mid"]
    assert multi.transform(X.head(5)).shape == (5, 3)
    num = registry.PickTarget(TARGET.build(TARGET.defaults(), ctx("continuous")), "second", "continuous").fit(X, y)
    assert list(num.get_feature_names_out(["c"])) == ["c"] and num.transformer_.target_type_ == "continuous"
    binary = registry.PickTarget(
        TARGET.build(TARGET.defaults(), ctx("binary", ("no", "yes"))), "first", "binary", ("no", "yes")
    ).fit(X, y.assign(first=(y["first"] > 0).astype(float)))
    assert list(binary.transformer_.classes_) == [0, 1]
    with pytest.raises(ValueError, match="needs the targets"):
        registry.PickTarget(TARGET.build(TARGET.defaults(), ctx("binary")), "first", "binary").fit(X)
