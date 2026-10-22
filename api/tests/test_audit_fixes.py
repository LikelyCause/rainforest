"""Fixes from the 2026-10 code audit that need no database."""

import numpy as np
import pytest


def test_a_non_numeric_quantile_alpha_is_a_problem_not_a_crash():
    from app.modeling.params import setting_problems

    for bad in ("x", None):
        problems = setting_problems("lightgbm", {"objective": "quantile", "alpha": bad})
        assert problems and "alpha" in problems[0]


def test_adopt_with_no_settings_rows_falls_back_to_the_suggestion():
    from app.assessment import adopt
    from app.semantics import SETTING_KEYS

    rows = [{"action": "keep", "semantic_type": "categorical", "target_task": None, "settings": {}}]
    plan = adopt(rows, lambda t, fam: {k: None for k in SETTING_KEYS} | {"encoding": "ordinal"})
    assert plan is not None and plan["settings"] and all(s["encoding"] == "ordinal" for s in plan["settings"].values())


@pytest.mark.parametrize("name", ["../../session_secret", "..", "a/b.webp", "%2e%2e"])
def test_avatar_names_cant_leave_the_avatar_folder(name):
    from fastapi import HTTPException

    from app.auth import avatar

    with pytest.raises(HTTPException) as e:
        avatar("1", name)
    assert e.value.status_code == 404


def test_target_chain_predicts_each_link_once_with_the_same_chain_features():
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    from app.modeling.train import TargetChain

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    a = (X[:, 0] > 0).astype(float)
    m = (X[:, 1] > 0).astype(float) + (X[:, 2] > 0.5)
    g = 3 * a + X[:, 1] + rng.normal(scale=0.1, size=300)
    chain = TargetChain(
        RandomForestClassifier(n_estimators=10, random_state=0),
        RandomForestRegressor(n_estimators=10, random_state=0),
        ["binary", "multiclass", "regression"],
        [0, 1, 2],
        cv=3,
        handles_nan=True,
    ).fit(X, np.column_stack([a, m, g]))
    calls = {"n": 0}
    for est in chain.estimators_:
        for name in ("predict", "predict_proba"):
            if hasattr(est, name):
                orig = getattr(est, name)

                def counted(Z, orig=orig):
                    calls["n"] += 1
                    return orig(Z)

                setattr(est, name, counted)
    outs = chain.predict_all(X[:50], [2, 3, 0])
    assert calls["n"] == 3
    # The chain features match what fit fed each link: compare with a manual walk.
    feats = np.zeros((50, 3))
    for pos, est in enumerate(chain.estimators_):
        t = chain.order_[pos]
        o = TargetChain._output(est, np.hstack((X[:50], feats[:, :pos])), chain.kinds[t])
        feats[:, pos] = chain._scaled(t, o) if chain.kinds[t] == "regression" else o
    assert np.allclose(outs[2], chain.estimators_[2].predict(np.hstack((X[:50], feats[:, :2]))))
