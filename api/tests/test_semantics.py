import pytest

from app.semantics import FAMILIES, NUMERIC, SEMANTIC_TYPES, allowed, defaults_for_type, suggest, validate_decision

STATS = {"null_pct": 0.2, "skewness": 3.1, "outlier_pct": 0.1, "min_value": 0.0, "n_unique": 40}


@pytest.mark.parametrize("family", list(FAMILIES))
def test_defaults_and_suggestions_are_valid_for_every_family(family):
    for t in SEMANTIC_TYPES:
        action = "drop" if t in ("identifier", "constant") else "keep"
        d = {"action": action, "semantic_type": t, **defaults_for_type(t, family)}
        if d["imputation"] == "constant" and t in NUMERIC:
            d["impute_value"] = "0"
        assert validate_decision(d, family) == [], (family, t)
        assert validate_decision(suggest(t, STATS, False, family), family) == [], (family, t)


def test_tree_family_never_scales_or_transforms():
    for t in SEMANTIC_TYPES:
        a = allowed(t, "tree")
        assert a["scaling"] == ["none"] and a["transform"] == ["none"]
    s = suggest("numeric_continuous", STATS, False, "tree")
    assert (s["imputation"], s["transform"], s["scaling"]) == ("none", "none", "none")
    bad = {**s, "scaling": "standard"}
    assert validate_decision(bad, "tree")


def test_linear_family_requires_imputation_and_onehot():
    assert "none" not in allowed("numeric_continuous", "linear")["imputation"]
    assert "ordinal" not in allowed("categorical", "linear")["encoding"]
    s = suggest("numeric_continuous", STATS, False, "linear")
    assert s["imputation"] == "median" and s["add_missing_indicator"] and s["transform"] == "log1p"
    assert s["scaling"] == "standard"


def test_action_does_not_depend_on_family():
    for t in SEMANTIC_TYPES:
        actions = {suggest(t, STATS, False, f)["action"] for f in FAMILIES}
        assert len(actions) == 1, t
    assert {suggest("numeric_continuous", {"null_pct": 0.99}, False, f)["action"] for f in FAMILIES} == {"drop"}
    assert suggest("categorical", {"n_unique": 5}, True, "tree")["action"] == "target"


def test_identifier_cannot_be_kept():
    d = {"action": "keep", "semantic_type": "identifier", **defaults_for_type("identifier")}
    assert validate_decision(d)
