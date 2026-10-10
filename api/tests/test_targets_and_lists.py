"""Target-derived state, dataset list summaries, export type counts and delete paths (no database)."""
import numpy as np
import polars as pl

from app.correlations import target_associations
from app.processing import LEAKAGE_THRESHOLD, target_suggestion
from app.profiling import _codes, corr_vector
from app.routers.datasets import _run_dirs, _targets, headline_metric
from app.routers.exports import features_by_type
from app.routers.fields import target_order
from app.schemas import DatasetPatch

STATS = {"null_pct": 0.0, "skewness": 0.1, "outlier_pct": 0.0, "n_unique": 40}


def test_leak_flag_follows_target_association():
    by_family, leak = target_suggestion("numeric_continuous", STATS, False, LEAKAGE_THRESHOLD + 0.01)
    assert leak and all(s["action"] == "drop" for s in by_family.values())
    assert "likely target leakage" in by_family["tree"]["reasons"][0]
    by_family, leak = target_suggestion("numeric_continuous", STATS, False, 0.5)
    assert not leak and by_family["tree"]["action"] == "keep"
    assert target_suggestion("numeric_continuous", STATS, False, None)[1] is False


def test_targets_are_never_leaks():
    by_family, leak = target_suggestion("binary", STATS, True, 1.0)
    assert not leak and all(s["action"] == "target" for s in by_family.values())


def test_suggestion_is_json_comparable():
    a, _ = target_suggestion("categorical", STATS, False, 0.99)
    b, _ = target_suggestion("categorical", STATS, False, 0.99)
    assert a == b and isinstance(a["tree"]["reasons"], list)


def test_target_order_keeps_preference_then_appends():
    assert target_order(["b", "gone", "a"], ["a", "b", "c"]) == ["b", "a", "c"]
    assert target_order([], ["a"]) == ["a"]


def test_upload_targets_default_to_none():
    assert _targets(None) == []
    assert _targets(" ") == []
    assert _targets("a, b,a,,") == ["a", "b"]


def test_patch_accepts_empty_target_list():
    assert DatasetPatch(target_columns=[]).target_columns == []
    assert DatasetPatch().target_columns is None


def test_headline_metric_prefers_binary_auc_for_first_chain_target():
    test = {"b": {"roc_auc": 0.7}, "a": {"roc_auc_ovr_macro": 0.8, "accuracy": 0.6}}
    assert headline_metric("done", ["a", "b"], test) == {"metric": "roc_auc_ovr_macro", "value": 0.8}
    assert headline_metric("done", ["b", "a"], test) == {"metric": "roc_auc", "value": 0.7}
    assert headline_metric("done", ["a"], {"a": {"accuracy": 1}}) == {"metric": "accuracy", "value": 1.0}
    assert headline_metric("running", ["a"], test) == {"metric": None, "value": None}
    assert headline_metric("done", None, None) == {"metric": None, "value": None}


def test_features_by_type_counts_fields_and_outputs():
    manifest = {
        "fields": [{"name": "n", "semantic_type": "numeric_continuous"},
                   {"name": "c", "semantic_type": "categorical"},
                   {"name": "id", "semantic_type": "identifier"}],
        "feature_source_field": {"n": "n", "n__isna": "n", "c_A": "c", "c_B": "c", "c_other": "c"},
    }
    assert features_by_type(manifest) == {
        "numeric_continuous": {"fields": 1, "features": 2},
        "categorical": {"fields": 1, "features": 3},
        "identifier": {"fields": 1, "features": 0},
    }


def test_run_dirs_include_unrecorded_and_stay_inside_models_dir(tmp_path):
    models = tmp_path / "models"
    for d in ("run3_lightgbm_1", "run3_mlp_2", "run30_lightgbm_1", "export1_x_1"):
        (models / d).mkdir(parents=True)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    runs = [{"id": 3, "model_dir": str(models / "export1_x_1")}, {"id": 4, "model_dir": str(outside)}]
    assert {p.name for p in _run_dirs(models, runs)} == {"run3_lightgbm_1", "run3_mlp_2", "export1_x_1"}


def test_corr_vector_matches_type():
    s = pl.Series("x", ["1", "2", None, "4"])
    num, codes = corr_vector(s, "numeric_continuous", None, None, 50)
    assert codes is None and np.isnan(num[2]) and num[3] == 4.0
    num, codes = corr_vector(pl.Series("t", ["a", "b", "a", None]), "categorical", None, np.array([0, 2, 3]), 50)
    assert num is None and codes.tolist() == _codes(pl.Series("t", ["a", "a", None]), 50).tolist()
    assert corr_vector(pl.Series("t", ["free text"]), "text", None, None, 50) == (None, None)


def test_target_associations_pair_new_targets_with_everything():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    x_leak = y + rng.normal(0, 0.01, 500)
    noise = rng.normal(0, 1, 500)
    cat = (y * 2 + rng.integers(0, 2, 500)).astype(np.int32)
    other_target = rng.integers(0, 3, 500).astype(np.int32)
    assocs = list(target_associations([(0, y.astype(np.int32)), (4, other_target)],
                                      [(1, x_leak), (2, noise)], [(3, cat)]))
    pairs = {(min(a.i, a.j), max(a.i, a.j)): a for a in assocs}
    assert set(pairs) == {(0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (3, 4)}
    assert pairs[(0, 1)].value > 0.95 and pairs[(0, 2)].value < 0.2
    assert pairs[(0, 3)].method == "cramers_v" and pairs[(0, 1)].method == "correlation_ratio"


def test_headline_metric_uses_r2_for_a_regression_first_target():
    from app.routers.datasets import headline_metric
    assert headline_metric("done", ["price"], {"price": {"rmse": 3.0, "r2": 0.7}}) == {"metric": "r2", "value": 0.7}
