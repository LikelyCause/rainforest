"""Associations that the leak screen relies on: pairwise Spearman under MAR, codes for missing values, and the
uncapped screen of high-cardinality and text fields against targets."""
import numpy as np
import polars as pl
from scipy.stats import spearmanr

from app.correlations import cramers_v, cramers_v_sparse, screen_associations, spearman_matrix
from app.processing import LEAKAGE_THRESHOLD
from app.profiling import _codes, profile_column, screen_codes


def test_spearman_ranks_within_the_pairs_rows_under_mar():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    y = x ** 3
    y[np.abs(x) < 1] = np.nan  # missing at random given x
    rho, n = spearman_matrix(np.column_stack([x, y]))
    m = ~np.isnan(y)
    assert rho[0, 1] > 0.999 and n[0, 1] == m.sum()


def test_spearman_matches_scipy_with_ties_and_differing_missingness():
    rng = np.random.default_rng(1)
    n = 2000
    a = rng.normal(0, 1, n)
    X = np.column_stack([a, np.round(a * 2), a + rng.normal(0, 0.5, n), rng.normal(0, 1, n)])
    for k in range(X.shape[1]):
        X[rng.random(n) < 0.25, k] = np.nan
    rho, _ = spearman_matrix(X, refine_at=0.0)
    for i in range(4):
        for j in range(i + 1, 4):
            m = ~np.isnan(X[:, i]) & ~np.isnan(X[:, j])
            assert abs(rho[i, j] - spearmanr(X[m, i], X[m, j]).statistic) < 1e-9


def test_codes_keep_missing_apart_from_other():
    codes = _codes(pl.Series("c", ["a", None, "b", "a", "z"]), 1)
    assert codes.tolist() == [0, 2, 1, 0, 1]


def test_high_cardinality_leak_is_found_by_the_uncapped_screen():
    n = 20_000
    levels = np.arange(n) % 500
    target = (levels % 3).astype(np.int32)
    capped = _codes(pl.Series("c", [f"L{v}" for v in levels]), 50)
    assert cramers_v(capped, target) < 0.5  # what the capped association sees
    field = screen_codes(pl.Series("c", [f"L{v}" for v in levels]), None)
    assert cramers_v_sparse(field, target) > LEAKAGE_THRESHOLD


def test_sparse_cramers_v_matches_the_dense_one():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 12, 4000)
    b = (a % 4 + rng.integers(0, 2, 4000)).astype(np.int64)
    assert abs(cramers_v_sparse(a, b) - cramers_v(a, b)) < 1e-9
    assert cramers_v_sparse(rng.integers(0, 4000, 4000), rng.integers(0, 2, 4000)) < 0.1  # near-unique, no signal


def test_text_that_repeats_the_label_is_screened():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 3000)
    case = rng.integers(0, 50, 3000)
    notes = [f"application {'approved' if v else 'declined'} after a full manual review of the documents, case {k}"
             for v, k in zip(y, case)]
    p = profile_column(pl.Series("notes", notes), 0, False, None, None, 50)
    t = profile_column(pl.Series("y", y.tolist()), 1, True, None, None, 50)
    assert p.inferred_type == "text" and p.screen_codes is not None and t.screen_codes is not None
    (assoc,) = list(screen_associations([(1, t.screen_codes)], [(0, p.screen_codes)]))
    assert assoc.method == "cramers_v" and assoc.value > LEAKAGE_THRESHOLD


def test_only_targets_text_and_high_cardinality_fields_get_screen_codes():
    rng = np.random.default_rng(5)
    small = profile_column(pl.Series("c", rng.choice(list("abc"), 2000).tolist()), 0, False, None, None, 50)
    big = profile_column(pl.Series("c", [f"v{i % 300}" for i in range(3000)]), 0, False, None, None, 50)
    num = profile_column(pl.Series("x", rng.normal(0, 1, 2000)), 0, False, None, None, 50)
    assert small.screen_codes is None and num.screen_codes is None and big.screen_codes is not None
