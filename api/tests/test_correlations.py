import numpy as np
from scipy.stats import spearmanr

from app.correlations import compute_associations, correlation_ratio, cramers_v, spearman_matrix


def test_spearman_matches_scipy_pairwise_complete():
    rng = np.random.default_rng(1)
    a = rng.normal(size=2000)
    b = a * 0.5 + rng.normal(size=2000)
    c = -np.exp(a) + rng.normal(scale=0.1, size=2000)
    X = np.column_stack([a, b, c])
    X[rng.random(2000) < 0.1, 1] = np.nan
    rho, n = spearman_matrix(X)
    mask = ~np.isnan(X[:, 1])
    expected_ab = spearmanr(a[mask], b[mask]).statistic
    assert abs(rho[0, 1] - expected_ab) < 1e-9
    assert rho[0, 2] < -0.95
    assert n[0, 1] == mask.sum()


def test_cramers_v_bounds():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 4, 5000)
    assert cramers_v(a, a) > 0.99
    assert cramers_v(a, rng.integers(0, 4, 5000)) < 0.05


def test_correlation_ratio():
    rng = np.random.default_rng(3)
    codes = rng.integers(0, 3, 5000)
    x = codes * 10 + rng.normal(size=5000)
    eta, n = correlation_ratio(x.astype(float), codes)
    assert eta > 0.95 and n == 5000


def test_compute_associations_covers_all_pair_kinds():
    rng = np.random.default_rng(4)
    num = [(0, rng.normal(size=500)), (1, rng.normal(size=500))]
    cat = [(2, rng.integers(0, 3, 500)), (3, rng.integers(0, 3, 500))]
    methods = {a.method for a in compute_associations(num, cat)}
    assert methods == {"spearman", "cramers_v", "correlation_ratio"}
