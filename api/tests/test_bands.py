import numpy as np
import pytest

from app.bands import assign, band_labels, edges_for, quantile_edges, thin_bands


def test_quantile_edges_split_evenly():
    assert quantile_edges(np.arange(100, dtype=float), 4) == [24.75, 49.5, 74.25]


def test_quantile_edges_collapse_duplicates():
    v = np.array([0.0] * 90 + list(range(1, 11)), dtype=float)
    edges = quantile_edges(v, 5)
    assert edges == [1.0]
    assert np.unique(assign(v, edges)).size == 2


def test_quantile_edges_of_a_constant_column_are_empty():
    assert quantile_edges(np.zeros(20), 4) == []


def test_quantile_edges_ignore_nan():
    assert quantile_edges(np.array([np.nan, 1, 2, 3, 4]), 2) == [2.5]


def test_labels_and_assignment():
    edges = [10.0, 25.0]
    assert band_labels(edges) == ["< 10", "[10, 25)", "≥ 25"]
    got = assign(np.array([3, 10, 24.9, 25, 99, np.nan]), edges)
    assert got[:5].tolist() == [0, 1, 1, 2, 2] and np.isnan(got[5])


def test_labels_stay_distinct_for_close_edges():
    assert len(set(band_labels([1000000.1, 1000000.2]))) == 3


def test_thin_bands():
    assert thin_bands(np.array([1.0, 2, 3, 50, 60]), [10, 20]) == ["[10, 20)"]


def test_edges_for_uses_given_edges_or_training_quantiles():
    assert edges_for({"edges": [1, 2]}, np.arange(10.0)) == [1.0, 2.0]
    assert edges_for({"k": 2}, np.arange(10.0)) == [4.5]
    with pytest.raises(ValueError, match="one band"):
        edges_for({"k": 4}, np.zeros(50))


def test_equal_sized_cut_points_are_rounded_for_readable_labels():
    v = np.random.default_rng(0).gamma(4.0, 9.0, size=1000)
    edges = quantile_edges(v, 4)
    assert all(len(f"{e:g}".replace(".", "").lstrip("0")) <= 4 for e in edges)
    counts = np.bincount(assign(v, edges).astype(int))
    assert counts.min() >= 240  # still close to equal-sized


def test_rounding_never_moves_a_cut_onto_the_minimum():
    v = np.array([100.0001] * 50 + [100.0002] * 50)
    assert np.unique(assign(v, quantile_edges(v, 2))).size == 2


@pytest.mark.parametrize("cap", [99.99999, 1234.56])
def test_rounding_never_empties_a_band_at_a_repeated_value(cap):
    rng = np.random.default_rng(1)
    v = np.r_[rng.uniform(0, cap, 700), np.full(300, cap)]
    edges = quantile_edges(v, 5)
    assert thin_bands(v, edges) == []
