import numpy as np
import polars as pl
import pytest

from app.profiling import detect_datetime_format, infer_type, profile_column, profile_parquet


def _infer(values, name="col"):
    s = pl.Series(name, values)
    nn = s.drop_nulls()
    return infer_type(s, name, int(nn.n_unique()), nn.len())[0]


@pytest.mark.parametrize(
    "values,expected",
    [
        (list(range(5000)), "identifier"),
        ([1.5, 2.25, 3.125] * 400 + [float(i) / 7 for i in range(800)], "numeric_continuous"),
        ([i % 5 for i in range(2000)], "numeric_discrete"),
        (["a", "b", "c", "d"] * 500, "categorical"),
        (["yes", "no"] * 500, "binary"),
        ([True, False, None] * 100, "binary"),
        (["2024-01-%02d" % (1 + i % 28) for i in range(1000)], "datetime"),
        (["the quick brown fox jumps over the lazy dog %d" % i for i in range(1000)], "text"),
        (["x"] * 100, "constant"),
        ([None] * 100, "constant"),
        (["%d.50" % i for i in range(3000)], "numeric_continuous"),
    ],
)
def test_infer_type(values, expected):
    assert _infer(values) == expected


def test_detects_us_date_format():
    fmt, rate = detect_datetime_format(pl.Series(["01/31/2024", "12/01/2023", "07/04/2020"]))
    assert fmt == "%m/%d/%Y" and rate == 1.0


def test_numeric_profile_has_exact_counts_and_charts():
    rng = np.random.default_rng(0)
    x = rng.lognormal(3, 1, 10_000)
    x[:500] = np.nan
    p = profile_column(pl.Series("income", x), 0, False, None, None, 50)
    assert p.inferred_type == "numeric_continuous"
    assert p.summary["null_count"] == 500
    assert p.summary["skewness"] > 2
    assert p.charts["distribution"]["kind"] == "histogram"
    assert "log_distribution" in p.charts
    assert p.corr_numeric is not None and p.corr_numeric.shape == (10_000,)


def test_numeric_target_keeps_its_type_and_text_target_becomes_labels():
    p = profile_column(pl.Series("column_b", [0, 1, 2] * 300), 0, True, None, None, 50)
    assert p.is_target and p.inferred_type == "numeric_discrete"
    s = profile_column(pl.Series("tier", ["gold", "silver", "bronze free text"] * 300), 0, True, None, None, 50)
    assert s.is_target and s.inferred_type in ("categorical", "binary")


def test_profile_sample_dataset(sample_parquet):
    n, profiles = profile_parquet(sample_parquet, ["column_a", "column_b"], 2000, 2000, 50)
    assert n == 4000
    by = {p.name: p for p in profiles}
    assert by["customer_id"].inferred_type == "identifier"
    assert by["account_uuid"].inferred_type == "identifier"
    assert by["schema_version"].inferred_type == "constant"
    assert by["customer_review"].inferred_type == "text"
    assert by["birth_date_us"].inferred_type == "datetime"
    assert by["column_a"].is_target and by["column_b"].is_target
