import numpy as np
import polars as pl
import pytest

from app.parquet_rows import read_rows


@pytest.fixture(scope="module")
def grouped(sample_parquet, tmp_path_factory):
    path = tmp_path_factory.mktemp("rows") / "grouped.parquet"
    pl.read_parquet(sample_parquet).write_parquet(path, row_group_size=700)
    return path


@pytest.mark.parametrize("which", ["spread", "one_group", "edges", "empty"])
def test_read_rows_matches_a_full_read(grouped, which):
    full = pl.read_parquet(grouped)
    idx = {
        "spread": np.sort(np.random.default_rng(0).choice(full.height, 300, replace=False)),
        "one_group": np.arange(710, 760),
        "edges": np.array([0, 699, 700, 1399, full.height - 1]),
        "empty": np.array([], dtype=np.int64),
    }[which]
    got = read_rows(grouped, full.columns, idx)
    assert got.schema == full.schema
    assert got.equals(full[idx])


def test_read_rows_column_subset_and_no_columns(grouped):
    got = read_rows(grouped, ["column_b", "column_a"], np.array([3, 5]))
    assert got.columns == ["column_b", "column_a"] and got.height == 2
    assert read_rows(grouped, [], np.array([1])).width == 0
