import pytest
from fastapi import HTTPException

from app.config import get_settings
from app.routers.datasets import create_sample, sample_limits
from app.schemas import SampleRequest

USER = {"id": 1, "name": "Admin", "email": None, "role": "lead"}


def test_sample_over_the_cell_budget_is_refused():
    assert sample_limits() == {"max_cells": 60_000_000}
    with pytest.raises(HTTPException) as e:
        create_sample(SampleRequest(rows=2_000_000, fields=400), USER)
    assert e.value.status_code == 422
    assert "at most 60,000,000" in e.value.detail and "use up to 150,000 rows" in e.value.detail


def test_cell_budget_is_configurable(monkeypatch):
    monkeypatch.setenv("SAMPLE_MAX_CELLS", "100000")
    get_settings.cache_clear()
    try:
        assert sample_limits() == {"max_cells": 100_000}
        with pytest.raises(HTTPException):
            create_sample(SampleRequest(rows=3_000, fields=40), USER)
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


def test_numeric_target_adds_column_c_and_changes_nothing_else():
    import numpy as np
    import polars as pl

    from app.sample_data import generate
    a = generate(2000, 40, seed=5)
    b = generate(2000, 40, seed=5, numeric_target=True)
    assert b.columns[-1] == "column_c" and b.drop("column_c").equals(a)
    assert b["column_c"].dtype == pl.Float64 and b["column_c"].n_unique() > 1000
    assert abs(np.corrcoef(b["column_c"].to_numpy(), b["credit_score"].to_numpy())[0, 1]) > 0.2


def test_the_numeric_target_counts_against_the_cell_budget(monkeypatch):
    monkeypatch.setenv("SAMPLE_MAX_CELLS", "120000")
    get_settings.cache_clear()
    try:
        with pytest.raises(HTTPException):
            create_sample(SampleRequest(rows=3_000, fields=40, numeric_target=True), USER)
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()
