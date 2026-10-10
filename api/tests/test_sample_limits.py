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
