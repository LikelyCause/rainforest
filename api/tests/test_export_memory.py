"""Export memory bounds: per-transformer float32 output, byte-budgeted chunks and fit sample."""
import json

import numpy as np
import polars as pl
import pytest

from app import profiling
from app.config import get_settings
from app.semantics import suggest
from app.transform import (
    FieldSpec, build_preprocessor, coercion_exprs, estimated_width, rows_for_budget, run_export, to_pandas_frame,
    transform_into,
)


@pytest.fixture(scope="module")
def profiles(sample_parquet):
    return profiling.profile_parquet(sample_parquet, ["column_a", "column_b"], 4000, 2000, 50)[1]


def _specs(profiles, family):
    specs, targets = [], []
    for p in profiles:
        s = suggest(p.inferred_type, p.summary, p.is_target, family)
        if s["action"] == "target":
            targets.append(p.name)
        elif s["action"] == "keep":
            specs.append(FieldSpec(p.name, s["semantic_type"], s["imputation"], s.get("impute_value"),
                                   s["add_missing_indicator"], s["transform"], s["scaling"], s["encoding"],
                                   s["params"], p.stats.get("datetime_format")))
    return specs, targets


@pytest.mark.parametrize("family", ["tree", "linear"])
def test_transform_into_matches_column_transformer(sample_parquet, profiles, family):
    specs, _ = _specs(profiles, family)
    df = pl.read_parquet(sample_parquet)
    coercers = coercion_exprs(specs, df.schema)
    frame = to_pandas_frame(df, specs, coercers)
    pre = build_preprocessor(specs).fit(frame)
    expected = pre.transform(frame).astype(np.float32)
    got = transform_into(pre, frame, expected.shape[1])
    assert got.flags.f_contiguous and got.dtype == np.float32
    np.testing.assert_array_equal(got, expected)
    assert estimated_width(specs) >= expected.shape[1]


def test_rows_for_budget():
    assert rows_for_budget(512, 1600) == (512 << 20) // (8 * 1600)
    assert rows_for_budget(0, 10) == 1_000


@pytest.fixture
def budgets(monkeypatch):
    def set_(chunk_mb: int, fit_mb: int):
        monkeypatch.setenv("EXPORT_CHUNK_MB", str(chunk_mb))
        monkeypatch.setenv("EXPORT_FIT_MB", str(fit_mb))
        get_settings.cache_clear()

    yield set_
    monkeypatch.undo()
    get_settings.cache_clear()


def test_chunk_budget_keeps_output_identical(sample_parquet, profiles, tmp_path, budgets):
    specs, targets = _specs(profiles, "tree")
    budgets(256, 512)
    run_export(sample_parquet, specs, targets, tmp_path / "big", {"seed": 1}, lambda *_: None)
    budgets(0, 512)  # the 1,000-row floor
    run_export(sample_parquet, specs, targets, tmp_path / "small", {"seed": 1}, lambda *_: None)
    big, small = (json.loads((tmp_path / d / "manifest.json").read_text()) for d in ("big", "small"))
    assert big["chunk_rows"] >= 4_000 and small["chunk_rows"] == 1_000  # one chunk vs four
    a, b = (pl.read_parquet(tmp_path / d / "model_ready.parquet") for d in ("big", "small"))
    assert a.equals(b)
    assert big["nan_features"] == small["nan_features"]


def test_fit_budget_caps_the_fit_sample(sample_parquet, profiles, tmp_path, budgets):
    specs, targets = _specs(profiles, "tree")
    budgets(256, 0)
    manifest = run_export(sample_parquet, specs, targets, tmp_path, {"seed": 1}, lambda *_: None)
    assert manifest["fit_sample_rows"] == 1_000
