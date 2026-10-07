from pathlib import Path

import pytest

from app.sample_data import generate


@pytest.fixture(scope="session")
def sample_parquet(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("data") / "sample.parquet"
    generate(n_rows=4000, n_fields=60, seed=7, target_kind="mixed").write_parquet(path)
    return path
