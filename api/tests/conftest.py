from pathlib import Path

import pytest

from app.sample_data import generate


@pytest.fixture(scope="session")
def sample_parquet(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("data") / "sample.parquet"
    generate(n_rows=4000, n_fields=60, seed=7, target_kind="mixed").write_parquet(path)
    return path


ADMIN = {"username": "admin", "password": "test-admin-password"}


@pytest.fixture(scope="session")
def admin_login():
    """Configure the admin sign-in for API tests; returns a function that signs a TestClient in."""
    from app.config import get_settings

    mp = pytest.MonkeyPatch()
    mp.setenv("RAINFOREST_ADMIN_USER", ADMIN["username"])
    mp.setenv("RAINFOREST_ADMIN_PASSWORD", ADMIN["password"])
    get_settings.cache_clear()

    def login(client):
        r = client.post("/api/auth/login", json=ADMIN)
        assert r.status_code == 200, r.text
        return client

    yield login
    mp.undo()
    get_settings.cache_clear()
