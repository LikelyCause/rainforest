from pathlib import Path

import pytest

from app.sample_data import generate


@pytest.fixture(scope="session")
def sample_parquet(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("data") / "sample.parquet"
    generate(n_rows=4000, n_fields=60, seed=7, target_kind="mixed").write_parquet(path)
    return path


@pytest.fixture(scope="session")
def login_as():
    """Pick-mode sign-in for API tests: `login_as(client, role="lead", name=None)` signs the TestClient in as an
    active user with that role (the built-in Admin for a lead, else one created on first use) and returns it."""
    from app.config import get_settings

    mp = pytest.MonkeyPatch()
    mp.setenv("RAINFOREST_AUTH_MODE", "pick")
    get_settings.cache_clear()

    def login(client, role: str = "lead", name: str | None = None):
        from app import users

        name = name or ("Admin" if role == "lead" else f"Test {role}")
        user = next((u for u in users.list_all() if u["name"] == name and u["role"] == role and u["active"]), None)
        user = user or users.create(name, None, role)
        r = client.post("/api/auth/pick", json={"user_id": user["id"]})
        assert r.status_code == 200, r.text
        client.user = r.json()
        return client

    yield login
    mp.undo()
    get_settings.cache_clear()
