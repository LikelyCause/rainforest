from pathlib import Path

import pytest

from app.sample_data import generate


@pytest.fixture(scope="session")
def sample_parquet(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("data") / "sample.parquet"
    generate(n_rows=4000, n_fields=60, seed=7, target_kind="mixed").write_parquet(path)
    return path


def set_phase(dataset_id: int, phase: str) -> dict:
    """Put a dataset's current recipe straight into `phase`, skipping the team process, for tests about what
    happens to team decisions and exports there (a finalized version is followed by a new one in `phase`)."""
    from app import recipes
    from app.db import connection

    with connection() as conn:
        r = recipes.current(conn, dataset_id, lock="update")
        if r["phase"] == phase:
            return r
        if r["phase"] == "finalized":
            return recipes.start_version(conn, dataset_id, None, r["family"], phase)
        if phase == "finalized":
            return recipes.finalize(conn, r, r["created_by"])
        return conn.execute("UPDATE meta.recipes SET phase = %s WHERE id = %s RETURNING *", (phase, r["id"])).fetchone()


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
