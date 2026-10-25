"""start.py's database check: create the database when managed, insist on it existing when not."""

import psycopg
import pytest
import start

URI = "postgresql://u:p@db:5432/tracking"


class FakeConn:
    def __init__(self, exists: bool):
        self.exists = exists
        self.statements: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        self.statements.append(query if isinstance(query, str) else query.as_string())
        return self

    def fetchone(self):
        return (1,) if self.exists else None


@pytest.fixture
def conn(monkeypatch):
    c = FakeConn(exists=False)
    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: c)
    return c


def test_unmanaged_missing_database_exits_naming_the_switch(conn):
    with pytest.raises(SystemExit, match=r"tracking.*MLFLOW_MANAGED_DATABASE"):
        start.ensure_database(URI, managed=False)
    assert not any("CREATE DATABASE" in s for s in conn.statements)


def test_managed_missing_database_is_created(conn):
    start.ensure_database(URI, managed=True)
    assert any(s.startswith("CREATE DATABASE") and "tracking" in s for s in conn.statements)


def test_existing_database_is_left_alone(conn):
    conn.exists = True
    start.ensure_database(URI, managed=False)
    assert not any("CREATE DATABASE" in s for s in conn.statements)
