"""RAINFOREST_MANAGED_ROLES=false: the SQL console's role is created by an operator, bootstrap only grants to it.

Needs an empty database: RAINFOREST_TEST_DATABASE_URL=postgresql://user:pw@host:port/db pytest
"""

import os

import psycopg
import pytest
from psycopg import errors, sql
from psycopg.conninfo import make_conninfo

from app import db
from app.config import get_settings

DB_URL = os.environ.get("RAINFOREST_TEST_DATABASE_URL")
ROLE_STATE = "SELECT a.rolpassword, r.rolconfig FROM pg_roles r JOIN pg_authid a ON a.oid = r.oid WHERE r.rolname = %s"


@pytest.fixture
def database(tmp_path, monkeypatch):
    if not DB_URL:
        pytest.skip("set RAINFOREST_TEST_DATABASE_URL to run the API tests")
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    db.close_pool()
    db.bootstrap()  # the schema; the role named in READONLY_ROLE comes from the test
    yield
    db.close_pool()
    get_settings.cache_clear()


def _unmanaged(monkeypatch, role: str, password: str = "y"):
    monkeypatch.setenv("RAINFOREST_MANAGED_ROLES", "false")
    monkeypatch.setenv("READONLY_ROLE", role)
    monkeypatch.setenv("READONLY_PASSWORD", password)
    get_settings.cache_clear()


def test_unmanaged_roles_requires_the_role_to_exist(database, monkeypatch):
    _unmanaged(monkeypatch, "nobody_ro")
    with pytest.raises(SystemExit, match=r"nobody_ro does not exist.*CREATE ROLE nobody_ro LOGIN PASSWORD"):
        db.bootstrap()
    with psycopg.connect(DB_URL, autocommit=True) as conn:
        assert conn.execute("SELECT 1 FROM pg_roles WHERE rolname = 'nobody_ro'").fetchone() is None


def test_unmanaged_roles_grants_to_an_existing_role_and_leaves_it_alone(database, monkeypatch):
    with psycopg.connect(DB_URL, autocommit=True) as admin:
        admin.execute("CREATE ROLE kept_ro LOGIN PASSWORD 'x'")
        try:
            before = admin.execute(ROLE_STATE, ("kept_ro",)).fetchone()
            _unmanaged(monkeypatch, "kept_ro")
            db.bootstrap()

            assert admin.execute(ROLE_STATE, ("kept_ro",)).fetchone() == before  # password and settings untouched
            with psycopg.connect(make_conninfo(DB_URL, user="kept_ro", password="x")) as conn:
                conn.execute("SELECT count(*) FROM meta.datasets")
                conn.rollback()
                for table in db.PRIVATE_TABLES:
                    with pytest.raises(errors.InsufficientPrivilege):
                        conn.execute(sql.SQL("SELECT 1 FROM {}").format(sql.Identifier("meta", table)))
                    conn.rollback()
        finally:
            admin.execute("DROP OWNED BY kept_ro")
            admin.execute("DROP ROLE kept_ro")
