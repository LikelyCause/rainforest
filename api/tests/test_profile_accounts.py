"""Accounts from .env, the Viewer role, and profile pictures."""
import pytest

from app.users import Account, parse_accounts

from .team_helpers import DB_URL, app_, people  # noqa: F401


# ------------------------------------------------------------------------------------------- parsing
def test_parse_accounts():
    got = parse_accounts("tanner#2:Tanner:lead; test-member:Test Member:member;test-viewer:Test Viewer:viewer")
    assert got == [Account("tanner", 2, "Tanner", "lead"), Account("test-member", None, "Test Member", "member"),
                   Account("test-viewer", None, "Test Viewer", "viewer")]


@pytest.mark.parametrize("text,needle", [
    ("a:A:lead;a:B:member", "twice"), ("a:A:lead;x:X:boss", "role"), ("a:A:lead;bad", "key:Name:role"),
    ("Tanner:Tanner:lead", "key"), ("m:M:member", "lead"), ("a#x:A:lead", "id"), ("a::lead", "name"),
])
def test_parse_refuses(text, needle):
    with pytest.raises(ValueError, match=needle):
        parse_accounts(text)


# ------------------------------------------------------------------------------------------- sync
needs_db = pytest.mark.skipif(not DB_URL, reason="set RAINFOREST_TEST_DATABASE_URL to run the API tests")


@pytest.fixture
def conn(app_):
    """A transaction rolled back after the test, so the shared test users are untouched."""
    from app.db import connection
    with connection() as c:
        yield c
        c.rollback()


def _row(conn, uid):
    return conn.execute("SELECT id, key, name, role, active, session_version FROM meta.users WHERE id = %s",
                        (uid,)).fetchone()


def _new(conn, name, role="member"):
    return conn.execute("INSERT INTO meta.users (name, role) VALUES (%s, %s) RETURNING id", (name, role)).fetchone()["id"]


@needs_db
def test_sync_claims_creates_deactivates_and_is_idempotent(conn):
    from app.users import sync_accounts
    lee, dana = _new(conn, "Lee Lead", "lead"), _new(conn, "Dana Scientist")
    v_dana = _row(conn, dana)["session_version"]
    accounts = parse_accounts(f"tanner#{lee}:Tanner:lead;test-member:Test Member:member;test-viewer:Test Viewer:viewer")
    sync_accounts(conn, accounts)
    t = _row(conn, lee)
    assert (t["key"], t["name"], t["role"], t["active"]) == ("tanner", "Tanner", "lead", True)
    d = _row(conn, dana)
    assert not d["active"] and d["session_version"] == v_dana + 1
    keys = {r["key"]: r for r in conn.execute("SELECT * FROM meta.users WHERE active").fetchall()}
    assert set(keys) == {"tanner", "test-member", "test-viewer"} and keys["test-viewer"]["role"] == "viewer"
    before = conn.execute("SELECT id, name, role, active, session_version FROM meta.users ORDER BY id").fetchall()
    assert sync_accounts(conn, accounts) == {"created": [], "updated": [], "deactivated": [], "reactivated": []}
    assert conn.execute("SELECT id, name, role, active, session_version FROM meta.users ORDER BY id").fetchall() == before


@needs_db
def test_sync_removed_key_deactivates_and_returning_key_reactivates(conn):
    from app.users import sync_accounts
    sync_accounts(conn, parse_accounts("boss:Boss:lead;tm:Test Member:member"))
    tm = conn.execute("SELECT id FROM meta.users WHERE key = 'tm'").fetchone()["id"]
    sync_accounts(conn, parse_accounts("boss:Boss:lead"))
    assert not _row(conn, tm)["active"]
    sync_accounts(conn, parse_accounts("boss:Boss:lead;tm:Test Member:member"))
    assert _row(conn, tm)["active"]


@needs_db
def test_sync_role_change_signs_out(conn):
    from app.users import sync_accounts
    sync_accounts(conn, parse_accounts("boss:Boss:lead;tm:Test Member:member"))
    tm = _row(conn, conn.execute("SELECT id FROM meta.users WHERE key = 'tm'").fetchone()["id"])
    sync_accounts(conn, parse_accounts("boss:Boss:lead;tm:Test Member:viewer"))
    after = _row(conn, tm["id"])
    assert after["role"] == "viewer" and after["session_version"] == tm["session_version"] + 1


@needs_db
def test_accounts_from_env_are_not_edited_in_the_app(people, monkeypatch):
    from app.config import get_settings
    lead = people["lead"]
    monkeypatch.setenv("RAINFOREST_USERS", "boss:Boss:lead")
    get_settings.cache_clear()
    try:
        assert lead.post("/api/users", json={"name": "X", "role": "member"}).status_code == 409
        assert lead.patch(f"/api/users/{lead.user['id']}", json={"name": "Y"}).status_code == 409
    finally:
        monkeypatch.delenv("RAINFOREST_USERS")
        get_settings.cache_clear()


@needs_db
def test_picker_lists_key_and_picture(people):
    users = people["lead"].get("/api/auth/users").json()
    assert all({"id", "name", "role", "key", "avatar_url"} <= set(u) for u in users)
