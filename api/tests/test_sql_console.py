"""SQL console against a real Postgres: one statement only, read-only, bounded, nothing leaks between requests.

Needs an empty database: RAINFOREST_TEST_DATABASE_URL=postgresql://user:pw@host:port/db pytest
"""
import os

import pytest

DB_URL = os.environ.get("RAINFOREST_TEST_DATABASE_URL")


@pytest.fixture(scope="module")
def client(tmp_path_factory, admin_login):
    if not DB_URL:
        pytest.skip("set RAINFOREST_TEST_DATABASE_URL to run the API tests")
    from fastapi.testclient import TestClient

    from app import db
    from app.config import get_settings

    mp = pytest.MonkeyPatch()
    mp.setenv("DATABASE_URL", DB_URL)
    mp.setenv("DATA_DIR", str(tmp_path_factory.mktemp("data")))
    mp.setenv("SQL_TIMEOUT_MS", "2000")
    get_settings.cache_clear()
    db.close_pool()
    from app.main import app

    with TestClient(app) as c:
        yield admin_login(c)
    db.close_pool()
    mp.undo()
    get_settings.cache_clear()


def _sql(client, q: str):
    return client.post("/api/sql", json={"sql": q})


# All but the last hid a second statement from the old literal/comment-stripping guard.
SMUGGLED = [
    "SELECT '--'; SET statement_timeout = 0; SELECT pg_sleep(1e5)",
    "SELECT '/*'; SET statement_timeout = 0; SELECT '*/'",
    "SELECT E'it\\'s'; SET statement_timeout = 0; SELECT ''",
    "SELECT $$ ' $$; SET statement_timeout = 0; SELECT ''",
    "SELECT $tag$ -- $tag$; SET statement_timeout = 0",
    "SELECT 1; COMMIT; SET SESSION CHARACTERISTICS AS TRANSACTION READ WRITE",
]


@pytest.mark.parametrize("q", SMUGGLED)
def test_second_statement_is_rejected(client, q):
    r = _sql(client, q)
    assert r.status_code == 400, r.text
    assert r.json()["detail"]["message"] == "Run one statement at a time."
    assert _sql(client, "SELECT current_setting('statement_timeout') AS t").json()["rows"] == [["2s"]]


@pytest.mark.parametrize("q", ["SET statement_timeout = 0", "COMMIT", "CREATE TEMP TABLE t (x int)",
                               "DELETE FROM meta.datasets", "EXPLAIN SELECT 1"])
def test_only_queries_run(client, q):
    r = _sql(client, q)
    assert r.status_code == 400, r.text
    assert "SELECT" in r.json()["detail"]["detail"]


@pytest.mark.parametrize("q, rows", [
    ("SELECT $$;$$ AS s", [[";"]]),
    ("SELECT ';' AS s, '100%' AS p;", [[";", "100%"]]),
    ("WITH a AS (SELECT 1 AS x) SELECT x FROM a", [[1]]),
    ("VALUES (1, 'a')", [[1, "a"]]),
])
def test_single_queries_with_tricky_text_run(client, q, rows):
    r = _sql(client, q)
    assert r.status_code == 200, r.text
    assert r.json()["rows"] == rows


def test_writes_are_refused_even_through_functions(client):
    r = _sql(client, "SELECT * FROM meta.datasets FOR UPDATE")
    assert r.status_code == 400 and "read-only" in r.json()["detail"]["message"]


def test_only_the_shown_rows_are_computed(client):
    # Row 2002 onwards divides by zero; a fully buffered result would hit it, a streamed one stops at 2001.
    r = _sql(client, "SELECT g, CASE WHEN g > 2001 THEN 1 / (g - g) END AS boom FROM generate_series(1, 5000000) g")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["row_count"] == 2000 and body["truncated"] is True and body["rows"][-1] == [2000, None]


def test_timeout_and_session_state_do_not_leak(client):
    r = _sql(client, "SELECT pg_sleep(5)")
    assert r.status_code == 400 and "statement timeout" in r.json()["detail"]["message"]
    r = _sql(client, "SELECT set_config('statement_timeout', '0', false), pg_advisory_lock(4242)")
    assert r.status_code == 200, r.text
    after = _sql(client, "SELECT current_setting('statement_timeout') AS t, "
                         "(SELECT count(*) FROM pg_locks WHERE locktype = 'advisory') AS locks").json()
    assert after["rows"] == [["2s", 0]]


def test_console_needs_a_session(client):
    from fastapi.testclient import TestClient

    from app.main import app

    assert TestClient(app).post("/api/sql", json={"sql": "SELECT 1"}).status_code == 401
