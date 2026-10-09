"""Sign-in, the session cookie, and the request checks (host, origin, content type). No database needed:
the app is driven without its startup hook, and only routes that don't touch Postgres are called."""
import json
import os
import stat
import time

import pytest
from fastapi.testclient import TestClient

from app import auth
from app.config import get_settings
from app.main import app

from .conftest import ADMIN


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch, admin_login):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("RAINFOREST_SESSION_SECRET", raising=False)
    monkeypatch.delenv("RAINFOREST_ALLOWED_HOSTS", raising=False)
    get_settings.cache_clear()
    auth.limiter.clear()
    yield monkeypatch
    get_settings.cache_clear()
    auth.limiter.clear()


@pytest.fixture
def client():
    return TestClient(app)  # no `with`: skips the lifespan, which needs Postgres


def _login(client, **kw):
    return client.post("/api/auth/login", json={**ADMIN, **kw})


def test_protected_routes_need_a_session(client):
    for method, path in [("GET", "/api/options"), ("GET", "/api/datasets"), ("GET", "/api/auth/me"),
                         ("GET", "/api/exports/1/files/data"), ("GET", "/api/runs/1/files/model"),
                         ("POST", "/api/sql"), ("DELETE", "/api/datasets/1"), ("GET", "/docs")]:
        r = client.request(method, path)
        assert r.status_code == 401, (method, path, r.status_code)
        assert r.json()["detail"] == "Sign in required."
    r = client.post("/api/datasets/upload", files={"file": ("a.csv", b"a\n1\n", "text/csv")})
    assert r.status_code == 401


def test_login_me_logout(client):
    assert _login(client, password="wrong").status_code == 401
    assert _login(client, username="root").status_code == 401
    r = _login(client)
    assert r.status_code == 200 and r.json() == {"user": "admin"}
    cookie = r.headers["set-cookie"]
    assert cookie.startswith(f"{auth.COOKIE}=") and "HttpOnly" in cookie and "SameSite=strict" in cookie
    assert "Max-Age=604800" in cookie and "Path=/" in cookie and "Secure" not in cookie
    assert client.get("/api/auth/me").json() == {"user": "admin"}
    assert client.get("/api/options").status_code == 200

    r = client.post("/api/auth/logout")
    assert r.status_code == 204 and "Max-Age=0" in r.headers["set-cookie"]
    assert client.get("/api/auth/me").status_code == 401


def test_cookie_is_secure_behind_https(client):
    r = client.post("/api/auth/login", json=ADMIN, headers={"X-Forwarded-Proto": "https"})
    assert "Secure" in r.headers["set-cookie"]


def test_forged_expired_and_stale_sessions_are_refused(client, env):
    token = _login(client).cookies[auth.COOKIE]
    payload, sig = token.rsplit(".", 1)
    for bad in (f"{payload}.{sig[:-2]}xx", f"{payload}x.{sig}", "nonsense", auth.make_token(now=time.time() - 8 * 86400)):
        assert TestClient(app, cookies={auth.COOKIE: bad}).get("/api/options").status_code == 401, bad
    assert TestClient(app, cookies={auth.COOKIE: token}).get("/api/options").status_code == 200
    env.setenv("RAINFOREST_ADMIN_PASSWORD", "rotated")
    get_settings.cache_clear()
    assert client.get("/api/options").status_code == 401, "changing the password ends existing sessions"


@pytest.mark.parametrize("password", [None, "", auth.PLACEHOLDER_PASSWORD])
def test_no_password_refuses_everything_but_health(client, env, monkeypatch, password):
    if password is None:
        env.delenv("RAINFOREST_ADMIN_PASSWORD")
    else:
        env.setenv("RAINFOREST_ADMIN_PASSWORD", password)
    get_settings.cache_clear()
    monkeypatch.setattr("app.main.db.fetch_one", lambda *a, **k: {"ok": 1})
    assert client.get("/health").json() == {"status": "ok"}
    for r in (client.get("/api/options"), client.get("/api/auth/me"), _login(client, password=password or "")):
        assert r.status_code == 503
        assert "RAINFOREST_ADMIN_PASSWORD" in r.json()["detail"]


def test_session_secret_is_generated_once_and_private(tmp_path, env):
    auth._stored_secret.cache_clear()
    first = auth.session_secret()
    path = tmp_path / "session_secret"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    auth._stored_secret.cache_clear()  # as a second worker or a restart would
    assert auth.session_secret() == first and len(first) == 64
    env.setenv("RAINFOREST_SESSION_SECRET", "from-the-environment")
    get_settings.cache_clear()
    assert auth.session_secret() == b"from-the-environment"


def test_rate_limit_per_address(client):
    for _ in range(auth.LOGIN_MAX_FAILURES):
        assert _login(client, password="nope").status_code == 401
    blocked = _login(client)
    assert blocked.status_code == 429 and int(blocked.headers["retry-after"]) > 0
    other = client.post("/api/auth/login", json=ADMIN, headers={"X-Forwarded-For": "203.0.113.9"})
    assert other.status_code == 200


def test_cross_origin_writes_are_refused(client):
    for headers in ({"Origin": "http://evil.example"}, {"Origin": "null"}, {"Referer": "http://evil.example/page"},
                    {"Origin": "http://testserver:9999"}):
        r = client.post("/api/auth/login", json=ADMIN, headers=headers)
        assert r.status_code == 403, headers
    for headers in ({"Origin": "http://testserver"}, {"Referer": "http://testserver/d/1/review"},
                    {"Origin": "http://localhost:5173"}, {},
                    {"Host": "10.0.0.5:8080", "Origin": "http://10.0.0.5:8080"},
                    {"Host": "rf.example.com", "Origin": "https://rf.example.com"}):
        assert client.post("/api/auth/login", json=ADMIN, headers=headers).status_code == 200, headers


def test_bodies_must_be_json_or_multipart_for_upload(client):
    _login(client)
    body = json.dumps(ADMIN)
    for ctype in ("text/plain", "application/x-www-form-urlencoded", None):
        headers = {"Content-Type": ctype} if ctype else {}
        r = client.post("/api/auth/login", content=body, headers=headers)
        assert r.status_code == 415, ctype
    r = client.post("/api/sql", content=json.dumps({"sql": "SELECT 1"}), headers={"Content-Type": "text/plain"})
    assert r.status_code == 415
    r = client.post("/api/datasets/upload", json={"file": "x"})
    assert r.status_code == 415 and "multipart" in r.json()["detail"]
    assert client.post("/api/auth/login", content=body,
                       headers={"Content-Type": "application/json; charset=utf-8"}).status_code == 200


def test_allowed_hosts(client, env, monkeypatch):
    env.setenv("RAINFOREST_ALLOWED_HOSTS", "rf.example.com, *.lan")
    get_settings.cache_clear()
    monkeypatch.setattr("app.main.db.fetch_one", lambda *a, **k: {"ok": 1})
    assert client.get("/api/options").status_code == 400
    assert client.get("/health").status_code == 200
    for host in ("rf.example.com", "rf.example.com:8080", "box.lan", "localhost:8000", "127.0.0.1"):
        assert client.get("/api/options", headers={"Host": host}).status_code == 401, host
    assert client.get("/api/options", headers={"Host": "evil.example"}).status_code == 400
