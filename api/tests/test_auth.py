"""Pick-mode sign-in, the session cookie, the request checks (host, origin, content type) and role gates. No database
needed: the app is driven without its startup hook, and the user store is an in-memory fake."""

import json
import os
import stat
import time

import pytest
from fastapi.testclient import TestClient

from app import auth
from app.config import get_settings
from app.main import app


class FakeUsers:
    def __init__(self):
        self.rows = {
            1: {"id": 1, "name": "Admin", "email": None, "role": "lead", "active": True, "session_version": 1},
            2: {
                "id": 2,
                "name": "Dana",
                "email": "dana@example.com",
                "role": "member",
                "active": True,
                "session_version": 1,
            },
        }

    def session_user(self, user_id, version):
        u = self.rows.get(user_id)
        if u and u["active"] and u["session_version"] == version:
            return {k: u[k] for k in ("id", "name", "email", "role")}
        return None

    def get(self, user_id):
        return self.rows.get(user_id)

    def pickable(self):
        return [{k: u[k] for k in ("id", "name", "role")} for u in self.rows.values() if u["active"]]

    def touch(self, user_id):
        pass


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAINFOREST_AUTH_MODE", "pick")
    for name in (
        "RAINFOREST_SESSION_SECRET",
        "RAINFOREST_ALLOWED_HOSTS",
        "RAINFOREST_OIDC_ISSUER",
        "RAINFOREST_OIDC_CLIENT_ID",
        "RAINFOREST_OIDC_CLIENT_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def store(monkeypatch):
    fake = FakeUsers()
    for name in ("session_user", "get", "pickable", "touch"):
        monkeypatch.setattr(f"app.users.{name}", getattr(fake, name))
    return fake


@pytest.fixture
def client():
    return TestClient(app)  # no `with`: skips the lifespan, which needs Postgres


def _pick(client, user_id=1, **kw):
    return client.post("/api/auth/pick", json={"user_id": user_id}, **kw)


def test_protected_routes_need_a_session(client):
    for method, path in [
        ("GET", "/api/options"),
        ("GET", "/api/datasets"),
        ("GET", "/api/auth/me"),
        ("GET", "/api/users"),
        ("GET", "/api/auth/check"),
        ("GET", "/api/exports/1/files/data"),
        ("GET", "/api/runs/1/files/model"),
        ("POST", "/api/sql"),
        ("DELETE", "/api/datasets/1"),
        ("GET", "/docs"),
    ]:
        r = client.request(method, path)
        assert r.status_code == 401, (method, path, r.status_code)
        assert r.json()["detail"] == "Sign in required."
    r = client.post("/api/datasets/upload", files={"file": ("a.csv", b"a\n1\n", "text/csv")})
    assert r.status_code == 401


def test_pick_me_logout(client):
    cfg = client.get("/api/auth/config").json()
    assert cfg["mode"] == "pick" and cfg["problem"] is None
    assert client.get("/api/auth/users").json() == [
        {"id": 1, "name": "Admin", "role": "lead"},
        {"id": 2, "name": "Dana", "role": "member"},
    ]
    assert _pick(client, 99).status_code == 404
    r = _pick(client, 2)
    assert r.status_code == 200 and r.json()["name"] == "Dana"
    cookie = r.headers["set-cookie"]
    assert cookie.startswith(f"{auth.COOKIE}=") and "HttpOnly" in cookie and "SameSite=strict" in cookie
    assert "Max-Age=604800" in cookie and "Path=/" in cookie and "Secure" not in cookie
    assert client.get("/api/auth/me").json() == {
        "id": 2,
        "name": "Dana",
        "email": "dana@example.com",
        "role": "member",
        "mode": "pick",
        "can_rename": True,
    }
    assert client.get("/api/options").status_code == 200
    assert client.get("/api/auth/check").status_code == 204

    r = client.post("/api/auth/logout")
    assert r.status_code == 200 and r.json() == {"logout_url": None} and "Max-Age=0" in r.headers["set-cookie"]
    assert client.get("/api/auth/me").status_code == 401
    assert client.get("/api/auth/check").status_code == 401


def test_deactivated_users_cant_pick_or_keep_their_session(client, store):
    _pick(client, 2)
    store.rows[2]["active"] = False
    assert client.get("/api/auth/me").status_code == 401
    assert _pick(client, 2).status_code == 404
    assert [u["id"] for u in client.get("/api/auth/users").json()] == [1]


def test_session_version_bump_ends_sessions(client, store):
    _pick(client, 2)
    store.rows[2]["session_version"] = 2  # what a role change or deactivation does
    assert client.get("/api/auth/me").status_code == 401
    _pick(client, 2)
    assert client.get("/api/auth/me").status_code == 200


def test_cookie_is_secure_behind_https(client):
    assert "Secure" in _pick(client, headers={"X-Forwarded-Proto": "https"}).headers["set-cookie"]


def test_forged_expired_and_foreign_tokens_are_refused(client, env):
    token = _pick(client).cookies[auth.COOKIE]
    payload, sig = token.rsplit(".", 1)
    state = auth._seal("oidc-state", {"u": 1, "v": 1, "exp": int(time.time()) + 60})
    for bad in (
        f"{payload}.{sig[:-2]}xx",
        f"{payload}x.{sig}",
        "nonsense",
        state,
        auth.make_token(1, 1, now=time.time() - 8 * 86400),
    ):
        assert TestClient(app, cookies={auth.COOKIE: bad}).get("/api/options").status_code == 401, bad
    assert TestClient(app, cookies={auth.COOKIE: token}).get("/api/options").status_code == 200
    env.setenv("RAINFOREST_SESSION_SECRET", "rotated")
    get_settings.cache_clear()
    assert client.get("/api/options").status_code == 401, "rotating the secret ends existing sessions"


def test_lead_only_routes(client):
    _pick(client, 2)
    r = client.delete("/api/datasets/1")
    assert r.status_code == 403 and "lead" in r.json()["detail"]
    for method, path, body in [("POST", "/api/users", {"name": "X"}), ("PATCH", "/api/users/1", {"name": "X"})]:
        assert client.request(method, path, json=body).status_code == 403, path


def test_pick_endpoints_are_off_in_oidc_mode(client, env):
    env.setenv("RAINFOREST_AUTH_MODE", "oidc")
    env.setenv("RAINFOREST_OIDC_ISSUER", "https://idp.example")
    env.setenv("RAINFOREST_OIDC_CLIENT_ID", "rainforest")
    env.setenv("RAINFOREST_OIDC_CLIENT_SECRET", "s")
    get_settings.cache_clear()
    assert client.get("/api/auth/users").status_code == 404
    assert _pick(client).status_code == 404


def test_unconfigured_oidc_refuses_everything_but_health_and_config(client, env, monkeypatch):
    env.setenv("RAINFOREST_AUTH_MODE", "oidc")
    get_settings.cache_clear()
    monkeypatch.setattr("app.main.db.fetch_one", lambda *a, **k: {"ok": 1})
    assert client.get("/health").json() == {"status": "ok"}
    cfg = client.get("/api/auth/config").json()
    assert cfg["mode"] == "oidc" and "RAINFOREST_OIDC_ISSUER" in cfg["problem"]
    for r in (client.get("/api/options"), client.get("/api/auth/me"), client.get("/api/auth/oidc/login")):
        assert r.status_code == 503
        assert "RAINFOREST_OIDC_ISSUER" in r.json()["detail"]


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


def test_cross_origin_writes_are_refused(client):
    for headers in (
        {"Origin": "http://evil.example"},
        {"Origin": "null"},
        {"Referer": "http://evil.example/page"},
        {"Origin": "http://testserver:9999"},
    ):
        assert _pick(client, headers=headers).status_code == 403, headers
    for headers in (
        {"Origin": "http://testserver"},
        {"Referer": "http://testserver/d/1/review"},
        {"Origin": "http://localhost:5173"},
        {},
        {"Host": "10.0.0.5:8080", "Origin": "http://10.0.0.5:8080"},
        {"Host": "rf.example.com", "Origin": "https://rf.example.com"},
    ):
        assert _pick(client, headers=headers).status_code == 200, headers


def test_auth_check_for_nginx(client):
    """auth_request forwards the original method and headers but no body; the origin check still applies."""
    _pick(client)
    for method in ("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"):
        r = client.request(method, "/api/auth/check", headers={"Content-Length": "120", "Origin": "http://testserver"})
        assert r.status_code == 204, method
    assert client.post("/api/auth/check", headers={"Origin": "http://evil.example"}).status_code == 403


def test_bodies_must_be_json_or_multipart_for_upload(client):
    _pick(client)
    body = json.dumps({"user_id": 1})
    for ctype in ("text/plain", "application/x-www-form-urlencoded", None):
        headers = {"Content-Type": ctype} if ctype else {}
        assert client.post("/api/auth/pick", content=body, headers=headers).status_code == 415, ctype
    r = client.post("/api/sql", content=json.dumps({"sql": "SELECT 1"}), headers={"Content-Type": "text/plain"})
    assert r.status_code == 415
    r = client.post("/api/datasets/upload", json={"file": "x"})
    assert r.status_code == 415 and "multipart" in r.json()["detail"]
    assert (
        client.post(
            "/api/auth/pick", content=body, headers={"Content-Type": "application/json; charset=utf-8"}
        ).status_code
        == 200
    )


def test_allowed_hosts(client, env, monkeypatch):
    env.setenv("RAINFOREST_ALLOWED_HOSTS", "rf.example.com, *.lan")
    get_settings.cache_clear()
    monkeypatch.setattr("app.main.db.fetch_one", lambda *a, **k: {"ok": 1})
    assert client.get("/api/options").status_code == 400
    assert client.get("/health").status_code == 200
    for host in ("rf.example.com", "rf.example.com:8080", "box.lan", "localhost:8000", "127.0.0.1"):
        assert client.get("/api/options", headers={"Host": host}).status_code == 401, host
    assert client.get("/api/options", headers={"Host": "evil.example"}).status_code == 400


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "/"),
        ("/d/1/review", "/d/1/review"),
        ("//evil.example", "/"),
        ("https://evil.example", "/"),
        ("/\\evil.example", "/"),
        ("/login?next=/x", "/"),
        ("relative", "/"),
    ],
)
def test_safe_next(value, expected):
    assert auth.safe_next(value) == expected
