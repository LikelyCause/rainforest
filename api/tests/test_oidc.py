"""OIDC sign-in against an in-process provider (tests/oidc_provider.py). The redirect and token checks need no
database; the flows that provision users run against RAINFOREST_TEST_DATABASE_URL and skip without it."""
import os
import time
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

from app import auth, oidc
from app.config import get_settings
from app.main import app

from .oidc_provider import CLIENT_ID, CLIENT_SECRET, ISSUER, FakeProvider, new_key

DB_URL = os.environ.get("RAINFOREST_TEST_DATABASE_URL")
OIDC_ENV = {
    "RAINFOREST_AUTH_MODE": "oidc", "RAINFOREST_OIDC_ISSUER": ISSUER, "RAINFOREST_OIDC_CLIENT_ID": CLIENT_ID,
    "RAINFOREST_OIDC_CLIENT_SECRET": CLIENT_SECRET, "RAINFOREST_OIDC_PROVIDER_NAME": "Keycloak",
    "RAINFOREST_OIDC_REDIRECT_URL": "http://testserver/api/auth/oidc/callback",
}
LEAD = {"sub": "u-lead", "name": "Lee Lead", "email": "lead@example.com", "groups": ["rainforest-lead"]}
MEMBER = {"sub": "u-ds", "name": "Dana Scientist", "email": "ds@example.com", "groups": ["/rainforest-member"]}


@pytest.fixture
def provider():
    p = FakeProvider()
    oidc.transport = p.transport()
    oidc.reset()
    yield p
    oidc.transport = None
    oidc.reset()


@pytest.fixture
def oidc_env(tmp_path, monkeypatch, provider):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    for k, v in OIDC_ENV.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()


def start_login(client, next_path="/d/1/review"):
    r = client.get("/api/auth/oidc/login", params={"next": next_path}, follow_redirects=False)
    assert r.status_code == 303, r.text
    return r, {k: v[0] for k, v in parse_qs(urlsplit(r.headers["location"]).query).items()}


def sign_in(client, provider, claims, next_path="/d/1/review"):
    r, q = start_login(client, next_path)
    code = provider.authorize(r.headers["location"], claims)
    return client.get("/api/auth/oidc/callback", params={"state": q["state"], "code": code}, follow_redirects=False)


# ------------------------------------------------------------------------------------- no database
def test_login_redirect_carries_pkce_state_and_nonce(oidc_env):
    client = TestClient(app)
    r, q = start_login(client)
    assert r.headers["location"].startswith(f"{ISSUER}/protocol/openid-connect/auth?")
    assert q["response_type"] == "code" and q["client_id"] == CLIENT_ID
    assert q["redirect_uri"] == OIDC_ENV["RAINFOREST_OIDC_REDIRECT_URL"]
    assert "openid" in q["scope"].split()
    assert q["code_challenge_method"] == "S256" and len(q["code_challenge"]) == 43
    assert len(q["state"]) >= 40 and len(q["nonce"]) >= 40 and q["nonce"] != q["state"]
    cookie = r.headers["set-cookie"]
    assert cookie.startswith(f"{auth.STATE_COOKIE}=") and "HttpOnly" in cookie and "SameSite=lax" in cookie
    assert "Path=/api/auth/oidc" in cookie and "Max-Age=600" in cookie
    _, q2 = start_login(client)
    assert q2["state"] != q["state"] and q2["code_challenge"] != q["code_challenge"]


def test_config_names_the_provider(oidc_env):
    cfg = TestClient(app).get("/api/auth/config").json()
    assert cfg == {"mode": "oidc", "provider_name": "Keycloak", "lead_group": "rainforest-lead",
                   "member_group": "rainforest-member", "problem": None, "accounts_from_env": False}


def test_callback_refuses_a_missing_or_mismatched_state(oidc_env, provider):
    client = TestClient(app)
    r, q = start_login(client)
    code = provider.authorize(r.headers["location"], LEAD)
    for params in ({"state": "forged", "code": code}, {"code": code}):
        bad = client.get("/api/auth/oidc/callback", params=params, follow_redirects=False)
        assert bad.status_code == 303 and bad.headers["location"] == "/login?error=state"
        assert [c for c in bad.headers.get_list("set-cookie") if c.startswith(f"{auth.COOKIE}=")] == []
    # Without the browser's state cookie (another browser, or a forged link) the right state is refused too.
    stranger = TestClient(app)
    bad = stranger.get("/api/auth/oidc/callback", params={"state": q["state"], "code": code}, follow_redirects=False)
    assert bad.headers["location"] == "/login?error=state"
    assert provider.token_requests == []


def test_callback_reports_a_provider_error(oidc_env, provider):
    client = TestClient(app)
    _, q = start_login(client)
    r = client.get("/api/auth/oidc/callback", params={"state": q["state"], "error": "access_denied"},
                   follow_redirects=False)
    assert r.headers["location"] == "/login?error=provider"


def test_code_from_another_login_fails_pkce(oidc_env, provider):
    """The verifier is derived from this browser's state, so a code issued for a different authorization
    request is refused by the provider's PKCE check."""
    client = TestClient(app)
    _, q_a = start_login(client)
    other = TestClient(app)
    r_b, _ = start_login(other)
    code_b = provider.authorize(r_b.headers["location"], LEAD)
    r = client.get("/api/auth/oidc/callback", params={"state": q_a["state"], "code": code_b}, follow_redirects=False)
    assert r.headers["location"] == "/login?error=token"
    assert provider.token_requests[-1]["code_verifier"]


def test_logout_returns_the_providers_end_session_url(oidc_env):
    r = TestClient(app).post("/api/auth/logout")
    url = r.json()["logout_url"]
    q = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
    assert url.startswith(f"{ISSUER}/protocol/openid-connect/logout?")
    assert q == {"client_id": CLIENT_ID, "post_logout_redirect_uri": "http://testserver/login"}


class TestIdTokenChecks:
    @pytest.fixture(autouse=True)
    def _env(self, oidc_env):
        pass

    def claims(self, provider, nonce="n-1"):
        return {"sub": "abc", "nonce": nonce}

    def test_valid(self, provider):
        assert oidc.verify_id_token(provider.id_token(self.claims(provider)), "n-1")["sub"] == "abc"

    @pytest.mark.parametrize("change, message", [
        ({"iss": "https://evil.example/realms/rainforest"}, "issuer"),
        ({"aud": "someone-else"}, "audience"),
        ({"exp": int(time.time()) - 3600, "iat": int(time.time()) - 7200}, "expired"),
        ({"exp": None}, "exp"),
        ({"iat": None}, "iat"),
        ({"nonce": "other"}, "nonce"),
        ({"aud": [CLIENT_ID, "other"]}, "azp"),
    ])
    def test_rejected_claims(self, provider, change, message):
        token = provider.id_token(self.claims(provider), **change)
        with pytest.raises(oidc.OIDCError, match=f"(?i){message}"):
            oidc.verify_id_token(token, "n-1")

    def test_several_audiences_with_matching_azp_pass(self, provider):
        token = provider.id_token(self.claims(provider), aud=[CLIENT_ID, "other"], azp=CLIENT_ID)
        assert oidc.verify_id_token(token, "n-1")["sub"] == "abc"

    def test_signature_must_verify_against_the_providers_keys(self, provider):
        forged = provider.id_token(self.claims(provider), key=new_key())
        with pytest.raises(oidc.OIDCError, match="Signature"):
            oidc.verify_id_token(forged, "n-1")
        unknown = provider.id_token(self.claims(provider), key=new_key(), kid="k-unknown")
        with pytest.raises(oidc.OIDCError, match="doesn't publish"):
            oidc.verify_id_token(unknown, "n-1")

    def test_symmetric_and_unsigned_tokens_are_refused(self, provider):
        import jwt
        hs = jwt.encode({"sub": "abc", "iss": ISSUER, "aud": CLIENT_ID, "nonce": "n-1",
                         "iat": int(time.time()), "exp": int(time.time()) + 60}, CLIENT_SECRET, algorithm="HS256")
        none = jwt.encode({"sub": "abc"}, None, algorithm="none")
        for token in (hs, none):
            with pytest.raises(oidc.OIDCError, match="not allowed"):
                oidc.verify_id_token(token, "n-1")

    def test_rotated_key_is_fetched(self, provider):
        oidc.verify_id_token(provider.id_token(self.claims(provider)), "n-1")  # caches the JWKS
        provider.key, provider.kid = new_key(), "k2"
        assert oidc.verify_id_token(provider.id_token(self.claims(provider)), "n-1")["sub"] == "abc"

    def test_discovery_must_name_the_configured_issuer(self, provider):
        provider.issuer_in_discovery = "https://idp.test/realms/other"
        with pytest.raises(oidc.OIDCError, match="names issuer"):
            oidc.metadata()


@pytest.mark.parametrize("groups, role", [
    (["rainforest-lead", "rainforest-member"], "lead"), (["/rainforest-member"], "member"),
    ("rainforest-lead", "lead"), (["/org/rainforest-lead"], "lead"), (["other"], None), (None, None),
])
def test_role_from_groups(oidc_env, groups, role):
    assert oidc.role_from_claims({"groups": groups}) == role


# ----------------------------------------------------------------------------------------- database
@pytest.fixture(scope="module")
def db_app(tmp_path_factory):
    if not DB_URL:
        pytest.skip("set RAINFOREST_TEST_DATABASE_URL to run the API tests")
    from app import db

    mp = pytest.MonkeyPatch()
    mp.setenv("DATABASE_URL", DB_URL)
    mp.setenv("DATA_DIR", str(tmp_path_factory.mktemp("data")))
    for k, v in OIDC_ENV.items():
        mp.setenv(k, v)
    get_settings.cache_clear()
    db.close_pool()
    with TestClient(app):
        yield app
    db.close_pool()
    mp.undo()
    get_settings.cache_clear()


@pytest.fixture
def flow(db_app, provider):
    get_settings.cache_clear()  # the module env, in case a no-database test above changed it
    return provider


def _user(sub):
    from app.db import fetch_one
    return fetch_one("SELECT * FROM meta.users WHERE oidc_subject = %s", (sub,))


def test_first_sign_in_provisions_the_user(flow):
    client = TestClient(app)
    r = sign_in(client, flow, LEAD)
    assert r.status_code == 303 and r.headers["location"] == "/d/1/review"
    cookies = {c.split("=", 1)[0]: c for c in r.headers.get_list("set-cookie")}
    assert "SameSite=strict" in cookies[auth.COOKIE] and "HttpOnly" in cookies[auth.COOKIE]
    assert "Max-Age=0" in cookies[auth.STATE_COOKIE], "the one-time state cookie is cleared"
    me = client.get("/api/auth/me").json()
    assert {k: me[k] for k in ("name", "email", "role", "mode")} == {
        "name": "Lee Lead", "email": "lead@example.com", "role": "lead", "mode": "oidc"}
    assert _user("u-lead")["last_seen_at"] is not None

    member = TestClient(app)
    sign_in(member, flow, MEMBER)
    assert member.get("/api/auth/me").json()["role"] == "member"
    assert member.delete("/api/datasets/999999").status_code == 403


def test_no_group_gets_the_forbidden_page_and_no_account(flow):
    client = TestClient(app)
    r = sign_in(client, flow, {"sub": "u-outsider", "name": "Out Sider", "groups": ["other"]})
    assert r.headers["location"] == "/forbidden?reason=no_group"
    assert [c for c in r.headers.get_list("set-cookie") if c.startswith(f"{auth.COOKIE}=")] == []
    assert client.get("/api/auth/me").status_code == 401
    assert _user("u-outsider") is None


def test_role_and_profile_refresh_on_every_sign_in(flow):
    old = TestClient(app)
    sign_in(old, flow, {**MEMBER, "sub": "u-mover"})
    assert old.get("/api/auth/me").json()["role"] == "member"

    new = TestClient(app)
    sign_in(new, flow, {**MEMBER, "sub": "u-mover", "name": "Dana Renamed", "groups": ["rainforest-lead"]})
    me = new.get("/api/auth/me").json()
    assert me["role"] == "lead" and me["name"] == "Dana Renamed"
    assert old.get("/api/auth/me").status_code == 401, "a role change ends the user's other sessions"

    again = TestClient(app)
    sign_in(again, flow, {**MEMBER, "sub": "u-mover", "name": "Dana Renamed", "groups": ["rainforest-lead"]})
    assert new.get("/api/auth/me").status_code == 200, "signing in again with the same role keeps other sessions"

    gone = sign_in(TestClient(app), flow, {**MEMBER, "sub": "u-mover", "groups": []})
    assert gone.headers["location"] == "/forbidden?reason=no_group"
    assert new.get("/api/auth/me").status_code == 401, "leaving both groups ends the user's sessions"


def test_deactivation_logs_out_and_blocks_sign_in(flow):
    lead = TestClient(app)
    sign_in(lead, flow, LEAD)
    member = TestClient(app)
    sign_in(member, flow, {**MEMBER, "sub": "u-leaver"})
    uid = member.get("/api/auth/me").json()["id"]

    r = lead.patch(f"/api/users/{uid}", json={"name": "Renamed"})
    assert r.status_code == 409 and "identity provider" in r.json()["detail"]
    assert lead.post("/api/users", json={"name": "New"}).status_code == 409
    r = lead.patch(f"/api/users/{uid}", json={"active": False})
    assert r.status_code == 200 and r.json()["active"] is False and r.json()["sso"] is True
    assert member.get("/api/auth/me").status_code == 401
    r = sign_in(TestClient(app), flow, {**MEMBER, "sub": "u-leaver"})
    assert r.headers["location"] == "/forbidden?reason=deactivated"

    assert lead.patch(f"/api/users/{uid}", json={"active": True}).status_code == 200
    back = TestClient(app)
    sign_in(back, flow, {**MEMBER, "sub": "u-leaver"})
    assert back.get("/api/auth/me").json()["id"] == uid
