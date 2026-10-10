"""An in-process OpenID provider for tests, served through httpx.MockTransport: discovery, JWKS, and a token
endpoint that checks the client secret, redirect URI and PKCE verifier before issuing a signed ID token."""
from __future__ import annotations

import base64
import hashlib
import json
import time
from urllib.parse import parse_qs, urlsplit

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

ISSUER = "https://idp.test/realms/rainforest"
CLIENT_ID = "rainforest"
CLIENT_SECRET = "test-client-secret-long-enough-for-hs256"


def new_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


class FakeProvider:
    def __init__(self):
        self.key = new_key()
        self.kid = "k1"
        self.pending: dict[str, dict] = {}  # code -> what the authorize request carried
        self.token_requests: list[dict] = []
        self.issuer_in_discovery = ISSUER

    # -- what a test drives -------------------------------------------------------------------------------
    def authorize(self, location: str, claims: dict) -> str:
        """Play the user signing in at the provider: returns the code it would redirect back with."""
        q = {k: v[0] for k, v in parse_qs(urlsplit(location).query).items()}
        code = f"code-{len(self.pending)}"
        self.pending[code] = {"query": q, "claims": claims}
        return code

    def id_token(self, claims: dict, *, key=None, kid=None, alg="RS256", **overrides) -> str:
        now = int(time.time())
        body = {"iss": ISSUER, "aud": CLIENT_ID, "iat": now, "exp": now + 300, **claims, **overrides}
        body = {k: v for k, v in body.items() if v is not None}
        return jwt.encode(body, key or self.key, algorithm=alg, headers={"kid": kid or self.kid})

    # -- HTTP --------------------------------------------------------------------------------------------
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def jwks(self) -> dict:
        jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(self.key.public_key()))
        return {"keys": [{**jwk, "kid": self.kid, "use": "sig", "alg": "RS256"}]}

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/.well-known/openid-configuration"):
            return httpx.Response(200, json={
                "issuer": self.issuer_in_discovery,
                "authorization_endpoint": f"{ISSUER}/protocol/openid-connect/auth",
                "token_endpoint": f"{ISSUER}/protocol/openid-connect/token",
                "jwks_uri": f"{ISSUER}/protocol/openid-connect/certs",
                "end_session_endpoint": f"{ISSUER}/protocol/openid-connect/logout",
                "id_token_signing_alg_values_supported": ["RS256"],
            })
        if path.endswith("/certs"):
            return httpx.Response(200, json=self.jwks())
        if path.endswith("/token") and request.method == "POST":
            return self._token(request)
        return httpx.Response(404)

    def _token(self, request: httpx.Request) -> httpx.Response:
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        self.token_requests.append(form)
        expected = "Basic " + base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
        if request.headers.get("authorization") != expected:
            return httpx.Response(401, json={"error": "invalid_client"})
        grant = self.pending.pop(form.get("code", ""), None)
        if grant is None or form.get("grant_type") != "authorization_code":
            return httpx.Response(400, json={"error": "invalid_grant"})
        q = grant["query"]
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(form.get("code_verifier", "").encode()).digest()).decode().rstrip("=")
        if q.get("code_challenge_method") != "S256" or challenge != q.get("code_challenge"):
            return httpx.Response(400, json={"error": "invalid_grant", "error_description": "PKCE verification failed"})
        if form.get("redirect_uri") != q.get("redirect_uri"):
            return httpx.Response(400, json={"error": "invalid_grant", "error_description": "redirect_uri mismatch"})
        claims = {"nonce": q.get("nonce"), **grant["claims"]}
        return httpx.Response(200, json={"access_token": "at", "token_type": "Bearer",
                                         "id_token": self.id_token(claims)})
