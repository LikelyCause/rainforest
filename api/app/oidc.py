"""A small OpenID Connect relying party: authorization-code flow with PKCE, ID token verification against the
provider's JWKS. Only what Rainforest needs, so every check is visible here."""

from __future__ import annotations

import base64
import hashlib
import hmac
import threading
import time
from urllib.parse import urlencode

import httpx
import jwt

from .config import get_settings

DISCOVERY_TTL = 3600
JWKS_MIN_REFRESH = 60  # an unknown key id (rotation) refetches the JWKS at most this often
LEEWAY = 60
# Asymmetric algorithms only: a symmetric one would make the client secret a signing key.
ALGORITHMS = ["RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "ES512", "EdDSA"]

# Tests swap in an httpx.MockTransport.
transport: httpx.BaseTransport | None = None

_lock = threading.Lock()
_cache: dict = {}


class OIDCError(Exception):
    pass


def _client() -> httpx.Client:
    return httpx.Client(transport=transport, timeout=10)


def config_problem() -> str | None:
    s = get_settings()
    missing = [
        n
        for n, v in (
            ("RAINFOREST_OIDC_ISSUER", s.oidc_issuer),
            ("RAINFOREST_OIDC_CLIENT_ID", s.oidc_client_id),
            ("RAINFOREST_OIDC_CLIENT_SECRET", s.oidc_client_secret),
        )
        if not v
    ]
    if missing:
        return f"Sign-in is set to OIDC but {', '.join(missing)} is not set. Set it for the api service and restart it."
    return None


def reset() -> None:
    with _lock:
        _cache.clear()


def metadata() -> dict:
    s = get_settings()
    issuer = s.oidc_issuer.rstrip("/")
    with _lock:
        cached = _cache.get("meta")
        if cached and cached[0] > time.time() and cached[1]["issuer"].rstrip("/") == issuer:
            return cached[1]
    url = s.oidc_discovery_url or f"{issuer}/.well-known/openid-configuration"
    try:
        with _client() as c:
            r = c.get(url)
            r.raise_for_status()
            meta = r.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise OIDCError(f"Couldn't read the provider's discovery document at {url}: {exc}") from exc
    # The discovery document must name the issuer we were configured with, or tokens can't be checked against it.
    if str(meta.get("issuer", "")).rstrip("/") != issuer:
        raise OIDCError(f"Discovery at {url} names issuer {meta.get('issuer')!r}, not {s.oidc_issuer!r}")
    for key in ("authorization_endpoint", "token_endpoint", "jwks_uri"):
        if not meta.get(key):
            raise OIDCError(f"Discovery at {url} has no {key}")
    with _lock:
        _cache["meta"] = (time.time() + DISCOVERY_TTL, meta)
    return meta


def _jwks(force: bool = False) -> jwt.PyJWKSet:
    with _lock:
        cached = _cache.get("jwks")
        if cached and (not force or _cache.get("jwks_forced", 0) + JWKS_MIN_REFRESH > time.time()):
            return cached
        if force:
            _cache["jwks_forced"] = time.time()
    try:
        with _client() as c:
            r = c.get(metadata()["jwks_uri"])
            r.raise_for_status()
            keys = jwt.PyJWKSet.from_dict(r.json())
    except (httpx.HTTPError, ValueError, jwt.PyJWTError) as exc:
        raise OIDCError(f"Couldn't read the provider's signing keys: {exc}") from exc
    with _lock:
        _cache["jwks"] = keys
    return keys


def _signing_key(kid: str | None) -> jwt.PyJWK:
    for force in (False, True):
        keys = _jwks(force)
        matches = [k for k in keys.keys if kid is None or k.key_id == kid]
        if matches:
            return matches[0]
    raise OIDCError("The ID token was signed with a key the provider doesn't publish")


def pkce_challenge(verifier: str) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")


def authorization_url(state: str, nonce: str, verifier: str, redirect_uri: str) -> str:
    s = get_settings()
    params = {
        "response_type": "code",
        "client_id": s.oidc_client_id,
        "redirect_uri": redirect_uri,
        "scope": s.oidc_scopes,
        "state": state,
        "nonce": nonce,
        "code_challenge": pkce_challenge(verifier),
        "code_challenge_method": "S256",
    }
    endpoint = metadata()["authorization_endpoint"]
    return f"{endpoint}{'&' if '?' in endpoint else '?'}{urlencode(params)}"


def exchange_code(code: str, verifier: str, redirect_uri: str) -> dict:
    s = get_settings()
    try:
        with _client() as c:
            r = c.post(
                metadata()["token_endpoint"],
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "code_verifier": verifier,
                },
                auth=(s.oidc_client_id, s.oidc_client_secret),
            )
    except httpx.HTTPError as exc:
        raise OIDCError(f"The token request failed: {exc}") from exc
    if r.status_code != 200:
        raise OIDCError(f"The provider refused the code ({r.status_code}): {r.text[:300]}")
    try:
        body = r.json()
    except ValueError as exc:
        raise OIDCError("The token response wasn't JSON") from exc
    if not body.get("id_token"):
        raise OIDCError("The token response has no id_token (is 'openid' in the scopes?)")
    return body


def verify_id_token(token: str, nonce: str) -> dict:
    """Signature (provider's JWKS, asymmetric algorithms only), issuer, audience, expiry, issued-at and nonce."""
    s = get_settings()
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as exc:
        raise OIDCError(f"Malformed ID token: {exc}") from exc
    if header.get("alg") not in ALGORITHMS:
        raise OIDCError(f"ID token algorithm {header.get('alg')!r} is not allowed")
    key = _signing_key(header.get("kid"))
    try:
        claims = jwt.decode(
            token,
            key,
            algorithms=[header["alg"]],
            audience=s.oidc_client_id,
            issuer=metadata()["issuer"],
            leeway=LEEWAY,
            options={"require": ["iss", "aud", "exp", "iat", "sub"]},
        )
    except jwt.PyJWTError as exc:
        raise OIDCError(f"ID token rejected: {exc}") from exc
    aud = claims["aud"] if isinstance(claims["aud"], list) else [claims["aud"]]
    if len(aud) > 1 and claims.get("azp") != s.oidc_client_id:
        raise OIDCError("ID token has several audiences but wasn't issued to this client (azp)")
    if not hmac.compare_digest(str(claims.get("nonce", "")).encode(), nonce.encode()):
        raise OIDCError("ID token nonce doesn't match this sign-in")
    return claims


def role_from_claims(claims: dict) -> str | None:
    s = get_settings()
    raw = claims.get(s.oidc_groups_claim) or []
    groups = {
        g.strip("/").rsplit("/", 1)[-1] if g.startswith("/") else g
        for g in ([raw] if isinstance(raw, str) else raw)
        if isinstance(g, str)
    }
    if s.lead_group in groups:
        return "lead"
    if s.member_group in groups:
        return "member"
    return None


def profile_from_claims(claims: dict) -> tuple[str, str | None]:
    email = claims.get("email") if isinstance(claims.get("email"), str) else None
    name = next(
        (claims[k] for k in ("name", "preferred_username") if isinstance(claims.get(k), str) and claims[k]),
        email or claims["sub"],
    )
    return name[:200], email


def end_session_url(post_logout_redirect_uri: str) -> str | None:
    """Where to send the browser to end the provider's session too, or None when it has no end_session_endpoint."""
    try:
        endpoint = metadata().get("end_session_endpoint")
    except OIDCError:
        return None
    if not endpoint:
        return None
    params = {"client_id": get_settings().oidc_client_id, "post_logout_redirect_uri": post_logout_redirect_uri}
    return f"{endpoint}{'&' if '?' in endpoint else '?'}{urlencode(params)}"
