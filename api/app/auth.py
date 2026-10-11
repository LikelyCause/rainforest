"""Sign-in (OIDC or pick-a-user), the session cookie, the `current_user` / `require_lead` dependencies, and the checks
every request passes before reaching a route."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from functools import lru_cache
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from . import oidc, users
from .config import get_settings

log = logging.getLogger(__name__)

COOKIE = "rf_session"
STATE_COOKIE = "rf_oidc"
SESSION_TTL = 7 * 24 * 3600
STATE_TTL = 10 * 60
OIDC_PATH = "/api/auth/oidc"
OPEN_PATHS = {"/health", "/api/auth/config", "/api/auth/logout", "/api/auth/users", "/api/auth/pick",
              f"{OIDC_PATH}/login", f"{OIDC_PATH}/callback"}
CHECK_PATH = "/api/auth/check"
UPLOAD_PATH = "/api/datasets/upload"
AVATAR_UPLOAD = re.compile(r"/api/users/\d+/avatar")
AVATAR_MAX_BODY = 5 * 1024 * 1024 + 64 * 1024
AVATAR_PREFIX = "/api/auth/avatars/"  # open to GET: the picker shows pictures before anyone signs in
DEV_ORIGINS = {"http://localhost:5173", "http://127.0.0.1:5173"}
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
# A viewer reads only. These writes are reads in disguise (Explore, the field finder) or their own profile; every
# other write is refused, the SQL console included (it reads every table) and any route added later.
VIEWER_ALLOWED = [(m, re.compile(p)) for m, p in (
    ("POST", r"/api/datasets/\d+/analyses"), ("POST", r"/api/datasets/\d+/fields/query"),
    ("PUT", r"/api/users/\d+/avatar"), ("DELETE", r"/api/users/\d+/avatar"), ("PATCH", r"/api/auth/me"),
)]
# MLflow's UI loads runs, models and traces with POSTs that only search; a viewer may make those and no other.
MLFLOW_SEARCH = re.compile(r"/mlflow/(ajax-)?api/2\.0/mlflow/"
                           r"(runs/search|experiments/search|logged-models/search|traces/search|"
                           r"metrics/get-history-bulk-interval)")
VIEWER_REFUSED = "Viewers can look but not change anything."


def viewer_refused(user: dict, method: str, path: str, original_method: str | None = None,
                   original_uri: str | None = None) -> bool:
    if user.get("role") != "viewer":
        return False
    if path == CHECK_PATH:  # nginx's auth_request for MLflow: judge the request it stands for
        unsafe = (original_method or method).upper() in UNSAFE_METHODS
        uri = (original_uri or "").split("?")[0]
        return unsafe and not ((original_method or "").upper() == "POST" and ".." not in uri
                               and MLFLOW_SEARCH.fullmatch(uri))
    return method in UNSAFE_METHODS and not any(m == method and p.fullmatch(path) for m, p in VIEWER_ALLOWED)


def config_problem() -> str | None:
    return oidc.config_problem() if get_settings().auth_mode == "oidc" else None


# ------------------------------------------------------------------------------------------ signing
def session_secret() -> bytes:
    s = get_settings()
    if s.session_secret:
        return s.session_secret.encode()
    return _stored_secret(s.data_dir.resolve())


@lru_cache
def _stored_secret(data_dir: Path) -> bytes:
    """Generated once and shared by every worker through the data volume. Not kept in the database,
    because the SQL console's read-only role can read all of meta.*."""
    path = data_dir / "session_secret"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        tmp = data_dir / f".session_secret.{os.getpid()}.{secrets.token_hex(4)}"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_hex(32))
        try:
            os.link(tmp, path)  # atomic; loses cleanly to a worker that got there first
        except FileExistsError:
            pass
        finally:
            tmp.unlink()
    return path.read_text().strip().encode()


def _mac(purpose: str, payload: str) -> str:
    # One key per purpose, so a value signed for one use (an OIDC state) is never accepted as another (a session).
    key = hmac.new(session_secret(), purpose.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(hmac.new(key, payload.encode(), hashlib.sha256).digest()).decode().rstrip("=")


def _seal(purpose: str, claims: dict) -> str:
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"{payload}.{_mac(purpose, payload)}"


def _open(purpose: str, token: str | None) -> dict | None:
    if not token or "." not in token:
        return None
    payload, sig = token.rsplit(".", 1)
    if not hmac.compare_digest(sig.encode(), _mac(purpose, payload).encode()):
        return None
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except ValueError:
        return None
    if not isinstance(claims, dict) or not isinstance(claims.get("exp"), int) or claims["exp"] < time.time():
        return None
    return claims


def make_token(user_id: int, version: int, now: float | None = None) -> str:
    return _seal("session", {"u": user_id, "v": version, "exp": int((now or time.time()) + SESSION_TTL)})


def session_claims(token: str | None) -> tuple[int, int] | None:
    claims = _open("session", token)
    if claims is None or not isinstance(claims.get("u"), int) or not isinstance(claims.get("v"), int):
        return None
    return claims["u"], claims["v"]


def is_https(request: Request) -> bool:
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    return proto.split(",")[0].strip().lower() == "https"


# ----------------------------------------------------------------------------------- request checks
def _split_host(value: str) -> tuple[str, int | None]:
    try:
        u = urlsplit("//" + value.strip())
        return (u.hostname or "", u.port)
    except ValueError:
        return ("", None)


def host_allowed(host_header: str, allowed: str) -> bool:
    patterns = [p.strip().lower() for p in allowed.split(",") if p.strip()]
    if "*" in patterns:
        return True
    host, _ = _split_host(host_header)
    if host in LOCAL_HOSTS:
        return True
    return any(host == p or (p.startswith("*.") and host.endswith(p[1:])) for p in patterns)


def origin_allowed(request: Request) -> bool:
    """Same-origin (or the Vite dev server) for browser requests; non-browser clients send neither header."""
    origin = request.headers.get("origin") or request.headers.get("referer")
    if origin is None:
        return True
    try:
        o = urlsplit(origin)
        o_port = o.port
    except ValueError:
        return False
    if o.scheme not in ("http", "https") or not o.hostname:
        return False
    if f"{o.scheme}://{o.netloc}" in DEV_ORIGINS:
        return True
    default = 443 if o.scheme == "https" else 80
    for value in (request.headers.get("host"), request.headers.get("x-forwarded-host")):
        if not value:
            continue
        host, port = _split_host(value.split(",")[0])
        if host == o.hostname and (port or default) == (o_port or default):
            return True
    return False


def body_type_problem(request: Request) -> str | None:
    """A body without a JSON (or, for uploads, multipart) type could come from a cross-site form or no-cors fetch."""
    has_body = "transfer-encoding" in request.headers or request.headers.get("content-length", "0") not in ("", "0")
    if not has_body:
        return None
    ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if request.url.path == UPLOAD_PATH or AVATAR_UPLOAD.fullmatch(request.url.path):
        return None if ctype == "multipart/form-data" else "Uploads must be sent as multipart/form-data."
    return None if ctype == "application/json" else "Send the request body as application/json."


async def check_request(request: Request) -> JSONResponse | None:
    """Refuse the request, or return None and leave the signed-in user (if any) in request.state.user."""
    s = get_settings()
    path = request.url.path
    if path == "/health":
        return None
    if not host_allowed(request.headers.get("host", ""), s.allowed_hosts):
        return JSONResponse({"detail": "Host not allowed. Add it to RAINFOREST_ALLOWED_HOSTS."}, 400)
    if request.method in UNSAFE_METHODS:
        if not origin_allowed(request):
            return JSONResponse({"detail": "Cross-origin request refused."}, 403)
        # The check nginx's auth_request makes never carries a body.
        problem = None if path == CHECK_PATH else body_type_problem(request)
        if problem:
            return JSONResponse({"detail": problem}, 415)
        # Checked before the body is read: the route only sees it once it's spooled to disk.
        if AVATAR_UPLOAD.fullmatch(path) and int(request.headers.get("content-length") or 0) > AVATAR_MAX_BODY:
            return JSONResponse({"detail": "Pictures can be 5 MB at most."}, 413)
    if path in OPEN_PATHS or (path.startswith(AVATAR_PREFIX) and request.method in ("GET", "HEAD")):
        return None
    problem = config_problem()
    if problem:
        return JSONResponse({"detail": problem}, 503)
    ids = session_claims(request.cookies.get(COOKIE))
    user = await run_in_threadpool(users.session_user, *ids) if ids else None
    if user is None:
        return JSONResponse({"detail": "Sign in required."}, 401)
    if viewer_refused(user, request.method, path, request.headers.get("x-original-method"),
                      request.headers.get("x-original-uri")):
        return JSONResponse({"detail": VIEWER_REFUSED}, 403)
    request.state.user = user
    return None


class SecurityMiddleware:
    """Host, origin, content-type and session checks, ahead of routing so downloads are covered too."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            denied = await check_request(Request(scope))
            if denied is not None:
                await denied(scope, receive, send)
                return
        await self.app(scope, receive, send)


# -------------------------------------------------------------------------------------- dependencies
def current_user(request: Request) -> dict:
    """The signed-in user, {id, name, email, role}. The middleware has already refused requests without one."""
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(401, "Sign in required.")
    return user


def require_lead(user: Annotated[dict, Depends(current_user)]) -> dict:
    if user["role"] != "lead":
        raise HTTPException(403, "Only a lead can do this.")
    return user


CurrentUser = Annotated[dict, Depends(current_user)]
LeadUser = Annotated[dict, Depends(require_lead)]


# ------------------------------------------------------------------------------------------- routes
router = APIRouter(prefix="/api/auth", tags=["auth"])


def _set_cookie(response: Response, request: Request, name: str, value: str, max_age: int, *,
                samesite: str = "strict", path: str = "/") -> None:
    response.set_cookie(name, value, max_age=max_age, path=path, httponly=True, samesite=samesite,
                        secure=is_https(request))


def _sign_in(response: Response, request: Request, user: dict) -> None:
    _set_cookie(response, request, COOKIE, make_token(user["id"], user["session_version"]), SESSION_TTL)


def safe_next(value: str | None) -> str:
    """Only same-app paths, so the sign-in redirect can't be pointed at another site."""
    if not value or not value.startswith("/") or value.startswith("//") or "\\" in value or value.startswith("/login"):
        return "/"
    return value


def _require_mode(mode: str) -> None:
    if get_settings().auth_mode != mode:
        raise HTTPException(404, f"Not available: sign-in mode is {get_settings().auth_mode!r}")


@router.get("/config")
def auth_config() -> dict:
    """What the sign-in page needs before anyone is signed in."""
    s = get_settings()
    return {"mode": s.auth_mode, "provider_name": s.oidc_provider_name if s.auth_mode == "oidc" else None,
            "lead_group": s.lead_group, "member_group": s.member_group, "problem": config_problem(),
            "accounts_from_env": users.managed_by_env()}


def _rename_refused() -> str | None:
    """Why nobody can rename themselves here, or None: names come from .env or the identity provider otherwise."""
    if users.managed_by_env():
        return users.ENV_MANAGED
    if get_settings().auth_mode == "oidc":
        return "Your name comes from the identity provider; change it there."
    return None


@router.get("/me")
def me(user: CurrentUser) -> dict:
    return {**user, "mode": get_settings().auth_mode, "can_rename": _rename_refused() is None}


class ProfileIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)


@router.patch("/me")
def update_me(body: ProfileIn, user: CurrentUser) -> dict:
    """Change your own display name (pick mode without RAINFOREST_USERS; anyone, Viewers included)."""
    if problem := _rename_refused():
        raise HTTPException(409, problem)
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "A name can't be blank.")
    users.update(user["id"], {"name": name})
    return me({**user, "name": name})


@router.api_route("/check", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"], status_code=204)
def check() -> Response:
    """For nginx auth_request: 204 with a valid session; the middleware answers 401 (or 403 cross-origin) otherwise."""
    return Response(status_code=204)


@router.post("/logout")
def logout(request: Request) -> JSONResponse:
    """Clear the session cookie. In OIDC mode also returns the provider's sign-out URL for the browser to visit."""
    response = JSONResponse({"logout_url": _provider_logout_url(request)})
    _set_cookie(response, request, COOKIE, "", 0)
    return response


def _provider_logout_url(request: Request) -> str | None:
    if get_settings().auth_mode != "oidc" or config_problem():
        return None
    redirect = urlsplit(redirect_uri(request))
    return oidc.end_session_url(f"{redirect.scheme}://{redirect.netloc}/login")


@router.get("/avatars/{user_id}/{name}")
def avatar(user_id: str, name: str) -> FileResponse:
    from . import avatars

    path = avatars.path_for(int(user_id), name) if user_id.isdigit() else None
    if path is None:
        raise HTTPException(404, "No such picture")
    # The name is the picture's hash, so a changed picture has a new URL and this one can be cached for good.
    return FileResponse(path, media_type="image/webp", headers={
        "Cache-Control": "public, max-age=31536000, immutable", "X-Content-Type-Options": "nosniff",
        "Content-Security-Policy": "default-src 'none'; sandbox"})


# Pick mode ---------------------------------------------------------------------------------------
class PickIn(BaseModel):
    user_id: int


@router.get("/users")
def pick_list() -> list[dict]:
    """Active users for the pick-a-user sign-in page (pick mode only)."""
    _require_mode("pick")
    return users.pickable()


@router.post("/pick")
def pick(body: PickIn, request: Request, response: Response) -> dict:
    _require_mode("pick")
    user = users.get(body.user_id)
    if user is None or not user["active"]:
        raise HTTPException(404, "No active user with that id")
    users.touch(user["id"])
    _sign_in(response, request, user)
    shown = {k: user.get(k) for k in ("id", "name", "email", "role", "key", "avatar_url")}
    return shown | {"mode": "pick", "can_rename": _rename_refused() is None}


# OIDC mode ---------------------------------------------------------------------------------------
def redirect_uri(request: Request) -> str:
    s = get_settings()
    if s.oidc_redirect_url:
        return s.oidc_redirect_url
    host = (request.headers.get("x-forwarded-host") or request.headers.get("host", "")).split(",")[0].strip()
    return f"{'https' if is_https(request) else 'http'}://{host}{OIDC_PATH}/callback"


def _derived(purpose: str, state: str) -> str:
    # The PKCE verifier and nonce follow from the state and the server's secret, so neither is stored anywhere.
    return _mac(purpose, state)


def _to_login(error: str) -> RedirectResponse:
    return RedirectResponse(f"/login?error={error}", status_code=303)


@router.get("/oidc/login")
def oidc_login(request: Request, next: str | None = None) -> RedirectResponse:
    _require_mode("oidc")
    if problem := config_problem():
        raise HTTPException(503, problem)
    state = secrets.token_urlsafe(32)
    try:
        url = oidc.authorization_url(state, _derived("nonce", state), _derived("pkce", state), redirect_uri(request))
    except oidc.OIDCError as exc:
        raise HTTPException(502, str(exc)) from exc
    response = RedirectResponse(url, status_code=303)
    # Lax, not Strict: the browser has to send it on the provider's cross-site redirect back to the callback.
    _set_cookie(response, request, STATE_COOKIE,
                _seal("oidc-state", {"s": state, "n": safe_next(next), "exp": int(time.time() + STATE_TTL)}),
                STATE_TTL, samesite="lax", path=OIDC_PATH)
    return response


@router.get("/oidc/callback")
def oidc_callback(request: Request, state: str | None = None, code: str | None = None,
                  error: str | None = None) -> RedirectResponse:
    _require_mode("oidc")
    pending = _open("oidc-state", request.cookies.get(STATE_COOKIE))
    if pending is None or not state or not hmac.compare_digest(str(pending.get("s")).encode(), state.encode()):
        response = _to_login("state")
    elif error or not code:
        log.warning("OIDC provider returned an error: %s", error)
        response = _to_login("provider")
    else:
        try:
            tokens = oidc.exchange_code(code, _derived("pkce", state), redirect_uri(request))
            claims = oidc.verify_id_token(tokens["id_token"], _derived("nonce", state))
            name, email = oidc.profile_from_claims(claims)
            user = users.sign_in_oidc(claims["sub"], name, email, oidc.role_from_claims(claims))
        except oidc.OIDCError as exc:
            log.warning("OIDC sign-in failed: %s", exc)
            response = _to_login("token")
        except users.SignInRefused as exc:
            response = RedirectResponse(f"/forbidden?reason={exc.reason}", status_code=303)
        else:
            response = RedirectResponse(safe_next(pending.get("n")), status_code=303)
            _sign_in(response, request, user)
    _set_cookie(response, request, STATE_COOKIE, "", 0, samesite="lax", path=OIDC_PATH)
    return response
