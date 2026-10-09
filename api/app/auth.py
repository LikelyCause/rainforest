"""Single-admin sign-in, the session cookie, and the checks every request passes before reaching a route."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from collections import deque
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .config import get_settings

COOKIE = "rf_session"
SESSION_TTL = 7 * 24 * 3600
OPEN_PATHS = {"/health", "/api/auth/login", "/api/auth/logout"}
UPLOAD_PATH = "/api/datasets/upload"
DEV_ORIGINS = {"http://localhost:5173", "http://127.0.0.1:5173"}
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
LOGIN_MAX_FAILURES = 10
LOGIN_WINDOW_S = 15 * 60

PLACEHOLDER_PASSWORD = "set-a-long-password-here"  # the value in .env.example
NO_PASSWORD = ("Sign-in is not configured. Set RAINFOREST_ADMIN_PASSWORD for the api service "
               "(in .env when using Docker Compose) and restart it.")


# ------------------------------------------------------------------------------------------ session
def admin_password() -> str:
    """The configured password, or "" when unset or still the .env.example placeholder."""
    password = get_settings().admin_password
    return "" if password == PLACEHOLDER_PASSWORD else password


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


def _sign(payload: str) -> str:
    s = get_settings()
    # Keyed on the credentials too, so changing the admin user or password ends every session.
    key = hmac.new(session_secret(), f"session\0{s.admin_user}\0{s.admin_password}".encode(), hashlib.sha256)
    mac = hmac.new(key.digest(), payload.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).decode().rstrip("=")


def make_token(now: float | None = None) -> str:
    claims = {"u": get_settings().admin_user, "exp": int((now or time.time()) + SESSION_TTL)}
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"{payload}.{_sign(payload)}"


def session_user(token: str | None) -> str | None:
    if not token or "." not in token:
        return None
    payload, sig = token.rsplit(".", 1)
    if not hmac.compare_digest(sig.encode(), _sign(payload).encode()):
        return None
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except ValueError:
        return None
    if not isinstance(claims, dict) or not isinstance(claims.get("exp"), int) or claims["exp"] < time.time():
        return None
    return claims.get("u") if claims.get("u") == get_settings().admin_user else None


# ------------------------------------------------------------------------------------- rate limit
class LoginLimiter:
    """Failed sign-ins per client address in a sliding window. In-process, so each worker counts its own."""

    def __init__(self, max_failures: int = LOGIN_MAX_FAILURES, window_s: float = LOGIN_WINDOW_S):
        self.max_failures, self.window_s = max_failures, window_s
        self._failures: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, ip: str, now: float) -> deque[float]:
        q = self._failures.setdefault(ip, deque())
        while q and q[0] <= now - self.window_s:
            q.popleft()
        return q

    def retry_after(self, ip: str) -> int:
        now = time.monotonic()
        with self._lock:
            q = self._recent(ip, now)
            return int(q[0] + self.window_s - now) + 1 if len(q) >= self.max_failures else 0

    def failed(self, ip: str) -> None:
        now = time.monotonic()
        with self._lock:
            if len(self._failures) > 10_000:
                for k in [k for k, q in self._failures.items() if not q or q[-1] <= now - self.window_s]:
                    del self._failures[k]
            self._recent(ip, now).append(now)

    def clear(self, ip: str | None = None) -> None:
        with self._lock:
            if ip is None:
                self._failures.clear()
            else:
                self._failures.pop(ip, None)


limiter = LoginLimiter()


def client_ip(request: Request) -> str:
    # nginx appends its peer to X-Forwarded-For; the API's own port is bound to localhost.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.rsplit(",", 1)[-1].strip()
    return request.client.host if request.client else "unknown"


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
    if request.url.path == UPLOAD_PATH:
        return None if ctype == "multipart/form-data" else "Uploads must be sent as multipart/form-data."
    return None if ctype == "application/json" else "Send the request body as application/json."


def check_request(request: Request) -> JSONResponse | None:
    s = get_settings()
    path = request.url.path
    if path == "/health":
        return None
    if not host_allowed(request.headers.get("host", ""), s.allowed_hosts):
        return JSONResponse({"detail": "Host not allowed. Add it to RAINFOREST_ALLOWED_HOSTS."}, 400)
    if request.method in UNSAFE_METHODS:
        if not origin_allowed(request):
            return JSONResponse({"detail": "Cross-origin request refused."}, 403)
        problem = body_type_problem(request)
        if problem:
            return JSONResponse({"detail": problem}, 415)
    if path in OPEN_PATHS:
        return None
    if not admin_password():
        return JSONResponse({"detail": NO_PASSWORD}, 503)
    if session_user(request.cookies.get(COOKIE)) is None:
        return JSONResponse({"detail": "Sign in required."}, 401)
    return None


class SecurityMiddleware:
    """Host, origin, content-type and session checks, ahead of routing so downloads are covered too."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            denied = check_request(Request(scope))
            if denied is not None:
                await denied(scope, receive, send)
                return
        await self.app(scope, receive, send)


# ------------------------------------------------------------------------------------------- routes
router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginIn(BaseModel):
    username: str = Field(max_length=200)
    password: str = Field(max_length=1000)


def _set_cookie(response: Response, request: Request, value: str, max_age: int) -> None:
    response.set_cookie(COOKIE, value, max_age=max_age, path="/", httponly=True, samesite="strict",
                        secure=is_https(request))


@router.post("/login")
def login(body: LoginIn, request: Request, response: Response) -> dict:
    s = get_settings()
    if not admin_password():
        raise HTTPException(503, NO_PASSWORD)
    ip = client_ip(request)
    wait = limiter.retry_after(ip)
    if wait:
        raise HTTPException(429, "Too many failed sign-ins. Try again in a few minutes.",
                            headers={"Retry-After": str(wait)})
    user_ok = hmac.compare_digest(body.username.encode(), s.admin_user.encode())
    password_ok = hmac.compare_digest(body.password.encode(), s.admin_password.encode())
    if not (user_ok and password_ok):
        limiter.failed(ip)
        raise HTTPException(401, "Wrong username or password.")
    limiter.clear(ip)
    _set_cookie(response, request, make_token(), SESSION_TTL)
    return {"user": s.admin_user}


@router.post("/logout", status_code=204)
def logout(request: Request) -> Response:
    response = Response(status_code=204)
    _set_cookie(response, request, "", 0)
    return response


@router.get("/me")
def me() -> dict:
    return {"user": get_settings().admin_user}
