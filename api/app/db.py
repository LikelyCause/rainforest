"""Postgres access: a shared connection pool, schema bootstrap, small query helpers."""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from .config import Settings, get_settings

log = logging.getLogger(__name__)

_pool: ConnectionPool | None = None
_ro_pool: ConnectionPool | None = None
_SCHEMA_LOCK_KEY = 7_311_204  # arbitrary advisory-lock id for schema bootstrap
# Hold people's own assessments (or say who assesses what) while they are blind: never readable from the console.
PRIVATE_TABLES = ("dataset_assessors", "assessments", "assessment_settings", "assessment_batches", "review_comments")


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        s = get_settings()
        _pool = ConnectionPool(
            s.database_url,
            min_size=1,
            max_size=s.db_pool_max,
            kwargs={"row_factory": dict_row},
            open=True,
        )
    return _pool


def close_pool() -> None:
    global _pool, _ro_pool
    if _pool is not None:
        _pool.close()
        _pool = None
    if _ro_pool is not None:
        _ro_pool.close()
        _ro_pool = None


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    """Pooled connection; commits on success, rolls back on error."""
    with get_pool().connection() as conn:
        yield conn


def fetch_all(query: str | sql.Composable, params: Any = None) -> list[dict]:
    with connection() as conn:
        return conn.execute(query, params).fetchall()


def fetch_one(query: str | sql.Composable, params: Any = None) -> dict | None:
    with connection() as conn:
        return conn.execute(query, params).fetchone()


def execute(query: str | sql.Composable, params: Any = None) -> None:
    with connection() as conn:
        conn.execute(query, params)


def jsonb(value: Any) -> Jsonb:
    return Jsonb(json_safe(value))


def json_safe(value: Any) -> Any:
    """What JSONB can store: NaN and ±inf become null (a metric that couldn't be computed), and NUL characters,
    which Postgres text can't hold, are dropped."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value.replace("\x00", "") if "\x00" in value else value
    if isinstance(value, dict):
        return {json_safe(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    return value


def readonly_conninfo() -> str:
    s = get_settings()
    return make_conninfo(s.database_url, user=s.readonly_role, password=s.readonly_password)


def _reset_readonly(conn: psycopg.Connection) -> None:
    # A console query can take session-level advisory locks or set session GUCs; none may outlive it.
    conn.execute("DISCARD ALL")


def get_readonly_pool() -> ConnectionPool:
    """Small pool for the SQL console's read-only role; its size also caps concurrent console queries."""
    global _ro_pool
    if _ro_pool is None:
        s = get_settings()
        _ro_pool = ConnectionPool(
            readonly_conninfo(),
            min_size=1,
            max_size=s.sql_pool_max,
            kwargs={"autocommit": True, "prepare_threshold": None, "connect_timeout": 5},
            reset=_reset_readonly,
            open=True,
        )
    return _ro_pool


MIGRATIONS_DIR = Path(__file__).parent / "migrations"
VIEWS_SQL = Path(__file__).parent / "views.sql"


def apply_migrations(conn: psycopg.Connection) -> list[str]:
    """Apply migrations/NNN_*.sql in order, each exactly once, recording them in meta.schema_migrations."""
    conn.execute("CREATE SCHEMA IF NOT EXISTS meta")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS meta.schema_migrations (
               version    TEXT PRIMARY KEY,
               applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"""
    )
    applied = {r["version"] for r in conn.execute("SELECT version FROM meta.schema_migrations").fetchall()}
    done: list[str] = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        if path.stem in applied:
            continue
        conn.execute(path.read_text())
        conn.execute("INSERT INTO meta.schema_migrations (version) VALUES (%s)", (path.stem,))
        done.append(path.stem)
    conn.execute(VIEWS_SQL.read_text())
    return done


def role_exists(conn: psycopg.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (name,)).fetchone() is not None


def ensure_readonly_role(conn: psycopg.Connection, s: Settings) -> None:
    """Create the SQL console's role and set its password and read-only limits. Needs CREATEROLE."""
    role = sql.Identifier(s.readonly_role)
    if not role_exists(conn, s.readonly_role):
        conn.execute(sql.SQL("CREATE ROLE {} LOGIN").format(role))
    conn.execute(sql.SQL("ALTER ROLE {} WITH LOGIN PASSWORD {}").format(role, sql.Literal(s.readonly_password)))
    conn.execute(sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(role))
    conn.execute(sql.SQL("ALTER ROLE {} SET statement_timeout = {}").format(role, sql.Literal(f"{s.sql_timeout_ms}ms")))


def grant_readonly(conn: psycopg.Connection, s: Settings) -> None:
    """What the console's role may read. Runs whoever created the role: these tables are the app's to protect."""
    role = sql.Identifier(s.readonly_role)
    conn.execute(sql.SQL("GRANT USAGE ON SCHEMA meta, raw TO {}").format(role))
    conn.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA meta, raw TO {}").format(role))
    conn.execute(sql.SQL("ALTER DEFAULT PRIVILEGES IN SCHEMA meta, raw GRANT SELECT ON TABLES TO {}").format(role))
    # The two grants above cover these too (raw tables appear at runtime, so the default stays).
    conn.execute(
        sql.SQL("REVOKE ALL ON {} FROM {}").format(
            sql.SQL(", ").join(sql.Identifier("meta", t) for t in PRIVATE_TABLES), role
        )
    )


def bootstrap() -> None:
    """Migrate the schema and (re)configure the read-only role. Safe to run from several workers."""
    from . import experiments, recipes, users

    s = get_settings()
    if s.auth_mode == "pick" and s.users.strip():
        try:
            users.parse_accounts(s.users)  # before any write, so a bad list changes nothing
        except ValueError as exc:
            raise SystemExit(f"RAINFOREST_USERS: {exc}") from None
    with connection() as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (_SCHEMA_LOCK_KEY,))
        try:
            applied = apply_migrations(conn)
            recipes.backfill_snapshots(conn)
            conn.execute("SELECT meta.import_orphan_experiments()")
            experiments.backfill_copies(conn)
            if s.auth_mode == "pick" and s.users.strip():
                try:
                    done = users.sync_accounts(conn, users.parse_accounts(s.users))
                except ValueError as exc:
                    conn.rollback()
                    raise SystemExit(f"RAINFOREST_USERS: {exc}") from None
                if any(done.values()):
                    log.info("accounts from RAINFOREST_USERS: %s", done)
            else:
                # Someone has to be able to sign in (pick mode) and manage users on a fresh or emptied install.
                conn.execute(
                    "INSERT INTO meta.users (name, role) SELECT 'Admin', 'lead' "
                    "WHERE NOT EXISTS (SELECT 1 FROM meta.users)"
                )
            if s.managed_roles:
                ensure_readonly_role(conn, s)
            elif not role_exists(conn, s.readonly_role):
                quoted = sql.Identifier(s.readonly_role).as_string(conn)
                raise SystemExit(
                    f"READONLY_ROLE {s.readonly_role} does not exist and RAINFOREST_MANAGED_ROLES is false; "
                    f"create it: CREATE ROLE {quoted} LOGIN PASSWORD '…'"
                )
            grant_readonly(conn, s)
            conn.commit()
        except BaseException:
            conn.rollback()  # an aborted transaction can't run the unlock below
            raise
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (_SCHEMA_LOCK_KEY,))
            conn.commit()
    log.info("schema ready (migrations applied this start: %s)", ", ".join(applied) or "none")
