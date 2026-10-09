"""Postgres access: a shared connection pool, schema bootstrap, small query helpers."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from .config import get_settings

log = logging.getLogger(__name__)

_pool: ConnectionPool | None = None
_SCHEMA_LOCK_KEY = 7_311_204  # arbitrary advisory-lock id for schema bootstrap


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
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


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
    return Jsonb(value)


def readonly_conninfo() -> str:
    s = get_settings()
    return make_conninfo(s.database_url, user=s.readonly_role, password=s.readonly_password)


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


def bootstrap() -> None:
    """Migrate the schema and (re)configure the read-only role. Safe to run from several workers."""
    s = get_settings()
    with connection() as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (_SCHEMA_LOCK_KEY,))
        try:
            applied = apply_migrations(conn)
            role = sql.Identifier(s.readonly_role)
            exists = conn.execute(
                "SELECT 1 FROM pg_roles WHERE rolname = %s", (s.readonly_role,)
            ).fetchone()
            if not exists:
                conn.execute(sql.SQL("CREATE ROLE {} LOGIN").format(role))
            conn.execute(
                sql.SQL("ALTER ROLE {} WITH LOGIN PASSWORD {}").format(
                    role, sql.Literal(s.readonly_password)
                )
            )
            conn.execute(sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(role))
            conn.execute(
                sql.SQL("ALTER ROLE {} SET statement_timeout = {}").format(
                    role, sql.Literal(f"{s.sql_timeout_ms}ms")
                )
            )
            conn.execute(sql.SQL("GRANT USAGE ON SCHEMA meta, raw TO {}").format(role))
            conn.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA meta, raw TO {}").format(role))
            conn.execute(
                sql.SQL("ALTER DEFAULT PRIVILEGES IN SCHEMA meta, raw GRANT SELECT ON TABLES TO {}").format(role)
            )
            conn.commit()
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (_SCHEMA_LOCK_KEY,))
            conn.commit()
    log.info("schema ready (migrations applied this start: %s)", ", ".join(applied) or "none")


def mark_interrupted_jobs() -> None:
    """Jobs run in-process; anything still 'running' at startup was killed by a restart."""
    with connection() as conn:
        # A profiled dataset only loses the associations of targets chosen later; it stays usable.
        conn.execute(
            """UPDATE meta.datasets
                  SET status = 'ready', progress = 1, status_message = 'Ready', updated_at = now(),
                      warnings = array_append(warnings, 'Associations for the new target(s) were interrupted '
                                              'by an API restart; choose the targets again to retry.')
                WHERE status = 'correlating' AND profiled_at IS NOT NULL
                  AND updated_at < now() - interval '2 minutes'"""
        )
        conn.execute(
            """UPDATE meta.datasets
                  SET status = 'failed', error = 'Interrupted by an API restart; re-upload or reprocess.',
                      updated_at = now()
                WHERE status IN ('pending', 'ingesting', 'profiling', 'correlating')
                  AND updated_at < now() - interval '2 minutes'"""
        )
        for table in ("exports", "model_runs", "analyses"):
            conn.execute(
                sql.SQL(
                    """UPDATE meta.{} SET status = 'failed', error = 'Interrupted by an API restart.',
                              finished_at = now()
                        WHERE status IN ('pending', 'running') AND updated_at < now() - interval '2 minutes'"""
                ).format(sql.Identifier(table))
            )
