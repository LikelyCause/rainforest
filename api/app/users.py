"""Users: lookups for the session check, OIDC provisioning, and the edits leads make on the Team page."""
from __future__ import annotations

from fastapi import HTTPException

from .db import connection, fetch_all, fetch_one

PUBLIC_COLS = "id, name, email, role"
LIST_COLS = "id, name, email, role, active, oidc_subject IS NOT NULL AS sso, created_at, last_seen_at"


class SignInRefused(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason  # "no_group" | "deactivated"


def session_user(user_id: int, version: int) -> dict | None:
    """The active user a session cookie names, if the cookie is still current for them."""
    row = fetch_one(
        f"""SELECT {PUBLIC_COLS}, (last_seen_at IS NULL OR last_seen_at < now() - interval '5 minutes') AS stale
              FROM meta.users WHERE id = %s AND active AND session_version = %s""",
        (user_id, version),
    )
    if row is None:
        return None
    if row.pop("stale"):
        touch(row["id"])
    return row


def touch(user_id: int) -> None:
    with connection() as conn:
        conn.execute("UPDATE meta.users SET last_seen_at = now() WHERE id = %s", (user_id,))


def get(user_id: int) -> dict | None:
    return fetch_one(f"SELECT {LIST_COLS}, session_version FROM meta.users WHERE id = %s", (user_id,))


def pickable() -> list[dict]:
    return fetch_all("SELECT id, name, role FROM meta.users WHERE active ORDER BY lower(name), id")


def list_all() -> list[dict]:
    return fetch_all(f"SELECT {LIST_COLS} FROM meta.users ORDER BY active DESC, lower(name), id")


def create(name: str, email: str | None, role: str) -> dict:
    return fetch_one(f"INSERT INTO meta.users (name, email, role) VALUES (%s, %s, %s) RETURNING {LIST_COLS}",
                     (name, email, role))


def update(user_id: int, changes: dict) -> dict:
    """Apply name / email / role / active. Deactivating or changing the role signs the user out everywhere, and the
    last active lead can't be demoted or deactivated (nobody could manage users after that)."""
    with connection() as conn:
        conn.execute("LOCK TABLE meta.users IN SHARE ROW EXCLUSIVE MODE")
        row = conn.execute("SELECT * FROM meta.users WHERE id = %s", (user_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "User not found")
        new = {**row, **changes}
        if row["role"] == "lead" and row["active"] and (new["role"] != "lead" or not new["active"]):
            others = conn.execute("SELECT count(*) AS n FROM meta.users WHERE role = 'lead' AND active AND id <> %s",
                                  (user_id,)).fetchone()["n"]
            if others == 0:
                raise HTTPException(409, "This is the only active lead. Make someone else a lead first.")
        bump = new["role"] != row["role"] or (row["active"] and not new["active"])
        return conn.execute(
            f"""UPDATE meta.users SET name = %s, email = %s, role = %s, active = %s,
                       session_version = session_version + %s
                 WHERE id = %s RETURNING {LIST_COLS}""",
            (new["name"], new["email"], new["role"], new["active"], int(bump), user_id),
        ).fetchone()


def sign_in_oidc(subject: str, name: str, email: str | None, role: str | None) -> dict:
    """Find or create the user for an ID token's subject and refresh their name, email and role from it.
    `role` None (in neither group) refuses the sign-in and ends any sessions the user still has."""
    with connection() as conn:
        row = conn.execute("SELECT id, active, role FROM meta.users WHERE oidc_subject = %s FOR UPDATE",
                           (subject,)).fetchone()
        if row is None:
            if role is None:
                raise SignInRefused("no_group")
            return conn.execute(
                f"""INSERT INTO meta.users (name, email, role, oidc_subject, last_seen_at)
                    VALUES (%s, %s, %s, %s, now()) RETURNING {PUBLIC_COLS}, session_version""",
                (name, email, role, subject),
            ).fetchone()
        if not row["active"]:
            raise SignInRefused("deactivated")
        if role is None:
            conn.execute("UPDATE meta.users SET session_version = session_version + 1 WHERE id = %s", (row["id"],))
            conn.commit()
            raise SignInRefused("no_group")
        return conn.execute(
            f"""UPDATE meta.users SET name = %s, email = %s, role = %s, last_seen_at = now(),
                       session_version = session_version + %s
                 WHERE id = %s RETURNING {PUBLIC_COLS}, session_version""",
            (name, email, role, int(role != row["role"]), row["id"]),
        ).fetchone()
