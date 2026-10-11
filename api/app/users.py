"""Users: lookups for the session check, accounts defined in .env, OIDC provisioning, and the edits leads make on the
Team page."""
from __future__ import annotations

import re
from dataclasses import dataclass

from fastapi import HTTPException

from .config import get_settings
from .db import connection, fetch_all, fetch_one

PUBLIC_COLS = "id, name, email, role, key, avatar"
LIST_COLS = "id, name, email, role, active, oidc_subject IS NOT NULL AS sso, created_at, last_seen_at, key, avatar"
ROLES = ("lead", "member", "viewer")
KEY_RE = re.compile(r"[a-z0-9-]+")


@dataclass(frozen=True)
class Account:
    key: str
    claim_id: int | None
    name: str
    role: str


def parse_accounts(text: str) -> list[Account]:
    """`key[#id]:Name:role;…`, checked as a whole: a ValueError names the first problem, so a bad line changes
    nothing."""
    out: list[Account] = []
    for raw in (e.strip() for e in text.split(";")):
        if not raw:
            continue
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError(f"{raw!r} isn't key:Name:role")
        head, name, role = (p.strip() for p in parts)
        key, _, claim = head.partition("#")
        if not KEY_RE.fullmatch(key):
            raise ValueError(f"{raw!r}: the key must be lowercase letters, digits and dashes")
        if claim and not claim.isdigit():
            raise ValueError(f"{raw!r}: the part after # must be an account id")
        if not name:
            raise ValueError(f"{raw!r}: the name is empty")
        if role not in ROLES:
            raise ValueError(f"{raw!r}: the role must be one of {', '.join(ROLES)}")
        if any(a.key == key for a in out):
            raise ValueError(f"the key {key!r} appears twice")
        if claim and any(a.claim_id == int(claim) for a in out):
            raise ValueError(f"two entries claim #{claim}")
        out.append(Account(key, int(claim) if claim else None, name, role))
    if not any(a.role == "lead" for a in out):
        raise ValueError("there must be at least one lead")
    return out


ENV_MANAGED = "Accounts come from RAINFOREST_USERS in .env; change them there and restart."


def managed_by_env() -> bool:
    return get_settings().auth_mode == "pick" and bool(get_settings().users.strip())


def sync_accounts(conn, accounts: list[Account]) -> dict[str, list[str]]:
    """Make the active accounts exactly `accounts`: claim (once) the account an entry names by id, update names and
    roles (a role change signs that account out), create new ones, reactivate returning ones, and deactivate every
    other account, ending its sessions. Nothing is deleted, so history stays attached. The caller commits."""
    conn.execute("LOCK TABLE meta.users IN SHARE ROW EXCLUSIVE MODE")
    done: dict[str, list[str]] = {"created": [], "updated": [], "deactivated": [], "reactivated": []}
    for a in accounts:
        if a.claim_id is not None and conn.execute("SELECT 1 FROM meta.users WHERE key = %s", (a.key,)).fetchone() is None:
            # A claim that can't be honoured would quietly make a fresh account and deactivate the one meant, so it
            # stops the start-up instead (the caller's transaction rolls everything back).
            target = conn.execute("SELECT key FROM meta.users WHERE id = %s", (a.claim_id,)).fetchone()
            if target is None:
                raise ValueError(f"{a.key}#{a.claim_id}: there's no account #{a.claim_id}")
            if target["key"] is not None:
                raise ValueError(f"{a.key}#{a.claim_id}: account #{a.claim_id} already has the key {target['key']!r}")
            conn.execute("UPDATE meta.users SET key = %s WHERE id = %s", (a.key, a.claim_id))
        row = conn.execute("SELECT * FROM meta.users WHERE key = %s", (a.key,)).fetchone()
        if row is None:
            conn.execute("INSERT INTO meta.users (key, name, role) VALUES (%s, %s, %s)", (a.key, a.name, a.role))
            done["created"].append(a.key)
            continue
        if (row["name"], row["role"]) != (a.name, a.role):
            conn.execute("""UPDATE meta.users SET name = %s, role = %s,
                                   session_version = session_version + (role <> %s)::int WHERE id = %s""",
                         (a.name, a.role, a.role, row["id"]))
            done["updated"].append(a.key)
        if not row["active"]:
            conn.execute("UPDATE meta.users SET active = true WHERE id = %s", (row["id"],))
            done["reactivated"].append(a.key)
    gone = conn.execute(
        """UPDATE meta.users SET active = false, session_version = session_version + 1
            WHERE active AND (key IS NULL OR NOT key = ANY(%s)) RETURNING coalesce(key, name) AS k""",
        ([a.key for a in accounts],)).fetchall()
    done["deactivated"] = [r["k"] for r in gone]
    return done


def avatar_url(row: dict) -> str | None:
    return f"/api/auth/avatars/{row['id']}/{row['avatar']}.webp" if row.get("avatar") else None


def public(row: dict) -> dict:
    """A user as the app shows them: no stored picture hash, its URL instead."""
    return {**{k: v for k, v in row.items() if k != "avatar"}, "avatar_url": avatar_url(row)}


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
    return public(row)


def touch(user_id: int) -> None:
    with connection() as conn:
        conn.execute("UPDATE meta.users SET last_seen_at = now() WHERE id = %s", (user_id,))


def get(user_id: int) -> dict | None:
    row = fetch_one(f"SELECT {LIST_COLS}, session_version FROM meta.users WHERE id = %s", (user_id,))
    return public(row) if row else None


def pickable() -> list[dict]:
    """The picker's accounts: leads first, then members, then viewers, each by name."""
    return [public(r) for r in fetch_all(
        """SELECT id, name, role, key, avatar FROM meta.users WHERE active
            ORDER BY array_position(ARRAY['lead', 'member', 'viewer'], role), lower(name), id""")]


def list_all() -> list[dict]:
    return [public(r) for r in fetch_all(f"SELECT {LIST_COLS} FROM meta.users ORDER BY active DESC, lower(name), id")]


def create(name: str, email: str | None, role: str) -> dict:
    return public(fetch_one(f"INSERT INTO meta.users (name, email, role) VALUES (%s, %s, %s) RETURNING {LIST_COLS}",
                            (name, email, role)))


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
        return public(conn.execute(
            f"""UPDATE meta.users SET name = %s, email = %s, role = %s, active = %s,
                       session_version = session_version + %s
                 WHERE id = %s RETURNING {LIST_COLS}""",
            (new["name"], new["email"], new["role"], new["active"], int(bump), user_id),
        ).fetchone())


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
