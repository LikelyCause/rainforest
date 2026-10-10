"""The team: everyone signed in can list it; leads add, rename, re-role and deactivate users. In OIDC mode the
provider owns names, emails and roles, so leads can only deactivate or reactivate."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import users
from ..auth import CurrentUser, LeadUser
from ..config import get_settings
from ..schemas import UserIn, UserPatch

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("")
def list_users(_: CurrentUser) -> list[dict]:
    return users.list_all()


@router.post("", status_code=201)
def create_user(body: UserIn, _: LeadUser) -> dict:
    if get_settings().auth_mode == "oidc":
        raise HTTPException(409, "Users come from the identity provider; they are added when they first sign in.")
    return users.create(body.name.strip(), (body.email or "").strip() or None, body.role)


@router.patch("/{user_id}")
def update_user(user_id: int, body: UserPatch, _: LeadUser) -> dict:
    changes = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None or k == "email"}
    if "name" in changes:
        changes["name"] = changes["name"].strip()
    if "email" in changes:
        changes["email"] = (changes["email"] or "").strip() or None
    if get_settings().auth_mode == "oidc" and set(changes) - {"active"}:
        raise HTTPException(409, "Names, emails and roles come from the identity provider; "
                                 "here a user can only be deactivated or reactivated.")
    return users.update(user_id, changes)
