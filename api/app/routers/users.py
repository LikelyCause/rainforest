"""The team: everyone signed in can list it; leads add, rename, re-role and deactivate users. In OIDC mode the
provider owns names, emails and roles, so leads can only deactivate or reactivate."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from .. import avatars, users
from ..auth import CurrentUser, LeadUser
from ..config import get_settings
from ..schemas import UserIn, UserPatch

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("")
def list_users(_: CurrentUser) -> list[dict]:
    return users.list_all()


ENV_MANAGED = "Accounts come from RAINFOREST_USERS in .env; change them there and restart."


def _may_change_picture(user: dict, user_id: int) -> None:
    if user["id"] != user_id and user["role"] != "lead":
        raise HTTPException(403, "You can change your own picture; a lead can change anyone's.")
    if users.get(user_id) is None:
        raise HTTPException(404, "User not found")


@router.put("/{user_id}/avatar")
def set_avatar(user_id: int, request: Request, user: CurrentUser, file: UploadFile = File(...)) -> dict:
    """Upload a JPEG, PNG or WebP (5 MB at most) as the account's picture: you for yourself, a lead for anyone."""
    _may_change_picture(user, user_id)
    if int(request.headers.get("content-length") or 0) > avatars.MAX_BYTES + 64 * 1024:
        raise HTTPException(413, "Pictures can be 5 MB at most.")
    data = file.file.read(avatars.MAX_BYTES + 1)
    if len(data) > avatars.MAX_BYTES:
        raise HTTPException(413, "Pictures can be 5 MB at most.")
    avatars.store(user_id, data)
    return users.get(user_id)


@router.delete("/{user_id}/avatar")
def remove_avatar(user_id: int, user: CurrentUser) -> dict:
    _may_change_picture(user, user_id)
    avatars.remove(user_id)
    return users.get(user_id)


@router.post("", status_code=201)
def create_user(body: UserIn, _: LeadUser) -> dict:
    if users.managed_by_env():
        raise HTTPException(409, ENV_MANAGED)
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
    if users.managed_by_env():
        raise HTTPException(409, ENV_MANAGED)
    if get_settings().auth_mode == "oidc" and set(changes) - {"active"}:
        raise HTTPException(409, "Names, emails and roles come from the identity provider; "
                                 "here a user can only be deactivated or reactivated.")
    return users.update(user_id, changes)
