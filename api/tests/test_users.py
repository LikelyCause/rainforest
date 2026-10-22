"""Users in pick mode against a real Postgres (RAINFOREST_TEST_DATABASE_URL): the Team API, role gates, sessions
ending on deactivation or a role change, the migration's first lead, and who-did-what columns on every write."""

import os
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from .conftest import set_phase

DB_URL = os.environ.get("RAINFOREST_TEST_DATABASE_URL")


@pytest.fixture(scope="module")
def app_(tmp_path_factory, login_as):
    if not DB_URL:
        pytest.skip("set RAINFOREST_TEST_DATABASE_URL to run the API tests")
    from app import db
    from app.config import get_settings

    mp = pytest.MonkeyPatch()
    mp.setenv("DATABASE_URL", DB_URL)
    mp.setenv("DATA_DIR", str(tmp_path_factory.mktemp("data")))
    get_settings.cache_clear()
    db.close_pool()
    from app.main import app

    with TestClient(app):
        yield app
    db.close_pool()
    mp.undo()
    get_settings.cache_clear()


@pytest.fixture
def lead(app_, login_as):
    return login_as(TestClient(app_))


@pytest.fixture
def member(app_, login_as):
    return login_as(TestClient(app_), "member")


def _new_user(lead, **body):
    r = lead.post("/api/users", json={"name": f"User {time.monotonic_ns()}", **body})
    assert r.status_code == 201, r.text
    return r.json()


def _as(app_, user_id):
    c = TestClient(app_)
    assert c.post("/api/auth/pick", json={"user_id": user_id}).status_code == 200
    return c


def test_migration_made_admin_the_first_lead(app_):
    from app.db import fetch_one

    first = fetch_one("SELECT name, role, active FROM meta.users ORDER BY id LIMIT 1")
    assert first == {"name": "Admin", "role": "lead", "active": True}


def test_pick_list_is_public_and_minimal(app_, lead):
    u = _new_user(lead, email="private@example.com")
    listed = TestClient(app_).get("/api/auth/users").json()
    assert {"id": u["id"], "name": u["name"], "role": "member", "key": None, "avatar_url": None} in listed
    assert all(set(x) == {"id", "name", "role", "key", "avatar_url"} for x in listed)


def test_lead_manages_the_team(app_, lead):
    u = _new_user(lead, email=" a@example.com ", role="member")
    assert u["email"] == "a@example.com" and u["role"] == "member" and u["active"] and not u["sso"]
    r = lead.patch(f"/api/users/{u['id']}", json={"name": "  Renamed  ", "email": ""})
    assert r.status_code == 200 and r.json()["name"] == "Renamed" and r.json()["email"] is None
    listed = {x["id"]: x for x in lead.get("/api/users").json()}
    assert listed[u["id"]]["name"] == "Renamed"
    assert lead.patch("/api/users/999999", json={"name": "x"}).status_code == 404
    assert lead.post("/api/users", json={"name": ""}).status_code == 422
    assert lead.patch(f"/api/users/{u['id']}", json={"role": "owner"}).status_code == 422


def test_members_can_list_but_not_manage(app_, member):
    assert member.get("/api/users").status_code == 200
    assert member.post("/api/users", json={"name": "x"}).status_code == 403
    assert member.patch(f"/api/users/{member.user['id']}", json={"role": "lead"}).status_code == 403


def test_deactivation_and_role_change_end_sessions(app_, lead):
    u = _new_user(lead)
    c = _as(app_, u["id"])
    assert lead.patch(f"/api/users/{u['id']}", json={"name": "Still here"}).status_code == 200
    assert c.get("/api/auth/me").json()["name"] == "Still here", "a rename keeps the session"

    assert lead.patch(f"/api/users/{u['id']}", json={"role": "lead"}).status_code == 200
    assert c.get("/api/auth/me").status_code == 401
    c = _as(app_, u["id"])
    assert c.get("/api/auth/me").json()["role"] == "lead"

    assert lead.patch(f"/api/users/{u['id']}", json={"active": False}).status_code == 200
    assert c.get("/api/auth/me").status_code == 401
    assert c.post("/api/auth/pick", json={"user_id": u["id"]}).status_code == 404
    assert u["id"] not in [x["id"] for x in TestClient(app_).get("/api/auth/users").json()]


def test_the_last_active_lead_stays(app_, lead):
    from app.db import fetch_all

    others = [
        r["id"]
        for r in fetch_all("SELECT id FROM meta.users WHERE role = 'lead' AND active AND id <> %s", (lead.user["id"],))
    ]
    for uid in others:
        assert lead.patch(f"/api/users/{uid}", json={"active": False}).status_code == 200
    for change in ({"active": False}, {"role": "member"}):
        r = lead.patch(f"/api/users/{lead.user['id']}", json=change)
        assert r.status_code == 409 and "only active lead" in r.json()["detail"]
    for uid in others:
        assert lead.patch(f"/api/users/{uid}", json={"active": True}).status_code == 200


def _csv(n=400) -> str:
    rng = np.random.default_rng(3)
    return "x,z,c,y\n" + "".join(
        f"{a:.3f},{b:.3f},{'pqr'[i % 3]},{'ab'[i % 2]}\n"
        for i, (a, b) in enumerate(zip(rng.normal(size=n), rng.normal(size=n)))
    )


def _wait(fn, timeout=120.0):
    deadline = time.monotonic() + timeout
    while not (out := fn()):
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.3)
    return out


def test_writes_record_the_acting_user(app_, lead, member, login_as):
    from app.db import fetch_all, fetch_one

    # Team decisions in review are a lead's to set, so the writer here is a second lead.
    writer = login_as(TestClient(app_), "lead", name="Writer lead")
    uid = writer.user["id"]
    ds = writer.post(
        "/api/datasets/upload", files={"file": ("u.csv", _csv(), "text/csv")}, data={"targets": "y"}
    ).json()
    ds = _wait(lambda: (d := writer.get(f"/api/datasets/{ds['id']}").json())["status"] in ("ready", "failed") and d)
    assert ds["status"] == "ready" and ds["created_by"] == uid
    set_phase(ds["id"], "review")
    try:
        fields = {f["name"]: f for f in writer.get(f"/api/datasets/{ds['id']}/fields").json()}
        fid = fields["x"]["id"]
        cur = writer.get(f"/api/fields/{fid}").json()["decision"]
        keys = (
            "action",
            "semantic_type",
            "imputation",
            "impute_value",
            "add_missing_indicator",
            "transform",
            "scaling",
            "encoding",
            "params",
            "notes",
            "reviewed",
        )
        assert (
            writer.put(f"/api/fields/{fid}/decision", json={**{k: cur[k] for k in keys}, "action": "drop"}).status_code
            == 200
        )
        hist = writer.get(f"/api/fields/{fid}/history").json()
        assert hist[0]["user_id"] == uid and hist[0]["user_name"] == writer.user["name"]

        bulk = writer.post(
            f"/api/datasets/{ds['id']}/decisions/bulk", json={"field_ids": [fields["z"]["id"]], "reviewed": True}
        ).json()
        batch = fetch_one("SELECT user_id FROM meta.bulk_batches WHERE id = %s", (bulk["batch_id"],))
        assert batch["user_id"] == uid
        assert writer.post(f"/api/datasets/{ds['id']}/decisions/bulk/{bulk['batch_id']}/undo").status_code == 200
        assert writer.patch(f"/api/datasets/{ds['id']}", json={"target_columns": ["y", "c"]}).status_code == 200
        users = {
            r["user_id"]
            for r in fetch_all("SELECT DISTINCT user_id FROM meta.decision_history WHERE dataset_id = %s", (ds["id"],))
        }
        assert users == {uid}, "single saves, bulk, undo and target changes are all attributed"
        assert writer.patch(f"/api/datasets/{ds['id']}", json={"target_columns": ["y"]}).status_code == 200

        an = writer.post(
            f"/api/datasets/{ds['id']}/analyses", json={"kind": "importance", "params": {"target": "y", "sample": 1000}}
        ).json()
        assert an["created_by"] == uid

        set_phase(ds["id"], "finalized")
        exp = writer.post(f"/api/datasets/{ds['id']}/exports", json={}).json()
        assert exp["created_by"] == uid
        assert (
            _wait(lambda: (x := writer.get(f"/api/exports/{exp['id']}").json())["status"] in ("done", "failed") and x)[
                "status"
            ]
            == "done"
        )
        run = writer.post(
            f"/api/exports/{exp['id']}/runs", json={"model": "random_forest", "cv": 0, "params": {"n_estimators": 10}}
        ).json()
        assert run["created_by"] == uid
        assert (
            _wait(lambda: (x := writer.get(f"/api/runs/{run['id']}").json())["status"] in ("done", "failed") and x)[
                "status"
            ]
            == "done"
        )
        assert (
            _wait(lambda: (x := writer.get(f"/api/analyses/{an['id']}").json())["status"] in ("done", "failed") and x)[
                "status"
            ]
            == "done"
        )

        r = member.delete(f"/api/datasets/{ds['id']}")
        assert r.status_code == 403 and "lead" in r.json()["detail"]
    finally:
        assert lead.delete(f"/api/datasets/{ds['id']}").status_code == 200


def test_anyone_renames_themselves_in_pick_mode(app_, lead):
    viewer = _new_user(lead, role="viewer")
    c = _as(app_, viewer["id"])
    assert c.get("/api/auth/me").json()["can_rename"] is True
    r = c.patch("/api/auth/me", json={"name": "  Renamed Viewer  "})
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "Renamed Viewer" and r.json()["role"] == "viewer"
    assert "session_version" not in r.json()
    assert c.get("/api/auth/me").json()["name"] == "Renamed Viewer"  # same session, not signed out
    assert c.patch("/api/auth/me", json={"name": "   "}).status_code == 422
    assert c.patch("/api/auth/me", json={"name": "x", "role": "lead"}).json()["role"] == "viewer"  # only the name


def test_renaming_yourself_is_refused_when_names_come_from_elsewhere(app_, lead, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("RAINFOREST_USERS", "admin#1:Admin:lead")
    get_settings.cache_clear()
    try:
        assert lead.get("/api/auth/me").json()["can_rename"] is False
        r = lead.patch("/api/auth/me", json={"name": "Nope"})
        assert r.status_code == 409 and "RAINFOREST_USERS" in r.json()["detail"]
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()
