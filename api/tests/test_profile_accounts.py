"""Accounts from .env, the Viewer role, and profile pictures."""
import pytest

from app.users import Account, parse_accounts

from .team_helpers import DB_URL, app_, people  # noqa: F401


# ------------------------------------------------------------------------------------------- parsing
def test_parse_accounts():
    got = parse_accounts("tanner#2:Tanner:lead; test-member:Test Member:member;test-viewer:Test Viewer:viewer")
    assert got == [Account("tanner", 2, "Tanner", "lead"), Account("test-member", None, "Test Member", "member"),
                   Account("test-viewer", None, "Test Viewer", "viewer")]


@pytest.mark.parametrize("text,needle", [
    ("a:A:lead;a:B:member", "twice"), ("a:A:lead;x:X:boss", "role"), ("a:A:lead;bad", "key:Name:role"),
    ("Tanner:Tanner:lead", "key"), ("m:M:member", "lead"), ("a#x:A:lead", "id"), ("a::lead", "name"),
])
def test_parse_refuses(text, needle):
    with pytest.raises(ValueError, match=needle):
        parse_accounts(text)


# ------------------------------------------------------------------------------------------- sync
needs_db = pytest.mark.skipif(not DB_URL, reason="set RAINFOREST_TEST_DATABASE_URL to run the API tests")


@pytest.fixture
def conn(app_):
    """A transaction rolled back after the test, so the shared test users are untouched."""
    from app.db import connection
    with connection() as c:
        yield c
        c.rollback()


def _row(conn, uid):
    return conn.execute("SELECT id, key, name, role, active, session_version FROM meta.users WHERE id = %s",
                        (uid,)).fetchone()


def _new(conn, name, role="member"):
    return conn.execute("INSERT INTO meta.users (name, role) VALUES (%s, %s) RETURNING id", (name, role)).fetchone()["id"]


@needs_db
def test_sync_claims_creates_deactivates_and_is_idempotent(conn):
    from app.users import sync_accounts
    lee, dana = _new(conn, "Lee Lead", "lead"), _new(conn, "Dana Scientist")
    v_dana = _row(conn, dana)["session_version"]
    accounts = parse_accounts(f"tanner#{lee}:Tanner:lead;test-member:Test Member:member;test-viewer:Test Viewer:viewer")
    sync_accounts(conn, accounts)
    t = _row(conn, lee)
    assert (t["key"], t["name"], t["role"], t["active"]) == ("tanner", "Tanner", "lead", True)
    d = _row(conn, dana)
    assert not d["active"] and d["session_version"] == v_dana + 1
    keys = {r["key"]: r for r in conn.execute("SELECT * FROM meta.users WHERE active").fetchall()}
    assert set(keys) == {"tanner", "test-member", "test-viewer"} and keys["test-viewer"]["role"] == "viewer"
    before = conn.execute("SELECT id, name, role, active, session_version FROM meta.users ORDER BY id").fetchall()
    assert sync_accounts(conn, accounts) == {"created": [], "updated": [], "deactivated": [], "reactivated": []}
    assert conn.execute("SELECT id, name, role, active, session_version FROM meta.users ORDER BY id").fetchall() == before


@needs_db
def test_sync_removed_key_deactivates_and_returning_key_reactivates(conn):
    from app.users import sync_accounts
    sync_accounts(conn, parse_accounts("boss:Boss:lead;tm:Test Member:member"))
    tm = conn.execute("SELECT id FROM meta.users WHERE key = 'tm'").fetchone()["id"]
    sync_accounts(conn, parse_accounts("boss:Boss:lead"))
    assert not _row(conn, tm)["active"]
    sync_accounts(conn, parse_accounts("boss:Boss:lead;tm:Test Member:member"))
    assert _row(conn, tm)["active"]


@needs_db
def test_sync_role_change_signs_out(conn):
    from app.users import sync_accounts
    sync_accounts(conn, parse_accounts("boss:Boss:lead;tm:Test Member:member"))
    tm = _row(conn, conn.execute("SELECT id FROM meta.users WHERE key = 'tm'").fetchone()["id"])
    sync_accounts(conn, parse_accounts("boss:Boss:lead;tm:Test Member:viewer"))
    after = _row(conn, tm["id"])
    assert after["role"] == "viewer" and after["session_version"] == tm["session_version"] + 1


@needs_db
def test_accounts_from_env_are_not_edited_in_the_app(people, monkeypatch):
    from app.config import get_settings
    lead = people["lead"]
    monkeypatch.setenv("RAINFOREST_USERS", "boss:Boss:lead")
    get_settings.cache_clear()
    try:
        assert lead.post("/api/users", json={"name": "X", "role": "member"}).status_code == 409
        assert lead.patch(f"/api/users/{lead.user['id']}", json={"name": "Y"}).status_code == 409
    finally:
        monkeypatch.delenv("RAINFOREST_USERS")
        get_settings.cache_clear()


@needs_db
def test_picker_lists_key_and_picture(people):
    users = people["lead"].get("/api/auth/users").json()
    assert all({"id", "name", "role", "key", "avatar_url"} <= set(u) for u in users)


# ------------------------------------------------------------------------------------------- the Viewer role
import re  # noqa: E402

from fastapi.routing import APIRoute  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from .team_helpers import fields, finalized_trees, review_all, save, start, upload  # noqa: E402

VIEWER_MAY = [("POST", r"/api/datasets/\{dataset_id\}/analyses"), ("POST", r"/api/datasets/\{dataset_id\}/fields/query"),
              ("POST", r"/api/sql"), ("POST", r"/api/auth/pick"), ("POST", r"/api/auth/logout"),
              ("PUT", r"/api/users/\{user_id\}/avatar"), ("DELETE", r"/api/users/\{user_id\}/avatar")]


@pytest.fixture(scope="module")
def viewer(app_, login_as):
    return login_as(TestClient(app_), "viewer", name="Test viewer")


@needs_db
def test_every_write_route_refuses_a_viewer(app_, viewer):
    checked = 0
    for route in app_.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in route.methods & {"POST", "PUT", "PATCH", "DELETE"}:
            allowed = any(m == method and re.fullmatch(p, route.path) for m, p in VIEWER_MAY)
            if route.path.startswith("/api/auth/"):
                continue  # sign-in routes; /check is tested on its own
            path = re.sub(r"\{[^}]+\}", "1", route.path)
            r = viewer.request(method, path)
            if allowed:
                assert r.status_code != 403 or "own picture" in r.text, (method, route.path, r.text)
            else:
                assert r.status_code == 403 and "Viewer" in r.text, (method, route.path, r.status_code, r.text)
            checked += 1
    assert checked > 20


@needs_db
def test_mlflow_writes_are_refused_for_a_viewer(people, viewer):
    assert viewer.get("/api/auth/check", headers={"X-Original-Method": "DELETE"}).status_code == 403
    assert viewer.get("/api/auth/check", headers={"X-Original-Method": "GET"}).status_code == 204
    assert people["a"].get("/api/auth/check", headers={"X-Original-Method": "DELETE"}).status_code == 204


def _x(client, ds):
    fid = fields(client, ds)["x"]["id"]
    return client.get(f"/api/fields/{fid}").json()["decision"]["action"], fid


@needs_db
def test_viewer_never_sees_an_assessing_team_decision(people, viewer):
    from app.db import connection
    lead = people["lead"]
    ds, rid = finalized_trees(lead)
    assert _x(viewer, ds)[0] == "keep"
    start(lead, ds, "tree", [lead], start="baseline", source=rid)
    fid = fields(lead, ds)["x"]["id"]
    with connection() as conn:  # the team working copy moves on while v2 is assessed
        conn.execute("UPDATE meta.field_decisions SET action = 'drop' WHERE field_id = %s", (fid,))
        conn.execute("""INSERT INTO meta.decision_history (field_id, dataset_id, family, previous, current, user_id)
                        VALUES (%s, %s, 'tree', '{}', '{}', %s)""", (fid, ds["id"], lead.user["id"]))
        conn.commit()
    assert _x(people["c"], ds)[0] == "drop"  # a member who isn't assessing still sees the working copy
    assert _x(viewer, ds)[0] == "keep"  # the viewer sees v1
    assert fields(viewer, ds)["x"]["action"] == "keep"
    assert viewer.get(f"/api/fields/{fid}/history").json() == []
    r = viewer.post(f"/api/datasets/{ds['id']}/analyses", json={"kind": "map", "params": {"data": "raw"}})
    assert r.status_code == 200, r.text
    assert r.json()["params"]["decisions"] == f"baseline:{rid}"
    assert save(lead, fid, action="drop").status_code == 200  # the lead's own assessment
    review_all(lead, ds)
    assert lead.post(f"/api/datasets/{ds['id']}/assessment/submit").status_code == 200
    assert _x(viewer, ds)[0] == "drop"  # team review: everyone sees the team decision


@needs_db
def test_viewer_sees_suggestions_on_a_dataset_with_no_finalized_baseline(people, viewer):
    lead = people["lead"]
    ds = upload(lead)
    start(lead, ds, "tree", [lead])
    fid = fields(lead, ds)["x"]["id"]
    from app.db import connection
    with connection() as conn:
        conn.execute("UPDATE meta.field_decisions SET action = 'drop', source = 'user' WHERE field_id = %s", (fid,))
        conn.commit()
    got = viewer.get(f"/api/fields/{fid}").json()
    assert got["decision"]["action"] == got["field"]["suggestion"]["action"] == "keep"
    assert got["decision"]["source"] == "suggested"


@needs_db
def test_viewers_cant_assess(people, viewer):
    lead = people["lead"]
    ds = upload(lead)
    r = lead.post(f"/api/datasets/{ds['id']}/baselines", json={"family": "tree", "start": "scratch",
                  "source_recipe_id": None, "targets": ["y"], "assessors": [viewer.user["id"]]})
    assert r.status_code == 422 and "Viewer" in r.text
    start(lead, ds, "tree", [lead])
    r = lead.post(f"/api/datasets/{ds['id']}/assessors", json={"user_id": viewer.user["id"]})
    assert r.status_code == 422 and "Viewer" in r.text


# ------------------------------------------------------------------------------------------- pictures
import io  # noqa: E402

from PIL import Image  # noqa: E402


def _png(w=400, h=300, color=(200, 40, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "PNG")
    return buf.getvalue()


def _put(client, uid, data, name="me.png", ctype="image/png"):
    return client.put(f"/api/users/{uid}/avatar", files={"file": (name, data, ctype)})


@pytest.fixture
def member(people):
    return people["a"]


@needs_db
def test_avatar_is_stored_square_and_served(member, app_):
    from app.config import get_settings
    r = _put(member, member.user["id"], _png())
    assert r.status_code == 200, r.text
    url = r.json()["avatar_url"]
    got = TestClient(app_).get(url)  # open before sign-in: the picker shows it
    assert got.status_code == 200 and got.headers["content-type"] == "image/webp"
    assert "immutable" in got.headers["cache-control"] and got.headers["x-content-type-options"] == "nosniff"
    assert "sandbox" in got.headers["content-security-policy"]
    assert Image.open(io.BytesIO(got.content)).size == (256, 256)
    assert any(u["avatar_url"] == url for u in TestClient(app_).get("/api/auth/users").json())
    old = get_settings().avatars_dir / url.rsplit("/", 2)[-2:][0]
    r2 = _put(member, member.user["id"], _png(color=(10, 200, 10)))
    files = sorted(p.name for p in get_settings().avatars_dir.glob(f"{member.user['id']}-*.webp"))
    assert r2.json()["avatar_url"] != url and len(files) == 1, (old, files)
    assert member.delete(f"/api/users/{member.user['id']}/avatar").status_code == 200
    assert not list(get_settings().avatars_dir.glob(f"{member.user['id']}-*.webp"))
    assert TestClient(app_).get(url).status_code == 404


@needs_db
@pytest.mark.parametrize("data,name,code", [
    (b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>', "x.svg", 400),
    (b"<html><script>alert(1)</script></html>", "x.png", 400),
    (b"\x89PNG" + b"0" * (6 * 1024 * 1024), "big.png", 413),
], ids=["svg", "html-named-png", "too-big"])
def test_avatar_refuses_bad_files(member, data, name, code):
    from app.config import get_settings
    before = set(get_settings().avatars_dir.glob("*"))
    assert _put(member, member.user["id"], data, name).status_code == code
    assert set(get_settings().avatars_dir.glob("*")) == before


@needs_db
def test_avatar_refuses_a_pixel_bomb(member):
    buf = io.BytesIO()
    Image.new("1", (9000, 9000)).save(buf, "PNG")  # tiny file, 81 megapixels
    assert _put(member, member.user["id"], buf.getvalue()).status_code == 400


@needs_db
def test_avatar_permissions(people, viewer):
    a, c, lead = people["a"], people["c"], people["lead"]
    assert _put(c, a.user["id"], _png()).status_code == 403
    assert _put(lead, a.user["id"], _png()).status_code == 200
    assert _put(viewer, viewer.user["id"], _png()).status_code == 200
    assert _put(viewer, a.user["id"], _png()).status_code == 403


@needs_db
def test_avatar_paths_cant_escape(app_):
    c = TestClient(app_)
    assert c.get("/api/auth/avatars/1/../../session_secret").status_code in (401, 404)  # never the file
    assert c.get("/api/auth/avatars/1/0123456789abcdef.webp").status_code == 404
    assert c.get("/api/auth/avatars/x/0123456789abcdef.webp").status_code == 404
