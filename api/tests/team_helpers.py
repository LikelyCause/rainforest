"""Shared fixtures and helpers for the DB-backed team tests (baselines, experiment adjustments). Import the fixtures
into a test module (`from .team_helpers import app_, people`) to use them there."""

import os
import time
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient

DB_URL = os.environ.get("RAINFOREST_TEST_DATABASE_URL")
KEYS = (
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


@pytest.fixture(scope="module")
def people(app_, login_as):
    tag = uuid.uuid4().hex[:6]
    return {
        "lead": login_as(TestClient(app_)),
        "a": login_as(TestClient(app_), "member", name=f"Assessor A {tag}"),
        "b": login_as(TestClient(app_), "member", name=f"Assessor B {tag}"),
        "c": login_as(TestClient(app_), "member", name=f"Bystander C {tag}"),
    }


def csv_text(n=400, seed=3) -> str:
    """x (some missing), z, w continuous; c three levels; y binary (the target)."""
    rng = np.random.default_rng(seed)
    return "x,z,w,c,y\n" + "".join(
        f"{'' if i % 9 == 0 else f'{a:.3f}'},{b:.3f},{c:.3f},{'pqr'[i % 3]},{'ab'[i % 2]}\n"
        for i, (a, b, c) in enumerate(zip(rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)))
    )


def wait(fn, timeout=120.0):
    deadline = time.monotonic() + timeout
    while not (out := fn()):
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.3)
    return out


def upload(client, csv=None, targets="y") -> dict:
    ds = client.post(
        "/api/datasets/upload", files={"file": ("t.csv", csv or csv_text(), "text/csv")}, data={"targets": targets}
    ).json()
    ds = wait(lambda: (d := client.get(f"/api/datasets/{ds['id']}").json())["status"] in ("ready", "failed") and d)
    assert ds["status"] == "ready", ds
    return ds


def fields(client, ds, view: str | None = None) -> dict[str, dict]:
    q = f"?view={view}" if view else ""
    r = client.get(f"/api/datasets/{ds['id']}/fields{q}")
    assert r.status_code == 200, r.text
    return {f["name"]: f for f in r.json()}


def save(client, fid, view: str | None = None, **patch):
    q = f"?view={view}" if view else ""
    cur = client.get(f"/api/fields/{fid}{q}").json()["decision"]
    return client.put(f"/api/fields/{fid}/decision{q}", json={**{k: cur[k] for k in KEYS}, **patch})


def review_all(client, ds, view: str | None = None):
    q = f"?view={view}" if view else ""
    r = client.post(f"/api/datasets/{ds['id']}/decisions/bulk{q}", json={"filters": [], "reviewed": True})
    assert r.status_code == 200, r.text


def start(lead, ds, family, assessors, start="scratch", source=None, targets=("y",), expect=200) -> dict:
    r = lead.post(
        f"/api/datasets/{ds['id']}/baselines",
        json={
            "family": family,
            "start": start,
            "source_recipe_id": source,
            "targets": list(targets),
            "assessors": [u.user["id"] for u in assessors],
        },
    )
    assert r.status_code == expect, r.text
    return r.json()


def submit_and_finalize(lead, ds, *assessors) -> dict:
    """Every given assessor (default: the lead) reviews everything and submits; the lead finalizes."""
    for who in assessors or (lead,):
        review_all(who, ds)
        r = who.post(f"/api/datasets/{ds['id']}/assessment/submit")
        assert r.status_code == 200, r.text
    r = lead.post(f"/api/datasets/{ds['id']}/finalize", json={"accept_open": True})
    assert r.status_code == 200, r.text
    return r.json()


def finalized_trees(lead, csv=None) -> tuple[dict, int]:
    """A dataset with a finalized Trees baseline v1 (the lead assessing, suggestions accepted); returns it and the
    recipe id."""
    ds = upload(lead, csv)
    start(lead, ds, "tree", [lead])
    st = submit_and_finalize(lead, ds)
    return ds, st["recipe"]["id"]


def experiment(client, ds, rid, name="tries") -> dict:
    r = client.post(f"/api/datasets/{ds['id']}/experiments", json={"name": name, "source": f"recipe:{rid}"})
    assert r.status_code == 201, r.text
    return r.json()


def adjust(client, ds, x, name, expect=200, **patch):
    view = f"experiment:{x['id']}"
    fid = fields(client, ds, view)[name]["id"]
    r = save(client, fid, view, **patch)
    assert r.status_code == expect, r.text
    return r


def export(client, ds, x) -> dict:
    r = client.post(f"/api/datasets/{ds['id']}/exports", json={"experiment_id": x["id"]})
    assert r.status_code == 202, r.text
    e = wait(lambda: (e := client.get(f"/api/exports/{r.json()['id']}").json())["status"] in ("done", "failed") and e)
    assert e["status"] == "done", e.get("error")
    return e
