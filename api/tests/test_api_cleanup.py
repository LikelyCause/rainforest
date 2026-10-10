"""API behaviour against a real Postgres (RAINFOREST_TEST_DATABASE_URL): the leak screen on upload and on target
changes, decision provenance, bulk accept, export preview warnings, upload cleanup and job error state."""
import os
import time

import numpy as np
import pytest

DB_URL = os.environ.get("RAINFOREST_TEST_DATABASE_URL")


@pytest.fixture(scope="module")
def client(tmp_path_factory, admin_login):
    if not DB_URL:
        pytest.skip("set RAINFOREST_TEST_DATABASE_URL to run the API tests")
    from fastapi.testclient import TestClient

    from app import db
    from app.config import get_settings

    mp = pytest.MonkeyPatch()
    mp.setenv("DATABASE_URL", DB_URL)
    mp.setenv("DATA_DIR", str(tmp_path_factory.mktemp("data")))
    get_settings.cache_clear()
    db.close_pool()
    from app.main import app

    with TestClient(app) as c:
        yield admin_login(c)
    db.close_pool()
    mp.undo()
    get_settings.cache_clear()


def wait_until(fn, timeout=120.0, every=0.3):
    deadline = time.monotonic() + timeout
    while True:
        out = fn()
        if out:
            return out
        assert time.monotonic() < deadline, "timed out"
        time.sleep(every)


def upload(client, csv: str, targets: str | None = None, name="t.csv") -> dict:
    data = {"targets": targets} if targets else {}
    ds = client.post("/api/datasets/upload", files={"file": (name, csv, "text/csv")}, data=data).json()
    ds = wait_until(lambda: (d := client.get(f"/api/datasets/{ds['id']}").json())["status"] in ("ready", "failed") and d)
    assert ds["status"] == "ready", ds
    return ds


def fields_of(client, ds) -> dict[str, dict]:
    return {f["name"]: f for f in client.get(f"/api/datasets/{ds['id']}/fields").json()}


def sql(client, q: str) -> list[list]:
    r = client.post("/api/sql", json={"sql": q})
    assert r.status_code == 200, r.text
    return r.json()["rows"]


def test_raw_table_keeps_backslash_n_text_and_strips_nul_bytes(client):
    rows = "".join(f"{i},plain{i % 7}\n" for i in range(200))
    csv = "id,note\n" + rows + "200,\\N\n201,a\x00b\n202,\n"
    ds = upload(client, csv)
    try:
        table = ds["raw_table"]
        got = dict(sql(client, f"SELECT id, note FROM {table} WHERE id >= 200 ORDER BY id"))
        assert got == {200: "\\N", 201: "ab", 202: None}
        assert any("NUL" in w for w in ds["warnings"])
    finally:
        client.delete(f"/api/datasets/{ds['id']}")


def test_upload_of_a_dataset_deleted_while_queued_is_removed(client):
    from app.config import get_settings
    from app.processing import process_dataset

    up = get_settings().uploads_dir / "gone.csv"
    up.write_text("a\n1\n")
    process_dataset(10**9, str(up), "csv")
    assert not up.exists()


def test_startup_sweep_removes_only_orphaned_old_uploads(client):
    from app.config import get_settings
    from app.db import execute, fetch_one
    from app.processing import sweep_uploads

    uploads = get_settings().uploads_dir
    old_orphan, new_orphan, waiting = uploads / "old.csv", uploads / "new.csv", uploads / "waiting.csv"
    for p in (old_orphan, new_orphan, waiting):
        p.write_text("a\n1\n")
    for p in (old_orphan, waiting):
        os.utime(p, (time.time() - 3600, time.time() - 3600))
    row = fetch_one("""INSERT INTO meta.datasets (name, file_format, status, upload_path)
                       VALUES ('w', 'csv', 'pending', %s) RETURNING id""", (str(waiting),))
    try:
        assert sweep_uploads() == [old_orphan]
        assert new_orphan.exists() and waiting.exists()
    finally:
        execute("DELETE FROM meta.datasets WHERE id = %s", (row["id"],))
        for p in (new_orphan, waiting):
            p.unlink(missing_ok=True)


def _leaky_csv(n=20_000, levels=300, seed=0) -> str:
    rng = np.random.default_rng(seed)
    lv = rng.integers(0, levels, n)
    y = np.where(lv % 2 == 0, "yes", "no")
    x = rng.normal(0, 1, n)
    return "merchant,x,y\n" + "".join(f"m{a},{b:.4f},{c}\n" for a, b, c in zip(lv, x, y))


def test_upload_flags_a_high_cardinality_leak(client):
    ds = upload(client, _leaky_csv(), targets="y")
    try:
        f = fields_of(client, ds)
        assert f["merchant"]["n_unique"] == 300 and f["merchant"]["leakage"]
        assert f["merchant"]["max_target_assoc"] > 0.95 and not f["x"]["leakage"]
    finally:
        client.delete(f"/api/datasets/{ds['id']}")


def test_choosing_a_target_later_screens_high_cardinality_fields(client):
    ds = upload(client, _leaky_csv(seed=1))
    try:
        assert not fields_of(client, ds)["merchant"]["leakage"]
        assert client.patch(f"/api/datasets/{ds['id']}", json={"target_columns": ["y"]}).status_code == 200
        wait_until(lambda: fields_of(client, ds)["merchant"]["leakage"], timeout=60)
        assoc = client.get(f"/api/fields/{fields_of(client, ds)['merchant']['id']}").json()["target_associations"]
        assert assoc[0]["name"] == "y" and assoc[0]["strength"] > 0.95
    finally:
        client.delete(f"/api/datasets/{ds['id']}")
