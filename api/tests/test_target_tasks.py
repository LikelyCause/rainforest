"""Target tasks in the decision tables: the trigger that keeps them coherent, its agreement with semantics, and the
paths that carry them."""
import pytest

from app.semantics import SEMANTIC_TYPES, default_task

from .team_helpers import (DB_URL, adjust, app_, experiment, export, fields, people, review_all, save,  # noqa: F401
                           start, submit_and_finalize, upload)

pytestmark = pytest.mark.skipif(not DB_URL, reason="set RAINFOREST_TEST_DATABASE_URL to run the API tests")


def _sql_default(conn, t, n):
    return conn.execute("SELECT meta.default_target_task(%s, %s) AS t", (t, n)).fetchone()["t"]


def test_sql_default_matches_python(app_):
    from app.db import connection
    with connection() as conn:
        for t in SEMANTIC_TYPES:
            for n in (None, 2, 100, 101, 5000):
                assert _sql_default(conn, t, n) == default_task(t, n), (t, n)


def test_trigger_defaults_clears_and_resets(app_, people):
    from app.db import connection
    lead = people["lead"]
    ds = upload(lead, "x,price,c\n" + "".join(f"{i},{i * 1.5 + 0.25},{'pq'[i % 2]}\n" for i in range(300)), targets="")
    f = fields(lead, ds)
    with connection() as conn:
        def row(name):
            return conn.execute("SELECT action, target_task, target_bins FROM meta.field_decisions WHERE field_id = %s",
                                (f[name]["id"],)).fetchone()
        conn.execute("UPDATE meta.field_decisions SET action = 'target' WHERE field_id = %s", (f["price"]["id"],))
        assert row("price")["target_task"] == "regression"
        conn.execute("UPDATE meta.field_decisions SET target_task = 'bins' WHERE field_id = %s", (f["price"]["id"],))
        assert row("price")["target_bins"] == {"k": 5}
        conn.execute("UPDATE meta.field_decisions SET target_task = 'regression' WHERE field_id = %s", (f["price"]["id"],))
        assert row("price")["target_bins"] is None
        conn.execute("UPDATE meta.field_decisions SET action = 'keep' WHERE field_id = %s", (f["price"]["id"],))
        assert row("price")["target_task"] is None and row("price")["target_bins"] is None
        conn.rollback()


def test_trigger_resets_task_the_type_no_longer_allows(app_, people):
    from app.db import connection
    lead = people["lead"]
    ds = upload(lead, "x,price\n" + "".join(f"{i},{i * 1.5 + 0.25}\n" for i in range(300)), targets="")
    fid = fields(lead, ds)["price"]["id"]
    with connection() as conn:
        conn.execute("UPDATE meta.field_decisions SET action = 'target', target_task = 'regression' WHERE field_id = %s",
                     (fid,))
        conn.execute("UPDATE meta.field_decisions SET semantic_type = 'categorical' WHERE field_id = %s", (fid,))
        got = conn.execute("SELECT target_task FROM meta.field_decisions WHERE field_id = %s", (fid,)).fetchone()
        # Categorical can't be a number, so the trigger offers its default; validation reports the level limit when
        # anyone saves it.
        assert got["target_task"] == "classes"
        conn.rollback()


def test_unknown_task_is_refused(app_, people):
    import psycopg
    from app.db import connection
    lead = people["lead"]
    ds = upload(lead, "x,price\n" + "".join(f"{i},{i * 1.5 + 0.25}\n" for i in range(300)), targets="")
    fid = fields(lead, ds)["price"]["id"]
    with connection() as conn, pytest.raises(psycopg.errors.CheckViolation):
        conn.execute("UPDATE meta.field_decisions SET action = 'target', target_task = 'nope' WHERE field_id = %s",
                     (fid,))


def test_every_target_has_a_task(app_):
    from app.db import connection
    with connection() as conn:
        bad = conn.execute("""SELECT count(*) AS n FROM meta.field_decisions
                               WHERE (action = 'target') <> (target_task IS NOT NULL)""").fetchone()["n"]
        assert bad == 0


PRICES = "x,z,price,c\n" + "".join(f"{i % 17},{(i * 7) % 11},{i * 1.5 + (i % 5)},{'pq'[i % 2]}\n" for i in range(400))


def _task(client, ds, name, view=None):
    d = fields(client, ds, view)[name]
    got = client.get(f"/api/fields/{d['id']}" + (f"?view={view}" if view else "")).json()["decision"]
    return got["target_task"], got["target_bins"]


def test_put_sets_validates_and_keeps_the_task(app_, people):
    lead = people["lead"]
    ds = upload(lead, PRICES, targets="")
    start(lead, ds, "tree", [lead], targets=("price",))
    fid = fields(lead, ds)["price"]["id"]
    assert _task(lead, ds, "price") == ("regression", None)
    assert save(lead, fid, target_task="bins", target_bins={"edges": [100, 300]}).status_code == 200
    assert _task(lead, ds, "price") == ("bins", {"edges": [100, 300]})
    r = save(lead, fid, target_task="classes")
    assert r.status_code == 422 and "continuous" in r.text
    r = save(lead, fid, target_task="bins", target_bins={"edges": [5, 1]})
    assert r.status_code == 422 and "ascending" in r.text


def test_put_without_task_keeps_it(app_, people):
    lead = people["lead"]
    ds = upload(lead, PRICES, targets="")
    start(lead, ds, "tree", [lead], targets=("price",))
    fid = fields(lead, ds)["price"]["id"]
    save(lead, fid, target_task="bins", target_bins={"k": 4})
    cur = lead.get(f"/api/fields/{fid}").json()["decision"]
    body = {k: cur[k] for k in ("action", "semantic_type", "imputation", "impute_value", "add_missing_indicator",
                                "transform", "scaling", "encoding", "params", "notes", "reviewed")}
    assert lead.put(f"/api/fields/{fid}/decision", json={**body, "notes": "n"}).status_code == 200
    assert _task(lead, ds, "price") == ("bins", {"k": 4})
    assert save(lead, fid, target_task="bins").status_code == 200
    assert _task(lead, ds, "price") == ("bins", {"k": 4})


def test_a_task_disagreement_goes_to_review(app_, people):
    lead, a, b = people["lead"], people["a"], people["b"]
    ds = upload(lead, PRICES, targets="")
    start(lead, ds, "tree", [a, b], targets=("price",))
    fid = fields(a, ds)["price"]["id"]
    save(a, fid, target_task="regression")
    save(b, fid, target_task="bins", target_bins={"k": 3})
    for who in (a, b):
        review_all(who, ds)
        assert who.post(f"/api/datasets/{ds['id']}/assessment/submit").status_code == 200
    queue = lead.get(f"/api/datasets/{ds['id']}/review").json()["queue"]
    row = next(q for q in queue if q["name"] == "price")
    assert {(x["target_task"], (x["target_bins"] or {}).get("k")) for x in row["assessments"]} == {
        ("regression", None), ("bins", 3)}


def test_agreed_task_is_adopted_finalized_and_carried(app_, people):
    """Review Focus 1: the task survives adoption, finalize, a new version, an experiment copy and a revert."""
    lead, a, b = people["lead"], people["a"], people["b"]
    ds = upload(lead, PRICES, targets="")
    start(lead, ds, "tree", [a, b], targets=("price",))
    for who in (a, b):
        save(who, fields(who, ds)["price"]["id"], target_task="bins", target_bins={"k": 4})
    st = submit_and_finalize(lead, ds, a, b)
    rid = st["recipe"]["id"]
    from app.db import fetch_one
    snap = fetch_one("SELECT snapshot FROM meta.recipes WHERE id = %s", (rid,))["snapshot"]
    assert snap["fields"]["price"]["decision"]["target_task"] == "bins"
    assert snap["fields"]["price"]["decision"]["target_bins"] == {"k": 4}
    assert _task(lead, ds, "price", f"baseline:{rid}") == ("bins", {"k": 4})
    x = experiment(lead, ds, rid)
    view = f"experiment:{x['id']}"
    assert _task(lead, ds, "price", view) == ("bins", {"k": 4})
    adjust(lead, ds, x, "price", target_task="regression")
    adj = lead.get(f"/api/experiments/{x['id']}").json()["adjustments"]
    assert any(c["key"] == "target_task" for f in adj["fields"] for c in f["changes"])
    fid = fields(lead, ds, view)["price"]["id"]
    r = lead.post(f"/api/experiments/{x['id']}/adjustments/revert", json={"field_ids": [fid]})
    assert r.status_code == 200, r.text
    assert _task(lead, ds, "price", view) == ("bins", {"k": 4})
    start(lead, ds, "tree", [lead], start="baseline", source=rid, targets=("price",))
    assert _task(lead, ds, "price") == ("bins", {"k": 4})


def test_bulk_review_keeps_the_task(app_, people):
    lead = people["lead"]
    ds = upload(lead, PRICES, targets="")
    start(lead, ds, "tree", [lead], targets=("price",))
    save(lead, fields(lead, ds)["price"]["id"], target_task="bins", target_bins={"k": 3}, reviewed=False)
    review_all(lead, ds)
    assert _task(lead, ds, "price") == ("bins", {"k": 3})


def test_reuse_carries_the_task(app_, people):
    lead = people["lead"]
    ds = upload(lead, PRICES, targets="")
    start(lead, ds, "tree", [lead], targets=("price",))
    save(lead, fields(lead, ds)["price"]["id"], target_task="bins", target_bins={"edges": [200, 400]})
    rid = submit_and_finalize(lead, ds)["recipe"]["id"]
    other = upload(lead, PRICES, targets="")
    start(lead, other, "tree", [lead], start="recipe", source=rid, targets=())
    assert _task(lead, other, "price") == ("bins", {"edges": [200, 400]})
