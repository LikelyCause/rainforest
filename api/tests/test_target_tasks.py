"""Target tasks in the decision tables: the trigger that keeps them coherent, its agreement with semantics, and the
paths that carry them."""
import pytest

from app.semantics import SEMANTIC_TYPES, default_task

from .team_helpers import DB_URL, app_, fields, people, save, upload  # noqa: F401

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
