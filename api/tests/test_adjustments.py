"""Experiment adjustments: a team experiment's own copy of its baseline, changed through view=experiment:<id>,
exported and logged as baseline + adjustments. Needs RAINFOREST_TEST_DATABASE_URL."""
from .team_helpers import app_, people  # noqa: F401  (fixtures)
from .team_helpers import adjust, experiment, export, fields, finalized_trees, save, upload


def test_experiment_copies_exist_and_go_with_their_experiment(people):
    from app.db import connection

    lead = people["lead"]
    ds, rid = finalized_trees(lead)
    x = experiment(lead, ds, rid)
    with connection() as conn:
        n = conn.execute("SELECT count(*) AS n FROM meta.experiment_decisions WHERE experiment_id = %s",
                         (x["id"],)).fetchone()["n"]
        assert n == len(fields(lead, ds))
        conn.execute("DELETE FROM meta.experiments WHERE id = %s", (x["id"],))
        assert conn.execute("SELECT count(*) AS n FROM meta.experiment_settings WHERE experiment_id = %s",
                            (x["id"],)).fetchone()["n"] == 0
