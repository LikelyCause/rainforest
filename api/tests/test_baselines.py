"""Baselines: one per model type, each assessed and finalized like a recipe version, started from scratch, from
another baseline, or from another dataset's. Needs RAINFOREST_TEST_DATABASE_URL."""
from .team_helpers import app_, people  # noqa: F401  (fixtures)


def test_recipes_have_a_family_and_versions_count_per_family(app_):
    from app.db import connection

    with connection() as conn:
        cols = {r["column_name"] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_schema='meta' AND table_name='recipes'")}
        assert "family" in cols
        idx = conn.execute("SELECT indexdef FROM pg_indexes WHERE schemaname='meta' AND tablename='recipes'").fetchall()
        assert any("(dataset_id, family, version)" in r["indexdef"] for r in idx)
        xcols = {r["column_name"] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_schema='meta' AND table_name='experiments'")}
        assert "is_default" in xcols
