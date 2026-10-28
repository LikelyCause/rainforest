"""Batch planning without a database (classes and skips, the grid and random samplers) and the migration 025 schema
and settings (the DB tests need RAINFOREST_TEST_DATABASE_URL)."""

import math
import os
import statistics

import pytest

from app import batches
from app.config import Settings

DB_URL = os.environ.get("RAINFOREST_TEST_DATABASE_URL")


def _counts(**kw) -> dict[str, dict[str, int]]:
    return {c: {"train": t, "valid": v, "test": s} for c, (t, v, s) in kw.items()}


# ---- classes ------------------------------------------------------------------------------------------------------


def test_plan_classes_trains_every_class_above_the_minimum():
    counts = _counts(A=(50, 10, 10), B=(10, 3, 2))
    assert batches.plan_classes(counts, None, 10) == (["A", "B"], [])


def test_plan_classes_skips_too_few_training_positives_with_the_reason():
    counts = _counts(A=(50, 10, 10), B=(9, 3, 2))
    assert batches.plan_classes(counts, None, 10) == (
        ["A"],
        [{"class": "B", "reason": "fewer than 10 positives in training"}],
    )


def test_plan_classes_skips_a_class_with_no_validation_positives():
    # Positives in training but none to calibrate and threshold on: trained, it could not be calibrated.
    counts = _counts(A=(50, 10, 10), B=(40, 0, 5))
    assert batches.plan_classes(counts, None, 10) == (["A"], [{"class": "B", "reason": "no validation positives"}])


def test_plan_classes_skips_everything_when_the_validation_split_is_empty():
    counts = {"A": {"train": 50, "valid": 0, "test": 10, "valid_rows": 0}}
    assert batches.plan_classes(counts, None, 10) == ([], [{"class": "A", "reason": "no validation rows"}])


def test_plan_classes_gives_each_class_one_reason():
    counts = {"A": {"train": 2, "valid": 0, "test": 0, "valid_rows": 0}}
    assert batches.plan_classes(counts, None, 10)[1] == [
        {"class": "A", "reason": "fewer than 10 positives in training"}
    ]


def test_plan_classes_lists_unticked_classes_as_not_selected():
    counts = _counts(A=(50, 10, 10), B=(9, 3, 2), C=(60, 12, 12))
    assert batches.plan_classes(counts, ["B", "C"], 10) == (
        ["C"],
        [
            {"class": "A", "reason": "not selected"},
            {"class": "B", "reason": "fewer than 10 positives in training"},
        ],
    )


def test_not_selected_wins_over_the_other_reasons():
    counts = _counts(A=(2, 0, 0), B=(60, 12, 12))
    assert batches.plan_classes(counts, ["B"], 10) == (["B"], [{"class": "A", "reason": "not selected"}])


def test_plan_classes_needs_the_training_and_validation_counts():
    with pytest.raises(ValueError, match="A.*train"):
        batches.plan_classes({"A": {"valid": 3, "test": 3}}, None, 10)
    with pytest.raises(ValueError, match="A.*valid"):
        batches.plan_classes({"A": {"train": 30, "test": 3}}, None, 10)


def test_plan_classes_refuses_a_class_the_target_does_not_have():
    with pytest.raises(ValueError, match="Z"):
        batches.plan_classes(_counts(A=(50, 10, 10)), ["Z"], 10)


def test_default_search_classes_are_the_most_frequent_the_median_and_the_rarest_above_the_minimum():
    counts = _counts(
        V01=(5000, 900, 900),
        V02=(1200, 200, 200),
        V03=(800, 100, 100),
        V04=(300, 50, 50),
        V05=(40, 8, 8),
        V06=(12, 3, 3),
        V07=(4, 1, 1),
    )
    # V07 is under the minimum, so V06 is the rarest; of the six eligible the median is V03.
    assert batches.search_classes_default(counts, 10) == ["V01", "V03", "V06"]


def test_default_search_classes_are_distinct_when_few_classes_qualify():
    assert batches.search_classes_default(_counts(A=(50, 10, 10), B=(20, 5, 5)), 10) == ["A", "B"]
    assert batches.search_classes_default(_counts(A=(50, 10, 10), B=(2, 1, 1)), 10) == ["A"]
    assert batches.search_classes_default(_counts(A=(5, 1, 1)), 10) == []


def test_default_search_classes_skip_a_class_that_could_not_be_trained():
    counts = _counts(A=(500, 90, 90), B=(400, 0, 80), C=(30, 6, 6))
    assert batches.search_classes_default(counts, 10) == ["A", "C"]


# ---- grid ---------------------------------------------------------------------------------------------------------


def test_grid_is_the_product_of_the_lists():
    g = batches.grid({"a": [1, 2, 3], "b": ["x", "y"], "c": [True]})
    assert len(g) == 6
    assert {"a": 3, "b": "y", "c": True} in g
    assert len({tuple(sorted(d.items())) for d in g}) == 6


def test_grid_refuses_a_range_and_names_the_setting():
    with pytest.raises(ValueError, match="learning_rate"):
        batches.grid({"learning_rate": {"low": 0.01, "high": 0.3, "log": True}, "n_estimators": [10, 20]})


def test_grid_refuses_an_empty_list_and_names_the_setting():
    with pytest.raises(ValueError, match="max_depth"):
        batches.grid({"max_depth": [], "n_estimators": [10]})


# ---- random -------------------------------------------------------------------------------------------------------

SPACE = {
    "learning_rate": {"low": 1e-3, "high": 1.0, "log": True},
    "max_depth": {"low": 2, "high": 12, "integer": True},
    "subsample": {"low": 0.5, "high": 1.0},
    "boosting": ["gbdt", "dart"],
}


def _ok(_s):
    return None


def test_random_samples_are_reproducible_by_seed():
    a, _ = batches.random_samples(SPACE, 20, 7, _ok)
    b, _ = batches.random_samples(SPACE, 20, 7, _ok)
    c, _ = batches.random_samples(SPACE, 20, 8, _ok)
    assert a == b and a != c and len(a) == 20


def test_random_samples_stay_within_the_ranges_and_take_list_values():
    samples, redraws = batches.random_samples(SPACE, 200, 1, _ok)
    assert redraws == 0
    for s in samples:
        assert 1e-3 <= s["learning_rate"] <= 1.0
        assert 2 <= s["max_depth"] <= 12
        assert 0.5 <= s["subsample"] <= 1.0
        assert s["boosting"] in ("gbdt", "dart")


def test_integer_ranges_give_ints_and_reach_both_ends():
    samples, _ = batches.random_samples({"d": {"low": 2, "high": 4, "integer": True}}, 3, 0, _ok, 10_000)
    assert all(type(s["d"]) is int for s in samples)
    assert {s["d"] for s in samples} == {2, 3, 4}


def test_a_log_integer_range_stays_inside_its_integer_bounds():
    space = {"leaves": {"low": 4, "high": 256, "log": True, "integer": True}}
    samples, _ = batches.random_samples(space, 40, 5, _ok, 10_000)
    assert all(type(s["leaves"]) is int and 4 <= s["leaves"] <= 256 for s in samples)


def test_a_log_integer_range_gives_its_end_integers_full_mass():
    # Rounding a draw over [low, high] would give the ends half the mass of their neighbours; 1..2 is the clearest case.
    counts = {1: 0, 2: 0}
    space = {"d": {"low": 1, "high": 2, "log": True, "integer": True}, "k": {"low": 0.0, "high": 1.0}}
    samples, _ = batches.random_samples(space, 4000, 4, _ok)
    for s in samples:
        counts[s["d"]] += 1
    # Log-uniform over [0.5, 2.5): P(1) = ln(1.5/0.5)/ln(5) = 0.68, P(2) = 0.32.
    assert math.isclose(counts[1] / 4000, math.log(3) / math.log(5), abs_tol=0.04)


def test_a_log_range_is_log_uniform():
    # Drawn uniformly the median would sit near 0.5; log-uniform puts it near the geometric mean, 0.0316.
    samples, _ = batches.random_samples({"lr": {"low": 1e-3, "high": 1.0, "log": True}}, 2000, 11, _ok)
    median = statistics.median(s["lr"] for s in samples)
    assert math.isclose(median, math.sqrt(1e-3 * 1.0), rel_tol=0.25)


def test_invalid_samples_are_redrawn_and_counted():
    def valid(s):
        return "learning_rate above 0.5" if s["lr"] > 0.5 else None

    samples, redraws = batches.random_samples({"lr": {"low": 0.0, "high": 1.0}}, 50, 2, valid)
    assert len(samples) == 50 and all(s["lr"] <= 0.5 for s in samples)
    assert redraws > 0


def test_a_search_that_cannot_find_enough_valid_samples_names_why():
    with pytest.raises(ValueError, match="learning_rate is never allowed") as e:
        batches.random_samples({"lr": {"low": 0.0, "high": 1.0}}, 5, 2, lambda s: "learning_rate is never allowed", 20)
    assert "20" in str(e.value)


def test_the_redraw_limit_defaults_to_ten_times_the_sample_count():
    calls = []

    def valid(s):
        calls.append(s)
        return "no"

    with pytest.raises(ValueError):
        batches.random_samples({"lr": {"low": 0.0, "high": 1.0}}, 3, 2, valid)
    assert len(calls) == 3 * 10 + 1


def test_a_small_list_space_cannot_supply_more_distinct_samples_than_it_has():
    with pytest.raises(ValueError, match="distinct"):
        batches.random_samples({"a": [1, 2]}, 5, 0, _ok)
    samples, _ = batches.random_samples({"a": [1, 2]}, 2, 0, _ok)
    assert sorted(s["a"] for s in samples) == [1, 2]


def test_repeated_draws_are_not_validated_again():
    for seed in range(10):
        seen = []

        def valid(s):
            seen.append(s["a"])
            return "no" if s["a"] == 1 else None

        batches.random_samples({"a": [1, 2, 3]}, 2, seed, valid)
        assert len(seen) == len(set(seen))


def test_a_scalar_where_a_list_is_expected_names_the_setting():
    for bad in (5, "gbdt", None):
        with pytest.raises(ValueError, match="boosting"):
            batches.random_samples({"boosting": bad}, 1, 0, _ok)


@pytest.mark.parametrize(
    "spec",
    [
        {"low": float("nan"), "high": 1.0},
        {"low": 0.0, "high": float("inf")},
        {"low": 5, "high": 1},
        {"low": 0, "high": 1, "log": True},
        {"low": 1},
        {"low": 0.5, "high": 0.9, "integer": True},
    ],
)
def test_malformed_ranges_are_refused_with_the_setting_named(spec):
    with pytest.raises(ValueError, match="x"):
        batches.random_samples({"x": spec}, 1, 0, _ok)


# ---- settings -----------------------------------------------------------------------------------------------------


def test_batch_and_deployment_settings_have_their_defaults():
    s = Settings()
    assert (s.ovr_min_positives, s.ovr_weak_pr_auc, s.batch_max_runs) == (10, 0.5, 500)
    assert (s.predict_max_records, s.deployment_cache) == (1000, 4)


# ---- migration 025 ------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def database(tmp_path_factory):
    if not DB_URL:
        pytest.skip("set RAINFOREST_TEST_DATABASE_URL to run the migration tests")
    from app import db
    from app.config import get_settings

    mp = pytest.MonkeyPatch()
    mp.setenv("DATABASE_URL", DB_URL)
    mp.setenv("DATA_DIR", str(tmp_path_factory.mktemp("data")))
    get_settings.cache_clear()
    db.close_pool()
    db.bootstrap()
    yield db
    db.close_pool()
    mp.undo()
    get_settings.cache_clear()


def _cols(db, table):
    return {
        r["column_name"]
        for r in db.fetch_all(
            "SELECT column_name FROM information_schema.columns WHERE table_schema = 'meta' AND table_name = %s",
            (table,),
        )
    }


def _constraint(db, table, name):
    return db.fetch_one(
        "SELECT pg_get_constraintdef(oid) AS def FROM pg_constraint WHERE conrelid = %s::regclass AND conname = %s",
        (f"meta.{table}", name),
    )["def"]


def test_migration_025_creates_the_batch_and_deployment_tables(database):
    assert {"experiment_id", "export_id", "kind", "name", "spec", "skipped", "status", "finished_at"} <= _cols(
        database, "run_batches"
    )
    assert {"models", "status", "batch_id", "retired_at", "export_id", "dataset_id"} <= _cols(database, "deployments")
    assert {"prefix", "hash", "last_used_at", "revoked_at"} <= _cols(database, "deployment_keys")
    assert {"filename", "has_labels", "metrics", "error", "owner", "finished_at"} <= _cols(database, "deployment_tests")
    assert "batch_id" in _cols(database, "model_runs")


def test_model_runs_may_be_cancelled(database):
    def_ = _constraint(database, "model_runs", "model_runs_status_check")
    assert "'cancelled'" in def_ and "'failed'" in def_


def test_batch_kinds_and_statuses_are_constrained(database):
    assert "one_vs_rest" in _constraint(database, "run_batches", "run_batches_kind_check")
    assert "'cancelled'" in _constraint(database, "run_batches", "run_batches_status_check")
    assert "'retired'" in _constraint(database, "deployments", "deployments_status_check")
    assert "'failed'" in _constraint(database, "deployment_tests", "deployment_tests_status_check")


def test_the_export_a_deployment_uses_cannot_be_deleted_under_it(database):
    rows = database.fetch_all(
        """SELECT confdeltype FROM pg_constraint
            WHERE conrelid = 'meta.deployments'::regclass AND contype = 'f' AND confrelid = 'meta.exports'::regclass"""
    )
    assert [r["confdeltype"] for r in rows] == ["r"]


def test_row_security_keeps_keys_and_private_experiments_out_of_the_console_role(database):
    # The console role is granted SELECT on every meta table; row security is what keeps key hashes and a private
    # experiment's batches and deployments away from it.
    tables = ["run_batches", "deployments", "deployment_keys", "deployment_tests"]
    rows = database.fetch_all(
        "SELECT relname, relrowsecurity FROM pg_class WHERE relnamespace = 'meta'::regnamespace AND relname = ANY(%s)",
        (tables,),
    )
    assert {r["relname"]: r["relrowsecurity"] for r in rows} == dict.fromkeys(tables, True)
    policies = {
        r["tablename"] for r in database.fetch_all("SELECT tablename FROM pg_policies WHERE schemaname = 'meta'")
    }
    assert "deployment_keys" not in policies and {"run_batches", "deployments", "deployment_tests"} <= policies


def test_the_console_role_has_no_privilege_on_keys(database):
    from app.config import get_settings

    assert "deployment_keys" in database.PRIVATE_TABLES  # test_bootstrap_roles and test_team check these revokes
    role = get_settings().readonly_role
    ok = database.fetch_one("SELECT has_table_privilege(%s, 'meta.deployment_keys', 'SELECT') AS ok", (role,))["ok"]
    assert ok is False
