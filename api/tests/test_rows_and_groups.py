"""Phase 1 of the gap analysis: row rules, grouped splits, one-vs-rest targets and coded-column inference."""

import numpy as np
import polars as pl
import pytest

from app.modeling.crossval import folds
from app.profiling import infer_type
from app.semantics import validate_decision
from app.transform import assign_splits, prepare_targets, row_rules

from .team_helpers import app_, people  # noqa: F401  (fixtures)


@pytest.fixture
def visits(tmp_path):
    rng = np.random.default_rng(1)
    n = 600
    patient = rng.integers(0, 150, n)
    df = pl.DataFrame(
        {
            "patient": patient,
            "visit": rng.integers(0, 10, n),
            "disch": rng.choice([1, 11, 3], n),
            "x": rng.normal(size=n).round(1),
            "y": rng.choice(["<30", ">30", "NO"], n, p=[0.2, 0.3, 0.5]),
        }
    )
    df = pl.concat([df, df.head(20)])  # 20 exact duplicates
    p = tmp_path / "d.parquet"
    df.write_parquet(p)
    return p, df


def test_row_rules_remove_duplicates_values_and_later_rows(visits):
    p, df = visits
    keep, groups, removed = row_rules(
        p,
        {
            "dedupe": True,
            "exclude": [{"field": "disch", "values": ["11"]}],
            "first_per": {"key": "patient", "order": "visit"},
        },
    )
    assert removed["duplicates"] == 20 and removed["disch in 11"] > 0 and groups is None
    kept = df.with_row_index("i").filter(pl.Series(keep))
    assert kept["patient"].n_unique() == kept.height and (kept["disch"] != 11).all()


def test_a_grouped_split_keeps_each_group_in_one_split(visits):
    p, df = visits
    keep, groups, _ = row_rules(p, {"group_by": "patient"})
    codes, info = assign_splits(df.select("y").to_pandas(), 0.15, 0.15, 3, keep=keep, groups=groups)
    per_group = pl.DataFrame({"g": groups, "s": codes}).group_by("g").agg(pl.col("s").n_unique())
    assert info["grouped"] and per_group["s"].max() == 1 and set(np.unique(codes)) == {0, 1, 2}


def test_cv_folds_keep_groups_whole():
    rng = np.random.default_rng(0)
    g = rng.integers(0, 40, 400)
    y = rng.integers(0, 2, 400)
    out, _, _ = folds(y, 400, 5, 1, "y", groups=g)
    for tr, te in out:
        assert not set(g[tr]) & set(g[te])


def test_a_class_target_can_be_predicted_one_vs_rest(visits):
    p, _ = visits
    d = {
        "action": "target",
        "semantic_type": "categorical",
        "target_task": "bins",
        "target_bins": {"positive": ["<30"]},
    }
    assert validate_decision(d, "tree", n_unique=3) == []
    assert validate_decision({**d, "target_bins": {"k": 4}}, "tree", n_unique=3)
    assert validate_decision({**d, "semantic_type": "numeric_continuous"}, "tree", n_unique=300)
    _, _, classes, _, _, info = prepare_targets(p, ["y"], {"y": {"task": "bins", "bins": {"positive": ["<30"]}}}, {}, 1)
    assert classes["y"] == ["(rest)", "<30"] and info["target_tasks"]["y"] == "classes"  # <30 is the positive class
    assert info["target_bins"]["y"] == {"positive": ["<30"]}


def test_coded_columns_are_inferred_as_codes_and_keys():
    rng = np.random.default_rng(0)
    diag = pl.Series([str(v) for v in rng.choice([250.83, 428, 414], 9800)] + ["V57", "E909"] * 100)
    assert infer_type(diag, "diag_1", diag.n_unique(), diag.len())[0] == "categorical"
    pat = pl.Series(rng.integers(0, 70_000, 100_000))
    assert infer_type(pat, "patient_nbr", pat.n_unique(), pat.len())[0] == "identifier"
    adm = pl.Series(rng.integers(1, 9, 5000))
    assert infer_type(adm, "admission_type_id", adm.n_unique(), adm.len())[0] == "categorical"
    count = pl.Series(rng.integers(1, 80, 5000))
    assert infer_type(count, "num_medications", count.n_unique(), count.len())[0] == "numeric_discrete"


def test_the_database_keeps_a_one_vs_rest_target(people):
    from .team_helpers import fields, save, start, upload

    lead = people["lead"]
    csv = "x,y\n" + "".join(f"{i % 7},{'abc'[i % 3]}\n" for i in range(300))
    ds = upload(lead, csv, targets="y")
    start(lead, ds, "tree", [lead])
    r = save(lead, fields(lead, ds)["y"]["id"], action="target", target_task="bins", target_bins={"positive": ["a"]})
    assert r.status_code == 200, r.text
    d = fields(lead, ds)
    got = lead.get(f"/api/fields/{d['y']['id']}").json()["decision"]
    assert (got["target_task"], got["target_bins"]) == ("bins", {"positive": ["a"]})


def test_binary_scores_carry_pr_auc_and_a_threshold_table():
    import numpy as np

    from app.modeling.train import evaluate

    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.1).astype(int)
    p1 = np.clip(0.1 + 0.3 * y + rng.normal(0, 0.15, 2000), 0, 1)
    m, d = evaluate(y, np.column_stack([1 - p1, p1]), ["no", "yes"])
    assert 0.1 < m["pr_auc"] <= 1 and len(d["thresholds"]) == 19
    best = max(d["thresholds"], key=lambda r: r["f1"])
    assert best["f1"] > m["f1"]  # the 0.5 cut-off isn't the best one for a rare class
