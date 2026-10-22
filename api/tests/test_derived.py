"""Derived fields: formulas, value groups and counts become ordinary fields of the dataset."""

import polars as pl
import pytest
from fastapi import HTTPException

from app import derived

from .team_helpers import app_, fields, people, start, submit_and_finalize, upload  # noqa: F401  (fixtures)


def test_a_formula_computes_ratios_and_leaves_impossible_values_missing():
    df = pl.DataFrame(
        {
            "water": [180.0, 160.0, 0.0],
            "cement": [300.0, 0.0, 0.0],
            "slag": [100.0, 0.0, 0.0],
            "odd-name": [1.0, 2.0, 3.0],
        }
    )
    s = derived.compute(df, "formula", {"expr": "water / (cement + slag) + 0 * `odd-name`"})
    assert s.to_list()[0] == pytest.approx(0.45) and s.to_list()[1] is None and s.to_list()[2] is None
    assert derived.compute(df, "formula", {"expr": "log1p(water) + clip(cement, 0, 100)"}).to_list()[
        0
    ] == pytest.approx(5.198497 + 100, rel=1e-5)


@pytest.mark.parametrize("expr", ["__import__('os')", "water.real", "open('x')", "water; 1", "lambda: 1"])
def test_a_formula_refuses_anything_but_arithmetic(expr):
    with pytest.raises(HTTPException):
        derived.check_spec("formula", {"expr": expr}, ["water"])


def test_groups_put_codes_into_chapters_by_range_and_value():
    df = pl.DataFrame({"diag": ["428", "250.83", "V57", None, "786", "E909"]})
    spec = {
        "source": "diag",
        "other": "Other",
        "groups": [
            {"label": "Circulatory", "ranges": [[390, 459.99], [785, 785.99]]},
            {"label": "Diabetes", "ranges": [[250, 250.99]]},
            {"label": "Respiratory", "ranges": [[460, 519.99], [786, 786.99]]},
        ],
    }
    assert derived.compute(df, "groups", spec).to_list() == [
        "Circulatory",
        "Diabetes",
        "Other",
        None,
        "Respiratory",
        "Other",
    ]


def test_numeric_group_labels_make_a_number():
    df = pl.DataFrame({"age": ["[0-10)", "[50-60)"]})
    s = derived.compute(
        df,
        "groups",
        {
            "source": "age",
            "other": "0",
            "groups": [{"label": "5", "values": ["[0-10)"]}, {"label": "55", "values": ["[50-60)"]}],
        },
    )
    assert s.dtype == pl.Float64 and s.to_list() == [5.0, 55.0]


def test_count_counts_matching_values_across_fields():
    df = pl.DataFrame({"a": ["Up", "No"], "b": ["Down", "Steady"], "c": ["No", None]})
    assert derived.compute(df, "count", {"fields": ["a", "b", "c"], "values": ["Up", "Down"]}).to_list() == [2, 0]


CSV = "w,c,y\n" + "".join(f"{150 + i % 40},{200 + (i * 7) % 300},{'ab'[i % 2]}\n" for i in range(400))


def test_a_derived_field_joins_the_dataset_and_a_new_version_assesses_it(people):
    lead = people["lead"]
    ds = upload(lead, CSV)
    start(lead, ds, "tree", [lead])
    r = lead.post(
        f"/api/datasets/{ds['id']}/derived", json={"name": "wc", "kind": "formula", "spec": {"expr": "w / c"}}
    )
    assert r.status_code == 409  # mid-assessment: added once finalized
    submit_and_finalize(lead, ds)
    p = lead.post(
        f"/api/datasets/{ds['id']}/derived/preview", json={"kind": "formula", "spec": {"expr": "w / c"}}
    ).json()
    assert p["type"] == "number" and len(p["sample"]) == 8
    r = lead.post(
        f"/api/datasets/{ds['id']}/derived", json={"name": "wc", "kind": "formula", "spec": {"expr": "w / c"}}
    )
    assert r.status_code == 201, r.text
    assert (
        lead.post(
            f"/api/datasets/{ds['id']}/derived", json={"name": "WC", "kind": "formula", "spec": {"expr": "w"}}
        ).status_code
        == 422
    )
    assert [d["name"] for d in lead.get(f"/api/datasets/{ds['id']}/derived").json()] == ["wc"]
    start(
        lead,
        ds,
        "tree",
        [lead],
        start="baseline",
        source=lead.get(f"/api/datasets/{ds['id']}/baselines").json()[0]["latest_finalized_id"],
    )
    f = fields(lead, ds)["wc"]
    assert f["action"] == "keep" and f["semantic_type"] == "numeric_continuous"
    assert lead.delete(f"/api/datasets/{ds['id']}/derived/{f['id']}").status_code == 409  # being assessed again


@pytest.mark.parametrize("reopen", [False, True])
def test_a_new_version_puts_fields_added_since_the_last_one_in_front_of_the_team(people, reopen):
    lead = people["lead"]
    ds = upload(lead, CSV)
    start(lead, ds, "tree", [lead])
    submit_and_finalize(lead, ds)
    assert (
        lead.post(
            f"/api/datasets/{ds['id']}/derived", json={"name": "wc", "kind": "formula", "spec": {"expr": "w / c"}}
        ).status_code
        == 201
    )
    body = {"family": "tree", "reopen": [fields(lead, ds)["w"]["id"]] if reopen else []}
    r = lead.post(f"/api/datasets/{ds['id']}/recipes", json=body)
    assert r.status_code == 200, r.text
    f = fields(lead, ds)
    assert not f["wc"]["reviewed"] and f["c"]["reviewed"]
    assert lead.get(f"/api/fields/{f['wc']['id']}").json()["locked"] is None  # open to decide, not carried over


def test_a_lead_marks_one_chosen_run_per_experiment_and_authors_note_runs(people):
    from .team_helpers import experiment, export, finalized_trees, wait

    lead, member = people["lead"], people["a"]
    ds, rid = finalized_trees(lead)
    e = export(lead, ds, experiment(lead, ds, rid, "choose"))
    ids = []
    for n in (10, 20):
        r = lead.post(
            f"/api/exports/{e['id']}/runs", json={"model": "lightgbm", "cv_folds": 0, "params": {"n_estimators": n}}
        ).json()
        ids.append(
            wait(lambda: (x := lead.get(f"/api/runs/{r['id']}").json())["status"] in ("done", "failed") and x)["id"]
        )
    best = lambda: next(x for x in lead.get("/api/datasets").json() if x["id"] == ds["id"])["best_run"]  # noqa: E731
    unchosen = best()  # nothing chosen yet: the best run on validation stands for the dataset
    assert unchosen["run_id"] in ids and not unchosen["chosen"] and unchosen["experiment_id"] == e["experiment_id"]
    assert member.patch(f"/api/runs/{ids[0]}", json={"chosen": True}).status_code == 403
    assert lead.patch(f"/api/runs/{ids[0]}", json={"chosen": True, "note": "best CV"}).json()["chosen_at"]
    assert (best()["run_id"], best()["chosen"]) == (ids[0], True)
    second = lead.patch(f"/api/runs/{ids[1]}", json={"chosen": True}).json()
    first = lead.get(f"/api/runs/{ids[0]}").json()
    assert second["chosen_at"] and first["chosen_at"] is None and first["note"] == "best CV"
    assert second["chosen_by_name"] == lead.get("/api/auth/me").json()["name"]
    listed = {x["id"]: x for x in lead.get(f"/api/datasets/{ds['id']}/experiments").json()}[second["experiment_id"]]
    assert listed["chosen"]["run_id"] == ids[1] and listed["chosen"]["model"] == "lightgbm"
    assert lead.get(f"/api/experiments/{second['experiment_id']}").json()["chosen"]["run_id"] == ids[1]
    assert member.patch(f"/api/runs/{ids[0]}", json={"note": "mine?"}).status_code == 403
