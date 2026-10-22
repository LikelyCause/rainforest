"""What a run is called (ridge regression for the linear model on numbers), what its downloaded model file is called,
and how many rows the export panel says its row rules leave to split."""
import math
from pathlib import Path

import pytest

from app.modeling.mlflow_log import model_file_name
from app.modeling.registry import MODELS, catalog, label_for

from .team_helpers import (DB_URL, app_, experiment, fields, people, save, start,  # noqa: F401
                           submit_and_finalize, upload, wait)

PRICES = "x,z,price,c\n" + "".join(f"{i % 17},{(i * 7) % 11},{i * 1.5 + (i % 5)},{'pq'[i % 2]}\n" for i in range(400))
MIGRATION = Path(__file__).parents[1] / "app" / "migrations" / "021_ridge_run_labels.sql"


def test_the_linear_model_is_named_for_its_targets():
    lr, rf = MODELS["logistic_regression"], MODELS["random_forest"]
    assert label_for(lr, ["binary"]) == label_for(lr, ["multiclass"]) == label_for(lr, []) == "Logistic regression"
    assert label_for(lr, ["regression"]) == label_for(lr, ["regression", "regression"]) == "Ridge regression"
    assert label_for(lr, ["binary", "regression"]) == "Logistic and ridge regression"
    # A random forest is one name for classes and numbers.
    assert {label_for(rf, k) for k in (["binary"], ["regression"], ["binary", "regression"])} == {"Random forest"}
    entry = next(m for m in catalog() if m["key"] == "logistic_regression")
    assert entry["regressor_label"] == "Ridge regression" and "numbers" in entry["regressor_description"]
    assert next(m for m in catalog() if m["key"] == "lightgbm")["regressor_label"] is None


def test_the_model_file_is_named_for_its_chain():
    assert model_file_name({"chain_kind": "target_chain"}) == "target_chain.joblib"
    assert model_file_name({"chain_kind": "classifier_chain"}) == "classifier_chain.joblib"
    # A report from before chain_kind: its target kinds decide; with neither, the classifier chain it then was.
    assert model_file_name({"target_kinds": {"y": "binary", "v": "regression"}}) == "target_chain.joblib"
    assert model_file_name({}) == model_file_name(None) == "classifier_chain.joblib"


def _linear_on_prices(lead):
    ds = upload(lead, PRICES, targets="")
    start(lead, ds, "linear", [lead], targets=("price",))
    rid = submit_and_finalize(lead, ds)["recipe"]["id"]
    x = experiment(lead, ds, rid, name="ridge")
    r = lead.post(f"/api/datasets/{ds['id']}/exports", json={"experiment_id": x["id"]})
    assert r.status_code == 202, r.text
    e = wait(lambda: (e := lead.get(f"/api/exports/{r.json()['id']}").json())["status"] in ("done", "failed") and e)
    assert e["status"] == "done", e.get("error")
    return ds, x, e


def _train(lead, e, model, **body):
    r = lead.post(f"/api/exports/{e['id']}/runs", json={"model": model, "cv": 0, "cv_folds": 0, **body})
    assert r.status_code == 202, r.text
    run = wait(lambda: (g := lead.get(f"/api/runs/{r.json()['id']}").json())["status"] in ("done", "failed") and g)
    assert run["status"] == "done", run.get("error")
    return r.json(), run


@pytest.mark.skipif(not DB_URL, reason="set RAINFOREST_TEST_DATABASE_URL to run the API tests")
def test_a_linear_run_on_a_number_is_ridge_regression_everywhere(people):
    from app.db import connection

    lead = people["lead"]
    ds, x, e = _linear_on_prices(lead)
    queued, run = _train(lead, e, "logistic_regression")
    # Named from the start, so a queued or failed run reads right too; the model key stays.
    assert queued["report"]["model_label"] == "Ridge regression" and queued["model"] == "logistic_regression"
    assert run["report"]["model_label"] == "Ridge regression" and run["report"]["chain_kind"] == "target_chain"
    assert lead.patch(f"/api/runs/{run['id']}", json={"chosen": True}).status_code == 200
    listed = next(i for i in lead.get(f"/api/datasets/{ds['id']}/experiments").json() if i["id"] == x["id"])
    assert listed["chosen"]["model_label"] == "Ridge regression"
    home = next(d for d in lead.get("/api/datasets").json() if d["id"] == ds["id"])
    assert home["best_run"]["model_label"] == "Ridge regression" and home["last_run"]["model_label"] == "Ridge regression"

    # The download says it is a target chain; the file on disk keeps its one name.
    r = lead.get(f"/api/runs/{run['id']}/files/model")
    assert r.status_code == 200
    assert f'filename="run{run["id"]}_logistic_regression_target_chain.joblib"' in r.headers["content-disposition"]
    assert (Path(run["model_dir"]) / "classifier_chain.joblib").exists()
    assert "report.json" in lead.get(f"/api/runs/{run['id']}/files/report").headers["content-disposition"]

    # A run stored before the fix carried the classifier's name; the migration renames it by its target kinds.
    with connection() as conn:
        conn.execute("""UPDATE meta.model_runs SET report = jsonb_set(report, '{model_label}', '"Logistic regression"')
                         WHERE id = %s""", (run["id"],))
        conn.execute(MIGRATION.read_text())
        label = conn.execute("SELECT report ->> 'model_label' AS l FROM meta.model_runs WHERE id = %s",
                             (run["id"],)).fetchone()["l"]
    assert label == "Ridge regression"


@pytest.mark.skipif(not DB_URL, reason="set RAINFOREST_TEST_DATABASE_URL to run the API tests")
def test_a_class_run_keeps_its_name_and_classifier_chain_file(people):
    lead = people["lead"]
    ds = upload(lead)
    start(lead, ds, "tree", [lead])
    rid = submit_and_finalize(lead, ds)["recipe"]["id"]
    x = experiment(lead, ds, rid, name="forest")
    r = lead.post(f"/api/datasets/{ds['id']}/exports", json={"experiment_id": x["id"]})
    e = wait(lambda: (e := lead.get(f"/api/exports/{r.json()['id']}").json())["status"] in ("done", "failed") and e)
    _, run = _train(lead, e, "random_forest", params={"n_estimators": 10})
    assert run["report"]["model_label"] == "Random forest"
    disp = lead.get(f"/api/runs/{run['id']}/files/model").headers["content-disposition"]
    assert f"run{run['id']}_random_forest_classifier_chain.joblib" in disp


# --- the split preview's row counts -------------------------------------------------------------------------------

def _split_counts(n: int, valid: float, test: float) -> dict[str, int]:
    """The UI's splitCounts (ui/src/experiments.ts): the held-out share rounds up, then test's part of it."""
    rest = math.ceil((valid + test) * n)
    t = math.ceil(test / (valid + test) * rest)
    return {"train": n - rest, "valid": rest - t, "test": t}


VISITS = "patient,visit,x,y\n" + "".join(
    f"{i % 90},{i // 90},{(i * 13) % 29},{'' if i % 50 == 7 else 'ab'[i % 2]}\n" for i in range(600)) + \
    "".join(f"{i % 90},{i // 90},{(i * 13) % 29},{'ab'[i % 2]}\n" for i in range(0, 40, 2))  # 20 exact duplicates


@pytest.mark.skipif(not DB_URL, reason="set RAINFOREST_TEST_DATABASE_URL to run the API tests")
def test_the_split_preview_counts_rows_after_the_row_rules(people):
    lead = people["lead"]
    ds = upload(lead, VISITS)
    start(lead, ds, "tree", [lead])
    rid = submit_and_finalize(lead, ds)["recipe"]["id"]
    x = experiment(lead, ds, rid, name="rows")
    url = f"/api/datasets/{ds['id']}/export-rows"

    plain = lead.post(url, json={"experiment_id": x["id"]}).json()
    assert plain["rows"] == plain["kept"] == 620 and plain["removed"] == {} and plain["groups"] is None
    assert plain["missing_target"] == 12 and plain["split_rows"] == 608  # rows 7, 57, … have no y

    rules = {"dedupe": True, "exclude": [{"field": "visit", "values": ["0"]}]}
    got = lead.post(url, json={**rules, "experiment_id": x["id"]}).json()
    assert got["rows"] == 620 and got["removed"]["duplicates"] == 20 and got["removed"]["visit in 0"] == 90
    assert got["kept"] == 620 - 20 - 90 and got["missing_target"] == 10  # rows 7 and 57 had visit 0
    assert got["split_rows"] == got["kept"] - got["missing_target"]
    assert lead.post(url, json={**rules, "experiment_id": x["id"]}).json() == got  # cached, same answer

    # The export agrees: it splits exactly the rows the preview counted, in the counts the preview gives.
    def exported(body):
        r = lead.post(f"/api/datasets/{ds['id']}/exports", json={**body, "experiment_id": x["id"]})
        assert r.status_code == 202, r.text
        e = wait(lambda: (e := lead.get(f"/api/exports/{r.json()['id']}").json())["status"] in ("done", "failed") and e)
        assert e["status"] == "done", e.get("error")
        return e["summary"]["split_counts"]

    sized = {**rules, "valid_size": 0.2, "test_size": 0.15, "seed": 7}
    exact = lead.post(url, json={**sized, "experiment_id": x["id"]}).json()
    assert exact["split"] == _split_counts(got["split_rows"], 0.2, 0.15) == exported(sized)

    # A grouped split moves whole groups; the preview runs the export's own split, so its counts are exact too.
    grouped = {"dedupe": True, "group_by": "patient", "valid_size": 0.15, "test_size": 0.15, "seed": 3}
    pv = lead.post(url, json={**grouped, "experiment_id": x["id"]}).json()
    assert pv["groups"] == 90 and pv["kept"] == 600 and pv["split_error"] is None
    assert pv["split"] == exported(grouped) and pv["split"] != _split_counts(pv["split_rows"], 0.15, 0.15)

    assert lead.post(url, json={"group_by": "nope", "experiment_id": x["id"]}).status_code == 422
    assert lead.post(url, json={"valid_size": 0.45, "test_size": 0.45, "experiment_id": x["id"]}).status_code == 422
    assert lead.post("/api/datasets/999999/export-rows", json={}).status_code == 404
