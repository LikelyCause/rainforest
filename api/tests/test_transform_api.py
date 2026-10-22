"""Registry transforms through the API: the option catalog, a decision refused for a model family, and a
target-aware option's target checked on save and in the export preview."""

import json

import numpy as np

from app.db import connection
from app.routers.exports import target_aware_problems

from .conftest import set_phase
from .test_api_cleanup import SAVED_KEYS, _numbers_csv, client, fields_of, upload  # noqa: F401


def test_options_publish_registry_transforms(client):
    cat = client.get("/api/options").json()
    assert cat["labels"]["transform"]["yeo_johnson"] == "Yeo-Johnson (power transform)"
    assert "yeo_johnson" in cat["allowed"]["linear"]["numeric_continuous"]["transform"]
    assert cat["allowed"]["tree"]["numeric_continuous"]["transform"] == ["none"]
    assert [t["key"] for t in cat["transforms"]] == ["yeo_johnson", "target"]


def test_a_trees_baseline_refuses_yeo_johnson(client):
    ds = upload(client, _numbers_csv(), targets="y")
    set_phase(ds["id"], "review")
    fid = fields_of(client, ds)["x"]["id"]
    cur = client.get(f"/api/fields/{fid}?family=tree").json()["decision"]
    r = client.put(
        f"/api/fields/{fid}/decision",
        json={**{k: cur[k] for k in SAVED_KEYS}, "family": "tree", "transform": "yeo_johnson"},
    )
    assert r.status_code == 422 and "not valid for numeric_continuous with tree models" in r.text


def _keep(name, encoding="target", **params):
    return {"name": name, "transform": "none", "encoding": encoding, "params": {"target": None, **params}}


def test_target_aware_problems_block_and_warn():
    keep = [_keep("diag_1", target="G2"), _keep("mjob"), _keep("age", encoding="onehot")]
    blocking, warnings = target_aware_problems(keep, ["G1", "G3"], ["G1", "G2", "G3"], "experiment", "Trees v1")
    assert blocking == ["Target encoding on `diag_1` uses `G2`, which this experiment doesn't predict."]
    assert warnings == []
    blocking, warnings = target_aware_problems(keep, ["G2", "G3"], ["G1", "G2", "G3"], "experiment", "Trees v1")
    assert blocking == []
    assert warnings == ["Target encoding on `mjob` follows the first target, now `G2` (`G1` in Trees v1)."]
    assert target_aware_problems(keep, ["G1"], None, "baseline", "Trees v1")[0] == [
        "Target encoding on `diag_1` uses `G2`, which this baseline doesn't predict."
    ]


def _coded_csv(n=600, seed=0) -> str:
    rng = np.random.default_rng(seed)
    return "c,x,y\n" + "".join(f"L{rng.integers(0, 40)},{rng.normal():.4f},{'ab'[i % 2]}\n" for i in range(n))


def test_the_field_response_lists_targets_and_saving_checks_them(client):
    ds = upload(client, _coded_csv(), targets="y")
    set_phase(ds["id"], "review")
    fid = fields_of(client, ds)["c"]["id"]
    got = client.get(f"/api/fields/{fid}?family=tree").json()
    assert got["targets"] == ["y"]
    body = {
        **{k: got["decision"][k] for k in SAVED_KEYS},
        "family": "tree",
        "encoding": "target",
        "params": {"target": "g9", "smooth": None, "cv": 5},
    }
    r = client.put(f"/api/fields/{fid}/decision", json=body)
    assert r.status_code == 422 and "params.target 'g9' isn't one of this source's targets (y)" in r.text
    ok = client.put(f"/api/fields/{fid}/decision", json={**body, "params": {**body["params"], "target": "y"}})
    assert ok.status_code == 200, ok.text


def test_the_export_preview_blocks_a_target_no_longer_predicted(client):
    ds = upload(client, _coded_csv(), targets="y")
    set_phase(ds["id"], "review")
    fid = fields_of(client, ds)["c"]["id"]
    got = client.get(f"/api/fields/{fid}?family=tree").json()
    body = {**{k: got["decision"][k] for k in SAVED_KEYS}, "family": "tree", "encoding": "target"}
    assert client.put(f"/api/fields/{fid}/decision", json={**body, "params": {"target": "y"}}).status_code == 200
    with connection() as conn:  # a target that was a target when saved and since stopped being one
        n = conn.execute(
            "UPDATE meta.field_settings SET params = %s WHERE field_id = %s AND family = 'tree'",
            (json.dumps({"target": "gone", "smooth": None, "cv": 5}), fid),
        ).rowcount
    assert n == 1
    p = client.get(f"/api/datasets/{ds['id']}/export-preview").json()
    assert "Target encoding on `c` uses `gone`, which this baseline doesn't predict." in p["blocking"]
