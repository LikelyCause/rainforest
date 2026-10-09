"""Explore analyses: known-answer checks on synthetic data, and the job/cache API when a test database is given.

API tests need an empty Postgres: RAINFOREST_TEST_DATABASE_URL=postgresql://user:pw@host:port/db pytest
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl
import pytest

from app.analysis import compute
from app.analysis.prep import AnalysisError, NotEnough, prepare_export, prepare_raw, sample_rows
from app.routers.explore import cluster_order
from app.schemas import AnalysisParams


def _frame(n: int = 1500, seed: int = 1, rare: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    classes = np.array(["a", "b", "c"])[y].astype(object)
    if rare:
        classes[:rare] = "rare"
    df = {
        "y": classes,
        "y_num": y * 2.0 + rng.normal(0, 0.5, n),
        "sep": y * 3.0 + rng.normal(0, 0.5, n),
        "cat_sep": np.where(rng.random(n) < 0.6, np.array(["p", "q", "r"])[y], "q").astype(object),
        "when": [f"2024-01-{1 + d:02d}" for d in rng.integers(0, 28, n)],
        "note": ["free text here"] * n,
        "row_key": np.arange(n),
    }
    for k in range(6):
        df[f"noise{k}"] = rng.normal(0, 1, n)
    df["noise_cat"] = rng.choice(["u", "v", "w", "x"], n).astype(object)
    return pl.DataFrame(df)


TYPES = {"y": "categorical", "y_num": "numeric_continuous", "sep": "numeric_continuous", "cat_sep": "categorical",
         "when": "datetime", "note": "text", "row_key": "identifier", "noise_cat": "categorical"}


def _fields(df: pl.DataFrame, targets=("y", "y_num")) -> list[dict]:
    return [{"id": i + 1, "name": c, "action": "target" if c in targets else "keep",
             "semantic_type": TYPES.get(c, "numeric_continuous"), "datetime_format": "%Y-%m-%d"}
            for i, c in enumerate(df.columns)]


@pytest.fixture(scope="module")
def parquet(tmp_path_factory):
    path = tmp_path_factory.mktemp("explore") / "d.parquet"
    _frame().write_parquet(path)
    return path


def _prep(parquet, target="y", sample=1000, kept=False):
    return prepare_raw(parquet, _fields(pl.read_parquet(parquet)), kept, target, sample)


def test_prep_types_skips_and_stratifies(parquet):
    p = _prep(parquet)
    names = {c.name: c for c in p.columns}
    assert p.n == 1000 and p.target_kind == "categorical" and p.classes == ["a", "b", "c"]
    assert {s["name"]: s["reason"] for s in p.skipped} == {"note": "free text", "row_key": "identifier"}
    assert "y_num" not in names, "other targets are never features"
    assert names["when"].kind == "numeric" and names["cat_sep"].kind == "categorical"
    full = pl.read_parquet(parquet)["y"].value_counts().sort("y")["count"].to_numpy() / 1500
    sampled = np.bincount(p.y, minlength=3) / p.n
    assert np.abs(full - sampled).max() < 0.002


def test_sample_rows_exact_size_and_drops_missing_target():
    y = np.array([0] * 70 + [1] * 25 + [-1] * 5)
    idx = sample_rows(100, 40, y, "categorical")
    assert idx.size == 40 and (y[idx] >= 0).all()
    assert np.bincount(y[idx]).tolist() == [29, 11]
    assert sample_rows(100, 500, y, "categorical").size == 95


def test_planted_separator_tops_feature_tests(parquet):
    r = compute.run_tests(_prep(parquet))
    top = r["rows"][0]
    assert (top["name"], top["test"], top["effect_name"]) == ("sep", "anova", "η²")
    assert top["effect"] > 0.8 and top["p_adj"] is not None and top["p_adj"] >= top["p"]
    assert top["secondary"]["test"] == "kruskal"
    assert [g["class"] for g in top["detail"]["groups"]] == ["a", "b", "c"]
    g = top["detail"]["groups"][0]
    assert g["min"] <= g["q1"] <= g["median"] <= g["q3"] <= g["max"]
    cat = next(row for row in r["rows"] if row["name"] == "cat_sep")
    assert cat["test"] == "chi2" and cat["detail"]["kind"] == "levels"
    assert len(cat["detail"]["counts"]) == 3 and len(cat["detail"]["counts"][0]) == len(cat["detail"]["levels"])
    effects = [row["effect"] for row in r["rows"]]
    assert effects == sorted(effects, reverse=True)
    noise = [row for row in r["rows"] if row["name"].startswith("noise")]
    assert all(row["effect"] < 0.05 for row in noise)
    json.dumps(r, allow_nan=False)


def test_numeric_target_uses_rank_correlation(parquet):
    r = compute.run_tests(_prep(parquet, target="y_num"))
    assert r["target_kind"] == "numeric" and r["classes"] == [] and r["excluded_classes"] == []
    top = r["rows"][0]
    assert top["name"] == "sep" and top["test"] == "spearman" and top["secondary"]["test"] == "pearson"
    assert top["detail"]["kind"] == "scatter" and len(top["detail"]["points"]) == 500
    cat = next(row for row in r["rows"] if row["name"] == "cat_sep")
    assert cat["test"] == "anova" and cat["detail"]["kind"] == "box"
    groups = [g["class"] for g in cat["detail"]["groups"]]
    assert groups[0] == "q" and sorted(groups) == ["p", "q", "r"]


def test_rare_classes_are_excluded_and_listed(tmp_path):
    path = tmp_path / "rare.parquet"
    _frame(rare=3).write_parquet(path)
    r = compute.run_tests(_prep(path, sample=1500))
    assert r["excluded_classes"] == [{"name": "rare", "n": 3}]
    assert r["classes"] == ["a", "b", "c"] and r["n"] == 1497


def test_tests_need_a_target_and_two_classes(parquet, tmp_path):
    with pytest.raises(NotEnough) as e:
        compute.run_tests(_prep(parquet, target=None))
    assert e.value.reason == "no_target" and str(e.value).startswith("Not enough")
    path = tmp_path / "one.parquet"
    _frame().with_columns(pl.lit("only").alias("y")).write_parquet(path)
    with pytest.raises(NotEnough) as e:
        compute.run_tests(_prep(path))
    assert e.value.reason == "single_class"


def test_mutual_information_and_pca(parquet):
    r = compute.run_importance(_prep(parquet))
    assert r["mi"][0]["name"] == "sep" and r["mi"][0]["field_id"] == 3
    assert [m["value"] for m in r["mi"]] == sorted((m["value"] for m in r["mi"]), reverse=True)
    pca = r["pca"]
    assert 0 < sum(pca["explained"]) <= 1 and pca["cumulative"][-1] <= 1
    assert pca["cumulative"] == sorted(pca["cumulative"])
    assert len(pca["components"]) == 5 and all(len(c["loadings"]) <= 20 for c in pca["components"])
    assert pca["n_features"] > len(r["mi"]), "categorical levels are one-hot for PCA"
    json.dumps(r, allow_nan=False)


def test_importance_without_target_is_pca_only(parquet):
    r = compute.run_importance(_prep(parquet, target=None))
    assert r["mi"] == [] and r["target"] is None and r["target_kind"] is None and r["pca"]["explained"]


def test_tsne_map(parquet):
    p = _prep(parquet)
    r = compute.run_map(p, "tsne", perplexity=20)
    assert r["n"] == len(r["points"]) == len(r["hover_values"]) == 1000
    assert r["classes"] == ["a", "b", "c"] and {pt[2] for pt in r["points"]} == {0, 1, 2}
    assert 0 < len(r["hover_fields"]) <= 5 and len(r["hover_values"][0]) == len(r["hover_fields"])
    assert compute.run_map(p, "tsne", perplexity=20)["points"] == r["points"], "fixed seed"
    json.dumps(r, allow_nan=False)


def test_map_without_target_and_with_numeric_target(parquet):
    r = compute.run_map(_prep(parquet, target=None), "tsne", perplexity=10)
    assert r["target_kind"] is None and r["classes"] == [] and all(pt[2] is None for pt in r["points"])
    r = compute.run_map(_prep(parquet, target="y_num"), "tsne", perplexity=10)
    assert r["target_kind"] == "numeric" and isinstance(r["points"][0][2], float)


def test_map_too_few_rows_for_perplexity(tmp_path):
    path = tmp_path / "tiny.parquet"
    _frame(n=40).write_parquet(path)
    with pytest.raises(NotEnough) as e:
        compute.run_map(_prep(path), "tsne", perplexity=50)
    assert e.value.reason == "too_few_rows"


def test_umap_missing_fails_clearly(parquet, monkeypatch):
    monkeypatch.setitem(sys.modules, "umap", None)
    with pytest.raises(AnalysisError, match="UMAP is not available"):
        compute.run_map(_prep(parquet), "umap")


def test_umap_map(parquet):
    pytest.importorskip("umap")
    r = compute.run_map(_prep(parquet), "umap", n_neighbors=15)
    assert r["method"] == "umap" and len(r["points"]) == 1000


def test_export_mode_reads_train_split_only(tmp_path):
    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 2, n)
    pl.DataFrame({
        "__row_id": np.arange(n), "__split": np.where(np.arange(n) < 400, "train", "test"),
        "f_sep": (y * 4 + rng.normal(0, 1, n)).astype(np.float32),
        "f_noise": rng.normal(0, 1, n).astype(np.float32),
        "f_onehot": (rng.random(n) < 0.3).astype(np.float32),
        "column_a": y.astype(np.int32),
    }).write_parquet(tmp_path / "model_ready.parquet")
    manifest = {"targets": ["column_a"], "target_classes": {"column_a": ["no", "yes"]}}
    p = prepare_export(tmp_path, manifest, "column_a", 1000)
    assert p.n == 400 and p.classes == ["no", "yes"] and [c.name for c in p.columns] == ["f_sep", "f_noise", "f_onehot"]
    r = compute.run_tests(p)
    assert r["rows"][0]["name"] == "f_sep" and r["rows"][0]["field_id"] is None
    with pytest.raises(AnalysisError):
        prepare_export(tmp_path, manifest, "column_b", 1000)


def test_correlated_pairs_cluster_together():
    s = np.full((6, 6), 0.05)
    for a, b in [(0, 3), (1, 4), (2, 5)]:
        s[a, b] = s[b, a] = 0.9
    order = cluster_order(s)
    assert sorted(order) == list(range(6))
    for a, b in [(0, 3), (1, 4), (2, 5)]:
        assert abs(order.index(a) - order.index(b)) == 1


def test_params_normalised_per_kind():
    p = AnalysisParams(perplexity=50, n_neighbors=40, method="tsne", kept=True)
    assert p.normalised("tests") == {"data": "raw", "kept": True, "target": None, "sample": 5000}
    assert p.normalised("map")["perplexity"] == 50 and "n_neighbors" not in p.normalised("map")
    umap = AnalysisParams(method="umap", n_neighbors=40).normalised("map")
    assert "perplexity" not in umap and umap["n_neighbors"] == 40
    assert AnalysisParams(data="export:3", kept=True).normalised("importance")["kept"] is False
    with pytest.raises(ValueError):
        AnalysisParams(sample=500)
    with pytest.raises(ValueError):
        AnalysisParams(data="export:x")


# ------------------------------------------------------------------------------------------- API
DB_URL = os.environ.get("RAINFOREST_TEST_DATABASE_URL")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
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
        yield c
    db.close_pool()
    mp.undo()
    get_settings.cache_clear()


def _wait(client, url: str, done, timeout: float = 180) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        body = client.get(url).json()
        if done(body):
            return body
        assert time.monotonic() < deadline, f"timed out waiting on {url}: {body}"
        time.sleep(0.3)


@pytest.fixture(scope="module")
def dataset(client):
    ds = client.post("/api/datasets/sample", json={"rows": 2000, "fields": 30, "target_kind": "binary"}).json()
    ds = _wait(client, f"/api/datasets/{ds['id']}", lambda d: d["status"] in ("ready", "failed"))
    assert ds["status"] == "ready", ds
    yield ds
    client.delete(f"/api/datasets/{ds['id']}")


def _analysis(client, ds_id: int, kind: str, **params) -> dict:
    recompute = params.pop("recompute", False)
    r = client.post(f"/api/datasets/{ds_id}/analyses", json={"kind": kind, "params": params, "recompute": recompute})
    assert r.status_code == 200, r.text
    return r.json()


def _finished(client, row: dict) -> dict:
    return _wait(client, f"/api/analyses/{row['id']}", lambda a: a["status"] in ("done", "failed"))


def test_api_map_is_cached_then_stale_then_recomputed(client, dataset):
    ds = dataset["id"]
    first = _analysis(client, ds, "map", target="column_a", sample=1000, perplexity=20)
    assert first["status"] in ("pending", "running", "done") and first["stale"] is False
    assert set(first) == {"id", "dataset_id", "kind", "params", "status", "progress", "status_message", "error",
                          "result", "created_at", "finished_at", "stale"}
    done = _finished(client, first)
    assert done["status"] == "done", done["error"]
    assert len(done["result"]["points"]) == 1000 and done["result"]["target"] == "column_a"
    again = _analysis(client, ds, "map", target="column_a", sample=1000, perplexity=20, n_neighbors=99)
    assert (again["id"], again["status"], again["stale"]) == (done["id"], "done", False)

    fields = client.get(f"/api/datasets/{ds}/fields").json()
    kept = next(f for f in fields if f["action"] == "keep")
    assert client.post(f"/api/datasets/{ds}/decisions/bulk",
                       json={"field_ids": [kept["id"]], "action": "drop"}).status_code == 200
    stale = _analysis(client, ds, "map", target="column_a", sample=1000, perplexity=20)
    assert (stale["id"], stale["stale"]) == (done["id"], True)
    assert client.get(f"/api/analyses/{done['id']}").json()["stale"] is True

    fresh = _analysis(client, ds, "map", target="column_a", sample=1000, perplexity=20, recompute=True)
    assert fresh["id"] != done["id"] and fresh["stale"] is False
    assert _finished(client, fresh)["status"] == "done"


def test_api_new_request_supersedes_running_one_of_same_kind(client, dataset):
    ds = dataset["id"]
    a = _analysis(client, ds, "map", sample=2000, perplexity=40, recompute=True)
    other = _analysis(client, ds, "importance", target="column_a", sample=1000, recompute=True)
    b = _analysis(client, ds, "map", sample=2000, perplexity=25, recompute=True)
    first = client.get(f"/api/analyses/{a['id']}").json()
    if first["status"] != "done":  # it can finish before the second request lands
        assert (first["status"], first["error"]) == ("failed", "Superseded")
    assert _finished(client, other)["status"] == "done"
    assert _finished(client, b)["status"] == "done"


def test_api_tests_without_target_is_not_enough(client, dataset):
    row = _finished(client, _analysis(client, dataset["id"], "tests"))
    assert row["status"] == "failed" and row["error"].startswith("Not enough")
    assert row["result"] == {"reason": "no_target"}


def test_api_tests_and_importance(client, dataset):
    tests = _finished(client, _analysis(client, dataset["id"], "tests", target="column_a", sample=1000))
    assert tests["status"] == "done" and tests["result"]["rows"]
    imp = _finished(client, _analysis(client, dataset["id"], "importance", sample=1000))
    assert imp["status"] == "done" and imp["result"]["mi"] == [] and imp["result"]["pca"]["explained"]


def test_api_tiny_dataset_is_not_enough(client):
    csv = "a,b,c,t\n" + "".join(f"{i},{i * 2 % 5},{'xy'[i % 2]},{i % 2}\n" for i in range(12))
    ds = client.post("/api/datasets/upload", files={"file": ("tiny.csv", csv, "text/csv")},
                     data={"targets": "t"}).json()
    ds = _wait(client, f"/api/datasets/{ds['id']}", lambda d: d["status"] in ("ready", "failed"))
    assert ds["status"] == "ready"
    try:
        row = _finished(client, _analysis(client, ds["id"], "map", sample=1000, perplexity=30))
        assert row["status"] == "failed" and row["result"] == {"reason": "too_few_rows"}
    finally:
        client.delete(f"/api/datasets/{ds['id']}")


def test_api_validation_and_missing(client, dataset):
    ds = dataset["id"]
    bad = client.post(f"/api/datasets/{ds}/analyses", json={"kind": "map", "params": {"sample": 10}})
    assert bad.status_code == 422
    assert client.post(f"/api/datasets/{ds}/analyses", json={"kind": "map", "params": {"target": "nope"}}
                       ).status_code == 422
    assert client.post(f"/api/datasets/{ds}/analyses", json={"kind": "map", "params": {"data": "export:999999"}}
                       ).status_code == 404
    assert client.post("/api/datasets/999999/analyses", json={"kind": "map"}).status_code == 404
    assert client.get("/api/analyses/999999").status_code == 404


def test_api_correlations(client, dataset):
    ds = dataset["id"]
    r = client.get(f"/api/datasets/{ds}/explore/correlations", params={"target": "column_a", "limit": 8})
    assert r.status_code == 200, r.text
    body = r.json()
    ids = [f["id"] for f in body["fields"]]
    assert len(ids) == 8 and body["fields"][0]["name"] == "column_a" and sorted(body["order"]) == sorted(ids)
    assert set(body["fields"][0]) == {"id", "name", "type", "action"}
    cell = body["cells"][0]
    assert set(cell) == {"a", "b", "method", "value", "strength", "n"} and cell["a"] in ids and cell["b"] in ids
    extra = next(f["id"] for f in client.get(f"/api/datasets/{ds}/fields").json() if f["id"] not in ids)
    plus = client.get(f"/api/datasets/{ds}/explore/correlations",
                      params={"target": "column_a", "limit": 8, "add": str(extra)}).json()
    assert len(plus["fields"]) == 9 and extra in plus["order"]
    kept = client.get(f"/api/datasets/{ds}/explore/correlations", params={"kept": 1}).json()
    assert all(f["action"] == "keep" for f in kept["fields"]) and len(kept["fields"]) <= 60
    assert client.get(f"/api/datasets/{ds}/explore/correlations", params={"add": "1,x"}).status_code == 422
    assert client.get(f"/api/datasets/{ds}/explore/correlations", params={"add": "999999"}).status_code == 422
    assert client.get("/api/datasets/999999/explore/correlations").status_code == 404
