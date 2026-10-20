"""Registry transforms through the API: the option catalog, and a decision refused for a model family."""
from .conftest import set_phase
from .test_api_cleanup import SAVED_KEYS, _numbers_csv, client, fields_of, upload  # noqa: F401


def test_options_publish_registry_transforms(client):
    cat = client.get("/api/options").json()
    assert cat["labels"]["transform"]["yeo_johnson"] == "Yeo-Johnson (power transform)"
    assert "yeo_johnson" in cat["allowed"]["linear"]["numeric_continuous"]["transform"]
    assert cat["allowed"]["tree"]["numeric_continuous"]["transform"] == ["none"]
    assert [t["key"] for t in cat["transforms"]] == ["yeo_johnson"]


def test_a_trees_baseline_refuses_yeo_johnson(client):
    ds = upload(client, _numbers_csv(), targets="y")
    set_phase(ds["id"], "review")
    fid = fields_of(client, ds)["x"]["id"]
    cur = client.get(f"/api/fields/{fid}?family=tree").json()["decision"]
    r = client.put(f"/api/fields/{fid}/decision",
                   json={**{k: cur[k] for k in SAVED_KEYS}, "family": "tree", "transform": "yeo_johnson"})
    assert r.status_code == 422 and "not valid for numeric_continuous with tree models" in r.text
