"""The sample library: real datasets that ship with the app. Needs RAINFOREST_TEST_DATABASE_URL for the API part."""

import polars as pl

from app.sample_library import LIBRARY, path_of

from .team_helpers import app_, fields, people, wait  # noqa: F401  (fixtures)


def test_every_entry_has_its_file_and_targets():
    for key, entry in LIBRARY.items():
        df = pl.read_parquet(path_of(key))
        assert df.height > 100 and set(entry["targets"]) <= set(df.columns), key
        assert entry["citation"] and entry["source_url"].startswith("https://archive.ics.uci.edu/")


def test_library_lists_its_datasets_and_adds_one(people):
    lead = people["lead"]
    lib = lead.get("/api/datasets/library").json()
    grades = next(x for x in lib if x["key"] == "student_grades")
    assert (grades["rows"], grades["columns"], grades["targets"]) == (649, 33, ["G1", "G2", "G3"])
    r = lead.post("/api/datasets/library/student_grades", json={})
    assert r.status_code == 202, r.text
    ds = wait(lambda: (d := lead.get(f"/api/datasets/{r.json()['id']}").json())["status"] in ("ready", "failed") and d)
    assert ds["status"] == "ready" and ds["name"] == "Student grades" and ds["target_columns"] == ["G1", "G2", "G3"]
    assert ds["baseline"]["setup"]
    assert fields(lead, ds)["G3"]["action"] == "target"
    assert lead.post("/api/datasets/library/nope", json={}).status_code == 404


def test_tab_separated_uploads_are_read(people):
    lead = people["lead"]
    tsv = "a\tb\ty\n" + "".join(f"{i}\t{i % 3}\t{'xy'[i % 2]}\n" for i in range(200))
    r = lead.post(
        "/api/datasets/upload", files={"file": ("t.tsv", tsv, "text/tab-separated-values")}, data={"targets": "y"}
    )
    assert r.status_code == 202, r.text
    ds = wait(lambda: (d := lead.get(f"/api/datasets/{r.json()['id']}").json())["status"] in ("ready", "failed") and d)
    assert ds["status"] == "ready" and set(fields(lead, ds)) == {"a", "b", "y"}


def test_concrete_entry_is_a_regression_dataset():
    e = LIBRARY["concrete_strength"]
    df = pl.read_parquet(path_of("concrete_strength"))
    assert df.height == 1030 and e["targets"] == ["strength_mpa"] and df["strength_mpa"].dtype == pl.Float64
    assert "10.24432/C5PK67" in e["citation"]
