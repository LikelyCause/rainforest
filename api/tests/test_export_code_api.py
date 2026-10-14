"""Export to code through the API: the script downloaded for an export rebuilds it from the uploaded file, a sample
the app generated or a library dataset, and says why when it can't be written. Needs RAINFOREST_TEST_DATABASE_URL."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
from fastapi.testclient import TestClient

from .team_helpers import adjust, app_, csv_text, experiment, people, start, submit_and_finalize, upload, wait  # noqa: F401


def _export(client, ds, **body) -> dict:
    r = client.post(f"/api/datasets/{ds['id']}/exports", json=body)
    assert r.status_code == 202, r.text
    e = wait(lambda: (e := client.get(f"/api/exports/{r.json()['id']}").json())["status"] in ("done", "failed") and e)
    assert e["status"] == "done", e.get("error")
    return e


def _script(client, export_id: int) -> str:
    r = client.get(f"/api/exports/{export_id}/files/code")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/x-python")
    assert f'filename="export{export_id}' in r.headers["content-disposition"]
    return r.text


def _rebuild(client, tmp: Path, export_id: int, *args: str) -> subprocess.CompletedProcess:
    """Run the export's script in a clean interpreter and compare its result with the export's own file."""
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "script.py").write_text(_script(client, export_id))
    (tmp / "theirs.parquet").write_bytes(client.get(f"/api/exports/{export_id}/files/data").content)
    r = subprocess.run([sys.executable, "-I", "script.py", *args, "--out", "mine.parquet", "--compare", "theirs.parquet"],
                       capture_output=True, text=True, cwd=tmp, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Identical to the app's export." in r.stdout and "Same values as theirs.parquet" in r.stdout
    assert pl.read_parquet(tmp / "mine.parquet").equals(pl.read_parquet(tmp / "theirs.parquet"))
    return r


def _csv(n: int = 600) -> str:
    """csv_text plus exact duplicates and a missing target."""
    lines = csv_text(n, seed=11).splitlines()
    lines[7] = ",".join(lines[7].split(",")[:-1] + [""])
    return "\n".join(lines + lines[40:45]) + "\n"


def test_the_script_of_an_adjusted_export_rebuilds_it_from_the_upload(app_, people, login_as, tmp_path):
    lead = people["lead"]
    csv = _csv()
    ds = upload(lead, csv)
    try:
        r = lead.post(f"/api/datasets/{ds['id']}/derived", json={"name": "zw", "kind": "formula",
                                                                 "spec": {"expr": "z * w"}})
        assert r.status_code == 201, r.text
        zw = r.json()["id"]
        start(lead, ds, "tree", [lead])
        rid = submit_and_finalize(lead, ds)["recipe"]["id"]
        x = experiment(lead, ds, rid)
        adjust(lead, ds, x, "x", imputation="median", add_missing_indicator=True)
        adjust(lead, ds, x, "c", notes='three levels """\nimport os')
        e = _export(lead, ds, experiment_id=x["id"], dedupe=True, exclude=[{"field": "c", "values": ["r"]}], seed=5,
                    valid_size=0.2, test_size=0.1)
        assert [d["name"] for d in e["config"]["derived"]] == ["zw"]
        text = _script(lead, e["id"])
        assert "keep.numeric('x', imputation='median', add_missing_indicator=True)" in text
        assert "# adjusted in this experiment: imputation none → median; add_missing_indicator false → true" in text
        assert "        # import os\n" in text
        (tmp_path / "t.csv").write_text(csv)
        _rebuild(lead, tmp_path / "a", e["id"], str(tmp_path / "t.csv"))

        # A viewer can take the script too, as they can the export's files.
        viewer = login_as(TestClient(app_), "viewer", name="Code viewer")
        assert viewer.get(f"/api/exports/{e['id']}/files/code").status_code == 200

        # Removing the derived field later doesn't break the export's script: it keeps the definition it was made with.
        assert lead.delete(f"/api/datasets/{ds['id']}/derived/{zw}").status_code == 200
        _rebuild(lead, tmp_path / "b", e["id"], str(tmp_path / "t.csv"))
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")


def test_a_generated_sample_is_made_again_by_the_script(people, tmp_path):
    from app.db import execute, fetch_one

    lead = people["lead"]
    r = lead.post("/api/datasets/sample", json={"rows": 2000, "fields": 30, "seed": 42, "target_kind": "binary"})
    assert r.status_code == 202, r.text
    ds = wait(lambda: (d := lead.get(f"/api/datasets/{r.json()['id']}").json())["status"] in ("ready", "failed") and d)
    try:
        spec = fetch_one("SELECT source_spec FROM meta.datasets WHERE id = %s", (ds["id"],))["source_spec"]
        assert spec == {"kind": "generated", "settings": {"rows": 2000, "fields": 30, "seed": 42,
                                                           "target_kind": "binary", "numeric_target": False}}
        start(lead, ds, "tree", [lead], targets=("column_a",))
        submit_and_finalize(lead, ds)
        e = _export(lead, ds)
        assert "return generate(n_rows=2000, n_fields=30, seed=42, target_kind='binary'" in _script(lead, e["id"])
        _rebuild(lead, tmp_path / "recorded", e["id"])

        # A sample made before the settings were recorded: worked out from its name and checked by making it again.
        execute("UPDATE meta.datasets SET source_spec = NULL WHERE id = %s", (ds["id"],))
        _rebuild(lead, tmp_path / "inferred", e["id"])
        assert fetch_one("SELECT source_spec FROM meta.datasets WHERE id = %s", (ds["id"],))["source_spec"] == spec

        execute("UPDATE meta.datasets SET source_spec = NULL, name = 'my sample' WHERE id = %s", (ds["id"],))
        r = lead.get(f"/api/exports/{e['id']}/files/code")
        assert r.status_code == 409 and "Generate the sample again" in r.json()["detail"]
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")


def test_a_library_dataset_is_read_from_its_file_or_rebuilt_from_its_source(people, tmp_path):
    from app.db import execute
    from app.sample_library import path_of

    lead = people["lead"]
    r = lead.post("/api/datasets/library/concrete_strength")
    assert r.status_code == 202, r.text
    ds = wait(lambda: (d := lead.get(f"/api/datasets/{r.json()['id']}").json())["status"] in ("ready", "failed") and d)
    try:
        start(lead, ds, "tree", [lead], targets=("strength_mpa",))
        submit_and_finalize(lead, ds)
        e = _export(lead, ds)
        text = _script(lead, e["id"])
        assert "return read_library(path, 'concrete_strength')" in text and '"fastexcel' in text
        # Offline here: give it the library's file rather than letting it download the UCI archive.
        _rebuild(lead, tmp_path / "file", e["id"], str(path_of("concrete_strength")))
        # Added before the source was recorded: recognised by its file name and its data.
        execute("UPDATE meta.datasets SET source_spec = NULL WHERE id = %s", (ds["id"],))
        assert "return read_library(path, 'concrete_strength')" in _script(lead, e["id"])
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")


def test_what_the_script_endpoint_refuses(people):
    lead = people["lead"]
    assert lead.get("/api/exports/999999/files/code").status_code == 404
    ds = upload(lead)
    try:
        start(lead, ds, "tree", [lead])
        submit_and_finalize(lead, ds)
        e = _export(lead, ds)
        from app.db import execute

        execute("UPDATE meta.exports SET status = 'running' WHERE id = %s", (e["id"],))
        assert lead.get(f"/api/exports/{e['id']}/files/code").status_code == 409
        execute("UPDATE meta.exports SET status = 'done' WHERE id = %s", (e["id"],))
        # The export's files removed: the script is still written, without the check against them.
        for f in Path(e["output_dir"]).iterdir():
            f.unlink()
        text = _script(lead, e["id"])
        assert "can't check its result" in text and "EXPECTED_OUTPUT = {}" in text
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")


def test_inputs_the_script_checks_are_the_apps_data(people):
    """The input summary the script checks is the app's own Parquet copy, column by column."""
    from app.db import connection, fetch_one
    from app.export_code import info_for
    from app.export_code.script import column_summary

    lead = people["lead"]
    ds = upload(lead)
    try:
        start(lead, ds, "tree", [lead])
        submit_and_finalize(lead, ds)
        e = _export(lead, ds)
        with connection() as conn:
            info = info_for(conn, e["id"])
        raw = pl.read_parquet(fetch_one("SELECT parquet_path FROM meta.datasets WHERE id = %s", (ds["id"],))["parquet_path"])
        assert info.expected_input["rows"] == raw.height
        assert info.expected_input["columns"] == {c: column_summary(raw[c]) for c in raw.columns}
        assert info.source == {"kind": "upload", "format": "csv", "file": "t.csv"}
        assert np.isclose(info.split["valid_size"], 0.15)
    finally:
        lead.delete(f"/api/datasets/{ds['id']}")
