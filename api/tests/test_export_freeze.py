"""An export freezes what scoring needs to read new records: the derived fields' formulas and each source field's
type and date format. The API tests need RAINFOREST_TEST_DATABASE_URL."""

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from app.semantics import suggest
from app.transform import ROW_ID, FieldSpec, run_export

from .team_helpers import (  # noqa: F401  (fixtures)
    adjust,
    app_,
    experiment,
    export,
    fields,
    finalized_trees,
    people,
    start,
    submit_and_finalize,
    wait,
)


def _dates(n: int, ambiguous: int) -> list[str]:
    """dd/mm/YYYY dates whose day is 12 or less in the first `ambiguous` rows, so only later rows say day-first."""
    return [f"{1 + i % 12 if i < ambiguous else 13 + i % 15:02d}/{1 + (i // 12) % 12:02d}/2020" for i in range(n)]


def _spec(name: str, t: str, **kw) -> FieldSpec:
    s = suggest(t, {}, False, "tree")
    keys = ("imputation", "impute_value", "add_missing_indicator", "transform", "scaling", "encoding", "params")
    return FieldSpec(name, t, **{**{k: s.get(k) for k in keys}, **kw})


def test_a_date_with_no_recorded_format_is_read_with_its_whole_columns_format_in_every_chunk(tmp_path):
    n = 3000
    rng = np.random.default_rng(0)
    src = tmp_path / "raw.parquet"
    pl.DataFrame({"d": _dates(n, 1500), "y": rng.choice(["a", "b"], n)}).write_parquet(src)
    out = tmp_path / "export"
    manifest = run_export(src, [_spec("d", "datetime")], ["y"], out, {"seed": 1, "chunk_rows": 1000}, lambda *_: None)
    assert manifest["fields"][0]["datetime_format"] == "%d/%m/%Y"
    data = pl.from_arrow(pq.read_table(out / "model_ready.parquet")).filter(pl.col(ROW_ID) == 1)
    assert data["d_day"].item() == 2 and data["d_month"].item() == 1  # 02/01/2020 is 2 January, in the first chunk


def _csv(n: int = 400) -> str:
    rng = np.random.default_rng(5)
    dates = _dates(n, n - 20)
    return "w,c,site_code,d,batch,y\n" + "".join(
        f"{150 + rng.normal():.3f},{200 + rng.normal() * 10:.3f},{1 + i % 7},{dates[i]},{'pq'[i % 2]}{i % 3},"
        f"{'ab'[i % 2]}\n"
        for i in range(n)
    )


WC2_BAND = {"source": "wc2", "other": "high", "groups": [{"label": "low", "ranges": [[0, 1.5]]}]}
DERIVED = [
    {"name": "wc", "kind": "formula", "spec": {"expr": "w / c"}},
    {"name": "wc2", "kind": "formula", "spec": {"expr": "wc * 2"}},
    {"name": "band", "kind": "groups", "spec": WC2_BAND},
    {"name": "hits", "kind": "count", "spec": {"fields": ["site_code"], "values": ["1", "2"]}},
]


def _with_derived(lead, removed: str | None = None):
    """A dataset with derived fields wc = w / c, wc2 = wc * 2, band (wc2's groups), hits (a count over site_code) and
    wsum = w + c, less `removed`, all assessed in a finalized version; and an experiment of that version that drops
    wsum, wc, w and batch, so wc and w are read only through the derived fields above them."""
    ds, _ = finalized_trees(lead, _csv())
    for d in [*DERIVED, {"name": "wsum", "kind": "formula", "spec": {"expr": "w + c"}}]:
        r = lead.post(f"/api/datasets/{ds['id']}/derived", json=d)
        assert r.status_code == 201, r.text
    if removed:
        r = lead.delete(f"/api/datasets/{ds['id']}/derived/{fields(lead, ds)[removed]['id']}")
        assert r.status_code == 200, r.text
    rid = lead.get(f"/api/datasets/{ds['id']}/baselines").json()[0]["latest_finalized_id"]
    start(lead, ds, "tree", [lead], start="baseline", source=rid)
    rid = submit_and_finalize(lead, ds)["recipe"]["id"]
    x = experiment(lead, ds, rid)
    for name in [n for n in ("wsum", "wc", "w", "batch") if n != removed]:
        adjust(lead, ds, x, name, action="drop")
    return ds, x


def _export(client, ds, x, **rules) -> dict:
    r = client.post(f"/api/datasets/{ds['id']}/exports", json={"experiment_id": x["id"], **rules})
    assert r.status_code == 202, r.text
    e = wait(lambda: (e := client.get(f"/api/exports/{r.json()['id']}").json())["status"] in ("done", "failed") and e)
    assert e["status"] == "done", e.get("error")
    return e


def test_an_export_freezes_the_derived_fields_it_reads_and_its_source_fields_types(people):
    lead = people["lead"]
    ds, x = _with_derived(lead)
    assert fields(lead, ds)["site_code"]["semantic_type"] == "categorical"
    e = _export(lead, ds, x, exclude=[{"field": "batch", "values": ["zz"]}])
    cfg = e["config"]
    assert cfg["derived"] == DERIVED  # wc only through wc2; wsum not at all
    schema = cfg["input_schema"]
    assert set(schema) == {"w", "c", "site_code", "d", "batch", "y"}
    assert schema["w"] == {"dtype": "Float64", "datetime_format": None, "integer_codes": False}
    assert schema["site_code"] == {"dtype": "Int64", "datetime_format": None, "integer_codes": True}
    assert schema["d"] == {"dtype": "String", "datetime_format": "%d/%m/%Y", "integer_codes": False}
    assert schema["batch"] == {"dtype": "String", "datetime_format": None, "integer_codes": False}
    assert schema["y"]["dtype"] == "String" and not schema["y"]["integer_codes"]

    f = fields(lead, ds)
    assert lead.delete(f"/api/datasets/{ds['id']}/derived/{f['hits']['id']}").status_code == 200
    assert "hits" not in fields(lead, ds)
    assert lead.get(f"/api/exports/{e['id']}").json()["config"]["derived"] == cfg["derived"]


def test_an_export_reading_a_derived_field_whose_input_was_removed_still_exports_and_says_so(people):
    lead = people["lead"]
    ds, x = _with_derived(lead, removed="wc")  # removing a derived field doesn't check what reads it
    e = _export(lead, ds, x)
    derived = {d["name"]: d for d in e["config"]["derived"]}
    assert list(derived) == ["wc2", "band", "hits"]
    assert derived["wc2"]["spec"] == {"expr": "wc * 2"} and "'wc'" in derived["wc2"]["unavailable"]
    assert "unavailable" not in derived["band"]
    assert "w" not in e["config"]["input_schema"]  # nothing left reads it


def test_an_export_from_before_the_freeze_finishes_without_either_key(people):
    from app.db import connection
    from app.routers.exports import _run

    lead = people["lead"]
    ds, rid = finalized_trees(lead, _csv())
    e = export(lead, ds, experiment(lead, ds, rid))
    with connection() as conn:
        conn.execute(
            """UPDATE meta.exports SET status = 'pending', config = config - 'derived' - 'input_schema'
                WHERE id = %s""",
            (e["id"],),
        )
    _run(e["id"])
    again = lead.get(f"/api/exports/{e['id']}").json()
    assert again["status"] == "done" and not {"derived", "input_schema"} & set(again["config"])


def test_an_export_pins_a_dates_whole_column_format_when_none_was_recorded(people):
    from app.db import connection

    lead = people["lead"]
    ds, rid = finalized_trees(lead, _csv())
    with connection() as conn:
        conn.execute(
            "UPDATE meta.fields SET stats = stats - 'datetime_format' WHERE dataset_id = %s AND name = 'd'",
            (ds["id"],),
        )
    e = export(lead, ds, experiment(lead, ds, rid))
    assert e["config"]["input_schema"]["d"]["datetime_format"] == "%d/%m/%Y"
    assert e["config"]["derived"] == []
