"""Export to code: the script rebuilds an export exactly, from the original file, outside the app (no database)."""

import ast
import dataclasses
import re
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from app import derived, ingest, profiling
from app.export_code import ExportInfo, render
from app.export_code import script as driver
from app.export_code.bundle import Bundle, BundleError, _Import, _Unit, undefined_names
from app.export_code.gather import _output_summary, fit_settings, input_expectation
from app.export_code.render import _literal, field_call, target_call
from app.export_code.script import keep, predict
from app.semantics import DEFAULT_PARAMS
from app.transform import FieldSpec, run_export

TASKS = {
    "label": {"task": "classes", "bins": None, "decimal_comma": False},
    "score": {"task": "regression", "bins": None, "decimal_comma": False},
    "grade": {"task": "bins", "bins": {"k": 4}, "decimal_comma": False},
    "tier": {"task": "bins", "bins": {"positive": ["a", "b"]}, "decimal_comma": False},
}
DERIVED = [
    {"name": "visits_per_age", "kind": "formula", "spec": {"expr": "(visits + 1) / (age + 1)"}},
    {
        "name": "age band",
        "kind": "groups",
        "spec": {
            "source": "age",
            "groups": [{"label": "young", "ranges": [[0, 30]]}, {"label": "middle", "ranges": [[30.0001, 55]]}],
            "other": "older",
        },
    },
    {"name": "yes_count", "kind": "count", "spec": {"fields": ["smoker", "member"], "values": ["yes"]}},
    {"name": "log_ipv", "kind": "formula", "spec": {"expr": "log1p(abs(visits_per_age) * 100)"}},
]


def messy_csv(n: int = 1500, seed: int = 5) -> str:
    """Semicolon-separated, a BOM and padded names in the header, '?' and blanks for missing values, decimal commas,
    day-first dates, epoch seconds, infinities, exact duplicate rows and a rare class."""
    rng = np.random.default_rng(seed)
    head = (
        "﻿ id ;age;income;visits;city;plan;segment;smoker;member;notes;comment;signup;seen_at;ratio;household;"
        "visit_no;label;score;grade;tier\n"
    )
    words = "great slow refund fast helpful broken price update support login waiting quality".split()
    rows = []
    for i in range(n):
        age = "?" if i % 17 == 0 else f"{rng.integers(18, 80)}"
        income = "" if i % 23 == 0 else f"{rng.normal(50_000, 15_000):.2f}".replace(".", ",")
        visits = f"{rng.poisson(3)}"
        city = "Nowhere" if i % 97 == 0 else str(rng.choice(["Oslo", "Lima", "Pune", "Kyiv", "Quito", "Accra"]))
        plan = str(rng.choice(["basic", "plus", "pro"]))
        segment = f"s{rng.integers(0, 40)}"
        smoker = "" if i % 13 == 0 else str(rng.choice(["yes", "no"]))
        member = str(rng.choice(["yes", "no"]))
        notes = " ".join(rng.choice(words, size=rng.integers(3, 9)))
        comment = "" if i % 7 == 0 else " ".join(rng.choice(words, size=rng.integers(1, 4)))
        day, month = rng.integers(1, 29), rng.integers(1, 13)
        signup = "?" if i % 31 == 0 else f"{day:02d}/{month:02d}/20{rng.integers(10, 24)}"
        seen = f"{1_600_000_000 + int(rng.integers(0, 90_000_000))}"
        ratio = "inf" if i % 211 == 0 else ("-inf" if i % 223 == 0 else f"{rng.normal():.4f}")
        household = f"h{i // 3}"
        label = "" if i % 41 == 0 else str(rng.choice(["low", "mid", "high"], p=[0.5, 0.3, 0.2]))
        score = f"{rng.normal(10, 3):.3f}"
        grade = f"{rng.integers(0, 101)}"
        tier = str(rng.choice(["a", "b", "c", "d"]))
        rows.append(
            ";".join(
                [
                    f"{i}",
                    age,
                    income,
                    visits,
                    city,
                    plan,
                    segment,
                    smoker,
                    member,
                    notes,
                    comment,
                    signup,
                    seen,
                    ratio,
                    household,
                    f"{i % 3}",
                    label,
                    score,
                    grade,
                    tier,
                ]
            )
        )
    rows += rows[100:110]  # exact duplicates
    return head + "\n".join(rows) + "\n"


def specs_for(raw: Path, family: str) -> list[FieldSpec]:
    df = pl.read_parquet(raw)
    comma = profiling.detect_decimal_comma(df["income"])
    fmt = profiling.column_datetime_format(df["signup"])
    epoch = profiling.column_datetime_format(df["seen_at"])
    fill = "none" if family == "tree" else "median"
    return [
        FieldSpec(
            "age",
            "numeric_discrete",
            fill,
            None,
            family != "tree",
            "none",
            "none" if family == "tree" else "standard",
            "none",
            {},
        ),
        FieldSpec(
            "income",
            "numeric_continuous",
            fill if family == "tree" else "mean",
            None,
            False,
            "none" if family == "tree" else "log1p",
            "none" if family == "tree" else "robust",
            "none",
            {},
            decimal_comma=comma,
        ),
        FieldSpec(
            "visits",
            "numeric_discrete",
            "most_frequent" if family != "tree" else "none",
            None,
            False,
            "clip" if family != "tree" else "none",
            "minmax" if family != "tree" else "none",
            "none",
            {},
        ),
        FieldSpec(
            "ratio",
            "numeric_continuous",
            "constant" if family != "tree" else "none",
            "0",
            False,
            "none",
            "quantile_normal" if family != "tree" else "none",
            "none",
            {},
        ),
        FieldSpec(
            "city",
            "categorical",
            "constant",
            "__missing__",
            False,
            "none",
            "none",
            "onehot",
            {"max_categories": 4, "min_frequency": 0.01},
        ),
        FieldSpec(
            "plan", "categorical", "most_frequent", None, False, "none", "none", "ordinal", {"min_frequency": 0.001}
        ),
        FieldSpec("segment", "categorical", "constant", "__missing__", False, "none", "none", "frequency", {}),
        FieldSpec("smoker", "binary", "constant", "__missing__", False, "none", "none", "binary", {}),
        FieldSpec("member", "binary", "most_frequent", None, False, "none", "none", "binary", {}),
        FieldSpec(
            "notes",
            "text",
            "constant",
            "",
            False,
            "none",
            "none",
            "tfidf_svd",
            {"svd_components": 6, "max_features": 500, "min_df": 2},
        ),
        FieldSpec("comment", "text", "constant", "", False, "none", "none", "text_stats", {}),
        FieldSpec(
            "signup",
            "datetime",
            "median" if family != "tree" else "none",
            None,
            False,
            "none",
            "standard" if family != "tree" else "none",
            "datetime_parts",
            {},
            fmt,
        ),
        FieldSpec(
            "seen_at",
            "datetime",
            "median" if family != "tree" else "none",
            None,
            False,
            "none",
            "none",
            "datetime_parts",
            {},
            epoch,
        ),
        FieldSpec("visits_per_age", "numeric_continuous", fill, None, False, "none", "none", "none", {}),
        FieldSpec("age band", "categorical", "constant", "__missing__", False, "none", "none", "onehot", {}),
        FieldSpec("yes_count", "numeric_discrete", fill, None, False, "none", "none", "none", {}),
        FieldSpec("log_ipv", "numeric_continuous", fill, None, False, "none", "none", "none", {}),
    ]


def app_export(tmp: Path, csv: str, family: str, config: dict, targets: list[str]) -> tuple[Path, Path, ExportInfo]:
    """The app's way: ingest the upload, add the derived fields, export. Returns the upload, the export's data and
    what the script is written from."""
    tmp.mkdir(parents=True, exist_ok=True)
    upload = tmp / "customers [v2].csv"  # brackets: a name, never a glob
    upload.write_text(csv)
    raw = tmp / "app" / "raw.parquet"
    ingest.to_parquet(upload, "csv", raw)
    originals = list(pl.read_parquet_schema(raw))
    for d in DERIVED:  # as routers/derived.add_derived does
        df = pl.read_parquet(raw)
        df.with_columns(derived.compute(df, d["kind"], d["spec"]).alias(d["name"])).write_parquet(raw)
    specs = specs_for(raw, family)
    out = tmp / "app" / "export"
    manifest = run_export(
        raw, specs, targets, out, config, lambda *_: None, family=family, target_tasks={t: TASKS[t] for t in targets}
    )
    info = ExportInfo(
        export_id=7,
        dataset_name="Customers 'n\" churn\nimport os",
        family=family,
        specs=specs,
        targets={t: TASKS[t] for t in targets},
        row_rules={
            k: config.get(k, v)
            for k, v in {"dedupe": False, "exclude": [], "first_per": None, "group_by": None}.items()
        },
        split={k: config[k] for k in ("valid_size", "test_size", "seed")},
        **fit_settings(config, manifest, specs),
        source={"kind": "upload", "format": "csv", "file": upload.name},
        derived=DERIVED,
        notes={"age": 'Ages over 80 are typos """\nimport os; os.system("echo pwned")', "city": " x = 1"},
        adjusted={"income": "imputation none → mean"},
        left_out=[("id", "identifier", "dropped"), ("household", "categorical", "dropped")],
        expected_input=input_expectation(raw, originals, [d["name"] for d in DERIVED]),
        expected_output=_output_summary(
            str(out / "model_ready.parquet"),
            (out / "model_ready.parquet").stat().st_mtime_ns,
            tuple(manifest["feature_names"]),
        ),
        baseline="Trees baseline v1",
        experiment="tries",
        adjustments=1,
        split_counts=manifest["split_counts"],
        n_features=manifest["n_features"],
    )
    return upload, out / "model_ready.parquet", info


def run_script(tmp: Path, info: ExportInfo, *args: str) -> subprocess.CompletedProcess:
    script = tmp / "export7_customers.py"
    script.write_text(render(info))
    # -I: no PYTHONPATH, no current directory, no user site: the script has nothing of the app's but itself.
    return subprocess.run(
        [sys.executable, "-I", str(script), *args], capture_output=True, text=True, cwd=tmp, timeout=600
    )


CONFIGS = {
    "rules": {
        "valid_size": 0.15,
        "test_size": 0.2,
        "seed": 3,
        "dedupe": True,
        "group_by": "household",
        "exclude": [{"field": "city", "values": ["Nowhere"]}],
        "chunk_rows": 400,
        "fit_sample_rows": 700,
    },
    "first_per": {
        "valid_size": 0.2,
        "test_size": 0.1,
        "seed": 11,
        "first_per": {"key": "household", "order": "visit_no"},
        "chunk_rows": 10_000,
    },
}


@pytest.mark.parametrize(
    "family,config,targets",
    [
        ("tree", "rules", ["label", "score"]),
        ("linear", "first_per", ["grade", "tier"]),
    ],
)
def test_script_rebuilds_the_export_exactly(tmp_path, family, config, targets):
    upload, model_ready, info = app_export(tmp_path, messy_csv(), family, CONFIGS[config], targets)
    r = run_script(
        tmp_path, info, str(upload), "--out", "mine.parquet", "--clean", "clean.parquet", "--compare", str(model_ready)
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Identical to the app's export." in r.stdout
    assert f"Same values as {model_ready}" in r.stdout
    mine, theirs = pl.read_parquet(tmp_path / "mine.parquet"), pl.read_parquet(model_ready)
    assert mine.schema == theirs.schema
    assert mine.equals(theirs)
    clean = pl.read_parquet(tmp_path / "clean.parquet")
    assert clean["__row_id"].to_list() == theirs["__row_id"].to_list()
    assert clean["__split"].to_list() == theirs["__split"].to_list()
    assert [c for c in clean.columns if c not in ("__row_id", "__split")] == [s.name for s in info.specs] + targets


def test_script_stops_on_different_data(tmp_path):
    upload, _, info = app_export(tmp_path, messy_csv(), "tree", CONFIGS["rules"], ["label"])
    lines = upload.read_text().splitlines()
    lines[5] = lines[5].replace(";plus;", ";pro;").replace(";basic;", ";pro;")
    lines[9] = lines[9].replace(";Oslo;", ";Lima;").replace(";Pune;", ";Lima;").replace(";Kyiv;", ";Lima;")
    upload.write_text("\n".join(lines[:-3]) + "\n")
    r = run_script(tmp_path, info, str(upload))
    assert r.returncode == 3
    assert "doesn't match what the app read" in r.stderr and "rows; the app's copy has" in r.stderr
    assert "Use the file that was uploaded" in r.stderr
    upload.write_text("\n".join(lines) + "\n")
    r = run_script(tmp_path, info, str(upload))
    assert r.returncode == 3 and "different values" in r.stderr
    # --no-check goes on, and still checks the result: these rows' plan and city changed.
    r = run_script(tmp_path, info, str(upload), "--no-check")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "Not checking the data" in r.stderr and "It differs from the app's export" in r.stderr
    r = run_script(tmp_path, info, str(tmp_path / "nowhere.csv"))
    assert r.returncode == 4 and "The rebuild failed." in r.stderr


def test_edited_split_fits_on_the_rows_the_app_would(tmp_path):
    """The script records the export's settings for the fit sample (Max rows to fit on, the memory budget), not how
    many rows that came to, so a changed split fits on the rows the app would fit on."""
    upload, _, info = app_export(tmp_path, messy_csv(), "linear", CONFIGS["first_per"], ["grade"])
    assert (info.fit_sample_rows, info.fit_mb) == (500_000, 512)
    text = render(info).replace(
        "SPLIT = {'valid_size': 0.2, 'test_size': 0.1, 'seed': 11}",
        "SPLIT = {'valid_size': 0.05, 'test_size': 0.05, 'seed': 11}",
    )
    assert "'valid_size': 0.05" in text
    script = tmp_path / "edited.py"
    script.write_text(text)
    r = subprocess.run(
        [sys.executable, "-I", str(script), str(upload), "--out", "edited.parquet"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=600,
    )
    assert r.returncode == 1, r.stdout + r.stderr
    config = {**CONFIGS["first_per"], "valid_size": 0.05, "test_size": 0.05}
    run_export(
        tmp_path / "app" / "raw.parquet",
        info.specs,
        ["grade"],
        tmp_path / "again",
        config,
        lambda *_: None,
        family="linear",
        target_tasks={"grade": TASKS["grade"]},
    )
    assert pl.read_parquet(tmp_path / "edited.parquet").equals(
        pl.read_parquet(tmp_path / "again" / "model_ready.parquet")
    )


def test_fit_settings_fall_back_to_what_the_export_did():
    specs = [FieldSpec("a", "numeric_continuous", "none", None, False, "none", "none", "none", {})]
    m = {
        "fit_sample_rows": 349,
        "chunk_rows": 200_000,
        "split_counts": {"train": 349},
        "n_features": 1,
        "fit_mb": 512,
        "chunk_mb": 256,
    }
    assert fit_settings({}, m, specs) == {
        "fit_sample_rows": 500_000,
        "fit_mb": 512,
        "chunk_rows": 200_000,
        "chunk_mb": 256,
    }
    # The budget was different then, and the settings don't account for the rows used: write the outcome.
    assert fit_settings({"fit_sample_rows": 100_000}, {**m, "fit_sample_rows": 300}, specs)["fit_sample_rows"] == 300
    assert fit_settings({"fit_sample_rows": 100_000}, {**m, "fit_sample_rows": 300}, specs)["fit_mb"] is None


def test_edited_decisions_rebuild_what_the_app_would_export(tmp_path):
    """Change a decision in the script: the result is what the app exports with that decision, and the check says it
    no longer matches the original export."""
    upload, _, info = app_export(tmp_path, messy_csv(), "tree", CONFIGS["rules"], ["label"])
    text = render(info).replace(
        "keep.categorical('city', encoding='onehot', max_categories=4)", "keep.categorical('city', encoding='ordinal')"
    )
    assert "keep.categorical('city', encoding='ordinal')" in text
    script = tmp_path / "edited.py"
    script.write_text(text)
    r = subprocess.run(
        [sys.executable, "-I", str(script), str(upload), "--out", "edited.parquet"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=600,
    )
    assert r.returncode == 1 and "It differs from the app's export" in r.stderr
    app_raw = tmp_path / "app" / "raw.parquet"
    specs = [
        FieldSpec(**{**s.__dict__, "encoding": "ordinal", "params": dict(DEFAULT_PARAMS["ordinal"])})
        if s.name == "city"
        else s
        for s in info.specs
    ]
    run_export(
        app_raw,
        specs,
        ["label"],
        tmp_path / "again",
        CONFIGS["rules"],
        lambda *_: None,
        target_tasks={"label": TASKS["label"]},
    )
    assert pl.read_parquet(tmp_path / "edited.parquet").equals(
        pl.read_parquet(tmp_path / "again" / "model_ready.parquet")
    )


def test_the_script_as_a_module(tmp_path):
    """Imported from Python: load_model_ready, load_clean and xy, and the fitted preprocessor that turns load_clean's
    features into the model-ready ones (for new rows, say)."""
    import importlib.util

    upload, model_ready, info = app_export(tmp_path, messy_csv(), "tree", CONFIGS["rules"], ["label", "score"])
    path = tmp_path / "export7_customers.py"
    path.write_text(render(info))
    spec = importlib.util.spec_from_file_location("export7_customers", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
        prepared = mod.prepare(upload)
        theirs = pl.read_parquet(model_ready)
        assert pl.from_pandas(prepared.data, nan_to_null=False).equals(theirs)
        assert prepared.feature_names == [
            c for c in theirs.columns if c not in ("__row_id", "__split", "label", "score")
        ]
        assert set(prepared.feature_source.values()) == {f.name for f in info.specs}
        assert prepared.target_classes["label"] == ["high", "low", "mid"]

        clean = mod.load_clean(upload)
        X, y = mod.xy(clean, "valid")
        assert list(y.columns) == ["label", "score"] and len(X) == (theirs["__split"] == "valid").sum()
        fresh = prepared.preprocessor.transform(X).astype(np.float32)
        expect = theirs.filter(pl.col("__split") == "valid").select(prepared.feature_names).to_numpy()
        assert np.array_equal(fresh, expect, equal_nan=True)
        X_ready, y_ready = mod.xy(prepared.data, "valid")
        assert list(X_ready.columns) == prepared.feature_names and y_ready["label"].dtype == np.int32
    finally:
        del sys.modules[spec.name]


# ---- the script's text ------------------------------------------------------------------------------------------------
def test_names_and_notes_never_become_code(tmp_path):
    _, _, info = app_export(tmp_path, messy_csv(300), "tree", CONFIGS["first_per"], ["label"])
    weird = 'x"""\nimport os\n\' y\\'
    info.specs = [FieldSpec(weird, "numeric_continuous", "none", None, False, "none", "none", "none", {}), *info.specs]
    info.notes[weird] = weird
    info.adjusted[weird] = weird
    info.left_out.append((weird, "constant", weird))
    info.experiment = weird
    text = render(info)
    plain = render(app_export(tmp_path / "plain", messy_csv(300), "tree", CONFIGS["first_per"], ["label"])[2])
    assert _top_level(text) == _top_level(plain)  # the same statements, whatever the names and notes hold
    module = types.ModuleType("export_script")
    sys.modules[module.__name__] = module
    try:
        exec(compile(text, "export_script", "exec"), module.__dict__)
    finally:
        del sys.modules[module.__name__]
    assert module.kept_fields()[0].name == weird
    assert undefined_names(text) == set()


def _top_level(text: str) -> list[tuple[str, ...]]:
    out = []
    for n in ast.parse(text).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            out.append((type(n).__name__, n.name))
        elif isinstance(n, ast.Assign):
            out.append(("Assign", *[t.id for t in n.targets if isinstance(t, ast.Name)]))
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out.append((type(n).__name__, *[a.asname or a.name for a in n.names]))
        else:
            out.append((type(n).__name__,))
    return out


def test_decisions_read_back_as_the_same_specs():
    specs = [
        FieldSpec("a", "numeric_continuous", "none", None, False, "none", "none", "none", {}),
        FieldSpec("b", "numeric_discrete", "constant", "0", True, "clip", "standard", "none", {}, decimal_comma=True),
        FieldSpec(
            "c",
            "categorical",
            "constant",
            "__missing__",
            False,
            "none",
            "none",
            "onehot",
            dict(DEFAULT_PARAMS["onehot"]),
        ),
        FieldSpec(
            "d",
            "categorical",
            "most_frequent",
            None,
            False,
            "none",
            "none",
            "onehot",
            {"max_categories": 9, "min_frequency": 0.01},
        ),
        FieldSpec("e", "categorical", "constant", "zz", False, "none", "none", "ordinal", {}),
        FieldSpec("f", "binary", "constant", "yes", False, "none", "none", "binary", {}),
        FieldSpec("g", "text", "constant", "", False, "none", "none", "tfidf_svd", {"svd_components": 4}),
        FieldSpec("h", "text", "constant", "", False, "none", "none", "text_stats", {}),
        FieldSpec("i", "datetime", "median", None, False, "none", "minmax", "datetime_parts", {}, "%d/%m/%Y"),
        FieldSpec("j", "categorical", "constant", None, True, "log1p", "none", "frequency", {}),  # odd, still exact
        FieldSpec("k", "numeric_continuous", "none", None, False, "none", "none", "onehot", {"x": 1}),
    ]
    ns = {"keep": keep, "predict": predict}
    for s in specs:
        call = field_call(s)
        assert eval(call, ns) == s, call
    assert field_call(specs[0]) == "keep.numeric('a')"
    assert field_call(specs[2]) == "keep.categorical('c', encoding='onehot')"
    assert (
        field_call(specs[3])
        == "keep.categorical('d', encoding='onehot', imputation='most_frequent', impute_value=None, max_categories=9)"
    )
    assert field_call(specs[9]).startswith("keep.field('j', 'categorical'")
    for task in [
        *TASKS.values(),
        {"task": "bins", "bins": {"edges": [1.5, 3.0]}, "decimal_comma": True},
        {"task": "classes", "bins": {"k": 3}, "decimal_comma": False},
    ]:
        call = target_call(task)
        assert eval(call, ns) == task, call
    assert target_call(TASKS["grade"]) == "predict.bands(4)"
    assert target_call(TASKS["tier"]) == "predict.one_vs_rest(['a', 'b'])"


# ---- the bundle -------------------------------------------------------------------------------------------------------
def test_bundle_copies_what_is_reached_and_nothing_else():
    b = Bundle()
    b.add("app.bands", "thin_bands")
    body = b.body()
    assert "def thin_bands(" in body and "def assign(" in body and "def _fmt(" in body
    assert "def quantile_edges(" not in body  # not reached from thin_bands
    assert b.requirements() == [f"numpy=={np.__version__}"]
    assert undefined_names("\n".join(b.import_lines()) + "\n" + body) == set()


def test_bundle_refuses_two_definitions_of_one_name():
    b = Bundle()
    b.add("app.bands", "band_labels")  # reaches bands._fmt
    with pytest.raises(BundleError, match="defined in both"):
        b._claim(_Unit("app.other", ("_fmt",), "def _fmt(v):\n    return str(v)", 1))


def test_undefined_names_sees_inside_lambdas():
    assert undefined_names("X = {'a': lambda: Missing()}\ndef f():\n    return X") == {"Missing"}


# ---- the checks -------------------------------------------------------------------------------------------------------
def _model_ready(n: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    f = rng.normal(size=n).astype(np.float32)
    f[3], f[4], f[7] = np.inf, -np.inf, np.nan
    return pd.DataFrame(
        {
            "__row_id": np.arange(n, dtype=np.int64),
            "__split": np.array(["train"] * n, dtype=object),
            "f": f,
            "label": rng.integers(0, 3, n).astype(np.int32),
            "score": rng.normal(10, 3, n),
        }
    )


def test_the_result_check_tells_rounding_from_real_differences(monkeypatch):
    data = _model_ready()
    expected = driver.output_summary(pl.from_pandas(data, nan_to_null=False), ["f"])
    assert "inf" not in _literal(expected)  # infinite features are counted, not summed, so the script can be written
    monkeypatch.setattr(driver, "EXPECTED_OUTPUT", expected)
    assert driver.output_problems(data) == ([], [])
    f, score = data["f"].to_numpy(), data["score"].to_numpy()
    last_digit = data.assign(
        f=np.where(np.isfinite(f), np.nextafter(f, np.float32(np.inf)), f), score=np.nextafter(score, np.inf)
    )
    assert driver.output_problems(last_digit) == ([], ["f", "score"])
    swapped = f.copy()
    swapped[100], swapped[200] = f[200], f[100]
    assert driver.output_problems(data.assign(f=swapped))[0]  # values in other rows aren't rounding
    moved = f.copy()
    moved[7], moved[8] = f[8], np.nan
    assert driver.output_problems(data.assign(f=moved))[0]
    shifted = f.copy()
    shifted[50] += 3.0
    assert driver.output_problems(data.assign(f=shifted))[0]
    label = data["label"].to_numpy().copy()
    label[0] = (label[0] + 1) % 3
    assert driver.output_problems(data.assign(label=label))[0]


def test_compare_allows_rounding_in_features_only(tmp_path):
    data = _model_ready()
    theirs = tmp_path / "theirs.parquet"
    pl.from_pandas(data, nan_to_null=False).write_parquet(theirs)
    f, score = data["f"].to_numpy(), data["score"].to_numpy()
    assert driver.compare_files(data, theirs) == ([], [])
    diffs, rounding = driver.compare_files(data.assign(f=f * np.float32(1 + 2e-5)), theirs)
    assert diffs == [] and rounding[0].startswith("f (by up to")
    assert driver.compare_files(data.assign(f=f + np.float32(0.01)), theirs)[0]
    assert driver.compare_files(data.assign(score=score * (1 + 1e-6)), theirs)[0]  # number targets: tight
    label = data["label"].to_numpy().copy()
    label[5] = (label[5] + 1) % 3
    assert driver.compare_files(data.assign(label=label), theirs)[0]


def test_the_command_line_by_source(tmp_path):
    upload, _, info = app_export(tmp_path, messy_csv(300), "tree", CONFIGS["first_per"], ["label"])
    # argparse formats help with %: a file name holding one must not break --help.
    r = run_script(tmp_path, dataclasses.replace(info, source={**info.source, "file": "growth 50%.csv"}), "--help")
    assert r.returncode == 0 and "growth 50%.csv" in r.stdout and "Exit status" in r.stdout
    generated = dataclasses.replace(
        info,
        source={
            "kind": "generated",
            "settings": {"rows": 300, "fields": 30, "seed": 42, "target_kind": "mixed", "numeric_target": False},
        },
    )
    r = run_script(tmp_path, generated, str(upload))
    assert r.returncode == 2 and "unrecognized arguments" in r.stderr  # a generated sample takes no file
    library = dataclasses.replace(
        info,
        source={
            "kind": "library",
            "key": "concrete_strength",
            "file": "concrete_strength.parquet",
            "source_url": "https://x",
        },
    )
    r = run_script(tmp_path, library, str(upload))
    assert r.returncode == 2 and "concrete_strength.parquet" in r.stderr


def test_encoding_settings_never_become_other_code():
    for params in (
        {"max_categories": 30, "min_frequency": 0.01, "class": 1},
        {"max_categories": 30, "\uff4din_frequency": 0.5},  # NFKC reads it as min_frequency
        {"max_categories": 30, "min_frequency": 0.01, "lambda": [1, {"x": 2}]},
    ):
        spec = FieldSpec("city", "categorical", "constant", "__missing__", False, "none", "none", "onehot", params)
        call = field_call(spec)
        ast.parse(call)
        assert "params=" in call and eval(call, {"keep": keep}) == spec, call


PEP723 = r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$"


def test_nothing_in_the_header_extends_the_script_metadata(tmp_path):
    _, _, info = app_export(tmp_path, messy_csv(300), "tree", CONFIGS["first_per"], ["label"])
    info.warnings = ["when\n///\nx", "/// script", "x" * 112 + " /// script", "x" * 112 + " ///"]
    info.dataset_name = "///"
    text = render(info)
    blocks = [m for m in re.finditer(PEP723, text) if m.group("type") == "script"]
    assert len(blocks) == 1
    meta = tomllib.loads(
        "".join(
            ln[2:] if ln.startswith("# ") else ln[1:] for ln in blocks[0].group("content").splitlines(keepends=True)
        )
    )
    assert meta["requires-python"] and meta["dependencies"]
    ast.parse(text)


# ---- the bundle, binding names --------------------------------------------------------------------------------------
def test_bundle_refuses_a_name_bound_to_two_things():
    b = Bundle()
    b.add("app.bands", "thin_bands")  # imports numpy as np
    with pytest.raises(BundleError, match="already binds"):
        b._external(_Import("numpy.random", None, "np"))
    with pytest.raises(BundleError, match="already binds"):
        b._claim(_Unit("app.other", ("np",), "np = 1", 1))
    b = Bundle(provided={"SPLIT"}, provider="app.export_code.script")
    with pytest.raises(BundleError, match="already binds"):
        b._claim(_Unit("app.other", ("SPLIT",), "SPLIT = '__split'", 1))
