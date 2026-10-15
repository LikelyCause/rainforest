"""What an export script runs: `export_code.render` copies this module into every script it writes, after the
decisions the export was made with, together with the app functions it calls (see `bundle`).

The script rebuilds the export outside the app: it reads the original data the way the app did, adds the derived
fields, checks the result against what the app read, then runs the export's own steps (row rules and split, targets,
preprocessing fitted on the same training rows, the transform) and checks its output against the export.

The names in PROVIDED are the decisions: every script defines them above this code, so here they are placeholders.
This module must stay bundleable: standard library, third-party packages and app functions only.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from ..derived import compute
from ..ingest import _sink
from ..parquet_rows import read_rows
from ..sample_library.build import build
from ..semantics import DEFAULT_PARAMS, FAMILIES, NON_FEATURE_TYPES
from ..transform import (ROW_ID, SPLIT_COL, FieldSpec, _feature_sources, build_preprocessor, coercion_exprs,
                         estimated_width, prepare_targets, readable_feature_names, rows_for_budget, safe_feature_names,
                         to_pandas_frame, transform_into)

PROVIDED = {"SCRIPT", "SOURCE", "raw_frame", "DERIVED", "FAMILY", "kept_fields", "targets", "ROW_RULES", "SPLIT",
            "FIT_SAMPLE_ROWS", "FIT_MB", "CHUNK_ROWS", "CHUNK_MB", "EXPECTED_INPUT", "EXPECTED_OUTPUT"}

# Exit codes: the result matches the app's export (exactly or but for floating-point rounding), it differs, the
# command line was wrong (argparse's own), the data doesn't read as the app read it, or the rebuild failed.
MATCHES, DIFFERS, USAGE, DATA_DIFFERS, FAILED = 0, 1, 2, 3, 4
# How far a float may move and still count as rounding: relative to the column's sums for the export's check, and
# relative to the column's largest value for --compare (scikit-learn's linear algebra differs a little by CPU).
OUTPUT_RTOL, INPUT_RTOL, COMPARE_RTOL = 1e-5, 1e-9, 1e-4

# ---- placeholders for PROVIDED (each script defines these) -----------------------------------------------------------
SCRIPT: dict[str, Any] = {"export_id": 0, "file": "export.py"}
SOURCE: dict[str, Any] = {"kind": "upload", "format": "csv", "file": "data.csv"}
DERIVED: list[dict[str, Any]] = []
FAMILY = "tree"
ROW_RULES: dict[str, Any] = {}
SPLIT: dict[str, Any] = {"valid_size": 0.15, "test_size": 0.15, "seed": 42}
FIT_SAMPLE_ROWS: int | None = None
FIT_MB: int | None = None
CHUNK_ROWS = 200_000
CHUNK_MB: int | None = None
EXPECTED_INPUT: dict[str, Any] = {}
EXPECTED_OUTPUT: dict[str, Any] = {}


def raw_frame(path: str | Path | None = None) -> pl.DataFrame:
    raise NotImplementedError


def kept_fields() -> list[FieldSpec]:
    raise NotImplementedError


def targets() -> dict[str, dict]:
    raise NotImplementedError


# ---- the decisions' vocabulary --------------------------------------------------------------------------------------
class keep:
    """One call per kept field. The arguments are the app's own settings; the ones left out take the app's default
    for that type. `keep.field` takes every setting, for anything the others can't say."""

    @staticmethod
    def numeric(name: str, *, discrete: bool = False, imputation: str = "none", impute_value: str | None = None,
                add_missing_indicator: bool = False, transform: str = "none", scaling: str = "none",
                decimal_comma: bool = False) -> FieldSpec:
        return FieldSpec(name, "numeric_discrete" if discrete else "numeric_continuous", imputation, impute_value,
                         add_missing_indicator, transform, scaling, "none", {}, None, decimal_comma)

    @staticmethod
    def categorical(name: str, *, encoding: str, imputation: str = "constant", impute_value: str | None = "__missing__",
                    params: dict | None = None, **more) -> FieldSpec:
        return FieldSpec(name, "categorical", imputation, impute_value, False, "none", "none", encoding,
                         _params(encoding, params, more))

    @staticmethod
    def binary(name: str, *, imputation: str = "constant", impute_value: str | None = "__missing__") -> FieldSpec:
        return FieldSpec(name, "binary", imputation, impute_value, False, "none", "none", "binary", {})

    @staticmethod
    def text(name: str, *, encoding: str = "tfidf_svd", impute_value: str | None = "", params: dict | None = None,
             **more) -> FieldSpec:
        return FieldSpec(name, "text", "constant", impute_value, False, "none", "none", encoding,
                         _params(encoding, params, more))

    @staticmethod
    def datetime(name: str, *, datetime_format: str | None = None, imputation: str = "none",
                 scaling: str = "none") -> FieldSpec:
        return FieldSpec(name, "datetime", imputation, None, False, "none", scaling, "datetime_parts", {},
                         datetime_format)

    @staticmethod
    def field(name: str, semantic_type: str, **settings) -> FieldSpec:
        return FieldSpec(name=name, semantic_type=semantic_type, **settings)


def _params(encoding: str, params: dict | None, more: dict) -> dict:
    return {**(dict(DEFAULT_PARAMS.get(encoding, {})) if params is None else dict(params)), **more}


class predict:
    """How a target is predicted: its classes, a number, bands of a number, or some classes against the rest."""

    @staticmethod
    def classes(*, decimal_comma: bool = False) -> dict:
        return {"task": "classes", "bins": None, "decimal_comma": decimal_comma}

    @staticmethod
    def number(*, decimal_comma: bool = False) -> dict:
        return {"task": "regression", "bins": None, "decimal_comma": decimal_comma}

    @staticmethod
    def bands(k: int | None = None, *, edges: list[float] | None = None, decimal_comma: bool = False) -> dict:
        """`k` bands with equal counts of training rows, or bands cut at `edges`."""
        return {"task": "bins", "bins": {"edges": list(edges)} if edges is not None else {"k": k},
                "decimal_comma": decimal_comma}

    @staticmethod
    def one_vs_rest(positive: list[str], *, decimal_comma: bool = False) -> dict:
        return {"task": "bins", "bins": {"positive": list(positive)}, "decimal_comma": decimal_comma}


# ---- reading the data -----------------------------------------------------------------------------------------------
class SourceMismatch(Exception):
    """The data read doesn't match what the app read for this export."""


class UsageError(ValueError):
    """The data file is missing or isn't the kind this export reads."""


def read_upload(path: str | Path | None, fmt: str) -> pl.DataFrame:
    """A file as the app ingested it: the same separator guess, missing-value markers, column names and types."""
    if path is None:
        raise UsageError(f"Give the path of {SOURCE.get('file') or 'the data file'}: python {SCRIPT['file']} PATH")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "raw.parquet"
        for note in _sink(Path(path), fmt, out):
            print(f"Note: {note}", file=sys.stderr)
        return pl.read_parquet(out)


def read_library(path: str | Path | None, key: str) -> pl.DataFrame:
    """One of the app's sample datasets: the given copy of its file, else rebuilt from its public source the way
    the app's copy was built."""
    if path is not None:
        if Path(path).suffix.lower() != ".parquet":
            raise UsageError(f"Give the app's own copy of this dataset, {key}.parquet (api/app/sample_library/data/ in "
                             f"Rainforest's code), or no file to rebuild it from its public source.")
        return read_upload(path, "parquet")
    print(f"Downloading {key} from its public source ({SOURCE.get('source_url')}) and building it as the app's copy "
          "was built…", file=sys.stderr)
    with tempfile.TemporaryDirectory() as tmp:
        built = Path(tmp) / "library.parquet"
        build(key).write_parquet(built, compression="zstd", compression_level=10)
        return read_upload(built, "parquet")


def read_source(path: str | Path | None = None, check: bool = True) -> pl.DataFrame:
    """The dataset as the app holds it: the data read the app's way plus the derived fields the export uses. Unless
    `check` is False, it must match what the app read, column by column."""
    df = raw_frame(path)
    for d in DERIVED:
        df = df.with_columns(compute(df, d["kind"], d["spec"]).alias(d["name"]))
    problems = input_problems(df) if check and EXPECTED_INPUT else []
    if problems:
        raise SourceMismatch(
            "This data doesn't match what the app read for this export:\n  - " + "\n  - ".join(problems[:12])
            + (f"\n  - and {len(problems) - 12} more" if len(problems) > 12 else "") + "\n" + _remedy())
    return df


def _remedy() -> str:
    kind = SOURCE["kind"]
    if kind == "generated":
        return ("The sample made again here differs from the app's. Pass check=False (--no-check) to go on; the result "
                "is still checked against the app's export.")
    if kind == "library":
        return ("Use the app's copy of the file, or no file to rebuild it from its public source; or pass check=False "
                "(--no-check) to go on anyway.")
    return "Use the file that was uploaded, or pass check=False (--no-check) to go on anyway."


def input_problems(df: pl.DataFrame) -> list[str]:
    want = EXPECTED_INPUT
    out = []
    if df.height != want["rows"]:
        out.append(f"{df.height:,} rows; the app's copy has {want['rows']:,}")
    missing = [c for c in want["columns"] if c not in df.columns]
    if missing:
        out.append(f"missing column(s): {', '.join(missing)}")
    extra = [c for c in df.columns if c not in want["columns"]]
    if extra:
        out.append(f"column(s) the app's copy doesn't have: {', '.join(extra)}")
    if out:
        return out
    for name, exp in want["columns"].items():
        if exp is None:  # a derived field: computed from the columns checked here, and the export's check covers it
            continue
        got = input_summary(df[name])
        if got["dtype"] != exp["dtype"]:
            out.append(f"{name}: read as {got['dtype']}; the app read {exp['dtype']}")
        elif got["nulls"] != exp["nulls"]:
            out.append(f"{name}: {got['nulls']:,} missing values; the app read {exp['nulls']:,}")
        elif got["digest"] != exp["digest"] and not ("sum" in exp and rounding_only(got, exp, INPUT_RTOL)):
            out.append(f"{name}: different values")
    return out


@contextmanager
def _staged(df: pl.DataFrame):
    """The dataset as a Parquet file, the form the export's steps read (the app's own copy has row groups of
    100,000 rows, like this one)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "raw.parquet"
        df.write_parquet(path, row_group_size=100_000)
        yield path


# ---- rebuilding the export ------------------------------------------------------------------------------------------
@dataclass
class Prepared:
    """The rebuilt export. `data` has the columns of the app's model_ready.parquet: __row_id (the row's position in
    the source, from 0), __split (train, valid or test), the features, then the targets as the models see them."""
    data: pd.DataFrame
    preprocessor: Any  # the fitted sklearn ColumnTransformer; .transform() takes load_clean()'s feature columns
    feature_names: list[str]
    feature_source: dict[str, str]  # feature -> the field it came from
    target_classes: dict[str, list[str]]  # class targets and bands: the label of each code
    split_info: dict[str, Any]


def _plan() -> tuple[list[FieldSpec], dict[str, dict], dict[str, Any], int]:
    specs = [f for f in kept_fields() if f.semantic_type not in NON_FEATURE_TYPES]
    tasks = targets()
    if not specs:
        raise ValueError("No fields are kept: nothing to export.")
    if not tasks:
        raise ValueError("No targets.")
    return specs, tasks, {**ROW_RULES, **SPLIT}, int(SPLIT.get("seed", 42))


def prepare(path: str | Path | None = None, check: bool = True) -> Prepared:
    """Rebuild the export from the data at `path`, step by step as the app's export does."""
    specs, tasks_in, config, seed = _plan()
    names = list(tasks_in)
    df = read_source(path, check)
    with _staged(df) as raw:
        del df
        _, y_enc, classes, split, split_info, info = prepare_targets(raw, names, tasks_in, config, seed)
        tasks = info["target_tasks"]
        info.pop("_groups")
        # Preprocessing is fitted on a sample of the training rows, drawn as the app drew it.
        train_idx = np.flatnonzero(split == 0)
        size = train_idx.size if FIT_SAMPLE_ROWS is None else min(FIT_SAMPLE_ROWS, train_idx.size)
        if FIT_MB is not None:  # fitting holds the sample's whole output, so wide exports fit on fewer rows
            size = min(size, rows_for_budget(FIT_MB, estimated_width(specs)))
        fit_idx = np.sort(np.random.default_rng(seed).choice(train_idx, size=size, replace=False))
        cols = [f.name for f in specs]
        pre = build_preprocessor(specs, seed)
        pre.fit(to_pandas_frame(read_rows(raw, cols, fit_idx), specs, coercion_exprs(specs), consume=True))
        feat_names = safe_feature_names(readable_feature_names(pre), reserved=[ROW_ID, SPLIT_COL, *names])
        coercers = coercion_exprs(specs)
        chunk_rows = CHUNK_ROWS if CHUNK_MB is None else min(CHUNK_ROWS, rows_for_budget(CHUNK_MB, len(feat_names)))
        blocks, kept_rows, offset = [], [], 0
        nan_counts = np.zeros(len(feat_names), dtype=np.int64)
        for batch in pq.ParquetFile(raw).iter_batches(batch_size=chunk_rows, columns=cols):
            chunk = pl.from_arrow(batch)
            rows = np.arange(offset, offset + chunk.height)
            offset += chunk.height
            keep_rows = split[rows] != 255
            if not keep_rows.any():
                continue
            X = transform_into(pre, to_pandas_frame(chunk.filter(pl.Series(keep_rows)), specs, coercers, consume=True),
                               len(feat_names))
            nan_counts += np.isnan(X).sum(axis=0)
            if not FAMILIES[FAMILY]["handles_nan"] and nan_counts.any():
                bad = [feat_names[k] for k in np.flatnonzero(nan_counts)][:10]
                raise ValueError(f"{FAMILY} models need complete data but these features still contain NaN: "
                                 f"{', '.join(bad)}. Set an imputation for the source fields.")
            blocks.append(X)
            kept_rows.append(rows[keep_rows])
    rows = np.concatenate(kept_rows) if kept_rows else np.zeros(0, dtype=np.int64)
    X = np.vstack(blocks) if blocks else np.zeros((0, len(feat_names)), dtype=np.float32)
    split_names = np.array(["train", "valid", "test"], dtype=object)
    data: dict[str, Any] = {ROW_ID: rows.astype(np.int64), SPLIT_COL: split_names[split[rows]]}
    data.update({n: X[:, k] for k, n in enumerate(feat_names)})
    data.update({t: y_enc[t].to_numpy()[rows].astype(np.float64 if tasks[t] == "regression" else np.int32)
                 for t in names})
    return Prepared(pd.DataFrame(data), pre, feat_names, _feature_sources(pre, feat_names),
                    {t: classes[t] for t in names if tasks[t] != "regression"}, split_info)


def load_model_ready(path: str | Path | None = None, check: bool = True) -> pd.DataFrame:
    """The export's model_ready data, rebuilt: the same rows, split, features and target codes as the app's."""
    return prepare(path, check).data


def load_clean(path: str | Path | None = None, check: bool = True) -> pd.DataFrame:
    """The export's rows with its decisions applied and nothing fitted: __row_id, __split, each kept field as the
    preprocessing receives it (numbers as floats, dates as datetimes, the rest as text, missing values still
    missing), and the targets as labels (numbers stay numbers; bands are cut later, on the training rows)."""
    specs, tasks_in, config, seed = _plan()
    names = list(tasks_in)
    df = read_source(path, check)
    with _staged(df) as raw:
        y_raw, _, _, split, _, _ = prepare_targets(raw, names, tasks_in, config, seed)
    rows = np.flatnonzero(split != 255)
    cols = [f.name for f in specs]
    features = to_pandas_frame(df.select(cols)[rows], specs, coercion_exprs(specs), consume=True)
    head = pd.DataFrame({ROW_ID: rows.astype(np.int64),
                         SPLIT_COL: np.array(["train", "valid", "test"], dtype=object)[split[rows]]})
    return pd.concat([head, features, y_raw.iloc[rows][names].reset_index(drop=True)], axis=1)


def xy(data: pd.DataFrame, split: str = "train") -> tuple[pd.DataFrame, pd.DataFrame]:
    """The features and the targets of one split (train, valid or test) of load_model_ready() or load_clean()."""
    part = data[data[SPLIT_COL] == split]
    names = list(targets())
    return part.drop(columns=[ROW_ID, SPLIT_COL, *names]), part[names]


# ---- checking -------------------------------------------------------------------------------------------------------
def column_digest(s: pl.Series) -> str:
    """A hash of a column's type, values and missing positions, the same on any machine for the same column."""
    h = hashlib.sha256(str(s.dtype).encode())
    h.update(np.packbits(s.is_null().to_numpy()).tobytes())
    v = s.drop_nulls()
    if v.dtype == pl.String:
        h.update(v.str.len_bytes().to_numpy().tobytes())
        h.update("".join(v.to_list()).encode())
    elif v.dtype == pl.Boolean:
        h.update(v.cast(pl.UInt8).to_numpy().tobytes())
    elif v.dtype.is_temporal():
        h.update(v.to_physical().to_numpy().tobytes())
    elif v.dtype.is_numeric():
        h.update(np.ascontiguousarray(v.to_numpy()).tobytes())
    else:
        h.update("\x1f".join(map(str, v.to_list())).encode())
    return h.hexdigest()[:20]


def _weights(n: int) -> np.ndarray:
    """A fixed weight in [-1, 1) for each row position, the same on every machine. A sum weighted by them changes when
    values move to other rows, which plain sums don't notice."""
    h = (np.arange(n, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)
    return h.astype(np.float64) / float(1 << 23) - 1.0


def column_summary(s: pl.Series, sums: bool = False) -> dict[str, Any]:
    """dtype, missing values and digest; with `sums`, what a float column is compared by when it isn't identical:
    where its NaN and infinite values are, and plain and row-weighted sums of the finite ones."""
    out: dict[str, Any] = {"dtype": str(s.dtype), "nulls": int(s.null_count()), "digest": column_digest(s)}
    if sums:
        x = s.cast(pl.Float64).to_numpy()
        finite = np.isfinite(x)
        v = np.where(finite, x, 0.0)
        wv = _weights(len(x)) * v
        special = hashlib.sha256(np.packbits(np.isnan(x)).tobytes() + np.packbits(np.isposinf(x)).tobytes()
                                 + np.packbits(np.isneginf(x)).tobytes()).hexdigest()[:20]
        out.update(nans=int(np.isnan(x).sum()), special=special, sum=float(v.sum()), abs=float(np.abs(v).sum()),
                   wsum=float(wv.sum()), wabs=float(np.abs(wv).sum()))
    return out


def input_summary(s: pl.Series) -> dict[str, Any]:
    """What the script checks a column of the data it reads against: sums too for floats, whose last digits a
    computed column (a generated sample) can round differently on another machine."""
    return column_summary(s, s.dtype.is_float())


def output_summary(df: pl.DataFrame, features: list[str]) -> dict[str, Any]:
    feats = set(features)
    return {"rows": df.height,
            "columns": {c: column_summary(df[c], c in feats or df[c].dtype.is_float()) for c in df.columns}}


def rounding_only(got: dict[str, Any], exp: dict[str, Any], rtol: float) -> bool:
    """Whether a float column that isn't identical differs only as floating-point rounding would: the same NaN and
    infinite values in the same rows, and plain, absolute and row-weighted sums within `rtol`."""
    if got["dtype"] != exp["dtype"] or got.get("special") != exp.get("special"):
        return False
    scale, wscale = max(exp["abs"], 1e-12), max(exp["wabs"], 1e-12)
    return (abs(got["sum"] - exp["sum"]) <= rtol * scale and abs(got["abs"] - exp["abs"]) <= rtol * scale
            and abs(got["wsum"] - exp["wsum"]) <= rtol * wscale)


def output_problems(data: pd.DataFrame, rtol: float = OUTPUT_RTOL) -> tuple[list[str], list[str]]:
    """(differences, columns that differ only by floating-point rounding) against the app's export."""
    want = EXPECTED_OUTPUT
    df = pl.from_pandas(data, nan_to_null=False)  # NaN stays NaN, as in the app's Parquet
    if df.height != want["rows"]:
        return [f"{df.height:,} rows; the app's export has {want['rows']:,}"], []
    if list(df.columns) != list(want["columns"]):
        missing = [c for c in want["columns"] if c not in df.columns]
        extra = [c for c in df.columns if c not in want["columns"]]
        return [f"columns differ (missing: {', '.join(missing) or 'none'}; not in the app's: "
                f"{', '.join(extra) or 'none'})"], []
    diffs, rounding = [], []
    for c, exp in want["columns"].items():
        got = column_summary(df[c], "sum" in exp)
        if got["digest"] == exp["digest"]:
            continue
        if "sum" in exp and rounding_only(got, exp, rtol):
            rounding.append(c)
        elif "sum" in exp and got["dtype"] == exp["dtype"] and got["special"] == exp["special"]:
            diffs.append(f"{c}: values differ (sum {got['sum']:.6g}; the app's {exp['sum']:.6g})")
        else:
            diffs.append(f"{c}: different values")
    return diffs, rounding


def compare_files(data: pd.DataFrame, path: str | Path, rtol: float = COMPARE_RTOL) -> tuple[list[str], list[str]]:
    """Value by value against the app's model_ready.parquet at `path`: (differences, float columns whose values
    differ only by floating-point rounding, each with its largest difference). Row ids, the split and class codes
    must be equal; a feature (float32) may differ by `rtol` of its column's largest value, a number target (float64)
    by INPUT_RTOL of it."""
    theirs = pl.read_parquet(path)
    mine = pl.from_pandas(data, nan_to_null=False)
    if mine.columns != theirs.columns:
        return [f"columns differ: {len(mine.columns)} here, {len(theirs.columns)} in {Path(path).name}"], []
    if mine.height != theirs.height:
        return [f"{mine.height:,} rows here, {theirs.height:,} in {Path(path).name}"], []
    diffs, rounding = [], []
    for c in mine.columns:
        a, b = mine[c], theirs[c]
        if a.dtype != b.dtype:
            diffs.append(f"{c}: {a.dtype} here, {b.dtype} there")
        elif a.dtype.is_float():
            tol = rtol if a.dtype == pl.Float32 else INPUT_RTOL  # features are float32; number targets float64
            x, y = a.to_numpy(), b.to_numpy()
            same = (x == y) | (np.isnan(x) & np.isnan(y))
            if same.all():
                continue
            finite = np.isfinite(x) & np.isfinite(y)
            gap = float(np.max(np.abs(x[finite] - y[finite]), initial=0.0))
            scale = float(np.max(np.abs(y[np.isfinite(y)]), initial=0.0))
            if (np.isfinite(x) == np.isfinite(y)).all() and (same | finite).all() and gap <= tol * max(scale, 1e-12):
                rounding.append(f"{c} (by up to {gap:.3g})")
            else:
                diffs.append(f"{c}: {int((~same).sum()):,} value(s) differ, by up to {gap:.3g}")
        elif not a.equals(b):
            diffs.append(f"{c}: {int((a != b).sum()):,} value(s) differ")
    return diffs, rounding


# ---- the command line -----------------------------------------------------------------------------------------------
def write_model_ready(data: pd.DataFrame, path: str | Path, regression: set[str]) -> None:
    """Parquet with the app's column types: int64 row ids, the split as text, float32 features (NaN where missing,
    not null), int32 class codes and float64 numbers."""
    names = set(targets())
    types = [pa.int64(), pa.string()] + [(pa.float64() if c in regression else pa.int32()) if c in names
                                         else pa.float32() for c in data.columns[2:]]
    arrays = [pa.array(data[c].to_numpy(), t) for c, t in zip(data.columns, types)]
    pq.write_table(pa.Table.from_arrays(arrays, schema=pa.schema(list(zip(data.columns, types)))), path,
                   compression="zstd")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog=SCRIPT["file"], description=f"Rebuild Rainforest export #{SCRIPT['export_id']}.",
                                 epilog="Exit status: 0 the result matches the app's export, 1 it differs, 2 the "
                                        "command line is wrong, 3 the data doesn't read as the app read it, 4 the "
                                        "rebuild failed.")
    kind = SOURCE["kind"]
    if kind == "upload":
        name = str(SOURCE.get("file") or "as uploaded").replace("%", "%%")  # argparse formats help with %
        ap.add_argument("source", help=f"the data file ({name})")
    elif kind == "library":
        ap.add_argument("source", nargs="?", help="the app's copy of the data file (default: rebuild it from its "
                                                  "public source)")
    ap.add_argument("--out", metavar="PARQUET", help="write the model-ready data (as the app's model_ready.parquet)")
    ap.add_argument("--clean", metavar="PARQUET", help="write the clean data: decisions applied, nothing fitted")
    ap.add_argument("--compare", metavar="PARQUET", help="compare value by value with the app's model_ready.parquet")
    ap.add_argument("--no-check", action="store_true",
                    help="go on when the data doesn't read as the app read it (the result is still checked)")
    args = ap.parse_args(argv)
    try:
        return _run(args, getattr(args, "source", None))
    except UsageError as exc:
        print(exc, file=sys.stderr)
        return USAGE
    except SourceMismatch as exc:
        print(exc, file=sys.stderr)
        return DATA_DIFFERS
    except Exception:  # noqa: BLE001 - a crash must not read as "differs"
        traceback.print_exc()
        print("The rebuild failed.", file=sys.stderr)
        return FAILED


def _run(args: argparse.Namespace, source: str | None) -> int:
    check = not args.no_check
    if not check:
        print("Not checking the data against what the app read (--no-check).", file=sys.stderr)
    prepared = prepare(source, check)
    data = prepared.data
    counts = data[SPLIT_COL].value_counts()
    print(f"Rebuilt export #{SCRIPT['export_id']}: {len(data):,} rows (train {counts.get('train', 0):,} · valid "
          f"{counts.get('valid', 0):,} · test {counts.get('test', 0):,}), {len(prepared.feature_names):,} features, "
          f"target(s) {', '.join(targets())}.")
    status = MATCHES
    if EXPECTED_OUTPUT:
        diffs, rounding = output_problems(data)
        if diffs:
            status = DIFFERS
            print("It differs from the app's export:\n  - " + "\n  - ".join(diffs[:20]), file=sys.stderr)
        elif rounding:
            print(f"Matches the app's export; {len(rounding)} column(s) differ only by floating-point rounding "
                  f"({', '.join(rounding[:5])}{'…' if len(rounding) > 5 else ''}).")
        else:
            print("Identical to the app's export.")
    else:
        print("Not checked against the app's export: the app no longer has its files.", file=sys.stderr)
    if args.compare:
        found, rounding = compare_files(data, args.compare)
        if found:
            status = DIFFERS
            print(f"Differs from {args.compare}:\n  - " + "\n  - ".join(found[:20]), file=sys.stderr)
        elif rounding:
            print(f"Same values as {args.compare} but for floating-point rounding in {len(rounding)} column(s): "
                  f"{', '.join(rounding[:5])}{'…' if len(rounding) > 5 else ''}.")
        else:
            print("Same values as " + args.compare)
    if args.out:
        write_model_ready(data, args.out, {t for t, v in targets().items() if v["task"] == "regression"})
        print(f"Wrote {args.out}")
    if args.clean:
        load_clean(source, check).to_parquet(args.clean, index=False)
        print(f"Wrote {args.clean}")
    return status
