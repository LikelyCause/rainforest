"""The one place raw records become predictions: a deployment's export converts each source field to the type it had
in the dataset, computes the derived fields from their frozen specs, and runs its own preprocessing
(`to_pandas_frame` → `transform_into`); then every deployed run scores the features.

A `Scorer` holds nothing that changes after loading, so request threads share one from the cache."""

from __future__ import annotations

import ast
import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from . import derived
from .config import get_settings
from .ingest import NULL_TOKENS
from .modeling.mlflow_log import MODEL_FILE
from .modeling.train import scores_of
from .parquet_rows import read_rows
from .profiling import coerce_numeric, parse_datetime
from .semantics import NUMERIC
from .transform import ROW_ID, SPLIT_COL, FieldSpec, coercion_exprs, to_pandas_frame, transform_into

REEXPORT = "Re-export to deploy"
TOLERANCE = 1e-9
CHECK_ROWS = 50_000
TRUE_TEXT = {"true": True, "false": False, "1": True, "0": False}
TRAINING_SAMPLE = 5_000


class ModelChanged(ValueError):
    """A deployed model's file isn't the one recorded when it was deployed."""


class Unreadable(OSError):
    """A joblib file that is missing or doesn't unpickle: truncated, overwritten, or not a pickle at all."""


def _unpickle(path: Path) -> Any:
    try:
        return joblib.load(path)
    except Exception as exc:  # noqa: BLE001  (bytes that aren't a pickle can raise nearly anything)
        raise Unreadable(f"{path.name} can't be read ({type(exc).__name__})") from exc


@dataclass(frozen=True)
class Model:
    run_id: int
    label: str
    kind: str  # binary (one-vs-rest), multiclass, regression
    labels: tuple[str, ...]  # what `scores` is keyed by: the positive class, every class, or the target
    threshold: float | None
    model_dir: str
    estimator: Any
    report: dict


@dataclass(frozen=True)
class Input:
    name: str
    dtype: Any
    datetime_format: str | None
    decimal_comma: bool | None  # set for a field the features read as a number: text that isn't one fails
    integer_codes: bool = False

    @property
    def integer_number(self) -> bool:
        """A number the dataset stored as integers: read as a float, so 7.5 stays 7.5."""
        return self.dtype.is_integer() and not self.integer_codes and not self.datetime_format


@dataclass
class ScoreResult:
    scores: dict[str, np.ndarray]
    top: list[tuple[str, float]] | None
    matches: list[list[str]] | None
    none_matched: list[bool] | None
    warnings: dict[str, int]


@dataclass(frozen=True)
class Scorer:
    export_id: int
    inputs: tuple[Input, ...]
    derived: tuple[dict, ...]
    specs: tuple[FieldSpec, ...]
    preprocessor: Any
    n_features: int
    models: tuple[Model, ...]

    @property
    def required(self) -> list[str]:
        return [i.name for i in self.inputs]

    def frame(self, df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
        """The source fields converted and the derived fields computed, and how many values of each field didn't
        convert."""
        if missing := [n for n in self.required if n not in df.columns]:
            raise ValueError(f"Missing field{'s' if len(missing) > 1 else ''}: {', '.join(map(repr, missing))}")
        typed, seen, warnings = [], [], {}
        for i in self.inputs:
            s, failed = convert(df[i.name], i)
            typed.append(s)
            seen.append(integer_text(df[i.name], s, bool(i.decimal_comma)) if i.integer_number else s)
            if failed:
                warnings[i.name] = failed
        frame, view = pl.DataFrame(typed), pl.DataFrame(seen)
        for d in self.derived:
            col = derived.compute(view, d["kind"], d["spec"]).alias(d["name"])
            view, frame = view.with_columns(col), frame.with_columns(col)
        return frame, warnings

    def features(self, df: pl.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
        """The export's features for the raw records in `df`, and the conversion failures per field."""
        frame, warnings = self.frame(df)
        specs = list(self.specs)
        # Fresh coercers per call: they count values as they go, and threads share the scorer.
        X = transform_into(self.preprocessor, to_pandas_frame(frame, specs, coercion_exprs(specs)), self.n_features)
        return X, warnings

    def outputs(self, df: pl.DataFrame) -> tuple[list[np.ndarray], dict[str, int]]:
        X, warnings = self.features(df)
        return [scores_of(m.estimator, m.report, X) for m in self.models], warnings

    def score(self, df: pl.DataFrame) -> ScoreResult:
        outputs, warnings = self.outputs(df)
        return summarise(self.models, outputs, warnings)


def summarise(models, outputs: list[np.ndarray], warnings: dict[str, int]) -> ScoreResult:
    """Scores keyed by class (or target), and per record: the best class among the class outputs, and the one-vs-rest
    classes at or above their own threshold, best first."""
    scores: dict[str, np.ndarray] = {}
    for m, out in zip(models, outputs):
        if m.kind == "multiclass":
            scores.update({label: out[:, k] for k, label in enumerate(m.labels)})
        else:
            scores[m.labels[0]] = out
    keys = [label for m in models if m.kind != "regression" for label in m.labels]
    top = None
    if keys:
        mat = np.column_stack([scores[k] for k in keys])
        best = mat.argmax(axis=1)
        top = [(keys[b], float(mat[i, b])) for i, b in enumerate(best)]
    binary = [m for m in models if m.kind == "binary"]
    matches = none_matched = None
    if binary:
        mat = np.column_stack([scores[m.labels[0]] for m in binary])
        hit = mat >= np.array([m.threshold for m in binary])
        order = np.argsort(-mat, axis=1, kind="stable")
        matches = [[binary[j].labels[0] for j in row if h[j]] for row, h in zip(order, hit)]
        none_matched = (~hit.any(axis=1)).tolist()
    return ScoreResult(scores, top, matches, none_matched, warnings)


# =========================================================================== conversion
def parse_dtype(text: str):
    """The polars type `str(dtype)` printed, e.g. "Int64" or "Datetime(time_unit='us', time_zone=None)"."""

    def build(n: ast.AST):
        if isinstance(n, ast.Name):
            return dtype_class(n.id)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            kw = {k.arg: ast.literal_eval(k.value) for k in n.keywords}
            return dtype_class(n.func.id)(*map(build, n.args), **kw)
        raise ValueError

    def dtype_class(name: str):
        cls = getattr(pl, name, None)
        if not (isinstance(cls, type) and issubclass(cls, pl.DataType)):
            raise ValueError
        return cls

    try:
        return build(ast.parse(text, mode="eval").body)
    except (SyntaxError, ValueError, TypeError) as exc:
        raise ValueError(f"Unknown field type {text!r}; {REEXPORT}") from exc


def missing_as_null(s: pl.Series) -> pl.Series:
    """`s` with what the upload reader reads as missing (its null tokens, the empty string among them) and NaN as
    null. Nothing else is missing: a blank category is a value, as it was in the dataset."""
    if s.dtype.is_nested() or s.dtype == pl.Object:
        raise ValueError(f"Field {s.name!r} takes a single value, not a list or an object")
    if s.dtype.is_float():
        return s.fill_nan(None)
    if s.dtype == pl.String:
        return pl.select(pl.when(~s.is_in(NULL_TOKENS)).then(s)).to_series().alias(s.name)
    return s


def whole_numbers(given: pl.Series, decimal_comma: bool = False) -> pl.Series:
    """Int64 of the values that are whole numbers. Integers and integer text are read exactly, past 2^53 too; only
    values like "7.0" (or "7,0" with `decimal_comma`) go through a float."""
    if given.dtype.is_integer() or given.dtype == pl.Boolean:
        return given.cast(pl.Int64, strict=False)
    num = coerce_numeric(given, decimal_comma)
    whole = pl.select(pl.when(num == num.round(0)).then(num)).to_series().cast(pl.Int64, strict=False)
    if given.dtype == pl.String:
        return given.str.strip_chars().cast(pl.Int64, strict=False).fill_null(whole)
    return whole


def integer_text(s: pl.Series, out: pl.Series, decimal_comma: bool = False) -> pl.Series:
    """An integer-typed number (`out`, its float) as derived fields read it from the dataset's integer column: a whole
    number as its integer text, so a groups or count value "30" matches 30.0 the way it matched 30."""
    whole = whole_numbers(missing_as_null(s), decimal_comma)
    text = pl.when(whole.is_not_null() & out.is_not_null()).then(whole.cast(pl.String)).otherwise(out.cast(pl.String))
    return pl.select(text).to_series().alias(s.name)


# RFC 3339's profile: T and Z in either case, and an offset only after a time (a date alone has no time zone).
ISO_8601 = r"^(\d{4}-\d{2}-\d{2})(?:[Tt ](\d{2}:\d{2})(:\d{2}(?:\.\d{1,9})?)?([Zz]|[+-]\d{2}:?\d{2})?)?$"


def iso_datetime(txt: pl.Series) -> pl.Series:
    """ISO 8601 text as naive UTC datetimes: a date alone is midnight, a time without an offset is UTC. Anything else
    is null; guessing a format from the other values sent would read 03/04 differently depending on them."""
    parts = pl.DataFrame({"t": txt.cast(pl.String).str.strip_chars()}).select(
        pl.col("t").str.extract_groups(ISO_8601).alias("p")
    )
    p = parts.unnest("p")
    off = pl.col("4")
    sign = pl.when(off.str.starts_with("-")).then(-1).otherwise(1)
    minutes = (
        pl.when(off.is_null() | (off.str.to_uppercase() == "Z"))
        .then(0)
        .otherwise(
            sign
            * (off.str.slice(1, 2).cast(pl.Int64, strict=False) * 60 + off.str.slice(-2).cast(pl.Int64, strict=False))
        )
    )
    stamp = pl.concat_str([pl.col("1"), pl.lit(" "), pl.col("2").fill_null("00:00"), pl.col("3").fill_null(":00")])
    naive = stamp.str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S%.f", strict=False)
    return p.select(naive - pl.duration(minutes=minutes)).to_series().alias(txt.name)


def as_temporal(s: pl.Series, dtype) -> pl.Series:
    """Dates or datetimes as `dtype`; an aware datetime goes through UTC, and a naive one is taken as UTC."""
    if isinstance(s.dtype, pl.Datetime) and s.dtype.time_zone is not None:
        s = s.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
        aware = s.cast(pl.Datetime(dtype.time_unit)).dt.replace_time_zone("UTC")
        return aware.dt.convert_time_zone(dtype.time_zone)
    return s.cast(dtype, strict=False)


def convert(s: pl.Series, to: Input) -> tuple[pl.Series, int]:
    """`s` as the dataset stored the field, and how many given values didn't convert (they become missing)."""
    given = missing_as_null(s)
    dtype = to.dtype
    if dtype.is_integer() and not to.integer_number:
        # "7.0" and "07" are code 7; 7.5 isn't any code, and truncating it would invent one.
        out = whole_numbers(given).cast(dtype, strict=False)
    elif dtype.is_numeric():
        out = coerce_numeric(given, bool(to.decimal_comma), keep_inf=True)
        if not dtype.is_integer():
            out = out.cast(dtype, strict=False)
    elif dtype == pl.Boolean:
        if given.dtype == pl.Boolean:
            out = given
        elif given.dtype.is_numeric():
            out = pl.select(pl.when(given == 1).then(True).when(given == 0).then(False)).to_series()
        else:
            txt = given.cast(pl.String).str.strip_chars().str.to_lowercase()
            out = txt.replace_strict(TRUE_TEXT, default=None, return_dtype=pl.Boolean)
    elif dtype.is_temporal():
        if given.dtype.is_temporal():
            out = as_temporal(given, dtype)
        elif given.dtype in (pl.String, pl.Null):
            out = as_temporal(iso_datetime(given), dtype)
        else:
            out = pl.Series(s.name, [None] * s.len(), dtype=dtype)
    elif dtype == pl.String:
        if given.dtype.is_temporal() and to.datetime_format:
            given = given.dt.strftime(to.datetime_format)
        out = given.cast(pl.String)
        ok = None
        if to.datetime_format:
            ok = parse_datetime(out, to.datetime_format).is_not_null()
        elif to.decimal_comma is not None:
            ok = coerce_numeric(out, to.decimal_comma, keep_inf=True).is_not_null()
        if ok is not None:
            out = pl.select(pl.when(ok).then(out)).to_series()
    else:
        out = given.cast(dtype, strict=False)
    return out.alias(s.name), int(given.is_not_null().sum() - out.is_not_null().sum())


# =========================================================================== loading
_cache: OrderedDict[int, Scorer] = OrderedDict()
_lock = threading.Lock()


def load(conn, deployment_or_spec: dict) -> Scorer:
    """The scorer of a deployment ({id, export_id, models}), from the cache of the last `DEPLOYMENT_CACHE` used; without
    an id (a deployment being created) it is built and not kept."""
    key = deployment_or_spec.get("id")
    if key is None:
        return _build(conn, deployment_or_spec["export_id"], deployment_or_spec["models"])
    with _lock:
        if (hit := _cache.get(key)) is not None:
            _cache.move_to_end(key)
            return hit
    # Built outside the lock: loading models takes a while, and two threads building one is only wasted work.
    scorer = _build(conn, deployment_or_spec["export_id"], deployment_or_spec["models"])
    with _lock:
        _cache[key] = scorer
        _cache.move_to_end(key)
        while len(_cache) > max(get_settings().deployment_cache, 1):
            _cache.popitem(last=False)
    return scorer


def forget(deployment_id: int) -> None:
    with _lock:
        _cache.pop(deployment_id, None)


def clear() -> None:
    with _lock:
        _cache.clear()


def inputs(conn, export_id: int) -> tuple[Input, ...]:
    """The source fields a record sent to a deployment of this export must carry, with the types they convert to."""
    return _frozen(conn, export_id)[3]


def _frozen(conn, export_id: int) -> tuple[Path, dict, tuple[FieldSpec, ...], tuple[Input, ...], list[dict]]:
    """The export's folder, manifest and field specs, its inputs, and the derived fields its features read."""
    exp = conn.execute("SELECT config, output_dir FROM meta.exports WHERE id = %s", (export_id,)).fetchone()
    if exp is None:
        raise ValueError("Export not found")
    cfg = exp["config"] or {}
    if "derived" not in cfg or "input_schema" not in cfg or not exp["output_dir"]:
        raise ValueError(REEXPORT)
    out_dir = Path(exp["output_dir"])
    manifest = json.loads((out_dir / "manifest.json").read_text())
    specs = tuple(FieldSpec(**f) for f in manifest["fields"])
    schema = cfg["input_schema"]
    frozen = cfg["derived"]
    # Only what the features read: a derived field a row rule or a target reads plays no part in scoring.
    needed, sources = derived.closure(frozen, [*schema, *(d["name"] for d in frozen)], [f.name for f in specs])
    for d in needed:
        if "unavailable" in d:
            raise ValueError(f"The derived field {d['name']!r} can't be computed from new records: {d['unavailable']}")
    if any(n not in schema for n in sources):
        raise ValueError(REEXPORT)
    by_name = {f.name: f for f in specs}

    def number_format(name: str) -> bool | None:
        f = by_name.get(name)
        return f.decimal_comma if f is not None and f.semantic_type in NUMERIC else None

    inputs = tuple(
        Input(
            n,
            parse_dtype(schema[n]["dtype"]),
            schema[n].get("datetime_format"),
            number_format(n),
            bool(schema[n].get("integer_codes")),
        )
        for n in sources
    )
    for i in inputs:
        # Nothing reads these from text yet, so a record's value would always be missing; refuse rather than score it.
        if what := {pl.Time: "a time of day", pl.Duration: "a duration"}.get(i.dtype.base_type()):
            raise ValueError(
                f"The field {i.name!r} is stored as {what}, which a deployment can't read from records; leave it out "
                "of the export's features to deploy"
            )
    return out_dir, manifest, specs, inputs, needed


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _build(conn, export_id: int, models: list[dict]) -> Scorer:
    out_dir, manifest, specs, inputs, needed = _frozen(conn, export_id)
    loaded = tuple(_model(conn, export_id, m) for m in models)
    keys = [label for m in loaded for label in m.labels]
    if len(set(keys)) < len(keys):
        raise ValueError("Two models give a score for the same class")
    return Scorer(
        export_id=export_id,
        inputs=inputs,
        derived=tuple(needed),
        specs=specs,
        preprocessor=_unpickle(out_dir / "preprocessor.joblib"),
        n_features=int(manifest["n_features"]),
        models=loaded,
    )


def _model(conn, export_id: int, m: dict) -> Model:
    run = conn.execute(
        "SELECT id, export_id, status, model_dir FROM meta.model_runs WHERE id = %s", (m["run_id"],)
    ).fetchone()
    if run is None or run["export_id"] != export_id:
        raise ValueError(f"Run {m['run_id']} is not a run of export {export_id}")
    if run["status"] != "done" or not run["model_dir"]:
        raise ValueError(f"Run {m['run_id']} hasn't finished")
    model_dir = Path(run["model_dir"])
    report = json.loads((model_dir / "report.json").read_text())
    order = report.get("order") or []
    if report.get("chain_kind") == "one_vs_rest":
        kind, labels = "binary", (report["positive_class"],)
        threshold = m.get("threshold")
        if threshold is None:
            threshold = (report.get("f1_threshold") or {}).get("threshold")
        if threshold is None:
            raise ValueError(f"Run {m['run_id']} has no F1 threshold to start from; give it a threshold")
    elif len(order) != 1:
        raise ValueError(f"Run {m['run_id']} chains several targets; deploy a run with one target")
    elif report["target_kinds"][order[0]] == "regression":
        kind, labels, threshold = "regression", (order[0],), None
    else:
        kind, labels, threshold = "multiclass", tuple(report["target_classes"][order[0]]), None
    label = m.get("label") or report.get("model_label") or f"Run {run['id']}"
    # Checked before unpickling: a file swapped in after deployment is neither trusted nor loaded.
    if m.get("sha256") and file_sha256(model_dir / MODEL_FILE) != m["sha256"]:
        raise ModelChanged(
            f"The model file of {label!r} (run {run['id']}) changed since it was deployed; deploy the run again"
        )
    return Model(
        run_id=int(run["id"]),
        label=label,
        kind=kind,
        labels=labels,
        threshold=None if threshold is None else float(threshold),
        model_dir=str(model_dir),
        estimator=_unpickle(model_dir / MODEL_FILE),
        report=report,
    )


# =========================================================================== equality check
def equality_check(conn, export_id: int, models: list[dict]) -> str | None:
    """None when scoring the raw values of the export's test rows reproduces each model's scores on the export's own
    `model_ready.parquet` within 1e-9, and its features exactly (as float32); else what differs first, scores before
    features. Without test rows it compares the validation rows,
    without those a sample of training rows, and with no rows at all it doesn't pass."""
    scorer = load(conn, {"export_id": export_id, "models": models})
    exp = conn.execute(
        """SELECT e.output_dir, e.created_at, d.parquet_path, d.profiled_at FROM meta.exports e
             JOIN meta.datasets d ON d.id = e.dataset_id WHERE e.id = %s""",
        (export_id,),
    ).fetchone()
    data = Path(exp["output_dir"]) / "model_ready.parquet"
    raw = Path(exp["parquet_path"])
    refreshed = exp["profiled_at"] is not None and exp["profiled_at"] > exp["created_at"]
    hint = " The dataset was refreshed after this export, so its rows may have changed; re-export to deploy."
    hint = hint if refreshed else ""
    if gone := [n for n in scorer.required if n not in pl.read_parquet_schema(raw)]:
        return f"The dataset no longer has {', '.join(map(repr, gone))}, which the export's features read.{hint}"
    positions = _rows_to_compare(pl.read_parquet(data, columns=[SPLIT_COL])[SPLIT_COL])
    if not positions.size:
        return "The export has no rows to compare scoring against."
    feats = json.loads((Path(exp["output_dir"]) / "manifest.json").read_text())["feature_names"]
    n_raw = pl.scan_parquet(raw).select(pl.len()).collect().item()
    for k in range(0, positions.size, CHECK_ROWS):
        part = read_rows(data, [ROW_ID, *feats], positions[k : k + CHECK_ROWS])
        ids = part[ROW_ID].to_numpy()
        if ids.max() >= n_raw:
            return f"Row {int(ids.max())} of the export is past the end of the dataset.{hint}"
        X = part.select(feats).to_numpy()
        got_X, _ = scorer.features(read_rows(raw, scorer.required, ids))
        for m in scorer.models:
            got = scores_of(m.estimator, m.report, got_X)
            if (msg := _differs(m, got, scores_of(m.estimator, m.report, X), ids)) is not None:
                return msg + hint
        # Scores can agree while a feature doesn't (a tree that never splits on it); training saw the export's.
        if (msg := _feature_differs(got_X, X, feats, ids)) is not None:
            return msg + hint
    return None


def _feature_differs(got: np.ndarray, want: np.ndarray, feats: list[str], ids: np.ndarray) -> str | None:
    g, w = got.astype(np.float32), want.astype(np.float32)
    bad = ~((g == w) | (np.isnan(g) & np.isnan(w)))
    if not bad.any():
        return None
    i, j = np.argwhere(bad)[0]
    return (
        f"The export's row {int(ids[i])} has {feats[j]!r} = {g[i, j]:.9g} from its raw values, but {w[i, j]:.9g} in "
        "the export's features."
    )


def _rows_to_compare(split: pl.Series) -> np.ndarray:
    """Positions of the test rows, else the validation rows, else a seeded sample of the training rows, sorted."""
    for name in ("test", "valid"):
        if (rows := np.flatnonzero((split == name).to_numpy())).size:
            return rows
    train = np.flatnonzero((split == "train").to_numpy())
    if train.size <= TRAINING_SAMPLE:
        return train
    return np.sort(np.random.default_rng(0).choice(train, TRAINING_SAMPLE, replace=False))


def _differs(m: Model, got: np.ndarray, want: np.ndarray, ids: np.ndarray) -> str | None:
    g, w = got.reshape(len(ids), -1), want.reshape(len(ids), -1)
    bad = ~((np.abs(g - w) <= TOLERANCE) | (np.isnan(g) & np.isnan(w)))
    if not bad.any():
        return None
    i, j = np.argwhere(bad)[0]
    what = f" for {m.labels[j]!r}" if m.kind == "multiclass" else ""
    return (
        f"{m.label!r} scores the export's row {int(ids[i])}{what} {g[i, j]:.10g} from its raw values, but "
        f"{w[i, j]:.10g} on the export's features."
    )
