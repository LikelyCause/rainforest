"""The one place raw records become predictions: a deployment's export converts each source field to the type it had
in the dataset, computes the derived fields from their frozen specs, and runs its own preprocessing
(`to_pandas_frame` → `transform_into`); then every deployed run scores the features.

A `Scorer` holds nothing that changes after loading, so request threads share one from the cache."""

from __future__ import annotations

import ast
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
from .modeling.mlflow_log import MODEL_FILE
from .modeling.train import predict_scores, scores_of
from .parquet_rows import read_rows
from .profiling import coerce_numeric, parse_datetime
from .semantics import NUMERIC
from .transform import ROW_ID, SPLIT_COL, FieldSpec, coercion_exprs, to_pandas_frame, transform_into

REEXPORT = "Re-export to deploy"
TOLERANCE = 1e-9
CHECK_ROWS = 50_000
TRUE_TEXT = {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False}


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

    def features(self, df: pl.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
        """The export's features for the raw records in `df`, and how many values of each field didn't convert."""
        if missing := [n for n in self.required if n not in df.columns]:
            raise ValueError(f"Missing field{'s' if len(missing) > 1 else ''}: {', '.join(map(repr, missing))}")
        cols, warnings = [], {}
        for i in self.inputs:
            s, failed = convert(df[i.name], i)
            cols.append(s)
            if failed:
                warnings[i.name] = failed
        frame = pl.DataFrame(cols)
        for d in self.derived:
            frame = frame.with_columns(derived.compute(frame, d["kind"], d["spec"]).alias(d["name"]))
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


def convert(s: pl.Series, to: Input) -> tuple[pl.Series, int]:
    """`s` as the dataset stored the field, and how many given values didn't convert (they become missing). NaN and
    blank text are missing to begin with."""
    given = s.fill_nan(None) if s.dtype.is_float() else s
    if given.dtype == pl.String:
        given = pl.select(pl.when(given.str.strip_chars() != "").then(given)).to_series()
    dtype = to.dtype
    if dtype.is_numeric():
        num = coerce_numeric(given, bool(to.decimal_comma), keep_inf=True)
        if dtype.is_integer():
            # "7.0" and "07" are 7; 7.5 isn't any integer, and truncating it would invent one.
            num = pl.select(pl.when(num == num.round(0)).then(num)).to_series()
        out = num.cast(dtype, strict=False)
    elif dtype == pl.Boolean:
        if given.dtype == pl.Boolean:
            out = given
        else:
            txt = given.cast(pl.String).str.strip_chars().str.to_lowercase()
            out = txt.replace_strict(TRUE_TEXT, default=None, return_dtype=pl.Boolean)
    elif dtype.is_temporal():
        out = parse_datetime(given, to.datetime_format).cast(dtype, strict=False)
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


def _build(conn, export_id: int, models: list[dict]) -> Scorer:
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
        Input(n, parse_dtype(schema[n]["dtype"]), schema[n].get("datetime_format"), number_format(n)) for n in sources
    )
    loaded = tuple(_model(conn, export_id, m) for m in models)
    keys = [label for m in loaded for label in m.labels]
    if len(set(keys)) < len(keys):
        raise ValueError("Two models give a score for the same class")
    return Scorer(
        export_id=export_id,
        inputs=inputs,
        derived=tuple(needed),
        specs=specs,
        preprocessor=joblib.load(out_dir / "preprocessor.joblib"),
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
            threshold = (report.get("f1_threshold") or {}).get("threshold", 0.5)
    elif len(order) != 1:
        raise ValueError(f"Run {m['run_id']} chains several targets; deploy a run with one target")
    elif report["target_kinds"][order[0]] == "regression":
        kind, labels, threshold = "regression", (order[0],), None
    else:
        kind, labels, threshold = "multiclass", tuple(report["target_classes"][order[0]]), None
    return Model(
        run_id=int(run["id"]),
        label=m.get("label") or report.get("model_label") or f"Run {run['id']}",
        kind=kind,
        labels=labels,
        threshold=None if threshold is None else float(threshold),
        model_dir=str(model_dir),
        estimator=joblib.load(model_dir / MODEL_FILE),
        report=report,
    )


# =========================================================================== equality check
def equality_check(conn, export_id: int, models: list[dict]) -> str | None:
    """None when scoring the raw values of the export's test rows (its validation rows when it has no test split)
    reproduces each model's scores on the export's own `model_ready.parquet` within 1e-9; else what differs first."""
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
    if gone := [n for n in scorer.required if n not in pl.read_parquet_schema(raw)]:
        return f"The dataset no longer has {', '.join(map(repr, gone))}, which the export's features read.{hint}"
    split = pl.read_parquet(data, columns=[SPLIT_COL])[SPLIT_COL]
    positions = np.flatnonzero((split == "test").to_numpy())
    if not positions.size:
        positions = np.flatnonzero((split == "valid").to_numpy())
    feats = json.loads((Path(exp["output_dir"]) / "manifest.json").read_text())["feature_names"]
    n_raw = _rows(raw)
    for k in range(0, positions.size, CHECK_ROWS):
        part = read_rows(data, [ROW_ID, *feats], positions[k : k + CHECK_ROWS])
        ids = part[ROW_ID].to_numpy()
        if ids.max() >= n_raw:
            return f"Row {int(ids.max())} of the export is past the end of the dataset.{hint}"
        X = part.select(feats).to_numpy()
        got, _ = scorer.outputs(read_rows(raw, scorer.required, ids))
        for m, g in zip(scorer.models, got):
            if (msg := _differs(m, g, predict_scores(Path(m.model_dir), X), ids)) is not None:
                return msg + (hint if refreshed else "")
    return None


def _rows(path: Path) -> int:
    return pl.scan_parquet(path).select(pl.len()).collect().item()


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
