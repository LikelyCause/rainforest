"""Seeded row samples of typed columns for the Explore analyses."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from ..parquet_rows import read_rows
from ..profiling import coerce_numeric, parse_datetime
from ..semantics import NUMERIC
from ..transform import ROW_ID, SPLIT_COL, _natural_key

SEED = 0
MAX_LEVELS = 20
# A numeric target with this few distinct values reads better as classes (ratings, counts of children, ...).
NUMERIC_TARGET_MIN_LEVELS = 20
OTHER, MISSING = "Other", "(missing)"
SKIP_REASONS = {"text": "free text", "identifier": "identifier", "constant": "constant"}


class AnalysisError(Exception):
    """A failure worth showing as-is, without an exception type in front."""


class NotEnough(Exception):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass
class Column:
    name: str
    field_id: int | None
    kind: str  # numeric | categorical
    values: np.ndarray  # float64 with NaN (numeric) or int32 codes into `levels` (categorical)
    levels: list[str] = field(default_factory=list)
    type: str = "numeric"  # the field's semantic type, shown next to its name


@dataclass
class Prepared:
    columns: list[Column]
    skipped: list[dict]
    target: str | None = None
    target_kind: str | None = None  # categorical | numeric
    y: np.ndarray | None = None  # int class codes or float values, aligned with column rows
    classes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        if self.y is not None:
            return int(self.y.size)
        return int(self.columns[0].values.size) if self.columns else 0


def numeric_column(name: str, field_id: int | None, values: np.ndarray, type_: str = "numeric") -> Column | None:
    x = np.asarray(values, dtype=np.float64)
    x = np.where(np.isfinite(x), x, np.nan)
    finite = x[~np.isnan(x)]
    if finite.size < 2 or finite.min() == finite.max():
        return None
    return Column(name, field_id, "numeric", x, type=type_)


def categorical_column(name: str, field_id: int | None, s: pl.Series, type_: str = "categorical") -> Column | None:
    as_str = s.cast(pl.String)
    counts = as_str.drop_nulls().value_counts(sort=True)
    top = counts[as_str.name].to_list()[:MAX_LEVELS]
    levels = list(top)
    mapping = {v: i for i, v in enumerate(top)}
    other = len(levels)
    if counts.height > MAX_LEVELS:
        levels.append(OTHER)
    codes = as_str.replace_strict(mapping, default=other, return_dtype=pl.Int32)
    if as_str.null_count():
        levels.append(MISSING)
        codes = codes.fill_null(len(levels) - 1)
    out = codes.to_numpy().astype(np.int32)
    if np.unique(out).size < 2:
        return None
    return Column(name, field_id, "categorical", out, levels, type_)


def target_values(
    s: pl.Series, semantic_type: str | None, task: str | None = None
) -> tuple[str, np.ndarray, list[str]]:
    """(kind, y, classes): float values with NaN for a numeric target, else codes with -1 for missing. A target predicted
    as classes is compared as classes whatever its type."""
    if semantic_type in NUMERIC and task != "classes":
        num = coerce_numeric(s)
        if num.drop_nulls().n_unique() > NUMERIC_TARGET_MIN_LEVELS:
            return "numeric", num.to_numpy().astype(np.float64), []
    as_str = s.cast(pl.String)
    classes = sorted(as_str.drop_nulls().unique().to_list(), key=_natural_key)
    codes = as_str.replace_strict({c: i for i, c in enumerate(classes)}, default=None, return_dtype=pl.Int32)
    return "categorical", codes.fill_null(-1).to_numpy().astype(np.int32), classes


def sample_rows(
    n_rows: int, k: int, y: np.ndarray | None = None, kind: str | None = None, seed: int = SEED
) -> np.ndarray:
    """Sorted row indices: proportional per class when `y` holds class codes; rows without a target are left out."""
    rng = np.random.default_rng(seed)
    if y is None:
        valid = np.arange(n_rows)
    elif kind == "numeric":
        valid = np.flatnonzero(~np.isnan(y))
    else:
        valid = np.flatnonzero(y >= 0)
    if valid.size <= k:
        return valid
    if kind != "categorical":
        return np.sort(rng.choice(valid, size=k, replace=False))
    classes, counts = np.unique(y[valid], return_counts=True)
    share = counts * k / valid.size
    alloc = np.floor(share).astype(int)
    alloc[np.argsort(-(share - alloc), kind="stable")[: k - alloc.sum()]] += 1
    picked = [rng.choice(valid[y[valid] == c], size=a, replace=False) for c, a in zip(classes, alloc) if a]
    return np.sort(np.concatenate(picked))


def prepare_raw(
    path: Path, fields: list[dict], kept: bool, target: str | None, sample: int, seed: int = SEED
) -> Prepared:
    """`fields`: dicts with id, name, action, semantic_type, target_task and datetime_format, in column order."""
    n_rows = pl.scan_parquet(path).select(pl.len()).collect().item()
    target_field = next((f for f in fields if f["name"] == target), None) if target else None
    if target and target_field is None:
        raise AnalysisError(f"{target!r} is not a field of this dataset.")
    use, skipped = [], []
    for f in fields:
        if f["name"] == target or f["action"] == "target" or (kept and f["action"] != "keep"):
            continue
        if f["semantic_type"] in SKIP_REASONS:
            skipped.append({"name": f["name"], "reason": SKIP_REASONS[f["semantic_type"]]})
        else:
            use.append(f)

    kind, y_full, classes = None, None, []
    if target_field is not None:
        kind, y_full, classes = target_values(
            pl.read_parquet(path, columns=[target]).to_series(),
            target_field["semantic_type"],
            target_field.get("target_task"),
        )
    idx = sample_rows(n_rows, sample, y_full, kind, seed)
    df = read_rows(path, [f["name"] for f in use], idx)

    columns: list[Column] = []
    for f in use:
        s = df[f["name"]]
        if s.dtype.is_float():
            s = s.fill_nan(None)
        t = f["semantic_type"]
        if t in NUMERIC:
            col = numeric_column(f["name"], f["id"], coerce_numeric(s).to_numpy(), t)
        elif t == "datetime":
            parsed = parse_datetime(s, f.get("datetime_format"))
            num = parsed.dt.epoch("s") if parsed.dtype.is_temporal() else coerce_numeric(parsed)
            col = numeric_column(f["name"], f["id"], num.cast(pl.Float64).to_numpy(), t)
        else:
            col = categorical_column(f["name"], f["id"], s, t)
        if col is None:
            skipped.append({"name": f["name"], "reason": "constant in sample"})
        else:
            columns.append(col)
    y = None if y_full is None else y_full[idx]
    return Prepared(columns, skipped, target, kind, y, classes)


def prepare_export(export_dir: Path, manifest: dict, target: str | None, sample: int, seed: int = SEED) -> Prepared:
    """Train-split rows of an export's model_ready.parquet; every feature is numeric there."""
    path = export_dir / "model_ready.parquet"
    if not path.exists():
        raise AnalysisError("The export's files were removed.")
    targets = list(manifest.get("targets") or [])
    if target and target not in targets:
        raise AnalysisError(f"{target!r} is not a target of this export.")
    train = np.flatnonzero((pl.read_parquet(path, columns=[SPLIT_COL]).to_series() == "train").to_numpy())
    y_full, kind = None, None
    if target:
        kind = "numeric" if (manifest.get("target_kinds") or {}).get(target) == "regression" else "categorical"
        col = pl.read_parquet(path, columns=[target]).to_series()
        y_full = (
            col.cast(pl.Float64).fill_null(np.nan).to_numpy()
            if kind == "numeric"
            else col.fill_null(-1).to_numpy().astype(np.int32)
        )[train]
    idx = sample_rows(train.size, sample, y_full, kind, seed)
    names = [c for c in pl.read_parquet_schema(path) if c not in (ROW_ID, SPLIT_COL, *targets)]
    df = read_rows(path, names, train[idx])
    columns, skipped = [], []
    for name in names:
        col = numeric_column(name, None, df[name].cast(pl.Float64).fill_nan(None).to_numpy())
        if col is None:
            skipped.append({"name": name, "reason": "constant in sample"})
        else:
            columns.append(col)
    classes = [str(c) for c in (manifest.get("target_classes") or {}).get(target, [])] if target else []
    y = None if y_full is None else y_full[idx]
    return Prepared(columns, skipped, target, kind, y, classes)


def design_matrix(columns: list[Column]) -> tuple[np.ndarray, list[str], list[int]]:
    """Median-filled standardised numerics plus one-hot levels; returns (X, feature names, owning column index)."""
    blocks, names, owners = [], [], []
    for k, c in enumerate(columns):
        if c.kind == "numeric":
            x = np.where(np.isnan(c.values), np.nanmedian(c.values), c.values)
            sd = x.std()
            blocks.append(((x - x.mean()) / (sd if sd > 0 else 1.0))[:, None])
            names.append(c.name)
            owners.append(k)
        else:
            onehot = (c.values[:, None] == np.arange(len(c.levels))[None, :]).astype(np.float64)
            present = onehot.any(axis=0)
            blocks.append(onehot[:, present])
            names += [f"{c.name}={lv}" for lv, p in zip(c.levels, present) if p]
            owners += [k] * int(present.sum())
    if not blocks:
        return np.zeros((0, 0)), [], []
    return np.hstack(blocks), names, owners


def code_matrix(columns: list[Column]) -> tuple[np.ndarray, np.ndarray]:
    """Median-filled numerics and integer category codes side by side, plus the discrete-feature mask."""
    cols = []
    for c in columns:
        if c.kind == "numeric":
            cols.append(np.where(np.isnan(c.values), np.nanmedian(c.values), c.values))
        else:
            cols.append(c.values.astype(np.float64))
    X = np.column_stack(cols) if cols else np.zeros((0, 0))
    return X, np.array([c.kind == "categorical" for c in columns], dtype=bool)


def display_values(c: Column) -> list:
    if c.kind == "categorical":
        return [c.levels[v] for v in c.values]
    if c.type == "datetime":
        return [
            None if np.isnan(v) else datetime.fromtimestamp(v, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            for v in c.values
        ]
    return [None if np.isnan(v) else round(float(v), 6) for v in c.values]
