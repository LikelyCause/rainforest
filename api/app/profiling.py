"""Column profiling and semantic-type inference.

Reads the dataset's Parquet copy one column at a time (memory stays ~1 column wide).
Counts (nulls, distinct values, min / max / mean) are exact over all rows; quantiles,
histograms, text statistics and correlation inputs use a seeded random row sample.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl
from scipy import stats as sstats

from .semantics import NUMERIC

ID_NAME_RE = re.compile(r"(^|[_\s.-])(id|uuid|guid|key|pk|identifier|code)$", re.IGNORECASE)
TOKEN_RE = r"[a-z][a-z']{2,}"
DATETIME_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S%.f",
    "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%dT%H:%M:%S%z",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%Y/%m/%d",
    "%d-%b-%Y",
    "%Y%m%d",
]
STOPWORDS = set(
    """the and for are but not you all any can had her was one our out has have this that with from
    they will would there their what about which when your into than then them these some been were
    its also just very more most such only over other said each where while because could should""".split()
)


@dataclass
class FieldProfile:
    name: str
    ordinal: int
    physical_dtype: str
    inferred_type: str
    confidence: float
    reasons: list[str]
    is_target: bool
    summary: dict[str, Any]  # promoted to typed DB columns
    stats: dict[str, Any]  # everything else (JSONB)
    charts: dict[str, Any]  # chart-ready payloads (JSONB)
    # Correlation inputs on the shared row sample (not persisted)
    corr_numeric: np.ndarray | None = field(default=None, repr=False)
    corr_codes: np.ndarray | None = field(default=None, repr=False)


# --------------------------------------------------------------------------- helpers
def _f(x: Any) -> float | None:
    """JSON-safe float."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _sample_series(s: pl.Series, idx: np.ndarray | None) -> pl.Series:
    return s if idx is None else s.gather(idx)


def sample_indices(n_rows: int, k: int, seed: int = 17) -> np.ndarray | None:
    if n_rows <= k:
        return None
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=k, replace=False))


def detect_datetime_format(s: pl.Series) -> tuple[str | None, float]:
    """Best strptime format for a string series and the share of values it parses."""
    probe = s.drop_nulls().head(5000).str.strip_chars()
    if probe.len() == 0:
        return None, 0.0
    best, best_rate = None, 0.0
    for fmt in DATETIME_FORMATS:
        try:
            parsed = probe.str.strptime(pl.Datetime("us"), fmt, strict=False)
        except Exception:  # noqa: BLE001 - polars raises for some format/type combos
            continue
        rate = 1 - parsed.null_count() / probe.len()
        if rate > best_rate:
            best, best_rate = fmt, rate
        if rate >= 0.999:
            break
    return best, best_rate


def parse_datetime(s: pl.Series, fmt: str | None) -> pl.Series:
    if s.dtype == pl.String:
        out = s.str.strip_chars().str.strptime(pl.Datetime("us"), fmt, strict=False)
    elif s.dtype == pl.Date:
        out = s.cast(pl.Datetime("us"))
    else:
        out = s
    if isinstance(out.dtype, pl.Datetime) and out.dtype.time_zone is not None:
        out = out.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    return out


def coerce_numeric(s: pl.Series) -> pl.Series:
    if s.dtype == pl.String:
        out = s.str.strip_chars().str.replace_all(",", "").cast(pl.Float64, strict=False)
    elif s.dtype == pl.Boolean:
        out = s.cast(pl.Float64)
    elif s.dtype.is_temporal():
        out = s.cast(pl.Datetime("us")).dt.epoch("s").cast(pl.Float64)
    else:
        out = s.cast(pl.Float64)
    return out.fill_nan(None)


# --------------------------------------------------------------------------- inference
def infer_type(
    s: pl.Series, name: str, n_unique: int, n_nonnull: int
) -> tuple[str, float, list[str], dict[str, Any]]:
    """Return (semantic_type, confidence, reasons, hints)."""
    hints: dict[str, Any] = {}
    dt = s.dtype
    if n_nonnull == 0:
        return "constant", 1.0, ["column is entirely missing"], hints
    if n_unique == 1:
        return "constant", 0.99, ["a single distinct value"], hints
    if dt == pl.Boolean:
        return "binary", 0.99, ["boolean dtype"], hints
    if n_unique == 2:
        return "binary", 0.95, ["exactly two distinct values"], hints
    if dt.is_temporal():
        return "datetime", 0.99, [f"{dt} dtype"], hints
    if dt.is_numeric():
        t, c, r = _infer_numeric(s.drop_nulls(), name, dt, n_unique, n_nonnull)
        return t, c, r, hints
    if dt == pl.String:
        return _infer_string(s, name, n_unique, n_nonnull)
    return "categorical", 0.4, [f"unrecognised dtype {dt}; treating as categorical"], hints


def _infer_numeric(nn: pl.Series, name: str, dt: pl.DataType, n_unique: int, n_nonnull: int):
    unique_ratio = n_unique / max(n_nonnull, 1)
    if dt.is_float():
        probe = nn.head(200_000)
        integral = bool((probe == probe.round(0)).all())
    else:
        integral = True
    if integral and n_nonnull >= 500:
        if unique_ratio >= 0.999 or (ID_NAME_RE.search(name) and unique_ratio > 0.9):
            why = "every value is unique" if unique_ratio >= 0.999 else "name looks like a key"
            monotonic = nn.head(100_000).is_sorted()
            return "identifier", 0.9 if monotonic else 0.75, [
                f"integer values; {why}" + ("; monotonically increasing" if monotonic else "")
            ]
    if integral:
        if n_unique <= 20:
            return "numeric_discrete", 0.7, [
                f"integer values with {n_unique} levels — could also be ordinal / categorical"
            ]
        if n_unique > 1000 and unique_ratio > 0.2:
            return "numeric_continuous", 0.75, [
                "integer-valued but high resolution — behaves like a continuous measure"
            ]
        return "numeric_discrete", 0.85, [f"integer values ({n_unique:,} distinct)"]
    if n_unique <= 10:
        return "numeric_discrete", 0.6, [f"only {n_unique} distinct non-integer values"]
    return "numeric_continuous", 0.92, ["floating point values"]


def _infer_string(s: pl.Series, name: str, n_unique: int, n_nonnull: int):
    hints: dict[str, Any] = {}
    nn = s.drop_nulls()
    probe = nn.head(20_000)
    unique_ratio = n_unique / max(n_nonnull, 1)

    as_num = probe.str.strip_chars().str.replace_all(",", "").cast(pl.Float64, strict=False)
    num_rate = 1 - as_num.null_count() / max(probe.len(), 1)
    if num_rate >= 0.98:
        t, c, r = _infer_numeric(
            coerce_numeric(nn).drop_nulls(), name, pl.Float64, n_unique, n_nonnull
        )
        hints["stored_as_text"] = True
        return t, round(c * 0.9, 2), [f"stored as text but {num_rate:.1%} of values parse as numbers", *r], hints

    fmt, dt_rate = detect_datetime_format(probe)
    if fmt and dt_rate >= 0.95:
        hints["datetime_format"] = fmt
        return "datetime", round(0.8 + 0.2 * dt_rate, 2), [
            f"{dt_rate:.1%} of values parse as dates with format {fmt}"
        ], hints

    lengths = probe.str.len_chars()
    words = probe.str.count_matches(r"\S+")
    avg_len = float(lengths.mean() or 0)
    avg_words = float(words.mean() or 0)
    if (avg_words >= 4 and unique_ratio > 0.2) or avg_len >= 50:
        conf = 0.95 if avg_words >= 8 else 0.8
        return "text", conf, [f"avg {avg_words:.1f} words / {avg_len:.0f} chars per value"], hints

    if n_nonnull >= 500 and avg_words < 2 and (
        unique_ratio >= 0.98 or (ID_NAME_RE.search(name) and unique_ratio > 0.5)
    ):
        why = f"{unique_ratio:.1%} of values are unique" if unique_ratio >= 0.98 else "name looks like a key"
        return "identifier", 0.85, [f"short tokens; {why}"], hints

    reasons = [f"{n_unique:,} distinct short string values"]
    conf = 0.9
    if n_unique > 100:
        reasons.append("high cardinality")
        conf = 0.7
    return "categorical", conf, reasons, hints


# --------------------------------------------------------------------------- statistics
def _histogram(x: np.ndarray, bins: int = 40) -> dict[str, Any]:
    if x.size == 0:
        return {"edges": [], "counts": []}
    lo, hi = np.percentile(x, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.min(x)), float(np.max(x))
    if lo == hi:
        hi = lo + 1.0
    counts, edges = np.histogram(x, bins=bins, range=(lo, hi))
    return {
        "edges": [float(e) for e in edges],
        "counts": [int(c) for c in counts],
        "below": int((x < lo).sum()),
        "above": int((x > hi).sum()),
    }


def _numeric_stats(full: pl.Series, sample: pl.Series, discrete: bool) -> tuple[dict, dict, dict]:
    nn = full.drop_nulls()
    x_full = nn.to_numpy()
    xs = sample.drop_nulls().to_numpy()
    if x_full.size == 0:
        return {}, {}, {}
    q = np.quantile(xs, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]) if xs.size else [None] * 7
    p01, p05, p25, p50, p75, p95, p99 = (float(v) if v is not None else None for v in q)
    iqr = (p75 - p25) if p75 is not None else 0.0
    outliers = float(((xs < p25 - 1.5 * iqr) | (xs > p75 + 1.5 * iqr)).mean()) if xs.size and iqr > 0 else 0.0
    skew = _f(sstats.skew(xs, bias=False)) if xs.size > 3 and np.std(xs) > 0 else 0.0
    kurt = _f(sstats.kurtosis(xs, bias=False)) if xs.size > 4 and np.std(xs) > 0 else 0.0
    summary = {
        "mean": _f(x_full.mean()),
        "std": _f(x_full.std(ddof=1)) if x_full.size > 1 else 0.0,
        "min_value": _f(x_full.min()),
        "max_value": _f(x_full.max()),
        "median": p50,
        "skewness": skew,
        "kurtosis": kurt,
        "zeros_pct": _f((x_full == 0).mean()),
        "outlier_pct": outliers,
    }
    stats = {
        "quantiles": {"p01": p01, "p05": p05, "p25": p25, "p50": p50, "p75": p75, "p95": p95, "p99": p99},
        "iqr": _f(iqr),
        "negative_pct": _f((x_full < 0).mean()),
        "cv": _f(summary["std"] / summary["mean"]) if summary["mean"] else None,
    }
    charts: dict[str, Any] = {}
    n_unique = int(nn.n_unique())
    if discrete and n_unique <= 60:
        vc = nn.value_counts(sort=False).sort(nn.name)
        charts["distribution"] = {
            "kind": "bars",
            "labels": [_fmt_num(v) for v in vc[nn.name].to_list()],
            "counts": vc["count"].to_list(),
        }
    else:
        charts["distribution"] = {"kind": "histogram", **_histogram(xs)}
    if (summary["min_value"] or 0) >= 0 and abs(skew or 0) > 1.5:
        charts["log_distribution"] = {"kind": "histogram", **_histogram(np.log1p(xs))}
    charts["box"] = {"p01": p01, "p25": p25, "p50": p50, "p75": p75, "p99": p99,
                     "min": summary["min_value"], "max": summary["max_value"]}
    return summary, stats, charts


def _fmt_num(v: Any) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _categorical_stats(full: pl.Series, top_k: int = 20) -> tuple[dict, dict, dict]:
    nn = full.drop_nulls().cast(pl.String)
    n = nn.len()
    if n == 0:
        return {}, {}, {}
    vc = nn.value_counts(sort=True)
    name = nn.name
    top = vc.head(top_k)
    counts = vc["count"].to_numpy()
    p = counts / counts.sum()
    entropy = float(-(p * np.log2(p)).sum())
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
    rare = int((p < 0.01).sum())
    lengths = nn.str.len_chars()
    summary = {
        "top_value": str(top[name][0]),
        "top_value_pct": float(top["count"][0] / n),
        "avg_length": _f(lengths.mean()),
    }
    stats = {
        "levels": len(counts),
        "entropy_bits": entropy,
        "normalized_entropy": entropy / max_entropy if max_entropy else 0.0,
        "rare_levels": rare,
        "rare_levels_share": float(p[p < 0.01].sum()),
        "top_values": [
            {"value": str(v), "count": int(c), "pct": float(c / n)}
            for v, c in zip(top[name].to_list(), top["count"].to_list())
        ],
        "other_count": int(n - top["count"].sum()),
    }
    charts = {
        "distribution": {
            "kind": "bars",
            "labels": [str(v) for v in top[name].to_list()],
            "counts": top["count"].to_list(),
            "other": stats["other_count"],
            "missing": int(full.null_count()),
        }
    }
    return summary, stats, charts


def _text_stats(sample: pl.Series) -> tuple[dict, dict, dict]:
    nn = sample.drop_nulls()
    if nn.len() == 0:
        return {}, {}, {}
    lengths = nn.str.len_chars().to_numpy()
    words = nn.str.count_matches(r"\S+").to_numpy()
    token_counter: Counter[str] = Counter()
    for toks in nn.head(50_000).str.to_lowercase().str.extract_all(TOKEN_RE).to_list():
        if toks:
            token_counter.update(t for t in toks if t not in STOPWORDS)
    top_tokens = token_counter.most_common(20)
    summary = {"avg_length": _f(lengths.mean())}
    stats = {
        "length": {"min": int(lengths.min()), "p50": _f(np.median(lengths)), "max": int(lengths.max())},
        "words": {"mean": _f(words.mean()), "p50": _f(np.median(words)), "max": int(words.max())},
        "empty_pct": _f((lengths == 0).mean()),
        "vocabulary_sampled": len(token_counter),
        "top_tokens": [{"token": t, "count": c} for t, c in top_tokens],
    }
    charts = {
        "distribution": {"kind": "histogram", "xlabel": "characters", **_histogram(lengths.astype(float), 30)},
        "tokens": {"kind": "bars", "labels": [t for t, _ in top_tokens], "counts": [c for _, c in top_tokens]},
    }
    return summary, stats, charts


def _datetime_stats(parsed: pl.Series, sample_parsed: pl.Series) -> tuple[dict, dict, dict]:
    nn = parsed.drop_nulls()
    if nn.len() == 0:
        return {}, {}, {}
    mn, mx = nn.min(), nn.max()
    snn = sample_parsed.drop_nulls()
    epoch = snn.dt.epoch("s").to_numpy().astype(float)
    weekday = np.bincount(snn.dt.weekday().to_numpy().astype(np.int64) - 1, minlength=7)[:7]
    head = nn.head(10_000)
    has_time = bool(((head.dt.hour() != 0) | (head.dt.minute() != 0) | (head.dt.second() != 0)).any())
    stats = {
        "min": mn.isoformat(),
        "max": mx.isoformat(),
        "span_days": _f((mx - mn).total_seconds() / 86400),
        "has_time_component": has_time,
    }
    charts = {
        "distribution": {"kind": "time_histogram", **_histogram(epoch, 40)},
        "weekday": {
            "kind": "bars",
            "labels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            "counts": [int(c) for c in weekday],
        },
    }
    summary = {"min_value": _f(epoch.min()) if epoch.size else None,
               "max_value": _f(epoch.max()) if epoch.size else None}
    return summary, stats, charts


def _codes(s: pl.Series, max_levels: int) -> np.ndarray:
    """Integer codes for association measures: top levels kept, rest -> 'other', null -> own code."""
    as_str = s.cast(pl.String)
    top = as_str.drop_nulls().value_counts(sort=True).head(max_levels)[as_str.name].to_list()
    mapping = {v: i for i, v in enumerate(top)}
    other, missing = len(top), len(top) + 1
    mapped = as_str.replace_strict(mapping, default=other, return_dtype=pl.Int32)
    return mapped.fill_null(missing).to_numpy().astype(np.int32)


# --------------------------------------------------------------------------- driver
def profile_column(
    s: pl.Series,
    ordinal: int,
    is_target: bool,
    sample_idx: np.ndarray | None,
    corr_idx: np.ndarray | None,
    corr_max_levels: int,
) -> FieldProfile:
    name = s.name
    physical = str(s.dtype)
    if s.dtype.is_float():
        s = s.fill_nan(None)
    n = s.len()
    nulls = s.null_count()
    n_nonnull = n - nulls
    n_unique = int(s.drop_nulls().n_unique()) if n_nonnull else 0

    t, conf, reasons, hints = infer_type(s, name, n_unique, n_nonnull)
    if is_target and t not in ("binary", "categorical"):
        reasons = [f"target column; values look {t}, treated as class labels", *reasons]
        t = "binary" if n_unique == 2 else "categorical"

    summary: dict[str, Any] = {
        "row_count": n,
        "null_count": nulls,
        "null_pct": nulls / n if n else 0.0,
        "n_unique": n_unique,
        "unique_pct": n_unique / n_nonnull if n_nonnull else 0.0,
    }
    stats: dict[str, Any] = {**hints}
    charts: dict[str, Any] = {}
    corr_numeric = corr_codes = None

    def _corr_sample(series: pl.Series) -> pl.Series:
        return series if corr_idx is None else series.gather(corr_idx)

    if t in NUMERIC:
        num = coerce_numeric(s)
        su, st, ch = _numeric_stats(num, _sample_series(num, sample_idx), t == "numeric_discrete")
        corr_numeric = _corr_sample(num).to_numpy().astype(np.float64)
    elif t in ("categorical", "binary"):
        su, st, ch = _categorical_stats(s)
        corr_codes = _codes(_corr_sample(s), corr_max_levels)
        if s.dtype.is_numeric() or s.dtype == pl.Boolean:
            stats["numeric_like"] = True
    elif t == "text":
        su, st, ch = _text_stats(_sample_series(s, sample_idx))
    elif t == "datetime":
        parsed = parse_datetime(s, hints.get("datetime_format"))
        su, st, ch = _datetime_stats(parsed, _sample_series(parsed, sample_idx))
        corr_numeric = (
            _corr_sample(parsed).dt.epoch("s").cast(pl.Float64).to_numpy().astype(np.float64)
        )
    elif t == "identifier":
        su, st, ch = {}, {}, {}
        lengths = s.drop_nulls().cast(pl.String).str.len_chars()
        su["avg_length"] = _f(lengths.mean())
    else:  # constant
        su, st, ch = {}, {}, {}
        first = s.drop_nulls().head(1).to_list()
        st["value"] = str(first[0]) if first else None

    summary.update(su)
    stats.update(st)
    charts.update(ch)
    stats["examples"] = [
        f"{v:.6g}" if isinstance(v, float) else str(v)[:200]
        for v in s.drop_nulls().unique(maintain_order=True).head(8).to_list()
    ]

    return FieldProfile(
        name=name,
        ordinal=ordinal,
        physical_dtype=physical,
        inferred_type=t,
        confidence=float(conf),
        reasons=reasons,
        is_target=is_target,
        summary=summary,
        stats=stats,
        charts=charts,
        corr_numeric=corr_numeric,
        corr_codes=corr_codes,
    )


def profile_parquet(
    parquet_path: Path,
    targets: list[str],
    sample_rows: int,
    corr_rows: int,
    corr_max_levels: int,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[int, list[FieldProfile]]:
    schema = pl.read_parquet_schema(parquet_path)
    n_rows = pl.scan_parquet(parquet_path).select(pl.len()).collect().item()
    sample_idx = sample_indices(n_rows, sample_rows, seed=17)
    corr_idx = sample_indices(n_rows, corr_rows, seed=29)
    target_set = set(targets)
    profiles: list[FieldProfile] = []
    names = list(schema.keys())
    for i, name in enumerate(names):
        if progress:
            progress(i, len(names), name)
        s = pl.read_parquet(parquet_path, columns=[name]).to_series()
        profiles.append(profile_column(s, i, name in target_set, sample_idx, corr_idx, corr_max_levels))
    return n_rows, profiles
