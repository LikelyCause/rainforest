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

from .semantics import NUMERIC, target_problem

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
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%Y/%m/%d",
    "%d-%b-%Y",
    "%Y%m%d",
]
DAY_FIRST = {"%m/%d/%Y": "%d/%m/%Y", "%m/%d/%Y %H:%M": "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M:%S": "%d/%m/%Y %H:%M:%S"}
MONTH_FIRST = {v: k for k, v in DAY_FIRST.items()}
YYYYMMDD = "%Y%m%d"
THOUSANDS_RE = r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$"
DECIMAL_COMMA_RE = r"^[+-]?\d+,\d+$"
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
    # Uncapped codes for the leak screen: targets, text, and categoricals with more than corr_max_levels levels.
    screen_codes: np.ndarray | None = field(default=None, repr=False)


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


def _spread(s: pl.Series, k: int, seed: int = 11) -> pl.Series:
    """Up to k values drawn from across the whole series, so sorted files don't fool a probe."""
    idx = sample_indices(s.len(), k, seed)
    return s if idx is None else s.gather(idx)


def _datetime_format(s: pl.Series) -> tuple[str | None, float, str | None]:
    """(format, share of a spread probe it parses, note) for a string series."""
    probe = _spread(s.drop_nulls(), 5000).str.strip_chars()
    if probe.len() == 0:
        return None, 0.0, None
    rates: dict[str, float] = {}
    for fmt in DATETIME_FORMATS:
        try:
            parsed = probe.str.strptime(pl.Datetime("us"), fmt, strict=False)
        except Exception:  # noqa: BLE001 - polars raises for some format/type combos
            continue
        rates[fmt] = 1 - parsed.null_count() / probe.len()
    best = max(rates, key=rates.get, default=None)  # ties go to the earlier format
    if best is None or rates[best] == 0:
        return None, 0.0, None
    note = None
    if best in DAY_FIRST or best in MONTH_FIRST:
        best, note = _day_month_order(s, best)
    return best, rates[best], note


def _day_month_order(s: pl.Series, fmt: str) -> tuple[str, str | None]:
    """Settle dd/mm vs mm/dd on the whole column: a part above 12 can only be a day."""
    parts = s.drop_nulls().str.strip_chars().str.extract_groups(r"^(\d{1,2})/(\d{1,2})/")
    first = parts.struct.field("1").cast(pl.Int32, strict=False)
    second = parts.struct.field("2").cast(pl.Int32, strict=False)
    day_first, month_first = int((first > 12).sum()), int((second > 12).sum())
    month_fmt = MONTH_FIRST.get(fmt, fmt)
    if day_first > month_first:
        return DAY_FIRST[month_fmt], None
    if month_first > day_first:
        return month_fmt, None
    return month_fmt, "day and month are ambiguous (no part above 12); assumed month-first"


def detect_datetime_format(s: pl.Series) -> tuple[str | None, float]:
    """Best strptime format for a string series and the share of a spread probe it parses."""
    fmt, rate, _ = _datetime_format(s)
    return fmt, rate


def _yyyymmdd_rate(s: pl.Series) -> float:
    """Share of non-null values that are valid YYYYMMDD dates (integers, or 8-digit strings)."""
    nn = s.drop_nulls()
    if nn.len() == 0:
        return 0.0
    if nn.dtype == pl.String:
        txt = nn.str.strip_chars()
        txt = txt.filter(txt.str.contains(r"^\d{8}$"))
    elif nn.dtype.is_numeric():
        x = nn.cast(pl.Float64)
        txt = x.filter((x >= 19000101) & (x <= 21001231) & (x == x.round(0))).cast(pl.Int64).cast(pl.String)
    else:
        return 0.0
    return txt.str.strptime(pl.Datetime("us"), YYYYMMDD, strict=False).drop_nulls().len() / nn.len()


EPOCH_TO_US = {"epoch_s": 1e6, "epoch_ms": 1e3, "epoch_us": 1.0, "epoch_ns": 1e-3}


def _numeric_datetime_format(s: pl.Series) -> str:
    """YYYYMMDD, or the epoch unit (s, ms, us, ns) that the values' magnitude implies."""
    if _yyyymmdd_rate(s) >= 0.99:
        return YYYYMMDD
    biggest = null_infinite(s.cast(pl.Float64).fill_nan(None)).abs().max() or 0
    return next((u for u, top in (("epoch_s", 1e11), ("epoch_ms", 1e14), ("epoch_us", 1e17)) if biggest < top),
                "epoch_ns")


def _numeric_to_datetime(s: pl.Series, fmt: str | None) -> pl.Series:
    """YYYYMMDD integers or epoch timestamps; without `fmt` it is judged from this series alone."""
    fmt = fmt if fmt == YYYYMMDD or fmt in EPOCH_TO_US else _numeric_datetime_format(s)
    if fmt == YYYYMMDD:
        ints = s.cast(pl.Float64).fill_nan(None).round(0).cast(pl.Int64, strict=False)
        return ints.cast(pl.String).str.strptime(pl.Datetime("us"), YYYYMMDD, strict=False)
    x = null_infinite(s.cast(pl.Float64).fill_nan(None))
    return pl.from_epoch((x * EPOCH_TO_US[fmt]).round(0).cast(pl.Int64), time_unit="us").alias(s.name)


def column_datetime_format(s: pl.Series) -> str | None:
    """The format to read a whole column as dates with, so every export chunk parses it the same way."""
    if s.dtype == pl.String:
        return detect_datetime_format(s.str.strip_chars())[0]
    if s.dtype.is_numeric():
        return _numeric_datetime_format(s)
    return None


def parse_datetime(s: pl.Series, fmt: str | None) -> pl.Series:
    if s.dtype == pl.String:
        txt = s.str.strip_chars()
        fmt = fmt or detect_datetime_format(txt)[0]
        if fmt is None:
            return pl.Series(s.name, [None] * s.len(), dtype=pl.Datetime("us"))
        out = txt.str.strptime(pl.Datetime("us"), fmt, strict=False)
    elif s.dtype == pl.Date:
        out = s.cast(pl.Datetime("us"))
    elif s.dtype.is_numeric():
        out = _numeric_to_datetime(s, fmt)
    else:
        out = s
    if isinstance(out.dtype, pl.Datetime) and out.dtype.time_zone is not None:
        out = out.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    return out


def detect_decimal_comma(s: pl.Series) -> bool:
    """True when a text column writes decimals with a comma ("1,5"), judged on the whole column.

    "1,500" reads either way, so only values that can't be thousands groups count as evidence, and any
    dot decimal rules the reading out.
    """
    txt = s.drop_nulls().str.strip_chars()
    with_comma = int(txt.str.contains(",", literal=True).sum())
    if not with_comma or txt.str.contains(".", literal=True).any():
        return False
    decimal = txt.str.contains(DECIMAL_COMMA_RE)
    return bool((decimal & ~txt.str.contains(THOUSANDS_RE)).any()) and int(decimal.sum()) * 2 >= with_comma


def null_infinite(s: pl.Series) -> pl.Series:
    return pl.select(pl.when(s.is_infinite()).then(None).otherwise(s)).to_series().alias(s.name)


def coerce_numeric(s: pl.Series, decimal_comma: bool | None = None, keep_inf: bool = False) -> pl.Series:
    """Float64 view of a column. Text that doesn't parse, NaN and (unless keep_inf) ±inf become null.

    Commas are thousands separators only in valid groups ("1,234.5"). With `decimal_comma` (detected from
    the series when None) "1,5" is 1.5. Any other value with a comma doesn't parse.
    """
    if s.dtype == pl.String:
        txt = s.str.strip_chars()
        if decimal_comma is None:
            decimal_comma = detect_decimal_comma(txt)
        if decimal_comma:
            fixed = pl.when(txt.str.contains(DECIMAL_COMMA_RE)).then(txt.str.replace(",", ".", literal=True))
        else:
            fixed = pl.when(txt.str.contains(THOUSANDS_RE)).then(txt.str.replace_all(",", "", literal=True))
        out = pl.select(fixed.otherwise(txt)).to_series().cast(pl.Float64, strict=False)
    elif s.dtype == pl.Boolean:
        out = s.cast(pl.Float64)
    elif s.dtype.is_temporal():
        out = s.cast(pl.Datetime("us")).dt.epoch("s").cast(pl.Float64)
    else:
        out = s.cast(pl.Float64)
    out = out.fill_nan(None).alias(s.name)
    return out if keep_inf else null_infinite(out)


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
    if dt.is_numeric() and _looks_yyyymmdd(s, name):
        hints["datetime_format"] = YYYYMMDD
        return "datetime", 0.85, ["integers that read as YYYYMMDD dates"], hints
    if dt.is_numeric() and n_unique >= 20 and (unit := _epoch_unit(s, name)):
        return "datetime", 0.7, [f"integers in the range of Unix timestamps in {unit} (2000–2100)"], hints
    if dt.is_numeric():
        t, c, r = _infer_numeric(s.drop_nulls(), name, dt, n_unique, n_nonnull)
        return t, c, r, hints
    if dt == pl.String:
        return _infer_string(s, name, n_unique, n_nonnull)
    return "categorical", 0.4, [f"unrecognised dtype {dt}; treating as categorical"], hints


# Epoch values for 2000-01-01 .. 2100-01-01 per unit, the magnitudes _numeric_to_datetime reads them in.
EPOCH_RANGES = {"seconds": (946_684_800, 4_102_444_800), "milliseconds": (946_684_800e3, 4_102_444_800e3),
                "microseconds": (946_684_800e6, 4_102_444_800e6), "nanoseconds": (946_684_800e9, 4_102_444_800e9)}


def _epoch_unit(s: pl.Series, name: str) -> str | None:
    """The unit when every value is a whole number inside one unit's 2000–2100 epoch range (and the name
    isn't key-like), else None."""
    if ID_NAME_RE.search(name):
        return None
    x = s.drop_nulls().cast(pl.Float64)
    lo, hi = x.min(), x.max()
    if lo is None or not bool((x == x.round(0)).all()):
        return None
    return next((u for u, (a, b) in EPOCH_RANGES.items() if a <= lo and hi <= b), None)


def _looks_yyyymmdd(s: pl.Series, name: str) -> bool:
    return not ID_NAME_RE.search(name) and _yyyymmdd_rate(s) >= 0.99


def _infer_numeric(nn: pl.Series, name: str, dt: pl.DataType, n_unique: int, n_nonnull: int):
    unique_ratio = n_unique / max(n_nonnull, 1)
    if dt.is_float():
        integral = bool((nn == nn.round(0)).all())
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
    probe = _spread(nn, 20_000)
    unique_ratio = n_unique / max(n_nonnull, 1)

    if _looks_yyyymmdd(nn, name):
        hints["datetime_format"] = YYYYMMDD
        return "datetime", 0.85, ["8-digit values that read as YYYYMMDD dates"], hints

    # Parse rates come from the whole column, so a clean head can't hide junk further down.
    decimal_comma = detect_decimal_comma(nn)
    as_num = coerce_numeric(nn, decimal_comma, keep_inf=True)
    num_rate = 1 - as_num.null_count() / max(nn.len(), 1)
    if num_rate >= 0.98:
        t, c, r = _infer_numeric(null_infinite(as_num).drop_nulls(), name, pl.Float64, n_unique, n_nonnull)
        hints["stored_as_text"] = True
        if decimal_comma:
            hints["decimal_comma"] = True
            r = ["comma used as the decimal separator", *r]
        return t, round(c * 0.9, 2), [f"stored as text but {num_rate:.1%} of values parse as numbers", *r], hints

    fmt, probe_rate, note = _datetime_format(nn)
    dt_rate = 1 - parse_datetime(nn, fmt).null_count() / max(nn.len(), 1) if fmt and probe_rate >= 0.9 else 0.0
    if fmt and dt_rate >= 0.95:
        hints["datetime_format"] = fmt
        return "datetime", round(0.8 + 0.2 * dt_rate, 2), [
            f"{dt_rate:.1%} of values parse as dates with format {fmt}", *([note] if note else [])
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
    x = x[np.isfinite(x)]
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
    # replace_strict sends nulls to `default` too, so missing has to be set before it.
    mapped = pl.select(pl.when(as_str.is_null()).then(missing)
                       .otherwise(as_str.replace_strict(mapping, default=other, return_dtype=pl.Int32)))
    return mapped.to_series().to_numpy().astype(np.int32)


def screen_codes(s: pl.Series, corr_idx: np.ndarray | None) -> np.ndarray:
    """A code per distinct value on the association sample, with no level cap (null is a value too)."""
    if s.dtype.is_float():
        s = s.fill_nan(None)
    if corr_idx is not None:
        s = s.gather(corr_idx)
    return s.cast(pl.String).fill_null("\0null").rank("dense").to_numpy().astype(np.int64)


def _failed_examples(raw: pl.Series, parsed: pl.Series, k: int = 5) -> list[str]:
    failed = raw.filter(raw.is_not_null() & parsed.is_null())
    return [str(v)[:100] for v in failed.unique(maintain_order=True).head(k).to_list()]


def corr_vector(s: pl.Series, t: str, datetime_format: str | None, corr_idx: np.ndarray | None,
                corr_max_levels: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(numeric, codes) association-sample vectors for a column already typed `t`, as profile_column builds them."""
    if s.dtype.is_float():
        s = s.fill_nan(None)
    if corr_idx is not None:
        s = s.gather(corr_idx)
    if t in NUMERIC:
        return coerce_numeric(s).to_numpy().astype(np.float64), None
    if t == "datetime":
        return parse_datetime(s, datetime_format).dt.epoch("s").cast(pl.Float64).to_numpy().astype(np.float64), None
    if t in ("categorical", "binary"):
        return None, _codes(s, corr_max_levels)
    return None, None


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
    inf_count = 0
    if s.dtype.is_float():
        s = s.fill_nan(None)
        inf_count = int(s.is_infinite().sum())
        s = null_infinite(s) if inf_count else s
    n = s.len()
    nulls = s.null_count()
    n_nonnull = n - nulls
    n_unique = int(s.drop_nulls().n_unique()) if n_nonnull else 0

    t, conf, reasons, hints = infer_type(s, name, n_unique, n_nonnull)
    if is_target and target_problem("categorical", n_unique):
        reasons = [f"named as a target but can't be a target: {target_problem('categorical', n_unique)}", *reasons]
        is_target = False
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

    unparseable = text_inf = 0
    if t in NUMERIC:
        raw = coerce_numeric(s, hints.get("decimal_comma", False), keep_inf=True)
        text_inf = int(raw.is_infinite().sum())
        num = null_infinite(raw)
        unparseable = raw.null_count() - nulls
        if unparseable:
            stats["unparseable_examples"] = _failed_examples(s, raw)
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
        unparseable = parsed.null_count() - nulls
        if unparseable:
            stats["unparseable_examples"] = _failed_examples(s, parsed)
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
    inf_count += text_inf
    if unparseable or text_inf:
        # The export reads these as missing, so the profile counts them as missing too (float infs already are).
        summary["null_count"] = nulls + unparseable + text_inf
        summary["null_pct"] = summary["null_count"] / n
    if unparseable:
        stats["unparseable_count"] = unparseable
        what = "dates" if t == "datetime" else "numbers"
        reasons = [*reasons, f"{unparseable:,} value(s) ({unparseable / n:.2%}) don't parse as {what} and are "
                   f"treated as missing"]
        conf = min(conf, 0.65)
    if inf_count:
        stats["inf_count"] = inf_count
        reasons = [*reasons, f"{inf_count:,} infinite value(s) are treated as missing"]
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
        screen_codes=screen_codes(s, corr_idx) if needs_screen(t, n_unique, is_target, corr_max_levels) else None,
    )


def needs_screen(t: str, n_unique: int, is_target: bool, corr_max_levels: int) -> bool:
    """Fields the capped associations can't vouch for: free text, and categoricals past the level cap."""
    return is_target or t == "text" or (t in ("categorical", "binary") and n_unique > corr_max_levels)


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
