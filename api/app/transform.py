"""Turn reviewed field decisions into a fitted sklearn preprocessing pipeline + model-ready Parquet.

Pipeline per field (only `action == keep`):
    numeric      : [SimpleImputer] -> [clip | signed-log1p] -> [scaler]   (+ optional missing indicator)
    categorical  : SimpleImputer -> OneHot | Ordinal | Frequency
    binary       : 0/1 (+ missing indicator when missing values get their own label)
    text         : TF-IDF -> TruncatedSVD   |   length / word-count stats
    datetime     : date parts + epoch days -> [SimpleImputer]

Leakage hygiene: the split is assigned first and every statistic (medians, vocabularies,
category levels, scaler params) is fitted on training rows only. Fitting uses a sample of
the training split; transformation streams through the full file in chunks.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from . import bands
from .config import get_settings
from .parquet_rows import read_rows
from .profiling import coerce_numeric, null_infinite, parse_datetime
from .semantics import (DEFAULT_BIN_COUNT, DEFAULT_FAMILY, DEFAULT_PARAMS, FAMILIES, GROUP_REST, NON_FEATURE_TYPES,
                        NUMERIC, grouped_target)

SPLIT_COL = "__split"
ROW_ID = "__row_id"
SPLIT_CODES = {"train": 0, "valid": 1, "test": 2}


# =========================================================================== transformers
class _NamesOneToOne:
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)


class SignedLog1p(_NamesOneToOne, TransformerMixin, BaseEstimator):
    """sign(x) * log1p(|x|): compresses heavy tails on both sides, NaN-safe."""

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.sign(X) * np.log1p(np.abs(X))


class Winsorizer(_NamesOneToOne, TransformerMixin, BaseEstimator):
    """Clip each column to its [lower, upper] training quantiles."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        self.low_ = np.nanquantile(X, self.lower, axis=0)
        self.high_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=np.float64), self.low_, self.high_)


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """Replace each level by its training frequency; unseen levels -> 0."""

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        self.n_features_in_ = df.shape[1]
        self.feature_names_in_ = np.asarray([str(c) for c in df.columns], dtype=object)
        self.maps_ = [df[c].astype(str).value_counts(normalize=True).to_dict() for c in df.columns]
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        out = np.zeros(df.shape, dtype=np.float64)
        for k, c in enumerate(df.columns):
            out[:, k] = df[c].astype(str).map(self.maps_[k]).fillna(0.0).to_numpy()
        return out

    def get_feature_names_out(self, input_features=None):
        names = input_features if input_features is not None else self.feature_names_in_
        return np.asarray([f"{n}_freq" for n in names], dtype=object)


class BinaryEncoder(TransformerMixin, BaseEstimator):
    """Two-level column -> 0/1 (1 = the later level in natural order, as targets are encoded).

    Missing and unseen values become the most frequent level (`fill="most_frequent"`), the level named by
    `fill`, or otherwise 0 plus a `__isna` indicator column, so missing never sits between the two levels.
    """

    def __init__(self, fill: str | None = "most_frequent"):
        self.fill = fill

    def fit(self, X, y=None):
        s = pd.Series(np.asarray(X, dtype=object).ravel())
        counts = s[s.notna()].astype(str).value_counts()
        top = counts.index[:2].tolist() or ["0"]
        self.levels_ = sorted(top, key=_natural_key)
        self.n_features_in_ = 1
        self.feature_names_in_ = np.asarray([str(getattr(X, "columns", ["x0"])[0])], dtype=object)
        if self.fill == "most_frequent":
            self.fill_code_, self.indicator_ = float(self.levels_.index(top[0])), False
        elif self.fill in self.levels_:
            self.fill_code_, self.indicator_ = float(self.levels_.index(self.fill)), False
        else:
            self.fill_code_, self.indicator_ = 0.0, True
        return self

    def transform(self, X):
        s = pd.Series(np.asarray(X, dtype=object).ravel())
        txt = s.where(s.isna(), s.astype(str))
        known = txt.isin(self.levels_).to_numpy()
        value = np.where(known, (txt == self.levels_[-1]).to_numpy(), self.fill_code_).astype(np.float64)
        if len(self.levels_) == 1:
            value = np.where(known, 0.0, self.fill_code_)
        return np.column_stack([value, ~known]).astype(np.float64) if self.indicator_ else value[:, None]

    def get_feature_names_out(self, input_features=None):
        name = (input_features if input_features is not None else self.feature_names_in_)[0]
        return np.asarray([f"{name}_{self.levels_[-1]}", *([f"{name}__isna"] if self.indicator_ else [])],
                          dtype=object)


class TextFeatures(TransformerMixin, BaseEstimator):
    """Single text column -> TF-IDF -> SVD components (and/or simple length statistics)."""

    def __init__(self, mode: str = "tfidf_svd", fill_value: str = "", svd_components: int = 32,
                 max_features: int = 20000, min_df: int = 3, random_state: int = 0):
        self.mode = mode
        self.fill_value = fill_value
        self.svd_components = svd_components
        self.max_features = max_features
        self.min_df = min_df
        self.random_state = random_state

    def _docs(self, X) -> list[str]:
        s = pd.Series(np.asarray(X, dtype=object).ravel())
        return s.where(s.notna(), self.fill_value).astype(str).tolist()

    def fit(self, X, y=None):
        self.n_features_in_ = 1
        docs = self._docs(X)
        if self.mode == "tfidf_svd":
            self.vectorizer_ = TfidfVectorizer(
                max_features=self.max_features, min_df=min(self.min_df, max(1, len(docs) // 100)),
                ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32, stop_words="english",
            )
            try:
                tf = self.vectorizer_.fit_transform(docs)
            except ValueError:  # empty vocabulary (e.g. all-empty column)
                self.vectorizer_, self.svd_ = None, None
                self.n_out_ = 0
                return self
            if tf.shape[1] < 2:  # SVD needs two terms; a single term is its own feature
                self.svd_, self.n_out_ = None, tf.shape[1]
                return self
            k = min(self.svd_components, tf.shape[1] - 1)
            self.svd_ = TruncatedSVD(n_components=k, random_state=self.random_state).fit(tf)
            self.n_out_ = k
        else:
            self.n_out_ = 2
        return self

    def transform(self, X):
        docs = self._docs(X)
        if self.mode == "tfidf_svd":
            if self.vectorizer_ is None:
                return np.zeros((len(docs), 0), dtype=np.float32)
            tf = self.vectorizer_.transform(docs)
            return (tf.toarray() if self.svd_ is None else self.svd_.transform(tf)).astype(np.float32)
        s = pd.Series(docs)
        return np.column_stack([s.str.len().to_numpy(float), s.str.split().str.len().fillna(0).to_numpy(float)])

    def get_feature_names_out(self, input_features=None):
        if self.mode == "tfidf_svd":
            return np.asarray([f"svd{i:02d}" for i in range(self.n_out_)], dtype=object)
        return np.asarray(["n_chars", "n_words"], dtype=object)


class DatetimeParts(TransformerMixin, BaseEstimator):
    """datetime64 column(s) -> year, month, day, weekday, hour, day-of-year, epoch days (NaN for NaT)."""

    PARTS = ("year", "month", "day", "weekday", "hour", "dayofyear", "epoch_days")

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        self.n_features_in_ = df.shape[1]
        self.feature_names_in_ = np.asarray([str(c) for c in df.columns], dtype=object)
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        blocks = []
        for c in df.columns:
            d = pd.to_datetime(df[c], errors="coerce")
            epoch = (d - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)
            blocks.append(np.column_stack([
                d.dt.year, d.dt.month, d.dt.day, d.dt.weekday, d.dt.hour, d.dt.dayofyear, epoch,
            ]).astype(np.float64))
        return np.hstack(blocks)

    def get_feature_names_out(self, input_features=None):
        names = input_features if input_features is not None else self.feature_names_in_
        return np.asarray([f"{n}_{p}" for n in names for p in self.PARTS], dtype=object)


# =========================================================================== building
@dataclass
class FieldSpec:
    name: str
    semantic_type: str
    imputation: str
    impute_value: str | None
    add_missing_indicator: bool
    transform: str
    scaling: str
    encoding: str
    params: dict[str, Any]
    datetime_format: str | None = None
    decimal_comma: bool = False


SCALERS = {
    "standard": lambda: StandardScaler(),
    "robust": lambda: RobustScaler(),
    "minmax": lambda: MinMaxScaler(),
    "quantile_normal": lambda: QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                                                   subsample=200_000, random_state=0),
}


def _numeric_pipeline(f: FieldSpec) -> Pipeline | str:
    steps: list[tuple[str, Any]] = []
    # Fill first, so a constant fill value is in the field's own units, not log or clipped ones.
    if f.imputation != "none":
        kw: dict[str, Any] = {"strategy": f.imputation, "keep_empty_features": True}
        if f.imputation == "constant":
            kw["fill_value"] = float(f.impute_value or 0)
        steps.append(("impute", SimpleImputer(**kw)))
    if f.transform == "clip":
        steps.append(("clip", Winsorizer()))
    elif f.transform == "log1p":
        steps.append(("log1p", SignedLog1p()))
    if f.scaling in SCALERS:
        steps.append(("scale", SCALERS[f.scaling]()))
    return Pipeline(steps) if steps else "passthrough"


def _categorical_pipeline(f: FieldSpec) -> Pipeline | BinaryEncoder:
    if f.semantic_type == "binary" or f.encoding == "binary":
        return BinaryEncoder("most_frequent" if f.imputation == "most_frequent" else f.impute_value)
    if f.imputation == "most_frequent":
        imputer = SimpleImputer(strategy="most_frequent", keep_empty_features=True)
    else:
        imputer = SimpleImputer(strategy="constant", fill_value=f.impute_value or "__missing__",
                                keep_empty_features=True)
    p = {**DEFAULT_PARAMS.get(f.encoding, {}), **(f.params or {})}
    enc: Any
    if f.encoding == "onehot":
        enc = OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False, dtype=np.float32,
                            max_categories=int(p.get("max_categories", 30)),
                            min_frequency=float(p.get("min_frequency", 0.01)))
    elif f.encoding == "ordinal":
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32,
                             min_frequency=float(p.get("min_frequency", 0.001)))
    else:
        enc = FrequencyEncoder()
    return Pipeline([("impute", imputer), ("encode", enc)])


def build_preprocessor(specs: list[FieldSpec], seed: int = 0) -> ColumnTransformer:
    transformers: list[tuple[str, Any, Any]] = []
    sources: dict[str, str] = {}  # transformer name -> source field
    # A step may not share a name with a ColumnTransformer argument (a field called "remainder").
    used: set[str] = set(ColumnTransformer([]).get_params(deep=False))

    def tname(field: str, suffix: str = "") -> str:
        # sklearn forbids "__" in step names; it is also the output-name separator.
        base = re.sub(r"_{2,}", "_", field + suffix).strip("_") or "field"
        cand, k = base, 2
        while cand in used:
            cand = f"{base}_{k}"
            k += 1
        used.add(cand)
        sources[cand] = field
        return cand

    for f in specs:
        t = f.semantic_type
        if t in NUMERIC:
            transformers.append((tname(f.name), _numeric_pipeline(f), [f.name]))
            if f.add_missing_indicator:
                transformers.append(
                    (tname(f.name, "_isna"), MissingIndicator(features="all", error_on_new=False), [f.name])
                )
        elif t in ("categorical", "binary"):
            transformers.append((tname(f.name), _categorical_pipeline(f), [f.name]))
        elif t == "text":
            p = {**DEFAULT_PARAMS["tfidf_svd"], **(f.params or {})}
            transformers.append((tname(f.name), TextFeatures(
                mode=f.encoding, fill_value=f.impute_value or "", svd_components=int(p["svd_components"]),
                max_features=int(p["max_features"]), min_df=int(p["min_df"]), random_state=seed,
            ), f.name))  # scalar column -> 1-D input
        elif t == "datetime":
            steps: list[tuple[str, Any]] = [("parts", DatetimeParts())]
            if f.imputation == "median":
                steps.append(("impute", SimpleImputer(strategy="median", keep_empty_features=True)))
            if f.scaling in SCALERS:
                steps.append(("scale", SCALERS[f.scaling]()))
            transformers.append((tname(f.name), Pipeline(steps), [f.name]))
    ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True,
                           sparse_threshold=0.0, n_jobs=None)
    ct.rainforest_sources_ = sources
    return ct


class NumericCoercer:
    """coerce_numeric with the field's profiled number format; counts the ±inf it turns into missing values."""

    def __init__(self, decimal_comma: bool = False):
        self.decimal_comma = decimal_comma
        self.inf_count = 0

    def __call__(self, s: pl.Series) -> pl.Series:
        raw = coerce_numeric(s, self.decimal_comma, keep_inf=True)
        self.inf_count += int(raw.is_infinite().sum())
        return null_infinite(raw)


def coercion_exprs(specs: list[FieldSpec]) -> dict[str, Callable[[pl.Series], pl.Series]]:
    """Deterministic, fit-free casting of raw columns into what the preprocessor expects."""
    out: dict[str, Callable[[pl.Series], pl.Series]] = {}
    for f in specs:
        if f.semantic_type in NUMERIC:
            out[f.name] = NumericCoercer(f.decimal_comma)
        elif f.semantic_type == "datetime":
            fmt = f.datetime_format
            out[f.name] = lambda s, fmt=fmt: parse_datetime(s, fmt)
        else:
            out[f.name] = lambda s: s.cast(pl.String)
    return out


def to_pandas_frame(df: pl.DataFrame, specs: list[FieldSpec], coercers, consume: bool = False) -> pd.DataFrame:
    """`consume` drops each column from `df` once converted, so the two copies never coexist in full."""
    data: dict[str, Any] = {}
    for f in specs:
        s = coercers[f.name](df.drop_in_place(f.name) if consume else df[f.name])
        if f.semantic_type in NUMERIC:
            data[f.name] = s.to_numpy().astype(np.float64)
        elif f.semantic_type == "datetime":
            data[f.name] = s.to_pandas()
        else:
            arr = s.to_numpy().astype(object)
            arr[pd.isna(arr)] = np.nan  # SimpleImputer's default missing marker for object arrays
            data[f.name] = arr
    return pd.DataFrame(data)


def readable_feature_names(pre: ColumnTransformer) -> list[str]:
    """Drop sklearn's redundant '<transformer>__' prefix when the inner name already starts with the field."""
    out: list[str] = []
    for name in pre.get_feature_names_out():
        tname, _, rest = str(name).partition("__")
        field = pre.rainforest_sources_.get(tname, tname)
        if rest.startswith("missingindicator_"):
            out.append(f"{field}__isna")
        elif rest == field or rest.startswith(field + "_"):
            out.append(rest)
        else:
            out.append(f"{field}__{rest}")
    return out


def safe_feature_names(names: list[str], reserved: Iterable[str] = ()) -> list[str]:
    """LightGBM rejects JSON special chars in feature names; also guarantee uniqueness.

    `reserved` names (targets, __row_id, __split) share the output file, so a feature never takes one.
    """
    out, seen = [], set(reserved)
    for n in names:
        base = re.sub(r"[^0-9A-Za-z_]+", "_", n).strip("_") or "f"
        cand, k = base, 2
        while cand in seen:
            cand = f"{base}_{k}"
            k += 1
        seen.add(cand)
        out.append(cand)
    return out


# =========================================================================== export
def estimated_width(specs: list[FieldSpec]) -> int:
    """Upper bound on the output features, known before fitting."""
    n = 0
    for f in specs:
        p = {**DEFAULT_PARAMS.get(f.encoding, {}), **(f.params or {})}
        if f.semantic_type in NUMERIC:
            n += 1 + int(f.add_missing_indicator)
        elif f.semantic_type == "text":
            n += int(p.get("svd_components", 32)) if f.encoding == "tfidf_svd" else 2
        elif f.semantic_type == "datetime":
            n += len(DatetimeParts.PARTS)
        elif f.semantic_type == "categorical" and f.encoding == "onehot":
            n += int(p.get("max_categories", 30))
        elif f.semantic_type == "binary":
            n += 2
        else:
            n += 1
    return max(n, 1)


def transform_into(pre: ColumnTransformer, frame: pd.DataFrame, n_out: int) -> np.ndarray:
    """`pre.transform(frame)` as column-major float32, filled one transformer at a time, so the float64 blocks
    and their hstack never exist at once. Column-major hands Arrow each feature as a contiguous buffer."""
    out = np.empty((len(frame), n_out), dtype=np.float32, order="F")
    for name, trans, cols in pre.transformers_:
        sl = pre.output_indices_[name]
        if sl.start == sl.stop or isinstance(trans, str):
            continue
        out[:, sl] = trans.transform(frame[cols])
    return out


def rows_for_budget(mb: int, width: int, floor: int = 1_000) -> int:
    """Rows whose dense float64 output fits in `mb` megabytes."""
    return max(floor, (mb << 20) // (8 * max(width, 1)))


def assign_splits(
    y: pd.DataFrame, valid_size: float, test_size: float, seed: int,
    strata_edges: dict[str, list[float]] | None = None, keep: np.ndarray | None = None,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return split codes per row (255 = dropped because a target is missing).

    Stratifies on the target combination, else on each target alone, else not at all. Strata with a single
    row are pooled, so one odd label doesn't switch stratification off. A numeric target (a key of `strata_edges`)
    stratifies on bands, so each split covers its range: the given cut points, or 5 equal-count bands when empty.
    """
    n = len(y)
    codes = np.full(n, 255, dtype=np.uint8)
    labeled = y.notna().all(axis=1).to_numpy()
    info: dict[str, Any] = {"dropped_unlabeled": int((~labeled).sum()), "stratified": False, "stratified_on": None}
    if keep is not None:
        labeled &= keep
    idx = np.flatnonzero(labeled)

    holdout = valid_size + test_size
    if holdout <= 0:
        codes[idx] = 0
        return codes, info
    if groups is not None:
        # Every row of a group lands in one split, so validation never sees a patient (or mix) training saw.
        from sklearn.model_selection import GroupShuffleSplit

        g = groups[idx]
        tr, rest = next(GroupShuffleSplit(1, test_size=holdout, random_state=seed).split(idx, groups=g))
        codes[idx[tr]] = 0
        rest_idx = idx[rest]
        if test_size <= 0 or valid_size <= 0:
            codes[rest_idx] = 1 if test_size <= 0 else 2
        else:
            v, t = next(GroupShuffleSplit(1, test_size=test_size / holdout, random_state=seed).split(
                rest_idx, groups=groups[rest_idx]))
            codes[rest_idx[v]], codes[rest_idx[t]] = 1, 2
        info["grouped"] = True
        return codes, info
    strata = y.copy()
    for c, edges in (strata_edges or {}).items():
        vals = strata[c].to_numpy(dtype=float)
        strata[c] = bands.assign(vals, edges if edges else bands.quantile_edges(vals, 5))
    labels = strata.loc[labeled].astype(str)
    candidates = [list(y.columns)] + ([[c] for c in y.columns] if y.shape[1] > 1 else []) + [None]
    for cols in candidates:
        keys = None if cols is None else _pool_singletons(labels[cols].agg("|".join, axis=1).to_numpy())
        try:
            train_idx, rest_idx, _, rest_keys = train_test_split(
                idx, idx if keys is None else keys, test_size=holdout, random_state=seed, stratify=keys
            )
        except ValueError:  # more strata than the holdout has rows
            continue
        info["stratified"], info["stratified_on"] = cols is not None, cols
        break
    codes[train_idx] = 0
    if test_size <= 0:
        codes[rest_idx] = 1
    elif valid_size <= 0:
        codes[rest_idx] = 2
    else:
        strat2 = None if info["stratified_on"] is None else _pool_singletons(rest_keys)
        try:
            v_idx, t_idx = train_test_split(rest_idx, test_size=test_size / holdout, random_state=seed, stratify=strat2)
        except ValueError:
            v_idx, t_idx = train_test_split(rest_idx, test_size=test_size / holdout, random_state=seed)
        codes[v_idx], codes[t_idx] = 1, 2
    return codes, info


def _pool_singletons(keys: np.ndarray) -> np.ndarray:
    values, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    lone = counts[inverse] < 2
    if not lone.any():
        return keys
    out = keys.astype(object)
    # Two or more lone rows form their own stratum; a single one joins the largest.
    out[lone] = "\0pooled" if lone.sum() >= 2 else values[counts.argmax()]
    return out


def check_target_classes(y: pd.DataFrame) -> None:
    """Every class needs two rows to reach both training and evaluation; fail before any work is done."""
    for t in y.columns:
        counts = y[t].dropna().astype(str).value_counts()
        lone = counts.index[counts < 2].tolist()
        if lone:
            shown = ", ".join(repr(v) for v in lone[:10]) + (f" and {len(lone) - 10} more" if len(lone) > 10 else "")
            raise ValueError(
                f"Target {t!r} has {len(lone)} class(es) with a single row: {shown}. Every class needs at least "
                "2 rows; fix or remove those rows, merge the class into another, or choose another target."
            )


def encode_targets(y: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Map each target's labels to 0..k-1 (sorted for determinism); NaN stays NaN."""
    encoded, classes = {}, {}
    for c in y.columns:
        s = y[c]
        labels = sorted(s.dropna().astype(str).unique().tolist(), key=_natural_key)
        mapping = {v: i for i, v in enumerate(labels)}
        encoded[c] = s.map(lambda v: mapping.get(str(v), np.nan) if pd.notna(v) else np.nan)
        classes[c] = labels
    return pd.DataFrame(encoded), classes


def row_rules(parquet_path: Path, config: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """The export's row rules: which rows stay (exact duplicates, excluded values, one row per key), the group code of
    each row when the split keeps a field's rows together, and how many rows each rule removed."""
    dedupe = bool(config.get("dedupe"))
    exclude = config.get("exclude") or []
    first = config.get("first_per") or None
    group_by = config.get("group_by") or None
    if not (dedupe or exclude or first or group_by):
        return None, None, {}
    df = pl.read_parquet(parquet_path).with_row_index("__i")
    keep = np.ones(df.height, dtype=bool)
    removed: dict[str, int] = {}
    if dedupe:
        dup = df.drop("__i").is_duplicated().to_numpy() & ~df.drop("__i").select(pl.struct(pl.all()).is_first_distinct()).to_series().to_numpy()
        removed["duplicates"] = int((dup & keep).sum())
        keep &= ~dup
    for rule in exclude:
        hit = df[rule["field"]].cast(pl.Utf8).is_in([str(v) for v in rule["values"]]).fill_null(False).to_numpy()
        removed[f"{rule['field']} in {', '.join(map(str, rule['values']))}"] = int((hit & keep).sum())
        keep &= ~hit
    if first:
        kept = df.filter(pl.Series(keep))
        if first.get("order"):
            kept = kept.sort([first["key"], first["order"]], nulls_last=True, maintain_order=True)
        firsts = set(kept.unique(subset=[first["key"]], keep="first", maintain_order=True)["__i"].to_list())
        not_first = keep & ~np.isin(df["__i"].to_numpy(), list(firsts))
        removed[f"later rows per {first['key']}"] = int(not_first.sum())
        keep &= ~not_first
    groups = None
    if group_by:
        groups = df[group_by].cast(pl.Utf8).fill_null("").rank("dense").to_numpy().astype(np.int64)
    return keep, groups, removed


def prepare_targets(parquet_path: Path, targets: list[str], target_tasks: dict[str, dict], config: dict[str, Any],
                    seed: int):
    """Read, check and encode the targets, and split the rows. Class targets become codes; a number stays a float; bands
    become codes of bands cut on the training rows. Numeric targets are read the way profiling read them, and a value
    that doesn't read as a number fails the export rather than silently losing its row."""
    tasks = {t: target_tasks.get(t, {}).get("task") or "classes" for t in targets}
    grouped = {t: pos for t in targets if (pos := grouped_target(target_tasks.get(t)))}
    for t in grouped:
        tasks[t] = "classes"
    numeric = [t for t in targets if tasks[t] != "classes"]
    raw = pl.read_parquet(parquet_path, columns=targets)
    y_raw = raw.to_pandas()
    for t, pos in grouped.items():
        col = y_raw[t]
        label = " / ".join(pos)
        y_raw[t] = col.where(col.isna(), np.where(col.astype(str).isin(pos), label, GROUP_REST))
    for t in numeric:
        num = coerce_numeric(raw[t], target_tasks[t].get("decimal_comma"))
        given = raw[t].fill_nan(None) if raw[t].dtype.is_float() else raw[t]
        if lost := int(given.is_not_null().sum() - num.is_not_null().sum()):
            raise ValueError(f"Target {t!r} has {lost:,} value(s) that don't read as numbers (or are infinite); fix "
                             "them or predict it as classes.")
        y_raw[t] = num.to_numpy()
    class_targets = [t for t in targets if tasks[t] == "classes"]
    check_target_classes(y_raw[class_targets])
    chosen = {t: [float(e) for e in (target_tasks[t].get("bins") or {}).get("edges", [])] for t in numeric}
    keep, groups, removed = row_rules(parquet_path, config)
    if keep is not None and not keep.any():
        raise ValueError("The row rules leave no rows to export.")
    split, split_info = assign_splits(y_raw, float(config.get("valid_size", 0.15)),
                                      float(config.get("test_size", 0.15)), seed, strata_edges=chosen, keep=keep,
                                      groups=groups)
    if removed:
        split_info["rows_removed"] = removed
    if config.get("group_by"):
        split_info["grouped_on"] = config["group_by"]
    train_rows = split == 0
    y_enc, classes = encode_targets(y_raw[class_targets])
    target_bins, target_stats = {}, {}
    for t in numeric:
        vals = y_raw[t].to_numpy(dtype=float)
        if tasks[t] == "bins":
            spec = target_tasks[t].get("bins") or {"k": DEFAULT_BIN_COUNT}
            edges = bands.edges_for(spec, vals[train_rows])
            if thin := bands.thin_bands(vals[split != 255], edges):
                raise ValueError(f"Target {t!r}: band(s) {', '.join(thin)} have fewer than 2 rows; move or remove "
                                 "cut points, or use fewer bands.")
            y_enc[t] = bands.assign(vals, edges)
            classes[t] = bands.band_labels(edges)
            target_bins[t] = {"edges": edges, "labels": classes[t], "k": spec.get("k")}
        else:
            y_enc[t] = vals
            tr = vals[train_rows & ~np.isnan(vals)]
            if np.unique(tr).size < 2:
                raise ValueError(f"Target {t!r} has a single value in the training rows; there is nothing to predict.")
            target_stats[t] = {"mean": float(tr.mean()), "std": float(tr.std()), "min": float(tr.min()),
                               "max": float(tr.max())}
    for t, pos in grouped.items():
        target_bins[t] = {"positive": pos}
    info = {"target_tasks": tasks, "target_bins": target_bins, "target_stats": target_stats, "_groups": groups}
    return y_raw, y_enc[targets], classes, split, split_info, info


def _natural_key(v: str):
    try:
        return (0, float(v), v)
    except ValueError:
        return (1, 0.0, v)


def run_export(
    parquet_path: Path,
    specs: list[FieldSpec],
    targets: list[str],
    out_dir: Path,
    config: dict[str, Any],
    progress: Callable[[float, str], None],
    family: str = DEFAULT_FAMILY,
    target_tasks: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """`target_tasks`: per target {"task", "bins", "decimal_comma"}; a target left out is predicted as classes."""
    import joblib
    import sklearn

    settings = get_settings()
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 42))
    chunk_rows = int(config.get("chunk_rows", 200_000))
    fit_rows = int(config.get("fit_sample_rows", 500_000))
    specs = [f for f in specs if f.semantic_type not in NON_FEATURE_TYPES]
    if not specs:
        raise ValueError("No fields are marked 'keep' — nothing to export.")
    if not targets:
        raise ValueError("No target fields — mark at least one field as a target.")
    if clash := {ROW_ID, SPLIT_COL} & set(targets):
        raise ValueError(f"Target name(s) {', '.join(sorted(clash))} are reserved for the export's own columns.")

    n_rows = pl.scan_parquet(parquet_path).select(pl.len()).collect().item()

    # 1. targets + split ------------------------------------------------------------
    progress(0.02, "Assigning train / valid / test split")
    y_raw, y_enc, classes, split, split_info, target_info = prepare_targets(parquet_path, targets, target_tasks or {},
                                                                            config, seed)
    tasks = target_info["target_tasks"]
    groups = target_info.pop("_groups")
    if groups is not None:
        np.save(out_dir / "groups.npy", groups)

    # 2. fit preprocessor on a training sample ---------------------------------------
    progress(0.08, "Fitting preprocessing pipeline on training rows")
    train_idx = np.flatnonzero(split == 0)
    rng = np.random.default_rng(seed)
    # Fitting materialises the whole fit sample's output, so wide exports fit on fewer rows.
    fit_cap = rows_for_budget(settings.export_fit_mb, estimated_width(specs))
    fit_idx = np.sort(rng.choice(train_idx, size=min(fit_rows, fit_cap, train_idx.size), replace=False))
    feature_cols = [f.name for f in specs]
    pre = build_preprocessor(specs, seed)
    pre.fit(to_pandas_frame(read_rows(parquet_path, feature_cols, fit_idx), specs, coercion_exprs(specs),
                            consume=True))
    coercers = coercion_exprs(specs)  # fresh, so inf counts cover the written rows only
    raw_names = readable_feature_names(pre)
    feat_names = safe_feature_names(raw_names, reserved=[ROW_ID, SPLIT_COL, *targets])
    feature_source = _feature_sources(pre, feat_names)
    chunk_rows = min(chunk_rows, rows_for_budget(settings.export_chunk_mb, len(feat_names)))

    # 3. stream transform -> Parquet ---------------------------------------------------
    data_path = out_dir / "model_ready.parquet"
    arrow_fields = (
        [pa.field(ROW_ID, pa.int64()), pa.field(SPLIT_COL, pa.string())]
        + [pa.field(n, pa.float32()) for n in feat_names]
        + [pa.field(t, pa.float64() if tasks[t] == "regression" else pa.int32()) for t in targets]
    )
    arrow_schema = pa.schema(arrow_fields)
    split_names = np.array(["train", "valid", "test"], dtype=object)
    written = 0
    nan_counts = np.zeros(len(feat_names), dtype=np.int64)
    pf = pq.ParquetFile(parquet_path)
    offset = 0
    with pq.ParquetWriter(data_path, arrow_schema, compression="zstd") as writer:
        for batch in pf.iter_batches(batch_size=chunk_rows, columns=feature_cols):
            df = pl.from_arrow(batch)
            del batch
            assert isinstance(df, pl.DataFrame)
            rows = np.arange(offset, offset + df.height)
            offset += df.height
            keep = split[rows] != 255
            if not keep.any():
                continue
            df = df.filter(pl.Series(keep))
            rows = rows[keep]
            frame = to_pandas_frame(df, specs, coercers, consume=True)
            del df
            X = transform_into(pre, frame, len(feat_names))
            del frame
            for k in range(X.shape[1]):
                nan_counts[k] += np.count_nonzero(np.isnan(X[:, k]))
            if not FAMILIES[family]["handles_nan"] and nan_counts.any():
                bad = [feat_names[k] for k in np.flatnonzero(nan_counts)][:10]
                raise ValueError(
                    f"{family} models need complete data but these features still contain NaN: {', '.join(bad)}. "
                    "Set an imputation for the source fields."
                )
            cols: list[Any] = [pa.array(rows, pa.int64()), pa.array(split_names[split[rows]], pa.string())]
            cols += [pa.array(X[:, k]) for k in range(X.shape[1])]
            cols += [pa.array(y_enc[t].to_numpy()[rows].astype(np.float64 if tasks[t] == "regression" else np.int32))
                     for t in targets]
            writer.write_table(pa.Table.from_arrays(cols, schema=arrow_schema))
            del X, cols
            written += int(keep.sum())
            progress(0.15 + 0.8 * offset / max(n_rows, 1), f"Transformed {offset:,} / {n_rows:,} rows")

    # 4. artefacts ----------------------------------------------------------------------
    joblib.dump(pre, out_dir / "preprocessor.joblib", compress=3)
    split_counts = {k: int((split == v).sum()) for k, v in SPLIT_CODES.items()}
    manifest = {
        "model_family": family,
        "contains_nan": bool(nan_counts.any()),
        "nan_features": {feat_names[k]: int(nan_counts[k]) for k in np.flatnonzero(nan_counts)},
        "inf_values": {name: c.inf_count for name, c in coercers.items()
                       if isinstance(c, NumericCoercer) and c.inf_count},
        "rows_written": written,
        "n_features": len(feat_names),
        "feature_names": feat_names,
        "feature_source_field": feature_source,
        "targets": targets,
        "target_classes": {t: classes[t] for t in targets if tasks[t] != "regression"},
        "target_kinds": {t: "regression" if tasks[t] == "regression"
                         else ("binary" if len(classes[t]) == 2 else "multiclass") for t in targets},
        **target_info,
        "split_column": SPLIT_COL,
        "split_counts": split_counts,
        "split_info": split_info,
        "fit_sample_rows": int(fit_idx.size),
        "chunk_rows": chunk_rows,
        "config": config,
        "fields": [f.__dict__ for f in specs],
        "versions": {"sklearn": sklearn.__version__, "polars": pl.__version__, "numpy": np.__version__},
        "files": {"data": "model_ready.parquet", "preprocessor": "preprocessor.joblib",
                  **({"groups": "groups.npy"} if groups is not None else {})},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default))
    progress(1.0, "Done")
    return manifest


def _feature_sources(pre: ColumnTransformer, feat_names: list[str]) -> dict[str, str]:
    """Map each output feature to the source field it came from."""
    out: dict[str, str] = {}
    for name, sl in pre.output_indices_.items():
        if name == "remainder":
            continue
        for k in range(sl.start, sl.stop):
            out[feat_names[k]] = pre.rainforest_sources_.get(name, name)
    return out


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if math.isnan(o) else float(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))
