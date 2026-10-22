"""Transformers a field's settings can name besides semantics.py's fixed options, one TransformSpec each.

A spec's key is what `transform` or `encoding` stores. The catalog, the checks, the export pipeline and the decision
panel read specs from here, so adding a transformer is adding a spec. Nothing from the app is imported but
modeling.params: semantics.py imports this module.
"""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone

from ..modeling.params import Setting, value_problem

DIMENSIONS = ("transform", "encoding")
_BUILTIN_KEY = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class TargetInfo:
    name: str
    kind: Literal["binary", "multiclass", "continuous"]
    classes: tuple[str, ...] = ()


@dataclass(frozen=True)
class BuildContext:
    seed: int = 0
    targets: tuple[TargetInfo, ...] = ()
    target: TargetInfo | None = None


class PickTarget(TransformerMixin, BaseEstimator):
    """Fits a target-aware transformer on one column of the export's encoded targets, so a spec's builder returns a
    plain sklearn transformer that takes a 1-D y. Multiclass codes reach it as labels, so its output names carry the
    class; binary ones as 0/1, so it encodes the share of the second class in target_classes."""

    def __init__(self, transformer=None, target: str = "", kind: str = "binary", classes: tuple[str, ...] = ()):
        self.transformer = transformer
        self.target = target
        self.kind = kind
        self.classes = classes

    def _y(self, y):
        if y is None:
            raise ValueError(f"{self.target}: fitting needs the targets (a DataFrame with a {self.target!r} column)")
        col = np.asarray(y[self.target])
        if self.kind == "continuous":
            return col.astype(np.float64)
        codes = col.astype(np.int64)
        return codes if self.kind == "binary" else np.asarray(self.classes, dtype=object)[codes]

    def fit(self, X, y=None):
        self.transformer_ = clone(self.transformer).fit(X, self._y(y))
        return self

    def fit_transform(self, X, y=None, **fit_params):
        # TargetEncoder.fit_transform is what cross-fits; fit followed by transform would not.
        self.transformer_ = clone(self.transformer)
        return self.transformer_.fit_transform(X, self._y(y))

    def transform(self, X):
        return self.transformer_.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.transformer_.get_feature_names_out(input_features)


@dataclass(frozen=True)
class TransformSpec:
    key: str
    dimension: str
    types: tuple[str, ...]
    families: tuple[str, ...]
    label: str
    help: str
    build: Callable[[dict[str, Any], BuildContext], Any]
    settings: tuple[Setting, ...] = ()
    handles_nan: bool = True
    needs_target: bool = False
    requires: str | None = None
    source: str = "builtin"

    @property
    def available(self) -> bool:
        return self.requires is None or importlib.util.find_spec(self.requires) is not None

    def defaults(self) -> dict[str, Any]:
        return {s.name: s.default for s in self.settings}

    def to_json(self) -> dict:
        return {
            "key": self.key,
            "dimension": self.dimension,
            "types": list(self.types),
            "families": list(self.families),
            "label": self.label,
            "help": self.help,
            "settings": [s.to_json() for s in self.settings],
            "handles_nan": self.handles_nan,
            "needs_target": self.needs_target,
            "source": self.source,
        }


_SPECS: dict[str, TransformSpec] = {}


def register(spec: TransformSpec) -> None:
    if spec.dimension not in DIMENSIONS:
        raise ValueError(f"{spec.key}: dimension must be one of {', '.join(DIMENSIONS)}")
    if spec.source == "builtin" and not _BUILTIN_KEY.match(spec.key):
        raise ValueError(f"{spec.key}: a built-in key is lower-case letters, digits and underscores")
    if spec.key in _SPECS:
        raise ValueError(f"{spec.key} is registered twice")
    names = [s.name for s in spec.settings]
    if len(set(names)) != len(names):
        raise ValueError(f"{spec.key}: a setting name is used twice")
    _SPECS[spec.key] = spec


def get(key: Any) -> TransformSpec | None:
    return _SPECS.get(key) if isinstance(key, str) else None


def specs(dimension: str | None = None) -> list[TransformSpec]:
    return [s for s in _SPECS.values() if s.available and dimension in (None, s.dimension)]


def params_for(spec: TransformSpec, params: dict[str, Any] | None) -> dict[str, Any]:
    """The spec's defaults, overridden by the stored values it knows. Other keys can be left over from a previous
    choice, as for encodings."""
    known = {s.name for s in spec.settings}
    return {**spec.defaults(), **{k: v for k, v in (params or {}).items() if k in known}}


def param_problems(spec: TransformSpec, params: dict[str, Any] | None) -> list[str]:
    by_name = {s.name: s for s in spec.settings}
    out = []
    for name, v in (params or {}).items():
        if name in by_name and v is not None and (problem := value_problem(by_name[name], v)):
            out.append(f"params.{problem}")
    return out
