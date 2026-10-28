"""External data sources a dataset can be built from: the `Source` protocol, the shapes it exchanges, and the named
connections configured in RAINFOREST_MONGO_SOURCES."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import quote

from ..config import get_settings

NAME_RE = re.compile(r"[a-z0-9-]+")
PREVIEW_SAMPLE = 200  # documents per collection
ColumnKind = Literal["number", "text", "datetime", "boolean", "json", "identifier", "mixed"]


class SourceError(Exception):
    """A failure talking to a source. The message is already redacted: it names the connection, never the URI."""


class BudgetExceeded(SourceError):
    def __init__(self, estimated: int, max_cells: int):
        super().__init__(
            f"The pull would read about {estimated:,} cells, more than the limit of {max_cells:,} (SOURCE_MAX_CELLS)"
        )
        self.estimated = estimated
        self.max = max_cells


class PullTimeout(SourceError):
    def __init__(self, timeout_s: int):
        super().__init__(f"The pull took longer than {timeout_s:,} seconds (SOURCE_PULL_TIMEOUT_S)")


@dataclass
class CollectionSpec:
    name: str
    label: str | None = None
    filter: dict | None = None
    projection: dict | None = None


@dataclass
class ColumnInfo:
    name: str
    kind: ColumnKind
    present_share: float


@dataclass
class PreviewResult:
    columns: list[ColumnInfo]
    per_source: list[dict] = field(default_factory=list)
    estimated_rows: int = 0
    estimated_cells: int = 0
    compared_on: list[str] = field(default_factory=list)  # `compared_columns` of the sample, by its kinds


@dataclass
class PullResult:
    per_source: list[dict]  # {label, documents} per collection, in the order pulled
    flatten_notes: dict  # arrays, mixed (columns), renamed ({original key path: column}), key_paths ([path, column])
    identifiers: list[str]  # columns of kind `identifier` by the preview's rule, over every pulled value


# Called with (label, documents pulled so far) between batches.
Progress = Callable[[str, int], None] | None


def chunk_path(out_dir: Path, label: str, n: int) -> Path:
    # Labels are typed by people; quoting keeps one a single file name.
    return out_dir / f"chunk_{quote(label, safe='')}_{n}.parquet"


def chunk_paths(out_dir: Path, label: str) -> list[Path]:
    out = []
    while (p := chunk_path(out_dir, label, len(out))).exists():
        out.append(p)
    return out


def compared_columns(
    columns: list[str], per_collection: list[set[str]], is_identifier: Callable[[str], bool]
) -> list[str]:
    """The columns deduplication compares rows on unless `match_on` names them: of `columns`, those every collection
    that returned documents has (`per_collection`), less `_id` and identifiers, since each collection's own keys
    would make every row distinct."""
    shared = set.intersection(*per_collection) if per_collection else set()
    return [c for c in columns if c in shared and c != "_id" and not is_identifier(c)]


def as_text(value: Any) -> str | None:
    """How a value of a mixed-type column reads as text. Chunks written before a column turned out mixed are
    converted with this too, from what Parquet gives back (floats for every number, UTC datetimes), so one value reads
    the same whichever chunk it landed in."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return str(int(value)) if math.isfinite(value) and value.is_integer() and abs(value) < 2**53 else repr(value)
    if isinstance(value, datetime):
        # The driver's datetimes are naive UTC; Parquet gives them back with the zone.
        return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).isoformat()
    return str(value)


class Source(Protocol):
    name: str
    kind: str

    def list_databases(self) -> list[str] | None:
        """Database names, or None when the account isn't allowed to list them."""

    def list_collections(self, database: str) -> list[dict]:
        """`{name, documents}` per collection."""

    def missing_collections(self, database: str, names: list[str]) -> list[str]:
        """Those of `names` the database has no collection (or view) of, in order."""

    def preview(
        self, database: str, collections: list[CollectionSpec], sample: int = PREVIEW_SAMPLE
    ) -> PreviewResult: ...

    def pull(
        self,
        database: str,
        collections: list[CollectionSpec],
        out_dir: Path,
        budget_cells: int,
        timeout_s: int,
        progress: Progress,
        names: list | None = None,
    ) -> PullResult:
        """`names`: the previous pull's `flatten_notes.key_paths`, whose column names this pull keeps."""


def parse_sources(text: str) -> dict[str, str]:
    """`name=uri;name=uri` as a whole: a ValueError names the first bad entry by its name (when that is a valid
    slug) or its position, and never carries any part of a connection string."""
    out: dict[str, str] = {}
    for pos, raw in enumerate(text.split(";"), 1):
        raw = raw.strip()
        if not raw:
            continue
        name, eq, uri = raw.partition("=")  # options in the URI hold "=" too, so only the first one splits
        name, uri = name.strip(), uri.strip()
        if not eq or not NAME_RE.fullmatch(name):
            raise ValueError(f"entry {pos} isn't name=uri with a name of lowercase letters, digits and dashes")
        if not uri:
            raise ValueError(f"{name!r}: the connection string is empty")
        if not re.match(r"mongodb(\+srv)?://[^\s/?@]", uri):
            raise ValueError(f"{name!r}: the connection string must use the mongodb or mongodb+srv scheme")
        if not uri.startswith("mongodb+srv://"):  # parsing an srv URI would query DNS
            from pymongo import uri_parser
            from pymongo.errors import ConfigurationError

            try:
                uri_parser.parse_uri(uri)
            except (ValueError, ConfigurationError):
                raise ValueError(f"{name!r}: not a valid MongoDB connection string") from None
        if name in out:
            raise ValueError(f"the name {name!r} appears twice")
        out[name] = uri
    return out


@lru_cache
def registry() -> dict[str, Source]:
    """The configured connections by name, built once per process. Building one opens no connection."""
    from .mongodb import MongoSource

    return {n: MongoSource(n, uri) for n, uri in parse_sources(get_settings().mongo_sources).items()}


def get(name: str) -> Source:
    return registry()[name]
