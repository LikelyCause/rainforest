"""External data sources a dataset can be built from: the `Source` protocol, the shapes it exchanges, and the named
connections configured in RAINFOREST_MONGO_SOURCES."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Protocol

from ..config import get_settings

NAME_RE = re.compile(r"[a-z0-9-]+")
ColumnKind = Literal["number", "text", "datetime", "boolean", "json", "identifier", "mixed"]


class SourceError(Exception):
    """A failure talking to a source. The message is already redacted: it names the connection, never the URI."""


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


class Source(Protocol):
    name: str
    kind: str

    def list_databases(self) -> list[str] | None:
        """Database names, or None when the account isn't allowed to list them."""

    def list_collections(self, database: str) -> list[dict]:
        """`{name, documents}` per collection."""

    def preview(self, database: str, collections: list[CollectionSpec], sample: int = 200) -> PreviewResult: ...

    def pull(
        self,
        database: str,
        collections: list[CollectionSpec],
        out_dir: Path,
        budget_cells: int,
        timeout_s: int,
        progress: Any,
    ) -> Any: ...


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
