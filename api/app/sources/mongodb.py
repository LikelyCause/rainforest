"""MongoDB connections: listing, preview, and the rules that turn a document into a flat row."""

from __future__ import annotations

import base64
import json
import math
import re
import time
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import polars as pl
from bson.binary import UUID_SUBTYPE, Binary
from bson.decimal128 import Decimal128
from bson.objectid import ObjectId
from pymongo import MongoClient, uri_parser
from pymongo.errors import ConfigurationError, ExecutionTimeout, OperationFailure, PyMongoError

from ..profiling import ID_NAME_RE
from . import (
    PREVIEW_SAMPLE,
    BudgetExceeded,
    CollectionSpec,
    ColumnInfo,
    ColumnKind,
    PreviewResult,
    Progress,
    PullResult,
    PullTimeout,
    SourceError,
    as_text,
    chunk_path,
    compared_columns,
)

SYSTEM_DATABASES = {"admin", "config", "local"}
CONNECT_TIMEOUT_MS = 10_000
QUERY_TIMEOUT_MS = 30_000
UNAUTHORIZED = 13  # AuthenticationFailed (18) and the rest are real errors, not "this account can't list"
PULL_BATCH = 5_000
CHUNK_DOCS = 50_000  # flat rows per chunk file; bounds the pull's memory
CREDENTIALS_RE = re.compile(r"mongodb(\+srv)?://[^@\s]+@", re.IGNORECASE)


def _text(value: Any) -> str | None:
    """The string a BSON identifier or binary value becomes."""
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, Binary):
        if value.subtype == UUID_SUBTYPE and len(value) == 16:
            return str(uuid.UUID(bytes=bytes(value)))
        return base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    return None


def _decimal(value: Decimal128) -> float:
    d = value.to_decimal()
    return float("nan") if d.is_snan() else float(d)  # float() refuses a signalling NaN


def _jsonable(value: Any) -> Any:
    """An array's elements with every BSON type replaced by what json.dumps can write."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal128):
        value = _decimal(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None  # NaN and inf aren't JSON
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    return _text(value) or str(value)


EMPTY_KEY = "_empty"


def _renamed(key: str) -> bool:
    return "." in key or key == ""


def _segment(key: str) -> str:
    """A key as one segment of a column name. An empty key gets a fixed name: Polars would name an empty column
    `column_<n>` by its position, which differs from chunk to chunk."""
    return key.replace(".", "_") if key else EMPTY_KEY


def flatten_document(doc: Mapping, notes: dict | None = None, paths: dict | None = None) -> dict[str, Any]:
    """One document as a flat row, by the table in docs/data-preparation.md. Nested documents become dotted paths,
    with dots inside a key replaced by `_` first (as `mongoexport --type=csv` names them) and an empty key named
    `_empty`; a key that then collides with a sibling gets a `_2` (`_3`, …) suffix. A key that wasn't renamed keeps
    its name over one that was, whichever comes first. Arrays become a JSON string.

    `notes`, when given, is filled for the caller: `arrays` (columns that hold an array), `ids` (columns that held an
    ObjectId, UUID or binary value) and `collisions` (the renamed columns, in order). `paths`, when given, maps each
    column to the tuple of original keys it came from. Missing keys are simply absent."""
    out: dict[str, Any] = {}

    def note(kind: str, path: str) -> None:
        if notes is None:
            return
        if kind == "collisions":
            notes.setdefault(kind, []).append(path)
        else:
            notes.setdefault(kind, set()).add(path)

    def segments(mapping: Mapping) -> dict[Any, str]:
        names = {key: str(key) for key in mapping}
        used = {seg for seg in names.values() if not _renamed(seg)}
        for key, seg in names.items():
            if not _renamed(seg):
                continue
            seg = base = _segment(seg)
            k = 2
            while seg in used:
                seg = f"{base}_{k}"
                k += 1
            used.add(seg)
            names[key] = seg
        return names

    def walk(prefix: str, keys: tuple[str, ...], mapping: Mapping) -> None:
        segs = segments(mapping)
        for key, value in mapping.items():
            seg = segs[key]
            path = f"{prefix}.{seg}" if prefix else seg
            if _renamed(str(key)) and seg != _segment(str(key)):
                note("collisions", path)
            if isinstance(value, Mapping):
                walk(path, (*keys, str(key)), value)
                continue
            if paths is not None:
                paths[path] = (*keys, str(key))
            if isinstance(value, (list, tuple)):
                out[path] = json.dumps(_jsonable(value), default=str)
                note("arrays", path)
            elif value is None:
                out[path] = None
            elif isinstance(value, Decimal128):
                out[path] = _decimal(value)
            elif isinstance(value, (bool, float, str, datetime)):
                out[path] = value
            elif isinstance(value, int):
                out[path] = int(value)  # Int64 is an int subclass
            else:
                text = _text(value)
                if text is not None:
                    note("ids", path)
                out[path] = text if text is not None else str(value)

    walk("", (), doc)
    return out


def _host_tokens(uri: str) -> list[str]:
    """The host:port and bare host names a connection string points at, longest first."""
    m = re.match(r"mongodb(\+srv)?://(?:.*@)?([^/?]+)", uri, re.IGNORECASE)
    if not m:
        return []
    tokens: set[str] = set()
    try:
        hosts = uri_parser.split_hosts(m.group(2))
    except (ValueError, ConfigurationError):
        hosts = [(h, None) for h in m.group(2).split(",")]
    for host, port in hosts:
        tokens.add(host)
        if port:
            tokens.add(f"{host}:{port}")
        if m.group(1) and host.count(".") >= 2:
            # An srv name resolves to member hosts under its parent domain (cluster0-shard-00-00.abc.mongodb.net).
            tokens.add("*." + host.split(".", 1)[1])
    return sorted(tokens, key=len, reverse=True)


def redact(message: str, name: str, hosts: tuple[str, ...] | list[str] = ()) -> str:
    """`message` with connection-string credentials replaced by `mongodb://***@`, every token in `hosts` (from
    `_host_tokens`; a `*.domain` token covers the subdomains) replaced by `<host>`, and the connection's name in
    front. Pure: MongoSource.redact supplies its own hosts."""
    out = CREDENTIALS_RE.sub("mongodb://***@", message)
    for token in hosts:
        if token.startswith("*."):
            pat = r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*\." + re.escape(token[2:]) + r"(?::\d+)?"
        else:
            pat = re.escape(token)
        out = re.sub(pat, "<host>", out, flags=re.IGNORECASE)
    return f"{name}: {out}"


class MongoSource:
    kind = "mongodb"

    def __init__(self, name: str, uri: str):
        self.name = name
        self._uri = uri
        self._hosts = _host_tokens(uri)
        self._client: MongoClient | None = None

    @property
    def client(self) -> MongoClient:
        # connect=False: the driver starts no monitor threads until the first operation.
        if self._client is None:
            self._client = MongoClient(
                self._uri,
                serverSelectionTimeoutMS=CONNECT_TIMEOUT_MS,
                readPreference="secondaryPreferred",
                connect=False,
            )
        return self._client

    def redact(self, message: str) -> str:
        return redact(message, self.name, self._hosts)

    @contextmanager
    def _guard(self) -> Iterator[None]:
        try:
            yield
        except PyMongoError as exc:
            raise SourceError(self.redact(str(exc))) from None

    def list_databases(self) -> list[str] | None:
        try:
            names = self.client.list_database_names()
        except OperationFailure as exc:
            if exc.code == UNAUTHORIZED:
                return None
            raise SourceError(self.redact(str(exc))) from None
        except PyMongoError as exc:
            raise SourceError(self.redact(str(exc))) from None
        return sorted(set(names) - SYSTEM_DATABASES)

    def list_collections(self, database: str) -> list[dict]:
        out = []
        with self._guard():
            db = self.client[database]
            for name in sorted(db.list_collection_names()):
                if name.startswith("system."):
                    continue
                out.append({"name": name, "documents": _estimated(db[name])})
        return out

    def missing_collections(self, database: str, names: list[str]) -> list[str]:
        with self._guard():
            # authorizedCollections: an account allowed `find` on some collections only may still list those.
            found = set(
                self.client[database].list_collection_names(filter={"name": {"$in": names}}, authorizedCollections=True)
            )
        return [n for n in names if n not in found]

    def preview(self, database: str, collections: list[CollectionSpec], sample: int = PREVIEW_SAMPLE) -> PreviewResult:
        present: dict[str, int] = {}
        categories: dict[str, set[str]] = {}
        sampled = 0
        per_source: list[dict] = []
        per_collection: list[set[str]] = []
        total_rows = total_cells = 0
        with self._guard():
            db = self.client[database]
            for spec in collections:
                coll = db[spec.name]
                counts, kinds, seen = self._sample(coll, spec, sample)
                documents = self._count(coll, spec)
                total_rows += documents
                total_cells += documents * len(counts)
                sampled += seen
                if seen:
                    per_collection.append(set(counts))
                per_source.append(
                    {
                        "name": spec.name,
                        "label": spec.label or spec.name,
                        "documents": documents,
                        "sampled": seen,
                        "columns": len(counts),
                    }
                )
                for col, n in counts.items():
                    present[col] = present.get(col, 0) + n
                    categories.setdefault(col, set()).update(kinds[col])
        # A collection that lacks a column still counts in the denominator.
        columns = [_column(col, categories[col], n / sampled if sampled else 0.0) for col, n in present.items()]
        kind_of = {c.name: c.kind for c in columns}
        compared = compared_columns(list(present), per_collection, lambda col: kind_of[col] == "identifier")
        return PreviewResult(columns, per_source, total_rows, total_cells, compared)

    @staticmethod
    def _sample(coll, spec: CollectionSpec, sample: int):
        names: dict[str, int] = {}
        categories: dict[str, set[str]] = {}
        seen = 0
        cursor = coll.find(spec.filter, spec.projection, batch_size=sample, max_time_ms=QUERY_TIMEOUT_MS).limit(sample)
        for doc in cursor:
            seen += 1
            notes: dict = {}
            row = flatten_document(doc, notes)
            for col, value in row.items():
                names.setdefault(col, 0)  # a column that is null everywhere still exists
                categories.setdefault(col, set())
                if value is not None:
                    names[col] += 1
                    categories[col].add(_category(col, value, notes))
        return names, categories, seen

    @staticmethod
    def _count(coll, spec: CollectionSpec) -> int:
        if not spec.filter:
            return _estimated(coll)
        try:
            return coll.count_documents(spec.filter, maxTimeMS=QUERY_TIMEOUT_MS)
        except ExecutionTimeout:
            return _estimated(coll)  # an upper bound is enough for the budget

    def pull(
        self,
        database: str,
        collections: list[CollectionSpec],
        out_dir: Path,
        budget_cells: int,
        timeout_s: int,
        progress: Progress,
    ) -> PullResult:
        """Every document of `collections` as flat rows in `out_dir/chunk_<label>_<n>.parquet`. The budget is the
        preview's estimate (documents x the columns a sample of them flattens to, the figure the dialog showed), so
        a pull over it is refused before any document is read in full."""
        deadline = time.monotonic() + timeout_s
        estimated = self.preview(database, collections).estimated_cells
        if estimated > budget_cells:
            raise BudgetExceeded(estimated, budget_cells)
        out_dir.mkdir(parents=True, exist_ok=True)
        chunks = _Chunks(out_dir)
        per_source = []
        with self._guard():
            db = self.client[database]
            for spec in collections:
                label = spec.label or spec.name
                remaining_ms = int((deadline - time.monotonic()) * 1000)
                if remaining_ms <= 0:
                    raise PullTimeout(timeout_s)
                # The server stops the query at the deadline too, so a stalled getMore can't outlive it.
                cursor = db[spec.name].find(
                    spec.filter,
                    spec.projection,
                    batch_size=PULL_BATCH,
                    no_cursor_timeout=False,
                    max_time_ms=remaining_ms,
                )
                documents = 0
                try:
                    for doc in cursor:
                        chunks.add(label, doc)
                        documents += 1
                        if documents % PULL_BATCH == 0:
                            if time.monotonic() > deadline:
                                raise PullTimeout(timeout_s)
                            if progress:
                                progress(label, documents)
                except ExecutionTimeout:
                    raise PullTimeout(timeout_s) from None
                finally:
                    cursor.close()
                chunks.flush(label)
                per_source.append({"label": label, "documents": documents})
                if progress:
                    progress(label, documents)
        return PullResult(per_source, chunks.notes(), chunks.identifiers())


_KINDS = {bool: "boolean", int: "number", float: "number", str: "text", datetime: "datetime"}


def _kind(t: type) -> str:
    kind = _KINDS.get(t)
    if kind is None:
        kind = next((k for base, k in _KINDS.items() if issubclass(t, base)), "text")
    return kind


class _Chunks:
    """Flat rows buffered and written CHUNK_DOCS at a time, with column names and types kept consistent across the
    whole pull.

    `flatten_document` names a document's columns from that document's keys alone, so two documents can give one
    key path different names (a dotted key renamed `_2` only when its literal twin is present). Here each original
    key path keeps the name it got first, and a later path that would take a name already in use gets the next free
    suffix. A column whose values span more than one kind (number, text, boolean, datetime) across the pull is
    mixed: written as text from then on, and earlier chunks are converted when they are assembled.

    Each column's `_category` set over every value is kept too, for `identifiers`. Arrays and BSON identifiers are
    written as strings, so those are noted per document; every other category follows from the value types a chunk
    already gathers."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.by_path: dict[tuple[str, ...], str] = {}
        self.taken: set[str] = set()
        self.order: dict[str, None] = {}
        self.kinds: dict[str, set[str]] = {}
        self.mixed: dict[str, None] = {}
        self.arrays: dict[str, None] = {}
        self.renamed: dict[str, str] = {}
        self.categories: dict[str, set[str]] = {}
        self.special: dict[str, int] = {}  # buffered values that were an array or a BSON identifier
        self.rows: list[dict] = []
        self.written: dict[str, int] = {}

    def _name(self, proposed: str, path: tuple[str, ...]) -> str:
        name = proposed
        if name in self.taken:
            name = base = ".".join(map(_segment, path))
            n = 2
            while name in self.taken:
                name = f"{base}_{n}"
                n += 1
        self.by_path[path] = name
        self.taken.add(name)
        self.order[name] = None
        if name != ".".join(path):
            self.renamed[".".join(path)] = name
        return name

    def add(self, label: str, doc: Mapping) -> None:
        notes: dict = {}
        paths: dict = {}
        flat = flatten_document(doc, notes, paths)
        row = {}
        for col, value in flat.items():
            path = paths[col]
            name = self.by_path.get(path)
            row[self._name(col, path) if name is None else name] = value
        for col in notes.get("arrays", ()):
            self.arrays[self._special(paths[col], "json")] = None
        for col in notes.get("ids", ()):
            self._special(paths[col], "identifier")
        self.rows.append(row)
        if len(self.rows) >= CHUNK_DOCS:
            self.flush(label)

    def _special(self, path: tuple[str, ...], category: str) -> str:
        name = self.by_path[path]
        self.categories.setdefault(name, set()).add(category)
        self.special[name] = self.special.get(name, 0) + 1
        return name

    def flush(self, label: str) -> None:
        if not self.rows:
            return
        rows, self.rows = self.rows, []
        special, self.special = self.special, {}
        present: set[str] = set()
        for row in rows:
            present.update(row)
        columns = []
        for name in (c for c in self.order if c in present):
            values = [row.get(name) for row in rows]
            found = {_kind(t) for t in set(map(type, values)) - {type(None)}}
            kinds = self.kinds.setdefault(name, set())
            kinds.update(found)
            if len(kinds) > 1:
                self.mixed[name] = None
            if special.get(name) and sum(isinstance(v, str) for v in values) == special[name]:
                found.discard("text")  # every string was an array or an identifier, categorised in `add`
            self.categories.setdefault(name, set()).update(found)
            columns.append(_series(name, values, name in self.mixed, kinds))
        n = self.written.get(label, 0)
        pl.DataFrame(columns).write_parquet(chunk_path(self.out_dir, label, n))
        self.written[label] = n + 1

    def notes(self) -> dict:
        return {"arrays": list(self.arrays), "mixed": list(self.mixed), "renamed": self.renamed}

    def identifiers(self) -> list[str]:
        return [c for c in self.order if column_kind(c, self.categories.get(c, set())) == "identifier"]


_DTYPES = {"boolean": pl.Boolean, "number": pl.Int64, "text": pl.String, "datetime": pl.Datetime("us", "UTC")}


def _series(name: str, values: list, mixed: bool, kinds: set[str]) -> pl.Series:
    if mixed:
        return pl.Series(name, [as_text(v) for v in values], dtype=pl.String)
    s = pl.Series(name, values, strict=False)
    if s.dtype == pl.Null:
        # Every value in this chunk is null: give the column the type the rest of the pull has.
        return s.cast(_DTYPES[next(iter(kinds))]) if kinds else s
    if kinds == {"datetime"}:
        # The driver returns naive datetimes in UTC.
        s = s.dt.replace_time_zone("UTC") if s.dtype.time_zone is None else s.dt.convert_time_zone("UTC")
        return s.cast(pl.Datetime("us", "UTC"))
    return s


def _estimated(coll) -> int:
    try:
        return coll.estimated_document_count()
    except OperationFailure:  # a view has no metadata count
        return coll.count_documents({}, maxTimeMS=QUERY_TIMEOUT_MS)


def _category(col: str, value: Any, notes: dict) -> str:
    if col in notes.get("arrays", ()):
        return "json"
    if col in notes.get("ids", ()):
        return "identifier"
    return _kind(type(value))


def column_kind(name: str, categories: set[str]) -> ColumnKind:
    """A column's kind from the `_category` of each of its values. The preview and the pull's deduplication both
    decide identifiers by it, so the comparison columns the dialog lists are the ones the pull compares."""
    if len(categories) > 1:
        return "mixed"
    if not categories:
        return "text"
    (kind,) = categories
    if kind in ("text", "number") and ID_NAME_RE.search(name):
        return "identifier"
    return kind  # type: ignore[return-value]


def _column(name: str, categories: set[str], share: float) -> ColumnInfo:
    return ColumnInfo(name, column_kind(name, categories), share)
