"""MongoDB connections: listing, preview, and the rules that turn a document into a flat row."""

from __future__ import annotations

import base64
import json
import re
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from bson.binary import UUID_SUBTYPE, Binary
from bson.decimal128 import Decimal128
from bson.objectid import ObjectId
from pymongo import MongoClient, uri_parser
from pymongo.errors import ConfigurationError, OperationFailure, PyMongoError

from ..profiling import ID_NAME_RE
from . import CollectionSpec, ColumnInfo, PreviewResult, SourceError

SYSTEM_DATABASES = {"admin", "config", "local"}
CONNECT_TIMEOUT_MS = 10_000
QUERY_TIMEOUT_MS = 30_000
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


def _jsonable(value: Any) -> Any:
    """An array's elements with every BSON type replaced by what json.dumps can write."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal128):
        return float(value.to_decimal())
    if isinstance(value, bool) or value is None or isinstance(value, (float, str)):
        return value
    if isinstance(value, int):
        return int(value)
    return _text(value) or str(value)


def flatten_document(doc: Mapping, notes: dict | None = None) -> dict[str, Any]:
    """One document as a flat row, by the table in docs/data-preparation.md. Nested documents become dotted paths,
    with dots inside a key replaced by `_` first (as `mongoexport --type=csv` names them); a key that then collides
    with an earlier sibling gets a `_2` (`_3`, …) suffix. Arrays become a JSON string.

    `notes`, when given, is filled for the caller: `arrays` (columns that hold an array), `ids` (columns that held an
    ObjectId, UUID or binary value) and `collisions` (the renamed columns, in order). Missing keys are simply absent."""
    out: dict[str, Any] = {}

    def note(kind: str, path: str) -> None:
        if notes is None:
            return
        if kind == "collisions":
            notes.setdefault(kind, []).append(path)
        else:
            notes.setdefault(kind, set()).add(path)

    def walk(prefix: str, mapping: Mapping) -> None:
        used: set[str] = set()
        for key, value in mapping.items():
            seg = base = str(key).replace(".", "_")
            k = 2
            while seg in used:
                seg = f"{base}_{k}"
                k += 1
            used.add(seg)
            path = f"{prefix}.{seg}" if prefix else seg
            if seg != base:
                note("collisions", path)
            if isinstance(value, Mapping):
                walk(path, value)
            elif isinstance(value, (list, tuple)):
                out[path] = json.dumps(_jsonable(value), default=str)
                note("arrays", path)
            elif value is None:
                out[path] = None
            elif isinstance(value, Decimal128):
                out[path] = float(value.to_decimal())
            elif isinstance(value, (bool, float, str, datetime)):
                out[path] = value
            elif isinstance(value, int):
                out[path] = int(value)  # Int64 is an int subclass
            else:
                text = _text(value)
                if text is not None:
                    note("ids", path)
                out[path] = text if text is not None else str(value)

    walk("", doc)
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
        except OperationFailure:
            return None
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
                try:
                    n = db[name].estimated_document_count()
                except OperationFailure:  # a view has no metadata count
                    n = db[name].count_documents({}, maxTimeMS=QUERY_TIMEOUT_MS)
                out.append({"name": name, "documents": n})
        return out

    def preview(self, database: str, collections: list[CollectionSpec], sample: int = 200) -> PreviewResult:
        columns: dict[str, ColumnInfo] = {}
        per_source: list[dict] = []
        total_rows = total_cells = 0
        with self._guard():
            db = self.client[database]
            for spec in collections:
                coll = db[spec.name]
                names, categories, seen = self._sample(coll, spec, sample)
                documents = self._count(coll, spec)
                total_rows += documents
                total_cells += documents * len(names)
                per_source.append(
                    {
                        "name": spec.name,
                        "label": spec.label or spec.name,
                        "documents": documents,
                        "sampled": seen,
                        "columns": len(names),
                    }
                )
                for col, present in names.items():
                    prior = columns.get(col)
                    kinds = categories[col]
                    info = _column(col, kinds, present / seen if seen else 0.0)
                    if prior is None:
                        columns[col] = info
                    else:
                        # The same column from another collection: keep a mismatch visible, average the share.
                        kind = prior.kind if prior.kind == info.kind else "mixed"
                        columns[col] = ColumnInfo(col, kind, (prior.present_share + info.present_share) / 2)
        return PreviewResult(list(columns.values()), per_source, total_rows, total_cells)

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
            return coll.estimated_document_count()
        try:
            return coll.count_documents(spec.filter, maxTimeMS=QUERY_TIMEOUT_MS)
        except OperationFailure:
            raise
        except PyMongoError:
            return coll.estimated_document_count()  # an upper bound is enough for the budget

    def pull(
        self,
        database: str,
        collections: list[CollectionSpec],
        out_dir: Path,
        budget_cells: int,
        timeout_s: int,
        progress: Any,
    ) -> Any:
        raise NotImplementedError


def _category(col: str, value: Any, notes: dict) -> str:
    if col in notes.get("arrays", ()):
        return "json"
    if col in notes.get("ids", ()):
        return "identifier"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, datetime):
        return "datetime"
    return "text"


def _column(name: str, kinds: set[str], share: float) -> ColumnInfo:
    if len(kinds) > 1:
        kind = "mixed"
    elif not kinds:
        kind = "text"
    else:
        (kind,) = kinds
        if kind == "bool":
            kind = "text"
        if kind in ("text", "number") and ID_NAME_RE.search(name):
            kind = "identifier"
    return ColumnInfo(name, kind, share)  # type: ignore[arg-type]
