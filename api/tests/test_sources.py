"""Named MongoDB connections, the flattening rules, and credential redaction. No test needs a server."""

import json
import uuid
from datetime import datetime
from decimal import Decimal

import pytest
from bson.binary import UUID_SUBTYPE, Binary
from bson.decimal128 import Decimal128
from bson.int64 import Int64
from bson.objectid import ObjectId
from pymongo.errors import ExecutionTimeout, OperationFailure, ServerSelectionTimeoutError

from app import db
from app.config import get_settings
from app.sources import CollectionSpec, SourceError, get, parse_sources, registry
from app.sources.mongodb import MongoSource, flatten_document, redact


def test_parse_sources_names_and_uris():
    assert parse_sources("vehicles=mongodb://u:p@h/db;archive=mongodb+srv://x@y/") == {
        "vehicles": "mongodb://u:p@h/db",
        "archive": "mongodb+srv://x@y/",
    }


def test_parse_sources_empty_and_options_with_equals():
    assert parse_sources("") == {}
    assert parse_sources(" ; ") == {}
    assert parse_sources("a=mongodb://h/?authSource=admin&tls=true") == {"a": "mongodb://h/?authSource=admin&tls=true"}


@pytest.mark.parametrize(
    "text,needle",
    [
        ("vehicles=", "vehicles"),
        ("Bad Name=mongodb://h/", "entry 1"),
        ("a=mongodb://h/;a=mongodb://h/", "a"),
        ("ok=mongodb://h/;mongodb://u:pw@h/?x=1", "entry 2"),
        ("v=http://u:pw@h/", "v"),
        ("v=mongodb://u:pw@h:notaport/", "v"),
    ],
)
def test_parse_sources_rejects_without_echoing_the_uri(text, needle):
    with pytest.raises(ValueError) as e:
        parse_sources(text)
    msg = str(e.value)
    assert needle in msg
    assert "mongodb://" not in msg and "pw" not in msg and "http://" not in msg


def test_flatten_rules():
    doc = {
        "_id": ObjectId("65f0a1b2c3d4e5f6a7b8c9d0"),
        "a": {"b": 1, "c.d": "x"},
        "arr": [1, {"k": 2}],
        "dec": Decimal128("1.5"),
        "when": datetime(2026, 1, 2),
        "n": None,
        "u": uuid.UUID(int=1),
    }
    assert flatten_document(doc) == {
        "_id": "65f0a1b2c3d4e5f6a7b8c9d0",
        "a.b": 1,
        "a.c_d": "x",
        "arr": '[1, {"k": 2}]',
        "dec": 1.5,
        "when": datetime(2026, 1, 2),
        "n": None,
        "u": "00000000-0000-0000-0000-000000000001",
    }


def test_flatten_numbers_binary_and_depth():
    out = flatten_document(
        {
            "i": Int64(7),
            "f": 2.5,
            "flag": True,
            "uuid4": Binary(uuid.UUID(int=2).bytes, UUID_SUBTYPE),
            "raw": Binary(b"\x00\x01"),
            "deep": {"a": {"b": {"c": {"d": 1}}}},
            "dots": {"x.y": {"z": 1}},
        }
    )
    assert out["i"] == 7 and isinstance(out["i"], int)
    assert out["f"] == 2.5 and out["flag"] is True
    assert out["uuid4"] == "00000000-0000-0000-0000-000000000002"
    assert out["raw"] == "AAE="
    assert out["deep.a.b.c.d"] == 1
    assert out["dots.x_y.z"] == 1


def test_flatten_array_elements_become_json_safe():
    oid = ObjectId("65f0a1b2c3d4e5f6a7b8c9d0")
    out = flatten_document(
        {"tags": [oid, datetime(2026, 1, 2, 3, 4), Decimal128("2.25"), {"u": uuid.UUID(int=1), "x": [Int64(1)]}, None]}
    )
    assert json.loads(out["tags"]) == [
        "65f0a1b2c3d4e5f6a7b8c9d0",
        "2026-01-02T03:04:00",
        2.25,
        {"u": "00000000-0000-0000-0000-000000000001", "x": [1]},
        None,
    ]
    assert flatten_document({"empty": []}) == {"empty": "[]"}


def test_flatten_key_collisions_are_suffixed_and_recorded():
    notes: dict = {}
    out = flatten_document({"a": {"c.d": 1, "c_d": 2, "c@d": 3}, "x.y": 4, "x_y": 5}, notes)
    assert out == {"a.c_d_2": 1, "a.c_d": 2, "a.c@d": 3, "x_y_2": 4, "x_y": 5}
    assert notes["collisions"] == ["a.c_d_2", "x_y_2"]


def test_flatten_collision_precedence_does_not_depend_on_key_order():
    dotted_first = flatten_document({"c.d": 1, "c_d": 2})
    literal_first = flatten_document({"c_d": 2, "c.d": 1})
    assert dotted_first == literal_first == {"c_d": 2, "c_d_2": 1}
    # A renamed key never takes a name a literal key later claims.
    assert flatten_document({"c.d": 1, "c_d": 2, "c_d_2": 3}) == {"c_d_3": 1, "c_d": 2, "c_d_2": 3}


def test_flatten_an_empty_key_gets_a_fixed_name():
    assert flatten_document({"": 1, "a": {"": 2}}) == {"_empty": 1, "a._empty": 2}
    # A literal key of that name keeps it, as over a dotted key's rename.
    assert flatten_document({"": 1, "_empty": 2}) == {"_empty": 2, "_empty_2": 1}
    paths: dict = {}
    flatten_document({"a": {"": 2}}, None, paths)
    assert paths == {"a._empty": ("a", "")}


def test_flatten_paths_name_the_original_keys():
    paths: dict = {}
    flatten_document({"a": {"c.d": 1, "c_d": 2}, "t": [1], "e": {}}, None, paths)
    assert paths == {"a.c_d_2": ("a", "c.d"), "a.c_d": ("a", "c_d"), "t": ("t",)}


def test_flatten_signalling_nan_and_non_finite_arrays():
    snan = Decimal128("sNaN")
    out = flatten_document(
        {"d": snan, "arr": [float("nan"), float("inf"), -float("inf"), snan, Decimal128("NaN"), 1.5]}
    )
    assert out["d"] != out["d"]
    assert out["arr"] == "[null, null, null, null, null, 1.5]"
    assert json.loads(out["arr"]) == [None, None, None, None, None, 1.5]


def test_flatten_notes_flag_arrays_and_identifiers():
    notes: dict = {}
    flatten_document(
        {
            "_id": ObjectId(),
            "arr": [1],
            "u": uuid.UUID(int=1),
            "b": Binary(b"x"),
            "s": "plain",
            "inner": {"o": ObjectId()},
        },
        notes,
    )
    assert notes["arrays"] == {"arr"}
    assert notes["ids"] == {"_id", "u", "b", "inner.o"}


def test_flatten_without_notes_still_works():
    assert flatten_document({"a": 1}) == {"a": 1}


def test_decimal128_nan_is_a_float():
    out = flatten_document({"d": Decimal128(Decimal("NaN"))})
    assert out["d"] != out["d"]


def test_redact_hides_credentials_and_hosts():
    out = redact("mongodb://user:secret@db1.example.com:27017/x timed out", "vehicles")
    assert "secret" not in out and "user" not in out
    assert out.startswith("vehicles: ")
    assert redact("...", "vehicles").startswith("vehicles: ")


def test_redact_srv_and_every_uri_in_the_message():
    out = redact("bad mongodb+srv://u:p%40ss@c.example.net/?a=b and again mongodb://x:y@h/", "archive")
    assert "p%40ss" not in out and "u:" not in out and "x:y" not in out
    assert out.count("mongodb://***@") == 2


def test_redact_replaces_the_connections_own_hosts():
    src = MongoSource("vehicles", "mongodb://u:p@db1.example.com:27017,db2.example.com/x?replicaSet=rs0")
    msg = "db1.example.com:27017: [Errno 61] Connection refused; db2.example.com:27017 timed out; DB1.EXAMPLE.COM"
    out = src.redact(msg)
    assert "example.com" not in out
    assert out.startswith("vehicles: ") and "<host>" in out
    assert "Connection refused" in out


def test_source_hosts_for_srv_need_no_dns():
    src = MongoSource("archive", "mongodb+srv://u:p@cluster0.abc.mongodb.net/?authSource=admin")
    assert "cluster0.abc.mongodb.net" not in src.redact("cluster0.abc.mongodb.net: not reachable")


class _Cursor:
    def __init__(self, docs):
        self.docs = docs

    def limit(self, n):
        self.docs = self.docs[:n]
        return self

    def __iter__(self):
        return iter(self.docs)


class _FakeCollection:
    def __init__(self, docs, view=False, count_times_out=False):
        self.docs = docs
        self.view = view
        self.count_times_out = count_times_out
        self.last_find = None

    def find(self, flt=None, projection=None, **kw):
        self.last_find = (flt, projection)
        return _Cursor(self.docs)

    def estimated_document_count(self):
        if self.view:
            raise OperationFailure("Namespace is a view, not a collection", code=166)
        return len(self.docs)

    def count_documents(self, flt, **kw):
        if self.count_times_out and flt:
            raise ExecutionTimeout("operation exceeded time limit", code=50)
        return len(self.docs)


class _FakeDb:
    def __init__(self, colls):
        self.colls = colls

    def list_collection_names(self, **kw):
        return [*self.colls, "system.views"]

    def __getitem__(self, name):
        return self.colls[name]


class _FakeClient:
    def __init__(self, dbs=None, list_error=None):
        self.dbs, self.list_error = dbs or {}, list_error

    def list_database_names(self):
        if self.list_error:
            raise self.list_error
        return ["admin", "config", "local", *self.dbs]

    def __getitem__(self, name):
        return self.dbs[name]


def _source(client):
    src = MongoSource("vehicles", "mongodb://u:pw@db1.example.com:27017/")
    src._client = client
    return src


def test_list_databases_and_collections():
    src = _source(_FakeClient({"cars": _FakeDb({"models": _FakeCollection([{"a": 1}, {"a": 2}])})}))
    assert src.list_databases() == ["cars"]
    assert src.list_collections("cars") == [{"name": "models", "documents": 2}]


def test_list_databases_none_when_the_account_cannot_list():
    assert _source(_FakeClient(list_error=OperationFailure("not authorized", code=13))).list_databases() is None


def test_list_databases_bad_credentials_is_an_error_not_none():
    err = OperationFailure("Authentication failed for u:pw@db1.example.com", code=18)
    with pytest.raises(SourceError) as e:
        _source(_FakeClient(list_error=err)).list_databases()
    assert str(e.value).startswith("vehicles: ") and "example.com" not in str(e.value)


def test_connection_errors_are_redacted_and_named():
    err = ServerSelectionTimeoutError(
        "db1.example.com:27017: timed out, Topology: mongodb://u:pw@db1.example.com:27017"
    )
    with pytest.raises(SourceError) as e:
        _source(_FakeClient(list_error=err)).list_databases()
    msg = str(e.value)
    assert msg.startswith("vehicles: ") and "pw" not in msg and "example.com" not in msg


def test_preview_columns_kinds_and_estimates():
    docs = [
        {"_id": ObjectId(), "age": 41, "name": "a", "tags": [1], "when": datetime(2026, 1, 1), "mix": 1},
        {"_id": ObjectId(), "age": 3.5, "name": "b", "tags": [], "when": datetime(2026, 1, 2), "mix": "x"},
        {"_id": ObjectId(), "age": 7, "extra": {"k": 1}, "mix": None},
        {"_id": ObjectId(), "age": 8},
    ]
    coll = _FakeCollection(docs)
    src = _source(_FakeClient({"cars": _FakeDb({"models": coll})}))
    res = src.preview("cars", [CollectionSpec("models", "models", {"age": {"$gt": 1}}, {"name": 1})], sample=3)
    cols = {c.name: c for c in res.columns}
    assert coll.last_find == ({"age": {"$gt": 1}}, {"name": 1})
    assert [c.name for c in res.columns] == ["_id", "age", "name", "tags", "when", "mix", "extra.k"]
    assert cols["_id"].kind == "identifier" and cols["_id"].present_share == 1.0
    assert cols["age"].kind == "number"
    assert cols["name"].kind == "text" and cols["name"].present_share == pytest.approx(2 / 3)
    assert cols["tags"].kind == "json" and cols["when"].kind == "datetime"
    assert cols["mix"].kind == "mixed" and cols["mix"].present_share == pytest.approx(2 / 3)
    assert res.estimated_rows == 4 and res.estimated_cells == 4 * 7
    assert res.per_source[0]["name"] == "models" and res.per_source[0]["documents"] == 4


def test_preview_counts_a_view_and_survives_a_count_timeout():
    docs = [{"a": 1}, {"a": 2}]
    db = _FakeDb({"v": _FakeCollection(docs, view=True), "slow": _FakeCollection(docs, count_times_out=True)})
    src = _source(_FakeClient({"cars": db}))
    assert src.list_collections("cars") == [{"name": "slow", "documents": 2}, {"name": "v", "documents": 2}]
    res = src.preview("cars", [CollectionSpec("v"), CollectionSpec("slow", filter={"a": 1})])
    assert [s["documents"] for s in res.per_source] == [2, 2]
    assert res.estimated_rows == 4


def test_preview_share_counts_collections_without_the_column():
    a = _FakeCollection([{"x": 1, "only_a": 1}] * 4)
    b = _FakeCollection([{"x": 1}] * 4)
    src = _source(_FakeClient({"d": _FakeDb({"a": a, "b": b})}))
    res = src.preview("d", [CollectionSpec("a"), CollectionSpec("b")])
    shares = {c.name: c.present_share for c in res.columns}
    assert shares == {"x": 1.0, "only_a": 0.5}


def test_preview_compares_on_columns_every_collection_has_less_identifiers():
    a = _FakeCollection([{"_id": ObjectId(), "code_id": f"k{i}", "x": i % 3, "y": "s", "only_a": 1} for i in range(50)])
    b = _FakeCollection([{"_id": ObjectId(), "code_id": "z", "x": 0, "y": "s"}])
    src = _source(_FakeClient({"d": _FakeDb({"a": a, "b": b})}))
    assert src.preview("d", [CollectionSpec("a"), CollectionSpec("b")]).compared_on == ["x", "y"]


def test_preview_booleans_are_their_own_kind_never_identifiers():
    coll = _FakeCollection([{"is_id": True, "active": False, "key": 7}, {"is_id": False, "active": True, "key": 8}])
    src = _source(_FakeClient({"d": _FakeDb({"c": coll})}))
    kinds = {c.name: c.kind for c in src.preview("d", [CollectionSpec("c")]).columns}
    assert kinds == {"is_id": "boolean", "active": "boolean", "key": "identifier"}


def test_registry_is_lazy_and_cached(monkeypatch):
    monkeypatch.setenv("RAINFOREST_MONGO_SOURCES", "vehicles=mongodb://u:p@127.0.0.1:1/;archive=mongodb://127.0.0.1:1/")
    get_settings.cache_clear()
    registry.cache_clear()
    try:
        reg = registry()
        assert list(reg) == ["vehicles", "archive"]
        assert reg["vehicles"].name == "vehicles" and reg["vehicles"].kind == "mongodb"
        assert registry() is reg and get("archive") is reg["archive"]
        assert reg["vehicles"]._client is None and reg["archive"]._client is None
        with pytest.raises(KeyError):
            get("nope")
    finally:
        monkeypatch.delenv("RAINFOREST_MONGO_SOURCES")
        get_settings.cache_clear()
        registry.cache_clear()


def test_registry_empty_without_the_setting(monkeypatch):
    monkeypatch.delenv("RAINFOREST_MONGO_SOURCES", raising=False)
    get_settings.cache_clear()
    registry.cache_clear()
    try:
        assert registry() == {}
    finally:
        get_settings.cache_clear()
        registry.cache_clear()


def test_a_malformed_setting_stops_startup_before_any_database_work(monkeypatch):
    monkeypatch.setenv("RAINFOREST_MONGO_SOURCES", "Bad Name=mongodb://u:topsecret@h/")
    get_settings.cache_clear()
    try:
        with pytest.raises(SystemExit) as e:
            db.bootstrap()
        assert "RAINFOREST_MONGO_SOURCES: " in str(e.value) and "entry 1" in str(e.value)
        assert "topsecret" not in str(e.value)
    finally:
        monkeypatch.delenv("RAINFOREST_MONGO_SOURCES")
        get_settings.cache_clear()


def test_settings_defaults():
    s = get_settings()
    assert s.mongo_sources == "" and s.source_max_cells == 600_000_000 and s.source_pull_timeout_s == 3600
