"""Bulk-edit planning, the shared field filter and request validation (no database)."""

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.routers.fields import plan_bulk, summarize_bulk
from app.routers.query import fields_where
from app.schemas import BulkDecisionIn, FieldFilter


def _where(*filters) -> str:
    return fields_where(7, [FieldFilter(**f) for f in filters]).as_string(None)


def test_where_scopes_to_dataset():
    assert _where() == "dataset_id = 7"


def test_in_is_typed_per_column_kind():
    assert _where({"column": "n_unique", "op": "in", "value": "2, 3"}).endswith(
        "\"n_unique\" = ANY('{2.0,3.0}'::float8[])"
    )
    assert _where({"column": "reviewed", "op": "in", "value": [True, "no"]}).endswith(
        "\"reviewed\" = ANY('{t,f}'::bool[])"
    )
    assert _where({"column": "action", "op": "in", "value": ["keep", "drop"]}).endswith(
        "\"action\" = ANY('{keep,drop}'::text[])"
    )


@pytest.mark.parametrize(
    "f",
    [
        {"column": "reviewed", "op": "=", "value": "maybe"},
        {"column": "null_pct", "op": "in", "value": "0.1,abc"},
        {"column": "no_such_column", "op": "=", "value": 1},
    ],
)
def test_bad_filters_are_422(f):
    with pytest.raises(HTTPException) as e:
        _where(f)
    assert e.value.status_code == 422


def test_comparison_and_null_ops():
    sql = _where({"column": "null_pct", "op": ">", "value": "0.2"}, {"column": "notes", "op": "is_null"})
    assert sql == 'dataset_id = 7 AND "null_pct" > 0.2 AND "notes" IS NULL'


@pytest.mark.parametrize(
    "body",
    [
        {"action": "drop"},
        {"action": "drop", "field_ids": [1], "filters": []},
        {"field_ids": [1]},
        {"field_ids": [], "action": "drop"},
        {"field_ids": list(range(5001)), "action": "drop"},
        {"filters": [], "action": "target"},
    ],
)
def test_invalid_bodies(body):
    with pytest.raises(ValidationError):
        BulkDecisionIn(**body)


def test_empty_filters_mean_whole_dataset():
    assert BulkDecisionIn(filters=[], reviewed=True).filters == []


def _row(fid, action="keep", stype="numeric_continuous", reviewed=False, suggestion=None):
    return {
        "field_id": fid,
        "name": f"f{fid}",
        "action": action,
        "semantic_type": stype,
        "reviewed": reviewed,
        "suggestion": suggestion or {},
    }


SETTINGS = {
    "imputation": "none",
    "impute_value": None,
    "add_missing_indicator": False,
    "transform": "none",
    "scaling": "none",
    "encoding": "none",
    "params": {},
}


def test_drop_skips_targets_and_unchanged():
    rows = [_row(1), _row(2, action="drop"), _row(3, action="target"), _row(4, reviewed=True)]
    planned = plan_bulk(rows, {}, BulkDecisionIn(field_ids=[1, 2, 3, 4], action="drop"))
    assert [c["field_id"] for c in planned] == [1, 4]
    s = summarize_bulk(len(rows), planned)
    assert (s["matched"], s["changed"], s["unchanged"], s["reviewed_changed"]) == (4, 2, 2, 1)
    assert s["transitions"] == {"keep→drop": 2}


def test_keep_never_applies_to_identifiers_but_reviewed_still_does():
    rows = [_row(1, action="drop", stype="identifier"), _row(2, action="drop")]
    planned = plan_bulk(rows, {}, BulkDecisionIn(field_ids=[1, 2], action="keep", reviewed=True))
    assert planned[0]["after"] == {"action": "drop", "semantic_type": "identifier", "reviewed": True}
    assert planned[1]["after"]["action"] == "keep"


def test_accept_suggestion_counts_settings_only_changes():
    sugg = {
        "tree": {**SETTINGS, "action": "keep", "semantic_type": "numeric_continuous"},
        "linear": {**SETTINGS, "action": "keep", "semantic_type": "numeric_continuous", "scaling": "standard"},
    }
    rows = [_row(1, reviewed=True, suggestion=sugg), _row(2, reviewed=True, suggestion=sugg)]
    settings = {
        1: {"tree": {**SETTINGS, "field_id": 1}, "linear": {**SETTINGS, "scaling": "standard"}},
        2: {"tree": {**SETTINGS}, "linear": {**SETTINGS, "scaling": "robust", "source": "suggested"}},
    }
    planned = plan_bulk(rows, settings, BulkDecisionIn(field_ids=[1, 2], accept_suggestion=True))
    assert [c["field_id"] for c in planned] == [2]
    assert list(planned[0]["settings"]) == ["linear"]
    assert summarize_bulk(2, planned)["transitions"] == {}


def test_accept_suggestion_keeps_settings_you_customised():
    sugg = {
        "tree": {**SETTINGS, "action": "keep", "semantic_type": "numeric_continuous"},
        "linear": {**SETTINGS, "action": "keep", "semantic_type": "numeric_continuous", "scaling": "standard"},
    }
    settings = {
        1: {"tree": {**SETTINGS, "source": "suggested"}, "linear": {**SETTINGS, "scaling": "robust", "source": "user"}}
    }
    planned = plan_bulk([_row(1, suggestion=sugg)], settings, BulkDecisionIn(field_ids=[1], accept_suggestion=True))
    assert planned[0]["after"]["reviewed"] and planned[0]["settings"] == {}


def test_accept_suggestion_rederives_customised_settings_when_the_type_changes():
    sugg = {"linear": {**SETTINGS, "action": "keep", "semantic_type": "numeric_continuous", "scaling": "standard"}}
    settings = {1: {"linear": {**SETTINGS, "encoding": "onehot", "source": "user"}}}
    planned = plan_bulk(
        [_row(1, stype="categorical", suggestion=sugg)], settings, BulkDecisionIn(field_ids=[1], accept_suggestion=True)
    )
    assert planned[0]["after"]["semantic_type"] == "numeric_continuous" and list(planned[0]["settings"]) == ["linear"]


def test_accept_suggestion_never_touches_target_role():
    sugg = {"tree": {**SETTINGS, "action": "target", "semantic_type": "binary"}}
    planned = plan_bulk(
        [_row(1, stype="binary", suggestion=sugg)],
        {1: {"tree": SETTINGS}},
        BulkDecisionIn(field_ids=[1], accept_suggestion=True),
    )
    assert planned[0]["after"] == {"action": "keep", "semantic_type": "binary", "reviewed": True}
