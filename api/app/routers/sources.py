from __future__ import annotations

from dataclasses import asdict

from bson import json_util
from fastapi import APIRouter, HTTPException

from .. import sources
from ..auth import CurrentUser
from ..config import get_settings
from ..schemas import SourceCollectionIn, SourcePreviewIn
from ..sources import PREVIEW_SAMPLE, CollectionSpec, PreviewResult, Source, SourceError

router = APIRouter(prefix="/api/sources", tags=["sources"])


def _source(name: str) -> Source:
    try:
        return sources.get(name)
    except KeyError:
        raise HTTPException(404, f"No source named {name}") from None


def _document(collection: str, field: str, text: str | None) -> dict | None:
    if text is None or not text.strip():
        return None
    try:
        value = json_util.loads(text)
    except Exception:  # noqa: BLE001 - a malformed {"$date": ...} wrapper raises whatever its decoder does
        value = None
    if not isinstance(value, dict):
        raise HTTPException(422, f"{collection}: {field} is not a JSON document")
    return value


def collection_specs(collections: list[SourceCollectionIn]) -> list[CollectionSpec]:
    """Every filter and projection parsed before anything touches a server, so a bad one is a 422 and never a
    half-run request."""
    return [
        CollectionSpec(
            c.name,
            filter=_document(c.name, "filter", c.filter),
            projection=_document(c.name, "projection", c.projection),
        )
        for c in collections
    ]


def match_columns(match_on: list[str] | None, preview: PreviewResult) -> list[str] | None:
    """`match_on` trimmed and without repeats, None when it names nothing. A name no sampled document has is a 422
    now, since the pull would only find out once everything is pulled."""
    names = list(dict.fromkeys(m.strip() for m in match_on or [] if m.strip())) or None
    sampled = {c.name for c in preview.columns}
    if unknown := [n for n in names or [] if n not in sampled]:
        raise HTTPException(
            422,
            f"Match on column not found in a sample of up to {PREVIEW_SAMPLE} documents per collection: {unknown[0]}",
        )
    return names


@router.get("")
def list_sources(user: CurrentUser) -> list[dict]:
    return [{"name": s.name, "kind": s.kind} for s in sources.registry().values()]


@router.get("/{name}/databases")
def list_databases(name: str, user: CurrentUser) -> dict:
    source = _source(name)
    try:
        names = source.list_databases()
    except SourceError as exc:
        raise HTTPException(502, str(exc)) from None
    return {"databases": names, "listable": names is not None}


@router.get("/{name}/databases/{database}/collections")
def list_collections(name: str, database: str, user: CurrentUser) -> list[dict]:
    source = _source(name)
    try:
        return source.list_collections(database)
    except SourceError as exc:
        raise HTTPException(502, str(exc)) from None


@router.post("/{name}/preview")
def preview(name: str, body: SourcePreviewIn, user: CurrentUser) -> dict:
    source = _source(name)
    specs = collection_specs(body.collections)
    try:
        result = source.preview(body.database, specs)
    except SourceError as exc:
        raise HTTPException(502, str(exc)) from None
    match_on = match_columns(body.match_on, result)
    max_cells = get_settings().source_max_cells
    return {
        **asdict(result),
        "compared_on": match_on or result.compared_on,
        "budget": {"max_cells": max_cells, "fits": result.estimated_cells <= max_cells},
    }
