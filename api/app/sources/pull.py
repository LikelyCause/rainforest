"""The pull job: a connected dataset's collections into chunked Parquet, assembled into `raw.parquet` with a `source`
column and duplicates left out, then processed like any other dataset."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import polars as pl
from bson import json_util
from pymongo.errors import PyMongoError

from .. import sources
from ..config import get_settings
from ..db import execute, fetch_one, jsonb
from ..processing import _Status, process_dataset
from . import CollectionSpec, Source, SourceError, as_text, chunk_paths, compared_columns

log = logging.getLogger(__name__)

SOURCE = "source"
RANK = "__source_rank"
NAMES_IN_WARNING = 20
EMPTY_SHOWN = '""'  # how a warning shows an empty key


def assemble(
    out_dir: Path,
    specs: list[CollectionSpec],
    add_source: bool,
    dedupe: bool,
    match_on: list[str] | None,
    dest: Path,
    *,
    identifiers: list[str],
    source_column: str | None = None,
) -> dict:
    """The chunks `pull` wrote, one frame in collection order, written to `dest`; the chunks are deleted.

    Deduplication compares `compared_columns` less the pull's `identifiers` (the rule the preview applies to its
    sample), or exactly `match_on`. Rows are in collection order, so keeping the first of each group keeps the row
    of the collection listed first. The label column is `source_column` when given (a refresh's: the name of the
    previous pull's); without it, `source` or the first `source_<n>` the documents don't have."""
    labels = [s.label or s.name for s in specs]
    chunks = {label: chunk_paths(out_dir, label) for label in labels}
    schemas = {p: pl.read_parquet_schema(p) for paths in chunks.values() for p in paths}
    if not schemas:
        raise ValueError("The collections returned no documents")
    # A column that turned out mixed is text in the chunks written after that and its own type before.
    text = {c for sch in schemas.values() for c, dt in sch.items() if dt == pl.String}
    typed = {c for sch in schemas.values() for c, dt in sch.items() if dt not in (pl.String, pl.Null)}
    recast = text & typed
    frames = []
    for rank, label in enumerate(labels):
        for path in chunks[label]:
            sch = schemas[path]
            frames.append(
                pl.scan_parquet(path).with_columns(
                    *[
                        pl.col(c).map_elements(as_text, return_dtype=pl.String)
                        for c in sch
                        if c in recast and sch[c] not in (pl.String, pl.Null)
                    ],
                    pl.lit(rank, pl.UInt32).alias(RANK),
                )
            )
    df = pl.concat(frames, how="diagonal_relaxed").collect()
    data_cols = [c for c in df.columns if c != RANK]

    compared: list[str] = []
    if dedupe:
        if match_on:
            missing = [c for c in match_on if c not in data_cols]
            if missing:
                raise ValueError(f"Match on names columns the data doesn't have: {', '.join(missing)}")
            compared = list(match_on)
        else:
            per_collection = [set().union(*(schemas[p] for p in chunks[lb])) for lb in labels if chunks[lb]]
            compared = compared_columns(data_cols, per_collection, set(identifiers).__contains__)
    before = _per_rank(df)
    if compared:
        df = df.unique(subset=compared, keep="first", maintain_order=True)
    after = _per_rank(df)
    removed = {labels[r]: n - after.get(r, 0) for r, n in before.items() if n > after.get(r, 0)}

    out_cols: list[pl.Expr] = []
    renamed: dict[str, str] = {}
    name = None
    if add_source:
        if source_column is None:
            name, n = SOURCE, 2
            while name in data_cols:  # the documents have a field of that name already
                name = f"{SOURCE}_{n}"
                n += 1
        else:
            # The name the first pull gave it, so its field keeps holding the labels; a key of the documents that
            # now has that name gives way instead.
            name = source_column
            if name in data_cols:
                n = 2
                while f"{name}_{n}" in data_cols:
                    n += 1
                renamed[name] = f"{name}_{n}"
        out_cols.append(
            pl.col(RANK).replace_strict(list(range(len(labels))), labels, return_dtype=pl.String).alias(name)
        )
    # A field that was null in every document has no type; text is what an upload would give it.
    out_cols += [
        (pl.col(c).cast(pl.String) if df.schema[c] == pl.Null else pl.col(c)).alias(renamed.get(c, c))
        for c in data_cols
    ]
    df = df.select(out_cols)

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        df.write_parquet(tmp, row_group_size=100_000)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    for path in schemas:
        path.unlink(missing_ok=True)
    return {
        "rows": df.height,
        "removed_duplicates": removed,
        "compared_on": [renamed.get(c, c) for c in compared],
        "source_column": name,
        "renamed": renamed,
    }


def _per_rank(df: pl.DataFrame) -> dict[int, int]:
    return dict(df.group_by(RANK).len().iter_rows())


def specs_from(collections: list[dict]) -> list[CollectionSpec]:
    """`dataset_sources.collections` as specs; filters and projections are stored as extended JSON."""
    return [
        CollectionSpec(
            c["name"],
            c.get("label") or c["name"],
            _bson(c.get("filter")),
            _bson(c.get("projection")),
        )
        for c in collections
    ]


def _bson(doc: dict | None) -> dict | None:
    return None if doc is None else json_util.loads(json.dumps(doc))


def names(items: list[str]) -> str:
    """Names for a warning, the first few of a long list."""
    more = len(items) - NAMES_IN_WARNING
    return ", ".join(items[:NAMES_IN_WARNING]) + (f" and {more} more" if more > 0 else "")


def warnings_for(notes: dict, removed: dict[str, int], dedupe: bool) -> list[str]:
    """One readable sentence per kind of thing the pull changed, or asked for and couldn't do."""
    out = []
    if notes.get("arrays"):
        out.append(f"Arrays stored as JSON text: {names(notes['arrays'])}")
    if notes.get("mixed"):
        out.append(f"Mixed types read as text: {names(notes['mixed'])}")
    if notes.get("renamed"):
        renamed = [f"{k or EMPTY_SHOWN} → {v}" for k, v in notes["renamed"].items()]
        out.append(f"Keys renamed: {names(renamed)}")
    if removed:
        out.append(f"Duplicates removed: {names([f'{k} {v:,}' for k, v in removed.items()])}")
    if dedupe and not notes.get("compared_on"):
        out.append("Duplicates were not removed: no shared columns to compare on")
    return out


def pull_into(dataset_id: int, src: dict, dest: Path, status: str = "ingesting", previous: dict | None = None) -> dict:
    """Pull a dataset's collections into `dest`, reporting progress under `status`. Returns what the pull row stores:
    rows, per_source (with each collection's removed duplicates) and flatten_notes. The chunk directory is gone
    afterwards, whatever happens.

    `previous` is a refresh's last successful pull (its per_source and flatten_notes). A collection that no longer
    exists, or that returns no documents after returning some, then fails the pull: taken as it is, it would delete
    its rows, and with them the fields only it has and their decisions. The label column keeps the name it had."""
    s = get_settings()
    st = _Status(dataset_id)
    pull_dir = s.datasets_dir / str(dataset_id) / "pull"
    shutil.rmtree(pull_dir, ignore_errors=True)
    try:
        registry = sources.registry()
        if src["connection"] not in registry:
            raise SourceError(f"{src['connection']}: no connection of that name is configured")
        source: Source = registry[src["connection"]]
        specs = specs_from(src["collections"])
        if previous is not None:
            _require_collections(source, src, specs)
        try:
            result = source.pull(
                src["database"],
                specs,
                pull_dir,
                s.source_max_cells,
                s.source_pull_timeout_s,
                lambda label, n: st.set(status, 0.0, f"Pulling {label}: {n:,} documents"),
            )
        except PyMongoError as exc:  # a source's pull wraps its own; this keeps any other path redacted
            raise SourceError(source.redact(str(exc))) from None
        if previous is not None:
            had = {p["label"]: p["documents"] for p in previous.get("per_source") or []}
            for spec, got in zip(specs, result.per_source):
                if got["documents"] == 0 and had.get(got["label"], 0) > 0:
                    raise SourceError(
                        f"{src['connection']}: collection {spec.name} returned no documents; its previous pull had "
                        f"{had[got['label']]:,}"
                    )
        st.set(status, 0.0, "Combining collections", force=True)
        summary = assemble(
            pull_dir,
            specs,
            src["add_source"],
            src["dedupe"],
            src.get("match_on"),
            dest,
            identifiers=result.identifiers,
            source_column=((previous or {}).get("flatten_notes") or {}).get("source_column"),
        )
    finally:
        shutil.rmtree(pull_dir, ignore_errors=True)
    removed = summary["removed_duplicates"]
    return {
        "rows": summary["rows"],
        "per_source": [{**p, "removed_duplicates": removed.get(p["label"], 0)} for p in result.per_source],
        "flatten_notes": {
            **result.flatten_notes,
            "renamed": {**result.flatten_notes["renamed"], **summary["renamed"]},
            "compared_on": summary["compared_on"],
            "source_column": summary["source_column"],
        },
        "removed_duplicates": removed,
    }


def _require_collections(source: Source, src: dict, specs: list[CollectionSpec]) -> None:
    present = {c["name"] for c in source.list_collections(src["database"])}
    if missing := [s.name for s in specs if s.name not in present]:
        one = len(missing) == 1
        raise SourceError(
            f"{src['connection']}: collection{'' if one else 's'} {', '.join(missing)} no longer "
            f"exist{'s' if one else ''} in the database"
        )


def run_connect(dataset_id: int) -> None:
    """The job behind a new connected dataset: pull, record the pull, then process `raw.parquet` like an upload."""
    src = fetch_one("SELECT * FROM meta.dataset_sources WHERE dataset_id = %s", (dataset_id,))
    if src is None:  # deleted while queued
        return
    pull_id = fetch_one(
        "INSERT INTO meta.dataset_pulls (dataset_id, status, created_by) VALUES (%s, 'running', %s) RETURNING id",
        (dataset_id, src["created_by"]),
    )["id"]
    execute(
        """UPDATE meta.datasets SET status = 'ingesting', progress = 0, error = NULL, status_message = %s,
                  updated_at = now() WHERE id = %s""",
        (f"Pulling from {src['connection']}", dataset_id),
    )
    dest = get_settings().datasets_dir / str(dataset_id) / "raw.parquet"
    try:
        pulled = pull_into(dataset_id, src, dest)
        execute(
            """UPDATE meta.dataset_pulls SET status = 'done', rows = %s, per_source = %s, flatten_notes = %s,
                      finished_at = now() WHERE id = %s""",
            (pulled["rows"], jsonb(pulled["per_source"]), jsonb(pulled["flatten_notes"]), pull_id),
        )
        execute(
            "UPDATE meta.datasets SET warnings = warnings || %s::text[], updated_at = now() WHERE id = %s",
            (warnings_for(pulled["flatten_notes"], pulled["removed_duplicates"], src["dedupe"]), dataset_id),
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("pulling dataset %s failed", dataset_id)
        # A SourceError is redacted already; nothing else carries a connection string.
        message = str(exc) or type(exc).__name__
        execute(
            """UPDATE meta.dataset_pulls SET status = 'failed', error = %s, finished_at = now() WHERE id = %s""",
            (message, pull_id),
        )
        execute(
            """UPDATE meta.datasets SET status = 'failed', error = %s, status_message = 'Failed', updated_at = now()
               WHERE id = %s""",
            (message, dataset_id),
        )
        return
    process_dataset(dataset_id, None, "parquet")
