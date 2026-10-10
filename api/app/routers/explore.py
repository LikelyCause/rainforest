from __future__ import annotations

from typing import Annotated

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform

from .. import experiments, jobs
from psycopg import sql

from ..analysis.runner import ROW_COLS, data_version, find_or_create, public, run_analysis
from ..assessment import Source, key_owner, source_for
from ..auth import CurrentUser
from ..db import connection, fetch_all
from ..schemas import AnalysisRequest, CorrelationQuery

router = APIRouter(prefix="/api", tags=["explore"])


def _ready_dataset(conn, dataset_id: int, lock: bool = False) -> dict:
    ds = conn.execute(
        f"SELECT id, status FROM meta.datasets WHERE id = %s{' FOR UPDATE' if lock else ''}", (dataset_id,)
    ).fetchone()
    if ds is None:
        raise HTTPException(404, "Dataset not found")
    if ds["status"] != "ready":
        raise HTTPException(409, "Dataset is not ready")
    return ds


def _export_visible(conn, data: str, user: dict) -> bool:
    """An "export:<id>" data source is usable only by someone who may see that export's experiment."""
    return conn.execute(f"SELECT 1 FROM meta.exports WHERE id = %s AND experiment_id IN "
                        f"({experiments.visible_ids(user['id'])})", (int(data.split(":", 1)[1]),)).fetchone() is not None


def _validate_source(conn, dataset_id: int, params: dict, user: dict) -> None:
    target = params["target"]
    if params["data"] == "raw":
        if target is not None and conn.execute(
            """SELECT 1 FROM meta.fields f JOIN meta.field_decisions d ON d.field_id = f.id
                WHERE f.dataset_id = %s AND f.name = %s AND d.action = 'target'""",
            (dataset_id, target),
        ).fetchone() is None:
            raise HTTPException(422, f"{target!r} is not a target of this dataset")
        return
    exp = conn.execute(
        "SELECT status, summary FROM meta.exports WHERE id = %s AND dataset_id = %s",
        (int(params["data"].split(":", 1)[1]), dataset_id),
    ).fetchone()
    if exp is None or not _export_visible(conn, params["data"], user):
        raise HTTPException(404, "Export not found")
    if exp["status"] != "done":
        raise HTTPException(409, "Export is not finished")
    if target is not None and target not in ((exp["summary"] or {}).get("targets") or []):
        raise HTTPException(422, f"{target!r} is not a target of this export")


def decision_source(conn, dataset_id: int, user: dict, kept: bool, view: str | None = None) -> Source:
    """Explore reads the team decision, except that "Kept only" while assessing uses the viewer's own choices, and
    `view` picks an experiment's copy or a finalized baseline instead."""
    if view:
        return source_for(conn, dataset_id, user, view=view)
    src = source_for(conn, dataset_id, user)
    return src if kept and src.personal else Source(src.recipe)


@router.post("/datasets/{dataset_id}/analyses")
def request_analysis(dataset_id: int, req: AnalysisRequest, user: CurrentUser) -> dict:
    """The cached result for these params if current, else the job computing it (started if needed)."""
    params = req.params.normalised(req.kind)
    with connection() as conn:
        _ready_dataset(conn, dataset_id, lock=True)  # serialises concurrent requests for one dataset
        if params["data"] == "raw":
            # Part of the cache key, so results computed from different people's choices never mix.
            params["decisions"] = decision_source(conn, dataset_id, user, params["kept"], req.view).key
        _validate_source(conn, dataset_id, params, user)
        row, created = find_or_create(conn, dataset_id, req.kind, params, req.recompute, user["id"])
    if created:
        jobs.submit(run_analysis, row["id"])
    return row


@router.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: int, user: CurrentUser) -> dict:
    with connection() as conn:
        row = conn.execute(f"SELECT {ROW_COLS} FROM meta.analyses WHERE id = %s", (analysis_id,)).fetchone()
        key = (row["params"].get("decisions") or "team") if row else "team"
        owner = key_owner(key)
        if key.startswith("experiment:"):
            try:
                experiments.get_visible(conn, int(key.split(":")[1]), user)
            except HTTPException:
                owner = -1  # not visible to this viewer: answer as for a missing analysis
        if row is None or (owner is not None and owner != user["id"]) or (
                row["params"].get("data", "raw").startswith("export:")
                and not _export_visible(conn, row["params"]["data"], user)):
            raise HTTPException(404, "Analysis not found")
        current = data_version(conn, row["dataset_id"], row["params"]) if row["status"] == "done" else None
    return public(row, current)


@router.get("/datasets/{dataset_id}/explore/correlations")
def correlations(dataset_id: int, q: Annotated[CorrelationQuery, Query()], user: CurrentUser) -> dict:
    """Stored associations for a capped, clustered set of fields: the target's strongest partners, or the
    fields with the strongest associations overall."""
    with connection() as conn:
        _ready_dataset(conn, dataset_id)
        dec = decision_source(conn, dataset_id, user, q.kept, q.view).decisions()
    fields = fetch_all(
        sql.SQL("""SELECT f.id, f.name, d.semantic_type AS type, d.action
                     FROM meta.fields f JOIN {} d ON d.field_id = f.id
                    WHERE f.dataset_id = %s ORDER BY f.ordinal""").format(dec),
        (dataset_id,),
    )
    by_id = {f["id"]: f for f in fields}
    target = next((f for f in fields if f["name"] == q.target), None) if q.target else None
    if q.target and target is None:
        raise HTTPException(422, f"Unknown field {q.target!r}")
    unknown = [i for i in q.add_ids if i not in by_id]
    if unknown:
        raise HTTPException(422, {"message": "Unknown field id(s)", "errors": unknown})
    target_id = target["id"] if target is not None else None
    # A field's score is its strongest association with the target, or with any eligible field.
    ranked = [r["id"] for r in fetch_all(
        sql.SQL("""WITH eligible AS (
               SELECT f.id FROM meta.fields f JOIN {} d ON d.field_id = f.id
                WHERE f.dataset_id = %(ds)s
                  AND (d.action = 'keep' OR (NOT %(kept)s AND d.action <> 'target') OR f.id = %(target)s)),
           sides AS (
               SELECT field_a AS me, field_b AS other, strength FROM meta.field_correlations WHERE dataset_id = %(ds)s
               UNION ALL
               SELECT field_b, field_a, strength FROM meta.field_correlations WHERE dataset_id = %(ds)s)
           SELECT me AS id, max(strength) AS score FROM sides
            WHERE me IN (SELECT id FROM eligible) AND other IN (SELECT id FROM eligible) AND strength <> 'NaN'
              AND (%(target)s::bigint IS NULL OR other = %(target)s)
            GROUP BY me ORDER BY score DESC, me LIMIT %(n)s""").format(dec),
        {"ds": dataset_id, "kept": q.kept, "target": target_id, "n": q.limit},
    )]
    chosen = [target_id] if target_id is not None else []
    chosen += ranked[: q.limit - len(chosen)]
    chosen += [i for i in q.add_ids if i not in chosen]

    pairs = fetch_all(
        """SELECT field_a AS a, field_b AS b, method, value, strength, n_obs AS n
             FROM meta.field_correlations
            WHERE dataset_id = %(ds)s AND field_a = ANY(%(ids)s) AND field_b = ANY(%(ids)s)
            ORDER BY field_a, field_b""",
        {"ds": dataset_id, "ids": chosen},
    )
    pos = {fid: k for k, fid in enumerate(chosen)}
    cells, strength = [], np.zeros((len(chosen), len(chosen)))
    for p in pairs:
        cells.append({**p, "value": float(p["value"]), "strength": float(p["strength"])})
        strength[pos[p["a"]], pos[p["b"]]] = strength[pos[p["b"]], pos[p["a"]]] = p["strength"]
    return {
        "fields": [{k: by_id[i][k] for k in ("id", "name", "type", "action")} for i in chosen],
        "order": [chosen[k] for k in cluster_order(strength)],
        "cells": cells,
    }


def cluster_order(strength: np.ndarray) -> list[int]:
    """Average-linkage leaf order on 1 - strength, so strongly associated fields sit next to each other."""
    n = strength.shape[0]
    if n < 3:
        return list(range(n))
    dist = 1.0 - np.clip(strength, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    z = optimal_leaf_ordering(linkage(condensed, method="average"), condensed)
    return [int(i) for i in leaves_list(z)]
