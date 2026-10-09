from __future__ import annotations

from typing import Annotated

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform

from .. import jobs
from ..analysis.runner import ROW_COLS, data_version, find_or_create, public, run_analysis
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


def _validate_source(conn, dataset_id: int, params: dict) -> None:
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
    if exp is None:
        raise HTTPException(404, "Export not found")
    if exp["status"] != "done":
        raise HTTPException(409, "Export is not finished")
    if target is not None and target not in ((exp["summary"] or {}).get("targets") or []):
        raise HTTPException(422, f"{target!r} is not a target of this export")


@router.post("/datasets/{dataset_id}/analyses")
def request_analysis(dataset_id: int, req: AnalysisRequest) -> dict:
    """The cached result for these params if current, else the job computing it (started if needed)."""
    params = req.params.normalised(req.kind)
    with connection() as conn:
        _ready_dataset(conn, dataset_id, lock=True)  # serialises concurrent requests for one dataset
        _validate_source(conn, dataset_id, params)
        row, created = find_or_create(conn, dataset_id, req.kind, params, req.recompute)
    if created:
        jobs.submit(run_analysis, row["id"])
    return row


@router.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: int) -> dict:
    with connection() as conn:
        row = conn.execute(f"SELECT {ROW_COLS} FROM meta.analyses WHERE id = %s", (analysis_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Analysis not found")
        current = data_version(conn, row["dataset_id"], row["params"]["data"]) if row["status"] == "done" else None
    return public(row, current)


@router.get("/datasets/{dataset_id}/explore/correlations")
def correlations(dataset_id: int, q: Annotated[CorrelationQuery, Query()]) -> dict:
    """Stored associations for a capped, clustered set of fields: the target's strongest partners, or the
    fields with the strongest associations overall."""
    with connection() as conn:
        _ready_dataset(conn, dataset_id)
    fields = fetch_all(
        """SELECT f.id, f.name, d.semantic_type AS type, d.action
             FROM meta.fields f JOIN meta.field_decisions d ON d.field_id = f.id
            WHERE f.dataset_id = %s ORDER BY f.ordinal""",
        (dataset_id,),
    )
    by_id = {f["id"]: f for f in fields}
    target = next((f for f in fields if f["name"] == q.target), None) if q.target else None
    if q.target and target is None:
        raise HTTPException(422, f"Unknown field {q.target!r}")
    unknown = [i for i in q.add_ids if i not in by_id]
    if unknown:
        raise HTTPException(422, {"message": "Unknown field id(s)", "errors": unknown})
    eligible = {f["id"] for f in fields
                if f["action"] == "keep" or (not q.kept and f["action"] != "target")}
    if target is not None:
        eligible.add(target["id"])

    pairs = fetch_all(
        """SELECT field_a AS a, field_b AS b, method, value, strength, n_obs AS n
             FROM meta.field_correlations WHERE dataset_id = %s""",
        (dataset_id,),
    )
    score: dict[int, float] = {}
    for p in pairs:
        if p["a"] not in eligible or p["b"] not in eligible:
            continue
        for me, other in ((p["a"], p["b"]), (p["b"], p["a"])):
            if target is None or other == target["id"]:
                score[me] = max(score.get(me, 0.0), p["strength"])
    ranked = sorted((i for i in score if target is None or i != target["id"]), key=lambda i: (-score[i], i))
    chosen = [target["id"]] if target is not None else []
    chosen += ranked[: q.limit - len(chosen)]
    chosen += [i for i in q.add_ids if i not in chosen]

    pos = {fid: k for k, fid in enumerate(chosen)}
    cells, strength = [], np.zeros((len(chosen), len(chosen)))
    for p in pairs:
        if p["a"] in pos and p["b"] in pos:
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
