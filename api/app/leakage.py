"""Leakage relative to a set of targets: which fields are nearly perfectly associated with any of them. A baseline or
an experiment checks its own targets this way, rather than the flags stored for the working copy's targets."""
from __future__ import annotations

from .processing import LEAKAGE_THRESHOLD


def leak_names(conn, dataset_id: int, targets: list[str]) -> set[str]:
    """Non-target fields whose association with one of `targets` reaches the leakage threshold."""
    if not targets:
        return set()
    rows = conn.execute(
        """SELECT DISTINCT o.name
             FROM meta.field_correlations c
             JOIN meta.fields t ON t.id IN (c.field_a, c.field_b) AND t.dataset_id = %(ds)s AND t.name = ANY(%(t)s)
             JOIN meta.fields o ON o.id IN (c.field_a, c.field_b) AND o.id <> t.id AND NOT o.name = ANY(%(t)s)
            WHERE c.dataset_id = %(ds)s AND c.strength >= %(th)s""",
        {"ds": dataset_id, "t": list(targets), "th": LEAKAGE_THRESHOLD},
    ).fetchall()
    return {r["name"] for r in rows}
