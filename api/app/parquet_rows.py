"""Read selected rows of a Parquet file a few column chunks at a time, so a sample never costs a full column read."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

COLUMNS_PER_READ = 4


def read_rows(path: Path, columns: list[str], idx: np.ndarray) -> pl.DataFrame:
    """Rows at the sorted positions `idx`, in that order. Peak memory is the result plus a few column chunks."""
    pf = pq.ParquetFile(path)
    idx = np.asarray(idx, dtype=np.int64)
    sizes = [pf.metadata.row_group(g).num_rows for g in range(pf.num_row_groups)]
    starts = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    parts: dict[str, list[pa.Array]] = {c: [] for c in columns}
    for g in range(pf.num_row_groups):
        lo, hi = np.searchsorted(idx, [starts[g], starts[g + 1]])
        if lo == hi:
            continue
        local = pa.array(idx[lo:hi] - starts[g])
        for k in range(0, len(columns), COLUMNS_PER_READ):
            batch = columns[k:k + COLUMNS_PER_READ]
            taken = pf.read_row_group(g, columns=batch).take(local)
            for c in batch:
                parts[c].append(taken.column(c).combine_chunks())
    if not columns:
        return pl.DataFrame()
    schema = pf.schema_arrow
    table = pa.table({c: pa.chunked_array(parts[c], type=schema.field(c).type) for c in columns})
    out = pl.from_arrow(table, rechunk=True)
    assert isinstance(out, pl.DataFrame)
    return out
