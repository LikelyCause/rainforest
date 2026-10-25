"""The lock a refresh holds (status `refreshing`): every write to the dataset and every new export, run or Explore
computation on it is refused until the refresh ends."""

from __future__ import annotations

from fastapi import HTTPException

REFRESHING = "A refresh of this dataset is running; try again once it finishes."


def refuse_refreshing(status: str | None) -> None:
    if status == "refreshing":
        raise HTTPException(409, REFRESHING)


def refuse_while_refreshing(conn, dataset_id: int) -> None:
    """409 while the dataset is refreshing. The row stays key-share locked until the transaction ends, so the refresh
    endpoint (which takes it FOR UPDATE) waits for this write and then sees it. Not for a transaction that later locks
    the row FOR UPDATE itself (recipes.lock_dataset): two of those would deadlock upgrading."""
    row = conn.execute("SELECT status FROM meta.datasets WHERE id = %s FOR KEY SHARE", (dataset_id,)).fetchone()
    refuse_refreshing(row and row["status"])
