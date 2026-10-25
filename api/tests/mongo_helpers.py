"""Test support for the MongoDB source: the seeded data and pointing the source registry at a server."""

from contextlib import contextmanager

import pytest
from pymongo.database import Database

from app.config import get_settings
from app.sample_mongo import seed as seed_database
from app.sources import registry


def seed(db: Database) -> None:
    seed_database(db)


@contextmanager
def use_sources(text: str):
    """RAINFOREST_MONGO_SOURCES=`text` with fresh settings and registry, put back (and rebuilt lazily) on exit."""
    mp = pytest.MonkeyPatch()
    mp.setenv("RAINFOREST_MONGO_SOURCES", text)
    _reset()
    try:
        yield
    finally:
        mp.undo()
        _reset()


def _reset() -> None:
    for source in registry().values():
        client = getattr(source, "_client", None)
        if client is not None:
            client.close()
    get_settings.cache_clear()
    registry.cache_clear()
