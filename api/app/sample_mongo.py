"""Seed data for the MongoDB source: two collections that exercise what a connected dataset has to cope with.

`curated` is clean. `massive` carries the same keys plus a deeper one that only some documents have, a column
whose type is mixed ("n/a" among numbers), and rows that repeat curated ones under a new `_id`, for deduplication.
Tests seed a throwaway database with it, and the Compose seeder runs the same function.

CLI:  python -m app.sample_mongo mongodb://host:27017 database
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from bson.objectid import ObjectId

CURATED_DOCS = 300
MASSIVE_DOCS = 500
MIXED_DOCS = 20  # "n/a" in rpm.mean; all among the first 200 so a 200-document preview sample sees them
DUPLICATE_DOCS = 50  # massive rows equal to curated ones except `_id`
SEED = 7

VEHICLES = ("sedan", "pickup", "van", "coupe", "wagon")
ENGINES = ("V6", "V8", "I4", "I4 turbo", "electric")
TAGS = ("fleet", "demo", "retrofit", "prototype", "recall")
EPOCH = datetime(2025, 1, 1)


def _object_id(collection: int, n: int) -> ObjectId:
    # Fixed ids keep two seedings of the same data identical.
    return ObjectId(f"{collection:08x}{n:016x}")


def curated_docs(n: int = CURATED_DOCS, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    docs = []
    for i in range(n):
        mean = round(rng.uniform(800, 3500), 1)
        docs.append(
            {
                "_id": _object_id(1, i),
                "vehicle": rng.choice(VEHICLES),
                "engine": rng.choice(ENGINES),
                "rpm": {"mean": mean, "max": round(mean * rng.uniform(1.2, 2.0), 1)},
                "tags": rng.sample(TAGS, rng.randint(1, 3)),
                "when": EPOCH + timedelta(minutes=rng.randrange(0, 60 * 24 * 365)),
            }
        )
    return docs


def massive_docs(curated: list[dict], n: int = MASSIVE_DOCS, seed: int = SEED + 1) -> list[dict]:
    rng = random.Random(seed)
    docs = []
    for i, doc in enumerate(curated_docs(n, seed)):  # same shape as curated, with values of its own
        doc["_id"] = _object_id(2, i)
        if 200 <= i < 200 + DUPLICATE_DOCS:
            doc = {**curated[i - 200], "_id": doc["_id"]}
        else:
            if i < 200 and i % (200 // MIXED_DOCS) == 5:
                doc["rpm"]["mean"] = "n/a"
            if i % 3 == 0:  # only some documents reach this deep
                doc["extra"] = {"depth": {"key": rng.choice("abcd")}}
        docs.append(doc)
    return docs


def seed(db) -> None:
    """Replace `curated` and `massive` in `db` (a pymongo Database) with the seed data."""
    curated = curated_docs()
    for name, docs in (("curated", curated), ("massive", massive_docs(curated))):
        db.drop_collection(name)
        db[name].insert_many(docs)


if __name__ == "__main__":
    import sys

    import pymongo

    if len(sys.argv) != 3:
        sys.exit(__doc__)
    seed(pymongo.MongoClient(sys.argv[1])[sys.argv[2]])
